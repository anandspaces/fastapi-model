"""Tests for cell_layout.validate — observe-only invariant checks.

Each invariant has:
  * a positive test (clean response → 0 violations of that invariant)
  * a negative test (single tampered field → exactly that invariant fires)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cell_layout import (
    Telemetry,
    ViolationRecord,
    _comment_fits,
    _expand_range,
    _range_bounds,
    _rects_overlap,
    validate,
)


# ── Tiny fake grid (avoids running cell_grid_service_v4 in unit tests) ──────

@dataclass
class FakeCell:
    cell_id: str
    row: int
    col: int
    writable: bool


@dataclass
class FakeGrid:
    page: int
    rows: int
    cols: int
    cell_size_pts: float
    cells: list[FakeCell]
    # The renderer needs these; the validator doesn't, but the dataclass
    # mirrors PageCellGrid so duck typing works.
    left_margin_pts: float = 0.0
    top_margin_pts: float = 0.0
    page_w_pts: float = 595.0
    page_h_pts: float = 842.0
    runs: list = field(default_factory=list)
    regions: list = field(default_factory=list)


def _build_fake_grid(
    page: int = 2, rows: int = 35, cols: int = 24, cell_size_pts: float = 24.0,
    *, non_writable: set[tuple[int, int]] | None = None,
) -> FakeGrid:
    nw = non_writable or set()
    cells = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            cid = ""  # not needed by validator
            cells.append(FakeCell(cid, r, c, writable=(r, c) not in nw))
    return FakeGrid(page=page, rows=rows, cols=cols, cell_size_pts=cell_size_pts, cells=cells)


def _clean_response(grid: FakeGrid) -> dict[str, Any]:
    return {
        "data": {
            "items": [
                {
                    "question_id": 1,
                    "marking": {
                        "page": grid.page,
                        "score_range": "C5:E6",
                        "score_box_range": "B4:F7",
                    },
                    "annotations": [
                        {
                            "page": grid.page,
                            "is_positive": False,
                            "comment": "Short remark.",
                            "comment_font_pts": 12.0,
                            "comment_range": "G3:K4",
                            "anchor": {
                                "type": "underline",
                                "range": "B7:H7",
                            },
                        }
                    ],
                }
            ]
        }
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _count(tel: Telemetry, invariant: str) -> int:
    return tel.counts.get(invariant, 0)


# ── Range helper unit tests ────────────────────────────────────────────────


def test_expand_range_single():
    assert _expand_range("B3") == [(3, 2)]


def test_expand_range_rect():
    cells = _expand_range("B3:D4")
    assert cells == [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]


def test_range_bounds():
    assert _range_bounds("B3:D5") == (3, 2, 5, 4)


def test_rects_overlap():
    assert _rects_overlap((1, 1, 3, 3), (3, 3, 5, 5))      # touch corner
    assert not _rects_overlap((1, 1, 2, 2), (3, 3, 4, 4))  # gap
    assert _rects_overlap((1, 1, 4, 4), (2, 2, 3, 3))      # contained


def test_comment_fits_simple():
    import fitz
    f = fitz.Font("helv")
    # 5 cells × 5 cells × 24pt = 120pt × 120pt; 12pt font → ~lots of room
    assert _comment_fits("Short.", f, 12.0, cols=5, rows=2, cell_size_pts=24.0)


def test_comment_fits_overflow_horizontal():
    import fitz
    f = fitz.Font("helv")
    # one extremely long word that can't fit any line
    long_word = "supercalifragilisticexpialidocious" * 4
    assert not _comment_fits(long_word, f, 12.0, cols=2, rows=3, cell_size_pts=24.0)


# ── End-to-end validator tests ─────────────────────────────────────────────


def test_clean_response_zero_violations():
    grid = _build_fake_grid()
    tel = validate(_clean_response(grid), [grid])
    assert tel.counts == {} or sum(tel.counts.values()) == 0
    assert tel.items_total == 1
    assert tel.annotations_total == 1


def test_i1_cell_out_of_grid():
    grid = _build_fake_grid(rows=10, cols=10)
    resp = _clean_response(grid)
    resp["data"]["items"][0]["annotations"][0]["comment_range"] = "G3:Z9"  # col Z = 26 > 10
    tel = validate(resp, [grid])
    assert _count(tel, "I1") >= 1


def test_i2_comment_on_nonwritable():
    # Mark cells (3,7)..(4,11) as non-writable  → comment_range "G3:K4" fully non-writable
    nw = {(r, c) for r in (3, 4) for c in range(7, 12)}
    grid = _build_fake_grid(non_writable=nw)
    tel = validate(_clean_response(grid), [grid])
    assert _count(tel, "I2") == 1


def test_i3_anchor_out_of_grid():
    grid = _build_fake_grid(rows=10, cols=10)
    resp = _clean_response(grid)
    # comment_range needs to stay valid → use small ranges
    resp["data"]["items"][0]["annotations"][0]["comment_range"] = "C3:D4"
    resp["data"]["items"][0]["marking"] = {"page": grid.page,
                                            "score_range": "B2:C3",
                                            "score_box_range": "A1:D4"}
    resp["data"]["items"][0]["annotations"][0]["anchor"] = {"type": "underline",
                                                            "range": "A20:Z20"}
    tel = validate(resp, [grid])
    assert _count(tel, "I3") >= 1


def test_i4_score_outside_box():
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    resp["data"]["items"][0]["marking"]["score_range"] = "Z30:Z31"  # outside box
    tel = validate(resp, [grid])
    assert _count(tel, "I4") == 1


def test_i5_font_out_of_range_low():
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    resp["data"]["items"][0]["annotations"][0]["comment_font_pts"] = 8.0
    tel = validate(resp, [grid])
    assert _count(tel, "I5") == 1


def test_i5_font_out_of_range_high():
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    resp["data"]["items"][0]["annotations"][0]["comment_font_pts"] = 18.0
    tel = validate(resp, [grid])
    assert _count(tel, "I5") == 1


def test_i6_overflow_text_too_long():
    grid = _build_fake_grid(cell_size_pts=24.0)
    resp = _clean_response(grid)
    # 1×1 cell at G3:G3 cannot fit a paragraph at 12pt
    resp["data"]["items"][0]["annotations"][0]["comment_range"] = "G3:G3"
    resp["data"]["items"][0]["annotations"][0]["comment"] = (
        "This is a very long teacher comment that absolutely cannot fit inside "
        "a single 24-point cell at 12 point font."
    )
    tel = validate(resp, [grid])
    assert _count(tel, "I6") >= 1


def test_i7_overlapping_comment_ranges():
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    item = resp["data"]["items"][0]
    item["annotations"].append({
        "page": grid.page,
        "is_positive": False,
        "comment": "Another remark.",
        "comment_font_pts": 12.0,
        "comment_range": "I3:M4",          # overlaps G3:K4
        "anchor": {"type": "none"},
    })
    tel = validate(resp, [grid])
    assert _count(tel, "I7") >= 1


def test_i8_exponent_caret_missing_word():
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    ann = resp["data"]["items"][0]["annotations"][0]
    ann["anchor"] = {"type": "exponent_caret", "range": "B7:B7", "extra": {}}
    tel = validate(resp, [grid])
    assert _count(tel, "I8") == 1


def test_i9_curly_brace_bad_side():
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    ann = resp["data"]["items"][0]["annotations"][0]
    ann["anchor"] = {"type": "curly_brace", "range": "A5:A10", "extra": {"side": "middle"}}
    tel = validate(resp, [grid])
    assert _count(tel, "I9") == 1


def test_telemetry_summary_shape():
    grid = _build_fake_grid()
    tel = validate(_clean_response(grid), [grid])
    s = tel.summary()
    assert s["items_total"] == 1
    assert s["annotations_total"] == 1
    assert s["violation_rate"] == 0.0
    assert "counts" in s


def test_violation_record_to_dict():
    rec = ViolationRecord(invariant="I1", page=2, item_index=0,
                          annotation_index=0, detail="x")
    d = rec.to_dict()
    assert d["invariant"] == "I1" and d["detail"] == "x"


def test_envelope_or_inner_dict():
    """Validator accepts both ``{"data": {...}}`` and the inner dict."""
    grid = _build_fake_grid()
    full = _clean_response(grid)
    inner = full["data"]
    a = validate(full, [grid])
    b = validate(inner, [grid])
    assert sum(a.counts.values()) == sum(b.counts.values()) == 0
    assert a.annotations_total == b.annotations_total == 1
