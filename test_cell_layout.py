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
    cell_layout_for_prompt,
    cell_layout_for_prompt_budgeted,
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


# ── cell_layout_for_prompt ─────────────────────────────────────────────────


@dataclass
class FakeRun:
    range_id: str


@dataclass
class FakeRegion:
    bbox_range_id: str


def _grid_with_runs_regions(page=2, rows=35, cols=24,
                             runs=None, regions=None):
    g = _build_fake_grid(page=page, rows=rows, cols=cols)
    g.runs = runs or []
    g.regions = regions or []
    return g


def test_cell_layout_for_prompt_empty():
    assert cell_layout_for_prompt([]) == []


def test_cell_layout_for_prompt_shape():
    g = _grid_with_runs_regions(
        page=3, rows=35, cols=24,
        runs=[FakeRun("A5:X5"), FakeRun("A6:X6")],
        regions=[FakeRegion("A5:X9"), FakeRegion("A22:X29")],
    )
    out = cell_layout_for_prompt([g])
    assert out == [{
        "page": 3, "rows": 35, "cols": 24,
        "writable_runs": ["A5:X5", "A6:X6"],
        "regions": ["A5:X9", "A22:X29"],
    }]


def test_cell_layout_for_prompt_missing_runs_regions_ok():
    """A grid with no runs / no regions still emits an entry with empty arrays."""
    g = _grid_with_runs_regions(page=1)
    out = cell_layout_for_prompt([g])
    assert out[0]["writable_runs"] == []
    assert out[0]["regions"] == []
    assert out[0]["page"] == 1


def test_cell_layout_for_prompt_token_budget():
    """Sanity-check the size on a worst-case dense grid (35 rows, 24 cols,
    every row a writable run). Stays under 2k chars per page so an 18-page
    PDF fits comfortably in Gemini's input limit."""
    runs = [FakeRun(f"A{r}:X{r}") for r in range(1, 36)]
    regions = [FakeRegion(f"A{r}:X{r+2}") for r in (1, 5, 10, 15, 20, 25, 30)]
    g = _grid_with_runs_regions(runs=runs, regions=regions)
    import json
    serialized = json.dumps(cell_layout_for_prompt([g]))
    # 35 runs × ~7 chars each + 7 regions × ~10 chars + envelope ≈ 350-450 chars
    assert len(serialized) < 2000, f"per-page payload {len(serialized)} chars"


def test_cell_layout_for_prompt_multipage():
    g1 = _grid_with_runs_regions(page=1, runs=[FakeRun("A1:X1")])
    g2 = _grid_with_runs_regions(page=2, runs=[FakeRun("B5:M5")])
    out = cell_layout_for_prompt([g1, g2])
    assert [p["page"] for p in out] == [1, 2]
    assert out[0]["writable_runs"] == ["A1:X1"]
    assert out[1]["writable_runs"] == ["B5:M5"]


def test_anchor_rows_list_detected_out_of_grid():
    """The wire shape uses anchor.rows[] (list of row ranges) instead of
    legacy anchor.range (single string). Validator handles both."""
    grid = _build_fake_grid(rows=10, cols=10)
    resp = _clean_response(grid)
    resp["data"]["items"][0]["annotations"][0]["comment_range"] = "C3:D4"
    resp["data"]["items"][0]["marking"] = {"page": grid.page,
                                            "score_range": "B2:C3",
                                            "score_box_range": "A1:D4"}
    # rows[]-shaped anchor with one entry out of grid
    resp["data"]["items"][0]["annotations"][0]["anchor"] = {
        "type": "underline",
        "rows": ["B7:G7", "A20:Z20"],   # second row way out of grid
    }
    tel = validate(resp, [grid])
    assert _count(tel, "I3") >= 1


def test_anchor_rows_list_clean_passes():
    """rows[]-shaped anchor with all cells in-grid should not fire I3."""
    grid = _build_fake_grid()
    resp = _clean_response(grid)
    resp["data"]["items"][0]["annotations"][0]["anchor"] = {
        "type": "underline",
        "rows": ["B7:G7", "B8:G8"],
    }
    tel = validate(resp, [grid])
    assert _count(tel, "I3") == 0


def test_cell_layout_for_prompt_range_ids_parseable():
    """Every emitted range ID round-trips through the same parser the
    validator uses — guards against accidental shape drift."""
    g = _grid_with_runs_regions(
        runs=[FakeRun("AA12:AC15"), FakeRun("D2:D2")],
        regions=[FakeRegion("Z1:Z35")],
    )
    out = cell_layout_for_prompt([g])
    for rid in out[0]["writable_runs"] + out[0]["regions"]:
        # _range_bounds raises on malformed input
        _range_bounds(rid)


def test_cell_layout_for_prompt_include_flags():
    g = _grid_with_runs_regions(
        runs=[FakeRun("A5:X5")], regions=[FakeRegion("A5:X9")],
    )
    runs_only = cell_layout_for_prompt([g], include_regions=False)
    assert "writable_runs" in runs_only[0]
    assert "regions" not in runs_only[0]

    regions_only = cell_layout_for_prompt([g], include_runs=False)
    assert "writable_runs" not in regions_only[0]
    assert "regions" in regions_only[0]


def test_cell_layout_for_prompt_budgeted_under_budget_keeps_runs():
    """Small payload → full layout (runs + regions) returned."""
    g = _grid_with_runs_regions(
        runs=[FakeRun("A5:X5"), FakeRun("A6:X6")],
        regions=[FakeRegion("A5:X9")],
    )
    out = cell_layout_for_prompt_budgeted([g], max_chars=10_000)
    assert "writable_runs" in out[0]
    assert "regions" in out[0]


def test_cell_layout_for_prompt_budgeted_over_budget_drops_runs():
    """Large payload → falls back to regions only."""
    runs = [FakeRun(f"A{r}:AZ{r}") for r in range(1, 100)]
    regions = [FakeRegion(f"A{r}:AZ{r+2}") for r in (1, 50, 99)]
    g = _grid_with_runs_regions(runs=runs, regions=regions)
    out = cell_layout_for_prompt_budgeted([g], max_chars=200)
    assert "writable_runs" not in out[0]
    assert out[0]["regions"] == [r.bbox_range_id for r in regions]
