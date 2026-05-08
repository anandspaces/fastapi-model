"""Tests for remark_cell_layout_service."""

from __future__ import annotations

import pytest

from cell_grid_service import (
    CELL_SIZE_PTS,
    Cell,
    PageCellGrid,
    cell_id_from_rc,
    rc_from_cell_id,
)
from remark_cell_layout_service import (
    _bbox_from_cell_ids,
    assign_bboxes_to_annotations,
    assign_cell_ids,
    estimate_remark_cell_width,
    pick_anchor_cell,
)


def _cell(
    row: int,
    col: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    page: int = 1,
    writable: bool = True,
) -> Cell:
    return Cell(
        cell_id=cell_id_from_rc(row, col),
        row=row,
        col=col,
        page=page,
        x_start_percent=x0,
        x_end_percent=x1,
        y_start_percent=y0,
        y_end_percent=y1,
        pdf_x1=0.0,
        pdf_y1=0.0,
        pdf_x2=20.0,
        pdf_y2=20.0,
        score=1.0 if writable else 0.0,
        ink_ratio=0.0 if writable else 0.5,
        writable=writable,
    )


def _uniform_grid(rows: int, cols: int, *, writable: bool = True) -> PageCellGrid:
    cells: list[Cell] = []
    cw = 100.0 / cols
    rh = 100.0 / rows
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            cells.append(
                _cell(
                    r,
                    c,
                    writable=writable,
                    x0=(c - 1) * cw,
                    x1=c * cw,
                    y0=(r - 1) * rh,
                    y1=r * rh,
                )
            )
    return PageCellGrid(
        page=1,
        rows=rows,
        cols=cols,
        cell_size_pts=CELL_SIZE_PTS,
        left_margin_pts=0.0,
        top_margin_pts=0.0,
        cells=cells,
    )


def test_estimate_remark_cell_width_empty():
    assert estimate_remark_cell_width("") == 0
    assert estimate_remark_cell_width("   ") == 0


def test_estimate_remark_cell_width_hello():
    n = estimate_remark_cell_width(
        "Hello world",
        font_size_pts=11.0,
        cell_size_pts=20.0,
        fontname="helv",
    )
    assert n >= 2


def test_pick_anchor_cell_centre():
    g = _uniform_grid(3, 5)
    # Middle of cell (2,3): col covers 40-60%, row 33.3-66.6
    r, c = pick_anchor_cell(g, x_pct=50.0, y_pct=50.0)
    assert (r, c) == (2, 3)


def test_assign_cell_ids_straight_run():
    g = _uniform_grid(4, 10, writable=True)
    # Short comment → 1–3 cells; anchor top-left writable
    ids = assign_cell_ids(
        g,
        comment="Ok.",
        x_start_percent=5.0,
        y_position_percent=10.0,
        font_size_pts=11.0,
        max_wrap_rows=3,
        fontname="helv",
    )
    assert len(ids) >= 1
    assert all(isinstance(x, str) for x in ids)
    # First cell should be row1 col1 anchor neighbourhood
    ar, ac = pick_anchor_cell(g, 5.0, 10.0)
    assert ids[0] == cell_id_from_rc(ar, ac)


def test_assign_cell_ids_wrap_when_blocked():
    """Non-writable cell forces wrap to next row at same start column."""
    cells: list[Cell] = []
    rows, cols = 3, 6
    cw = 100.0 / cols
    rh = 100.0 / rows
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            w = not (r == 1 and c == 4)  # block (1,4)
            cells.append(
                _cell(
                    r,
                    c,
                    writable=w,
                    x0=(c - 1) * cw,
                    x1=c * cw,
                    y0=(r - 1) * rh,
                    y1=r * rh,
                )
            )
    g = PageCellGrid(
        page=1,
        rows=rows,
        cols=cols,
        cell_size_pts=CELL_SIZE_PTS,
        left_margin_pts=0.0,
        top_margin_pts=0.0,
        cells=cells,
    )
    # Need ~5 cells (117pt @ 25pt cells); anchor row 1 has only 3 writable
    # cells before the (1,4) block, so layout must wrap to row 2.
    medium_comment = "Word " * 4
    ids = assign_cell_ids(
        g,
        comment=medium_comment,
        x_start_percent=15.0,
        y_position_percent=15.0,
        font_size_pts=11.0,
        max_wrap_rows=3,
        fontname="helv",
    )
    assert len(ids) >= 5
    rows_seen = {rc_from_cell_id(cid)[0] for cid in ids}
    assert len(rows_seen) >= 2


def test_assign_cell_ids_relocate_when_row_empty():
    """Anchor row has no writable cells — relocate to nearest run."""
    cells: list[Cell] = []
    rows, cols = 2, 5
    cw = 100.0 / cols
    rh = 100.0 / rows
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            w = r == 2  # only row 2 writable
            cells.append(
                _cell(
                    r,
                    c,
                    writable=w,
                    x0=(c - 1) * cw,
                    x1=c * cw,
                    y0=(r - 1) * rh,
                    y1=r * rh,
                )
            )
    g = PageCellGrid(
        page=1,
        rows=rows,
        cols=cols,
        cell_size_pts=CELL_SIZE_PTS,
        left_margin_pts=0.0,
        top_margin_pts=0.0,
        cells=cells,
    )
    ids = assign_cell_ids(
        g,
        comment="Hi there",  # ~2 cells, fits within row 2's 5 writable cells
        x_start_percent=50.0,
        y_position_percent=15.0,
        font_size_pts=11.0,
        max_wrap_rows=2,
        fontname="helv",
    )
    assert len(ids) >= 1
    for cid in ids:
        row, _c = rc_from_cell_id(cid)
        assert row == 2, f"expected relocate to row 2, got {cid}"


def test_assign_bboxes_to_annotations_inplace():
    g = _uniform_grid(3, 8)
    items = [
        {
            "question_id": 1,
            "annotations": [
                {
                    "page_index": 0,
                    "y_position_percent": 50.0,
                    "x_start_percent": 50.0,
                    "x_end_percent": 90.0,
                    "comment": "Short.",
                    "is_positive": False,
                    "cell_ids": ["legacy"],  # stripped by bbox path
                }
            ],
        }
    ]
    assign_bboxes_to_annotations(
        items,
        [g],
        font_size_pts=11.0,
        fontname="helv",
        max_wrap_rows=3,
    )
    ann = items[0]["annotations"][0]
    assert "cell_ids" not in ann
    assert "bbox" in ann
    bb = ann["bbox"]
    assert set(bb.keys()) == {"page", "x1_percent", "y1_percent", "x2_percent", "y2_percent"}
    assert 0.0 <= bb["x1_percent"] < bb["x2_percent"] <= 100.0
    assert 0.0 <= bb["y1_percent"] < bb["y2_percent"] <= 100.0
    assert bb["page"] == ann["page_index"] + 1


def test_assign_bboxes_distinct_for_consecutive_annotations_on_same_page():
    """Two annotations anchored on the same page must not collapse onto the same writable run."""
    cells: list[Cell] = []
    rows, cols = 3, 16
    cw = 100.0 / cols
    rh = 100.0 / rows
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            w = r == 2  # only the middle row is writable
            cells.append(
                _cell(
                    r,
                    c,
                    writable=w,
                    x0=(c - 1) * cw,
                    x1=c * cw,
                    y0=(r - 1) * rh,
                    y1=r * rh,
                )
            )
    g = PageCellGrid(
        page=1,
        rows=rows,
        cols=cols,
        cell_size_pts=CELL_SIZE_PTS,
        left_margin_pts=0.0,
        top_margin_pts=0.0,
        cells=cells,
    )
    items = [
        {
            "question_id": 1,
            "annotations": [
                {
                    "page_index": 0,
                    "y_position_percent": 50.0,
                    "x_start_percent": 5.0,
                    "x_end_percent": 25.0,
                    "comment": "First short remark.",
                    "is_positive": True,
                },
                {
                    "page_index": 0,
                    "y_position_percent": 50.0,
                    "x_start_percent": 5.0,
                    "x_end_percent": 25.0,
                    "comment": "Second short remark.",
                    "is_positive": False,
                },
            ],
        }
    ]
    assign_bboxes_to_annotations(
        items,
        [g],
        font_size_pts=11.0,
        fontname="helv",
        max_wrap_rows=1,
    )
    a = items[0]["annotations"][0].get("bbox")
    b = items[0]["annotations"][1].get("bbox")
    assert a is not None and b is not None, "both annotations should fit on the writable row"
    assert a != b, f"second annotation duplicated first bbox: {a}"
    # No x-overlap when both fit on the same row
    assert a["x2_percent"] <= b["x1_percent"] or b["x2_percent"] <= a["x1_percent"]


def test_bbox_collapses_multi_row_cell_ids():
    g = _uniform_grid(3, 8, writable=True)
    lookup = {c.cell_id: c for c in g.cells}
    ids = ["E1", "F1", "G1", "H1", "A2", "B2"]
    bbox = _bbox_from_cell_ids(g, ids)
    assert bbox is not None
    assert bbox["page"] == 1
    h1, a2 = lookup["H1"], lookup["A2"]
    row1_cells = [lookup[cid] for cid in ids if cid in ("E1", "F1", "G1", "H1")]
    y1 = min(c.y_start_percent for c in row1_cells)
    y2 = a2.y_end_percent
    assert bbox["x1_percent"] == round(a2.x_start_percent, 2)
    assert bbox["x2_percent"] == round(h1.x_end_percent, 2)
    assert bbox["y1_percent"] == round(y1, 2)
    assert bbox["y2_percent"] == round(y2, 2)


def test_assign_cell_ids_returns_empty_when_no_full_fit():
    """When no writable run can hold ``cells_needed`` cells, return [] (not a
    degenerate partial). This prevents tiny bboxes that cannot render the
    comment legibly.
    """
    cells: list[Cell] = []
    rows, cols = 4, 6
    cw = 100.0 / cols
    rh = 100.0 / rows
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            # Single isolated writable cell at (4,1) — too small for any meaningful comment.
            w = (r == 4 and c == 1)
            cells.append(
                _cell(
                    r,
                    c,
                    writable=w,
                    x0=(c - 1) * cw,
                    x1=c * cw,
                    y0=(r - 1) * rh,
                    y1=r * rh,
                )
            )
    g = PageCellGrid(
        page=1,
        rows=rows,
        cols=cols,
        cell_size_pts=CELL_SIZE_PTS,
        left_margin_pts=0.0,
        top_margin_pts=0.0,
        cells=cells,
    )
    long_comment = (
        "Explicitly state how these teachings act as an antidote to "
        "contemporary communal violence and caste discrimination."
    )
    ids = assign_cell_ids(
        g,
        comment=long_comment,
        x_start_percent=50.0,
        y_position_percent=50.0,
        font_size_pts=11.0,
        max_wrap_rows=3,
        fontname="helv",
    )
    assert ids == [], f"expected [] when no run fits the comment, got {ids}"


def test_assign_bboxes_fallback_slot_when_no_room():
    """When Tier 1 (cell-grid) cannot fit the comment, Tier 2 margin-slot bbox is used.

    The bbox must always be present and must sit inside either the right margin
    (x1 >= 87.5) or the left margin (x2 <= 12.5) — never absent.
    """
    cells: list[Cell] = []
    rows, cols = 4, 6
    cw = 100.0 / cols
    rh = 100.0 / rows
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            w = (r == 4 and c == 1)
            cells.append(
                _cell(
                    r,
                    c,
                    writable=w,
                    x0=(c - 1) * cw,
                    x1=c * cw,
                    y0=(r - 1) * rh,
                    y1=r * rh,
                )
            )
    g = PageCellGrid(
        page=1,
        rows=rows,
        cols=cols,
        cell_size_pts=CELL_SIZE_PTS,
        left_margin_pts=0.0,
        top_margin_pts=0.0,
        cells=cells,
    )
    items = [
        {
            "question_id": 1,
            "annotations": [
                {
                    "page_index": 0,
                    "y_position_percent": 50.0,
                    "x_start_percent": 50.0,
                    "x_end_percent": 90.0,
                    "comment": (
                        "Explicitly state how these teachings act as an antidote "
                        "to contemporary communal violence and caste discrimination."
                    ),
                    "is_positive": False,
                }
            ],
        }
    ]
    assign_bboxes_to_annotations(
        items,
        [g],
        font_size_pts=11.0,
        fontname="helv",
        max_wrap_rows=3,
    )
    ann = items[0]["annotations"][0]
    assert "bbox" in ann, "bbox must always be present — Tier 2 slot should fire"
    bb = ann["bbox"]
    assert bb["page"] == 1
    in_right = bb["x1_percent"] >= 87.5
    in_left  = bb["x2_percent"] <= 12.5
    assert in_right or in_left, (
        f"fallback bbox should be in a margin, got {bb}"
    )


def test_assign_bboxes_skips_marking_box_cells():
    """Annotation must not snap onto cells covered by the item's marking_box."""
    g = _uniform_grid(4, 8, writable=True)
    items = [
        {
            "question_id": 1,
            # Marking box covers the entire top row (row 1: y 0–25%)
            "marking_box_page": 1,
            "marking_box_x1_percent": 0.0,
            "marking_box_y1_percent": 0.0,
            "marking_box_x2_percent": 100.0,
            "marking_box_y2_percent": 25.0,
            "annotations": [
                {
                    "page_index": 0,
                    # Anchor inside the marking box → must relocate below it.
                    "y_position_percent": 12.0,
                    "x_start_percent": 50.0,
                    "x_end_percent": 90.0,
                    "comment": "Annotation should not collide with marking box.",
                    "is_positive": True,
                }
            ],
        }
    ]
    assign_bboxes_to_annotations(
        items,
        [g],
        font_size_pts=11.0,
        fontname="helv",
        max_wrap_rows=3,
    )
    bb = items[0]["annotations"][0].get("bbox")
    assert bb is not None
    # bbox must lie entirely below the marking_box (y1 >= 25%)
    assert bb["y1_percent"] >= 25.0, (
        f"annotation overlapped marking_box (y 0-25%); got bbox {bb}"
    )


def test_assign_bboxes_marking_box_reservation_respected_across_items():
    """Q2's annotations on the same page must avoid Q1's marking_box too."""
    g = _uniform_grid(6, 8, writable=True)
    items = [
        {
            "question_id": 1,
            # Q1 marking box covers row 1 (y 0–16.67%)
            "marking_box_page": 1,
            "marking_box_x1_percent": 0.0,
            "marking_box_y1_percent": 0.0,
            "marking_box_x2_percent": 100.0,
            "marking_box_y2_percent": 16.0,
            "annotations": [],
        },
        {
            "question_id": 2,
            # Q2 marking box covers row 2 (y 16.67–33.33%)
            "marking_box_page": 1,
            "marking_box_x1_percent": 0.0,
            "marking_box_y1_percent": 17.0,
            "marking_box_x2_percent": 100.0,
            "marking_box_y2_percent": 33.0,
            "annotations": [
                {
                    "page_index": 0,
                    "y_position_percent": 5.0,  # try to land in Q1's marking_box
                    "x_start_percent": 10.0,
                    "x_end_percent": 90.0,
                    "comment": "Should avoid both marking boxes.",
                    "is_positive": False,
                }
            ],
        },
    ]
    assign_bboxes_to_annotations(
        items,
        [g],
        font_size_pts=11.0,
        fontname="helv",
        max_wrap_rows=2,
    )
    bb = items[1]["annotations"][0].get("bbox")
    assert bb is not None
    # Must be below both marking boxes (y >= 33%)
    assert bb["y1_percent"] >= 33.0, f"bbox collided with reserved cells: {bb}"


def test_hindi_font_fallback_width():
    """Invalid font name triggers heuristic path inside measurement."""
    n = estimate_remark_cell_width(
        "प्रश्न",
        font_size_pts=11.0,
        cell_size_pts=20.0,
        fontname="definitely-not-a-real-font-xyz",
    )
    assert n >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
