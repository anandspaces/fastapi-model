"""Backend geometry for Pass-3 annotation comment placement.

Gemini supplies semantic anchors + preferred_side + priority; this module
computes ``comment_rows`` from FREE cells and the writable-run catalogue.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from cell_grid_service_v4 import PageCellGrid, WritableRun, cell_id_from_rc

from src.gemini_smart_ocr_v2.snap import cells_from_range_strings

log = logging.getLogger(__name__)

# Margin bands (Excel-style columns on A4 10pt grid — see CLAUDE.md).
_LEFT_MAX_COL = 11   # … K
_RIGHT_MIN_COL = 48  # AV …


def _run_cells(run: WritableRun) -> list[tuple[int, int]]:
    return [(run.row, c) for c in range(run.start_col, run.end_col + 1)]


def _run_fully_available(
    run: WritableRun,
    free_rc: set[tuple[int, int]],
    consumed: set[tuple[int, int]],
) -> bool:
    for rc in _run_cells(run):
        if rc not in free_rc or rc in consumed:
            return False
    return True


def _side_ok(run: WritableRun, preferred_side: str) -> bool:
    ps = (preferred_side or "right").lower()
    if ps == "left":
        return run.end_col <= _LEFT_MAX_COL
    if ps == "right":
        return run.start_col >= _RIGHT_MIN_COL
    if ps == "gap":
        w = run.end_col - run.start_col + 1
        return w >= 18 or (run.start_col <= 15 and run.end_col >= 40)
    return True


def _anchor_anchor_row(grid: PageCellGrid, anchor_rows: list[str]) -> int:
    cells = cells_from_range_strings(grid, [str(x) for x in anchor_rows])
    if not cells:
        return max(1, grid.rows // 2)
    rs = [r for r, _ in cells]
    return int(round(sum(rs) / len(rs)))


def _estimate_needed_cols(text_len: int, font_pts: float) -> int:
    # ~0.55 * font_pts width per character in a 10pt-wide cell column at ~16pt font
    chars_per_col = max(0.35, font_pts * 0.06)
    cols = int(math.ceil(text_len * chars_per_col / 8.0))
    return max(8, min(48, cols))


def _tokens_for_block(
    grid: PageCellGrid,
    row: int,
    col_lo: int,
    col_hi: int,
    free_rc: set[tuple[int, int]],
    consumed: set[tuple[int, int]],
    *,
    max_rows: int = 4,
) -> list[str]:
    """Build up to ``max_rows`` contiguous single-row ranges with full [col_lo,col_hi]."""
    out: list[str] = []
    for dr in range(max_rows):
        r = row + dr
        if r < 1 or r > grid.rows:
            break
        ok = True
        for c in range(col_lo, col_hi + 1):
            rc = (r, c)
            if rc not in free_rc or rc in consumed:
                ok = False
                break
        if not ok:
            break
        a = cell_id_from_rc(r, col_lo)
        b = cell_id_from_rc(r, col_hi)
        out.append(a if a == b else f"{a}:{b}")
    return out


def plan_comment_placement(
    grid: PageCellGrid,
    anchor: dict[str, Any],
    preferred_side: str,
    text_len: int,
    font_pts: float,
    free_rc: set[tuple[int, int]],
    consumed_page: set[tuple[int, int]],
) -> list[str]:
    """Return Excel-style ``comment_rows`` tokens placed in FREE cells near the anchor."""
    anchor_rows = list(anchor.get("rows") or [])
    anchor_row = _anchor_anchor_row(grid, anchor_rows)
    needed_cols = _estimate_needed_cols(text_len, max(14.0, font_pts))

    sides_order = [preferred_side.lower()]
    for s in ("right", "left", "gap"):
        if s not in sides_order:
            sides_order.append(s)

    best: tuple[float, WritableRun | None] = (-1.0, None)
    for side in sides_order:
        for run in grid.runs:
            if not _run_fully_available(run, free_rc, consumed_page):
                continue
            if not _side_ok(run, side):
                continue
            width = run.end_col - run.start_col + 1
            prox = 1.0 / (1.0 + abs(run.row - anchor_row) / 5.0)
            fit = min(1.0, width / max(needed_cols, 1))
            balance = 1.0 - abs(run.row / max(grid.rows, 1) - 0.5) * 0.5
            cont = min(1.0, width / 12.0)
            score = 0.4 * prox + 0.3 * fit + 0.2 * balance + 0.1 * cont
            if score > best[0]:
                best = (score, run)
        if best[1] is not None:
            break

    run = best[1]
    if run is None:
        log.debug("annotation_planner: no candidate run for anchor_row=%s", anchor_row)
        return []

    tokens = _tokens_for_block(
        grid,
        run.row,
        run.start_col,
        run.end_col,
        free_rc,
        consumed_page,
        max_rows=4,
    )
    return tokens


def replan_comment_rows(
    grid: PageCellGrid,
    ann: dict[str, Any],
    free_rc: set[tuple[int, int]],
    consumed_page: set[tuple[int, int]],
) -> list[str]:
    """Try alternate ``preferred_side`` values when validation finds no usable cells."""
    anchor = ann.get("anchor") or {}
    comment = str(ann.get("comment") or "")
    try:
        font_pts = float(ann.get("comment_font_pts", 14.0))
    except (TypeError, ValueError):
        font_pts = 14.0
    for side in ("right", "left", "gap"):
        rows = plan_comment_placement(
            grid, anchor, side, len(comment), font_pts, free_rc, consumed_page
        )
        if rows:
            return rows
    return []


def reserve_cells_for_tokens(
    grid: PageCellGrid,
    tokens: list[str],
    free_rc: set[tuple[int, int]],
    consumed_page: set[tuple[int, int]],
) -> None:
    """Remove planned cells from ``free_rc`` and add to ``consumed_page``."""
    for tok in tokens:
        for rc in cells_from_range_strings(grid, [str(tok)]):
            free_rc.discard(rc)
            consumed_page.add(rc)


def _priority_value(ann: dict[str, Any]) -> int:
    p = ann.get("priority", 2)
    try:
        return int(p)
    except (TypeError, ValueError):
        return 2


def reorder_annotations_by_priority(annotations: list[dict[str, Any]]) -> None:
    """Stable-sort annotations by priority (1 first) then original index."""
    indexed = list(enumerate(annotations))
    indexed.sort(key=lambda it: (_priority_value(it[1]), it[0]))
    annotations[:] = [it[1] for it in indexed]


def assign_comment_rows_for_evaluation(
    annotations: list[dict[str, Any]],
    grids_by_page: dict[int, PageCellGrid],
    initial_free_template: dict[int, set[tuple[int, int]]],
) -> None:
    """Populate ``comment_rows`` from geometry; respects ``priority`` (1=highest).

    Uses a **working copy** of FREE sets so full ``initial_free_template`` remains
    intact for ``validate_annotations_for_item`` (which performs real reservation).
    """
    working_free: dict[int, set[tuple[int, int]]] = {
        p: set(s) for p, s in initial_free_template.items()
    }
    consumed_by_page: dict[int, set[tuple[int, int]]] = {}

    reorder_annotations_by_priority(annotations)

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        try:
            page = int(ann.get("page", 1))
        except (TypeError, ValueError):
            continue
        grid = grids_by_page.get(page)
        if grid is None:
            continue

        preferred_side = str(ann.pop("preferred_side", "right")).lower()
        ann.pop("priority", None)

        anchor = ann.get("anchor") or {}
        comment = str(ann.get("comment") or "")
        try:
            font_pts = float(ann.get("comment_font_pts", 14.0))
        except (TypeError, ValueError):
            font_pts = 14.0

        fr = working_free.setdefault(page, set())
        consumed_pg = consumed_by_page.setdefault(page, set())

        planned = plan_comment_placement(
            grid,
            anchor,
            preferred_side,
            len(comment),
            font_pts,
            fr,
            consumed_pg,
        )
        ann["comment_rows"] = planned
        reserve_cells_for_tokens(grid, planned, fr, consumed_pg)
