"""Lay out examiner annotation remarks on the cell grid and expose placement as bbox.

Uses ``cell_grid_service`` grids + PyMuPDF text width measurement to estimate how many
grid cells a remark occupies at a nominal font size, then lays out cells in reading
order (wrap then relocate). The API surfaces a single ``bbox`` (percent + page) per
annotation instead of ``cell_ids`` so consumers need no grid metadata.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import fitz  # PyMuPDF

from cell_grid_service import (
    CELL_SIZE_PTS,
    Cell,
    PageCellGrid,
    cell_id_from_rc,
    find_writable_runs,
    rc_from_cell_id,
)

log = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name, "") or "").strip()
    return raw or default


# --- Tunables (override via env) -------------------------------------------------

REMARK_FONT_SIZE_PTS: float = _env_float("REMARK_FONT_SIZE_PTS", 11.0)
REMARK_FONTNAME_EN: str = _env_str("REMARK_FONTNAME_EN", "helv")
REMARK_FONTNAME_HI: str = _env_str("REMARK_FONTNAME_HI", "helv")
REMARK_MAX_WRAP_ROWS: int = max(1, _env_int("REMARK_MAX_WRAP_ROWS", 3))
REMARK_SEARCH_RADIUS_RUNS: int = max(1, _env_int("REMARK_SEARCH_RADIUS_RUNS", 8))


# --- Text width ------------------------------------------------------------------


def estimate_remark_cell_width(
    text: str,
    *,
    font_size_pts: float = REMARK_FONT_SIZE_PTS,
    cell_size_pts: float = CELL_SIZE_PTS,
    fontname: str = REMARK_FONTNAME_EN,
) -> int:
    """Return minimum number of grid cells (columns) needed for ``text`` on one line."""
    s = (text or "").strip()
    if not s:
        return 0
    w_pt = _measure_text_width_pt(s, fontname=fontname, font_size_pts=font_size_pts)
    return max(1, math.ceil(w_pt / float(cell_size_pts)))


def _measure_text_width_pt(
    text: str,
    *,
    fontname: str,
    font_size_pts: float,
) -> float:
    try:
        return float(
            fitz.get_text_length(text, fontname=fontname, fontsize=font_size_pts)
        )
    except Exception:
        # Unknown fontname (e.g. Devanagari not registered) — heuristic advance
        return len(text) * font_size_pts * 0.65


# --- Anchor ----------------------------------------------------------------------


def pick_anchor_cell(
    page_grid: PageCellGrid,
    x_pct: float,
    y_pct: float,
) -> tuple[int, int]:
    """Map annotation percentages to a 1-based (row, col) inside the grid."""
    x_pct = max(0.0, min(100.0, float(x_pct)))
    y_pct = max(0.0, min(100.0, float(y_pct)))

    best: tuple[int, int] | None = None
    best_d = float("inf")

    for cell in page_grid.cells:
        if (
            cell.x_start_percent <= x_pct <= cell.x_end_percent
            and cell.y_start_percent <= y_pct <= cell.y_end_percent
        ):
            cx = (cell.x_start_percent + cell.x_end_percent) / 2.0
            cy = (cell.y_start_percent + cell.y_end_percent) / 2.0
            d = (cx - x_pct) ** 2 + (cy - y_pct) ** 2
            if d < best_d:
                best_d = d
                best = (cell.row, cell.col)

    if best is not None:
        return best

    # Fallback: nearest cell centre in percent space
    for cell in page_grid.cells:
        cx = (cell.x_start_percent + cell.x_end_percent) / 2.0
        cy = (cell.y_start_percent + cell.y_end_percent) / 2.0
        d = (cx - x_pct) ** 2 + (cy - y_pct) ** 2
        if d < best_d:
            best_d = d
            best = (cell.row, cell.col)

    if best is not None:
        return best
    return (1, 1)


def _build_writable_set(page_grid: PageCellGrid) -> set[tuple[int, int]]:
    return {(c.row, c.col) for c in page_grid.cells if c.writable}


def _collect_cells_wrap(
    page_grid: PageCellGrid,
    writable: set[tuple[int, int]],
    start_row: int,
    start_col: int,
    cells_needed: int,
    max_wrap_rows: int,
) -> list[str]:
    """Greedy horizontal walk with row wrap at ``start_col`` (or col 1 if blocked)."""
    out: list[str] = []
    rows, cols = page_grid.rows, page_grid.cols

    for row_off in range(max_wrap_rows):
        r = start_row + row_off
        if r < 1 or r > rows:
            break
        if row_off == 0:
            c_start = start_col
        else:
            c_start = start_col if (r, start_col) in writable else 1

        c = c_start
        while c <= cols and len(out) < cells_needed:
            if (r, c) in writable:
                out.append(cell_id_from_rc(r, c))
                c += 1
            else:
                break
        if len(out) >= cells_needed:
            break
    return out


def _run_centre_rc(run: list[Any]) -> tuple[float, float]:
    rs = [c.row for c in run]
    cs = [c.col for c in run]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _relocate_collect(
    page_grid: PageCellGrid,
    writable: set[tuple[int, int]],
    anchor_row: int,
    anchor_col: int,
    cells_needed: int,
    max_wrap_rows: int,
) -> list[str]:
    """Pick nearest writable run and lay out from its left edge with wrapping."""
    runs = find_writable_runs(page_grid, min_length=1)
    if not runs:
        return []

    scored: list[tuple[float, int, list[Any]]] = []
    for run in runs:
        cr, cc = _run_centre_rc(run)
        dist = math.hypot(cr - anchor_row, cc - anchor_col)
        scored.append((dist, -len(run), run))
    scored.sort(key=lambda t: (t[0], t[1]))

    for _dist, _neglen, run in scored[: REMARK_SEARCH_RADIUS_RUNS]:
        run = sorted(run, key=lambda c: c.col)
        sr, sc = run[0].row, run[0].col
        got = _collect_cells_wrap(page_grid, writable, sr, sc, cells_needed, max_wrap_rows)
        if len(got) >= cells_needed:
            return got[:cells_needed]
    # Last resort: longest run anywhere — still must satisfy cells_needed.
    # A partial placement here would yield a bbox too small to render the comment
    # legibly (e.g. 4%-wide column on the page edge), so we'd rather drop bbox.
    by_len = sorted(runs, key=len, reverse=True)
    for run in by_len[:5]:
        run = sorted(run, key=lambda c: c.col)
        sr, sc = run[0].row, run[0].col
        got = _collect_cells_wrap(page_grid, writable, sr, sc, cells_needed, max_wrap_rows)
        if len(got) >= cells_needed:
            return got[:cells_needed]
    return []


def assign_cell_ids(
    page_grid: PageCellGrid,
    *,
    comment: str,
    x_start_percent: float,
    y_position_percent: float,
    font_size_pts: float = REMARK_FONT_SIZE_PTS,
    max_wrap_rows: int = REMARK_MAX_WRAP_ROWS,
    fontname: str = REMARK_FONTNAME_EN,
    excluded: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Compute ordered ``cell_ids`` for ``comment`` anchored at the annotation position.

    ``excluded`` lets the caller reserve cells already taken by previous annotations
    on the same page so consecutive remarks do not collapse onto the same writable
    run; cells in this set are removed from the writable pool before placement.
    """
    text = (comment or "").strip()
    if not text:
        return []

    cells_needed = estimate_remark_cell_width(
        text,
        font_size_pts=font_size_pts,
        cell_size_pts=page_grid.cell_size_pts,
        fontname=fontname,
    )
    writable = _build_writable_set(page_grid)
    if excluded:
        writable = writable - excluded
    if not writable:
        return []

    ar, ac = pick_anchor_cell(page_grid, x_start_percent, y_position_percent)

    # If anchor cell not writable, nudge right on same row to first writable
    sr, sc = ar, ac
    if (sr, sc) not in writable:
        found = False
        for c in range(sc, page_grid.cols + 1):
            if (sr, c) in writable:
                sc = c
                found = True
                break
        if not found:
            for c in range(1, sc):
                if (sr, c) in writable:
                    sc = c
                    found = True
                    break

    primary: list[str] = []
    if (sr, sc) in writable:
        primary = _collect_cells_wrap(
            page_grid, writable, sr, sc, cells_needed, max_wrap_rows
        )
    if len(primary) >= cells_needed:
        return primary[:cells_needed]

    relocated = _relocate_collect(
        page_grid,
        writable,
        ar,
        ac,
        cells_needed,
        max_wrap_rows,
    )
    if len(relocated) >= cells_needed:
        return relocated[:cells_needed]
    # No run on this page can hold the comment in full → drop placement so
    # ``assign_bboxes_to_annotations`` omits ``bbox`` rather than emitting one
    # too narrow to render the comment.
    return []


def _bbox_from_cell_ids(
    page_grid: PageCellGrid,
    cell_ids: list[str],
    *,
    cells_by_id: dict[str, Cell] | None = None,
) -> dict[str, Any] | None:
    """Bounding rectangle (percent, page 1-based) covering all laid-out cells."""
    if not cell_ids:
        return None
    lookup = cells_by_id or {c.cell_id: c for c in page_grid.cells}
    rects: list[Cell] = []
    for cid in cell_ids:
        c = lookup.get(cid.strip())
        if c is not None:
            rects.append(c)
    if not rects:
        return None
    x1 = min(c.x_start_percent for c in rects)
    y1 = min(c.y_start_percent for c in rects)
    x2 = max(c.x_end_percent for c in rects)
    y2 = max(c.y_end_percent for c in rects)
    return {
        "page": page_grid.page,
        "x1_percent": round(float(x1), 2),
        "y1_percent": round(float(y1), 2),
        "x2_percent": round(float(x2), 2),
        "y2_percent": round(float(y2), 2),
    }


# --- Slot pool (fallback bbox allocation) ----------------------------------------

_SLOT_POOL_SIZE     = 20    # horizontal strips across full page height
_SLOT_FALLBACK_X1   = 87.5  # right-margin x used only when a strip has zero writable cells
_SLOT_FALLBACK_X2   = 100.0


def _build_slot_pool(page_grid: PageCellGrid) -> list[dict[str, Any]]:
    """Pre-partition the full page into _SLOT_POOL_SIZE equal-height strips.

    Within each strip all writable cells are considered regardless of x position —
    free space can appear anywhere (mid-page gaps, wide diagram areas, etc.).
    Only when a strip contains no writable cells does the slot fall back to
    the right-margin geometry so there is always a non-empty bbox.

    Each slot is a dict with keys ``bbox`` and ``used``.
    """
    slots: list[dict[str, Any]] = []
    slot_h = 100.0 / _SLOT_POOL_SIZE

    for i in range(_SLOT_POOL_SIZE):
        y1 = i * slot_h
        y2 = min(100.0, (i + 1) * slot_h)

        # All writable cells in this horizontal strip — full page width
        strip_writable = [
            c for c in page_grid.cells
            if c.writable
            and c.y_start_percent >= y1
            and c.y_end_percent   <= y2
        ]
        if strip_writable:
            bbox: dict[str, Any] = {
                "page":        page_grid.page,
                "x1_percent":  round(min(c.x_start_percent for c in strip_writable), 2),
                "y1_percent":  round(min(c.y_start_percent for c in strip_writable), 2),
                "x2_percent":  round(max(c.x_end_percent   for c in strip_writable), 2),
                "y2_percent":  round(max(c.y_end_percent   for c in strip_writable), 2),
            }
        else:
            # No writable cells in this strip → right margin as geometric fallback
            bbox = {
                "page":        page_grid.page,
                "x1_percent":  _SLOT_FALLBACK_X1,
                "y1_percent":  round(y1, 2),
                "x2_percent":  _SLOT_FALLBACK_X2,
                "y2_percent":  round(y2, 2),
            }
        slots.append({"bbox": bbox, "used": False, "y_center": (y1 + y2) / 2.0})

    return slots


def _pop_nearest_slot(pool: list[dict[str, Any]], y_pct: float) -> dict[str, Any] | None:
    """Return and mark-used the slot whose centre is closest to ``y_pct``."""
    unused = [s for s in pool if not s["used"]]
    if not unused:
        return None
    best = min(unused, key=lambda s: abs(s["y_center"] - y_pct))
    best["used"] = True
    return best["bbox"]


def _synthesize_bbox(
    page_no: int,
    x1: float,
    x2: float,
    y: float,
    half_h: float = 1.5,
) -> dict[str, Any]:
    """Last-resort bbox synthesised purely from annotation coordinates."""
    return {
        "page":        page_no,
        "x1_percent":  round(max(0.0,   x1),          2),
        "y1_percent":  round(max(0.0,   y - half_h),  2),
        "x2_percent":  round(min(100.0, x2),          2),
        "y2_percent":  round(min(100.0, y + half_h),  2),
    }


def _cells_in_rect(
    page_grid: PageCellGrid,
    *,
    x1_pct: float,
    y1_pct: float,
    x2_pct: float,
    y2_pct: float,
) -> set[tuple[int, int]]:
    """Return ``(row, col)`` of every cell whose centre falls inside the percent rect."""
    if x2_pct < x1_pct:
        x1_pct, x2_pct = x2_pct, x1_pct
    if y2_pct < y1_pct:
        y1_pct, y2_pct = y2_pct, y1_pct
    out: set[tuple[int, int]] = set()
    for cell in page_grid.cells:
        cx = (cell.x_start_percent + cell.x_end_percent) / 2.0
        cy = (cell.y_start_percent + cell.y_end_percent) / 2.0
        if x1_pct <= cx <= x2_pct and y1_pct <= cy <= y2_pct:
            out.add((cell.row, cell.col))
    return out


def _reserve_marking_box_cells(
    items: list[dict[str, Any]],
    grid_by_page: dict[int, PageCellGrid],
    consumed_by_page: dict[int, set[tuple[int, int]]],
) -> None:
    """Pre-mark each item's ``marking_box_*`` cells as consumed.

    Without this, ``assign_cell_ids`` is free to lay an annotation directly on top of
    the score box because the writable pool only excludes cells used by earlier
    *annotations*. Reserving up-front means every annotation (this item's or any
    later item's on the same page) snaps to a different writable run.
    """
    for item in items:
        page_raw = item.get("marking_box_page")
        if page_raw is None:
            continue
        try:
            page_no = int(page_raw)
        except (TypeError, ValueError):
            continue
        grid = grid_by_page.get(page_no)
        if grid is None:
            continue
        try:
            x1 = float(item.get("marking_box_x1_percent"))
            y1 = float(item.get("marking_box_y1_percent"))
            x2 = float(item.get("marking_box_x2_percent"))
            y2 = float(item.get("marking_box_y2_percent"))
        except (TypeError, ValueError):
            continue
        rc = _cells_in_rect(
            grid,
            x1_pct=x1,
            y1_pct=y1,
            x2_pct=x2,
            y2_pct=y2,
        )
        if rc:
            consumed_by_page.setdefault(page_no, set()).update(rc)


def assign_bboxes_to_annotations(
    items: list[dict[str, Any]],
    page_grids: list[PageCellGrid],
    *,
    font_size_pts: float = REMARK_FONT_SIZE_PTS,
    fontname: str = REMARK_FONTNAME_EN,
    max_wrap_rows: int = REMARK_MAX_WRAP_ROWS,
) -> None:
    """Mutate each ``annotations[*]`` dict to add ``bbox`` (no ``cell_ids``).

    Every annotation is guaranteed to receive a ``bbox`` via a three-tier strategy:

    Tier 1 — ``assign_cell_ids``: finds writable cells for optimal comment placement.
    Tier 2 — margin slot pool: pre-partitioned non-overlapping strips in the right/left
              margin, each sized to fit a comment; nearest unused slot to the annotation
              y-position is claimed.  40 slots per page (20 right + 20 left) cover any
              realistic annotation density.
    Tier 3 — coordinate synthesis: bbox derived directly from annotation coordinates
              when no page grid is available.

    Cells used by earlier annotations (and the examiner score box) are still excluded
    from Tier 1 so Tier 1 placements remain collision-free.  Tier 2 slots are
    independent of the cell-grid consumed set — they rely on pre-partitioned geometry.
    """
    grid_by_page = {g.page: g for g in page_grids}
    lookup_cache: dict[int, dict[str, Cell]] = {
        g.page: {c.cell_id: c for c in g.cells} for g in page_grids
    }
    consumed_by_page: dict[int, set[tuple[int, int]]] = {}
    _reserve_marking_box_cells(items, grid_by_page, consumed_by_page)

    # Build slot pools once per page (Tier 2 fallback)
    slot_pools: dict[int, list[dict[str, Any]]] = {
        g.page: _build_slot_pool(g) for g in page_grids
    }

    for item in items:
        anns = item.get("annotations")
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            try:
                pi = int(ann.get("page_index", 0))
            except (TypeError, ValueError):
                continue
            page_no = pi + 1

            # Extract coordinates early — needed by all fallback tiers
            try:
                x0 = float(ann.get("x_start_percent", 50.0))
            except (TypeError, ValueError):
                x0 = 50.0
            try:
                x1_end = float(ann.get("x_end_percent", min(100.0, x0 + 20.0)))
            except (TypeError, ValueError):
                x1_end = min(100.0, x0 + 20.0)
            try:
                yp = float(ann.get("y_position_percent", 50.0))
            except (TypeError, ValueError):
                yp = 50.0
            comment = str(ann.get("comment", "") or "")
            ann.pop("cell_ids", None)

            grid = grid_by_page.get(page_no)
            if grid is None:
                # Tier 3: no grid for this page
                ann["bbox"] = _synthesize_bbox(page_no, x0, x1_end, yp)
                continue

            consumed = consumed_by_page.setdefault(page_no, set())

            # ── Tier 1: cell-grid placement ───────────────────────────────────
            bbox: dict[str, Any] | None = None
            try:
                ids = assign_cell_ids(
                    grid,
                    comment=comment,
                    x_start_percent=x0,
                    y_position_percent=yp,
                    font_size_pts=font_size_pts,
                    max_wrap_rows=max_wrap_rows,
                    fontname=fontname,
                    excluded=consumed,
                )
                for cid in ids:
                    try:
                        consumed.add(rc_from_cell_id(cid))
                    except ValueError:
                        continue
                bbox = _bbox_from_cell_ids(
                    grid,
                    ids,
                    cells_by_id=lookup_cache.get(page_no),
                )
            except Exception:
                log.warning(
                    "assign_bboxes_to_annotations assign_cell_ids failed page=%s comment_prefix=%r",
                    page_no,
                    comment[:80],
                    exc_info=True,
                )

            # ── Tier 2: nearest unused margin slot ────────────────────────────
            if bbox is None:
                bbox = _pop_nearest_slot(slot_pools.get(page_no, []), yp)

            # ── Tier 3: coordinate synthesis (slot pool exhausted or no grid) ─
            if bbox is None:
                bbox = _synthesize_bbox(page_no, x0, x1_end, yp)

            ann["bbox"] = bbox  # always set; never absent


def assign_cell_ids_to_annotations(
    items: list[dict[str, Any]],
    page_grids: list[PageCellGrid],
    *,
    font_size_pts: float = REMARK_FONT_SIZE_PTS,
    fontname: str = REMARK_FONTNAME_EN,
    max_wrap_rows: int = REMARK_MAX_WRAP_ROWS,
) -> None:
    """Mutate each ``annotations[*]`` dict to add ``cell_ids`` when possible.

    Deprecated: prefer :func:`assign_bboxes_to_annotations` for coordinate-only API
    responses (``bbox`` instead of ``cell_ids``).
    """
    grid_by_page = {g.page: g for g in page_grids}

    for item in items:
        anns = item.get("annotations")
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            try:
                pi = int(ann.get("page_index", 0))
            except (TypeError, ValueError):
                continue
            page_no = pi + 1
            grid = grid_by_page.get(page_no)
            if grid is None:
                ann["cell_ids"] = []
                continue
            try:
                x0 = float(ann.get("x_start_percent", 50.0))
            except (TypeError, ValueError):
                x0 = 50.0
            try:
                yp = float(ann.get("y_position_percent", 50.0))
            except (TypeError, ValueError):
                yp = 50.0
            comment = str(ann.get("comment", "") or "")
            try:
                ann["cell_ids"] = assign_cell_ids(
                    grid,
                    comment=comment,
                    x_start_percent=x0,
                    y_position_percent=yp,
                    font_size_pts=font_size_pts,
                    max_wrap_rows=max_wrap_rows,
                    fontname=fontname,
                )
            except Exception:
                log.warning(
                    "assign_cell_ids failed page=%s comment_prefix=%r",
                    page_no,
                    comment[:80],
                    exc_info=True,
                )
                ann["cell_ids"] = []
