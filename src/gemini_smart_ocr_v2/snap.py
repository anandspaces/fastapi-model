"""Pass-1 cell range parsing and snapping to PageCellGrid."""

from __future__ import annotations

import re
from typing import Any

from cell_grid_service_v4 import PageCellGrid, cell_id_from_rc, rc_from_cell_id


_ROW_RANGE_RE = re.compile(
    r"^\s*([A-Za-z]{1,3})(\d+)\s*:\s*([A-Za-z]{1,3})(\d+)\s*$"
)
_CELL_SINGLE_RE = re.compile(r"^\s*([A-Za-z]{1,3})(\d+)\s*$")


def parse_single_cell(s: str) -> tuple[int, int] | None:
    """Parse ``H15`` → (row, col)."""
    m = _CELL_SINGLE_RE.match((s or "").strip())
    if not m:
        return None
    letters, row_s = m.groups()
    row = int(row_s)
    _, col = rc_from_cell_id(f"{letters.upper()}{row}")
    return row, col


def parse_single_row_range(s: str) -> tuple[int, int, int] | None:
    """Parse ``H15:O15`` -> (row, start_col, end_col). Returns None if invalid."""
    m = _ROW_RANGE_RE.match((s or "").strip())
    if not m:
        return None
    c1_letters, r1, c2_letters, r2 = m.groups()
    if int(r1) != int(r2):
        return None  # multi-row rectangles need different handling
    row = int(r1)
    _, col_a = rc_from_cell_id(f"{c1_letters.upper()}{row}")
    _, col_b = rc_from_cell_id(f"{c2_letters.upper()}{row}")
    lo, hi = min(col_a, col_b), max(col_a, col_b)
    return row, lo, hi


def cells_from_range_strings(grid: PageCellGrid, ranges: list[str]) -> set[tuple[int, int]]:
    """Expand Pass-1 row-range strings (or single cells like ``H15``) to ``(row, col)`` set."""
    acc: set[tuple[int, int]] = set()
    for s in ranges:
        pr = parse_single_row_range(s)
        if pr is not None:
            row, lo, hi = pr
            for col in range(lo, hi + 1):
                if 1 <= row <= grid.rows and 1 <= col <= grid.cols:
                    acc.add((row, col))
            continue
        one = parse_single_cell(s)
        if one is not None:
            row, col = one
            if 1 <= row <= grid.rows and 1 <= col <= grid.cols:
                acc.add((row, col))
    return acc


def bounding_range_string(grid: PageCellGrid, ranges: list[str]) -> str | None:
    """Smallest A1:X9 rectangle covering all single-row ranges."""
    cells = cells_from_range_strings(grid, ranges)
    if not cells:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    a = cell_id_from_rc(min(rows), min(cols))
    b = cell_id_from_rc(max(rows), max(cols))
    return a if a == b else f"{a}:{b}"


def expand_row_range_to_cell_ids(row: int, c_lo: int, c_hi: int, grid: PageCellGrid) -> list[str]:
    """Expand inclusive column range on one row to cell IDs, clamped to grid."""
    out: list[str] = []
    for col in range(max(1, c_lo), min(grid.cols, c_hi) + 1):
        out.append(cell_id_from_rc(row, col))
    return out


def snap_row_ranges(
    ranges: list[str],
    grid: PageCellGrid,
    *,
    handwritten_answer: bool = False,
) -> list[str]:
    """Clamp ranges to the grid.

    v4 overlay: **writable** = green empty cells; **non-writable** = cells with ink.
    Handwritten answers must lie on ink cells → keep columns where ``not cell.writable``.
    Printed/marking/header blocks keep every in-grid column (ink or paper).
    """
    cell_by_id = {c.cell_id: c for c in grid.cells}
    kept_ranges: list[str] = []
    for raw in ranges:
        pr = parse_single_row_range(raw)
        rows_cols: list[tuple[int, int, int]] = []
        if pr is not None:
            row, lo, hi = pr
            if row < 1 or row > grid.rows:
                continue
            rows_cols.append((row, lo, hi))
        else:
            one = parse_single_cell(raw)
            if one is None:
                continue
            row, col = one
            rows_cols.append((row, col, col))

        for row, lo, hi in rows_cols:
            if row < 1 or row > grid.rows:
                continue
            ids: list[str] = []
            for col in range(lo, hi + 1):
                if col < 1 or col > grid.cols:
                    continue
                cid = cell_id_from_rc(row, col)
                cell = cell_by_id.get(cid)
                if cell is None:
                    continue
                if handwritten_answer and cell.writable:
                    # Ink cells only — v4 marks empty margins as writable (green).
                    continue
                ids.append(cid)
            if not ids:
                continue
            cols = sorted({cell_by_id[i].col for i in ids if i in cell_by_id})
            if not cols:
                continue
            a = cell_id_from_rc(row, cols[0])
            b = cell_id_from_rc(row, cols[-1])
            kept_ranges.append(a if a == b else f"{a}:{b}")
    return kept_ranges


def ordered_cell_ids_from_ranges(grid: PageCellGrid, ranges: list[str]) -> list[str]:
    """Expand row-range strings to sorted cell IDs (row-major, left-to-right)."""
    rc_list: list[tuple[int, int]] = []
    for s in ranges:
        pr = parse_single_row_range(s)
        if pr is not None:
            row, lo, hi = pr
            for col in range(lo, hi + 1):
                if 1 <= row <= grid.rows and 1 <= col <= grid.cols:
                    rc_list.append((row, col))
            continue
        one = parse_single_cell(s)
        if one is not None:
            row, col = one
            if 1 <= row <= grid.rows and 1 <= col <= grid.cols:
                rc_list.append((row, col))
    rc_list.sort(key=lambda t: (t[0], t[1]))
    out: list[str] = []
    seen: set[tuple[int, int]] = set()
    for rc in rc_list:
        if rc in seen:
            continue
        seen.add(rc)
        out.append(cell_id_from_rc(rc[0], rc[1]))
    return out


def leftmost_writable_cell_in_ranges(grid: PageCellGrid, ranges: list[str]) -> str | None:
    """Pick the top-left writable cell inside the union of single-row ranges."""
    cell_by_id = {c.cell_id: c for c in grid.cells}
    candidates: list[tuple[int, int]] = []
    for rc in cells_from_range_strings(grid, ranges):
        cid = cell_id_from_rc(rc[0], rc[1])
        cell = cell_by_id.get(cid)
        if cell is not None and cell.writable:
            candidates.append(rc)
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    r, c = candidates[0]
    return cell_id_from_rc(r, c)


def snap_blocks(blocks: list[dict[str, Any]], grid: PageCellGrid) -> list[dict[str, Any]]:
    """Mutate blocks in place: snap cells lists per kind."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        kind = str(b.get("kind") or "")
        cells_raw = b.get("cells")
        if not isinstance(cells_raw, list):
            cells_raw = []
        snapped = snap_row_ranges(
            [str(x) for x in cells_raw],
            grid,
            handwritten_answer=(kind == "handwritten_answer"),
        )
        if not snapped and kind == "handwritten_answer":
            snapped = snap_row_ranges(
                [str(x) for x in cells_raw],
                grid,
                handwritten_answer=False,
            )
        if not snapped and kind == "handwritten_answer":
            continue
        nb = dict(b)
        nb["cells"] = snapped
        out.append(nb)
    return out
