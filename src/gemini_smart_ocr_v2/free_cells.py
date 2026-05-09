"""Writable minus all occupied block cells — per-page FREE sets for Pass 3."""

from __future__ import annotations

import re
from typing import Any

from cell_grid_service_v4 import PageCellGrid, cell_id_from_rc

from .snap import cells_from_range_strings

_BLOCK_PAGE_RE = re.compile(r"^p(\d+)-", re.I)


def writable_rc_set(grid: PageCellGrid) -> set[tuple[int, int]]:
    return {(c.row, c.col) for c in grid.cells if c.writable}


def ranges_from_rc_set(grid: PageCellGrid, cells: set[tuple[int, int]]) -> list[str]:
    """Compact per-row horizontal segments for prompts (Excel-style ranges)."""
    if not cells:
        return []
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    out: list[str] = []
    for row in sorted(by_row.keys()):
        cols = sorted(set(by_row[row]))
        i = 0
        while i < len(cols):
            start_col = cols[i]
            end_col = cols[i]
            j = i + 1
            while j < len(cols) and cols[j] == end_col + 1:
                end_col = cols[j]
                j += 1
            a = cell_id_from_rc(row, start_col)
            b = cell_id_from_rc(row, end_col)
            out.append(a if a == b else f"{a}:{b}")
            i = j
    return out


def occupied_from_all_blocks(
    grids_by_page: dict[int, PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
) -> dict[int, set[tuple[int, int]]]:
    """Union of every block's cells on each page."""
    per_page: dict[int, set[tuple[int, int]]] = {}
    for bid, blk in flat_blocks.items():
        bid_str = str(blk.get("block_id") or bid)
        m = _BLOCK_PAGE_RE.match(bid_str)
        page = int(m.group(1)) if m else 1
        g = grids_by_page.get(page)
        if g is None:
            continue
        cells = blk.get("cells")
        if not isinstance(cells, list):
            continue
        rs = [str(x) for x in cells]
        per_page.setdefault(page, set()).update(cells_from_range_strings(g, rs))
    return per_page


def occupied_from_refs(
    grids_by_page: dict[int, PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    refs: list[str],
) -> dict[int, set[tuple[int, int]]]:
    """Union of cells for only the listed ``block_id`` refs."""
    per_page: dict[int, set[tuple[int, int]]] = {}
    for ref in refs:
        blk = flat_blocks.get(ref)
        if not blk:
            continue
        bid_str = str(blk.get("block_id") or ref)
        m = _BLOCK_PAGE_RE.match(bid_str)
        page = int(m.group(1)) if m else 1
        g = grids_by_page.get(page)
        if g is None:
            continue
        cells = blk.get("cells")
        if not isinstance(cells, list):
            continue
        rs = [str(x) for x in cells]
        per_page.setdefault(page, set()).update(cells_from_range_strings(g, rs))
    return per_page


def pass3_spatial_payload(
    grids: list[PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    answer_refs: list[str],
    pages: set[int],
) -> dict[str, Any]:
    """Pass-3 prompt attachment: global FREE cells + per-answer OCCUPIED cells."""
    by_page = {g.page: g for g in grids}
    free_sets = compute_free_cells(grids, flat_blocks)
    occ_answer = occupied_from_refs(by_page, flat_blocks, answer_refs)
    payload: dict[str, Any] = {}
    for p in sorted(pages):
        g = by_page.get(p)
        if g is None:
            continue
        payload[f"FREE_PAGE_{p}"] = ranges_from_rc_set(g, free_sets.get(p, set()))
        payload[f"OCCUPIED_PAGE_{p}"] = ranges_from_rc_set(g, occ_answer.get(p, set()))
    return payload


def compute_free_cells(
    grids: list[PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    *,
    items: list[dict[str, Any]] | None = None,
) -> dict[int, set[tuple[int, int]]]:
    """Per-page FREE = writable minus union of all Pass-1 ``flat_blocks`` cells.

    ``items`` is reserved for future per-question FREE tightening (same grid
    arithmetic as the plan); currently unused — spatial prompts use
    :func:`pass3_spatial_payload` for OCCUPIED vs this global FREE.
    """
    del items  # placeholder — avoids silent misuse while keeping a stable signature
    by_page = {g.page: g for g in grids}
    occupied = occupied_from_all_blocks(by_page, flat_blocks)
    free: dict[int, set[tuple[int, int]]] = {}
    for g in grids:
        w = writable_rc_set(g)
        occ = occupied.get(g.page, set())
        free[g.page] = w - occ
    return free


def free_and_occupied_payload(
    grids: list[PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    *,
    question_pages: set[int],
) -> dict[str, Any]:
    """Build JSON-serializable FREE_PAGE_N / OCCUPIED_PAGE_N for Gemini prompts."""
    by_page = {g.page: g for g in grids}
    occupied = occupied_from_all_blocks(by_page, flat_blocks)
    free_sets = compute_free_cells(grids, flat_blocks)
    payload: dict[str, Any] = {}
    for p in sorted(question_pages):
        g = by_page.get(p)
        if g is None:
            continue
        occ = occupied.get(p, set())
        fr = free_sets.get(p, set())
        payload[f"FREE_PAGE_{p}"] = ranges_from_rc_set(g, fr)
        payload[f"OCCUPIED_PAGE_{p}"] = ranges_from_rc_set(g, occ)
    return payload
