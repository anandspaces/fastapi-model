"""Pass-3 annotation validator for ``comment_rows``.

Parses **per-row** cell vocabulary (same as snap.py):

- Single-row span: ``H15:O15`` (start/end columns on one row).
- Single cell: ``H15``.
- **Ragged layout**: ``comment_rows`` is a list of such tokens — each entry can use
  a different column span on its row (multi-row, multi-width).

Rules:

1. Expand each token to ``(row, col)`` pairs; clamp to grid bounds (implicit in parsing).
2. Each cell must be in the page **FREE** set (writable gap cells for this Pass-3 context).
3. **Repair**: within each token, keep only a **contiguous left-to-right prefix** of
   cells that are still FREE and not yet **consumed on this page**; drop the rest of
   that token (invalid / overlaps earlier annotations).
4. **Reject**: if after repair an annotation has no cells left, drop the whole annotation.
5. **Track consumption**: mutates per-page FREE and a **per-page** consumed set so
   later annotations on the **same page** cannot reuse those cells (annotations on
   other pages are independent).
"""

from __future__ import annotations

import logging
from typing import Any

from cell_grid_service_v4 import PageCellGrid, cell_id_from_rc

from .snap import cells_from_range_strings, parse_single_cell, parse_single_row_range

log = logging.getLogger(__name__)


def _ranges_from_cell_set(grid: PageCellGrid, cells: set[tuple[int, int]]) -> list[str]:
    """Merge accepted cells into compact single-row range strings (preserves ragged rows)."""
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
            sc = cols[i]
            ec = cols[i]
            j = i + 1
            while j < len(cols) and cols[j] == ec + 1:
                ec = cols[j]
                j += 1
            a = cell_id_from_rc(row, sc)
            b = cell_id_from_rc(row, ec)
            out.append(a if a == b else f"{a}:{b}")
            i = j
    return out


def parse_comment_row_token(grid: PageCellGrid, token: str) -> tuple[list[tuple[int, int]], str | None]:
    """Parse one ``comment_rows`` entry into ordered ``(row, col)`` cells left-to-right.

    Returns ``(cells, error)`` where ``error`` is set if the token is unusable.
    """
    s = (token or "").strip()
    if not s:
        return [], "empty"

    pr = parse_single_row_range(s)
    if pr is not None:
        row, lo, hi = pr
        if row < 1 or row > grid.rows:
            return [], "row_oob"
        cells: list[tuple[int, int]] = []
        for col in range(lo, hi + 1):
            if col < 1 or col > grid.cols:
                return [], "col_oob"
            cells.append((row, col))
        return cells, None

    one = parse_single_cell(s)
    if one is None:
        return [], "unparseable"
    row, col = one
    if row < 1 or row > grid.rows or col < 1 or col > grid.cols:
        return [], "oob"
    return [(row, col)], None


def _prefix_free_cells(
    ordered_cells: list[tuple[int, int]],
    free_rc: set[tuple[int, int]],
    consumed_page: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Left-to-right contiguous prefix where each cell is FREE and not yet consumed on this page."""
    prefix: list[tuple[int, int]] = []
    for rc in ordered_cells:
        if rc not in free_rc or rc in consumed_page:
            break
        prefix.append(rc)
    return prefix


def validate_and_reserve_annotation_cells(
    grid: PageCellGrid,
    ann: dict[str, Any],
    free_rc: set[tuple[int, int]],
    consumed_page: set[tuple[int, int]],
) -> dict[str, Any] | None:
    """Repair ``comment_rows`` per token (prefix); reserve accepted cells on this page."""
    rows_raw = ann.get("comment_rows")
    if not isinstance(rows_raw, list):
        return None

    accepted: set[tuple[int, int]] = set()
    kept_row_strings: list[str] = []

    for raw in rows_raw:
        s = str(raw).strip()
        if not s:
            continue
        ordered, err = parse_comment_row_token(grid, s)
        if err or not ordered:
            log.debug("validate: skip token %r (%s)", s, err or "no cells")
            continue
        prefix = _prefix_free_cells(ordered, free_rc, consumed_page)
        if not prefix:
            continue
        for rc in prefix:
            free_rc.discard(rc)
            consumed_page.add(rc)
            accepted.add(rc)

        row = prefix[0][0]
        cols = sorted({c for r, c in prefix if r == row})
        if cols:
            a = cell_id_from_rc(row, cols[0])
            b = cell_id_from_rc(row, cols[-1])
            kept_row_strings.append(a if a == b else f"{a}:{b}")

    if not accepted:
        return None
    out = dict(ann)
    out["comment_rows"] = (
        kept_row_strings if kept_row_strings else _ranges_from_cell_set(grid, accepted)
    )
    return out


def validate_annotations_for_item(
    grids_by_page: dict[int, PageCellGrid],
    annotations: list[dict[str, Any]],
    initial_free: dict[int, set[tuple[int, int]]],
) -> list[dict[str, Any]]:
    """Validate in list order; FREE and per-page consumed sets shrink across annotations."""
    free_copy: dict[int, set[tuple[int, int]]] = {
        p: set(s) for p, s in initial_free.items()
    }
    consumed_by_page: dict[int, set[tuple[int, int]]] = {}
    cleaned: list[dict[str, Any]] = []
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
        fr = free_copy.setdefault(page, set())
        consumed_pg = consumed_by_page.setdefault(page, set())
        fixed = validate_and_reserve_annotation_cells(grid, ann, fr, consumed_pg)
        if fixed:
            cleaned.append(fixed)
        else:
            log.warning(
                "validate_annotations: dropped annotation page=%s (no valid FREE cells)",
                page,
            )
    return cleaned


def snap_anchor_rows_to_grid(grid: PageCellGrid, anchor: dict[str, Any]) -> dict[str, Any]:
    """Drop anchor row tokens that do not parse to in-grid single-row ranges."""
    if not isinstance(anchor, dict):
        return anchor
    rows = anchor.get("rows")
    if not isinstance(rows, list):
        return anchor
    kept: list[str] = []
    for r in rows:
        s = str(r).strip()
        if not s:
            continue
        if parse_single_row_range(s) or parse_single_cell(s):
            cells = cells_from_range_strings(grid, [s])
            if cells:
                kept.append(s)
    out = dict(anchor)
    if kept:
        out["rows"] = kept
    elif anchor.get("type") not in (None, "none"):
        out["type"] = "none"
        out.pop("rows", None)
    return out
