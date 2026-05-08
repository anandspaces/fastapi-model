"""Cell-grid service — FastAPI-facing wrapper around ``cell_grid_service_v4``.

Single entry point used by ``main.py``: turn a raw PDF into a per-page grid
(meta + writable runs + writable regions + optional per-cell list) and a
companion debug overlay PDF.

Cells are NOT exposed to the response by default — the per-page array is
~3500 entries which would dominate JSON size.  Pass ``include_cells=True``
for debug callers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cell_grid_service_v4 import (
    CELL_SIZE_PTS,
    PageCellGrid,
    analyze_pdf_cell_grid_v4,
    cell_id_from_rc,
    draw_cell_grid_overlay_v4,
    rc_from_cell_id,
)

log = logging.getLogger(__name__)

# Public re-exports — keep import surface small for callers.
__all__ = [
    "CELL_SIZE_PTS",
    "PageCellGrid",
    "build_cell_grid",
    "build_overlay_pdf",
    "cell_grid_meta_payload",
    "cell_grid_response_payload",
    "cell_id_from_rc",
    "rc_from_cell_id",
]


def build_cell_grid(
    pdf_bytes: bytes,
    *,
    cell_size_pts: float = CELL_SIZE_PTS,
    dpi: int = 300,
) -> list[PageCellGrid]:
    """Run v4 analysis on ``pdf_bytes`` and return per-page grids."""
    return analyze_pdf_cell_grid_v4(pdf_bytes, cell_size_pts=cell_size_pts, dpi=dpi)


def build_overlay_pdf(
    pdf_bytes: bytes,
    grids: list[PageCellGrid],
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Render the v4 grid as a vector overlay onto ``pdf_bytes`` for Gemini."""
    return draw_cell_grid_overlay_v4(pdf_bytes, grids, output_path, **kwargs)


def cell_grid_meta_payload(grids: list[PageCellGrid]) -> list[dict[str, Any]]:
    """Per-page meta block — the ONLY data the frontend needs to convert any
    cell id (from anywhere in the response) into a percent rectangle.

    Compact: just rows, cols, cell_size_pts, margins, page dimensions.
    """
    return [
        {
            "page": g.page,
            "rows": g.rows,
            "cols": g.cols,
            "cell_size_pts": g.cell_size_pts,
            "left_margin_pts": g.left_margin_pts,
            "top_margin_pts": g.top_margin_pts,
            "page_w_pts": g.page_w_pts,
            "page_h_pts": g.page_h_pts,
        }
        for g in grids
    ]


def cell_grid_response_payload(
    grids: list[PageCellGrid],
    *,
    include_cells: bool = False,
    include_runs: bool = True,
    include_regions: bool = True,
) -> list[dict[str, Any]]:
    """Full per-page payload (meta + runs + regions + optional cells).

    Use when the consumer (Gemini, tests, debug UI) needs the writable
    placement zones in the same response.
    """
    out: list[dict[str, Any]] = []
    for g in grids:
        page_payload: dict[str, Any] = {
            "page": g.page,
            "rows": g.rows,
            "cols": g.cols,
            "cell_size_pts": g.cell_size_pts,
            "left_margin_pts": g.left_margin_pts,
            "top_margin_pts": g.top_margin_pts,
            "page_w_pts": g.page_w_pts,
            "page_h_pts": g.page_h_pts,
        }
        if include_runs:
            page_payload["writable_runs"] = [r.to_dict() for r in g.runs]
        if include_regions:
            page_payload["writable_regions"] = [r.to_dict() for r in g.regions]
        if include_cells:
            page_payload["cells"] = [c.to_dict() for c in g.cells]
        out.append(page_payload)
    return out
