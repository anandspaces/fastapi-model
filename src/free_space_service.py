"""Free-space detection service: PDF/image → per-page annotatable zones.

Orchestrates PyMuPDF rasterization + free_space_utils grid scoring.
No Gemini, no DB, no auth — pure image analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np

from src.free_space_utils import (
    FreeZone,
    assign_annotations_to_free_zones,
    find_free_zones,
    get_grid_scores,
    merge_adjacent_zones,
    score_cell,
)

log = logging.getLogger(__name__)

_DEFAULT_ROWS = 20
_DEFAULT_COLS = 8
_DEFAULT_MIN_SCORE = 0.70  # Bug 2 fix: raised from 0.65 to compensate for looser std multiplier
_DEFAULT_TOP_N = 15
# Higher DPI keeps thin rules and strokes visible → cleaner grid scores and snapping.
_DEFAULT_DPI = 300


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------


def _rasterize_pdf_to_gray_arrays(
    pdf_bytes: bytes,
    dpi: int = _DEFAULT_DPI,
) -> list[np.ndarray]:
    """Convert every page of a PDF to a grayscale float32 numpy array (0=black, 1=white)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    arrays: list[np.ndarray] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        gray = (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width)
            .astype(np.float32)
            / 255.0
        )
        arrays.append(gray)
    doc.close()
    return arrays


def _png_bytes_to_gray_array(png_bytes: bytes) -> np.ndarray:
    """Decode a PNG/JPEG image (bytes) to a grayscale float32 array via PyMuPDF."""
    doc = fitz.open(stream=png_bytes, filetype="png")
    page = doc[0]
    pix = page.get_pixmap(colorspace=fitz.csGRAY)
    gray = (
        np.frombuffer(pix.samples, dtype=np.uint8)
        .reshape(pix.height, pix.width)
        .astype(np.float32)
        / 255.0
    )
    doc.close()
    return gray


# ---------------------------------------------------------------------------
# Per-page analysis
# ---------------------------------------------------------------------------


def analyze_page_free_space(
    gray: np.ndarray,
    *,
    rows: int = _DEFAULT_ROWS,
    cols: int = _DEFAULT_COLS,
    min_score: float = _DEFAULT_MIN_SCORE,
    top_n: int = _DEFAULT_TOP_N,
    merge: bool = True,
) -> list[FreeZone]:
    """Run grid analysis on a single grayscale page array.

    Returns a list of FreeZone objects sorted by score desc, then y asc.

    Args:
        gray:      Float32 array (h, w), values 0=black 1=white.
        rows/cols: Grid resolution.
        min_score: Emptiness threshold (0–1). Default 0.70 balances ruled-line noise vs empty cells.
        top_n:     Maximum zones returned per page.
        merge:     Whether to merge horizontally adjacent cells into wider zones.
    """
    h, w = gray.shape
    scores = get_grid_scores(gray, rows, cols)
    zones = find_free_zones(
        scores,
        rows,
        cols,
        img_w=w,
        img_h=h,
        min_score=min_score,
        top_n=top_n * 2,  # over-fetch before merge/trim
        exclude_left_cols=2,   # Bug 5 fix: 1→2 cols — A4 binding strip ≈ 15-20% = 2/8 cols
        exclude_bottom_rows=2, # Bug 5 fix: 1→2 rows — footer rules + contact line
    )
    if merge:
        zones = merge_adjacent_zones(zones, rows, cols)
    return zones[:top_n]


def analyze_pdf_free_space(
    pdf_bytes: bytes,
    *,
    rows: int = _DEFAULT_ROWS,
    cols: int = _DEFAULT_COLS,
    min_score: float = _DEFAULT_MIN_SCORE,
    top_n: int = _DEFAULT_TOP_N,
    dpi: int = _DEFAULT_DPI,
    merge: bool = True,
) -> list[list[FreeZone]]:
    """Analyse all pages of a PDF.

    Returns a list of per-page zone lists (0-indexed).
    E.g. result[0] = zones for page 1, result[1] = zones for page 2, ...
    """
    gray_arrays = _rasterize_pdf_to_gray_arrays(pdf_bytes, dpi=dpi)
    page_zones: list[list[FreeZone]] = []
    for page_idx, gray in enumerate(gray_arrays):
        zones = analyze_page_free_space(
            gray,
            rows=rows,
            cols=cols,
            min_score=min_score,
            top_n=top_n,
            merge=merge,
        )
        log.info(
            "free_space page=%d zones=%d (min_score=%.2f rows=%d cols=%d)",
            page_idx + 1,
            len(zones),
            min_score,
            rows,
            cols,
        )
        page_zones.append(zones)
    return page_zones


# ---------------------------------------------------------------------------
# Annotation snapping (re-export convenience)
# ---------------------------------------------------------------------------


def snap_items_annotations(
    items: list[dict[str, Any]],
    page_free_zones: list[list[FreeZone]],
    *,
    min_gap_pct: float = 8.0,  # Bug 3 fix: 12→8, matches assign_annotations_to_free_zones default
    max_per_zone: int = 2,
    max_shift_pct: float = 35.0,
) -> list[dict[str, Any]]:
    """Post-process smart-ocr items: assign each item's annotations to free zones.

    Uses capacity-aware greedy assignment with minimum gap enforcement and
    semantic bucket scoring to preserve intro / body / conclusion placement intent.
    """
    updated_items: list[dict[str, Any]] = []
    for item in items:
        out = dict(item)
        anns = out.get("annotations")
        if isinstance(anns, list) and anns:
            out["annotations"] = assign_annotations_to_free_zones(
                anns,
                page_free_zones,
                min_gap_pct=min_gap_pct,
                max_per_zone=max_per_zone,
                max_shift_pct=max_shift_pct,
            )
        updated_items.append(out)
    return updated_items


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def page_zones_to_api_response(
    page_zones: list[list[FreeZone]],
) -> list[dict[str, Any]]:
    """Convert page_zones to the API response shape."""
    return [
        {
            "pageIndex": page_idx,
            "freeZones": [z.to_dict() for z in zones],
        }
        for page_idx, zones in enumerate(page_zones)
    ]


def api_response_to_page_zones(
    pages: list[dict[str, Any]],
) -> list[list[FreeZone]]:
    """Reconstruct list[list[FreeZone]] from a /analyse/free-space pages array.

    Inverse of page_zones_to_api_response — used by the snap-annotations endpoint
    so callers can pass a cached free-space result without re-uploading the PDF.
    """
    if not pages:
        return []
    max_idx = max(int(p.get("pageIndex", 0)) for p in pages)
    result: list[list[FreeZone]] = [[] for _ in range(max_idx + 1)]
    for page in pages:
        idx = int(page.get("pageIndex", 0))
        if not (0 <= idx <= max_idx):
            continue
        zones: list[FreeZone] = []
        for z in page.get("freeZones", []):
            try:
                zones.append(
                    FreeZone(
                        x_start_percent=float(z["x_start_percent"]),
                        x_end_percent=float(z["x_end_percent"]),
                        y_start_percent=float(z["y_start_percent"]),
                        y_end_percent=float(z["y_end_percent"]),
                        score=float(z.get("score", 0.0)),
                        row=0,
                        col=0,
                    )
                )
            except (KeyError, TypeError, ValueError):
                pass
        result[idx] = zones
    return result
