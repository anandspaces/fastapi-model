"""Cell-Grid v4 — fine-grained writable-space analyser for answer-sheet PDFs.

Improvements over v1/v3:
    * 12-pt cells (~4.2 mm) — 4x finer than v3's 25-pt cells
    * Centered margins — leftover page space split equally on both sides
    * Stable score formula:   score = clip(mean - std, 0, 1)
    * Morphological cleanup of the ink mask before per-cell scoring:
        - close(3x3) merges single-pixel specks
        - erode(3x3) shrinks ink so cell-edge bleed does not disqualify cells
    * Emits THREE views per page:
        - cells     (every cell, with writable bool)
        - runs      (1-D contiguous writable runs per row)
        - regions   (2-D 4-connected components of writable cells)
    * Vector overlay PDF: faded full grid + bright writable cells + region
      outlines + tiny cell-ID labels — readable by Gemini for cell-id-based
      annotation placement.

Excel-style cell IDs: A1, B1, ..., Z1, AA1, AB1, ...
Range syntax: "A1" (single) or "C12:H12" / "D5:K9" (rect span).
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from scipy import ndimage

CELL_SIZE_PTS: float = 24.0
RASTER_DPI: int = 300
# Stricter thresholds so faint / light-gray ink still counts as "occupied":
#   * brightness cut-off raised 0.45 -> 0.75 to catch pencil & faded pen strokes
#   * min writable score raised 0.70 -> 0.88 (cells must be near-paper-white)
#   * max ink ratio tightened 0.04 -> 0.005 (almost zero ink pixels permitted)
INK_BRIGHTNESS_MAX: float = 0.75
WRITABLE_MIN_SCORE: float = 0.88
MAX_INK_PIXEL_RATIO: float = 0.005
MIN_RUN_LENGTH: int = 2
MIN_REGION_CELLS: int = 2


# ── Cell-ID helpers ─────────────────────────────────────────────────────────


def cell_id_from_rc(row: int, col: int) -> str:
    if row < 1 or col < 1:
        raise ValueError("row and col must be >= 1")
    letters: list[str] = []
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        letters.append(chr(65 + rem))
    return f"{''.join(reversed(letters))}{row}"


def rc_from_cell_id(cell_id: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", cell_id.strip())
    if not m:
        raise ValueError(f"Invalid cell_id: {cell_id}")
    letters = m.group(1).upper()
    row = int(m.group(2))
    col = 0
    for ch in letters:
        col = col * 26 + (ord(ch) - 64)
    return row, col


def range_id_from_cells(start: str, end: str) -> str:
    return start if start == end else f"{start}:{end}"


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class Cell:
    cell_id: str
    row: int
    col: int
    page: int
    x_start_percent: float
    x_end_percent: float
    y_start_percent: float
    y_end_percent: float
    pdf_x1: float
    pdf_y1: float
    pdf_x2: float
    pdf_y2: float
    score: float
    ink_ratio: float
    writable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WritableRun:
    page: int
    row: int
    start_col: int
    end_col: int
    start_cell_id: str
    end_cell_id: str
    range_id: str
    cell_count: int
    x_start_percent: float
    x_end_percent: float
    y_start_percent: float
    y_end_percent: float
    pdf_x1: float
    pdf_y1: float
    pdf_x2: float
    pdf_y2: float
    mean_score: float
    mean_ink_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WritableRegion:
    page: int
    region_id: str
    bbox_range_id: str
    cell_ids: list[str]
    cell_count: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    x_start_percent: float
    x_end_percent: float
    y_start_percent: float
    y_end_percent: float
    pdf_x1: float
    pdf_y1: float
    pdf_x2: float
    pdf_y2: float
    mean_score: float
    mean_ink_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageCellGrid:
    page: int
    rows: int
    cols: int
    cell_size_pts: float
    left_margin_pts: float
    top_margin_pts: float
    page_w_pts: float
    page_h_pts: float
    cells: list[Cell] = field(default_factory=list)
    runs: list[WritableRun] = field(default_factory=list)
    regions: list[WritableRegion] = field(default_factory=list)

    def to_dict(self, *, include_cells: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "page": self.page,
            "rows": self.rows,
            "cols": self.cols,
            "cell_size_pts": self.cell_size_pts,
            "left_margin_pts": self.left_margin_pts,
            "top_margin_pts": self.top_margin_pts,
            "page_w_pts": self.page_w_pts,
            "page_h_pts": self.page_h_pts,
            "runs": [r.to_dict() for r in self.runs],
            "regions": [r.to_dict() for r in self.regions],
        }
        if include_cells:
            d["cells"] = [c.to_dict() for c in self.cells]
        return d


# ── Core analysis ───────────────────────────────────────────────────────────


def analyze_pdf_cell_grid_v4(
    pdf_bytes: bytes,
    *,
    cell_size_pts: float = CELL_SIZE_PTS,
    dpi: int = RASTER_DPI,
    min_score: float = WRITABLE_MIN_SCORE,
    max_ink_ratio: float = MAX_INK_PIXEL_RATIO,
    ink_brightness_max: float = INK_BRIGHTNESS_MAX,
    min_run_length: int = MIN_RUN_LENGTH,
    min_region_cells: int = MIN_REGION_CELLS,
) -> list[PageCellGrid]:
    if cell_size_pts <= 0:
        raise ValueError("cell_size_pts must be > 0")
    if min_run_length < 1:
        raise ValueError("min_run_length must be >= 1")
    if min_region_cells < 1:
        raise ValueError("min_region_cells must be >= 1")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72.0
    results: list[PageCellGrid] = []

    try:
        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            cols = max(1, math.floor(page_w / cell_size_pts))
            rows = max(1, math.floor(page_h / cell_size_pts))

            left_margin = round((page_w - cols * cell_size_pts) / 2.0, 2)
            top_margin = round((page_h - rows * cell_size_pts) / 2.0, 2)

            gray = _rasterize_gray(page, dpi=dpi)
            ink = _build_clean_ink_mask(gray, ink_brightness_max=ink_brightness_max)

            score_grid, ink_grid = _grid_metrics_vectorised(
                gray=gray,
                ink=ink,
                rows=rows,
                cols=cols,
                cell_size_pts=cell_size_pts,
                left_margin=left_margin,
                top_margin=top_margin,
                scale=scale,
            )
            writable_grid = (score_grid >= min_score) & (ink_grid <= max_ink_ratio)

            cells = _build_cell_list(
                page_no=page_no,
                rows=rows,
                cols=cols,
                cell_size_pts=cell_size_pts,
                left_margin=left_margin,
                top_margin=top_margin,
                page_w=page_w,
                page_h=page_h,
                score_grid=score_grid,
                ink_grid=ink_grid,
                writable_grid=writable_grid,
            )

            runs = _extract_row_runs(
                page_no=page_no,
                writable_grid=writable_grid,
                score_grid=score_grid,
                ink_grid=ink_grid,
                cell_size_pts=cell_size_pts,
                left_margin=left_margin,
                top_margin=top_margin,
                page_w=page_w,
                page_h=page_h,
                min_run_length=min_run_length,
            )

            regions = _extract_regions(
                page_no=page_no,
                writable_grid=writable_grid,
                score_grid=score_grid,
                ink_grid=ink_grid,
                cell_size_pts=cell_size_pts,
                left_margin=left_margin,
                top_margin=top_margin,
                page_w=page_w,
                page_h=page_h,
                min_region_cells=min_region_cells,
            )

            results.append(
                PageCellGrid(
                    page=page_no,
                    rows=rows,
                    cols=cols,
                    cell_size_pts=cell_size_pts,
                    left_margin_pts=left_margin,
                    top_margin_pts=top_margin,
                    page_w_pts=round(page_w, 2),
                    page_h_pts=round(page_h, 2),
                    cells=cells,
                    runs=runs,
                    regions=regions,
                )
            )
    finally:
        doc.close()

    return results


# ── Image preparation ──────────────────────────────────────────────────────


def _rasterize_gray(page: fitz.Page, *, dpi: int) -> np.ndarray:
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    arr = (
        np.frombuffer(pix.samples, dtype=np.uint8)
        .reshape(pix.height, pix.width)
        .astype(np.float32)
        / 255.0
    )
    return arr


def _build_clean_ink_mask(gray: np.ndarray, *, ink_brightness_max: float) -> np.ndarray:
    """Binary ink mask after morphological cleanup — strict variant.

    1. Threshold gray < ink_brightness_max → raw ink (high cut-off catches
       faint pencil / faded pen strokes).
    2. Dilate 3x3 (1 iter) — grows faint specks so a halo of light gray
       around a stroke gets pulled into the ink mask.
    3. Close 3x3 (1 iter) — fills single-pixel gaps inside strokes.

    Erosion is intentionally omitted: the v3-style 1-px shrink lets cells
    that brush against ink pass the ink_ratio gate, which is exactly what
    we are trying to prevent here.
    """
    raw = gray < ink_brightness_max
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(raw, structure=structure, iterations=1)
    closed = ndimage.binary_closing(dilated, structure=structure, iterations=1)
    return closed


def _cell_metrics(gray_cell: np.ndarray, ink_cell: np.ndarray) -> tuple[float, float]:
    """Score and ink_ratio for a single cell — kept for tests / debugging.

    score = clip(mean(gray) - std(gray), 0, 1).  Stable on edge cells where
    v1's ``mean * (1 - std)`` collapses (CLAUDE.md "Critical Bug Fixes").

    ink_ratio = fraction of pixels inside the (cleaned) ink mask.
    """
    if gray_cell.size == 0:
        return 0.0, 1.0
    mean = float(np.mean(gray_cell))
    std = float(np.std(gray_cell))
    score = max(0.0, min(1.0, mean - std))
    ink_ratio = float(np.mean(ink_cell)) if ink_cell.size > 0 else 0.0
    return score, ink_ratio


def _grid_metrics_vectorised(
    *,
    gray: np.ndarray,
    ink: np.ndarray,
    rows: int,
    cols: int,
    cell_size_pts: float,
    left_margin: float,
    top_margin: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cell mean / std / ink_ratio in one vectorised pass.

    Uses integral images (summed-area tables) so each cell is computed in O(1)
    via four lookups, broadcast over the full grid. ~100x faster than the
    Python double loop for ~3500 cells/page.

    Returns (score_grid, ink_grid), both shape (rows, cols).
    """
    H, W = gray.shape

    ys = np.empty(rows + 1, dtype=np.int64)
    xs = np.empty(cols + 1, dtype=np.int64)
    for r in range(rows + 1):
        ys[r] = int((top_margin + r * cell_size_pts) * scale)
    for c in range(cols + 1):
        xs[c] = int((left_margin + c * cell_size_pts) * scale)
    np.clip(ys, 0, H, out=ys)
    np.clip(xs, 0, W, out=xs)

    gray64 = gray.astype(np.float64, copy=False)
    int_gray = _integral(gray64)
    int_gray2 = _integral(gray64 * gray64)
    int_ink = _integral(ink.astype(np.float64))

    sum_gray = _box_sum(int_gray, ys, xs)
    sum_gray2 = _box_sum(int_gray2, ys, xs)
    sum_ink = _box_sum(int_ink, ys, xs)

    counts = ((ys[1:] - ys[:-1])[:, None] * (xs[1:] - xs[:-1])[None, :]).astype(np.float64)
    counts_safe = np.where(counts > 0, counts, 1.0)

    mean = sum_gray / counts_safe
    var = np.maximum(0.0, sum_gray2 / counts_safe - mean * mean)
    std = np.sqrt(var)

    score = np.clip(mean - std, 0.0, 1.0)
    ink_ratio = sum_ink / counts_safe

    empty_mask = counts <= 0
    score = np.where(empty_mask, 0.0, score)
    ink_ratio = np.where(empty_mask, 1.0, ink_ratio)

    return score.astype(np.float32), ink_ratio.astype(np.float32)


def _integral(arr: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero-padded first row/col so box sums are
    expressed as a single subtraction of four corner values."""
    H, W = arr.shape
    out = np.zeros((H + 1, W + 1), dtype=np.float64)
    np.cumsum(arr, axis=0, out=out[1:, 1:])
    np.cumsum(out[1:, 1:], axis=1, out=out[1:, 1:])
    return out


def _box_sum(integral: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Vectorised box-sum lookup over a regular grid of cells.

    ``integral`` shape (H+1, W+1). ``ys`` length rows+1. ``xs`` length cols+1.
    Returns shape (rows, cols).
    """
    yy_top = ys[:-1, None]
    yy_bot = ys[1:, None]
    xx_lft = xs[None, :-1]
    xx_rgt = xs[None, 1:]
    return (
        integral[yy_bot, xx_rgt]
        - integral[yy_top, xx_rgt]
        - integral[yy_bot, xx_lft]
        + integral[yy_top, xx_lft]
    )


def _build_cell_list(
    *,
    page_no: int,
    rows: int,
    cols: int,
    cell_size_pts: float,
    left_margin: float,
    top_margin: float,
    page_w: float,
    page_h: float,
    score_grid: np.ndarray,
    ink_grid: np.ndarray,
    writable_grid: np.ndarray,
) -> list[Cell]:
    cells: list[Cell] = []
    for r in range(1, rows + 1):
        top_pts = top_margin + (r - 1) * cell_size_pts
        bot_pts = top_pts + cell_size_pts
        y_start_pct = round((top_pts / page_h) * 100.0, 3)
        y_end_pct = round((bot_pts / page_h) * 100.0, 3)
        for c in range(1, cols + 1):
            x_pts = left_margin + (c - 1) * cell_size_pts
            x_end_pts = x_pts + cell_size_pts
            score = round(float(score_grid[r - 1, c - 1]), 4)
            ratio = round(float(ink_grid[r - 1, c - 1]), 4)
            cells.append(
                Cell(
                    cell_id=cell_id_from_rc(r, c),
                    row=r,
                    col=c,
                    page=page_no,
                    x_start_percent=round((x_pts / page_w) * 100.0, 3),
                    x_end_percent=round((x_end_pts / page_w) * 100.0, 3),
                    y_start_percent=y_start_pct,
                    y_end_percent=y_end_pct,
                    pdf_x1=round(x_pts, 2),
                    pdf_y1=round(top_pts, 2),
                    pdf_x2=round(x_end_pts, 2),
                    pdf_y2=round(bot_pts, 2),
                    score=score,
                    ink_ratio=ratio,
                    writable=bool(writable_grid[r - 1, c - 1]),
                )
            )
    return cells


# ── Run / region extraction ────────────────────────────────────────────────


def _extract_row_runs(
    *,
    page_no: int,
    writable_grid: np.ndarray,
    score_grid: np.ndarray,
    ink_grid: np.ndarray,
    cell_size_pts: float,
    left_margin: float,
    top_margin: float,
    page_w: float,
    page_h: float,
    min_run_length: int,
) -> list[WritableRun]:
    rows, cols = writable_grid.shape
    runs: list[WritableRun] = []
    for r0 in range(rows):
        c0 = 0
        while c0 < cols:
            if not writable_grid[r0, c0]:
                c0 += 1
                continue
            c1 = c0
            while c1 + 1 < cols and writable_grid[r0, c1 + 1]:
                c1 += 1
            length = c1 - c0 + 1
            if length >= min_run_length:
                row = r0 + 1
                start_col = c0 + 1
                end_col = c1 + 1
                start_cell = cell_id_from_rc(row, start_col)
                end_cell = cell_id_from_rc(row, end_col)
                x_start = left_margin + (start_col - 1) * cell_size_pts
                x_end = left_margin + end_col * cell_size_pts
                y_start = top_margin + (row - 1) * cell_size_pts
                y_end = top_margin + row * cell_size_pts
                run_scores = score_grid[r0, c0 : c1 + 1]
                run_inks = ink_grid[r0, c0 : c1 + 1]
                runs.append(
                    WritableRun(
                        page=page_no,
                        row=row,
                        start_col=start_col,
                        end_col=end_col,
                        start_cell_id=start_cell,
                        end_cell_id=end_cell,
                        range_id=range_id_from_cells(start_cell, end_cell),
                        cell_count=length,
                        x_start_percent=round((x_start / page_w) * 100.0, 3),
                        x_end_percent=round((x_end / page_w) * 100.0, 3),
                        y_start_percent=round((y_start / page_h) * 100.0, 3),
                        y_end_percent=round((y_end / page_h) * 100.0, 3),
                        pdf_x1=round(x_start, 2),
                        pdf_y1=round(y_start, 2),
                        pdf_x2=round(x_end, 2),
                        pdf_y2=round(y_end, 2),
                        mean_score=round(float(np.mean(run_scores)), 4),
                        mean_ink_ratio=round(float(np.mean(run_inks)), 4),
                    )
                )
            c0 = c1 + 1
    return runs


def _extract_regions(
    *,
    page_no: int,
    writable_grid: np.ndarray,
    score_grid: np.ndarray,
    ink_grid: np.ndarray,
    cell_size_pts: float,
    left_margin: float,
    top_margin: float,
    page_w: float,
    page_h: float,
    min_region_cells: int,
    max_regions: int = 32,
    min_dim: int = 2,
) -> list[WritableRegion]:
    """Maximal axis-aligned all-writable rectangles → WritableRegion list.

    Connected components misbehave on answer sheets — handwriting leaves enough
    micro-whitespace that all writable cells form one giant blob. Maximal
    rectangles instead surface coherent placement zones (a margin column, a
    half-page diagram gap, an inter-paragraph band) that Gemini can pick from.

    Algorithm:
      1. Build per-column heights (consecutive writable rows ending at row r).
      2. Stack-based largest-rectangle-in-histogram per row, emitting every
         maximal rectangle.
      3. Filter: area ≥ ``min_region_cells`` and both dimensions ≥ ``min_dim``.
      4. Drop rectangles strictly contained in a larger emitted rectangle.
      5. Cap at ``max_regions`` (largest first).
    """
    raw = _maximal_rectangles(writable_grid, min_cells=min_region_cells, min_dim=min_dim)
    if not raw:
        return []
    raw.sort(key=lambda t: -((t[1] - t[0] + 1) * (t[3] - t[2] + 1)))
    deduped: list[tuple[int, int, int, int]] = []
    for r1, r2, c1, c2 in raw:
        contained = False
        for R1, R2, C1, C2 in deduped:
            if R1 <= r1 and R2 >= r2 and C1 <= c1 and C2 >= c2:
                contained = True
                break
        if not contained:
            deduped.append((r1, r2, c1, c2))
        if len(deduped) >= max_regions:
            break

    regions: list[WritableRegion] = []
    for idx, (r1, r2, c1, c2) in enumerate(deduped, start=1):
        row_start = r1 + 1
        row_end = r2 + 1
        col_start = c1 + 1
        col_end = c2 + 1
        cell_ids = [
            cell_id_from_rc(rr, cc)
            for rr in range(row_start, row_end + 1)
            for cc in range(col_start, col_end + 1)
        ]
        bbox_range = range_id_from_cells(
            cell_id_from_rc(row_start, col_start),
            cell_id_from_rc(row_end, col_end),
        )
        x_start = left_margin + (col_start - 1) * cell_size_pts
        x_end = left_margin + col_end * cell_size_pts
        y_start = top_margin + (row_start - 1) * cell_size_pts
        y_end = top_margin + row_end * cell_size_pts
        sub_score = score_grid[r1 : r2 + 1, c1 : c2 + 1]
        sub_ink = ink_grid[r1 : r2 + 1, c1 : c2 + 1]
        regions.append(
            WritableRegion(
                page=page_no,
                region_id=f"R{idx}",
                bbox_range_id=bbox_range,
                cell_ids=cell_ids,
                cell_count=len(cell_ids),
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                x_start_percent=round((x_start / page_w) * 100.0, 3),
                x_end_percent=round((x_end / page_w) * 100.0, 3),
                y_start_percent=round((y_start / page_h) * 100.0, 3),
                y_end_percent=round((y_end / page_h) * 100.0, 3),
                pdf_x1=round(x_start, 2),
                pdf_y1=round(y_start, 2),
                pdf_x2=round(x_end, 2),
                pdf_y2=round(y_end, 2),
                mean_score=round(float(np.mean(sub_score)), 4),
                mean_ink_ratio=round(float(np.mean(sub_ink)), 4),
            )
        )
    return regions


def _maximal_rectangles(
    writable: np.ndarray, *, min_cells: int, min_dim: int
) -> list[tuple[int, int, int, int]]:
    """Return raw maximal rectangle list.  Coordinates are 0-indexed and
    inclusive: (row_start, row_end, col_start, col_end).
    """
    rows, cols = writable.shape
    heights = np.zeros(cols, dtype=np.int64)
    found: set[tuple[int, int, int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if writable[r, c] else 0
        stack: list[tuple[int, int]] = []  # (col_index, height)
        c = 0
        while c <= cols:
            cur = int(heights[c]) if c < cols else 0
            start = c
            while stack and stack[-1][1] >= cur and (c == cols or stack[-1][1] > cur or cur == 0):
                col0, h = stack.pop()
                row_start = r - h + 1
                row_end = r
                col_start = col0
                col_end = c - 1
                width = col_end - col_start + 1
                height = h
                if (
                    height >= min_dim
                    and width >= min_dim
                    and width * height >= min_cells
                ):
                    found.add((row_start, row_end, col_start, col_end))
                start = col0
            if cur > 0 and (not stack or stack[-1][1] < cur):
                stack.append((start, cur))
            c += 1
    return list(found)


# ── Vector overlay ─────────────────────────────────────────────────────────


def _column_letters(col: int) -> str:
    return cell_id_from_rc(1, col)[:-1]


def draw_cell_grid_overlay_v4(
    pdf_bytes: bytes,
    page_grids: list[PageCellGrid],
    output_path: str | Path,
    *,
    show_full_grid: bool = True,
    show_writable: bool = True,
    show_regions: bool = True,
    show_axis_labels: bool = True,
    show_cell_labels: bool = False,
    axis_label_font_size: float = 5.0,
    cell_label_font_size: float = 3.0,
) -> Path:
    """Render the v4 grid onto the PDF as a vector overlay.

    Layers (bottom → top):
      faded gray  : grid lines (vertical + horizontal)
      bright green: writable cells (filled translucent rect)
      blue thick  : 2-D writable region bboxes with region-id + range label
      axis labels : column letters above row 1; row numbers left of col 1
                    (cheap — O(rows + cols) instead of O(rows * cols))

    Per-cell labels are off by default — Gemini can derive any cell id from
    axis labels. Pass ``show_cell_labels=True`` for debug overlays.

    Drawing is batched via ``page.new_shape()`` + ``shape.commit()`` so a
    3500-cell grid renders in a fraction of a second.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    by_page = {g.page: g for g in page_grids}
    faded = (0.78, 0.78, 0.78)
    green_fill = (0.78, 0.96, 0.78)
    green_stroke = (0.0, 0.55, 0.0)
    blue = (0.0, 0.30, 0.85)
    try:
        for page_no in range(1, doc.page_count + 1):
            grid = by_page.get(page_no)
            if grid is None:
                continue
            page = doc[page_no - 1]
            page_h = page.rect.height
            cs = grid.cell_size_pts
            top = grid.top_margin_pts
            left = grid.left_margin_pts

            if show_full_grid:
                shape = page.new_shape()
                for r in range(grid.rows + 1):
                    y = top + r * cs
                    shape.draw_line(
                        fitz.Point(left, y),
                        fitz.Point(left + grid.cols * cs, y),
                    )
                for c in range(grid.cols + 1):
                    x = left + c * cs
                    shape.draw_line(
                        fitz.Point(x, top),
                        fitz.Point(x, top + grid.rows * cs),
                    )
                shape.finish(color=faded, width=0.25)
                shape.commit()

            if show_writable:
                shape = page.new_shape()
                for cell in grid.cells:
                    if not cell.writable:
                        continue
                    shape.draw_rect(
                        fitz.Rect(cell.pdf_x1, cell.pdf_y1, cell.pdf_x2, cell.pdf_y2)
                    )
                shape.finish(
                    color=green_stroke,
                    fill=green_fill,
                    width=0.3,
                    fill_opacity=0.4,
                    stroke_opacity=0.55,
                )
                shape.commit()

            if show_regions and grid.regions:
                shape = page.new_shape()
                for rg in grid.regions:
                    shape.draw_rect(
                        fitz.Rect(rg.pdf_x1, rg.pdf_y1, rg.pdf_x2, rg.pdf_y2)
                    )
                shape.finish(color=blue, width=1.3)
                shape.commit()
                for rg in grid.regions:
                    page.insert_text(
                        (rg.pdf_x1 + 1.0, max(axis_label_font_size, rg.pdf_y1 - 1.2)),
                        f"{rg.region_id} {rg.bbox_range_id}",
                        fontsize=axis_label_font_size + 1.0,
                        color=blue,
                    )

            if show_axis_labels:
                fs = axis_label_font_size
                for c in range(1, grid.cols + 1):
                    x = left + (c - 1) * cs + cs / 2 - fs * 0.55
                    y = max(fs, top - 1.0)
                    page.insert_text(
                        (x, y),
                        _column_letters(c),
                        fontsize=fs,
                        color=(0.35, 0.35, 0.35),
                    )
                for r in range(1, grid.rows + 1):
                    x = max(0.5, left - fs * 1.6)
                    y = top + (r - 1) * cs + cs / 2 + fs * 0.35
                    if y > page_h - 0.5:
                        continue
                    page.insert_text(
                        (x, y),
                        str(r),
                        fontsize=fs,
                        color=(0.35, 0.35, 0.35),
                    )

            if show_cell_labels:
                for cell in grid.cells:
                    col = green_stroke if cell.writable else (0.55, 0.55, 0.55)
                    page.insert_text(
                        (
                            cell.pdf_x1 + 0.5,
                            min(page_h - 1.0, cell.pdf_y1 + cell_label_font_size + 0.3),
                        ),
                        cell.cell_id,
                        fontsize=cell_label_font_size,
                        color=col,
                    )

        out = Path(output_path)
        if out.exists():
            out.unlink()
        doc.save(out)
    finally:
        doc.close()
    return out


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_summary(grids: list[PageCellGrid]) -> None:
    for g in grids:
        writable = sum(1 for c in g.cells if c.writable)
        total = len(g.cells)
        print(
            f"page={g.page} rows={g.rows} cols={g.cols} "
            f"cells={total} writable={writable} "
            f"runs={len(g.runs)} regions={len(g.regions)} "
            f"left={g.left_margin_pts} top={g.top_margin_pts}",
            file=sys.stderr,
        )


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Cell-grid v4 analyser & overlay renderer.")
    p.add_argument("pdf", help="Input PDF path")
    p.add_argument("output", nargs="?", help="Output overlay PDF (default: <pdf>_grid_v4.pdf)")
    p.add_argument("--cell-size-pts", type=float, default=CELL_SIZE_PTS)
    p.add_argument("--dpi", type=int, default=RASTER_DPI)
    p.add_argument("--min-score", type=float, default=WRITABLE_MIN_SCORE)
    p.add_argument("--max-ink-ratio", type=float, default=MAX_INK_PIXEL_RATIO)
    p.add_argument("--ink-brightness-max", type=float, default=INK_BRIGHTNESS_MAX)
    p.add_argument("--min-run-length", type=int, default=MIN_RUN_LENGTH)
    p.add_argument("--min-region-cells", type=int, default=MIN_REGION_CELLS)
    p.add_argument("--no-full-grid", action="store_true", help="Hide faded full-grid layer")
    p.add_argument("--no-regions", action="store_true", help="Hide region rectangles")
    p.add_argument("--no-axis-labels", action="store_true", help="Hide column/row axis labels")
    p.add_argument("--cell-labels", action="store_true", help="Label every cell (slow on dense grids)")
    p.add_argument("--json", action="store_true", help="Print full JSON to stdout")
    p.add_argument("--include-cells", action="store_true", help="Include per-cell array in JSON output")
    args = p.parse_args()

    pdf_path = Path(args.pdf)
    pdf_bytes = pdf_path.read_bytes()

    grids = analyze_pdf_cell_grid_v4(
        pdf_bytes,
        cell_size_pts=args.cell_size_pts,
        dpi=args.dpi,
        min_score=args.min_score,
        max_ink_ratio=args.max_ink_ratio,
        ink_brightness_max=args.ink_brightness_max,
        min_run_length=args.min_run_length,
        min_region_cells=args.min_region_cells,
    )

    _print_summary(grids)

    out_path = Path(args.output) if args.output else pdf_path.with_name(f"{pdf_path.stem}_grid_v4.pdf")
    draw_cell_grid_overlay_v4(
        pdf_bytes,
        grids,
        out_path,
        show_full_grid=not args.no_full_grid,
        show_regions=not args.no_regions,
        show_axis_labels=not args.no_axis_labels,
        show_cell_labels=args.cell_labels,
    )
    print(f"wrote overlay → {out_path}", file=sys.stderr)

    if args.json:
        payload = [g.to_dict(include_cells=args.include_cells) for g in grids]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
