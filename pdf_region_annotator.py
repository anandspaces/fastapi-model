"""
pdf_region_annotator.py
-----------------------
Pass a PDF → get back a PDF with writable regions highlighted and
their PDF coordinates printed on each region.

Usage:
    python pdf_region_annotator.py input.pdf output.pdf

    # or from code:
    from pdf_region_annotator import annotate_pdf
    annotate_pdf("input.pdf", "output.pdf")

Dependencies:
    pip install pymupdf scipy numpy
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz
from scipy.ndimage import label

# ─── tuneable constants ───────────────────────────────────────────────────────

CELL_SIZE_PTS: float = 25.0
WRITABLE_MIN_SCORE: float = 0.85
INK_PIXEL_BRIGHTNESS_MAX: float = 0.45
MAX_INK_PIXEL_RATIO: float = 0.03
RASTER_DPI: int = 150          # 150 is fast enough for detection; raise to 300 for accuracy
MIN_REGION_CELLS: int = 3      # ignore blobs smaller than this
CONNECTIVITY: int = 8          # 4 = no diagonals, 8 = diagonals count


# ─── data models ─────────────────────────────────────────────────────────────

@dataclass
class Cell:
    cell_id: str
    row: int
    col: int
    page: int
    pdf_x1: float
    pdf_y1: float
    pdf_x2: float
    pdf_y2: float
    score: float
    ink_ratio: float
    writable: bool


@dataclass
class PageCellGrid:
    page: int
    rows: int
    cols: int
    cell_size_pts: float
    left_margin_pts: float
    top_margin_pts: float
    cells: list[Cell]


@dataclass
class WritableRegion:
    blob_id: int
    page: int
    cell_ids: list[str]
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    pdf_x1: float
    pdf_y1: float
    pdf_x2: float
    pdf_y2: float
    width_pts: float
    height_pts: float
    area_cells: int
    bbox_fill_ratio: float

    def __str__(self) -> str:
        return (
            f"Region #{self.blob_id} | page={self.page} | "
            f"cells={self.area_cells} | fill={self.bbox_fill_ratio:.2f} | "
            f"x1={self.pdf_x1:.1f} y1={self.pdf_y1:.1f} "
            f"x2={self.pdf_x2:.1f} y2={self.pdf_y2:.1f} | "
            f"w={self.width_pts:.1f}pt h={self.height_pts:.1f}pt"
        )


# ─── cell id helpers ──────────────────────────────────────────────────────────

def _cell_id(row: int, col: int) -> str:
    letters: list[str] = []
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        letters.append(chr(65 + rem))
    return f"{''.join(reversed(letters))}{row}"


# ─── rasterisation ────────────────────────────────────────────────────────────

def _rasterize_gray(pdf_bytes: bytes, *, dpi: int) -> list[np.ndarray]:
    """One float32 grayscale array per page, values in [0, 1]."""
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


# ─── cell scoring ─────────────────────────────────────────────────────────────

def _score_cell(patch: np.ndarray) -> float:
    """
    Brightness-based whitespace score in [0, 1].
    Uses mean - std instead of mean*(1-std) to avoid the flawed
    penalty collapse described in the review.
    """
    if patch.size == 0:
        return 0.0
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    return float(np.clip(mean - std, 0.0, 1.0))


def _cell_metrics(patch: np.ndarray, *, ink_brightness_max: float) -> tuple[float, float]:
    score = _score_cell(patch)
    if patch.size == 0:
        return 0.0, 0.0          # empty slice → not writable (not penalised as "full ink")
    ink_ratio = float(np.mean(patch < ink_brightness_max))
    return score, ink_ratio


# ─── grid analysis ────────────────────────────────────────────────────────────

def analyze_pdf_cell_grid(
    pdf_bytes: bytes,
    *,
    cell_size_pts: float = CELL_SIZE_PTS,
    min_score: float = WRITABLE_MIN_SCORE,
    max_ink_ratio: float = MAX_INK_PIXEL_RATIO,
    ink_brightness_max: float = INK_PIXEL_BRIGHTNESS_MAX,
    dpi: int = RASTER_DPI,
) -> list[PageCellGrid]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    gray_pages = _rasterize_gray(pdf_bytes, dpi=dpi)
    assert len(gray_pages) == doc.page_count, "Page count mismatch between renders"
    scale = dpi / 72.0

    results: list[PageCellGrid] = []
    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)

        cols = max(1, math.floor(page_w / cell_size_pts))
        rows = max(1, math.floor(page_h / cell_size_pts))

        # Centre the grid — split remainder evenly on both sides
        h_remainder = page_w - cols * cell_size_pts
        v_remainder = page_h - rows * cell_size_pts
        left_margin = round(h_remainder / 2.0, 2)
        top_margin = round(v_remainder / 2.0, 2)

        gray = gray_pages[page_idx]
        cells: list[Cell] = []

        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                x1 = left_margin + (c - 1) * cell_size_pts
                x2 = left_margin + c * cell_size_pts
                y1 = top_margin + (r - 1) * cell_size_pts
                y2 = top_margin + r * cell_size_pts

                x0_px = max(0, int(x1 * scale))
                x1_px = min(gray.shape[1], max(x0_px + 1, int(x2 * scale)))
                y0_px = max(0, int(y1 * scale))
                y1_px = min(gray.shape[0], max(y0_px + 1, int(y2 * scale)))

                score, ink_ratio = _cell_metrics(
                    gray[y0_px:y1_px, x0_px:x1_px],
                    ink_brightness_max=ink_brightness_max,
                )

                cells.append(Cell(
                    cell_id=_cell_id(r, c),
                    row=r, col=c, page=page_no,
                    pdf_x1=round(x1, 2), pdf_y1=round(y1, 2),
                    pdf_x2=round(x2, 2), pdf_y2=round(y2, 2),
                    score=round(score, 4),
                    ink_ratio=round(ink_ratio, 4),
                    writable=(score >= min_score) and (ink_ratio <= max_ink_ratio),
                ))

        results.append(PageCellGrid(
            page=page_no, rows=rows, cols=cols,
            cell_size_pts=cell_size_pts,
            left_margin_pts=left_margin,
            top_margin_pts=top_margin,
            cells=cells,
        ))

    doc.close()
    return results


# ─── connected-component region detection ────────────────────────────────────

def find_writable_regions(
    page_grid: PageCellGrid,
    *,
    connectivity: int = CONNECTIVITY,
    min_cells: int = MIN_REGION_CELLS,
) -> list[WritableRegion]:
    grid = np.zeros((page_grid.rows, page_grid.cols), dtype=np.uint8)
    for cell in page_grid.cells:
        if cell.writable:
            grid[cell.row - 1][cell.col - 1] = 1

    structure = (
        np.ones((3, 3), dtype=int)
        if connectivity == 8
        else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    )
    labeled, num_blobs = label(grid, structure=structure)

    cell_lookup = {(c.row, c.col): c for c in page_grid.cells}
    regions: list[WritableRegion] = []

    for blob_id in range(1, num_blobs + 1):
        positions = np.argwhere(labeled == blob_id)   # (N, 2) — 0-indexed
        if len(positions) < min_cells:
            continue

        rows_1 = positions[:, 0] + 1
        cols_1 = positions[:, 1] + 1
        blob_cells = [cell_lookup[(r, c)] for r, c in zip(rows_1.tolist(), cols_1.tolist())]

        row_min, row_max = int(rows_1.min()), int(rows_1.max())
        col_min, col_max = int(cols_1.min()), int(cols_1.max())

        pdf_x1 = min(c.pdf_x1 for c in blob_cells)
        pdf_y1 = min(c.pdf_y1 for c in blob_cells)
        pdf_x2 = max(c.pdf_x2 for c in blob_cells)
        pdf_y2 = max(c.pdf_y2 for c in blob_cells)

        bbox_area = (row_max - row_min + 1) * (col_max - col_min + 1)
        fill_ratio = round(len(blob_cells) / bbox_area, 3)

        regions.append(WritableRegion(
            blob_id=blob_id,
            page=page_grid.page,
            cell_ids=sorted([c.cell_id for c in blob_cells]),
            row_min=row_min, row_max=row_max,
            col_min=col_min, col_max=col_max,
            pdf_x1=pdf_x1, pdf_y1=pdf_y1,
            pdf_x2=pdf_x2, pdf_y2=pdf_y2,
            width_pts=round(pdf_x2 - pdf_x1, 2),
            height_pts=round(pdf_y2 - pdf_y1, 2),
            area_cells=len(blob_cells),
            bbox_fill_ratio=fill_ratio,
        ))

    regions.sort(key=lambda r: r.area_cells, reverse=True)
    return regions


# ─── console debug dump ───────────────────────────────────────────────────────

def print_region_report(regions: list[WritableRegion], page_no: int) -> None:
    print(f"\n{'='*65}")
    print(f"  PAGE {page_no}  —  {len(regions)} writable region(s) found")
    print(f"{'='*65}")
    for i, r in enumerate(regions, 1):
        solid = "✓ solid" if r.bbox_fill_ratio >= 0.8 else "⚠ irregular"
        print(f"\n  [{i}] blob_id={r.blob_id}  cells={r.area_cells}  fill={r.bbox_fill_ratio:.2f}  {solid}")
        print(f"      cell ids  : {', '.join(r.cell_ids)}")
        print(f"      grid span : rows {r.row_min}–{r.row_max}  cols {r.col_min}–{r.col_max}")
        print(f"      pdf coords: x1={r.pdf_x1:.2f}  y1={r.pdf_y1:.2f}  x2={r.pdf_x2:.2f}  y2={r.pdf_y2:.2f}")
        print(f"      size      : {r.width_pts:.2f} pt wide  x  {r.height_pts:.2f} pt tall")


# ─── PDF overlay writer ───────────────────────────────────────────────────────

# Colours (R, G, B) in [0, 1]
_GREEN  = (0.05, 0.65, 0.25)
_BLUE   = (0.10, 0.35, 0.90)
_ORANGE = (0.95, 0.50, 0.05)
_RED    = (0.85, 0.10, 0.10)
_WHITE  = (1.00, 1.00, 1.00)


def _region_color(region: WritableRegion) -> tuple[float, float, float]:
    """Color-code by fill ratio so you can spot quality at a glance."""
    if region.bbox_fill_ratio >= 0.9:
        return _GREEN
    if region.bbox_fill_ratio >= 0.7:
        return _BLUE
    if region.bbox_fill_ratio >= 0.5:
        return _ORANGE
    return _RED


def _draw_region_overlay(
    page: fitz.Page,
    region: WritableRegion,
    rank: int,
) -> None:
    color = _region_color(region)
    rect = fitz.Rect(region.pdf_x1, region.pdf_y1, region.pdf_x2, region.pdf_y2)

    # Semi-transparent fill
    page.draw_rect(rect, color=color, fill=color, fill_opacity=0.08, width=1.2)

    # Coordinate label — placed just inside the top-left corner
    coord_text = (
        f"#{rank}  ({region.area_cells} cells, fill={region.bbox_fill_ratio:.2f})\n"
        f"x1={region.pdf_x1:.1f}  y1={region.pdf_y1:.1f}\n"
        f"x2={region.pdf_x2:.1f}  y2={region.pdf_y2:.1f}\n"
        f"w={region.width_pts:.1f}pt  h={region.height_pts:.1f}pt"
    )

    # Small white backing rect so text is readable over any background
    label_h = 36
    label_w = min(region.width_pts - 2, 130)
    if label_w > 20 and region.height_pts > label_h:
        backing = fitz.Rect(
            region.pdf_x1 + 1,
            region.pdf_y1 + 1,
            region.pdf_x1 + 1 + label_w,
            region.pdf_y1 + 1 + label_h,
        )
        page.draw_rect(backing, color=_WHITE, fill=_WHITE, fill_opacity=0.75, width=0)

    text_rect = fitz.Rect(
        region.pdf_x1 + 2,
        region.pdf_y1 + 2,
        region.pdf_x1 + 2 + label_w,
        region.pdf_y1 + 2 + label_h + 10,
    )
    page.insert_textbox(
        text_rect,
        coord_text,
        fontsize=6.5,
        color=color,
        align=0,
    )

    # Corner tick marks for precise coordinate reference
    tick = 5.0
    for (ax, ay), (bx, by) in [
        # top-left
        ((rect.x0, rect.y0 + tick), (rect.x0, rect.y0)),
        ((rect.x0, rect.y0), (rect.x0 + tick, rect.y0)),
        # top-right
        ((rect.x1 - tick, rect.y0), (rect.x1, rect.y0)),
        ((rect.x1, rect.y0), (rect.x1, rect.y0 + tick)),
        # bottom-left
        ((rect.x0, rect.y1 - tick), (rect.x0, rect.y1)),
        ((rect.x0, rect.y1), (rect.x0 + tick, rect.y1)),
        # bottom-right
        ((rect.x1 - tick, rect.y1), (rect.x1, rect.y1)),
        ((rect.x1, rect.y1 - tick), (rect.x1, rect.y1)),
    ]:
        page.draw_line((ax, ay), (bx, by), color=color, width=1.5)


def _draw_cell_grid(page: fitz.Page, page_grid: PageCellGrid) -> None:
    """Light grey grid so you can see the cell boundaries."""
    for cell in page_grid.cells:
        rect = fitz.Rect(cell.pdf_x1, cell.pdf_y1, cell.pdf_x2, cell.pdf_y2)
        page.draw_rect(rect, color=(0.8, 0.8, 0.8), width=0.3)


# ─── legend ───────────────────────────────────────────────────────────────────

def _draw_legend(page: fitz.Page) -> None:
    pw = page.rect.width
    items = [
        (_GREEN,  "fill ≥ 0.90  solid rectangle"),
        (_BLUE,   "fill ≥ 0.70  good"),
        (_ORANGE, "fill ≥ 0.50  irregular"),
        (_RED,    "fill < 0.50  scattered"),
    ]
    x, y = pw - 170, 10
    page.draw_rect(fitz.Rect(x - 4, y - 4, pw - 4, y + len(items) * 13 + 4),
                   color=(0.9, 0.9, 0.9), fill=(0.98, 0.98, 0.98), width=0.5)
    for color, label_text in items:
        page.draw_rect(fitz.Rect(x, y + 1, x + 10, y + 9), color=color, fill=color, width=0)
        page.insert_text((x + 13, y + 9), label_text, fontsize=6.5, color=(0.2, 0.2, 0.2))
        y += 13


# ─── public API ───────────────────────────────────────────────────────────────

def annotate_pdf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cell_size_pts: float = CELL_SIZE_PTS,
    min_score: float = WRITABLE_MIN_SCORE,
    max_ink_ratio: float = MAX_INK_PIXEL_RATIO,
    dpi: int = RASTER_DPI,
    connectivity: int = CONNECTIVITY,
    min_cells: int = MIN_REGION_CELLS,
    draw_cell_grid: bool = True,
    verbose: bool = True,
) -> list[WritableRegion]:
    """
    Annotate *input_path* with writable-region overlays and save to *output_path*.
    Returns the full list of WritableRegion objects found across all pages.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    pdf_bytes = input_path.read_bytes()

    # ── analyse ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"[1/3] Analysing cell grid  (cell={cell_size_pts}pt, dpi={dpi}) …")
    page_grids = analyze_pdf_cell_grid(
        pdf_bytes,
        cell_size_pts=cell_size_pts,
        min_score=min_score,
        max_ink_ratio=max_ink_ratio,
        dpi=dpi,
    )

    if verbose:
        print(f"[2/3] Detecting connected writable regions …")
    all_regions: list[WritableRegion] = []
    regions_by_page: dict[int, list[WritableRegion]] = {}
    for pg in page_grids:
        regs = find_writable_regions(pg, connectivity=connectivity, min_cells=min_cells)
        regions_by_page[pg.page] = regs
        all_regions.extend(regs)
        if verbose:
            print_region_report(regs, pg.page)

    # ── draw overlay ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[3/3] Writing annotated PDF → {output_path} …")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for pg in page_grids:
        page = doc[pg.page - 1]
        if draw_cell_grid:
            _draw_cell_grid(page, pg)
        _draw_legend(page)
        for rank, region in enumerate(regions_by_page.get(pg.page, []), 1):
            _draw_region_overlay(page, region, rank)

    # Atomic-ish save: write to temp then rename
    tmp = output_path.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(output_path)

    if verbose:
        print(f"\nDone. {len(all_regions)} regions across {len(page_grids)} page(s).")
        print(f"Output: {output_path.resolve()}")

    return all_regions


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a PDF with writable-region bounding boxes and coordinates."
    )
    parser.add_argument("input",  help="Input PDF path")
    parser.add_argument("output", help="Output PDF path")
    parser.add_argument("--cell-size",    type=float, default=CELL_SIZE_PTS,       metavar="PT",
                        help=f"Cell size in points (default {CELL_SIZE_PTS})")
    parser.add_argument("--dpi",          type=int,   default=RASTER_DPI,          metavar="DPI",
                        help=f"Rasterisation DPI (default {RASTER_DPI})")
    parser.add_argument("--min-score",    type=float, default=WRITABLE_MIN_SCORE,  metavar="F",
                        help=f"Min whitespace score to mark writable (default {WRITABLE_MIN_SCORE})")
    parser.add_argument("--max-ink",      type=float, default=MAX_INK_PIXEL_RATIO, metavar="F",
                        help=f"Max ink pixel ratio (default {MAX_INK_PIXEL_RATIO})")
    parser.add_argument("--min-cells",    type=int,   default=MIN_REGION_CELLS,    metavar="N",
                        help=f"Min cells per region (default {MIN_REGION_CELLS})")
    parser.add_argument("--connectivity", type=int,   default=CONNECTIVITY,        choices=[4, 8],
                        help=f"Cell connectivity (default {CONNECTIVITY})")
    parser.add_argument("--no-grid",      action="store_true",
                        help="Skip drawing the cell grid overlay")
    parser.add_argument("--quiet",        action="store_true",
                        help="Suppress console output")

    args = parser.parse_args()
    annotate_pdf(
        args.input,
        args.output,
        cell_size_pts=args.cell_size,
        dpi=args.dpi,
        min_score=args.min_score,
        max_ink_ratio=args.max_ink,
        min_cells=args.min_cells,
        connectivity=args.connectivity,
        draw_cell_grid=not args.no_grid,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()