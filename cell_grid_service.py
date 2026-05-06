from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
import scipy.ndimage as ndi

CELL_SIZE_PTS: float = 20.0

# When you run:  python3 cell_grid_service.py
# edit these filenames if you want a different input/output in the project folder.
INPUT_PDF_RELATIVE_PATH = "test_pdf3.pdf"
OUTPUT_PDF_RELATIVE_PATH = "cell_grid_v1_pdf3.pdf"

WRITABLE_MIN_SCORE: float = 0.85
INK_PIXEL_BRIGHTNESS_MAX: float = 0.45
MAX_INK_PIXEL_RATIO: float = 0.05
MAX_INK_BLOB_FRACTION: float = 0.012
RASTER_DPI: int = 300

_OPEN_STRUCT_2X2 = np.ones((2, 2), dtype=bool)


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
class PageCellGrid:
    page: int
    rows: int
    cols: int
    cell_size_pts: float
    left_margin_pts: float
    top_margin_pts: float
    cells: list[Cell]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "rows": self.rows,
            "cols": self.cols,
            "cell_size_pts": self.cell_size_pts,
            "left_margin_pts": self.left_margin_pts,
            "top_margin_pts": self.top_margin_pts,
            "cells": [c.to_dict() for c in self.cells],
        }


def cell_id_from_rc(row: int, col: int) -> str:
    if row < 1 or col < 1:
        raise ValueError("row and col must be >= 1")
    letters = []
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


def analyze_pdf_cell_grid(
    pdf_bytes: bytes,
    *,
    cell_size_pts: float = CELL_SIZE_PTS,
    min_score: float = WRITABLE_MIN_SCORE,
    max_ink_ratio: float = MAX_INK_PIXEL_RATIO,
    max_blob_fraction: float = MAX_INK_BLOB_FRACTION,
    ink_brightness_max: float = INK_PIXEL_BRIGHTNESS_MAX,
    dpi: int = RASTER_DPI,
) -> list[PageCellGrid]:
    if cell_size_pts <= 0:
        raise ValueError("cell_size_pts must be > 0")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    gray_pages = _rasterize_pdf_to_gray_arrays(pdf_bytes, dpi=dpi)
    scale = dpi / 72.0

    results: list[PageCellGrid] = []
    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        cols = max(1, math.floor(page_w / cell_size_pts))
        rows = max(1, math.floor(page_h / cell_size_pts))
        left_margin = round(page_w - cols * cell_size_pts, 2)
        top_margin = round(page_h - rows * cell_size_pts, 2)
        gray = gray_pages[page_idx]
        cells: list[Cell] = []

        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                top_dist = top_margin + (r - 1) * cell_size_pts
                bottom_dist = top_margin + r * cell_size_pts
                x1 = left_margin + (c - 1) * cell_size_pts
                x2 = left_margin + c * cell_size_pts
                x0_px = max(0, int(x1 * scale))
                x1_px = min(gray.shape[1], max(x0_px + 1, int(x2 * scale)))
                y0_px = max(0, int(top_dist * scale))
                y1_px = min(gray.shape[0], max(y0_px + 1, int(bottom_dist * scale)))
                score, ink_ratio, blob_frac = _cell_metrics(
                    gray[y0_px:y1_px, x0_px:x1_px],
                    ink_brightness_max=ink_brightness_max,
                )
                score = round(score, 4)
                ink_ratio = round(ink_ratio, 4)
                blob_frac = round(blob_frac, 4)

                cells.append(
                    Cell(
                        cell_id=cell_id_from_rc(r, c),
                        row=r,
                        col=c,
                        page=page_no,
                        x_start_percent=round((x1 / page_w) * 100.0, 2),
                        x_end_percent=round((x2 / page_w) * 100.0, 2),
                        y_start_percent=round((top_dist / page_h) * 100.0, 2),
                        y_end_percent=round((bottom_dist / page_h) * 100.0, 2),
                        pdf_x1=round(x1, 2),
                        pdf_y1=round(top_dist, 2),
                        pdf_x2=round(x2, 2),
                        pdf_y2=round(bottom_dist, 2),
                        score=score,
                        ink_ratio=ink_ratio,
                        writable=(
                            (score >= min_score)
                            and (ink_ratio <= max_ink_ratio)
                            and (blob_frac <= max_blob_fraction)
                        ),
                    )
                )

        results.append(
            PageCellGrid(
                page=page_no,
                rows=rows,
                cols=cols,
                cell_size_pts=cell_size_pts,
                left_margin_pts=left_margin,
                top_margin_pts=top_margin,
                cells=cells,
            )
        )
    doc.close()
    return results


def find_writable_runs(page_grid: PageCellGrid, *, min_length: int = 1) -> list[list[Cell]]:
    rows: dict[int, list[Cell]] = {}
    for cell in page_grid.cells:
        if cell.writable:
            rows.setdefault(cell.row, []).append(cell)

    runs: list[list[Cell]] = []
    for row_cells in rows.values():
        sorted_cells = sorted(row_cells, key=lambda c: c.col)
        run: list[Cell] = []
        prev_col = None
        for cell in sorted_cells:
            if prev_col is None or cell.col == prev_col + 1:
                run.append(cell)
            else:
                if len(run) >= min_length:
                    runs.append(run)
                run = [cell]
            prev_col = cell.col
        if len(run) >= min_length:
            runs.append(run)
    runs.sort(key=len, reverse=True)
    return runs


def draw_cell_grid_overlay(
    pdf_bytes: bytes,
    page_grids: list[PageCellGrid],
    output_path: str | Path,
    *,
    draw_non_writable: bool = True,
    show_cell_ids: bool = True,
) -> Path:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    grid_by_page = {g.page: g for g in page_grids}
    for page_no in range(1, doc.page_count + 1):
        grid = grid_by_page.get(page_no)
        if grid is None:
            continue
        page = doc[page_no - 1]
        for cell in grid.cells:
            if not draw_non_writable and not cell.writable:
                continue
            color = (0.0, 0.7, 0.0) if cell.writable else (1.0, 0.0, 0.0)
            rect = fitz.Rect(cell.pdf_x1, cell.pdf_y1, cell.pdf_x2, cell.pdf_y2)
            page.draw_rect(rect, color=color, width=0.8)
            if show_cell_ids:
                page.insert_text(
                    (cell.pdf_x1 + 1.5, min(page.rect.height - 2, cell.pdf_y1 + 7)),
                    cell.cell_id,
                    fontsize=5,
                    color=color,
                )
    out = Path(output_path)
    if out.exists():
        out.unlink()
    doc.save(out)
    doc.close()
    return out


def _score_cell(cell: np.ndarray) -> float:
    """Brightness-only score in [0, 1]; empty cells read high."""
    if cell.size == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.mean(cell))))


def _cell_metrics(
    cell: np.ndarray, *, ink_brightness_max: float
) -> tuple[float, float, float]:
    """Return (score, cleaned ink_ratio, max_blob_fraction)."""
    if cell.size == 0:
        return 0.0, 1.0, 1.0
    h, w = int(cell.shape[0]), int(cell.shape[1])
    inner = cell[1:-1, 1:-1] if h > 2 and w > 2 else cell
    score = _score_cell(inner)
    if inner.size == 0:
        return score, 1.0, 1.0
    binary = inner < ink_brightness_max
    cleaned = ndi.binary_opening(binary, structure=_OPEN_STRUCT_2X2)
    ink_ratio = float(np.mean(cleaned))
    labeled, num_features = ndi.label(cleaned)
    if num_features == 0:
        max_blob_frac = 0.0
    else:
        counts = np.bincount(labeled.ravel())
        max_blob_frac = float(np.max(counts[1:])) / float(cleaned.size)
    return score, ink_ratio, max_blob_frac


def _rasterize_pdf_to_gray_arrays(pdf_bytes: bytes, *, dpi: int = RASTER_DPI) -> list[np.ndarray]:
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


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent
    _inp = _root / INPUT_PDF_RELATIVE_PATH
    _out = _root / OUTPUT_PDF_RELATIVE_PATH
    if not _inp.is_file():
        raise SystemExit(f"Input PDF not found: {_inp}")
    _bytes = _inp.read_bytes()
    # Slightly lower DPI here only so local preview finishes quickly (library default RASTER_DPI unchanged).
    _grids = analyze_pdf_cell_grid(_bytes, dpi=150)
    draw_cell_grid_overlay(_bytes, _grids, _out)
    print(_out.resolve())
