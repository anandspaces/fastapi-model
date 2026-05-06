from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np

CELL_SIZE_PTS: float = 25.0
WRITABLE_MIN_SCORE: float = 0.85
INK_PIXEL_BRIGHTNESS_MAX: float = 0.45
MAX_INK_PIXEL_RATIO: float = 0.03
RASTER_DPI: int = 300
MIN_RUN_LENGTH: int = 2


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
class PageWritableRuns:
    page: int
    rows: int
    cols: int
    cell_size_pts: float
    left_margin_pts: float
    top_margin_pts: float
    runs: list[WritableRun]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "rows": self.rows,
            "cols": self.cols,
            "cell_size_pts": self.cell_size_pts,
            "left_margin_pts": self.left_margin_pts,
            "top_margin_pts": self.top_margin_pts,
            "runs": [r.to_dict() for r in self.runs],
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


def range_id_from_cells(start_cell_id: str, end_cell_id: str) -> str:
    return start_cell_id if start_cell_id == end_cell_id else f"{start_cell_id}:{end_cell_id}"


def analyze_pdf_writable_runs(
    pdf_bytes: bytes,
    *,
    cell_size_pts: float = CELL_SIZE_PTS,
    min_score: float = WRITABLE_MIN_SCORE,
    max_ink_ratio: float = MAX_INK_PIXEL_RATIO,
    ink_brightness_max: float = INK_PIXEL_BRIGHTNESS_MAX,
    dpi: int = RASTER_DPI,
    min_run_length: int = MIN_RUN_LENGTH,
) -> list[PageWritableRuns]:
    if cell_size_pts <= 0:
        raise ValueError("cell_size_pts must be > 0")
    if min_run_length < 1:
        raise ValueError("min_run_length must be >= 1")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    gray_pages = _rasterize_pdf_to_gray_arrays(pdf_bytes, dpi=dpi)
    scale = dpi / 72.0
    all_pages: list[PageWritableRuns] = []

    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        cols = max(1, math.floor(page_w / cell_size_pts))
        rows = max(1, math.floor(page_h / cell_size_pts))
        left_margin = round(page_w - cols * cell_size_pts, 2)
        top_margin = round(page_h - rows * cell_size_pts, 2)
        gray = gray_pages[page_idx]
        run_rows: list[WritableRun] = []

        for r in range(1, rows + 1):
            writable_cols: list[int] = []
            scores_by_col: dict[int, float] = {}
            ink_by_col: dict[int, float] = {}

            for c in range(1, cols + 1):
                top_dist = top_margin + (r - 1) * cell_size_pts
                bottom_dist = top_margin + r * cell_size_pts
                x1 = left_margin + (c - 1) * cell_size_pts
                x2 = left_margin + c * cell_size_pts

                x0_px = max(0, int(x1 * scale))
                x1_px = min(gray.shape[1], max(x0_px + 1, int(x2 * scale)))
                y0_px = max(0, int(top_dist * scale))
                y1_px = min(gray.shape[0], max(y0_px + 1, int(bottom_dist * scale)))

                score, ink_ratio = _cell_metrics(
                    gray[y0_px:y1_px, x0_px:x1_px],
                    ink_brightness_max=ink_brightness_max,
                )
                if score >= min_score and ink_ratio <= max_ink_ratio:
                    writable_cols.append(c)
                    scores_by_col[c] = score
                    ink_by_col[c] = ink_ratio

            row_runs = _merge_row_runs(
                page_no=page_no,
                row=r,
                writable_cols=writable_cols,
                scores_by_col=scores_by_col,
                ink_by_col=ink_by_col,
                page_w=page_w,
                page_h=page_h,
                cell_size_pts=cell_size_pts,
                left_margin=left_margin,
                top_margin=top_margin,
            )
            run_rows.extend([run for run in row_runs if run.cell_count >= min_run_length])

        all_pages.append(
            PageWritableRuns(
                page=page_no,
                rows=rows,
                cols=cols,
                cell_size_pts=cell_size_pts,
                left_margin_pts=left_margin,
                top_margin_pts=top_margin,
                runs=run_rows,
            )
        )

    doc.close()
    return all_pages


def draw_writable_runs_overlay(
    pdf_bytes: bytes,
    page_runs: list[PageWritableRuns],
    output_path: str | Path,
    *,
    show_labels: bool = True,
) -> Path:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    runs_by_page = {p.page: p.runs for p in page_runs}
    for page_no in range(1, doc.page_count + 1):
        page = doc[page_no - 1]
        for run in runs_by_page.get(page_no, []):
            rect = fitz.Rect(run.pdf_x1, run.pdf_y1, run.pdf_x2, run.pdf_y2)
            color = (0.0, 0.7, 0.0)
            page.draw_rect(rect, color=color, width=1.1)
            if show_labels:
                page.insert_text(
                    (run.pdf_x1 + 1.5, min(page.rect.height - 2, run.pdf_y1 + 8)),
                    run.range_id,
                    fontsize=6,
                    color=color,
                )
    out = Path(output_path)
    if out.exists():
        out.unlink()
    doc.save(out)
    doc.close()
    return out


def _merge_row_runs(
    *,
    page_no: int,
    row: int,
    writable_cols: list[int],
    scores_by_col: dict[int, float],
    ink_by_col: dict[int, float],
    page_w: float,
    page_h: float,
    cell_size_pts: float,
    left_margin: float,
    top_margin: float,
) -> list[WritableRun]:
    if not writable_cols:
        return []
    cols = sorted(writable_cols)
    groups: list[list[int]] = []
    current = [cols[0]]
    for c in cols[1:]:
        if c == current[-1] + 1:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
    groups.append(current)

    runs: list[WritableRun] = []
    for g in groups:
        start_col, end_col = g[0], g[-1]
        start_cell = cell_id_from_rc(row, start_col)
        end_cell = cell_id_from_rc(row, end_col)
        x1 = left_margin + (start_col - 1) * cell_size_pts
        x2 = left_margin + end_col * cell_size_pts
        top_dist = top_margin + (row - 1) * cell_size_pts
        bottom_dist = top_margin + row * cell_size_pts
        score_vals = [scores_by_col[c] for c in g]
        ink_vals = [ink_by_col[c] for c in g]

        runs.append(
            WritableRun(
                page=page_no,
                row=row,
                start_col=start_col,
                end_col=end_col,
                start_cell_id=start_cell,
                end_cell_id=end_cell,
                range_id=range_id_from_cells(start_cell, end_cell),
                cell_count=len(g),
                x_start_percent=round((x1 / page_w) * 100.0, 2),
                x_end_percent=round((x2 / page_w) * 100.0, 2),
                y_start_percent=round((top_dist / page_h) * 100.0, 2),
                y_end_percent=round((bottom_dist / page_h) * 100.0, 2),
                pdf_x1=round(x1, 2),
                pdf_y1=round(top_dist, 2),
                pdf_x2=round(x2, 2),
                pdf_y2=round(bottom_dist, 2),
                mean_score=round(float(np.mean(score_vals)), 4),
                mean_ink_ratio=round(float(np.mean(ink_vals)), 4),
            )
        )
    return runs


def _score_cell(cell: np.ndarray) -> float:
    if cell.size == 0:
        return 0.0
    mean = float(np.mean(cell))
    std = float(np.std(cell))
    std_factor = min(1.0, std * 3.0)
    return max(0.0, min(1.0, mean * (1.0 - std_factor)))


def _cell_metrics(cell: np.ndarray, *, ink_brightness_max: float) -> tuple[float, float]:
    score = _score_cell(cell)
    if cell.size == 0:
        return score, 1.0
    ink_ratio = float(np.mean(cell < ink_brightness_max))
    return score, ink_ratio


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
