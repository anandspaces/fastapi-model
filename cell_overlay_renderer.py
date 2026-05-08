"""cell_overlay_renderer.py — fast writable-cell overlay PNGs for the grading prompt.

For each PDF page, render an overlay raster with:
  * faint grid lines covering the whole grid
  * translucent green fill on writable cells (single shape commit)
  * cell ID label inside each writable cell (single TextWriter commit per page)
  * axis labels (column letters + row numbers) along the top and left edges

Why this matters: Gemini's grading call uses these PNGs as the spatial reference,
so cell-ID labels MUST be readable in the raster. The implementation is batched
(one shape, one TextWriter per layer per page) to stay inside the perf budget:

    BUDGET: ≤ 600 ms / page P95 at 200 DPI, ~2k writable cells/page

The legacy ``cell_grid_service_v4 --cell-labels`` path issues one
``page.insert_text`` per cell and clocks ~30 s/page on the same input. Don't
regress to that pattern.

Public API:
    render_overlay_pngs(pdf_bytes, grids, *, dpi=200, label_font_path=None) -> list[bytes]

CLI:
    python3 cell_overlay_renderer.py input.pdf out_dir/ [--dpi 200] [--font path]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import fitz

from cell_grid_service_v4 import PageCellGrid, analyze_pdf_cell_grid_v4

# Visual constants — tuned so labels stay readable at 200 DPI on dense grids
# while cell fills are translucent enough to leave handwriting visible.
GREEN_FILL = (0.78, 0.96, 0.78)
GREEN_STROKE = (0.05, 0.55, 0.05)
GRID_GRAY = (0.78, 0.78, 0.78)
LABEL_COLOR = (0.05, 0.45, 0.05)
# Used for non-writable cell labels under label_every_cell=True. Faint enough
# to read against white but not fight handwriting in dark cells.
NONWRITABLE_LABEL_COLOR = (0.40, 0.40, 0.40)
AXIS_COLOR = (0.30, 0.30, 0.30)


# Per-cell labels are only emitted when they would render at ≥ this many points.
# Below this, the label font drops under ~4pt and becomes illegible in the
# raster at 200 DPI; axis labels alone are then the spatial reference.
MIN_CELL_LABEL_SIZE_PTS: float = 4.0

# Absolute floor — even label_every_cell skips below this size because no DPI
# can rescue 2pt glyphs.
ABS_MIN_CELL_LABEL_SIZE_PTS: float = 3.0

# JPEG output defaults. q-85 is the sweet spot for grid overlays — labels stay
# crisp, payload drops 3-5× vs PNG at the same DPI.
DEFAULT_JPEG_QUALITY: int = 85


def _label_font_size_pts(cell_size_pts: float) -> float:
    return max(2.4, min(6.0, cell_size_pts * 0.30))


def _axis_font_size_pts(cell_size_pts: float) -> float:
    return max(4.0, min(7.5, cell_size_pts * 0.40))


def _should_label_each_cell(cell_size_pts: float) -> bool:
    return _label_font_size_pts(cell_size_pts) >= MIN_CELL_LABEL_SIZE_PTS


def _column_letters(col: int) -> str:
    letters: list[str] = []
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def render_overlay_pngs(
    pdf_bytes: bytes,
    grids: list[PageCellGrid],
    *,
    dpi: int = 200,
    label_font_path: str | None = None,
    show_axis_labels: bool = True,
    label_every_cell: bool = False,
    image_format: str = "png",
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[bytes]:
    """Render one overlay image per page; returns bytes list in document order.

    Pages with no matching grid (e.g., a page outside the analyzer's range)
    are emitted as the original raster — caller can still rely on len(result)
    matching doc.page_count.

    ``label_every_cell``: when True, non-writable cells also get their cell ID
    drawn in faint gray. Used by the cell-overlay grading prompt where Gemini
    must be able to read any cell ID (anchors land on handwriting cells).

    ``image_format``: ``"png"`` (default, lossless) or ``"jpeg"``. JPEG q-85
    cuts payload 3-5× with no visible label degradation at ≥ 150 DPI — use it
    for the grading-prompt path where total upload size matters.

    ``jpeg_quality``: 1-100, only used when ``image_format='jpeg'``. Defaults
    to ``DEFAULT_JPEG_QUALITY`` (85).
    """
    fmt = image_format.lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise ValueError(f"image_format must be 'png' or 'jpeg', got {image_format!r}")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    by_page = {g.page: g for g in grids}

    label_font = (
        fitz.Font(fontfile=label_font_path)
        if label_font_path and Path(label_font_path).exists()
        else fitz.Font("helv")
    )

    out: list[bytes] = []
    try:
        for idx in range(doc.page_count):
            grid = by_page.get(idx + 1)
            page = doc[idx]
            if grid is not None:
                _draw_overlay(page, grid, label_font, show_axis_labels,
                              label_every_cell=label_every_cell)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
            if fmt == "png":
                out.append(pix.tobytes("png"))
            else:
                out.append(pix.tobytes("jpeg", jpg_quality=int(jpeg_quality)))
    finally:
        doc.close()
    return out


def _draw_overlay(
    page: fitz.Page,
    grid: PageCellGrid,
    label_font: fitz.Font,
    show_axis_labels: bool,
    *,
    label_every_cell: bool = False,
) -> None:
    cs = grid.cell_size_pts
    left = grid.left_margin_pts
    top = grid.top_margin_pts
    rows, cols = grid.rows, grid.cols
    label_size = _label_font_size_pts(cs)

    # ── Layer 1: faded full grid ────────────────────────────────────────
    shape = page.new_shape()
    for r in range(rows + 1):
        y = top + r * cs
        shape.draw_line(fitz.Point(left, y), fitz.Point(left + cols * cs, y))
    for c in range(cols + 1):
        x = left + c * cs
        shape.draw_line(fitz.Point(x, top), fitz.Point(x, top + rows * cs))
    shape.finish(color=GRID_GRAY, width=0.2)
    shape.commit()

    # ── Layer 2: writable cell fills (one commit) ──────────────────────
    writable_cells = [c for c in grid.cells if c.writable]
    if writable_cells:
        shape = page.new_shape()
        for cell in writable_cells:
            shape.draw_rect(
                fitz.Rect(cell.pdf_x1, cell.pdf_y1, cell.pdf_x2, cell.pdf_y2)
            )
        shape.finish(
            color=GREEN_STROKE,
            fill=GREEN_FILL,
            width=0.25,
            fill_opacity=0.32,
            stroke_opacity=0.55,
        )
        shape.commit()

    # ── Layer 3a: writable cell ID labels (green) ──────────────────────
    # Skipped on dense grids where labels would be illegible (< 4pt font),
    # UNLESS label_every_cell forces them on (the cell-overlay grading prompt
    # needs every cell ID readable regardless of cell size). Even with that
    # override, the absolute floor (~3pt) cannot be rescued by any DPI.
    label_writable = (
        writable_cells
        and label_size >= ABS_MIN_CELL_LABEL_SIZE_PTS
        and (label_every_cell or _should_label_each_cell(cs))
    )
    if label_writable:
        tw = fitz.TextWriter(page.rect, color=LABEL_COLOR)
        for cell in writable_cells:
            tw.append(
                fitz.Point(cell.pdf_x1 + 1.0, cell.pdf_y1 + label_size + 0.5),
                cell.cell_id,
                font=label_font,
                fontsize=label_size,
            )
        tw.write_text(page)

    # ── Layer 3b: non-writable cell labels (faint gray) ────────────────
    # Off by default. Enabled by label_every_cell so Gemini can reference
    # any cell ID (e.g. for anchor.rows on handwriting cells). One additional
    # TextWriter commit per page. Skipped below the absolute illegibility
    # floor — sub-3pt glyphs can't be rescued.
    if label_every_cell and label_size >= ABS_MIN_CELL_LABEL_SIZE_PTS:
        non_writable = [c for c in grid.cells if not c.writable]
        if non_writable:
            tw_nw = fitz.TextWriter(page.rect, color=NONWRITABLE_LABEL_COLOR)
            for cell in non_writable:
                tw_nw.append(
                    fitz.Point(cell.pdf_x1 + 1.0, cell.pdf_y1 + label_size + 0.5),
                    cell.cell_id,
                    font=label_font,
                    fontsize=label_size,
                )
            tw_nw.write_text(page)

    # ── Layer 4: axis labels (cheap O(rows + cols)) ────────────────────
    if show_axis_labels:
        axis_size = _axis_font_size_pts(cs)
        tw_axis = fitz.TextWriter(page.rect, color=AXIS_COLOR)
        for c in range(1, cols + 1):
            x = left + (c - 1) * cs + cs / 2 - axis_size * 0.55
            y = max(axis_size, top - 1.5)
            tw_axis.append(
                fitz.Point(x, y),
                _column_letters(c),
                font=label_font,
                fontsize=axis_size,
            )
        for r in range(1, rows + 1):
            x = max(0.5, left - axis_size * 1.6)
            y = top + (r - 1) * cs + cs / 2 + axis_size * 0.35
            if y > page.rect.height - 0.5:
                continue
            tw_axis.append(
                fitz.Point(x, y),
                str(r),
                font=label_font,
                fontsize=axis_size,
            )
        tw_axis.write_text(page)


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf")
    p.add_argument("out_dir")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cell-size-pts", type=float, default=None,
                   help="Override cell size; defaults to cell_grid_service_v4 default")
    p.add_argument("--font", default=None,
                   help="TTF for cell labels (defaults to PyMuPDF helv)")
    p.add_argument("--no-axis", action="store_true")
    p.add_argument("--label-every-cell", action="store_true",
                   help="Label non-writable cells too (for cell-overlay grading prompt)")
    p.add_argument("--format", default="png", choices=("png", "jpeg"),
                   help="Image format. JPEG is 3-5× smaller; use for upload payloads.")
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY,
                   help="JPEG quality 1-100 (only used when --format=jpeg)")
    args = p.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_bytes = pdf_path.read_bytes()

    t0 = time.perf_counter()
    grids = (
        analyze_pdf_cell_grid_v4(pdf_bytes, cell_size_pts=args.cell_size_pts)
        if args.cell_size_pts is not None
        else analyze_pdf_cell_grid_v4(pdf_bytes)
    )
    t_grid = time.perf_counter() - t0

    t0 = time.perf_counter()
    images = render_overlay_pngs(
        pdf_bytes,
        grids,
        dpi=args.dpi,
        label_font_path=args.font,
        show_axis_labels=not args.no_axis,
        label_every_cell=args.label_every_cell,
        image_format=args.format,
        jpeg_quality=args.jpeg_quality,
    )
    t_render = time.perf_counter() - t0

    ext = "jpg" if args.format == "jpeg" else "png"
    for i, img in enumerate(images, start=1):
        (out_dir / f"page_{i:03d}.{ext}").write_bytes(img)

    n_writable = sum(sum(1 for c in g.cells if c.writable) for g in grids)
    n_pages = len(images)
    avg_ms = 1000.0 * t_render / max(1, n_pages)
    total_kb = sum(len(b) for b in images) // 1024
    print(
        f"grid={t_grid:.2f}s  render={t_render:.2f}s  "
        f"pages={n_pages}  writable_cells={n_writable}  avg={avg_ms:.1f} ms/page  "
        f"total={total_kb} KB",
        file=sys.stderr,
    )
    print(f"wrote {n_pages} {args.format.upper()}s → {out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
