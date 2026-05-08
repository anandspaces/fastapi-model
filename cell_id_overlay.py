"""cell_id_overlay.py — minimal cell-grid overlay (every cell labelled with its ID).

No writability scoring, no regions, no runs. For every PDF page it:
  * computes a 12-pt grid centered horizontally
  * draws faint cell borders
  * writes the Excel-style cell ID inside every cell

Usage:
    python3 cell_id_overlay.py input.pdf output.pdf \\
        [--cell-size-pts 12] [--font fonts/HomemadeApple-Regular.ttf] \\
        [--label-size 3.2] [--axis-labels]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import fitz


def col_letters(col: int) -> str:
    letters: list[str] = []
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def render(
    pdf_in: str,
    pdf_out: str,
    *,
    cell_size_pts: float,
    font_path: str | None,
    label_size: float,
    axis_labels: bool,
    grid_color: tuple[float, float, float],
    label_color: tuple[float, float, float],
) -> None:
    have_font = bool(font_path and Path(font_path).exists())
    font_alias = "hand" if have_font else "helv"

    doc = fitz.open(pdf_in)
    total_cells = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        cols = max(1, math.floor(pw / cell_size_pts))
        rows = max(1, math.floor(ph / cell_size_pts))
        left = round((pw - cols * cell_size_pts) / 2.0, 2)
        top = round((ph - rows * cell_size_pts) / 2.0, 2)

        if have_font:
            page.insert_font(fontname=font_alias, fontfile=font_path)

        # Faint grid lines
        shape = page.new_shape()
        for r in range(rows + 1):
            y = top + r * cell_size_pts
            shape.draw_line(
                fitz.Point(left, y),
                fitz.Point(left + cols * cell_size_pts, y),
            )
        for c in range(cols + 1):
            x = left + c * cell_size_pts
            shape.draw_line(
                fitz.Point(x, top),
                fitz.Point(x, top + rows * cell_size_pts),
            )
        shape.finish(color=grid_color, width=0.2)
        shape.commit()

        # Per-cell label (centered)
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                cid = f"{col_letters(c)}{r}"
                cx = left + (c - 1) * cell_size_pts + cell_size_pts / 2
                cy = top + (r - 1) * cell_size_pts + cell_size_pts / 2
                page.insert_text(
                    (cx - len(cid) * label_size * 0.27, cy + label_size * 0.32),
                    cid,
                    fontname=font_alias,
                    fontsize=label_size,
                    color=label_color,
                )
                total_cells += 1

        if axis_labels:
            fs = max(4.0, cell_size_pts * 0.45)
            for c in range(1, cols + 1):
                x = left + (c - 1) * cell_size_pts + cell_size_pts / 2 - fs * 0.55
                y = max(fs, top - 1.0)
                page.insert_text((x, y), col_letters(c), fontname=font_alias,
                                 fontsize=fs, color=(0.35, 0.35, 0.35))
            for r in range(1, rows + 1):
                x = max(0.5, left - fs * 1.6)
                y = top + (r - 1) * cell_size_pts + cell_size_pts / 2 + fs * 0.35
                if y > ph - 0.5:
                    continue
                page.insert_text((x, y), str(r), fontname=font_alias,
                                 fontsize=fs, color=(0.35, 0.35, 0.35))

        print(
            f"  page {page_idx + 1}: {rows}x{cols}  "
            f"(left={left} top={top} pw={pw:.1f} ph={ph:.1f})",
            file=sys.stderr,
        )

    doc.save(pdf_out, garbage=4, deflate=True)
    doc.close()
    print(f"wrote {pdf_out}  ({total_cells} cells across {len(doc)} pages)",
          file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf")
    p.add_argument("output_pdf")
    p.add_argument("--cell-size-pts", type=float, default=12.0)
    p.add_argument("--font", default="fonts/HomemadeApple-Regular.ttf")
    p.add_argument("--label-size", type=float, default=3.2)
    p.add_argument("--axis-labels", action="store_true",
                   help="Draw column letters / row numbers along page edges")
    p.add_argument("--grid-color", default="0.78,0.78,0.78")
    p.add_argument("--label-color", default="0.45,0.45,0.45")
    args = p.parse_args()

    def parse_rgb(s: str) -> tuple[float, float, float]:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--*-color expects R,G,B got {s!r}")
        return parts[0], parts[1], parts[2]

    render(
        args.pdf,
        args.output_pdf,
        cell_size_pts=args.cell_size_pts,
        font_path=args.font,
        label_size=args.label_size,
        axis_labels=args.axis_labels,
        grid_color=parse_rgb(args.grid_color),
        label_color=parse_rgb(args.label_color),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
