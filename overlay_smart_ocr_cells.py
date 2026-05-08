"""overlay_smart_ocr_cells.py — render smart-ocr annotation cells onto a PDF.

For every annotation in the response, the script paints each referenced cell
(from cell_ids) as a translucent rectangle and writes the cell ID inside it.
Per-item, it also draws an outline of the marking_box region and a score
circle at marking_x/y_position_percent.

Usage:
    python3 overlay_smart_ocr_cells.py input.pdf response.json output.pdf \\
        [--font fonts/HomemadeApple-Regular.ttf] \\
        [--label-size 6] [--alpha 0.22]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

from cell_grid_service_v4 import rc_from_cell_id

GREEN = (0.10, 0.55, 0.20)
RED = (0.80, 0.15, 0.15)
BLUE = (0.20, 0.25, 0.75)


def cell_rect(meta: dict, cell_id: str) -> fitz.Rect:
    row, col = rc_from_cell_id(cell_id)
    cs = meta["cell_size_pts"]
    lm = meta["left_margin_pts"]
    tm = meta["top_margin_pts"]
    return fitz.Rect(
        lm + (col - 1) * cs,
        tm + (row - 1) * cs,
        lm + col * cs,
        tm + row * cs,
    )


def pct_to_xy(pw: float, ph: float, x_pct: float, y_pct: float) -> tuple[float, float]:
    return pw * x_pct / 100.0, ph * y_pct / 100.0


def pct_to_rect(pw: float, ph: float, x1p, y1p, x2p, y2p) -> fitz.Rect:
    return fitz.Rect(pw * x1p / 100.0, ph * y1p / 100.0, pw * x2p / 100.0, ph * y2p / 100.0)


def render(
    pdf_in: str,
    response_json: str,
    pdf_out: str,
    *,
    font_path: str | None,
    label_size: float,
    alpha: float,
) -> None:
    api = json.loads(Path(response_json).read_text())
    data = api.get("data", api)
    meta_by_page = {m["page"]: m for m in data.get("cellGridMeta", [])}
    items = data.get("items", [])

    have_font = bool(font_path and Path(font_path).exists())
    font_obj = fitz.Font(fontfile=font_path) if have_font else fitz.Font("helv")
    font_alias = "hand" if have_font else "helv"

    def text_width(s: str, sz: float) -> float:
        return font_obj.text_length(s, fontsize=sz)

    # Group annotations by their bbox.page (1-indexed PDF page)
    anns_by_page: dict[int, list[dict]] = {}
    for item in items:
        for ann in item.get("annotations") or []:
            if not ann.get("cell_ids"):
                continue
            page = (ann.get("bbox") or {}).get("page")
            if page is None:
                continue
            anns_by_page.setdefault(page, []).append(ann)

    items_by_mark_page: dict[int, list[dict]] = {}
    items_by_box_page: dict[int, list[dict]] = {}
    for item in items:
        if item.get("marking_page"):
            items_by_mark_page.setdefault(item["marking_page"], []).append(item)
        if item.get("marking_box_page"):
            items_by_box_page.setdefault(item["marking_box_page"], []).append(item)

    doc = fitz.open(pdf_in)
    n_anns = n_cells = n_oob = 0

    for page_idx in range(len(doc)):
        page_no = page_idx + 1
        page = doc[page_idx]
        pw = page.rect.width
        ph = page.rect.height
        meta = meta_by_page.get(page_no)

        if have_font:
            page.insert_font(fontname=font_alias, fontfile=font_path)

        # 1) Annotation cells ------------------------------------------------
        if meta:
            for ann in anns_by_page.get(page_no, []):
                color = GREEN if ann.get("is_positive") else RED
                n_anns += 1
                for cid in ann["cell_ids"]:
                    try:
                        r, c = rc_from_cell_id(cid)
                    except Exception as exc:
                        print(f"  page {page_no}: bad cell_id {cid!r}: {exc}", file=sys.stderr)
                        continue
                    if r < 1 or r > meta["rows"] or c < 1 or c > meta["cols"]:
                        n_oob += 1
                        print(
                            f"  page {page_no}: cell {cid} out of grid "
                            f"({meta['rows']}x{meta['cols']})",
                            file=sys.stderr,
                        )
                    rect = cell_rect(meta, cid)
                    page.draw_rect(
                        rect,
                        color=color,
                        fill=color,
                        fill_opacity=alpha,
                        width=0.35,
                    )
                    tw = text_width(cid, label_size)
                    cx = rect.x0 + rect.width / 2 - tw / 2
                    cy = rect.y0 + rect.height / 2 + label_size * 0.32
                    page.insert_text(
                        (cx, cy),
                        cid,
                        fontname=font_alias,
                        fontsize=label_size,
                        color=color,
                    )
                    n_cells += 1

        # 2) Marking-box outline --------------------------------------------
        for item in items_by_box_page.get(page_no, []):
            xs = [
                item.get("marking_box_x1_percent"),
                item.get("marking_box_y1_percent"),
                item.get("marking_box_x2_percent"),
                item.get("marking_box_y2_percent"),
            ]
            if any(v is None for v in xs):
                continue
            rect = pct_to_rect(pw, ph, *xs)
            page.draw_rect(rect, color=BLUE, fill=None, width=0.9)
            tag = f"Q{item.get('question_id', '?')}"
            page.insert_text(
                (rect.x0 + 2, max(rect.y0 - 2, 6)),
                tag,
                fontname=font_alias,
                fontsize=8,
                color=BLUE,
            )

        # 3) Score circle ----------------------------------------------------
        for item in items_by_mark_page.get(page_no, []):
            mx = item.get("marking_x_position_percent")
            my = item.get("marking_y_position_percent")
            if mx is None or my is None:
                continue
            cx, cy = pct_to_xy(pw, ph, mx, my)
            page.draw_circle((cx, cy), 16, color=BLUE, width=1.2)
            max_m = item.get("max_marks") or 0
            if max_m:
                awarded = item.get("marks_awarded", 0) or 0
                label = f"{awarded:g}/{max_m:g}"
            else:
                label = (item.get("status") or "?").upper()[:3]
            sz = 10.0
            tw = text_width(label, sz)
            page.insert_text(
                (cx - tw / 2, cy + sz * 0.32),
                label,
                fontname=font_alias,
                fontsize=sz,
                color=BLUE,
            )

    doc.save(pdf_out, garbage=4, deflate=True)
    doc.close()

    print(
        f"wrote {pdf_out}  "
        f"({n_anns} annotations, {n_cells} cells, {n_oob} out-of-range)",
        file=sys.stderr,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf")
    p.add_argument("response_json")
    p.add_argument("output_pdf")
    p.add_argument("--font", default="fonts/HomemadeApple-Regular.ttf")
    p.add_argument("--label-size", type=float, default=6.0)
    p.add_argument("--alpha", type=float, default=0.22)
    args = p.parse_args()

    render(
        args.pdf,
        args.response_json,
        args.output_pdf,
        font_path=args.font,
        label_size=args.label_size,
        alpha=args.alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
