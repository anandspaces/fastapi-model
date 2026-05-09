#!/usr/bin/env python3
"""
annotate_from_response.py — Overlay annotation marks from a smart-OCR API
response JSON onto a student answer-sheet PDF.

Usage:
    python3 annotate_from_response.py input.pdf response.json font.ttf output.pdf [--dpi 250]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GridMeta:
    page: int
    rows: int
    cols: int
    cell_size_pts: float
    left_margin_pts: float
    top_margin_pts: float
    page_w_pts: float
    page_h_pts: float


@dataclass
class Annotation:
    page: int
    is_positive: bool
    comment: str
    comment_font_pts: float
    comment_rows: list[str]
    anchor_type: str        # ellipse|underline|tick|arrow|curly_brace|none
    anchor_rows: list[str]


# ---------------------------------------------------------------------------
# Cell-ID / range parsing helpers
# ---------------------------------------------------------------------------

def _col_letters_to_int(letters: str) -> int:
    """Convert column letters to 1-based int: A→1, Z→26, AA→27, etc."""
    result = 0
    for ch in letters.upper():
        result = result * 26 + (ord(ch) - ord('A') + 1)
    return result


def _parse_cell(cell: str) -> tuple[int, int]:
    """Parse a cell ID like 'AA34' → (row=34, col=27)."""
    i = 0
    while i < len(cell) and cell[i].isalpha():
        i += 1
    if i == 0 or i == len(cell):
        raise ValueError(f"Cannot parse cell ID: {cell!r}")
    col = _col_letters_to_int(cell[:i])
    row = int(cell[i:])
    return row, col


def _parse_range(token: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse a range token like 'A34:J35' → ((34,1),(35,10))."""
    if ':' not in token:
        # Treat as single-cell range
        rc = _parse_cell(token.strip())
        return rc, rc
    left, right = token.strip().split(':', 1)
    return _parse_cell(left.strip()), _parse_cell(right.strip())


def _range_to_rect(token: str, meta: GridMeta) -> Optional[fitz.Rect]:
    """Convert a range token to a PDF fitz.Rect in points."""
    try:
        (r1, c1), (r2, c2) = _parse_range(token)
    except ValueError as e:
        print(f"[WARN] Unparseable range {token!r}: {e}", file=sys.stderr)
        return None

    x1 = meta.left_margin_pts + (c1 - 1) * meta.cell_size_pts
    y1 = meta.top_margin_pts + (r1 - 1) * meta.cell_size_pts
    x2 = meta.left_margin_pts + c2 * meta.cell_size_pts
    y2 = meta.top_margin_pts + r2 * meta.cell_size_pts
    return fitz.Rect(x1, y1, x2, y2)


def _union_rects(rects: list[fitz.Rect]) -> Optional[fitz.Rect]:
    """Return the bounding union of a list of rects."""
    if not rects:
        return None
    result = rects[0]
    for r in rects[1:]:
        result = result | r
    return result


def _ranges_to_union_rect(tokens: list[str], meta: GridMeta) -> Optional[fitz.Rect]:
    rects = []
    for tok in tokens:
        r = _range_to_rect(tok, meta)
        if r is not None:
            rects.append(r)
    return _union_rects(rects)


def _ellipse_outer_rect(anchor_rows: list[str], meta: GridMeta) -> Optional[fitz.Rect]:
    """Same padding as _draw_ellipse — for leader lines and connectors."""
    rect = _ranges_to_union_rect(anchor_rows, meta)
    if rect is None:
        return None
    H_PAD = 4.0
    target_h = max(rect.height + 8.0, rect.width / 3.0)
    V_PAD = max(H_PAD, (target_h - rect.height) / 2.0)
    return fitz.Rect(
        rect.x0 - H_PAD, rect.y0 - V_PAD,
        rect.x1 + H_PAD, rect.y1 + V_PAD,
    )


def _cubic_bezier_points(
    p0: fitz.Point, p1: fitz.Point, p2: fitz.Point, p3: fitz.Point, n: int = 24
) -> list[fitz.Point]:
    """Sample a cubic Bézier for polyline drawing."""
    pts: list[fitz.Point] = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        x = (
            u**3 * p0.x
            + 3 * u**2 * t * p1.x
            + 3 * u * t**2 * p2.x
            + t**3 * p3.x
        )
        y = (
            u**3 * p0.y
            + 3 * u**2 * t * p1.y
            + 3 * u * t**2 * p2.y
            + t**3 * p3.y
        )
        pts.append(fitz.Point(x, y))
    return pts


def _draw_polyline_bezier(
    page: fitz.Page,
    p0: fitz.Point,
    p1: fitz.Point,
    p2: fitz.Point,
    p3: fitz.Point,
    color: tuple,
    width: float,
) -> None:
    pts = _cubic_bezier_points(p0, p1, p2, p3)
    page.draw_polyline(pts, color=color, width=width)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_RED = (0.85, 0.05, 0.05)


def _ink_color(is_positive: bool) -> tuple[float, float, float]:
    # All annotation marks and comments use red ink.
    return _RED


# ---------------------------------------------------------------------------
# Drawing primitives (pure PyMuPDF / points)
# ---------------------------------------------------------------------------

def _draw_ellipse(page: fitz.Page, anchor_rows: list[str], meta: GridMeta,
                  color: tuple) -> None:
    """Draw an organic-looking ellipse around the bounding rect of anchor cells."""
    r = _ellipse_outer_rect(anchor_rows, meta)
    if r is None:
        return

    # Draw three slightly offset ovals for an organic feel (aspect ≤ ~3:1)
    offsets = [
        (0.0,  0.0),
        (0.6, -0.4),
        (-0.5,  0.5),
    ]
    widths = [2.0, 1.2, 0.9]
    for (dx, dy), w in zip(offsets, widths):
        shifted = fitz.Rect(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy)
        page.draw_oval(shifted, color=color, width=w)


def _draw_underline(page: fitz.Page, anchor_rows: list[str], meta: GridMeta,
                    color: tuple) -> None:
    """Draw a gentle sine-wave underline along the bottom of each anchor row range."""
    AMPLITUDE = 1.5   # pts
    PERIOD = 20.0     # pts
    WIDTH = 1.5

    for tok in anchor_rows:
        rect = _range_to_rect(tok, meta)
        if rect is None:
            continue

        y_base = rect.y1 - 1.0   # bottom edge of the row band
        x_start = rect.x0
        x_end = rect.x1
        span = x_end - x_start
        if span <= 0:
            continue

        # Build polyline with ~2pt steps
        steps = max(int(span / 2), 2)
        points = []
        for i in range(steps + 1):
            t = i / steps
            x = x_start + t * span
            phase = (x - x_start) / PERIOD * 2 * math.pi
            y = y_base + AMPLITUDE * math.sin(phase)
            points.append(fitz.Point(x, y))

        page.draw_polyline(points, color=color, width=WIDTH)


def _draw_tick(page: fitz.Page, anchor_rows: list[str], meta: GridMeta,
               color: tuple) -> None:
    """Draw a ✓ checkmark at the centre of the bounding rect."""
    rect = _ranges_to_union_rect(anchor_rows, meta)
    if rect is None:
        return

    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2

    # Lower-left origin for the tick
    ox = cx - 7.0
    oy = cy + 3.0

    # Short foot stroke: down-right ~6pt
    foot_end = fitz.Point(ox + 5.0, oy + 3.0)
    # Long arm stroke: from foot_end up-right ~14pt
    arm_end = fitz.Point(ox + 5.0 + 10.0, oy + 3.0 - 12.0)

    page.draw_line(fitz.Point(ox, oy), foot_end, color=color, width=2.0)
    page.draw_line(foot_end, arm_end, color=color, width=2.0)


def _draw_arrow(
    page: fitz.Page,
    anchor_rows: list[str],
    comment_rows: list[str],
    meta: GridMeta,
    color: tuple,
    *,
    shaft_len: float = 30.0,
) -> None:
    """Bold arrow from margin toward anchor rect (comment side → ink)."""
    anchor_rect = _ranges_to_union_rect(anchor_rows, meta)
    comment_rect = _ranges_to_union_rect(comment_rows, meta)
    if anchor_rect is None:
        return

    acx = (anchor_rect.x0 + anchor_rect.x1) / 2
    acy = (anchor_rect.y0 + anchor_rect.y1) / 2
    if comment_rect is not None:
        ccx = (comment_rect.x0 + comment_rect.x1) / 2
        comment_left_of_anchor = ccx < acx
    else:
        comment_left_of_anchor = False

    # Shaft from outside anchor pointing inward
    if comment_left_of_anchor:
        start = fitz.Point(anchor_rect.x0 - shaft_len, acy)
        end = fitz.Point(anchor_rect.x0, acy)
        inward = fitz.Point(1.0, 0.0)
    else:
        start = fitz.Point(anchor_rect.x1 + shaft_len, acy)
        end = fitz.Point(anchor_rect.x1, acy)
        inward = fitz.Point(-1.0, 0.0)

    page.draw_line(start, end, color=color, width=2.0)

    # Arrowhead at ``end`` pointing along shaft into the anchor
    vx = end.x - start.x
    vy = end.y - start.y
    ln = math.hypot(vx, vy) or 1.0
    ux, uy = vx / ln, vy / ln
    px, py = -uy, ux
    for sign in (-1.0, 1.0):
        page.draw_line(
            end,
            fitz.Point(end.x - 8.0 * ux + sign * 4.0 * px, end.y - 8.0 * uy + sign * 4.0 * py),
            color=color,
            width=2.0,
        )


def _draw_curly_brace(
    page: fitz.Page,
    anchor_rows: list[str],
    comment_rows: list[str],
    meta: GridMeta,
    color: tuple,
) -> None:
    """Draw a tutor-style { or } spanning anchor_rows; side from comment position."""
    anchor_rect = _ranges_to_union_rect(anchor_rows, meta)
    if anchor_rect is None:
        return

    comment_rect = _ranges_to_union_rect(comment_rows, meta)
    if comment_rect is None:
        ccx = anchor_rect.x0 - 50.0
    else:
        ccx = (comment_rect.x0 + comment_rect.x1) / 2
    acx = (anchor_rect.x0 + anchor_rect.x1) / 2
    left_brace = ccx < acx  # "{" on left of anchor block

    y_top = anchor_rect.y0
    y_bot = anchor_rect.y1
    y_mid = (y_top + y_bot) / 2.0
    tip = 10.0

    if left_brace:
        bx = anchor_rect.x0
        tip_x = bx - tip
        # Upper half of '{'
        _draw_polyline_bezier(
            page,
            fitz.Point(bx, y_top),
            fitz.Point(tip_x, y_top),
            fitz.Point(tip_x, y_mid),
            fitz.Point(tip_x, y_mid),
            color,
            2.0,
        )
        _draw_polyline_bezier(
            page,
            fitz.Point(tip_x, y_mid),
            fitz.Point(tip_x, y_mid),
            fitz.Point(tip_x, y_bot),
            fitz.Point(bx, y_bot),
            color,
            2.0,
        )
    else:
        bx = anchor_rect.x1
        tip_x = bx + tip
        _draw_polyline_bezier(
            page,
            fitz.Point(bx, y_top),
            fitz.Point(tip_x, y_top),
            fitz.Point(tip_x, y_mid),
            fitz.Point(tip_x, y_mid),
            color,
            2.0,
        )
        _draw_polyline_bezier(
            page,
            fitz.Point(tip_x, y_mid),
            fitz.Point(tip_x, y_mid),
            fitz.Point(tip_x, y_bot),
            fitz.Point(bx, y_bot),
            color,
            2.0,
        )


def _edge_midpoint_facing(rect: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    """Midpoint of the edge of ``rect`` that faces ``toward``."""
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    dx = toward.x - cx
    dy = toward.y - cy
    if abs(dx) >= abs(dy):
        if dx >= 0:
            return fitz.Point(rect.x1, cy)
        return fitz.Point(rect.x0, cy)
    if dy >= 0:
        return fitz.Point(cx, rect.y1)
    return fitz.Point(cx, rect.y0)


def _edge_anchor_points(
    comment_rect: fitz.Rect,
    anchor_rect: fitz.Rect,
) -> tuple[fitz.Point, fitz.Point]:
    """Leader from comment edge (facing anchor centre) to anchor edge (facing comment centre)."""
    cc = fitz.Point((comment_rect.x0 + comment_rect.x1) / 2, (comment_rect.y0 + comment_rect.y1) / 2)
    ac = fitz.Point((anchor_rect.x0 + anchor_rect.x1) / 2, (anchor_rect.y0 + anchor_rect.y1) / 2)
    start = _edge_midpoint_facing(comment_rect, ac)
    end = _edge_midpoint_facing(anchor_rect, cc)
    return start, end


def _draw_leader_line(
    page: fitz.Page,
    comment_rows: list[str],
    anchor_rows: list[str],
    anchor_type: str,
    meta: GridMeta,
    color: tuple,
) -> None:
    """Thin arrow-tipped line from comment box to anchor (ellipse padding rect)."""
    if anchor_type in ("none", "arrow", "curly_brace"):
        return
    comment_rect = _ranges_to_union_rect(comment_rows, meta)
    if comment_rect is None:
        return

    if anchor_type == "ellipse":
        anchor_outer = _ellipse_outer_rect(anchor_rows, meta)
    else:
        anchor_outer = _ranges_to_union_rect(anchor_rows, meta)
    if anchor_outer is None:
        return

    start, end = _edge_anchor_points(comment_rect, anchor_outer)
    page.draw_line(start, end, color=color, width=0.8)

    ang = math.atan2(end.y - start.y, end.x - start.x)
    for da in (math.radians(150), math.radians(-150)):
        dx = 5.0 * math.cos(ang + da)
        dy = 5.0 * math.sin(ang + da)
        page.draw_line(end, fitz.Point(end.x + dx, end.y + dy), color=color, width=0.8)


def _draw_comment_box(
    page: fitz.Page,
    comment_rows: list[str],
    meta: GridMeta,
    color: tuple,
) -> None:
    """Dashed outline around the union of comment_rows."""
    rect = _ranges_to_union_rect(comment_rows, meta)
    if rect is None or rect.is_empty:
        return
    page.draw_rect(rect, color=color, width=0.5, dashes="[2 2] 0")


# ---------------------------------------------------------------------------
# Comment text rendering
# ---------------------------------------------------------------------------

_registered_fonts: set[int] = set()   # set of fitz.Page xref values


def _ensure_font(page: fitz.Page, fontname: str, font_path: str) -> None:
    """Register the TTF font on this page (once per page)."""
    page_xref = page.xref
    if page_xref not in _registered_fonts:
        page.insert_font(fontname=fontname, fontfile=font_path)
        _registered_fonts.add(page_xref)


def _render_comment(page: fitz.Page, comment: str, comment_rows: list[str],
                    font_pts: float, meta: GridMeta, color: tuple,
                    font_path: str, fontname: str = "hwfont") -> None:
    """Render comment text inside the bounding rect of comment_rows.

    The cell ranges are often narrow (left/right margin cols only), so the
    rect height is expanded to accommodate the text.  Font size is reduced
    progressively (down to 8 pt) if the text still overflows.
    """
    rects = [_range_to_rect(tok, meta) for tok in comment_rows]
    rects = [r for r in rects if r is not None and not r.is_empty]
    if not rects:
        return

    union = _union_rects(rects)
    if union is None or union.is_empty:
        return

    _ensure_font(page, fontname, font_path)

    # Production display: 16–18 pt regardless of JSON payload.
    font_pts = max(16.0, min(18.0, float(font_pts)))

    # Estimate how many lines the comment needs at the given font size and
    # available width, then expand the rect height to fit.
    def _expanded_rect(fs: float) -> fitz.Rect:
        avail_w = max(union.width, 10.0)
        # HomemadeApple is a wide font; ~0.7 pt per char at given size is a
        # reasonable approximation for line-wrapping estimation.
        chars_per_line = max(1, int(avail_w / (fs * 0.70)))
        num_lines = math.ceil(len(comment) / chars_per_line) + 1
        needed_h = num_lines * fs * 1.5
        new_y1 = min(meta.page_h_pts - 2.0, union.y0 + max(union.height, needed_h))
        return fitz.Rect(union.x0, union.y0, union.x1, new_y1)

    # Try progressively smaller font sizes until text fits.
    for fs in range(int(font_pts), 9, -1):
        rect = _expanded_rect(float(fs))
        rc = page.insert_textbox(
            rect, comment,
            fontname=fontname,
            fontsize=float(fs),
            color=color,
            align=0,
        )
        if rc >= 0:
            return

    # Last resort at 8 pt with a generous rect extending to page bottom.
    rect = fitz.Rect(union.x0, union.y0, union.x1, meta.page_h_pts - 2.0)
    rc = page.insert_textbox(
        rect, comment,
        fontname=fontname,
        fontsize=8.0,
        color=color,
        align=0,
    )
    if rc < 0:
        print(
            f"[WARN] Comment still overflows at 8 pt (page {page.number + 1}): "
            f"{comment[:40]!r}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _load_grid_meta(data: dict) -> dict[int, GridMeta]:
    """Parse cellGridMeta array into a page→GridMeta dict."""
    metas: dict[int, GridMeta] = {}
    for entry in data.get("cellGridMeta", []):
        try:
            gm = GridMeta(
                page=int(entry["page"]),
                rows=int(entry["rows"]),
                cols=int(entry["cols"]),
                cell_size_pts=float(entry["cell_size_pts"]),
                left_margin_pts=float(entry["left_margin_pts"]),
                top_margin_pts=float(entry["top_margin_pts"]),
                page_w_pts=float(entry["page_w_pts"]),
                page_h_pts=float(entry["page_h_pts"]),
            )
            metas[gm.page] = gm
        except (KeyError, ValueError, TypeError) as e:
            print(f"[WARN] Skipping bad cellGridMeta entry: {e}", file=sys.stderr)
    return metas


def _load_annotations(data: dict) -> list[Annotation]:
    """Flatten data.items[*].annotations into a list of Annotation objects."""
    annotations: list[Annotation] = []
    for item in data.get("items", []):
        for raw in item.get("annotations", []):
            try:
                anchor = raw.get("anchor") or {}
                ann = Annotation(
                    page=int(raw["page"]),
                    is_positive=bool(raw.get("is_positive", False)),
                    comment=str(raw.get("comment", "")),
                    comment_font_pts=float(raw.get("comment_font_pts", 12.0)),
                    comment_rows=list(raw.get("comment_rows") or []),
                    anchor_type=str(anchor.get("type", "none")).lower(),
                    anchor_rows=list(anchor.get("rows") or []),
                )
                annotations.append(ann)
            except (KeyError, ValueError, TypeError) as e:
                print(f"[WARN] Skipping bad annotation: {e} — {raw!r}", file=sys.stderr)
    return annotations


# ---------------------------------------------------------------------------
# Per-page drawing dispatcher
# ---------------------------------------------------------------------------

def _draw_annotation(page: fitz.Page, ann: Annotation, meta: GridMeta,
                     font_path: str) -> None:
    color = _ink_color(ann.is_positive)

    # --- Anchor mark ---
    if ann.anchor_type == "ellipse":
        if ann.anchor_rows:
            _draw_ellipse(page, ann.anchor_rows, meta, color)
        else:
            print(
                f"[WARN] page {ann.page}: ellipse anchor has no rows, skipping mark",
                file=sys.stderr,
            )
    elif ann.anchor_type == "underline":
        if ann.anchor_rows:
            _draw_underline(page, ann.anchor_rows, meta, color)
        else:
            print(
                f"[WARN] page {ann.page}: underline anchor has no rows, skipping mark",
                file=sys.stderr,
            )
    elif ann.anchor_type == "tick":
        if ann.anchor_rows:
            _draw_tick(page, ann.anchor_rows, meta, color)
        else:
            print(
                f"[WARN] page {ann.page}: tick anchor has no rows, skipping mark",
                file=sys.stderr,
            )
    elif ann.anchor_type == "arrow":
        if ann.anchor_rows:
            _draw_arrow(
                page, ann.anchor_rows, ann.comment_rows, meta, color,
            )
        else:
            print(
                f"[WARN] page {ann.page}: arrow anchor has no rows, skipping mark",
                file=sys.stderr,
            )
    elif ann.anchor_type == "curly_brace":
        if ann.anchor_rows:
            _draw_curly_brace(
                page, ann.anchor_rows, ann.comment_rows, meta, color,
            )
        else:
            print(
                f"[WARN] page {ann.page}: curly_brace anchor has no rows, skipping mark",
                file=sys.stderr,
            )
    elif ann.anchor_type == "none":
        pass  # intentionally no mark
    else:
        print(
            f"[WARN] page {ann.page}: unknown anchor type {ann.anchor_type!r}, skipping mark",
            file=sys.stderr,
        )

    # --- Comment text, dashed box, leader (order per plan) ---
    if ann.comment and ann.comment_rows:
        _render_comment(
            page, ann.comment, ann.comment_rows, ann.comment_font_pts,
            meta, color, font_path,
        )
        _draw_comment_box(page, ann.comment_rows, meta, color)
        _draw_leader_line(
            page,
            ann.comment_rows,
            ann.anchor_rows,
            ann.anchor_type,
            meta,
            color,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def annotate(
    input_pdf: str,
    response_json: str,
    font_ttf: str,
    output_pdf: str,
) -> None:
    print(f"[INFO] Loading response JSON: {response_json}", file=sys.stderr)
    with open(response_json, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("data", raw)   # tolerate both {data:{...}} and flat shape

    grid_metas = _load_grid_meta(data)
    annotations = _load_annotations(data)

    print(
        f"[INFO] Loaded {len(annotations)} annotation(s) across "
        f"{len(grid_metas)} grid page(s)",
        file=sys.stderr,
    )

    # Group annotations by page
    by_page: dict[int, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_page[ann.page].append(ann)

    print(f"[INFO] Opening PDF: {input_pdf}", file=sys.stderr)
    doc = fitz.open(input_pdf)

    for page_no_1based, page_annotations in sorted(by_page.items()):
        if page_no_1based not in grid_metas:
            print(
                f"[WARN] No cellGridMeta for page {page_no_1based}, "
                f"skipping {len(page_annotations)} annotation(s)",
                file=sys.stderr,
            )
            continue

        page_idx = page_no_1based - 1
        if page_idx < 0 or page_idx >= doc.page_count:
            print(
                f"[WARN] Page {page_no_1based} out of range "
                f"(PDF has {doc.page_count} page(s)), skipping",
                file=sys.stderr,
            )
            continue

        meta = grid_metas[page_no_1based]
        page = doc[page_idx]

        print(
            f"[INFO] Drawing {len(page_annotations)} annotation(s) on page {page_no_1based}",
            file=sys.stderr,
        )
        for ann in page_annotations:
            _draw_annotation(page, ann, meta, font_ttf)

    print(f"[INFO] Saving annotated PDF: {output_pdf}", file=sys.stderr)
    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    print("[INFO] Done.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay annotation marks from a smart-OCR response JSON onto a PDF."
    )
    parser.add_argument("input_pdf", help="Path to the student answer-sheet PDF")
    parser.add_argument("response_json", help="Path to the smart-OCR API response JSON")
    parser.add_argument("font_ttf", help="Path to the handwriting TTF font file")
    parser.add_argument("output_pdf", help="Where to write the annotated PDF")
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI hint (currently informational; PyMuPDF operates in pts). Default: 250",
    )
    args = parser.parse_args()

    annotate(args.input_pdf, args.response_json, args.font_ttf, args.output_pdf)


if __name__ == "__main__":
    _cli()
