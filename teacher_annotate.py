#!/usr/bin/env python3
"""
teacher_annotate.py — Unified UPSC answer-sheet annotation engine.

Single-file pipeline:
  PDF in  →  rasterise  →  detect layout (lines/gaps/strips/word-gaps/free-rects)
                       →  resolve remarks (auto coords + zone fallback)
                       →  draw teacher-authentic primitives + handwriting text
                       →  PDF out

CLI:
  python3 teacher_annotate.py input.pdf remarks.json output.pdf [--dpi 250]
  python3 teacher_annotate.py input.pdf --analyze-only          # print layout JSON
  python3 teacher_annotate.py input.pdf --scaffold-dense        # emit dense remarks JSON
  python3 teacher_annotate.py input.pdf remarks.json output.pdf --debug-freespace
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1   Imports + Constants
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any

import cv2
import fitz
import img2pdf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

# Severity → BGR colour (cv2 uses BGR; we store as RGB and convert at draw time)
RED = (190, 20, 20)
RED_SOFT = (210, 55, 55)
GREEN = (15, 130, 45)

SEVERITY_COLOR = {
    "critical": RED,
    "suggestion": RED_SOFT,
    "positive": GREEN,
}

# Page zone fractions (constant across PDFs)
LEFT_MARGIN_FRAC = 0.135
RIGHT_MARGIN_FRAC = 0.875

# Drawing priority (lower draws first)
DRAW_PRIORITY = {
    "score_circle": 0,
    "tick": 1,
    "curly_brace": 2,
    "underline_remark": 3,
    "circle_remark": 4,
    "exponent_insert": 5,
    "arrow_remark": 6,
    "text": 7,
}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(_THIS_DIR, "fonts")


def _font_path() -> str:
    candidates = [
        os.path.join(_FONTS_DIR, "HomemadeApple-Regular.ttf"),
        os.path.join(_THIS_DIR, "HomemadeApple-Regular.ttf"),
        os.path.join(_THIS_DIR, "HomemadeApple.ttf"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"HomemadeApple-Regular.ttf not found in {_FONTS_DIR!r} or {_THIS_DIR!r}"
    )


FONT_PATH = _font_path()


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def _seed(x: float, y: float) -> None:
    """Deterministic per-position randomness. Same (x,y) → same wobble."""
    random.seed(int(x) * 31337 + int(y))


def _color_for(severity: str) -> tuple[int, int, int]:
    return SEVERITY_COLOR.get(severity, RED_SOFT)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2   Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GapInfo:
    after_line: int          # 1-indexed: gap follows this line
    y_top: int               # px
    y_bot: int               # px
    height: int              # px
    y_center: int            # px


@dataclass
class FreeStrip:
    kind: str                # 'left_margin' | 'right_margin' | 'inter_line_gap'
    y_top: int; y_bot: int
    x_left: int; x_right: int
    height: int
    after_line: int = 0
    before_line: int = 0


@dataclass
class WordGap:
    line: int                # 1-indexed
    x_center: int            # px
    word_left_end: int
    word_right_start: int
    width: int


@dataclass
class FreeRect:
    x1: int; y1: int; x2: int; y2: int
    area: int
    centre_x_pct: float
    centre_y_pct: float
    nearest_line: int


@dataclass
class LayoutMap:
    page: int
    W: int; H: int; dpi: int
    scale: float                                # dpi/150
    line_ys: list[int]                          # px, 0-indexed list
    line_half_height: int                       # LH
    gaps: list[GapInfo]                         # body gaps (annotator-grade)
    left_strips: list[FreeStrip]
    right_strips: list[FreeStrip]
    word_gaps: list[WordGap]
    free_rects: list[FreeRect]
    lmx: int; rmx: int                          # margin boundaries (px)

    # ── helpers ─────────────────────────────────────────────────────────
    def line_y(self, n: int) -> int:
        if not self.line_ys:
            return 0
        return self.line_ys[max(0, min(n - 1, len(self.line_ys) - 1))]

    def best_gap_for_line(self, line: int, max_dist: int = 5) -> GapInfo | None:
        below = [g for g in self.gaps
                 if line <= g.after_line <= line + max_dist]
        above = [g for g in self.gaps
                 if line - max_dist <= g.after_line < line]
        if below:
            return min(below, key=lambda g: g.after_line)
        if above:
            return max(above, key=lambda g: g.after_line)
        return None

    def word_gaps_on_line(self, line: int) -> list[WordGap]:
        return sorted([g for g in self.word_gaps if g.line == line],
                      key=lambda g: g.x_center)

    def best_score_position(self) -> tuple[float, int]:
        """Return (cx_pct, cy_line) for a top-area score circle."""
        top_right = [r for r in self.free_rects
                     if r.centre_x_pct > 60 and r.centre_y_pct < 25]
        if top_right:
            best = max(top_right, key=lambda r: r.area)
            return best.centre_x_pct, max(1, best.nearest_line)
        # Fallback: top-left of detected text
        return 8.0, 1

    def summary(self) -> str:
        return (f"Page {self.page}: {len(self.line_ys)} lines | "
                f"{len(self.gaps)} body gaps | "
                f"{len(self.left_strips)} L-strips, {len(self.right_strips)} R-strips | "
                f"{len(self.word_gaps)} word-gaps | {len(self.free_rects)} free-rects")


@dataclass
class ResolvedRemark:
    """All coordinates resolved to absolute pixel positions."""
    type: str
    severity: str
    color: tuple[int, int, int]                 # RGB
    line_idx: int = 0                            # 0-indexed
    x_start_px: int = 0
    x_end_px: int = 0
    cx_px: int = 0
    cy_px: int = 0
    text: str = ""
    remark: str = ""
    remark_zone: str = "right_margin"
    gap: GapInfo | None = None
    zone: str = "left_margin"                    # for tick / curly_brace
    line_start_idx: int = 0
    line_end_idx: int = 0
    raw: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3   PDF Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_pages_pil(pdf_in: str, dpi: int) -> list[Image.Image]:
    """Rasterise PDF to RGB PIL images. PyMuPDF only — no poppler dependency."""
    doc = fitz.open(pdf_in)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pages: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if pix.n == 4:
            img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
        else:
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4   Layout Intelligence — build_layout_map() runs ONCE per page
# ─────────────────────────────────────────────────────────────────────────────

def _ink_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)


def detect_lines(gray: np.ndarray, dpi: int) -> list[int]:
    """Connected-component centroid clustering → text baseline y-positions (px)."""
    bw = _ink_mask(gray)
    sc = dpi / 150.0
    n, _, stats, centroids = cv2.connectedComponentsWithStats(bw)
    char_ys = sorted(
        centroids[i][1] for i in range(1, n)
        if (int(12 * sc) < stats[i][3] < int(90 * sc) and
            int(5 * sc) < stats[i][2] < int(150 * sc) and
            int(40 * sc * sc) < stats[i][4] < int(8000 * sc * sc))
    )
    if not char_ys:
        return []
    cg = int(22 * sc); mt = int(35 * sc)
    raw: list[int] = []
    bucket = [char_ys[0]]
    for y in char_ys[1:]:
        if y - bucket[-1] < cg:
            bucket.append(y)
        else:
            if len(bucket) >= 3:
                raw.append(int(np.median(bucket)))
            bucket = [y]
    if len(bucket) >= 3:
        raw.append(int(np.median(bucket)))
    if not raw:
        return []
    merged = [raw[0]]
    for y in raw[1:]:
        if y - merged[-1] > mt:
            merged.append(y)
    return merged


def detect_gaps(line_ys: list[int], H: int, lh: int, min_gap: int) -> list[GapInfo]:
    """Inter-line gaps (annotator-grade — matches old detect_gaps)."""
    gaps: list[GapInfo] = []
    for i in range(len(line_ys) - 1):
        gt = line_ys[i] + lh + 4
        gb = line_ys[i + 1] - lh - 4
        if gb - gt >= min_gap:
            gaps.append(GapInfo(
                after_line=i + 1,
                y_top=gt, y_bot=gb,
                height=gb - gt, y_center=(gt + gb) // 2,
            ))
    if line_ys:
        gt = line_ys[-1] + lh + 4
        if H - 4 - gt >= min_gap:
            gaps.append(GapInfo(
                after_line=len(line_ys),
                y_top=gt, y_bot=H - 4,
                height=H - 4 - gt, y_center=(gt + H - 4) // 2,
            ))
    return gaps


def detect_strips(gray: np.ndarray, dpi: int, W: int,
                  lmx: int, rmx: int
                  ) -> tuple[list[FreeStrip], list[FreeStrip]]:
    """Technique 1 — morphological dilation → free margin strips."""
    bw = _ink_mask(gray)
    sc = dpi / 150.0
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(30 * sc), 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(6 * sc)))
    dil = cv2.dilate(cv2.dilate(bw, h_kern), v_kern)

    def _runs(occ: np.ndarray, min_h: int) -> list[tuple[int, int]]:
        runs, in_f, start = [], False, 0
        for y, is_occ in enumerate(occ):
            if not is_occ and not in_f:
                start = y; in_f = True
            elif is_occ and in_f:
                if y - start >= min_h:
                    runs.append((start, y))
                in_f = False
        if in_f and len(occ) - start >= min_h:
            runs.append((start, len(occ)))
        return runs

    min_h = max(30, int(35 * sc))
    lm_occ = dil[:, :lmx].sum(axis=1) > 80
    rm_occ = dil[:, rmx:].sum(axis=1) > 80

    def _build(runs, kind, x_left, x_right) -> list[FreeStrip]:
        return [FreeStrip(kind=kind, y_top=y0, y_bot=y1,
                          x_left=x_left, x_right=x_right, height=y1 - y0)
                for y0, y1 in runs]

    return (_build(_runs(lm_occ, min_h), "left_margin", 4, lmx - 4),
            _build(_runs(rm_occ, min_h), "right_margin", rmx + 4, W - 4))


def detect_word_gaps(gray: np.ndarray, line_ys: list[int], dpi: int,
                     H: int, lmx: int, rmx: int) -> list[WordGap]:
    """Technique 2 — column projection → inter-word gaps per line."""
    sc = dpi / 150.0
    LH = max(12, H // 55)
    min_gap_px = max(8, int(10 * sc))
    min_word_px = max(12, int(15 * sc))
    merge_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(20 * sc), 1))
    out: list[WordGap] = []

    for line_idx, line_y in enumerate(line_ys):
        line_num = line_idx + 1
        y_top = max(0, line_y - LH); y_bot = min(H, line_y + LH)
        band_gray = gray[y_top:y_bot, lmx:rmx]
        bw = cv2.adaptiveThreshold(
            cv2.GaussianBlur(band_gray, (3, 3), 0), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        dil = cv2.dilate(bw, merge_kern)
        proj = dil.sum(axis=0).astype(np.float32)
        if proj.max() == 0:
            continue
        thr = proj.max() * 0.05
        words: list[tuple[int, int]] = []
        in_word = False; ws = 0
        for x, v in enumerate(proj):
            if v > thr and not in_word:
                ws = x; in_word = True
            elif v <= thr and in_word:
                if x - ws >= min_word_px:
                    words.append((ws, x))
                in_word = False
        if in_word and len(proj) - ws >= min_word_px:
            words.append((ws, len(proj)))

        for i in range(len(words) - 1):
            gap_start = words[i][1]; gap_end = words[i + 1][0]
            if gap_end - gap_start < min_gap_px:
                continue
            raw = bw.sum(axis=0).astype(np.float32)
            raw_thr = raw.max() * 0.03 if raw.max() > 0 else 1
            left_end = gap_start
            for x in range(min(gap_start + int(25 * sc), len(raw)) - 1,
                           max(0, gap_start - int(5 * sc)), -1):
                if raw[x] > raw_thr:
                    left_end = x; break
            right_start = gap_end
            for x in range(max(0, gap_end - int(25 * sc)),
                           min(len(raw), gap_end + int(5 * sc))):
                if raw[x] > raw_thr:
                    right_start = x; break
            cx = lmx + (left_end + right_start) // 2
            out.append(WordGap(
                line=line_num, x_center=cx,
                word_left_end=lmx + left_end,
                word_right_start=lmx + right_start,
                width=right_start - left_end,
            ))
    return out


def detect_free_rects(gray: np.ndarray, line_ys: list[int], dpi: int,
                      W: int, H: int, min_area: int = 2000) -> list[FreeRect]:
    """Technique 3 — Canny + dilated edges + inverted mask → free rectangles."""
    sc = dpi / 150.0
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 30, 90)
    dil_px = max(8, int(12 * sc))
    dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
    dil_edges = cv2.dilate(edges, dil_kern)
    ink_expanded = cv2.dilate(_ink_mask(gray), dil_kern)
    occupied = cv2.bitwise_or(dil_edges, ink_expanded)
    free_mask = cv2.bitwise_not(occupied)
    close_px = max(15, int(20 * sc))
    close_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, close_kern)

    contours, _ = cv2.findContours(
        free_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[FreeRect] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        cont_area = cv2.contourArea(cnt)
        if cont_area / max(area, 1) < 0.35:
            continue
        x2, y2 = x + w, y + h
        nearest = 1
        if line_ys:
            cy = (y + y2) // 2
            nearest = min(range(len(line_ys)),
                          key=lambda i: abs(line_ys[i] - cy)) + 1
        rects.append(FreeRect(
            x1=x, y1=y, x2=x2, y2=y2, area=area,
            centre_x_pct=round((x + x2) / 2 / W * 100, 2),
            centre_y_pct=round((y + y2) / 2 / H * 100, 2),
            nearest_line=nearest,
        ))
    return sorted(rects, key=lambda r: r.area, reverse=True)


def build_layout_map(pil_img: Image.Image, page_no: int, dpi: int) -> LayoutMap:
    """Build the complete LayoutMap for one page. Run ONCE per page."""
    W, H = pil_img.size
    gray = np.array(pil_img.convert("L"))
    sc = dpi / 150.0
    LH = max(12, H // 55)
    lmx = int(W * LEFT_MARGIN_FRAC); rmx = int(W * RIGHT_MARGIN_FRAC)
    min_gap = max(40, int(55 * sc))

    line_ys = detect_lines(gray, dpi)
    gaps = detect_gaps(line_ys, H, LH, min_gap)
    left_strips, right_strips = detect_strips(gray, dpi, W, lmx, rmx)
    word_gaps = detect_word_gaps(gray, line_ys, dpi, H, lmx, rmx)
    free_rects = detect_free_rects(gray, line_ys, dpi, W, H)

    return LayoutMap(
        page=page_no, W=W, H=H, dpi=dpi, scale=sc,
        line_ys=line_ys, line_half_height=LH,
        gaps=gaps, left_strips=left_strips, right_strips=right_strips,
        word_gaps=word_gaps, free_rects=free_rects,
        lmx=lmx, rmx=rmx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5   Snapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_ink_bottom(gray: np.ndarray, line_y: int, lh: int,
                    x1: int, x2: int, pad: int = 4) -> int:
    y_top = max(0, line_y - lh)
    y_bot = min(gray.shape[0], line_y + lh + int(lh * 0.4))
    x1c = max(0, x1); x2c = min(gray.shape[1], x2)
    band = gray[y_top:y_bot, x1c:x2c]
    bw = cv2.adaptiveThreshold(
        cv2.GaussianBlur(band, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    rs = bw.sum(axis=1)
    thr = max(rs) * 0.06 if max(rs) > 0 else 1
    last = 0
    for i in range(len(rs) - 1, -1, -1):
        if rs[i] > thr:
            last = i; break
    return y_top + last + pad


def find_ink_top(gray: np.ndarray, line_y: int, lh: int, x1: int, x2: int) -> int:
    y_top = max(0, line_y - lh); y_bot = min(gray.shape[0], line_y + lh)
    x1c = max(0, x1); x2c = min(gray.shape[1], x2)
    band = gray[y_top:y_bot, x1c:x2c]
    bw = cv2.adaptiveThreshold(
        cv2.GaussianBlur(band, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    rs = bw.sum(axis=1)
    thr = max(rs) * 0.06 if max(rs) > 0 else 1
    for i in range(len(rs)):
        if rs[i] > thr:
            return y_top + i
    return line_y


def snap_to_word_boundaries(gray: np.ndarray, line_y: int, lh: int,
                            lmx: int, rmx: int, x_start: int, x_end: int,
                            dpi: int) -> tuple[int, int]:
    """Expand x_start/x_end to actual word boundaries that overlap [x_start,x_end]."""
    sc = dpi / 150.0
    y1 = max(0, line_y - lh); y2 = min(gray.shape[0], line_y + lh)
    band = gray[y1:y2, lmx:rmx]
    bw = cv2.adaptiveThreshold(
        cv2.GaussianBlur(band, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    mk = int(20 * sc)
    dil = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (mk, 1)))
    c = dil.sum(axis=0).astype(float)
    if c.max() == 0:
        return x_start, x_end
    thr = c.max() * 0.08
    in_w = False; ws = 0; words: list[tuple[int, int]] = []
    for x2c, v in enumerate(c):
        if v > thr and not in_w:
            ws = x2c; in_w = True
        elif v <= thr and in_w:
            if x2c - ws > int(15 * sc):
                raw = bw[:, ws:x2c].sum(axis=0).astype(float)
                ol = next((i for i, v2 in enumerate(raw) if v2 > 0), 0)
                or_ = next((i for i, v2 in enumerate(reversed(raw)) if v2 > 0), 0)
                words.append((ws + ol + lmx, x2c - or_ + lmx))
            in_w = False
    if in_w:
        words.append((ws + lmx, len(c) + lmx))
    if not words:
        return x_start, x_end
    ov = [(a, b) for a, b in words if a < x_end and b > x_start]
    if not ov:
        return x_start, x_end
    return min(a for a, _ in ov), max(b for _, b in ov)


def snap_exponent_gap(layout: LayoutMap, line: int, target_x: int) -> int:
    """Snap target x-px to the nearest detected word gap on `line`."""
    gaps = layout.word_gaps_on_line(line)
    if not gaps:
        return target_x
    return min(gaps, key=lambda g: abs(g.x_center - target_x)).x_center


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6   Drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def wavy_underline(arr: np.ndarray, gray: np.ndarray, line_y: int, lh: int,
                   x1: int, x2: int, color: tuple[int, int, int], thickness: int,
                   dpi: int, next_line_y: int | None = None, pad_px: int = 4) -> None:
    """
    Wavy underline following the actual ink-bottom profile of the line.

    BUG FIX (vs annotate_pdf_final.py): the line band's `y_bot` is clamped to
    `next_line_y - lh - 4` to prevent the wave bleeding into the next line of
    text on tightly-spaced pages.
    """
    x1c = max(0, x1); x2c = min(gray.shape[1], x2)
    if x2c <= x1c:
        return
    y_top = max(0, line_y - lh)
    if next_line_y is not None:
        y_bot = min(gray.shape[0], next_line_y - lh - 4)
    else:
        y_bot = min(gray.shape[0], line_y + lh + int(lh * 0.15))
    if y_bot <= y_top + 4:
        y_bot = min(gray.shape[0], line_y + lh + 2)

    band = gray[y_top:y_bot, x1c:x2c]
    bw = cv2.adaptiveThreshold(
        cv2.GaussianBlur(band, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    bottoms: list[float | None] = []
    for c in range(bw.shape[1]):
        nz = np.nonzero(bw[:, c])[0]
        bottoms.append(float(y_top + nz[-1]) if len(nz) > 0 else None)

    filled: list[float | None] = bottoms[:]
    last: float | None = None
    for i, v in enumerate(filled):
        if v is not None: last = v
        elif last is not None: filled[i] = last
    last = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None: last = filled[i]
        elif last is not None: filled[i] = last

    if all(v is None for v in filled):
        y_flat = line_y + lh + pad_px
        cv2.line(arr, (x1c, y_flat), (x2c, y_flat), _bgr(color), thickness, cv2.LINE_AA)
        return

    arr_f = np.array([v if v is not None else line_y + lh for v in filled], dtype=float)
    win = max(40, int(100 * dpi / 150))
    robust = np.zeros_like(arr_f)
    for i in range(len(arr_f)):
        w0 = max(0, i - win // 2); w1 = min(len(arr_f), i + win // 2)
        vals = [arr_f[j] for j in range(w0, w1) if filled[j] is not None]
        robust[i] = np.percentile(vals, 82) if vals else arr_f[i]

    sigma = max(25.0, 55.0 * dpi / 150)
    smoothed = gaussian_filter1d(robust, sigma=sigma) + pad_px
    for i in range(len(smoothed)):
        if filled[i] is not None:
            smoothed[i] = max(smoothed[i], filled[i] + 1)

    xs = list(range(x1c, x2c, 4))
    ys = [int(smoothed[min(j, len(smoothed) - 1)]) for j in range(0, x2c - x1c, 4)]
    pts = np.array(list(zip(xs, ys)), np.int32).reshape(-1, 1, 2)
    cv2.polylines(arr, [pts], False, _bgr(color), thickness, cv2.LINE_AA)


def organic_ellipse(arr: np.ndarray, cx: int, cy: int, rx: int, ry: int,
                    color: tuple[int, int, int], thickness: int, dpi: int) -> None:
    """Hand-drawn-looking ellipse via parametric + 3 Fourier harmonics + tilt."""
    _seed(cx, cy)
    tilt = math.radians(random.uniform(-8, 8))
    wobble = 0.045 + random.uniform(0, 0.02)
    h1, p1 = wobble, random.uniform(0, 2 * math.pi)
    h2, p2 = wobble * 0.55, random.uniform(0, 2 * math.pi)
    h3, p3 = wobble * 0.3, random.uniform(0, 2 * math.pi)
    start = random.uniform(-0.18, 0.18)
    total = 2 * math.pi + random.uniform(0.05, 0.18)
    N = max(80, int(120 * dpi / 150))
    pts = []
    for i in range(N + 1):
        t = start + total * i / N
        rw = (1 + h1 * math.sin(5 * t + p1)
                + h2 * math.sin(11 * t + p2)
                + h3 * math.sin(17 * t + p3))
        xp = rx * rw * math.cos(t); yp = ry * rw * math.sin(t)
        xr = xp * math.cos(tilt) - yp * math.sin(tilt)
        yr = xp * math.sin(tilt) + yp * math.cos(tilt)
        pts.append((int(cx + xr), int(cy + yr)))
    p = np.array(pts, np.int32).reshape(-1, 1, 2)
    cv2.polylines(arr, [p], False, _bgr(color), thickness, cv2.LINE_AA)


def natural_tick(arr: np.ndarray, cx: int, cy: int, size: int,
                 color: tuple[int, int, int], thickness: int) -> None:
    """Two curved Bezier strokes — short foot + long ascending arm."""
    _seed(cx, cy)

    def bez(p0, cp, p1, n=30):
        out = []
        for i in range(n + 1):
            t = i / n
            out.append((int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cp[0] + t * t * p1[0]),
                       int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cp[1] + t * t * p1[1])))
        return out

    s = size
    p0 = (cx - s // 2, cy - s // 8)
    p1 = (cx - s // 8, cy + s // 2)
    cp = (cx - s // 2 + int(s * 0.15) + random.randint(-2, 2),
          cy + s // 8 + random.randint(-2, 2))
    p2 = p1
    p3 = (cx + s // 2 + random.randint(-3, 3), cy - s // 2 + random.randint(-3, 3))
    cp2 = (cx + int(s * 0.1) + random.randint(-2, 2),
           cy + int(s * 0.2) + random.randint(-2, 2))
    cv2.polylines(arr, [np.array(bez(p0, cp, p1, 20), np.int32).reshape(-1, 1, 2)],
                  False, _bgr(color), thickness, cv2.LINE_AA)
    cv2.polylines(arr, [np.array(bez(p2, cp2, p3, 40), np.int32).reshape(-1, 1, 2)],
                  False, _bgr(color), thickness, cv2.LINE_AA)


def _free_band_centre(free_bands: list[tuple[int, int]],
                      y_lo: int, y_hi: int) -> int | None:
    """Find the centre of a free band overlapping [y_lo, y_hi]. None if no overlap."""
    best = None; best_overlap = 0
    a, b = min(y_lo, y_hi), max(y_lo, y_hi)
    for s, e in free_bands:
        ov = max(0, min(e, b) - max(s, a))
        if ov > best_overlap:
            best_overlap = ov
            best = (s + e) // 2
    return best


def organic_arrow(arr: np.ndarray, sx: int, sy: int, dx: int, dy: int,
                  color: tuple[int, int, int], th: int,
                  free_bands: list[tuple[int, int]], H: int) -> None:
    """
    Quadratic Bezier arrow that bows through whitespace.

    BUG FIX: when free_bands are provided, the control point is snapped to the
    centre of the closest free band between source and dest, so the arrow curves
    through actual blank space instead of cutting across text.
    """
    dist = math.hypot(dx - sx, dy - sy)
    if dist < 5:
        return
    going_right = dx > sx + 50
    going_down = dy > sy + 80

    if going_right:
        bow = min(dist * 0.32, H * 0.055)
        cpx = (sx + dx) // 2
        cpy = min(sy, dy) - int(bow)
        if dy > sy: cpy = sy - int(bow * 0.55)
    elif going_down:
        bow = min(dist * 0.28, H * 0.05)
        cpx = max(sx, dx) + int(bow)
        cpy = (sy + dy) // 2
    else:
        ang = math.atan2(dy - sy, dx - sx); perp = ang + math.pi / 2
        bow = dist * 0.28
        cpx = int((sx + dx) // 2 + bow * math.cos(perp))
        cpy = int((sy + dy) // 2 + bow * math.sin(perp))

    # Snap cpy to a free band when one exists between source and dest
    if free_bands:
        snap = _free_band_centre(free_bands, min(sy, dy), max(sy, dy))
        if snap is not None:
            cpy = int(0.5 * cpy + 0.5 * snap)

    pts = []
    N = 100
    for i in range(N + 1):
        t = i / N
        x = int((1 - t) ** 2 * sx + 2 * (1 - t) * t * cpx + t * t * dx)
        y = int((1 - t) ** 2 * sy + 2 * (1 - t) * t * cpy + t * t * dy)
        pts.append((x, y))
    p = np.array(pts, np.int32).reshape(-1, 1, 2)
    cv2.polylines(arr, [p], False, _bgr(color), th, cv2.LINE_AA)
    ang = math.atan2(pts[-1][1] - pts[-4][1], pts[-1][0] - pts[-4][0])
    alen = max(14, th * 6)
    for da in (0.45, -0.45):
        ax = int(dx - alen * math.cos(ang + da))
        ay = int(dy - alen * math.sin(ang + da))
        cv2.line(arr, (dx, dy), (ax, ay), _bgr(color), th, cv2.LINE_AA)


def curly_brace(arr: np.ndarray, y_top: int, y_bot: int, bx: int, side: str,
                color: tuple[int, int, int], thickness: int, tip_w: int) -> None:
    """Cubic Bezier curly brace ({ or }) of given height at x=bx."""
    Hb = y_bot - y_top
    tip = bx + (tip_w if side == "left" else -tip_w)
    mid = y_top + Hb // 2
    pts = []

    def cub(a, b, c, d, t):
        return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d

    N = 80
    for i in range(N + 1):
        t = i / N
        pts.append((int(cub(bx, tip, tip, tip, t)),
                    int(cub(y_top, y_top, mid - Hb // 8, mid, t))))
    for i in range(N + 1):
        t = i / N
        pts.append((int(cub(tip, tip, tip, bx, t)),
                    int(cub(mid, mid + Hb // 8, y_bot, y_bot, t))))
    cv2.polylines(arr, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                  False, _bgr(color), thickness, cv2.LINE_AA)


def exponent_caret(arr: np.ndarray, cx: int, caret_y: int, s: int,
                   color: tuple[int, int, int], thickness: int) -> None:
    """Caret ^ for missing-word insertion."""
    cv2.line(arr, (cx, caret_y), (cx - s, caret_y + s),
             _bgr(color), thickness, cv2.LINE_AA)
    cv2.line(arr, (cx, caret_y), (cx + s, caret_y + s),
             _bgr(color), thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7   Text layer — handwriting font, ProximityLayout, place_in_gap
# ─────────────────────────────────────────────────────────────────────────────

def _pil_font(sz: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, max(8, sz))


def _measure(text: str, sz: int) -> tuple[int, int]:
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d.textbbox((0, 0), text, font=_pil_font(sz))
    return bb[2] - bb[0], bb[3] - bb[1]


def paste_hw(layer: Image.Image, text: str, x: int, y: int, sz: int,
             color: tuple[int, int, int]) -> tuple[int, int]:
    fnt = _pil_font(sz)
    tw, th = _measure(text, sz)
    pad = max(4, sz // 4)
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad, pad), text, font=fnt, fill=color + (255,))
    layer.paste(img, (int(x), int(y)), img)
    return img.size


class ProximityLayout:
    """
    Non-overlapping text placement in margins.

    BUG FIX: `_slots` is pre-seeded with the OCCUPIED ranges of each margin
    (computed from the inverse of the detected free strips), so newly placed
    remark text never lands on top of existing handwriting in the margin.
    """

    def __init__(self, layout: LayoutMap):
        self.W = layout.W; self.H = layout.H
        self.lmx = layout.lmx; self.rmx = layout.rmx
        self._slots: dict[str, list[tuple[int, int]]] = {
            "left_margin": self._occupied_ranges(layout.left_strips, layout.H),
            "right_margin": self._occupied_ranges(layout.right_strips, layout.H),
        }

    @staticmethod
    def _occupied_ranges(strips: list[FreeStrip], H: int) -> list[tuple[int, int]]:
        """Invert free strips → occupied ranges (so we DON'T place text there)."""
        if not strips:
            return []  # no strips known → margin is treated as fully placeable
        free_sorted = sorted([(s.y_top, s.y_bot) for s in strips])
        occupied: list[tuple[int, int]] = []
        cur = 0
        for y0, y1 in free_sorted:
            if y0 > cur + 4:
                occupied.append((cur, y0))
            cur = max(cur, y1)
        if H - cur > 4:
            occupied.append((cur, H))
        return occupied

    def zone_x(self, zone: str) -> int:
        return 4 if zone == "left_margin" else self.rmx + 6

    def zone_w(self, zone: str) -> int:
        return self.lmx - 8 if zone == "left_margin" else self.W - self.rmx - 10

    def _ok(self, zone: str, y0: int, y1: int, m: int = 6) -> bool:
        for pt, pb in self._slots[zone]:
            if not (y1 + m <= pt or y0 - m >= pb):
                return False
        return True

    def find_y(self, zone: str, pref: int, h: int, m: int = 6) -> int:
        if self._ok(zone, pref, pref + h, m):
            self._slots[zone].append((pref, pref + h))
            self._slots[zone].sort()
            return pref
        step = max(4, h // 6)
        for off in range(step, self.H, step):
            for d in (-1, 1):
                y = pref + d * off
                if y < 2 or y + h > self.H - 2:
                    continue
                if self._ok(zone, y, y + h, m):
                    self._slots[zone].append((y, y + h))
                    self._slots[zone].sort()
                    return y
        # No clean slot — append at end (best effort)
        y = self._slots[zone][-1][1] + m if self._slots[zone] else max(2, pref)
        self._slots[zone].append((y, y + h))
        return y

    def render(self, layer: Image.Image, lines_str: list[str], zone: str,
               preferred_y: int, sz: int,
               color: tuple[int, int, int]) -> tuple[int, int, int]:
        zw = self.zone_w(zone); zx = self.zone_x(zone)
        lh = _measure("A", sz)[1] + max(2, sz // 8)
        cw = max(1, int(zw / max(1, _measure("m", sz)[0])))
        rows: list[str] = []
        for ln in lines_str:
            rows += textwrap.wrap(ln, cw) or [ln]
        total = len(rows) * lh
        y0 = self.find_y(zone, preferred_y - total // 2, total)
        for i, row in enumerate(rows):
            paste_hw(layer, row, zx, y0 + i * lh, sz, color)
        return zx, y0, total


def place_in_gap(layer: Image.Image, text: str, gap: GapInfo, body_l: int,
                 body_r: int, sz: int,
                 color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Wrap text and centre-place inside a body gap.
    BUG FIX: previously rendered as a single un-wrapped line that ran off the page.
    """
    avail_w = max(120, body_r - body_l - 30)
    cw = max(1, int(avail_w / max(1, _measure("m", sz)[0])))
    rows = textwrap.wrap(text, cw) or [text]
    lh = _measure("A", sz)[1] + max(3, sz // 7)
    total = len(rows) * lh
    if total < gap.height - 8:
        y0 = gap.y_top + (gap.height - total) // 2
    else:
        y0 = gap.y_top + 4
    x0 = body_l + 30
    for i, row in enumerate(rows):
        paste_hw(layer, row, x0, y0 + i * lh, sz, color)
    return x0, y0, total


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8   RemarkResolver — JSON spec → ResolvedRemark with absolute coords
# ─────────────────────────────────────────────────────────────────────────────

class RemarkResolver:
    """Resolves auto/auto_line/auto_gap and zone fallback to absolute pixels."""

    def __init__(self, layout: LayoutMap):
        self.layout = layout

    def resolve_all(self, raw_remarks: list[dict]) -> list[ResolvedRemark]:
        out: list[ResolvedRemark] = []
        for raw in raw_remarks:
            try:
                out.append(self._resolve_one(raw))
            except Exception as e:
                print(f"[resolver] dropped remark {raw}: {e}", file=sys.stderr)
        out.sort(key=lambda r: (DRAW_PRIORITY.get(r.type, 9),
                                r.line_idx, r.cy_px))
        return out

    def _resolve_one(self, r: dict) -> ResolvedRemark:
        L = self.layout
        t = r.get("type")
        sev = r.get("severity", "suggestion")
        col = _color_for(sev)
        line = r.get("line", r.get("from_line", r.get("line_start", 1)))
        line_idx = max(0, min(len(L.line_ys) - 1, int(line) - 1)) if L.line_ys else 0

        out = ResolvedRemark(type=t, severity=sev, color=col,
                             line_idx=line_idx, raw=r)

        # Resolve x_start / x_end
        if "x_start" in r:
            out.x_start_px = self._x(r["x_start"], L)
        if "x_end" in r:
            out.x_end_px = self._x(r["x_end"], L)
        # auto_line → expand to full body width
        if r.get("x_start") == "auto_line":
            out.x_start_px = L.lmx + int(L.W * 0.005)
        if r.get("x_end") == "auto_line":
            out.x_end_px = L.rmx - int(L.W * 0.005)

        # Resolve cx / cy_line for score_circle
        if t == "score_circle":
            if r.get("cx") in (None, "auto"):
                cx_pct, cy_line = L.best_score_position()
                out.cx_px = int(L.W * cx_pct / 100)
                out.cy_px = L.line_y(cy_line)
            else:
                out.cx_px = int(L.W * float(r["cx"]) / 100)
                cy_line_v = r.get("cy_line")
                if cy_line_v in (None, "auto"):
                    _, cy_line = L.best_score_position()
                    out.cy_px = L.line_y(cy_line)
                else:
                    out.cy_px = L.line_y(int(cy_line_v))
            out.text = str(r.get("text", ""))
            return out

        # Resolve cy_px from line for everything else
        out.cy_px = L.line_y(int(line))

        # zone for tick / curly_brace
        out.zone = r.get("zone", "left_margin")

        # remark_zone with gap fallback
        rz = r.get("remark_zone", "right_margin")
        if rz == "gap":
            gap = L.best_gap_for_line(int(r.get("line_end", line)))
            if gap and gap.height >= max(26, L.H // 65) + 10:
                out.gap = gap
                out.remark_zone = "gap"
            else:
                out.remark_zone = "right_margin"  # fallback
                print(f"[resolver] line {line}: gap unavailable → right_margin",
                      file=sys.stderr)
        else:
            out.remark_zone = rz

        out.remark = r.get("remark", "")
        out.text = r.get("text", "")

        # exponent_insert: snap to word gap
        if t == "exponent_insert":
            xp = r.get("x_pct")
            if xp in (None, "auto_gap"):
                wgs = L.word_gaps_on_line(int(line))
                if wgs:
                    mid = (L.lmx + L.rmx) // 2
                    out.cx_px = min(wgs, key=lambda g: abs(g.x_center - mid)).x_center
                else:
                    out.cx_px = (L.lmx + L.rmx) // 2
            else:
                target = int(L.W * float(xp) / 100)
                out.cx_px = snap_exponent_gap(L, int(line), target)

        # arrow_remark from_x
        if t == "arrow_remark":
            fx = r.get("from_x", 50)
            out.cx_px = int(L.W * float(fx) / 100)

        # curly_brace line_start / line_end
        if t == "curly_brace":
            ls = int(r.get("line_start", 1))
            le = int(r.get("line_end", ls + 1))
            out.line_start_idx = max(0, ls - 1)
            out.line_end_idx = max(0, le - 1)

        # underline_remark line_end
        if t == "underline_remark" and "line_end" in r:
            out.line_end_idx = max(0, int(r["line_end"]) - 1)

        return out

    @staticmethod
    def _x(val: Any, L: LayoutMap) -> int:
        if isinstance(val, str):
            if val == "auto_line":
                return 0
            try:
                return int(L.W * float(val) / 100)
            except ValueError:
                return 0
        return int(L.W * float(val) / 100)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9   Page Annotator — dispatch loop
# ─────────────────────────────────────────────────────────────────────────────

def _free_bands_for_arrows(layout: LayoutMap) -> list[tuple[int, int]]:
    """Combined free-y bands across the whole width for arrow control-point routing."""
    return [(s.y_top, s.y_bot) for s in
            layout.right_strips + layout.left_strips] + \
           [(g.y_top, g.y_bot) for g in layout.gaps]


def draw_page(pil_img: Image.Image, resolved: list[ResolvedRemark],
              layout: LayoutMap) -> Image.Image:
    L = layout
    W, H = L.W, L.H
    sc = L.scale
    LH = L.line_half_height
    LN_TH = max(2, int(2 * sc))
    TICK_SZ = max(10, int(H // 40)); TICK_TH = max(2, int(3 * sc))
    TIP_W = max(10, int(18 * sc)); ARR_TH = max(2, int(2.5 * sc))
    S_REM = max(26, int(H // 65))
    S_MAR = max(18, int(H // 100))
    S_INS = max(20, int(H // 85))
    S_SCR = max(36, int(H // 44))

    arr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    text_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layout_helper = ProximityLayout(L)
    free_bands = _free_bands_for_arrows(L)
    gray = np.array(pil_img.convert("L"))

    for r in resolved:
        c = r.color
        t = r.type

        if t == "score_circle":
            txt = r.text
            tw, th = _measure(txt, S_SCR)
            rad = max(tw, th) // 2 + max(16, int(20 * sc))
            organic_ellipse(arr, r.cx_px, r.cy_px, rad, rad, c,
                            max(2, int(3 * sc)), L.dpi)
            paste_hw(text_l, txt, r.cx_px - tw // 2 - 2, r.cy_px - th // 2 - 2,
                     S_SCR, c)

        elif t == "tick":
            cx = int(W * 0.07) if r.zone == "left_margin" else int(W * 0.93)
            natural_tick(arr, cx, r.cy_px, TICK_SZ, c, TICK_TH)

        elif t == "circle_remark":
            cy = r.cy_px
            xs, xe = snap_to_word_boundaries(
                gray, cy, LH, L.lmx, L.rmx, r.x_start_px, r.x_end_px, L.dpi)
            cx = (xs + xe) // 2
            itop = find_ink_top(gray, cy, LH, xs, xe)
            ibot = find_ink_bottom(gray, cy, LH, xs, xe, pad=2)
            rx = (xe - xs) // 2 + max(6, int(8 * sc))
            ry = (ibot - itop) // 2 + max(4, int(5 * sc))
            oval_cy = (itop + ibot) // 2
            organic_ellipse(arr, cx, oval_cy, rx, ry, c, LN_TH, L.dpi)

            if r.remark:
                if r.remark_zone == "gap" and r.gap is not None:
                    tx, ty, _ = place_in_gap(text_l, r.remark, r.gap,
                                             L.lmx, L.rmx, S_REM, c)
                    organic_arrow(arr, cx, oval_cy + ry,
                                  tx, ty + S_REM // 2, c, ARR_TH, free_bands, H)
                else:
                    _, y0, th2 = layout_helper.render(
                        text_l, [r.remark], r.remark_zone, oval_cy, S_MAR, c)
                    ax = cx + rx if "right" in r.remark_zone else cx - rx
                    bx2 = (L.rmx + 4) if "right" in r.remark_zone else (L.lmx - 4)
                    organic_arrow(arr, ax, oval_cy, bx2, y0 + th2 // 2,
                                  c, ARR_TH, free_bands, H)

        elif t == "underline_remark":
            line_y = r.cy_px
            x1, x2 = r.x_start_px, r.x_end_px
            next_y = (L.line_ys[r.line_idx + 1]
                      if r.line_idx + 1 < len(L.line_ys) else None)
            wavy_underline(arr, gray, line_y, LH, x1, x2, c, LN_TH, L.dpi,
                           next_line_y=next_y)
            anchor_x = x2
            anchor_y = find_ink_bottom(gray, line_y, LH, x1, x2)

            if r.line_end_idx and r.line_end_idx > r.line_idx:
                # multi-line underline (use same x by default)
                line_y2 = L.line_ys[r.line_end_idx]
                next_y2 = (L.line_ys[r.line_end_idx + 1]
                           if r.line_end_idx + 1 < len(L.line_ys) else None)
                x1b = int(L.W * float(r.raw.get("x_start2", r.raw.get("x_start", 14))) / 100)
                x2b = int(L.W * float(r.raw.get("x_end2", r.raw.get("x_end", 75))) / 100)
                wavy_underline(arr, gray, line_y2, LH, x1b, x2b, c, LN_TH, L.dpi,
                               next_line_y=next_y2)
                anchor_x = x2b
                anchor_y = find_ink_bottom(gray, line_y2, LH, x1b, x2b)

            if r.remark:
                if r.remark_zone == "gap" and r.gap is not None:
                    tx, ty, _ = place_in_gap(text_l, r.remark, r.gap,
                                             L.lmx, L.rmx, S_REM, c)
                    organic_arrow(arr, anchor_x, anchor_y, tx, ty + S_REM // 2,
                                  c, ARR_TH, free_bands, H)
                else:
                    _, y0, th2 = layout_helper.render(
                        text_l, [r.remark], r.remark_zone, anchor_y, S_MAR, c)
                    bx2 = (L.rmx + 4) if "right" in r.remark_zone else (L.lmx - 4)
                    organic_arrow(arr, anchor_x, anchor_y, bx2, y0 + th2 // 2,
                                  c, ARR_TH, free_bands, H)

        elif t == "exponent_insert":
            cx = r.cx_px; line_y = r.cy_px
            itop = find_ink_top(gray, line_y, LH, cx - 20, cx + 20)
            s = max(8, int(H // 78)); caret_y = itop - s - 6
            exponent_caret(arr, cx, caret_y, s, c, LN_TH)
            if r.text:
                tw, th = _measure(r.text, S_INS)
                paste_hw(text_l, r.text, cx - tw // 2, caret_y - th - 4, S_INS, c)

        elif t == "arrow_remark":
            from_y = r.cy_px; from_x = r.cx_px
            if r.remark:
                _, y0, th2 = layout_helper.render(
                    text_l, [r.remark], r.remark_zone, from_y, S_MAR, c)
                to_x = (L.rmx + 4) if "right" in r.remark_zone else (L.lmx - 4)
                organic_arrow(arr, from_x, from_y, to_x, y0 + th2 // 2,
                              c, ARR_TH, free_bands, H)

        elif t == "curly_brace":
            zone = r.zone
            y_top = L.line_y(r.line_start_idx + 1) - LH
            y_bot = L.line_y(r.line_end_idx + 1) + LH
            bx = L.lmx - max(2, int(3 * sc)) if zone == "left_margin" else \
                 L.rmx + max(2, int(3 * sc))
            side = "left" if zone == "left_margin" else "right"
            curly_brace(arr, y_top, y_bot, bx, side, c, LN_TH, TIP_W)
            if r.remark:
                layout_helper.render(text_l, [r.remark], zone,
                                     (y_top + y_bot) // 2, S_MAR, c)

        elif t == "text":
            zone = r.zone if r.zone in ("left_margin", "right_margin") \
                else "right_margin"
            ay = r.cy_px if r.cy_px else int(H * 0.5)
            layout_helper.render(text_l, r.text.split("\n"), zone, ay, S_MAR, c)

    base = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    return Image.alpha_composite(base, text_l).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  Pipeline + scaffold_dense (auto-fill placement stress test)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-canned remark text, sized to specific zones.
_DENSE_GAP_REMARKS = [
    "Logical inconsistency. This claim contradicts the opening argument and weakens the overall thesis.",
    "Critical factual error. Verify against authoritative sources before final submission.",
    "Argument lacks depth. Add socio-economic dimensions and supporting data.",
    "Conclusion is generic. Restate the central thesis with concrete recommendations.",
    "Fundamental misreading. Revise from standard references before reattempting this topic.",
    "Strong analytical framing here, but the supporting evidence is insufficient for a top-tier answer.",
    "This paragraph drifts from the question demand. Refocus on the specific evaluation criteria.",
]

_DENSE_MARGIN_REMARKS = [
    "Cite the specific article or commission report.",
    "Lacks comparative perspective. Add at least one counter-argument.",
    "Use precise UPSC terminology — 'sovereignty', not 'authority'.",
    "Add data, year, or quantitative anchor.",
    "Argument incomplete. Address the second half of the question.",
    "Transition between paragraphs is abrupt; use linking phrases.",
    "Strong point — sustain this analytical depth throughout.",
]


def _pick_remark(pool: list[str], idx: int) -> str:
    return pool[idx % len(pool)]


def scaffold_dense(layout: LayoutMap) -> list[dict]:
    """
    Auto-generate up to ~8 remarks per page covering ALL detected free spaces.
    Used to stress-test placement logic.
    """
    remarks: list[dict] = []
    if not layout.line_ys:
        return remarks

    # 1. score_circle (auto position)
    cx_pct, cy_line = layout.best_score_position()
    remarks.append({
        "type": "score_circle",
        "cx": cx_pct, "cy_line": cy_line,
        "text": "7/10", "severity": "critical",
    })

    # 2. tick on first detected line, left margin
    remarks.append({
        "type": "tick", "line": 1, "zone": "left_margin",
        "severity": "critical",
    })

    sev_cycle = ["critical", "suggestion"]

    # 3. body gap remarks (one per gap, max 3)
    n = len(layout.line_ys)
    for i, gap in enumerate(layout.gaps[:3]):
        line = max(1, gap.after_line)
        # Anchor on the line ABOVE the gap so the arrow points down into it.
        # Alternate between underline_remark and circle_remark
        kind = "underline_remark" if i % 2 == 0 else "circle_remark"
        body = {
            "type": kind,
            "line": min(line, n),
            "x_start": 16, "x_end": 60,
            "remark": _pick_remark(_DENSE_GAP_REMARKS, i),
            "remark_zone": "gap",
            "severity": sev_cycle[i % 2],
        }
        remarks.append(body)

    # 4. right-margin strips (one annotation per strip, max 3)
    used_lines: set[int] = {r.get("line", -1) for r in remarks}
    for i, strip in enumerate(layout.right_strips[:3]):
        # Find the line whose y is closest to the strip centre
        cy = (strip.y_top + strip.y_bot) // 2
        if not layout.line_ys:
            continue
        line = min(range(len(layout.line_ys)),
                   key=lambda j: abs(layout.line_ys[j] - cy)) + 1
        if line in used_lines:
            continue
        used_lines.add(line)
        remarks.append({
            "type": "underline_remark",
            "line": line,
            "x_start": 14, "x_end": 65,
            "remark": _pick_remark(_DENSE_MARGIN_REMARKS, i),
            "remark_zone": "right_margin",
            "severity": sev_cycle[i % 2],
        })

    # 5. left-margin strips (one annotation per strip, max 2 — leave room for brace)
    for i, strip in enumerate(layout.left_strips[:2]):
        cy = (strip.y_top + strip.y_bot) // 2
        if not layout.line_ys:
            continue
        line = min(range(len(layout.line_ys)),
                   key=lambda j: abs(layout.line_ys[j] - cy)) + 1
        if line in used_lines:
            continue
        used_lines.add(line)
        remarks.append({
            "type": "underline_remark",
            "line": line,
            "x_start": 14, "x_end": 65,
            "remark": _pick_remark(_DENSE_MARGIN_REMARKS, i + 3),
            "remark_zone": "left_margin",
            "severity": sev_cycle[i % 2],
        })

    # 6. exponent_insert on first line ≥ 4 with a word gap
    for line in range(4, n + 1):
        if layout.word_gaps_on_line(line):
            remarks.append({
                "type": "exponent_insert",
                "line": line, "x_pct": "auto_gap",
                "text": "(cite?)", "severity": "suggestion",
            })
            break

    # 7. curly_brace if there are ≥ 4 lines that fit a margin strip y-range
    if layout.left_strips and n >= 4:
        big = max(layout.left_strips, key=lambda s: s.height)
        if big.height >= int(layout.H * 0.10):
            ls_line = next((j + 1 for j, y in enumerate(layout.line_ys)
                            if y >= big.y_top), 1)
            le_line = next((j + 1 for j, y in enumerate(reversed(layout.line_ys))
                            if y <= big.y_bot), n)
            le_line = n - le_line + 1 if le_line else n
            if le_line - ls_line >= 3:
                remarks.append({
                    "type": "curly_brace",
                    "line_start": ls_line, "line_end": le_line,
                    "zone": "left_margin",
                    "remark": "Whole para off-target",
                    "severity": "critical",
                })

    return remarks


def process_pdf(pdf_in: str, remarks_path: str | None, pdf_out: str,
                dpi: int = 250, *, dense: bool = False,
                analyze_only: bool = False) -> dict | None:
    print(f"[+] Loading {pdf_in} at {dpi} DPI…", file=sys.stderr)
    pages = load_pdf_pages_pil(pdf_in, dpi=dpi)
    print(f"[+] {len(pages)} page(s) {pages[0].size[0]}×{pages[0].size[1]}px",
          file=sys.stderr)

    layouts: list[LayoutMap] = []
    for i, img in enumerate(pages, 1):
        L = build_layout_map(img, i, dpi)
        layouts.append(L)
        print(f"[+] {L.summary()}", file=sys.stderr)

    if analyze_only:
        return {
            "pages": [
                {
                    "page": L.page,
                    "lines": len(L.line_ys),
                    "gaps": [(g.after_line, g.height) for g in L.gaps],
                    "right_strips": [(s.y_top, s.y_bot, s.height)
                                     for s in L.right_strips],
                    "left_strips": [(s.y_top, s.y_bot, s.height)
                                    for s in L.left_strips],
                    "word_gap_count": len(L.word_gaps),
                    "free_rect_count": len(L.free_rects),
                }
                for L in layouts
            ]
        }

    if dense:
        spec = {"pages": [{"page": L.page, "remarks": scaffold_dense(L)}
                          for L in layouts]}
        return spec

    # Real annotation flow
    if not remarks_path:
        raise ValueError("remarks_path is required unless --scaffold-dense or --analyze-only")
    with open(remarks_path) as f:
        spec = json.load(f)
    page_map = {p["page"]: p.get("remarks", []) for p in spec.get("pages", [])}

    annotated: list[Image.Image] = []
    for img, L in zip(pages, layouts):
        raw = page_map.get(L.page, [])
        resolved = RemarkResolver(L).resolve_all(raw)
        print(f"[+] Page {L.page}: {len(resolved)} remarks resolved",
              file=sys.stderr)
        annotated.append(draw_page(img, resolved, L))

    print("[+] Writing PDF…", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(annotated):
            p = os.path.join(tmp, f"p{i:03d}.png")
            img.save(p, "PNG")
            paths.append(p)
        with open(pdf_out, "wb") as f:
            f.write(img2pdf.convert(paths))
    print(f"[+] → {pdf_out}", file=sys.stderr)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(prog="teacher_annotate")
    p.add_argument("pdf", help="input PDF path")
    p.add_argument("remarks", nargs="?", help="remarks JSON (omit with --analyze-only or --scaffold-dense)")
    p.add_argument("output", nargs="?", help="output PDF path")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--analyze-only", action="store_true",
                   help="print layout JSON, no annotation")
    p.add_argument("--scaffold-dense", action="store_true",
                   help="emit auto-filled remarks JSON to stdout, no annotation")
    args = p.parse_args()

    if args.analyze_only:
        out = process_pdf(args.pdf, None, "/dev/null",
                          dpi=args.dpi, analyze_only=True)
        print(json.dumps(out, indent=2))
        return

    if args.scaffold_dense:
        out = process_pdf(args.pdf, None, "/dev/null",
                          dpi=args.dpi, dense=True)
        print(json.dumps(out, indent=2))
        return

    if not args.remarks or not args.output:
        p.error("remarks JSON and output PDF are required for annotation mode")
    process_pdf(args.pdf, args.remarks, args.output, dpi=args.dpi)


if __name__ == "__main__":
    _cli()
