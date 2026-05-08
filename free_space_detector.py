"""
freespace_detector.py  —  Answer-sheet free space detector
============================================================
Replaces Script 2's rigid 25-pt cell grid with proper computer vision.

Three techniques, each targeting a different annotation placement need:

  TECHNIQUE 1 — Morphological dilation + inversion
    → finds margin free bands and inter-line gaps
    → feeds ProximityLayout and find_gap_near in Script 1

  TECHNIQUE 2 — Connected component projection
    → finds intra-line word gaps (exact x-positions)
    → feeds snap_exponent_gap, snap_circle_words in Script 1

  TECHNIQUE 3 — Adaptive threshold + contour bounding boxes
    → finds large 2D free rectangles anywhere on the page
    → feeds score_circle placement and gap text blocks in Script 1

Output: a FreeSpaceMap per page — drop-in replacement for PageCellGrid.
The map exposes coordinates as both PDF-point and percent (0-100),
directly usable in remarks.json for Script 1.

Usage:
    from freespace_detector import analyze_pdf_freespace, to_remarks_coords
    maps = analyze_pdf_freespace(pdf_bytes, dpi=250)
    for page_map in maps:
        print(page_map.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import fitz  # PyMuPDF


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES  (all coords in both pixel space and percent 0-100)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FreeStrip:
    """A horizontal free band — used for margin and inter-line gap placement."""
    kind: str            # "left_margin" | "right_margin" | "inter_line_gap"
    y_top_px: int
    y_bot_px: int
    x_left_px: int
    x_right_px: int
    # percent equivalents (0-100 scale, as used in remarks.json)
    y_top_pct: float
    y_bot_pct: float
    x_left_pct: float
    x_right_pct: float
    height_px: int
    width_px: int
    # which Script-1 line numbers this gap falls between (1-indexed)
    after_line: int = 0   # 0 = before first line
    before_line: int = 0  # 0 = after last line

    def centre_y_pct(self) -> float:
        return (self.y_top_pct + self.y_bot_pct) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WordGap:
    """
    A horizontal gap between two words on a line.
    Used for exponent_insert and split-point detection.
    """
    line: int            # 1-indexed Script-1 line number
    x_gap_px: int        # pixel x of the gap centre
    x_gap_pct: float     # percent equivalent
    word_left_end_px: int
    word_right_start_px: int
    gap_width_px: int


@dataclass
class FreeRect:
    """
    A large 2-D rectangle of free space, detected by contour analysis.
    Used for placing score circles, multi-line remark blocks, etc.
    """
    x1_px: int; y1_px: int
    x2_px: int; y2_px: int
    x1_pct: float; y1_pct: float
    x2_pct: float; y2_pct: float
    area_px: int
    centre_x_pct: float
    centre_y_pct: float
    # nearest Script-1 line
    nearest_line: int = 0


@dataclass
class FreeSpaceMap:
    """Complete free-space analysis for one PDF page."""
    page: int
    W: int              # pixel width at analysis DPI
    H: int              # pixel height at analysis DPI
    dpi: int
    page_w_pts: float   # PDF point dimensions
    page_h_pts: float

    # ── Technique 1 results ──
    left_margin_strips: list[FreeStrip] = field(default_factory=list)
    right_margin_strips: list[FreeStrip] = field(default_factory=list)
    inter_line_gaps: list[FreeStrip] = field(default_factory=list)

    # ── Technique 2 results ──
    word_gaps: list[WordGap] = field(default_factory=list)

    # ── Technique 3 results ──
    free_rects: list[FreeRect] = field(default_factory=list)

    # Detected line y-positions (pixels, 1-indexed externally)
    line_ys: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Page {self.page}: "
            f"{len(self.line_ys)} lines | "
            f"{len(self.left_margin_strips)} L-strips | "
            f"{len(self.right_margin_strips)} R-strips | "
            f"{len(self.inter_line_gaps)} gaps | "
            f"{len(self.word_gaps)} word-gaps | "
            f"{len(self.free_rects)} free-rects"
        )

    def best_gap_for_line(self, line: int, prefer: str = "below",
                          max_dist: int = 5) -> FreeStrip | None:
        """Find the closest inter-line gap to a given 1-indexed line number."""
        below = [g for g in self.inter_line_gaps
                 if line <= g.after_line <= line + max_dist]
        above = [g for g in self.inter_line_gaps
                 if line - max_dist <= g.after_line < line]
        if prefer == "below" and below:
            return min(below, key=lambda g: g.after_line)
        if above:
            return max(above, key=lambda g: g.after_line)
        if below:
            return min(below, key=lambda g: g.after_line)
        return None

    def word_gaps_on_line(self, line: int) -> list[WordGap]:
        return sorted([g for g in self.word_gaps if g.line == line],
                      key=lambda g: g.x_gap_px)

    def largest_free_rects(self, n: int = 5) -> list[FreeRect]:
        return sorted(self.free_rects, key=lambda r: r.area_px, reverse=True)[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(pix: fitz.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF pixmap to a uint8 grayscale numpy array."""
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    elif pix.n == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Binary mask: 255 = ink present, 0 = blank.
    Uses adaptive threshold so faint ruled lines count as ink.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        31, 10
    )
    return bw


def _detect_lines(gray: np.ndarray, dpi: int) -> list[int]:
    """
    Detect text baseline y-positions (pixels).
    Uses connected component centroids — same algorithm as Script 1 detect_lines.
    Returns sorted list of pixel y-values (0-indexed).
    """
    bw = _ink_mask(gray)
    sc = dpi / 150.0
    n, _, stats, centroids = cv2.connectedComponentsWithStats(bw)

    char_ys = sorted(
        centroids[i][1] for i in range(1, n)
        if (int(12 * sc) < stats[i][3] < int(90 * sc) and
            int(5 * sc) < stats[i][2] < int(150 * sc) and
            int(40 * sc**2) < stats[i][4] < int(8000 * sc**2))
    )
    if not char_ys:
        return []

    cg = int(22 * sc)
    mt = int(35 * sc)
    raw, bucket = [], [char_ys[0]]
    for y in char_ys[1:]:
        if y - bucket[-1] < cg:
            bucket.append(y)
        else:
            if len(bucket) >= 3:
                raw.append(int(np.median(bucket)))
            bucket = [y]
    if len(bucket) >= 3:
        raw.append(int(np.median(bucket)))

    merged = [raw[0]]
    for y in raw[1:]:
        if y - merged[-1] > mt:
            merged.append(y)
    return merged


def _page_zones(W: int, dpi: int) -> tuple[int, int]:
    """Left margin boundary and right margin start (pixels)."""
    LMX = int(W * 0.135)
    RMX = int(W * 0.875)
    return LMX, RMX


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNIQUE 1 — Morphological dilation → margin strips & inter-line gaps
# ─────────────────────────────────────────────────────────────────────────────

def _technique1_strips(
    gray: np.ndarray,
    line_ys: list[int],
    dpi: int,
    W: int, H: int,
) -> tuple[list[FreeStrip], list[FreeStrip], list[FreeStrip]]:
    """
    Algorithm:
      1. Get ink mask
      2. Dilate horizontally (merge nearby ink into word blobs)
      3. Dilate vertically (merge ink into line blobs)
      4. Per-row: check occupancy in left margin zone, right margin zone,
         and body zone separately
      5. Find contiguous free runs (min height threshold) in each zone
      6. Tag inter-line gaps with which line numbers they fall between

    Returns: (left_strips, right_strips, gap_strips)
    """
    bw = _ink_mask(gray)
    sc = dpi / 150.0
    LMX, RMX = _page_zones(W, dpi)
    LH = max(12, H // 55)

    # Horizontal dilation — merges words into solid line blobs
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(30 * sc), 1))
    # Vertical dilation — small to avoid merging adjacent lines
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(6 * sc)))

    dil_h = cv2.dilate(bw, h_kern)
    dil_hv = cv2.dilate(dil_h, v_kern)

    def row_occupied(zone_arr: np.ndarray, thr: int = 50) -> np.ndarray:
        return zone_arr.sum(axis=1) > thr

    lm_occ = row_occupied(dil_hv[:, :LMX], thr=80)
    rm_occ = row_occupied(dil_hv[:, RMX:], thr=80)
    body_occ = row_occupied(dil_hv[:, LMX:RMX], thr=200)

    def find_free_runs(occ: np.ndarray, min_h: int = 30) -> list[tuple[int, int]]:
        runs = []
        in_free, start = False, 0
        for y, is_occ in enumerate(occ):
            if not is_occ and not in_free:
                start = y
                in_free = True
            elif is_occ and in_free:
                if y - start >= min_h:
                    runs.append((start, y))
                in_free = False
        if in_free and H - start >= min_h:
            runs.append((start, H))
        return runs

    min_h = max(30, int(35 * sc))

    def runs_to_strips(runs, kind, x_left, x_right) -> list[FreeStrip]:
        strips = []
        for y0, y1 in runs:
            strips.append(FreeStrip(
                kind=kind,
                y_top_px=y0, y_bot_px=y1,
                x_left_px=x_left, x_right_px=x_right,
                y_top_pct=round(y0 / H * 100, 2),
                y_bot_pct=round(y1 / H * 100, 2),
                x_left_pct=round(x_left / W * 100, 2),
                x_right_pct=round(x_right / W * 100, 2),
                height_px=y1 - y0,
                width_px=x_right - x_left,
            ))
        return strips

    left_strips = runs_to_strips(
        find_free_runs(lm_occ, min_h), "left_margin", 4, LMX - 4)
    right_strips = runs_to_strips(
        find_free_runs(rm_occ, min_h), "right_margin", RMX + 4, W - 4)

    # Inter-line gaps: body zone, tagged with line numbers
    gap_runs = find_free_runs(body_occ, min_h=max(35, int(40 * sc)))
    gap_strips = []
    for y0, y1 in gap_runs:
        # Find which line numbers bracket this gap
        after = sum(1 for ly in line_ys if ly + LH < y0)
        before = sum(1 for ly in line_ys if ly - LH > y1)
        s = FreeStrip(
            kind="inter_line_gap",
            y_top_px=y0, y_bot_px=y1,
            x_left_px=LMX, x_right_px=RMX,
            y_top_pct=round(y0 / H * 100, 2),
            y_bot_pct=round(y1 / H * 100, 2),
            x_left_pct=round(LMX / W * 100, 2),
            x_right_pct=round(RMX / W * 100, 2),
            height_px=y1 - y0,
            width_px=RMX - LMX,
            after_line=after,
            before_line=len(line_ys) - before,
        )
        gap_strips.append(s)

    return left_strips, right_strips, gap_strips


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNIQUE 2 — Connected component projection → word gaps per line
# ─────────────────────────────────────────────────────────────────────────────

def _technique2_word_gaps(
    gray: np.ndarray,
    line_ys: list[int],
    dpi: int,
    W: int, H: int,
) -> list[WordGap]:
    """
    Algorithm (per line):
      1. Extract a horizontal band ±LH around the line centroid
      2. Threshold to get ink pixels
      3. Compute column-wise ink projection profile
      4. Apply horizontal dilation to merge within-word ink blobs
      5. Find transitions (ink→gap→ink) in the projection
      6. Each gap centre becomes a WordGap with exact pixel & percent coords

    This gives sub-pixel-accurate inter-word gap positions for:
      - exponent_insert (needs exact x where caret goes)
      - circle_remark snapping (needs word boundaries)
    """
    sc = dpi / 150.0
    LH = max(12, H // 55)
    LMX, RMX = _page_zones(W, dpi)
    min_gap_px = max(8, int(10 * sc))
    min_word_px = max(12, int(15 * sc))

    # Morphological kernel: merge ink within a word (~20px at 150dpi)
    merge_kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(20 * sc), 1))

    gaps: list[WordGap] = []

    for line_idx, line_y in enumerate(line_ys):
        line_num = line_idx + 1
        y_top = max(0, line_y - LH)
        y_bot = min(H, line_y + LH)
        band_gray = gray[y_top:y_bot, LMX:RMX]

        bw = cv2.adaptiveThreshold(
            cv2.GaussianBlur(band_gray, (3, 3), 0), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

        # Dilate horizontally to form word blobs
        dil = cv2.dilate(bw, merge_kern)

        # Column projection: sum of ink pixels in each column
        proj = dil.sum(axis=0).astype(np.float32)
        if proj.max() == 0:
            continue

        thr = proj.max() * 0.05   # 5% of peak = ink present in this column

        # Find word extents (runs where projection > threshold)
        in_word = False
        word_start = 0
        words: list[tuple[int, int]] = []  # (x_start, x_end) in band coords

        for x, v in enumerate(proj):
            if v > thr and not in_word:
                word_start = x
                in_word = True
            elif v <= thr and in_word:
                if x - word_start >= min_word_px:
                    words.append((word_start, x))
                in_word = False
        if in_word and len(proj) - word_start >= min_word_px:
            words.append((word_start, len(proj)))

        # Each consecutive pair of words → one gap
        for i in range(len(words) - 1):
            gap_start = words[i][1]    # end of left word
            gap_end = words[i + 1][0]  # start of right word
            gap_w = gap_end - gap_start
            if gap_w < min_gap_px:
                continue

            # Refine: find actual ink edges using raw (non-dilated) projection
            raw_proj = bw.sum(axis=0).astype(np.float32)
            raw_thr = raw_proj.max() * 0.03 if raw_proj.max() > 0 else 1

            # Rightmost ink column of left word
            left_ink_end = gap_start
            for x in range(min(gap_start + int(25*sc), len(raw_proj)) - 1,
                           max(0, gap_start - int(5*sc)), -1):
                if raw_proj[x] > raw_thr:
                    left_ink_end = x
                    break

            # Leftmost ink column of right word
            right_ink_start = gap_end
            for x in range(max(0, gap_end - int(25*sc)),
                           min(len(raw_proj), gap_end + int(5*sc))):
                if raw_proj[x] > raw_thr:
                    right_ink_start = x
                    break

            gap_centre_band = (left_ink_end + right_ink_start) // 2
            gap_centre_px = LMX + gap_centre_band

            gaps.append(WordGap(
                line=line_num,
                x_gap_px=gap_centre_px,
                x_gap_pct=round(gap_centre_px / W * 100, 2),
                word_left_end_px=LMX + left_ink_end,
                word_right_start_px=LMX + right_ink_start,
                gap_width_px=right_ink_start - left_ink_end,
            ))

    return gaps


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNIQUE 3 — Canny edge detection + morphology → free rectangle map
# ─────────────────────────────────────────────────────────────────────────────

def _technique3_free_rects(
    gray: np.ndarray,
    line_ys: list[int],
    dpi: int,
    W: int, H: int,
    min_area_px: int = 2000,
) -> list[FreeRect]:
    """
    Algorithm:
      1. Canny edge detection — finds all ink/object boundaries
      2. Dilate edges heavily — expands obstacle regions, closes gaps
      3. Invert → free space mask (free = 255, occupied = 0)
      4. Morphological close → fills tiny free pockets (not truly free)
      5. Find contours of remaining free regions
      6. Fit minimum bounding rect to each contour
      7. Filter by area, keep only axis-aligned rectangles

    This is the most direct application of the general free-space detection
    literature to the PDF annotation domain.
    """
    sc = dpi / 150.0
    LMX, RMX = _page_zones(W, dpi)

    # ── Step 1: Canny edge detection ─────────────────────────────────────────
    # Blur first to suppress noise from paper grain
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=90)

    # ── Step 2: Dilate edges — obstacles expand to block space around ink ─────
    # Kernel size ∝ dpi: at 250dpi, ~15px dilation ≈ 1.5mm clearance
    dil_px = max(8, int(12 * sc))
    dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
    dilated_edges = cv2.dilate(edges, dil_kern)

    # Also incorporate full ink mask (catches filled regions Canny misses)
    ink = _ink_mask(gray)
    ink_expanded = cv2.dilate(ink, dil_kern)

    occupied = cv2.bitwise_or(dilated_edges, ink_expanded)

    # ── Step 3: Invert → free space mask ──────────────────────────────────────
    free_mask = cv2.bitwise_not(occupied)

    # ── Step 4: Morphological close → remove tiny isolated free pockets ───────
    close_px = max(15, int(20 * sc))
    close_kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_px, close_px))
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, close_kern)

    # ── Step 5 & 6: Contours → bounding rectangles ────────────────────────────
    contours, _ = cv2.findContours(
        free_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: list[FreeRect] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area_px:
            continue

        # Use actual contour area vs bounding rect area — filter out thin slivers
        contour_area = cv2.contourArea(cnt)
        fill_ratio = contour_area / max(area, 1)
        if fill_ratio < 0.35:   # less than 35% filled = sliver, skip
            continue

        x2, y2 = x + w, y + h

        # Find nearest line number
        nearest_line = 1
        if line_ys:
            cy = (y + y2) // 2
            nearest_line = min(range(len(line_ys)),
                               key=lambda i: abs(line_ys[i] - cy)) + 1

        rects.append(FreeRect(
            x1_px=x, y1_px=y, x2_px=x2, y2_px=y2,
            x1_pct=round(x / W * 100, 2),
            y1_pct=round(y / H * 100, 2),
            x2_pct=round(x2 / W * 100, 2),
            y2_pct=round(y2 / H * 100, 2),
            area_px=area,
            centre_x_pct=round((x + x2) / 2 / W * 100, 2),
            centre_y_pct=round((y + y2) / 2 / H * 100, 2),
            nearest_line=nearest_line,
        ))

    return sorted(rects, key=lambda r: r.area_px, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pdf_freespace(
    pdf_bytes: bytes,
    dpi: int = 250,
    min_strip_height_px: int = 30,
    min_rect_area_px: int = 2000,
) -> list[FreeSpaceMap]:
    """
    Run all three free-space detection techniques on every page.

    Returns a list of FreeSpaceMap (one per page) whose coordinate fields
    are in both pixel space and percent (0-100), compatible with Script 1's
    remarks.json format.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    results: list[FreeSpaceMap] = []

    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        page_w_pts = float(page.rect.width)
        page_h_pts = float(page.rect.height)

        # Rasterise to grayscale at analysis DPI
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        gray = (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width)
        )
        H, W = gray.shape

        # Shared line detection (same algorithm as Script 1)
        line_ys = _detect_lines(gray, dpi)

        # Run all three techniques
        left_strips, right_strips, gap_strips = _technique1_strips(
            gray, line_ys, dpi, W, H)

        word_gaps = _technique2_word_gaps(
            gray, line_ys, dpi, W, H)

        free_rects = _technique3_free_rects(
            gray, line_ys, dpi, W, H,
            min_area_px=min_rect_area_px)

        results.append(FreeSpaceMap(
            page=page_no,
            W=W, H=H, dpi=dpi,
            page_w_pts=page_w_pts,
            page_h_pts=page_h_pts,
            left_margin_strips=left_strips,
            right_margin_strips=right_strips,
            inter_line_gaps=gap_strips,
            word_gaps=word_gaps,
            free_rects=free_rects,
            line_ys=line_ys,
        ))

    doc.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  COORDINATE BRIDGE — FreeSpaceMap → remarks.json entries
# ─────────────────────────────────────────────────────────────────────────────

def to_remarks_coords(page_map: FreeSpaceMap) -> dict:
    """
    Converts a FreeSpaceMap into a dictionary of pre-computed coordinate
    suggestions that can be directly inserted into remarks.json for Script 1.

    Returns a dict with:
      "suggested_score_position"  — best place for a score circle (top-right)
      "margin_anchors"            — right margin y-positions for remark text
      "gap_anchors"               — inter-line gaps, sorted by position
      "word_gap_by_line"          — {line_num: [x_pct, ...]} for exponent inserts
    """
    # Best score circle position: largest free rect in top-right quadrant
    top_right_rects = [
        r for r in page_map.free_rects
        if r.centre_x_pct > 60 and r.centre_y_pct < 20
    ]
    score_pos = None
    if top_right_rects:
        best = max(top_right_rects, key=lambda r: r.area_px)
        score_pos = {
            "cx": best.centre_x_pct,
            "cy": best.centre_y_pct,
            "cy_line": best.nearest_line,
        }

    # Margin anchors: right margin strips, sorted top-to-bottom
    margin_anchors = [
        {
            "y_pct": s.centre_y_pct(),
            "height_px": s.height_px,
            "x_pct": s.x_left_pct,
            "after_line": s.after_line,
        }
        for s in sorted(page_map.right_margin_strips,
                        key=lambda s: s.y_top_px)
    ]

    # Gap anchors: inter-line gaps
    gap_anchors = [
        {
            "after_line": g.after_line,
            "before_line": g.before_line,
            "y_top_pct": g.y_top_pct,
            "y_bot_pct": g.y_bot_pct,
            "height_px": g.height_px,
        }
        for g in sorted(page_map.inter_line_gaps, key=lambda g: g.after_line)
    ]

    # Word gaps by line number
    word_gap_by_line: dict[int, list[float]] = {}
    for wg in page_map.word_gaps:
        word_gap_by_line.setdefault(wg.line, []).append(wg.x_gap_pct)

    return {
        "page": page_map.page,
        "suggested_score_position": score_pos,
        "margin_anchors": margin_anchors,
        "gap_anchors": gap_anchors,
        "word_gap_by_line": word_gap_by_line,
        "line_count": len(page_map.line_ys),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG VISUALIZER — draw all detections on the page image
# ─────────────────────────────────────────────────────────────────────────────

def visualize_freespace(
    pdf_bytes: bytes,
    page_maps: list[FreeSpaceMap],
    output_path: str,
    dpi: int = 250,
):
    """
    Renders each page with colour-coded overlays:
      • Blue  semi-transparent  — left/right margin free strips
      • Green semi-transparent  — inter-line gaps
      • Red   semi-transparent  — Technique 3 free rectangles
      • Cyan  dots              — word gap centres
      • Yellow lines            — detected text baselines
    Saves as a multi-page PDF for inspection.
    """
    import tempfile, os
    import img2pdf
    from PIL import Image, ImageDraw
    from pdf2image import convert_from_bytes

    pages_pil = convert_from_bytes(pdf_bytes, dpi=dpi)
    page_map_by_no = {pm.page: pm for pm in page_maps}
    annotated_paths = []

    with tempfile.TemporaryDirectory() as tmp:
        for i, pil_img in enumerate(pages_pil, 1):
            pm = page_map_by_no.get(i)
            if pm is None:
                continue
            overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Blue: margin strips
            for s in pm.left_margin_strips + pm.right_margin_strips:
                draw.rectangle(
                    [s.x_left_px, s.y_top_px, s.x_right_px, s.y_bot_px],
                    fill=(50, 100, 255, 55))

            # Green: inter-line gaps
            for g in pm.inter_line_gaps:
                draw.rectangle(
                    [g.x_left_px, g.y_top_px, g.x_right_px, g.y_bot_px],
                    fill=(30, 200, 80, 70))

            # Red: free rects (Technique 3)
            for r in pm.free_rects[:20]:
                draw.rectangle(
                    [r.x1_px, r.y1_px, r.x2_px, r.y2_px],
                    outline=(220, 40, 40, 200), width=2)

            # Cyan: word gap centres
            for wg in pm.word_gaps:
                r = 5
                draw.ellipse(
                    [wg.x_gap_px - r, pm.line_ys[wg.line - 1] - r,
                     wg.x_gap_px + r, pm.line_ys[wg.line - 1] + r],
                    fill=(0, 220, 220, 200))

            # Yellow: detected baselines
            for ly in pm.line_ys:
                draw.line([(0, ly), (pm.W, ly)], fill=(240, 200, 0, 130), width=1)

            combined = Image.alpha_composite(
                pil_img.convert("RGBA"), overlay).convert("RGB")
            path = os.path.join(tmp, f"page_{i:03d}.png")
            combined.save(path, "PNG")
            annotated_paths.append(path)

        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(annotated_paths))

    print(f"[freespace] Visualization saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python freespace_detector.py input.pdf [dpi] [viz_output.pdf]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    viz_out = sys.argv[3] if len(sys.argv) > 3 else None

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"[+] Analyzing {pdf_path} at {dpi} DPI …")
    maps = analyze_pdf_freespace(pdf_bytes, dpi=dpi)

    for pm in maps:
        print(f"    {pm.summary()}")
        coords = to_remarks_coords(pm)
        print(f"    Remarks coords: {json.dumps(coords, indent=2)}")

    if viz_out:
        visualize_freespace(pdf_bytes, maps, viz_out, dpi=dpi)