"""Smart OCR: classify → type-aware OCR → dedupe → structure (sections → flat items)."""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from src.gemini_copy_ocr import (
    copy_ocr_max_pages,
    copy_ocr_parallel_workers,
    copy_ocr_raster_dpi,
    count_pdf_pages,
    rasterize_pdf_to_png_pages,
)
from src.gemini_extract import MODEL_ID
from src.grid_overlay import batch_draw_grid, grid_to_pct

log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


_PAGE_TYPES = frozenset(
    {"DUPLICATE", "CORRECTION", "PARAGRAPH", "WORD_LIST", "UNKNOWN"}
)

# --- Annotation schemas ---------------------------------------------------------------

_ANCHOR_MARK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "type": types.Schema(
            type=types.Type.STRING,
            description="One of: ellipse, underline, tick",
        ),
        "cx": types.Schema(type=types.Type.NUMBER, description="Grid x center (1-50)"),
        "cy": types.Schema(type=types.Type.NUMBER, description="Grid y center (1-50)"),
        "rx": types.Schema(
            type=types.Type.NUMBER,
            description="x-radius in grid units; for underline = half horizontal span; for tick = 0",
        ),
        "ry": types.Schema(
            type=types.Type.NUMBER,
            description="y-radius; for underline = 0; for tick = 0",
        ),
    },
    required=["type", "cx", "cy", "rx", "ry"],
    property_ordering=["type", "cx", "cy", "rx", "ry"],
)

_REMARK_BOX_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "x1": types.Schema(type=types.Type.NUMBER, description="Top-left grid x (1-50)"),
        "y1": types.Schema(type=types.Type.NUMBER, description="Top-left grid y (1-50)"),
        "x2": types.Schema(type=types.Type.NUMBER, description="Bottom-right grid x (1-50)"),
        "y2": types.Schema(type=types.Type.NUMBER, description="Bottom-right grid y (1-50)"),
        "comment": types.Schema(type=types.Type.STRING, description="Teacher comment text; empty string if none"),
    },
    required=["x1", "y1", "x2", "y2", "comment"],
    property_ordering=["x1", "y1", "x2", "y2", "comment"],
)

_LAYOUT_ZONE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "y1": types.Schema(type=types.Type.INTEGER, description="First grid row of handwriting band (6-45)"),
        "y2": types.Schema(type=types.Type.INTEGER, description="Last grid row of handwriting band (6-45)"),
    },
    required=["y1", "y2"],
    property_ordering=["y1", "y2"],
)

_CONTENT_LINE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "y":  types.Schema(type=types.Type.INTEGER, description="Baseline grid row of this handwriting line (6-45)"),
        "x1": types.Schema(type=types.Type.INTEGER, description="Leftmost grid column with ink on this line (1-37)"),
        "x2": types.Schema(type=types.Type.INTEGER, description="Rightmost grid column with ink on this line (1-37)"),
    },
    required=["y", "x1", "x2"],
    property_ordering=["y", "x1", "x2"],
)

_CLASSIFY_ANNOTATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "page_type": types.Schema(
            type=types.Type.STRING,
            description=(
                "Exactly one of: DUPLICATE, CORRECTION, PARAGRAPH, WORD_LIST, UNKNOWN. "
                "DUPLICATE = visually identical layout to another sheet in this booklet. "
                "CORRECTION = short line/sentence fixes, numbered one-line corrections. "
                "PARAGRAPH = long prose answers, अर्थ/प्रयोग style blocks. "
                "WORD_LIST = word pairs, विलोम, उपसर्ग/प्रत्यय tables. "
                "UNKNOWN = none of the above clearly fits."
            ),
        ),
        "content_bands": types.Schema(
            type=types.Type.ARRAY,
            items=_LAYOUT_ZONE_SCHEMA,
            description=(
                "Contiguous y-row bands (rows 6–45 only) that contain student handwriting. "
                "Top rows 1-5 and bottom rows 46-50 are always margin — never include them. "
                "Right margin columns 38-50 are always structurally free and are NOT content."
            ),
        ),
        "content_lines": types.Schema(
            type=types.Type.ARRAY,
            items=_CONTENT_LINE_SCHEMA,
            description=(
                "Every individual handwriting line on the page. "
                "y = baseline row of the line, x1/x2 = leftmost/rightmost column with ink. "
                "List ALL lines top-to-bottom — one entry per written line."
            ),
        ),
        "anchor_marks": types.Schema(
            type=types.Type.ARRAY,
            items=_ANCHOR_MARK_SCHEMA,
            description="4-7 SUGGESTED teacher markup positions on student handwriting.",
        ),
        "remarks": types.Schema(
            type=types.Type.ARRAY,
            items=_REMARK_BOX_SCHEMA,
            description="3-5 free-whitespace bounding boxes in FREE ZONES only (see content_bands). Spread them across the full vertical range of the page — early, middle, and late y positions.",
        ),
    },
    required=["page_type", "content_bands", "anchor_marks", "remarks"],
    property_ordering=["page_type", "content_bands", "content_lines", "anchor_marks", "remarks"],
)


def _build_remark_comment_bank(language: str, page_type: str) -> str:
    """Return a comment-bank block for injection into the classify+annotation prompt."""
    lang = (language or "en").strip().lower()
    if lang == "hi":
        positive   = ["शुद्ध", "सही", "उत्तम", "बहुत अच्छा", "सटीक", "उचित", "संपूर्ण", "स्पष्ट"]
        partial    = ["अपूर्ण", "और विस्तार करें", "संक्षिप्त", "उदाहरण दें", "स्पष्ट करें", "क्रम ठीक करें"]
        negative   = ["गलत", "अशुद्ध", "सुधारें", "पुनः लिखें", "विलोम नहीं", "अर्थ गलत"]
        structural = ["प्रश्न का उत्तर नहीं", "विषयांतर", "लेखन अस्पष्ट"]
    else:
        positive   = ["Correct", "Good", "Well done", "Precise", "Accurate", "Clear", "Complete", "Apt"]
        partial    = ["Incomplete", "Elaborate", "Too brief", "Give example", "Clarify", "Revise order"]
        negative   = ["Wrong", "Incorrect", "Revise", "Rewrite", "Not the antonym", "Meaning incorrect"]
        structural = ["Off-topic", "Not answered", "Illegible"]

    if page_type == "WORD_LIST":
        strategy = ("For WORD_LIST: positive next to correct pairs, negative next to wrong pairs, "
                    "partial when half-right.")
    elif page_type == "CORRECTION":
        strategy = ("For CORRECTION: positive/negative per numbered item; partial for incomplete fixes.")
    else:
        strategy = ("For PARAGRAPH: positive beside strong sentences, partial where elaboration needed, "
                    "structural where student went off-topic.")

    return "\n".join([
        "COMMENT BANK — always pick the closest fit; do NOT output empty comments.",
        f"  Positive  : {', '.join(repr(c) for c in positive)}",
        f"  Partial   : {', '.join(repr(c) for c in partial)}",
        f"  Negative  : {', '.join(repr(c) for c in negative)}",
        f"  Structural: {', '.join(repr(c) for c in structural)}",
        "",
        strategy,
        "If none fit, write a 1-3 word teacher note in the same language.",
        'NEVER output comment="" — every remark box must have a meaningful comment.',
    ])


REMARK_COMMENT_BANKS: dict[str, dict[str, list[str]]] = {
    "hi": {
        "positive":   ["शुद्ध", "सही", "उत्तम", "बहुत अच्छा", "सटीक", "उचित", "संपूर्ण", "स्पष्ट", "सराहनीय"],
        "partial":    ["अपूर्ण", "और विस्तार करें", "संक्षिप्त", "उदाहरण दें", "स्पष्ट करें", "क्रम ठीक करें"],
        "negative":   ["गलत", "अशुद्ध", "सुधारें", "पुनः लिखें", "विलोम नहीं", "अर्थ गलत"],
        "structural": ["प्रश्न का उत्तर नहीं", "विषयांतर", "लेखन अस्पष्ट"],
    },
    "en": {
        "positive":   ["Correct", "Good", "Well done", "Precise", "Accurate", "Clear", "Complete", "Apt"],
        "partial":    ["Incomplete", "Elaborate", "Too brief", "Give example", "Clarify", "Revise order"],
        "negative":   ["Wrong", "Incorrect", "Revise", "Rewrite", "Not the antonym", "Meaning wrong"],
        "structural": ["Off-topic", "Not answered", "Illegible"],
    },
}


def _build_classify_annotation_prompt(page_num: int, total_pages: int, language: str) -> str:
    lang = (language or "en").strip().lower()
    script = (
        "Hindi (Devanagari) and/or English may appear."
        if lang == "hi"
        else "English and/or Hindi may appear."
    )
    comment_bank = _build_remark_comment_bank(language, "UNKNOWN")

    return f"""You are simulating a human Hindi exam teacher marking a handwritten answer booklet.
Page {page_num} of {total_pages}. {script}
A 50×50 reference grid is drawn. Columns 1–50: left→right. Rows 1–50: top→bottom.

════════════════════════════════════════════════════════
STEP 1 — Classify the page. Choose exactly one page_type:
  DUPLICATE  — layout visually identical to another sheet already in this booklet.
  CORRECTION — mostly short numbered sentence/line corrections.
  PARAGRAPH  — long prose answers, अर्थ/प्रयोग style blocks.
  WORD_LIST  — word pairs, विलोम, उपसर्ग/प्रत्यय tables.
  UNKNOWN    — none of the above clearly fits.

If DUPLICATE → output exactly:
{{"page_type":"DUPLICATE","content_bands":[],"anchor_marks":[],"remarks":[]}}
and stop.

════════════════════════════════════════════════════════
STEP 2 — Map handwriting layout (rows 6–45 only).
  • Ignore printed question text, ruled lines, boxes.
  • Rows 1–8 are top margin (printed header/roll box) — never list as content.
  • Rows 46–50 are bottom margin — never list as content.
  • Columns 38–50 are always the right margin — content stays in columns 1–37.

2a. content_bands — group rows into contiguous handwriting bands.
  Record each band: {{"y1": first_row_with_ink, "y2": last_row_with_ink}}
  Example: writing rows 9–22, gap, then rows 28–43:
  content_bands: [{{"y1":9,"y2":22}},{{"y1":28,"y2":43}}]
  IMPORTANT: The last band's y2 MUST reach the final handwriting row including
  conclusion paragraphs in the lower half. Scan rows 30–45 explicitly.

2b. content_lines — list EVERY individual handwriting line, one entry per written line.
  For each line record:
    y  = the grid row at the INK BASELINE of that line (bottom edge of the letters)
    x1 = leftmost grid column where ink starts on this line
    x2 = rightmost grid column where ink ends on this line
  List ALL lines top-to-bottom. Do not skip any line, including short one-word lines.
  Example (3 lines in a paragraph):
  content_lines: [{{"y":9,"x1":5,"x2":32}},{{"y":11,"x1":5,"x2":30}},{{"y":13,"x1":5,"x2":18}}]

════════════════════════════════════════════════════════
STEP 3 — Place ANCHOR MARKS (6–10) — BE DENSE, cover the full page.

A real teacher marks differently depending on what is written:

  ── UNDERLINE — MANDATORY: mark EVERY semantically important term ───────────
  • Underline under EVERY key vocabulary word, concept, technical term, or
    correctly-used phrase the student has written.
  • For PARAGRAPH: place 4–6 underlines distributed across the full answer body —
    early lines, middle lines, AND late lines. Never cluster all underlines at top.
  • For CORRECTION: underline every corrected verb/noun form.
  • For WORD_LIST: underline correct answer words in each pair row.
  • cy = BOTTOM EDGE of the word's ink (baseline row), not its vertical centre.
  • rx = half-width of the underlined phrase in grid units. cx ∈ [8, 34].
  • (cx − rx) ≥ 6 and (cx + rx) ≤ 36.
  • MINIMUM 4 underlines per page even on short answers; use the most
    content-rich lines if there are fewer words.

  ── TICK (✓) — Place next to each CORRECT student point or pair ──────────
  • cx must be in columns 5–8 (left edge, beside numbered items) OR
    immediately after the last character of the answer line (cx ~ end of text).
  • For WORD_LIST: one tick per correct word-pair row, at the row's baseline.
  • For CORRECTION: one tick per correct numbered sentence, at line baseline.
  • For PARAGRAPH: tick beside a strong concluding sentence.
  • If no answer is clearly correct, emit ZERO ticks — never fake a tick.
  • cx ∈ [5, 8] for left-margin ticks. rx = 0, ry = 0.

  ── ELLIPSE — Highlight a multi-word phrase (2–4 words) ──────────────────
  • Use for a short correct phrase (not a single word, not a whole sentence).
  • ry ≤ 1 grid unit (tight horizontal oval). cx ∈ [8, 34].

  PAGE-TYPE STRATEGY:
  • WORD_LIST   → prefer ticks + ellipses (correct vs wrong pairs) + underlines on key words
  • CORRECTION  → prefer ticks on correct lines + ellipses on wrong words + underlines on corrections
  • PARAGRAPH   → DENSE underlines (4–6) distributed top-to-bottom + ellipses on key ideas;
                  ticks only on strong conclusions
  • UNKNOWN     → minimum 4 underlines + 1 ellipse

  MANDATORY RULES:
  • Total anchor_marks ≥ 6 (aim for 8–10 on paragraph pages).
  • cx for ALL marks must be 8–34 EXCEPT left-margin ticks which use cx 5–8.
  • For underlines: cy is the BOTTOM of the word's ink, not the row centre.
  • NEVER place marks on printed question text, headers, or blank rows.
  • First mark must be ≥ row 13 (or wherever student handwriting starts).
  • DISTRIBUTE marks evenly: do not cluster all in the top third.

════════════════════════════════════════════════════════
STEP 4 — Place REMARK BOXES — ONE PER CONTENT BAND (minimum 4 total).

{comment_bank}

REMARK BOX GEOMETRY — WIDE FLAT STRIPS, NOT TALL BOXES:
  Each remark box must be a WIDE HORIZONTAL STRIP — like a single annotation line:
  • Height: y2 − y1 = 2 to 3 grid rows only (never taller than 4 rows).
  • Width : x2 − x1 ≥ 10 grid units (wide enough to read the comment clearly).
  • Right margin strip example: x1=39, x2=49, y1=12, y2=14
  • Left  margin strip example: x1=1,  x2=11, y1=20, y2=22
  • Inline gap strip example  : x1=5,  x2=37, y1=24, y2=26   ← full content-width strip

  Think of each remark as a sticky note STRIP along the margin or inside a blank gap —
  it should span horizontally rather than stack vertically.

PLACEMENT RULES — ONE REMARK PER CONTENT BAND:
  1. For EACH content_band you identified in STEP 2, place exactly ONE remark box.
     Its y-centre must fall within that band's [y1, y2] range.
  2. Add extra remarks for inline gaps (blank rows between bands) — use full-width strip.
  3. RIGHT margin zone : x1=39, x2=49, y1/y2 within the band. Use for ODD-numbered bands.
  4. LEFT margin zone  : x1=1,  x2=11, y1/y2 within the band. Use for EVEN-numbered bands.
  5. INLINE GAP strip  : x1=5, x2=37; spans the entire blank gap between two bands.
  6. NEVER overlap a content_band (for non-margin remarks).
  7. NEVER y1 < 9 or y2 > 45. x2 > x1. y2 = y1 + 2 (preferred) or y1 + 3 max.
  8. Width ≥ 10 units always.

  MINIMUM COUNT: max(4, number_of_content_bands) remark boxes.

  DISTRIBUTION — spread across the full vertical range:
  • Bands in rows 9–20   → RIGHT margin strip
  • Bands in rows 20–30  → LEFT margin strip  (or inline gap strip if gap present)
  • Bands in rows 30–38  → RIGHT margin strip
  • Bands in rows 38–45  → LEFT margin strip

  REMARK CONTENT rules:
  • Pick the single most fitting comment from the COMMENT BANK above.
  • Keep comments ≤ 4 words (they must fit in a wide strip).
  • Match the comment to the SPECIFIC text at that y-position on the page.
  • Right-margin strip : {{"x1":39,"y1":10,"x2":49,"y2":12,"comment":"सही"}}
  • Left-margin strip  : {{"x1":1, "y1":20,"x2":11,"y2":22,"comment":"Incomplete"}}
  • Inline-gap strip   : {{"x1":5, "y1":23,"x2":37,"y2":25,"comment":"अपूर्ण"}}

All coordinates: integers in [1, 50]."""


def _parse_content_bands(raw_bands: Any) -> list[dict[str, float]]:
    """Parse content_bands array from Gemini response into validated grid-coordinate dicts."""
    if not isinstance(raw_bands, list):
        return []
    bands: list[dict[str, float]] = []
    for b in raw_bands:
        if not isinstance(b, dict):
            continue
        try:
            y1 = max(1.0, min(50.0, float(b.get("y1", 1))))
            y2 = max(1.0, min(50.0, float(b.get("y2", 50))))
            if y2 > y1:
                bands.append({"y1": y1, "y2": y2})
        except (TypeError, ValueError):
            pass
    return bands


def _merge_content_bands(
    bands: list[dict[str, float]], gap_threshold: float = 4.0
) -> list[dict[str, float]]:
    """Merge consecutive bands separated by ≤ gap_threshold grid rows.

    Eliminates over-segmentation caused by inter-line whitespace within a
    single paragraph being detected as separate bands.
    """
    if not bands:
        return bands
    sorted_b = sorted(bands, key=lambda b: b["y1"])
    merged = [dict(sorted_b[0])]
    for b in sorted_b[1:]:
        if b["y1"] - merged[-1]["y2"] <= gap_threshold:
            merged[-1]["y2"] = max(merged[-1]["y2"], b["y2"])
        else:
            merged.append(dict(b))
    return merged


def _extend_last_band(
    bands: list[dict[str, float]],
    anchor_cys_grid: list[float],
    page_type: str,
) -> list[dict[str, float]]:
    """Extend the last content band to cover text below the final anchor mark.

    PARAGRAPH and CORRECTION pages often have a conclusion paragraph that Gemini
    stops short of. Extends to max(last_y2, max_anchor_cy) + 7, capped at grid 44.
    """
    if not bands or page_type not in ("PARAGRAPH", "CORRECTION"):
        return bands
    bands = [dict(b) for b in bands]
    last = bands[-1]
    max_anchor = max(anchor_cys_grid, default=last["y2"])
    new_y2 = min(44.0, max(last["y2"], max_anchor) + 7.0)
    if new_y2 > last["y2"]:
        log.info(
            "  extend_last_band  y2 %.0f → %.0f  (max_anchor=%.1f type=%s)",
            last["y2"], new_y2, max_anchor, page_type,
        )
        last["y2"] = new_y2
    return bands


def _remarks_from_anchor_marks(
    anchor_marks_grid: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive right-margin remark slots from anchor mark y-positions.

    Clusters anchor marks within _ANCHOR_CLUSTER_GAP_GRID rows and places one
    slot per cluster at the cluster centroid. Every remark is vertically aligned
    with something the teacher actually annotated.

    Fallback: 3 evenly spaced slots when no anchor marks are present.
    """
    if not anchor_marks_grid:
        return [
            {"x1": 39.0, "y1": 10.0, "x2": 49.0, "y2": 14.0, "comment": "",
             "_text_y1_grid": 10.0, "_text_y2_grid": 14.0, "_cluster_size": 0},
            {"x1": 39.0, "y1": 24.0, "x2": 49.0, "y2": 28.0, "comment": "",
             "_text_y1_grid": 24.0, "_text_y2_grid": 28.0, "_cluster_size": 0},
            {"x1": 39.0, "y1": 38.0, "x2": 49.0, "y2": 42.0, "comment": "",
             "_text_y1_grid": 38.0, "_text_y2_grid": 42.0, "_cluster_size": 0},
        ]

    cys = sorted(float(a.get("cy", 25)) for a in anchor_marks_grid)

    clusters: list[list[float]] = []
    for cy in cys:
        if clusters and cy - clusters[-1][-1] <= _ANCHOR_CLUSTER_GAP_GRID:
            clusters[-1].append(cy)
        else:
            clusters.append([cy])

    slots: list[dict[str, Any]] = []
    for cluster in clusters:
        cy = sum(cluster) / len(cluster)
        y1 = max(6.0, cy - 2.0)
        y2 = min(45.0, y1 + 4.0)
        slots.append({
            "x1": 39.0, "y1": y1, "x2": 49.0, "y2": y2, "comment": "",
            "_text_y1_grid": min(cluster),
            "_text_y2_grid": max(cluster),
            "_cluster_size": len(cluster),
        })

    # Ensure at least 3 remark slots so every page gets a minimum of 3 remarks.
    _MIN_SLOTS = 3
    if len(slots) < _MIN_SLOTS:
        occupied = {round(s["y1"]) for s in slots}
        candidates = [10.0, 20.0, 30.0, 38.0, 14.0, 26.0]
        for cy in candidates:
            if len(slots) >= _MIN_SLOTS:
                break
            y1 = max(6.0, cy - 2.0)
            if round(y1) in occupied:
                continue
            y2 = min(45.0, y1 + 4.0)
            slots.append({
                "x1": 39.0, "y1": y1, "x2": 49.0, "y2": y2, "comment": "",
                "_text_y1_grid": y1, "_text_y2_grid": y2, "_cluster_size": 0,
            })
            occupied.add(round(y1))
        slots.sort(key=lambda s: s["y1"])

    return slots


def _filter_overlapping_remarks(
    remarks: list[dict[str, Any]],
    content_bands: list[dict[str, float]],
    page_num: int = 0,
) -> list[dict[str, Any]]:
    """Remove remark boxes that overlap with identified handwriting zones.

    Right-margin remarks (x1 >= 38) are always kept — student writing stays in
    columns 1-37, so the right margin is free at any y position.
    Horizontal-gap remarks are dropped if their y-span intersects any content band.
    """
    if not content_bands:
        return remarks
    result: list[dict[str, Any]] = []
    for r in remarks:
        x1_grid = float(r.get("x1", 5))
        x2_grid = float(r.get("x2", 50))
        y1_grid = float(r.get("y1", 1))
        y2_grid = float(r.get("y2", 50))
        if x1_grid >= 38:
            log.info(
                "  remark_filter[p%s] KEPT  right_margin x[%.0f-%.0f] y[%.0f-%.0f]",
                page_num, x1_grid, x2_grid, y1_grid, y2_grid,
            )
            result.append(r)
            continue
        overlapping_band = next(
            (b for b in content_bands if y1_grid < b["y2"] and y2_grid > b["y1"]), None
        )
        if overlapping_band is None:
            log.info(
                "  remark_filter[p%s] KEPT  gap         x[%.0f-%.0f] y[%.0f-%.0f]",
                page_num, x1_grid, x2_grid, y1_grid, y2_grid,
            )
            result.append(r)
        else:
            log.info(
                "  remark_filter[p%s] DROP  overlap     x[%.0f-%.0f] y[%.0f-%.0f]"
                "  hits_band y[%.0f-%.0f]",
                page_num, x1_grid, x2_grid, y1_grid, y2_grid,
                overlapping_band["y1"], overlapping_band["y2"],
            )
    return result


_REM_RIGHT_X1 = 76.0   # right margin left edge  (~grid 39 → pct)
_REM_RIGHT_X2 = 96.0   # right margin right edge (~grid 49 → pct)
_REM_LEFT_X1  =  2.0   # left margin left edge   (~grid 2  → pct)
_REM_LEFT_X2  = 20.0   # left margin right edge  (~grid 11 → pct)
_REM_Y_MIN    = 16.0   # top guard: keep remarks below the printed header (~grid row 9)
_REM_Y_MAX    = 94.0   # bottom guard (% from top) — margin remarks can reach near page bottom
_REM_MIN_H    =  4.0   # minimum remark box height in %
_ANCHOR_CLUSTER_GAP_GRID = 3   # anchor marks within 3 grid rows → one remark slot
_CONTENT_X_MIN = 5.0   # leftmost grid column of student answer area
_CONTENT_X_MAX = 37.0  # rightmost grid column of student answer area (38+ is right margin)

# --- Font-aware remark height constants ---
_FONT_SIZE_PT          = 14.0   # rendering font size in points
_LINE_HEIGHT_FACTOR    = 1.25   # line height = font_size × factor
_BOX_PADDING_PT        =  4.0   # top+bottom padding inside the remark box (pt)
_AVG_CHAR_WIDTH_FACTOR = 0.55   # avg char width = font_size × factor
_PAGE_HEIGHT_PT        = 842.0  # A4 page height (pt)
_PAGE_WIDTH_PT         = 595.0  # A4 page width (pt)
_INTER_REMARK_GAP_PCT  =  1.5   # minimum clear gap between consecutive remarks (%)


def _remark_height_pct(comment: str, col_x1_pct: float, col_x2_pct: float) -> float:
    """Compute the remark box height (%) needed to fit `comment` at _FONT_SIZE_PT.

    Uses A4 dimensions, the column's percentage width, and char-wrap arithmetic.
    Always returns at least _REM_MIN_H.
    """
    col_width_pt   = (col_x2_pct - col_x1_pct) / 100.0 * _PAGE_WIDTH_PT
    usable_pt      = max(20.0, col_width_pt - _BOX_PADDING_PT)
    chars_per_line = max(8, int(usable_pt / (_FONT_SIZE_PT * _AVG_CHAR_WIDTH_FACTOR)))
    n_lines        = math.ceil(len(comment) / chars_per_line) if comment else 1
    h_pt           = n_lines * _FONT_SIZE_PT * _LINE_HEIGHT_FACTOR + _BOX_PADDING_PT
    return max(_REM_MIN_H, h_pt / _PAGE_HEIGHT_PT * 100.0)


def _first_free_slot(
    placed: list[tuple[float, float]], ideal_y1: float, height: float
) -> float | None:
    """Return the first y1 ≥ ideal_y1 where [y1, y1+height] doesn't overlap any placed interval.

    Scans placed intervals (sorted) and jumps past each conflict until a free gap is
    found or _REM_Y_MAX is exceeded. Returns None when no slot fits.
    """
    y1 = ideal_y1
    for _ in range(20):
        y2 = y1 + height
        if y2 > _REM_Y_MAX:
            return None
        conflict = next((p for p in placed if p[0] < y2 and p[1] > y1), None)
        if conflict is None:
            return y1
        y1 = conflict[1]
    return None


def _spread_remarks_two_column(
    remarks_pct: list[dict[str, Any]],
    page_num: int = 0,
) -> list[dict[str, Any]]:
    """Re-layout a page's remarks into right + left margin with no overlaps.

    Uses interval-based free-slot search: each remark independently finds the first
    non-overlapping slot in right and left columns, then places on whichever side is
    closer to its ideal y. True bidirectional — naturally alternates when one side is
    densely packed. Remarks that can't fit in either column are dropped silently.
    """
    if not remarks_pct:
        return []

    sorted_r = sorted(remarks_pct, key=lambda r: r.get("y1_pct", 0.0))

    log.info(
        "spread[p%s] input  %s", page_num,
        " ".join(
            f"x[{r.get('x1_pct',0):.0f}-{r.get('x2_pct',0):.0f}]"
            f"y[{r.get('y1_pct',0):.0f}-{r.get('y2_pct',0):.0f}]"
            for r in sorted_r
        ),
    )

    right_placed: list[tuple[float, float]] = []
    left_placed:  list[tuple[float, float]] = []
    result: list[dict[str, Any]] = []

    for orig in sorted_r:
        r       = dict(orig)
        comment = r.get("comment", "") or ""
        h_right = _remark_height_pct(comment, _REM_RIGHT_X1, _REM_RIGHT_X2)
        h_left  = _remark_height_pct(comment, _REM_LEFT_X1,  _REM_LEFT_X2)
        ideal   = max(_REM_Y_MIN, r.get("y1_pct", _REM_Y_MIN))

        ry1 = _first_free_slot(right_placed, ideal, h_right)
        ly1 = _first_free_slot(left_placed,  ideal, h_left)

        right_fits = ry1 is not None
        left_fits  = ly1 is not None

        # Fallback: when forward scan overflows, try bottom-clamped position in each column.
        # This avoids dropping remarks that can still fit anchored to the column's bottom.
        if not right_fits:
            bc = _REM_Y_MAX - h_right
            if bc >= _REM_Y_MIN and not any(p[0] < _REM_Y_MAX and p[1] > bc for p in right_placed):
                ry1 = bc
                right_fits = True
                log.info("  spread[p%s] R_bottom_clamp ideal_y=%.0f → y1=%.0f", page_num, ideal, bc)

        if not left_fits:
            bc = _REM_Y_MAX - h_left
            if bc >= _REM_Y_MIN and not any(p[0] < _REM_Y_MAX and p[1] > bc for p in left_placed):
                ly1 = bc
                left_fits = True
                log.info("  spread[p%s] L_bottom_clamp ideal_y=%.0f → y1=%.0f", page_num, ideal, bc)

        if not right_fits and not left_fits:
            log.info("  spread[p%s] DROP  truly_no_space ideal_y=%.0f", page_num, ideal)
            continue

        # Pick whichever side's free slot starts closer to ideal y.
        # When equal or right is nearer, prefer right.
        if right_fits and (not left_fits or ry1 <= ly1):
            col = "R"
            y1, y2 = ry1, ry1 + h_right
            right_placed.append((y1, y2 + _INTER_REMARK_GAP_PCT))
            right_placed.sort()
            r.update({"x1_pct": _REM_RIGHT_X1, "x2_pct": _REM_RIGHT_X2,
                       "y1_pct": round(y1, 2), "y2_pct": round(y2, 2)})
        else:
            col = "L"
            y1, y2 = ly1, ly1 + h_left
            left_placed.append((y1, y2 + _INTER_REMARK_GAP_PCT))
            left_placed.sort()
            r.update({"x1_pct": _REM_LEFT_X1, "x2_pct": _REM_LEFT_X2,
                       "y1_pct": round(y1, 2), "y2_pct": round(y2, 2)})

        log.info(
            "  spread[p%s] %s  ideal_y=%.0f → y[%.0f-%.0f](h=%.1f)  push=%.0f  comment_len=%s",
            page_num, col, ideal, y1, y2,
            h_right if col == "R" else h_left,
            y1 - ideal, len(comment),
        )
        result.append(r)

    log.info(
        "spread[p%s] output R=%s L=%s", page_num,
        " ".join(f"y[{y1:.0f}-{y2:.0f}]" for y1, y2 in right_placed),
        " ".join(f"y[{y1:.0f}-{y2:.0f}]" for y1, y2 in left_placed),
    )
    return result


def _recalc_connector(r: dict[str, Any]) -> None:
    """Recompute connector fields after spread may have changed the remark's column."""
    x1 = float(r.get("x1_pct", 76.0))
    x2 = float(r.get("x2_pct", 96.0))
    if x2 < 74.0:
        r["connector_type"] = "none"
        r.pop("connector_cy_pct", None)
        return
    text_y1 = float(r.get("text_y1_pct", 0.0))
    text_y2 = float(r.get("text_y2_pct", 0.0))
    span = text_y2 - text_y1
    if span >= 4.0:
        r["connector_type"] = "brace"
        r.pop("connector_cy_pct", None)
    else:
        r["connector_type"] = "arrow"
        r["connector_cy_pct"] = round((text_y1 + text_y2) / 2.0, 2)
    r["connector_x_pct"] = 73.0 if x1 >= 70.0 else 23.0


def _deoverlap_item_remarks(remarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-spread all remarks for one item after enrichment to eliminate any new overlaps.

    Groups by page, then runs _spread_remarks_two_column on each page's complete
    remark set. Inline gap remarks (x2_pct < 74%) bypass spread — they already have a
    precise full-width placement inside a handwriting gap. After spread, every remark's
    connector fields are recalculated to match the final column assignment.
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for r in remarks:
        pg = int(r.get("page", 0))
        by_page.setdefault(pg, []).append(r)

    result: list[dict[str, Any]] = []
    for pg in sorted(by_page.keys()):
        inline = [r for r in by_page[pg] if float(r.get("x2_pct", 96)) < 74.0]
        margin = [r for r in by_page[pg] if float(r.get("x2_pct", 96)) >= 74.0]
        spread = _spread_remarks_two_column(margin, page_num=pg)
        for r in spread:
            _recalc_connector(r)
        result.extend(spread)
        result.extend(inline)
    return result


def _parse_classify_annotation_response(raw: str, page_num: int) -> dict[str, Any]:
    """Parse combined classify+annotation JSON → {page_type, anchor_marks, remarks}."""
    base = _parse_annotation_response(raw, page_num)
    page_type = "UNKNOWN"
    content_bands: list[dict[str, float]] = []
    content_lines_parsed: list[dict[str, int]] = []
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("page_type") is not None:
            page_type = _coerce_page_type(str(data["page_type"]))
            raw_anchor_cys = [
                float(a.get("cy", 25))
                for a in data.get("anchor_marks", [])
                if isinstance(a, dict)
            ]
            content_bands = _parse_content_bands(data.get("content_bands", []))
            content_bands = _merge_content_bands(content_bands)
            content_bands = _extend_last_band(content_bands, raw_anchor_cys, page_type)

            # Parse individual content lines for coordinate logging.
            raw_lines = data.get("content_lines", [])
            content_lines_parsed: list[dict[str, int]] = []
            for ln in raw_lines if isinstance(raw_lines, list) else []:
                if not isinstance(ln, dict):
                    continue
                try:
                    content_lines_parsed.append({
                        "y":  max(1, min(50, int(ln.get("y",  25)))),
                        "x1": max(1, min(50, int(ln.get("x1",  1)))),
                        "x2": max(1, min(50, int(ln.get("x2", 37)))),
                    })
                except (TypeError, ValueError):
                    pass
            content_lines_parsed.sort(key=lambda l: l["y"])

            # Drop ellipses that sit in blank inter-paragraph gaps (not around any word).
            if content_bands:
                def _cy_in_band(cy_g: float) -> bool:
                    return any(b["y1"] <= cy_g <= b["y2"] for b in content_bands)
                base["anchor_marks"] = [
                    a for a in base["anchor_marks"]
                    if a["type"] != "ellipse" or _cy_in_band(float(a.get("cy", 25)))
                ]

            break

    # ── Diagnostic: layout analysis ──────────────────────────────────────────
    sorted_bands = sorted(content_bands, key=lambda b: b["y1"])
    free_zones: list[str] = []
    prev = 6.0
    for band in sorted_bands:
        if band["y1"] > prev + 0.5:
            free_zones.append(f"gap[{prev:.0f}-{band['y1']:.0f}]")
        prev = max(prev, band["y2"])
    if prev < 45.0:
        free_zones.append(f"gap[{prev:.0f}-45]")
    free_zones.insert(0, "right_margin[38-49,y6-45]")  # always available

    log.info(
        "layout[p%s] type=%-10s bands=%s",
        page_num, page_type,
        " ".join(f"y[{b['y1']:.0f}-{b['y2']:.0f}]" for b in sorted_bands) or "(none)",
    )
    log.info(
        "layout[p%s] free_zones=%s",
        page_num, " ".join(free_zones),
    )
    # Log every detected handwriting line with its grid bounding box.
    if content_lines_parsed:
        log.info(
            "layout[p%s] lines(%s)= %s",
            page_num,
            len(content_lines_parsed),
            "  ".join(
                f"row{l['y']}:col[{l['x1']}-{l['x2']}]"
                for l in content_lines_parsed
            ),
        )
    # ─────────────────────────────────────────────────────────────────────────

    if page_type == "DUPLICATE":
        return {"page_type": "DUPLICATE", "anchor_marks": [], "remarks": []}

    # Margin remark slots: derived from anchor-mark clusters (reliable positioning).
    derived = _remarks_from_anchor_marks(base["anchor_marks"])
    log.info(
        "layout[p%s] remarks_derived=%s",
        page_num,
        " ".join(
            f"y[{r.get('y1',0):.0f}-{r.get('y2',0):.0f}]" for r in derived
        ) or "(none)",
    )

    # Inline gap remarks: taken from Gemini's STEP 5 output (x1 < 38 = content-area x).
    # Margin suggestions (x1 >= 38) are ignored in favour of derived slots above.
    inline_gaps = [
        r for r in base["remarks"]
        if float(r.get("x1", 50)) < 38
    ]
    if inline_gaps:
        log.info(
            "layout[p%s] inline_gaps=%s",
            page_num,
            " ".join(f"x[{r.get('x1',0):.0f}-{r.get('x2',0):.0f}]y[{r.get('y1',0):.0f}-{r.get('y2',0):.0f}]"
                     for r in inline_gaps),
        )

    all_remarks = derived + inline_gaps
    base["remarks"] = _filter_overlapping_remarks(all_remarks, content_bands, page_num)
    return {"page_type": page_type, **base}


def _classify_and_annotate_page(
    api_key: str,
    grid_png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
) -> dict[str, Any]:
    """Combined classify + annotation call using grid PNG.

    Returns {"page_type": str, "anchor_marks": [...], "remarks": [...]}.
    Falls back to page_type="UNKNOWN" and empty lists on failure.
    """
    client = genai.Client(api_key=api_key)
    prompt = _build_classify_annotation_prompt(page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=grid_png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_CLASSIFY_ANNOTATION_SCHEMA,
    )
    last_raw = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(model=MODEL_ID, contents=parts, config=cfg)
        last_raw = (getattr(resp, "text", None) or "").strip()
        if not last_raw:
            if attempt < 2:
                time.sleep(0.5)
            continue
        log.info("classify_annotate[p%s] raw_response=%s", page_num, last_raw)
        result = _parse_classify_annotation_response(last_raw, page_num)
        if result["page_type"] != "UNKNOWN" or result["anchor_marks"] or result["remarks"]:
            return result
        if attempt < 2:
            time.sleep(0.5)
    log.warning("classify_and_annotate: fallback page=%s raw=%r", page_num, last_raw[:200])
    return {"page_type": "UNKNOWN", "anchor_marks": [], "remarks": []}


def _parse_annotation_response(raw: str, page_num: int) -> dict[str, Any]:
    """Parse annotation-only JSON → {anchor_marks, remarks}."""
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        raw_anchors = data.get("anchor_marks") or []
        raw_remarks = data.get("remarks") or []

        anchor_marks: list[dict[str, Any]] = []
        for a in raw_anchors if isinstance(raw_anchors, list) else []:
            if not isinstance(a, dict):
                continue
            t = str(a.get("type", "")).strip().lower()
            if t not in ("ellipse", "underline", "tick"):
                continue
            try:
                cx = float(a.get("cx", 25))
                cy = float(a.get("cy", 25))
                rx = float(a.get("rx", 2))
                ry = float(a.get("ry", 2))
                # Clamp anchor marks to the answer content columns [_CONTENT_X_MIN, _CONTENT_X_MAX]
                if t == "underline":
                    ry = 0.0
                    rx = min(rx, cx - _CONTENT_X_MIN, _CONTENT_X_MAX - cx)
                    rx = max(0.5, rx)
                    cx = max(_CONTENT_X_MIN + rx, min(_CONTENT_X_MAX - rx, cx))
                elif t in ("circle", "ellipse"):
                    cx = max(_CONTENT_X_MIN + rx, min(_CONTENT_X_MAX - rx, cx))
                else:  # tick
                    cx = max(_CONTENT_X_MIN, min(_CONTENT_X_MAX, cx))
                anchor_marks.append({"type": t, "cx": cx, "cy": cy, "rx": rx, "ry": ry})
            except (TypeError, ValueError):
                pass

        remarks: list[dict[str, Any]] = []
        for r in raw_remarks if isinstance(raw_remarks, list) else []:
            if not isinstance(r, dict):
                continue
            try:
                _gs = 50.0
                x1 = max(1.0, min(_gs, float(r.get("x1", 5))))
                y1 = max(1.0, min(_gs, float(r.get("y1", 80))))
                x2 = max(1.0, min(_gs, float(r.get("x2", 45))))
                y2 = max(1.0, min(_gs, float(r.get("y2", 90))))
                if x2 - x1 < 6:
                    x2 = min(_gs, x1 + 6)
                if y2 - y1 < 3:
                    y2 = min(_gs, y1 + 3)
                remarks.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "comment": str(r.get("comment", ""))})
            except (TypeError, ValueError):
                pass

        return {"anchor_marks": anchor_marks, "remarks": remarks}

    log.warning("parse_annotation_response failed page=%s raw=%r", page_num, (raw or "")[:200])
    return {"anchor_marks": [], "remarks": []}


def _coerce_page_type(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "_")
    if t in _PAGE_TYPES:
        return t
    return "UNKNOWN"


# --- Stage 2: type-aware OCR prompts ---------------------------------------------------

def _build_ocr_prompt_generic(page_num: int, total_pages: int, language: str) -> str:
    lang = (language or "en").strip().lower()
    script_note = (
        "Preserve Hindi (Devanagari) and English exactly — do NOT transliterate."
        if lang == "hi"
        else "Preserve English and Hindi exactly — do NOT transliterate."
    )
    return f"""You are performing OCR for page {page_num} of {total_pages} from a handwritten exam copy.

Transcribe ALL visible text on this page exactly as written.
- Preserve line breaks and paragraph breaks.
- Keep question labels like Q1, (1), प्रश्न 1, उत्तर exactly.
- Do not translate or transliterate text.
- For tables/lists, keep one row per line; use " | " between cells when needed.
- If a word is unreadable, use [illegible] only for that span.
- Do not invent, summarize, or correct content.

{script_note}

Return ONLY valid JSON:
{{"text":"full page transcript with \\n for line breaks"}}"""


def _build_ocr_prompt_correction(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (CORRECTION style): Prefer one line per numbered response where possible.
If numbering is visible, preserve it exactly."""


def _build_ocr_prompt_paragraph(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (PARAGRAPH style): Keep question numbers/labels visible before each answer block."""


def _build_ocr_prompt_word_list(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (WORD_LIST style): Keep one pair/row per line.
Preserve column order and separators."""


def _ocr_prompt_for_page_type(
    page_type: str, page_num: int, total_pages: int, language: str
) -> str:
    if page_type == "CORRECTION":
        return _build_ocr_prompt_correction(page_num, total_pages, language)
    if page_type == "PARAGRAPH":
        return _build_ocr_prompt_paragraph(page_num, total_pages, language)
    if page_type == "WORD_LIST":
        return _build_ocr_prompt_word_list(page_num, total_pages, language)
    return _build_ocr_prompt_generic(page_num, total_pages, language)


# --- Stage 3: structure schema (sections) then flatten to legacy items -----------------

_QUESTION_BLOCK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "question_id": types.Schema(type=types.Type.INTEGER),
        "question": types.Schema(type=types.Type.STRING),
        "student_answer": types.Schema(type=types.Type.STRING),
        "answer_type": types.Schema(
            type=types.Type.STRING,
            description="One of: correction, paragraph, word_list",
        ),
        "start_page": types.Schema(type=types.Type.INTEGER),
        "start_y_position_percent": types.Schema(type=types.Type.NUMBER),
        "end_page": types.Schema(type=types.Type.INTEGER),
        "end_y_position_percent": types.Schema(type=types.Type.NUMBER),
        "marking_page": types.Schema(type=types.Type.INTEGER),
        "marking_x_position_percent": types.Schema(type=types.Type.NUMBER),
        "marking_y_position_percent": types.Schema(type=types.Type.NUMBER),
    },
    required=[
        "question_id",
        "question",
        "student_answer",
        "answer_type",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
    property_ordering=[
        "question_id",
        "question",
        "student_answer",
        "answer_type",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
)

_SECTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "section_name": types.Schema(type=types.Type.STRING),
        "questions": types.Schema(
            type=types.Type.ARRAY,
            items=_QUESTION_BLOCK_SCHEMA,
        ),
    },
    required=["section_name", "questions"],
    property_ordering=["section_name", "questions"],
)

_STRUCTURE_ROOT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sections": types.Schema(
            type=types.Type.ARRAY,
            items=_SECTION_SCHEMA,
        ),
    },
    required=["sections"],
    property_ordering=["sections"],
)

_ANSWER_TYPES = frozenset({"correction", "paragraph", "word_list"})

_QUESTION_NUM_RE = re.compile(
    r"""
    (?:
        \b(?:que(?:stion)?|q)\s*[:\.\-]?\s*(\d{1,3})\b       # Que:7 / Q.3 / Question 10
      | \bप्रश्न\s*[:\.\-]?\s*(\d{1,3})\b                     # प्रश्न 5
      | ^\s*\((\d{1,3})\)\s*[:\.\-]                          # (3):
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def _extract_question_number(question_text: str) -> int | None:
    """Parse the original question number from its leading label text."""
    m = _QUESTION_NUM_RE.search(question_text or "")
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            try:
                n = int(g, 10)
                return n if 1 <= n <= 500 else None
            except (TypeError, ValueError):
                return None
    return None


def _structure_item_sort_key(item: dict[str, Any]) -> tuple[int | float, ...]:
    sp = int(item.get("start_page", 1))
    sy = float(item.get("start_y_position_percent", 0.0))
    qi = item.get("question_id")
    if qi is None:
        return (sp, sy, 999999)
    try:
        return (sp, sy, int(qi))
    except (TypeError, ValueError):
        return (sp, sy, 999998)


_OCR_PAGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"text": types.Schema(type=types.Type.STRING)},
    required=["text"],
    property_ordering=["text"],
)


# --- JSON / text helpers ---------------------------------------------------------------

def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if m:
        return m.group(1).strip()
    return t


def _repair_json(text: str) -> str:
    t = _strip_json_fence(text)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    stripped = re.sub(r"\\.", "", t)
    if stripped.count('"') % 2 != 0:
        t += '"'
    open_b = t.count("{") - t.count("}")
    open_sq = t.count("[") - t.count("]")
    if open_b > 0:
        t += "}" * open_b
    if open_sq > 0:
        t += "]" * open_sq
    return t


def _parse_ocr_page_text(raw: str, page_num: int) -> str:
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("text") is not None:
            text = str(parsed.get("text", "")).strip()
            if text:
                return text
    raise ValueError(f"Could not parse OCR text JSON for page {page_num}.")


def _clamp_pct(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(100.0, x))


def _page_num(v: Any, total_pages: int, fallback: int = 1) -> int:
    try:
        p = int(v)
    except (TypeError, ValueError):
        p = fallback
    return max(1, min(total_pages, p))


def _normalize_answer_type(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in _ANSWER_TYPES:
        return s
    if s in ("correction", "short", "line"):
        return "correction"
    if s in ("word_list", "wordlist", "list", "pairs", "tabular"):
        return "word_list"
    if s in ("paragraph", "prose", "long"):
        return "paragraph"
    return "paragraph"


def _estimate_expected_questions(page_blocks: list[str]) -> int | None:
    """Heuristic: infer approximate question count from visible numbering labels."""
    nums: set[int] = set()
    patterns = (
        r"\bq(?:uestion)?\s*[:.\-]?\s*(\d{1,3})\b",
        r"(?:^|\n)\s*\(?(\d{1,3})\)?\s*[.)\-:]\s+",
        r"(?:प्रश्न|उ\.?\s*प्र\.?)\s*[:.\-]?\s*(\d{1,3})\b",
    )
    for block in page_blocks:
        body = _extract_page_body(block)
        for pat in patterns:
            for m in re.finditer(pat, body, flags=re.IGNORECASE):
                try:
                    n = int(m.group(1))
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= 200:
                    nums.add(n)
    if not nums:
        return None
    return max(nums)


def _normalize_flat_item(
    item: dict[str, Any],
    total_pages: int,
    section_name: str,
) -> dict[str, Any]:
    start_page = _page_num(item.get("start_page"), total_pages, 1)
    end_page = _page_num(item.get("end_page"), total_pages, start_page)
    if end_page < start_page:
        end_page = start_page

    question_text = str(item.get("question", "")).strip()
    label_num = _extract_question_number(question_text)
    try:
        model_qid = int(item.get("question_id", 0))
    except (TypeError, ValueError):
        model_qid = 0
    if label_num is not None:
        qid: int | None = label_num
    elif model_qid >= 1:
        qid = model_qid
    else:
        qid = None

    ans = str(item.get("student_answer", "")).strip()
    start_sy = _clamp_pct(item.get("start_y_position_percent", 0))
    end_sy = _clamp_pct(item.get("end_y_position_percent", 100))

    # Score mark always on the start page at the y where the answer begins.
    # Ignore Gemini's marking_page to avoid cross-page y-mismatch.
    marking_page = start_page
    my = start_sy

    return {
        "question_id": qid,
        "question": question_text,
        "student_answer": ans,
        "is_attempted": bool(ans),
        "section_name": section_name.strip(),
        "answer_type": _normalize_answer_type(item.get("answer_type")),
        "start_page": start_page,
        "start_y_position_percent": start_sy,
        "end_page": end_page,
        "end_y_position_percent": end_sy,
        "marking_page": marking_page,
        "marking_x_position_percent": 85.0,  # always right margin (center of 76–96% band)
        "marking_y_position_percent": my,
    }


def _parse_structure_sections(raw: str, total_pages: int) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    parsed: Any = None
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        if not candidate.strip():
            continue
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as e:
            last_err = e
    else:
        msg = str(last_err) if last_err else "invalid JSON"
        log.warning("smart_ocr structure JSON parse failed: %s", msg)
        raise ValueError(msg) from last_err

    if not isinstance(parsed, dict):
        raise ValueError("Structure response must be a JSON object.")
    sections = parsed.get("sections")
    if not isinstance(sections, list):
        raise ValueError("Structure response missing sections array.")

    out: list[dict[str, Any]] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        section_name = str(sec.get("section_name", "")).strip() or "अज्ञात अनुभाग"
        questions = sec.get("questions")
        if not isinstance(questions, list):
            continue
        for row in questions:
            if not isinstance(row, dict):
                continue
            norm = _normalize_flat_item(row, total_pages, section_name)
            out.append(norm)
    if not out:
        raise ValueError("No question-answer blocks detected.")
    out.sort(key=_structure_item_sort_key)

    used_ids: set[int] = {
        i
        for i in (item.get("question_id") for item in out)
        if isinstance(i, int)
    }
    fallback_counter = 1
    for item in out:
        if item.get("question_id") is not None:
            continue
        while fallback_counter in used_ids:
            fallback_counter += 1
        item["question_id"] = fallback_counter
        used_ids.add(fallback_counter)
        fallback_counter += 1
    return out


# --- Dedup -----------------------------------------------------------------------------

_PAGE_HEADER_RE = re.compile(r"^===\s*PAGE\s+(\d+)\s*===\s*$", re.MULTILINE | re.IGNORECASE)


def _extract_page_body(block: str) -> str:
    """Body text after === PAGE n === line."""
    m = _PAGE_HEADER_RE.search(block)
    if not m:
        return block.strip()
    return block[m.end() :].strip()


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(
        None, _normalize_for_compare(a), _normalize_for_compare(b)
    ).ratio()


def _deduplicate_page_texts(
    page_blocks: list[str],
    classifications: list[str],
    request_id: str,
    *,
    sim_threshold: float = 0.92,
) -> list[str]:
    """Drop near-duplicate page bodies; keep === PAGE n === headers for stable numbering."""
    out = list(page_blocks)
    for i in range(len(out)):
        page_no = i + 1
        header = f"=== PAGE {page_no} ==="
        body = _extract_page_body(out[i])

        if classifications[i] == "DUPLICATE" and i > 0:
            out[i] = (
                f"{header}\n"
                f"[OMITTED: page classified DUPLICATE — duplicate of earlier sheet; "
                f"OCR skipped to save tokens.]\n"
            )
            log.info("smart_ocr[%s] dedup: DUPLICATE class page=%s", request_id, page_no)
            continue

        if i == 0 or "[OMITTED:" in body:
            continue

        prev_body = _extract_page_body(out[i - 1])
        if "[OMITTED:" in prev_body:
            continue

        if _similarity(prev_body, body) >= sim_threshold:
            out[i] = (
                f"{header}\n"
                f"[OMITTED: near-duplicate of page {page_no - 1} (similarity >= {sim_threshold}).]\n"
            )
            log.info(
                "smart_ocr[%s] dedup: similarity collapse page=%s vs %s",
                request_id,
                page_no,
                page_no - 1,
            )
    return out


# --- Gemini calls: OCR ----------------------------------------------------------------


def _ocr_single_page(
    api_key: str,
    png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
    page_type: str,
) -> str:
    client = genai.Client(api_key=api_key)
    prompt = _ocr_prompt_for_page_type(page_type, page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_OCR_PAGE_SCHEMA,
    )
    page_text = ""
    last_raw = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        last_raw = (getattr(resp, "text", None) or "").strip()
        if not last_raw:
            if attempt < 2:
                time.sleep(0.5)
            continue
        try:
            page_text = _parse_ocr_page_text(last_raw, page_num)
            break
        except ValueError:
            if attempt < 2:
                time.sleep(0.5)

    if not page_text:
        raise RuntimeError(
            f"OCR failed for page {page_num} after 2 attempts; raw={last_raw[:200]!r}"
        )

    return f"=== PAGE {page_num} ===\n{page_text}"


def _annotations_grid_to_pct(
    anchor_marks: list[dict[str, Any]],
    remarks: list[dict[str, Any]],
    page: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert grid coords (1-50) to percentage coords (0-100) and attach page number."""
    out_anchors: list[dict[str, Any]] = []
    for a in anchor_marks:
        cx_pct, cy_pct = grid_to_pct(a["cx"], a["cy"])
        rx_pct, _ = grid_to_pct(a["rx"] + 1, 1)   # radius: shift from origin
        _, ry_pct = grid_to_pct(1, a["ry"] + 1)
        out_anchors.append({
            "type": a["type"],
            "page": page,
            "cx_pct": round(cx_pct, 2),
            "cy_pct": round(cy_pct, 2),
            "rx_pct": round(rx_pct, 2),
            "ry_pct": round(ry_pct, 2),
        })

    out_remarks: list[dict[str, Any]] = []
    for r in remarks:
        x1_pct, y1_pct = grid_to_pct(r["x1"], r["y1"])
        x2_pct, y2_pct = grid_to_pct(r["x2"], r["y2"])

        # Text span: stored in grid coords by _remarks_from_anchor_marks
        tg1 = r.get("_text_y1_grid", r["y1"])
        tg2 = r.get("_text_y2_grid", r["y2"])
        _, text_y1_pct = grid_to_pct(1, tg1)
        _, text_y2_pct = grid_to_pct(1, tg2)
        cluster_size = r.get("_cluster_size", 0)
        span_pct = text_y2_pct - text_y1_pct

        # Inline gap remark spans the content area (not a margin) — no connector needed.
        is_inline = x2_pct < 74.0
        conn_cy: float | None = None
        if is_inline:
            conn_type = "none"
            is_right = False
        elif cluster_size >= 2 or span_pct >= 4.0:
            conn_type = "brace"
            is_right = x1_pct >= 70.0
        else:
            conn_type = "arrow"
            conn_cy = round((text_y1_pct + text_y2_pct) / 2.0, 2)
            is_right = x1_pct >= 70.0
        conn_x = 73.0 if is_right else 23.0

        rem_out: dict[str, Any] = {
            "page": page,
            "x1_pct": round(x1_pct, 2),
            "y1_pct": round(y1_pct, 2),
            "x2_pct": round(x2_pct, 2),
            "y2_pct": round(y2_pct, 2),
            "comment": r.get("comment", ""),
            "text_y1_pct": round(text_y1_pct, 2),
            "text_y2_pct": round(text_y2_pct, 2),
            "connector_type": conn_type,
            "connector_x_pct": conn_x,
        }
        if conn_cy is not None:
            rem_out["connector_cy_pct"] = conn_cy
        out_remarks.append(rem_out)

    return out_anchors, out_remarks


def _structure_qa(
    client: genai.Client,
    pages_payload_json: str,
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    lang_note = (
        "Text is Hindi/English mixed; preserve wording in student_answer when present."
        if (language or "en").strip().lower() == "hi"
        else "Text may be Hindi/English mixed; preserve wording in student_answer when present."
    )
    expected_hint = (
        f"The exam has approximately {expected_questions} questions. "
        "You MUST extract all question-answer pairs and not stop early.\n"
        if expected_questions
        else ""
    )
    prompt = f"""You are structuring a handwritten exam answer book from OCR text only.

Input is JSON with this shape:
{{"pages":[{{"page":1,"text":"..."}}, ...]}}
where page numbers are physical page numbers 1..{total_pages}.

PHASE 2 ONLY (this structure pass) — SEGREGATE INTRO / COVER / ADMIN SHEETS:
The prior OCR pass transcribed every page verbatim. Here you must IGNORE front-matter
when building graded answer rows:
- Typical non-answer pages: candidate particulars, roll/barcode, serial seals, instructions
  ("read carefully", "do not write in margin"), timing, rough-work-only areas, blank
  continuation headers, invigilator notes, etc.
- Do NOT emit any question row whose text is only that boilerplate — it is not an exam answer.
- Answers almost always start where real question labels appear (Q1, Que:1, प्रश्न 1,
  Section A/B, Paper codes) together with the student's handwritten response.
- If OCR mixes boilerplate and real answers on one page, attach coordinates and text to
  the actual question/answer blocks only; do not invent phantom Q… rows from cover text.

TASK:
1. Read pages in order and identify TOP-LEVEL questions only.
   A top-level question is introduced by a label like:
     Q1, Q:2, Que:3, Que 4, Q.5, प्रश्न 1, (1), etc.

   CRITICAL — DO NOT split these into separate items:
   - Sub-parts of a case study (labeled 1. 2. 3. or (a)(b)(c) under the same Que: N)
   - Continuation of an answer on the next page
   - Numbered points within an answer

   A new top-level question starts ONLY when a new Que/Q/प्रश्न label appears
   that is at the SAME level as the main question headers, NOT when a
   numbered sub-part (1. / 2. / 3.) appears inside a question or its answer.

2. For case study questions with sub-parts (e.g. "Questions:- 1. ... 2. ... 3. ..."):
   - Treat the entire case study as ONE item.
   - student_answer must include ALL sub-part answers concatenated in order.
   - question must include the full case study stem AND all sub-part labels.

3. Group questions into sections using section_name in Hindi or short English.

4. For each question emit:
   - question_id: the integer taken from that question stem's label in the OCR.
     Examples: Que:7 → 7, Q.3 → 3, प्रश्न 5 → 5.
     Do NOT substitute sequential 1,2,3 … when the paper shows Que:10, Que:11, etc. Use the printed numbering.
     If no number is visible in the stem, guess from context; the server may assign a gap id when still unknown.
     NEVER invent a row for a booklet question that never appears in the OCR (if Q7 is absent everywhere, omit question_id 7 entirely).
   - question: full question text including any sub-part labels
   - student_answer: complete answer text, joining sub-part answers with \\n\\n — candidate handwriting only (do NOT paste the printed stem here; that belongs solely in ``question``).
   - answer_type: correction | paragraph | word_list
   - start_page, start_y_position_percent (0–100 from page TOP): top of visible question header/label block.
   - end_page, end_y_position_percent: bottom edge of handwritten answer ink for this question (not blank margin).
   - marking_page, marking_x_position_percent, marking_y_position_percent: anchor for examiner remark overlay on ONE page.

5. MARKING COORDINATE QUALITY (MUST ADAPT PER COPY; NO FIXED FORMULA):
   - Infer the handwritten answer zone visually from this specific copy/page, then place marking_y inside that zone.
   - Do NOT use any fixed global offset from top/bottom; derive placement from visible layout for each question.
   - Keep marking away from printed question text, headings, instructions, and footer rules; place it near student's ink.
   - For long answers spanning pages, choose marking_page where the answer body is clearest and densest.
   - For sparse/short answers, place marking near the actual written line/phrase, not on empty printed area.
   - If unsure between two positions, choose the one deeper in handwritten content (still readable and margin-safe).

6. BLANK / UNATTEMPTED ANSWERS — CRITICAL:
   - If there is no handwritten answer (blank area, only the question printed, "Ans:-" with nothing after):
     set student_answer to exactly "" (empty string). Do NOT invent or paraphrase filler text.
     Do NOT put teacher feedback, scores, or commentary inside student_answer — that field is OCR content only.
   - Still emit one row per such question so numbering stays aligned — do NOT omit unanswered questions.

{lang_note}
{expected_hint}

CRITICAL: Extract every top-level question from the OCR in order through the last page, including unanswered ones. Do not emit extra items for sub-parts of a single case study.

WORKED EXAMPLE of case study grouping:
  OCR contains:
    "Que:9 You are a young IAS officer... public image.
     Questions:-
     1. Should there be ethical boundaries...?
     2. How will you handle the situation?
     3. What things should public servant keep in mind?"

  CORRECT — emit ONE item:
    question_id: 9
    question: "Que:9 You are a young IAS officer... Questions:- 1. Should there be... 2. How will you... 3. What things..."
    student_answer: "<answer to 1>\\n\\n<answer to 2>\\n\\n<answer to 3>"

  WRONG — do NOT emit three separate items with question_ids 9, 10, 11.

CRITICAL JSON:
- Valid JSON only. Use \\n inside strings, never raw newlines inside JSON strings.
- Escape " as \\" inside strings.
- Return one object: {{"sections":[{{"section_name":"...","questions":[...]}}]}}
"""

    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=pages_payload_json),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=65536,
        response_mime_type="application/json",
        response_schema=_STRUCTURE_ROOT_SCHEMA,
    )
    last_err: Exception | None = None
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw:
            raise ValueError("Structure pass returned an empty response.")
        try:
            return _parse_structure_sections(raw, total_pages)
        except ValueError as e:
            last_err = e
            log.warning("structure_qa parse attempt %s/2 failed: %s", attempt, e)
            if attempt < 2:
                time.sleep(0.4)
    raise ValueError(f"Structure pass failed after 2 attempts: {last_err}") from last_err


def _structure_qa_with_fallback(
    client: genai.Client,
    pages_payload: dict[str, Any],
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    pages_payload_json = json.dumps(pages_payload, ensure_ascii=False)
    rows = _structure_qa(
        client,
        pages_payload_json,
        language,
        total_pages,
        expected_questions=expected_questions,
    )
    if expected_questions and len(rows) < max(1, int(expected_questions * 0.8)):
        pages = pages_payload.get("pages", [])
        if not isinstance(pages, list) or len(pages) < 2:
            return rows
        log.warning(
            "structure_qa undercount got=%s expected~%s; retry split",
            len(rows),
            expected_questions,
        )
        mid = len(pages) // 2
        first = {"pages": pages[:mid]}
        second = {"pages": pages[mid:]}
        rows1 = _structure_qa(
            client,
            json.dumps(first, ensure_ascii=False),
            language,
            total_pages,
            expected_questions=None,
        )
        rows2 = _structure_qa(
            client,
            json.dumps(second, ensure_ascii=False),
            language,
            total_pages,
            expected_questions=None,
        )
        rows = rows1 + rows2
        rows.sort(key=_structure_item_sort_key)
        seen_ids: set[Any] = set()
        deduped: list[dict[str, Any]] = []
        for item in rows:
            qid = item.get("question_id")
            if qid not in seen_ids:
                deduped.append(item)
                seen_ids.add(qid)
        rows = deduped
    return rows


# --- Remarks enrichment ---------------------------------------------------------------


def enrich_remarks_from_annotations(items: list[dict[str, Any]]) -> None:
    """Fill remarks[].comment from the evaluation annotations array.

    Called after ``merge_evaluations_into_items()`` has placed grading-generated
    ``annotations`` onto each item. Each annotation carries a ``comment`` string and
    a visual position (page_index 0-based, y_position_percent). This function:

    1. Matches every annotation to the nearest unmatched remark box on the same page
       (by y-centre distance).  If the best match is within 20 percentage points, the
       remark's comment is set from the annotation.
    2. Annotations that find no close-enough remark box get a **new remark entry**
       built from the annotation's own coords, clamped to the item's answer-body zone
       so injected remarks never land on the printed question-text area.

    Mutates items in-place; returns None.
    """
    _GUARD = 8.0  # % to skip past printed question-text zone from start_y

    for item in items:
        remarks: list[dict[str, Any]] = item.get("remarks") or []
        annotations: list[dict[str, Any]] = item.get("annotations") or []
        if not annotations:
            continue

        item_start_page = int(item.get("start_page", 1))
        item_end_page   = int(item.get("end_page", item_start_page))
        item_sy         = float(item.get("start_y_position_percent", 0.0))
        item_ey         = float(item.get("end_y_position_percent", 100.0))

        matched_remark_indices: set[int] = set()

        for ann in annotations:
            try:
                target_page = int(ann.get("page_index", 0)) + 1   # 0-based → 1-based
                ann_y: float = float(ann.get("y_position_percent", 50))
            except (TypeError, ValueError):
                continue

            best_idx: int | None = None
            best_dist = float("inf")
            for i, rem in enumerate(remarks):
                if i in matched_remark_indices:
                    continue
                if rem.get("comment"):   # already has a comment from a previous annotation
                    continue
                if rem.get("page") != target_page:
                    continue
                y_centre = (float(rem.get("y1_pct", 0)) + float(rem.get("y2_pct", 0))) / 2.0
                dist = abs(ann_y - y_centre)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist <= 20.0:
                remarks[best_idx]["comment"] = str(ann.get("comment", ""))
                matched_remark_indices.add(best_idx)
            else:
                # No nearby remark box — create one in the right margin.
                # Determine the effective answer-body y-range for this page so the
                # injected remark never lands on the printed question-text zone.
                if target_page == item_start_page == item_end_page:
                    body_y_min = item_sy + _GUARD
                    body_y_max = item_ey
                elif target_page == item_start_page:
                    body_y_min = item_sy + _GUARD
                    body_y_max = 100.0
                elif target_page == item_end_page:
                    body_y_min = _REM_Y_MIN
                    body_y_max = item_ey
                else:
                    body_y_min = _REM_Y_MIN
                    body_y_max = _REM_Y_MAX

                if body_y_min >= body_y_max:
                    log.info(
                        "enrich_remarks: skip ann y=%.0f page=%s — no body room",
                        ann_y, target_page,
                    )
                    continue

                ann_y_clamped = max(body_y_min, min(body_y_max, ann_y))
                log.info(
                    "enrich_remarks: new remark ann_y=%.0f → clamped=%.0f page=%s body[%.0f-%.0f]",
                    ann_y, ann_y_clamped, target_page, body_y_min, body_y_max,
                )

                try:
                    y1 = max(_REM_Y_MIN, ann_y_clamped - 5.0)
                    y2 = min(_REM_Y_MAX, ann_y_clamped + 5.0)
                    if y2 - y1 < _REM_MIN_H:
                        y2 = min(_REM_Y_MAX, y1 + _REM_MIN_H)
                    remarks.append({
                        "page": target_page,
                        "x1_pct": _REM_RIGHT_X1,
                        "y1_pct": round(y1, 2),
                        "x2_pct": _REM_RIGHT_X2,
                        "y2_pct": round(y2, 2),
                        "comment": str(ann.get("comment", "")),
                        "text_y1_pct": round(ann_y_clamped - 2.0, 2),
                        "text_y2_pct": round(ann_y_clamped + 2.0, 2),
                        "connector_type": "arrow",
                        "connector_x_pct": 73.0,
                        "connector_cy_pct": round(ann_y_clamped, 2),
                    })
                except (TypeError, ValueError):
                    pass

        item["remarks"] = _deoverlap_item_remarks(remarks)


# --- Public API ------------------------------------------------------------------------


def _nudge_mark_past_remarks(item: dict[str, Any]) -> None:
    """If the score mark at marking_y overlaps a right-margin remark, push it below."""
    my = float(item.get("marking_y_position_percent", 0.0))
    mp = int(item.get("marking_page", item.get("start_page", 1)))
    remarks = item.get("remarks") or []
    right = sorted(
        [r for r in remarks
         if r.get("page") == mp and float(r.get("x1_pct", 0)) >= 70.0],
        key=lambda r: r["y1_pct"],
    )
    for r in right:
        y1, y2 = float(r["y1_pct"]), float(r["y2_pct"])
        if y1 <= my < y2:
            my = y2 + 1.0
    item["marking_y_position_percent"] = round(min(my, 95.0), 2)


def _debug_render_annotations(
    grid_pages: list[bytes],
    items: list[dict[str, Any]],
    request_id: str,
) -> None:
    """Draw final anchor marks + remark boxes onto saved grid images for visual verification.

    Saves `temp/{request_id}_p{n}_annotated.png` for every page that has at least one mark.
    Color key:
      underline → red line       circle/ellipse → green oval
      tick       → blue dot      remark box     → orange rectangle
    """
    try:
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont

        _temp_dir = Path("temp")

        # Build page → PIL Image map from the in-memory grid_pages
        page_images: dict[int, Any] = {}
        for idx, page_bytes in enumerate(grid_pages):
            import io as _io
            page_images[idx + 1] = _Image.open(_io.BytesIO(page_bytes)).convert("RGB")

        dirty: set[int] = set()

        for item in items:
            for am in item.get("anchor_marks") or []:
                pg = int(am.get("page", 1))
                img = page_images.get(pg)
                if img is None:
                    continue
                W, H = img.width, img.height
                draw = _ImageDraw.Draw(img)
                cx = am["cx_pct"] / 100.0 * W
                cy = am["cy_pct"] / 100.0 * H
                rx = max(4.0, am["rx_pct"] / 100.0 * W)
                ry = max(4.0, am["ry_pct"] / 100.0 * H) if am.get("ry_pct", 0) else 4.0
                t = am["type"]
                if t == "underline":
                    draw.line([(cx - rx, cy), (cx + rx, cy)], fill=(220, 20, 20), width=3)
                elif t in ("circle", "ellipse"):
                    draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)],
                                 outline=(0, 160, 0), width=2)
                elif t == "tick":
                    r = 8
                    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                                 fill=(30, 80, 220))
                dirty.add(pg)

            for rem in item.get("remarks") or []:
                pg = int(rem.get("page", 1))
                img = page_images.get(pg)
                if img is None:
                    continue
                W, H = img.width, img.height
                draw = _ImageDraw.Draw(img)
                x1 = rem["x1_pct"] / 100.0 * W
                y1 = rem["y1_pct"] / 100.0 * H
                x2 = rem["x2_pct"] / 100.0 * W
                y2 = rem["y2_pct"] / 100.0 * H
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 120, 0), width=2)
                # Draw connector anchor point
                conn_x = rem.get("connector_x_pct", 73.0) / 100.0 * W
                ty1 = rem.get("text_y1_pct", rem["y1_pct"]) / 100.0 * H
                ty2 = rem.get("text_y2_pct", rem["y2_pct"]) / 100.0 * H
                draw.line([(x1, (ty1 + ty2) / 2), (conn_x, (ty1 + ty2) / 2)],
                          fill=(255, 180, 0), width=1)
                dirty.add(pg)

        import io as _io2
        for pg in dirty:
            img = page_images[pg]
            buf = _io2.BytesIO()
            img.save(buf, format="PNG", optimize=False)
            out = _temp_dir / f"{request_id}_p{pg}_annotated.png"
            out.write_bytes(buf.getvalue())
            log.info("debug_render: saved %s", out)

    except Exception as exc:  # never crash the main pipeline
        log.warning("debug_render: failed — %s", exc)


def smart_ocr_extract_student_answers(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Classify pages → type-aware OCR → dedupe → structure into items.

    Cover / intro segregation is done in the structure (phase 2) Gemini pass — not here.

    Response shape unchanged for HTTP layer: ``{{"items": [...], "page_count": N}}``.
    Each item includes legacy position fields plus ``section_name`` and ``answer_type``.
    """
    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    # OCR quality needs 300 DPI; grid overlay (for annotation coords) works at any DPI.
    dpi = max(300, copy_ocr_raster_dpi())
    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=dpi, request_id=request_id)
    max_workers = max(1, min(copy_ocr_parallel_workers(), total_pages))

    # Draw the 50×50 reference grid onto every page (fast parallel PIL compositing).
    grid_pages = batch_draw_grid(png_pages, max_workers=max_workers)
    log.info("smart_ocr[%s] grid overlay drawn dpi=%s pages=%s", request_id, dpi, total_pages)

    # # Save grid images for inspection.
    # _temp_dir = Path("temp")
    # _temp_dir.mkdir(exist_ok=True)
    # for _i, _gp in enumerate(grid_pages):
    #     _out = _temp_dir / f"{request_id}_p{_i + 1}.png"
    #     _out.write_bytes(_gp)
    # log.info("smart_ocr[%s] grid images saved to temp/ (%s files)", request_id, len(grid_pages))

    # --- Stage 1: OCR + classify+annotate submitted simultaneously ---
    # classify+annotate uses grid PNG → page_type + anchor_marks + remarks in one call.
    # OCR uses plain PNG for text quality; waits briefly for page_type to short-circuit
    # DUPLICATE pages and avoid wasting OCR tokens.
    page_blocks: list[str] = [""] * total_pages
    page_anchor_marks: list[list[dict[str, Any]]] = [[] for _ in range(total_pages)]
    page_remarks: list[list[dict[str, Any]]] = [[] for _ in range(total_pages)]
    classifications: list[str] = ["UNKNOWN"] * total_pages
    # Per-page event set the moment classify+annotate stores its result.
    classify_events: list[threading.Event] = [threading.Event() for _ in range(total_pages)]
    _CLASSIFY_WAIT_TIMEOUT = 12.0  # seconds; upper bound for classify+annotate round-trip

    def _classify_annotate_job(idx: int) -> tuple[int, dict[str, Any]]:
        pno = idx + 1
        try:
            result = _classify_and_annotate_page(
                api_key, grid_pages[idx], pno, total_pages, language
            )
            classifications[idx] = result["page_type"]
            log.info(
                "smart_ocr[%s] classify+annotate page=%s/%s type=%s anchors=%s remarks=%s",
                request_id, pno, total_pages, result["page_type"],
                len(result["anchor_marks"]), len(result["remarks"]),
            )
        except Exception as e:
            log.warning("smart_ocr[%s] classify+annotate failed page=%s: %s", request_id, pno, e)
            result = {"page_type": "UNKNOWN", "anchor_marks": [], "remarks": []}
            classifications[idx] = "UNKNOWN"
        finally:
            classify_events[idx].set()
        return idx, result

    def _ocr_job(idx: int) -> tuple[int, str]:
        pno = idx + 1
        # Wait for classification so DUPLICATE pages can skip the OCR Gemini call.
        classify_events[idx].wait(timeout=_CLASSIFY_WAIT_TIMEOUT)
        pt = classifications[idx]
        if pt == "DUPLICATE" and idx > 0:
            header = f"=== PAGE {pno} ==="
            block = (
                f"{header}\n[OMITTED: page classified DUPLICATE — duplicate of earlier sheet; "
                f"OCR skipped to save tokens.]\n"
            )
            log.info("smart_ocr[%s] ocr page=%s DUPLICATE skipped", request_id, pno)
            return idx, block
        block = _ocr_single_page(api_key, png_pages[idx], pno, total_pages, language, pt)
        log.info(
            "smart_ocr[%s] ocr page=%s/%s class=%s chars=%s",
            request_id, pno, total_pages, pt, len(block),
        )
        return idx, block

    # 2× workers: N classify+annotate + N OCR jobs all overlap in the same pool.
    with ThreadPoolExecutor(max_workers=max_workers * 2) as pool:
        cls_ann_futs = [pool.submit(_classify_annotate_job, i) for i in range(total_pages)]
        ocr_futs     = [pool.submit(_ocr_job, i) for i in range(total_pages)]
        for fut in as_completed(cls_ann_futs):
            idx, result = fut.result()
            anchors_pct, remarks_pct = _annotations_grid_to_pct(
                result["anchor_marks"], result["remarks"], idx + 1
            )
            page_anchor_marks[idx] = anchors_pct
            page_remarks[idx] = _spread_remarks_two_column(remarks_pct, page_num=idx + 1)
        for fut in as_completed(ocr_futs):
            idx, block = fut.result()
            page_blocks[idx] = block

    # Deduplicate near-identical neighboring pages (unchanged logic, uses text only).
    page_blocks = _deduplicate_page_texts(page_blocks, classifications, request_id)

    log.info(
        "smart_ocr[%s] after ocr+dedup total_chars=%s",
        request_id,
        sum(len(b) for b in page_blocks),
    )

    # --- Stage 2: build per-page OCR JSON payload for structure pass (text-only, unchanged) ---
    pages_payload = {"pages": []}
    for i, block in enumerate(page_blocks):
        pages_payload["pages"].append({"page": i + 1, "text": _extract_page_body(block)})
    pages_payload_json = json.dumps(pages_payload, ensure_ascii=False)
    payload_tokens_est = len(pages_payload_json) // 4
    expected_questions = _estimate_expected_questions(page_blocks)
    log.info(
        "smart_ocr[%s] structure input_tokens_est=%s output_cap=%s expected_questions=%s",
        request_id, payload_tokens_est, 65536, expected_questions,
    )

    # --- Stage 3: structure from OCR JSON payload (unchanged) ---
    structure_client = genai.Client(api_key=api_key)
    rows = _structure_qa_with_fallback(
        structure_client,
        pages_payload,
        language,
        total_pages,
        expected_questions=expected_questions,
    )

    log.info(
        "smart_ocr[%s] stage3 structure complete questions=%s",
        request_id, len(rows),
    )

    # --- Merge per-page anchor_marks and remarks into items ---
    # Collect all annotations whose page falls within [item.start_page, item.end_page].

    def _clip_marks_to_answer(
        marks: list[dict[str, Any]],
        start_page: int,
        end_page: int,
        start_y: float,
        end_y: float,
        y_key: str = "cy_pct",
    ) -> list[dict[str, Any]]:
        """Keep only marks whose page+y fall within the student's answer range.

        - page outside [start_page, end_page]          → exclude
        - start_page < page < end_page                 → keep (full page is answer body)
        - page == start_page == end_page               → keep if y in [start_y, end_y]
        - page == start_page (multi-page, first page)  → keep if y >= start_y
        - page == end_page   (multi-page, last page)   → keep if y <= end_y
        """
        result: list[dict[str, Any]] = []
        for m in marks:
            pg = m.get("page", 0)
            if pg < start_page or pg > end_page:
                continue
            if start_page < pg < end_page:
                result.append(m)
                continue
            if y_key == "cy_pct":
                y = float(m.get("cy_pct", 50.0))
            else:
                # Use the CENTER of the remark box so boundary remarks are handled
                # correctly — a box that straddles start_y is included only if its
                # center is inside the answer range.
                y1_r = float(m.get("y1_pct", 0.0))
                y2_r = float(m.get("y2_pct", y1_r))
                y = (y1_r + y2_r) / 2.0
            if pg == start_page and pg == end_page:
                if start_y <= y <= end_y:
                    result.append(m)
            elif pg == start_page:
                if y >= start_y:
                    result.append(m)
            else:
                if y <= end_y:
                    result.append(m)
        return result
    all_anchors: list[dict[str, Any]] = []
    all_remarks: list[dict[str, Any]] = []
    for idx in range(total_pages):
        all_anchors.extend(page_anchor_marks[idx])
        all_remarks.extend(page_remarks[idx])

    for item in rows:
        sp  = int(item.get("start_page", 1))
        ep  = int(item.get("end_page", sp))
        sy  = float(item.get("start_y_position_percent", 0.0))
        ey  = float(item.get("end_y_position_percent", 100.0))
        qid = item.get("question_id", "?")

        # Build per-page bounding description for logging
        page_spans: list[str] = []
        for pg in range(sp, ep + 1):
            if sp == ep:
                page_spans.append(f"p{pg}[y{sy:.0f}-{ey:.0f}%]")
            elif pg == sp:
                page_spans.append(f"p{pg}[y{sy:.0f}-100%]")
            elif pg == ep:
                page_spans.append(f"p{pg}[y0-{ey:.0f}%]")
            else:
                page_spans.append(f"p{pg}[y0-100%]")
        log.info("item[q%s] bbox  %s", qid, " ".join(page_spans))

        item["anchor_marks"] = _clip_marks_to_answer(
            all_anchors, sp, ep, sy, ey, y_key="cy_pct"
        )
        # Push the effective start_y past the printed question-text zone.
        # Remarks within _REMARK_ANSWER_GUARD_PCT of start_y land on the question
        # header, not on student handwriting — exclude them.
        _REMARK_ANSWER_GUARD_PCT = 8.0
        remark_sy = min(sy + _REMARK_ANSWER_GUARD_PCT, ey)
        item["remarks"] = _clip_marks_to_answer(
            all_remarks, sp, ep, remark_sy, ey, y_key="center"
        )

        log.info(
            "item[q%s] anchors=%s",
            qid,
            " ".join(
                f"p{a['page']}({a['type']}cy={a.get('cy_pct',0):.0f}%)"
                for a in item["anchor_marks"]
            ) or "(none)",
        )
        log.info(
            "item[q%s] remarks=%s",
            qid,
            " ".join(
                f"p{r['page']}y[{r.get('y1_pct',0):.0f}-{r.get('y2_pct',0):.0f}%]"
                for r in item["remarks"]
            ) or "(none)",
        )

        _nudge_mark_past_remarks(item)

    # _debug_render_annotations(grid_pages, rows, request_id)
    return {"items": rows, "page_count": total_pages}
