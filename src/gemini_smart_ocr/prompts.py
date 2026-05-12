"""Prompt builders for the three smart-OCR Gemini stages."""

from __future__ import annotations


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
        strategy = "For CORRECTION: positive/negative per numbered item; partial for incomplete fixes."
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


def build_classify_annotation_prompt(page_num: int, total_pages: int, language: str) -> str:
    lang = (language or "en").strip().lower()
    script = "Hindi (Devanagari) and/or English may appear." if lang == "hi" \
        else "English and/or Hindi may appear."
    comment_bank = _build_remark_comment_bank(language, "UNKNOWN")

    return f"""You are a Hindi exam teacher marking handwritten page {page_num}/{total_pages}. {script}
A 50×50 grid overlays the page (columns 1–50 left→right, rows 1–50 top→bottom).

STEP 1 — page_type (pick one):
  DUPLICATE  — layout visually identical to an earlier sheet → output
               {{"page_type":"DUPLICATE","content_bands":[],"anchor_marks":[],"remarks":[]}} and stop.
  CORRECTION — short numbered sentence/line fixes.
  PARAGRAPH  — long prose, अर्थ/प्रयोग blocks.
  WORD_LIST  — word pairs, विलोम, उपसर्ग/प्रत्यय tables.
  UNKNOWN    — none fits.

STEP 2 — Layout (rows 6–45 ONLY; rows 1–8 + 46–50 = margin; cols 38–50 = right margin).
  content_bands: contiguous handwriting bands. Each = {{"y1":first_ink_row,"y2":last_ink_row}}.
    Scan rows 30–45 explicitly; the FINAL band's y2 must reach the last ink row.
  content_lines: EVERY handwriting line, top-to-bottom, one entry per line.
    Each = {{"y":ink_baseline_row,"x1":left_ink_col,"x2":right_ink_col}}.

STEP 3 — Anchor marks (6–10 total; distribute top-to-bottom, none above row 13).
  UNDERLINE (mandatory, 4–6 per page; on PARAGRAPH spread across body, not clustered).
    cy = bottom edge of word ink (baseline). cx ∈ [8,34].
    rx = half-width of phrase in grid units; (cx−rx)≥6 and (cx+rx)≤36. ry = 0.
  TICK ✓ — next to correct items only (skip if none correct).
    cx ∈ [5,8] for left-margin ticks, or at end of answer line. rx = ry = 0.
    WORD_LIST → one per correct pair row. CORRECTION → one per correct numbered line.
    PARAGRAPH → one beside a strong concluding sentence.
  ELLIPSE — multi-word correct phrase (2–4 words). ry ≤ 1 (tight horizontal). cx ∈ [8,34].
  Per page-type emphasis: WORD_LIST→ticks+ellipses; CORRECTION→ticks+underlines;
    PARAGRAPH→dense underlines + a few ellipses; UNKNOWN→≥4 underlines + 1 ellipse.

STEP 4 — Remark boxes (max(4, num_content_bands) total; one per band).
{comment_bank}
  Geometry: WIDE STRIPS — height (y2−y1) = 2–3 rows, width (x2−x1) ≥ 10.
    9 ≤ y1, y2 ≤ 45. Never overlap a content_band (margin strips excepted).
  Three placement zones — alternate to spread the page:
    RIGHT margin : x1=39, x2=49 (use for odd-numbered bands).
    LEFT margin  : x1=1,  x2=11 (use for even-numbered bands).
    INLINE GAP   : x1=5,  x2=37 (spans the blank gap between two bands).
  Distribution suggestion: bands 9–20→RIGHT, 20–30→LEFT/INLINE, 30–38→RIGHT, 38–45→LEFT.
  Comment: pick the closest fit from COMMENT BANK, ≤ 4 words, match the y-position text.
    Never output comment="".

All coordinates are integers in [1, 50]."""


# --- Stage 1B: type-aware OCR prompts --------------------------------------------------


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


def ocr_prompt_for_page_type(page_type: str, page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    if page_type == "CORRECTION":
        return f"""{base}

FOCUS (CORRECTION style): Prefer one line per numbered response where possible.
If numbering is visible, preserve it exactly."""
    if page_type == "PARAGRAPH":
        return f"""{base}

FOCUS (PARAGRAPH style): Keep question numbers/labels visible before each answer block."""
    if page_type == "WORD_LIST":
        return f"""{base}

FOCUS (WORD_LIST style): Keep one pair/row per line.
Preserve column order and separators."""
    return base


# --- Stage 2: structure prompt --------------------------------------------------------


def build_structure_prompt(language: str, total_pages: int, expected_questions: int | None) -> str:
    lang_note = (
        "Text is Hindi/English mixed; preserve student_answer wording verbatim."
        if (language or "en").strip().lower() == "hi"
        else "Text may be Hindi/English mixed; preserve student_answer wording verbatim."
    )
    expected_hint = (
        f"Paper has ~{expected_questions} questions — extract all of them, do not stop early.\n"
        if expected_questions else ""
    )
    return f"""You convert OCR text of a handwritten exam booklet into structured question rows.
Input shape: {{"pages":[{{"page":N,"text":"..."}}, ...]}} with N in 1..{total_pages}.

IGNORE intro / cover / admin pages (roll boxes, instructions, "do not write in margin",
rough-work areas, invigilator notes). Do NOT invent question rows from cover boilerplate.

Identify TOP-LEVEL questions only (labels like Q1, Q:2, Que:3, Que 4, Q.5, प्रश्न 1, (1)).
DO NOT split: sub-parts of a case study (1./2./3. or (a)(b)(c) under one Que:N),
cross-page continuations, or numbered points inside an answer.
Case study with sub-parts → ONE item: full stem + all sub-part labels in ``question``;
sub-answers joined with "\\n\\n" in ``student_answer``.

Per-question fields:
- question_id: integer FROM the printed label (Que:7→7, प्रश्न 5→5). Use printed numbering,
  never substitute sequential 1,2,3. Omit a row entirely if the question never appears in OCR.
- question: full question text including sub-part labels.
- student_answer: candidate handwriting only (do NOT paste the printed stem). Empty string ""
  if blank/unattempted — still emit the row to keep numbering aligned. No grader commentary.
- answer_type: correction | paragraph | word_list.
- start_page, start_y_position_percent (0–100 from page top): top of the question header.
- end_page, end_y_position_percent: bottom edge of handwritten ink for this question.
- marking_page, marking_x_position_percent, marking_y_position_percent: anchor for examiner
  remark, placed near actual handwriting (not on printed text/header/footer). Adapt per page.

{lang_note}
{expected_hint}
Return one JSON object: {{"sections":[{{"section_name":"...","questions":[...]}}]}}.
Use \\n inside strings, escape " as \\". Section names in Hindi or short English.
"""
