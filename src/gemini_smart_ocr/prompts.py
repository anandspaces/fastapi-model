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
    return f"""You are structuring a handwritten exam answer book from OCR text only.

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
