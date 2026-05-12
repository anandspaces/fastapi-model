"""Prompt builders for the smart-OCR Gemini steps."""

from __future__ import annotations


# --- Step 2: per-page OCR + Q&A extraction (intro flag on page 1) ---------------------


def build_step2_page_prompt(
    page_num: int,
    total_pages: int,
    language: str,
    *,
    is_first_page: bool,
) -> str:
    lang = (language or "en").strip().lower()
    script_note = (
        "Preserve Hindi (Devanagari) and English exactly — do NOT transliterate."
        if lang == "hi"
        else "Preserve English and Hindi exactly — do NOT transliterate."
    )
    intro_clause = (
        "STEP A — INTRO CHECK. This IS page 1. Set is_intro=true iff the page is a "
        "cover / instructions / roll-number panel / blank rough-work sheet with NO "
        "graded student handwriting. Otherwise set is_intro=false. "
        if is_first_page else
        "STEP A — INTRO CHECK. This is NOT page 1; always set is_intro=false. "
    )
    return f"""You are processing page {page_num} of {total_pages} from a handwritten exam booklet.
A 50×50 reference grid is composited on the image (columns 1–50 left→right, rows 1–50 top→bottom).

{intro_clause}When is_intro=true, return empty text "" and questions=[].

STEP B — OCR. Transcribe ALL visible handwriting on this page into ``text``:
- Preserve line breaks (\\n) and paragraph breaks.
- Keep printed labels like Q1, (1), प्रश्न 1, उत्तर exactly.
- For tables/lists, keep one row per line; use " | " between cells when needed.
- Use [illegible] only for unreadable spans.
- Do not invent, translate, or correct content.

STEP C — Per-page Q&A records. List every question that appears on THIS page.

A "TOP-LEVEL question" MUST have a printed numeric label that you can SEE on this page —
one of these forms (case-insensitive, allow surrounding punctuation / mangled OCR):
  Q1, Q.1, Q:1, Que1, Que:1, Que.1, Quel:1, प्रश्न 1, प्र.1, (1), 1.

If you see a label, copy it verbatim (including the digit) into ``question_label`` —
DO NOT strip the prefix. Even mangled forms like "Quel>1" or "Q1." must be kept verbatim.
Set ``is_continuation=false``.

If you do NOT see a printed numeric label on this page but handwriting IS present,
this page is a continuation of a question whose label was on an earlier page.
Emit exactly ONE record with ``question_label=""``, ``question_text=""``, and
``is_continuation=true``. The ``student_answer`` field holds all the handwriting
on this page for that continuation.

CRITICAL — these are NEVER new questions; they are continuations of the prior question:
  * sub-section headings written by the student like "In administration :-",
    "Today's society :-", "Conclusion :-", "Examples :-", "Significance :-"
  * numbered list points inside an answer (1./2./3., (a)/(b)/(c))
  * case-study sub-parts (a)(b)(c) under the same printed Que:N
  * any line of handwriting on a page with no printed Q-label
For all of these set ``is_continuation=true`` and put the text into ``student_answer``.

- ``student_answer`` is the handwriting transcript for THIS question on THIS page only
  (empty string only if blank).
- ``start_y_position_percent`` = top of the question (or its continuation) on this page (0–100).
- ``end_y_position_percent`` = bottom of the student handwriting on this page.
- ``answer_type`` ∈ {{correction, paragraph, word_list}}.
- If the page contains no handwriting at all (rough work / blank), questions=[].

{script_note}

Return ONLY valid JSON matching the response schema."""


# --- Step 3: per-question-id grading + remarking + anchor remarking -------------------


def build_step3_item_prompt(
    *,
    language: str,
    subject: str,
    teacher_instructions: str,
    item: dict,
    valid_pages: list[int],
    check_level: str,
    strictness_line: str,
) -> str:
    lang = (language or "en").strip().lower()
    script_note = (
        "Comments may be in Hindi or English; preserve student wording verbatim in summaries."
        if lang == "hi"
        else "Comments should be in concise English unless the student wrote in Hindi."
    )
    qid = item.get("question_id", "?")
    qtext = (item.get("question") or "").strip()
    sans = (item.get("student_answer") or "").strip() or "[BLANK]"
    sp = int(item.get("start_page", 1))
    ep = int(item.get("end_page", sp))
    sy = float(item.get("start_y_position_percent", 0.0))
    ey = float(item.get("end_y_position_percent", 100.0))
    pages_csv = ", ".join(str(p) for p in valid_pages)

    return f"""You are an expert UPSC examiner marking ONE answer. {strictness_line}

SUBJECT: {subject}
CHECK LEVEL: {check_level}

TEACHER MODEL ANSWER + INSTRUCTIONS (the source of truth — use this to grade and to
ground every comment in what is missing / wrong / well done):
{teacher_instructions}

THIS QUESTION (Q{qid}):
{qtext or '[no printed question text captured]'}

STUDENT'S HANDWRITTEN ANSWER (OCR transcript spanning pages {sp}–{ep},
y={sy:.1f}%–{ey:.1f}% on those pages):
\"\"\"
{sans}
\"\"\"

YOU WILL RECEIVE ONE GRID-OVERLAID IMAGE PER PAGE in this answer's range. Each image
is preceded by a "--- PAGE N ---" header. Valid pages for this item: {pages_csv}.
Every anchor_mark and remark you emit MUST include the correct ``page`` value from
that list. Grid coordinates: 1–50 left→right (cx, x1, x2), 1–50 top→bottom (cy, y1, y2).

OUTPUT EXACTLY ONE JSON OBJECT MATCHING THE SCHEMA, WITH THESE PARTS:

1) GRADING — fill ``marks_awarded``, ``max_marks`` (from the TEACHER block above for
   THIS question; never invent a different cap), ``status`` ∈ {{correct, partial, wrong, unattempted}},
   ``feedback`` (specific — name what is missing or wrong vs. the model answer), and
   ``student_answer_summary`` (1–3 sentences).

2) ANCHOR MARKS (4–10 total, distributed across the item's pages):
   - UNDERLINE (mandatory, 4–6 per page that has student handwriting). On paragraph
     answers, spread underlines across body lines — do not cluster.
     cy = baseline grid row of the underlined phrase; cx ∈ [8, 34];
     rx = half-width of the phrase in grid units, (cx-rx) ≥ 6 and (cx+rx) ≤ 36; ry = 0.
   - TICK ✓ next to a correct claim or correct list item. cx ∈ [5, 8] for left-margin
     ticks, or near the end of an answer line (cx ∈ [30, 36]). rx = ry = 0.
   - ELLIPSE around a 2–4 word correct phrase. cx ∈ [8, 34], ry ≤ 1 (tight horizontal).

3) REMARKS (one per non-trivial body band):
   - Place ONLY in free zones:
     * RIGHT margin : x1=39, x2=49, height 2–3 grid rows (use for odd-numbered bands).
     * LEFT margin  : x1=1,  x2=11, height 2–3 grid rows (use for even-numbered bands).
     * INLINE GAP   : x1=5,  x2=37, spanning a blank gap between two content bands.
   - 9 ≤ y1, y2 ≤ 45. NEVER overlap a content band (margin strips excepted).
   - ``comment`` is a model-answer-grounded teacher note (≤ ~20 words) — for example
     "Missing 'Categorical Imperative' from model answer." or "Good linkage to Sarvodaya."
     Empty string only for pure visual underline-arrow boxes with no text.
   - Page rule: place the remark on the SAME page as the student writing it refers to.

4) ANNOTATIONS — mirror your remarks for the frontend. For each remark you emit, also
   emit one annotation: ``page_index = page - 1`` (zero-based), ``y_position_percent``
   ≈ midpoint of the related student writing on that page, ``x_start_percent`` and
   ``x_end_percent`` describing the in-text span being commented on (typically
   15–70% for inline content, 70–95% when the remark sits in the right margin),
   ``comment`` matches the remark, ``is_positive`` = true for praise / tick / underline
   of a correct claim, false otherwise.

{script_note}

Return ONLY JSON. All numeric coordinates are integers or floats in [1, 50] for the
grid fields and [0, 100] for the annotation percent fields."""
