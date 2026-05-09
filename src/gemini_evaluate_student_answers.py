"""Stage 3: grade student OCR items against a teacher answer model (Dart-parity)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from src.gemini_extract import MODEL_ID

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher instruction builder
# ---------------------------------------------------------------------------


def format_answer_model_as_teacher_instructions(
    questions: list[dict[str, Any]],
    title: str,
) -> str:
    """Serialize a model's questions list into the prompt block the evaluator reads.

    Each question dict shape (from QuestionPayload / SQLite JSON):
      questionNo, title, desc, instruction_name, pageNum, marks, diagramDescriptions
    """
    lines: list[str] = [f"Subject / Paper: {title or 'General'}", ""]
    for q in questions:
        qno = q.get("questionNo") or q.get("question_no") or "?"
        qlabel = str(qno).strip()
        if qlabel != "?" and not qlabel.upper().startswith("Q"):
            qlabel = f"Q{qlabel}"
        qtitle = (q.get("title") or "").strip()
        desc = (q.get("desc") or "").strip()
        instr_name = str(q.get("instruction_name") or "").strip()
        marks = q.get("marks") or 0
        diagrams = q.get("diagramDescriptions") or []

        lines.append(f"{qlabel}. {qtitle}")
        if desc:
            lines.append(f"   Model booklet (ideal answer): {desc}")
        if instr_name:
            lines.append(
                f"   Instructions (examiner marking key — weigh like the booklet text): "
                f"{instr_name}"
            )
        if diagrams:
            lines.append(f"   Diagrams/Key points: {'; '.join(str(d) for d in diagrams)}")
        lines.append(f"   MAX MARKS ALLOWED: {marks}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_CHECK_LEVEL_ALLOWED = frozenset({"moderate", "hard"})


def normalize_evaluation_check_level(raw: str) -> str:
    """Return canonical ``Moderate`` or ``Hard``, or empty string if invalid (caller rejects)."""
    s = (raw or "Moderate").strip().lower()
    if s not in _CHECK_LEVEL_ALLOWED:
        return ""
    return "Hard" if s == "hard" else "Moderate"


def evaluation_strictness_instruction(check_level: str) -> str:
    """Same semantics as ``GeminiService`` Moderate vs Hard grading lines."""
    if (check_level or "").strip().lower() == "hard":
        return (
            "- **EVALUATION STRICTNESS: HARD.** Be extremely strict. All answers must be "
            "strictly evaluated and normally score less than 50% of the total marks unless "
            "they are absolutely perfect without any flaws."
        )
    return (
        "- **EVALUATION STRICTNESS: MODERATE.** Grade normally, but keep medium or average "
        "answers around or below 50% of the total marks."
    )


def _build_evaluation_prompt_with_overlay(
    subject: str,
    teacher_instructions: str,
    student_json: str,
    *,
    check_level: str = "Moderate",
) -> str:
    """Cell-overlay grading prompt — Gemini sees the per-page raster with every
    cell ID printed inside it, and emits cell-ID-native placement directly.

    Used when ``SMART_OCR_OVERLAY_PROMPT=1``. Companion to
    :func:`_build_evaluation_prompt` (the legacy percent-coord prompt) — both
    coexist while the new path is dialed up.

    The caller MUST attach the overlay images (one ``image/jpeg`` Part per
    page, page-1 first) to the multimodal payload after this prompt's text
    Part. The prompt body references them as the spatial vocabulary.
    """
    strict_line = evaluation_strictness_instruction(check_level)
    return f"""You are a strict but fair {subject} examiner-mentor (UPSC Civil Services / mains-answer ethos). Your feedback reads like seasoned script evaluation — professional, restrained, clinically useful — never cheerleading.

You receive:
  1. TEACHER INSTRUCTIONS / MODEL KEY for each question.
  2. STUDENT'S OCR-EXTRACTED ANSWERS.
  3. CELL-OVERLAY IMAGES — one per PDF page, attached after this message in
     page order. Each is the original answer sheet with a cell grid drawn over
     it. Every cell shows its cell ID (Excel-style: A1, B1, …, Z1, AA1, AB1, …).
     Cells tinted GREEN are writable (no handwriting underneath). White cells
     contain handwriting or printed content.

Your job: grade the student against the teacher key, place examiner-style
annotations directly onto the cell grid, and emit one structured row per
teacher question.

═════════════════════════════════════════════════════════════════════════
MARKING RULES
═════════════════════════════════════════════════════════════════════════
{strict_line}
- max_marks MUST exactly match "MAX MARKS ALLOWED" in TEACHER INSTRUCTIONS.
- Booklet vs instruction_name: when a question includes "Instructions
  (examiner marking key)" via instruction_name, factor that into gaps,
  positives, marks, feedback, and annotations.
- Award marks_awarded as a DECIMAL in multiples of 0.5.
- NEVER exceed MAX MARKS. Perfect answer → exactly the max.
- Conceptual flexibility: do not blindly string-match; award full marks if
  underlying logic is identical.
- FEEDBACK TONE: understated, examiner-like — name one concrete merit only
  if warranted, foreground exact gaps. Forbidden flattery phrases
  ('Excellent', 'Outstanding', 'Brilliant'). 25–40 words.

═════════════════════════════════════════════════════════════════════════
ANNOTATION TONE (per-annotation `comment`)
═════════════════════════════════════════════════════════════════════════
- Mentor script: precise, corrective, syllabus-aware — cite what would earn
  the next increment of marks.
- Use is_positive: true SPARINGLY (only distinctive merit).
- Substantive answers should skew toward developmental comments, not applause.

═════════════════════════════════════════════════════════════════════════
SPATIAL VOCABULARY  (every spatial output is a list of single-row ranges)
═════════════════════════════════════════════════════════════════════════
- Cell IDs come straight off the overlay raster you are looking at.
- A "row range" covers consecutive cells on ONE row, written as
  "<start>:<end>", e.g. "E13:S13" means columns E through S on row 13.
- Multi-row content is a LIST of row ranges, one per row, each spanning
  exactly the cells that row contains:
        ["E13:S13", "E14:T14", "E15:Q15"]
  This faithfully captures ragged right edges — do NOT collapse rows
  into one big rectangle.
- Coordinates do not cross pages; each annotation belongs to exactly one
  page.

YOU OUTPUT ONLY CELL IDS. No percent or pixel coordinates anywhere.

═════════════════════════════════════════════════════════════════════════
ANSWER_SPAN  →  per-page list of row ranges where the student's ink sits
═════════════════════════════════════════════════════════════════════════
Walk row-by-row down each page. For every row that contains the student's
handwriting (NOT printed question text, NOT margin rules), emit one row
range covering the cells from the leftmost inked cell to the rightmost
inked cell. Skip blank rows between paragraphs.

═════════════════════════════════════════════════════════════════════════
COMMENT PLACEMENT  →  comment_rows
═════════════════════════════════════════════════════════════════════════
- comment_rows is a LIST of single-row ranges, one per row of the comment.
- Every cell in every range MUST be GREEN (writable) on the overlay.
- Place the comment NEXT TO the handwriting it critiques — usually in the
  LEFT margin (cols A–D) or the RIGHT margin (last few columns) of the same
  row band as the relevant answer rows.
- Comment text wraps row by row.

Capacity (HomemadeApple at comment_font_pts, given cell side L pts):
    chars_per_cell  ≈ floor(L / 6.5)
    lines_per_cell  ≈ 1 (for L ≤ 24)

NO TWO annotations on the same page may share any cell across their
comment_rows.

Examples:
  "comment_rows": ["A13:D13", "A14:D14", "A15:D15", "A16:D16"]   // left margin
  "comment_rows": ["U25:X25", "U26:X26", "U27:X27"]              // right margin

═════════════════════════════════════════════════════════════════════════
COMMENT FONT  →  comment_font_pts
═════════════════════════════════════════════════════════════════════════
Float in [11.0, 15.0]. Default 12.0. Larger only for critical points where
comment_rows can absorb the larger text.

═════════════════════════════════════════════════════════════════════════
ANCHOR MARK  →  anchor.type, anchor.rows, anchor.extra
═════════════════════════════════════════════════════════════════════════
Pick exactly ONE anchor mark per annotation. The mark sits ON the
student's content — anchor.rows MAY include white (non-writable) cells.

  type             rows shape                           extra (required)
  ──────────────  ───────────────────────────────────  ─────────────────────
  circle          1 row, 2–6 cols                      —
  tick            1 row, 1 cell                        —
  underline       1 row, N cols (the inked sentence)   —
  curly_brace     N rows, 1 col, in margin             side: "left"|"right"
  exponent_caret  1 row, 1 cell                        missing_word: "<word>"
  none            (omit anchor.rows entirely)          —

When to use each:
  - circle           → wrong word, misspelling, phrase to highlight
  - tick             → unambiguous merit (use SPARINGLY)
  - underline        → a sentence/clause to emphasise or correct
  - curly_brace      → paragraph-level remark spanning multiple rows
  - exponent_caret   → missing word at a specific gap
  - none             → comment without a mark (layout/structural critique)

═════════════════════════════════════════════════════════════════════════
MARKING  (the score circle and the marking strip)
═════════════════════════════════════════════════════════════════════════
"marking": {{
  "page": N,
  "score_range": "AC18:AE19",       // small rect of cells the score circle wraps
  "score_box_range": "Y17:AH20"     // outer rect, the marking strip / stamp area
}}
score_range MUST sit inside score_box_range.

═════════════════════════════════════════════════════════════════════════
ANNOTATION DENSITY
═════════════════════════════════════════════════════════════════════════
- Substantive answer (~6–8+ lines or multi-paragraph): AT LEAST 4 annotations.
  Cover three structural zones when each exists:
    1) Opening / introduction
    2) Body / core evidence
    3) Conclusion / synthesis
- Short answer (≤3 lines): AT LEAST 2 annotations anchored to concrete phrases.
- Per-page density: 2–4 annotations on a dense page.

═════════════════════════════════════════════════════════════════════════
UNATTEMPTED QUESTIONS
═════════════════════════════════════════════════════════════════════════
status: "unattempted", marks_awarded: 0, feedback: "Question not attempted."
student_answer: "", student_answer_summary: "",
answer_span: [], marking: null, annotations: []

Emit one row per teacher question — never omit one.

═════════════════════════════════════════════════════════════════════════
INPUT
═════════════════════════════════════════════════════════════════════════
TEACHER INSTRUCTIONS / MODEL KEY ({subject}):
{teacher_instructions}

STUDENT'S EXTRACTED ANSWERS:
{student_json}

The CELL-OVERLAY IMAGES follow this prompt in page order (page 1 first).

═════════════════════════════════════════════════════════════════════════
OUTPUT  (JSON array — one object per teacher question, in order)
═════════════════════════════════════════════════════════════════════════
[
  {{
    "question_id": 1,
    "question": "...full question text...",
    "max_marks": 8,
    "marks_awarded": 4.5,
    "status": "partial",
    "student_answer_summary": "...",
    "feedback": "Insightful specific feedback (25–40 words).",

    "answer_span": [
      {{ "page": 2, "rows": [
          "E13:S13","E14:T14","E15:Q15","E16:V16","E17:V17","E18:T18",
          "F25:M25","F26:T26","F27:S27","F28:T28"
      ]}}
    ],

    "marking": {{
      "page": 2,
      "score_range": "AC18:AE19",
      "score_box_range": "Y17:AH20"
    }},

    "annotations": [
      {{
        "page": 2,
        "is_positive": false,
        "comment": "Define EI before applying it.",
        "comment_font_pts": 12.0,
        "comment_rows": ["A13:D13","A14:D14","A15:D15","A16:D16"],
        "anchor": {{ "type": "underline", "rows": ["E13:S13"] }}
      }},
      {{
        "page": 2,
        "is_positive": false,
        "comment": "Add 'self-regulation' here.",
        "comment_font_pts": 12.0,
        "comment_rows": ["P14:T15"],
        "anchor": {{
          "type": "exponent_caret",
          "rows": ["M14:M14"],
          "extra": {{ "missing_word": "self-regulation" }}
        }}
      }}
    ]
  }}
]

═════════════════════════════════════════════════════════════════════════
INVARIANTS YOUR OUTPUT MUST SATISFY
═════════════════════════════════════════════════════════════════════════
1.  Every cell ID is within its page's grid.
2.  Each entry in any "rows" array is a SINGLE-ROW range "X<n>:Y<n>"
    (start row == end row). Multi-row content is a list of these.
3.  Every cell in comment_rows is GREEN (writable) on the overlay.
4.  anchor.rows cells are valid grid cells (writability not required).
5.  marking.score_range ⊆ marking.score_box_range.
6.  comment_font_pts ∈ [11.0, 15.0].
7.  No two annotations on the same page share any cell in their comment_rows.
8.  exponent_caret has non-empty extra.missing_word.
9.  curly_brace has extra.side ∈ {{"left","right"}}.

If you cannot satisfy an invariant, prefer dropping the anchor
(type: "none") or shrinking comment_font_pts.

═════════════════════════════════════════════════════════════════════════
JSON FORMATTING
═════════════════════════════════════════════════════════════════════════
- Output ONLY a valid JSON array. No prose.
- Escape literal " as \\". Use \\n for line breaks inside strings.
- No trailing commas. No comments.
- status ∈ {{"correct","partial","wrong","unattempted"}}.
- Use the student's language for every comment.
"""


def _build_evaluation_prompt(
    subject: str,
    teacher_instructions: str,
    student_json: str,
    *,
    check_level: str = "Moderate",
) -> str:
    strict_line = evaluation_strictness_instruction(check_level)
    return f"""You are a strict but fair {subject} examiner-mentor (UPSC Civil Services / mains-answer ethos). Your feedback reads like seasoned script evaluation — professional, restrained, clinically useful — never cheerleading.

You are given:
1. TEACHER INSTRUCTIONS / MODEL KEY — for each question this includes the **model booklet (ideal answer)** and any parallel **instructions (examiner marking key)**. Both are authoritative: use the booklet as the reference script; use instructions for rubric, dimensions, quotations, scheme nuance the teacher expects enforced.
2. A STUDENT's OCR-extracted answers (student_answer) from their handwritten copy.
   Rows with completely blank answers may be omitted from this JSON — you must still emit "unattempted" with 0 marks for every TEACHER question. Do not invent OCR text in ``student_answer_summary``.

Your job: Based on the TEACHER INSTRUCTIONS, grade the student's extracted answers. Find each question's matching student answer and award marks. Match the student's answer to the specific model key question by content or question number, even if the student answered them out of order (e.g. if the student answered Q5 first, match it to Q5 in the model key). If a question is present in the TEACHER INSTRUCTIONS but missing from the student's answers, it MUST be included in the output with a status of "unattempted" and 0 marks.

MARKING RULES:
{strict_line}
- The `max_marks` in your output MUST exactly match the "MAX MARKS ALLOWED" for each question in the TEACHER INSTRUCTIONS.
- **Booklet vs instruction_name:** When a question includes markdown **Instructions (examiner marking key)** (from field ``instruction_name``), factor that content into gaps, positives, marks, `feedback`, and `annotations` the same way you use the **Model booklet (ideal answer)** — it is part of the model key, not optional commentary.
- Award `marks_awarded` as a DECIMAL in multiples of 0.5 (0, 0.5, 1, 1.5, ...).
- NEVER EXCEED the "MAX MARKS ALLOWED" for a question. If a student's answer is perfect, give exactly the MAX MARKS ALLOWED.
- CONCEPTUAL FLEXIBILITY: Evaluate with deep human intelligence! Do not blindly string-match; award full marks if the underlying meaning and logic is identical.
- FEEDBACK TONE (`feedback`): Understated, examiner-like — name one concrete merit only if warranted, foreground **exact gaps**, depth not yet delivered, sharper examples or dimensions the scheme expects. Forbidden flattery phrases (e.g. 'Excellent','Outstanding','Brilliant','Great job','Very good' as standalone hype). Aim 25–40 words unless the script is trivially short.

TONE FOR ANNOTATION COMMENTS (`comment`):
- Mentor script: precise, corrective, syllabus-aware — cite what would earn the next increment of marks.
- Do **not** over-praise listings, quotes, or basic definitions; lukewarm 'good' wastes the candidate's time.
- Use `is_positive`: true sparingly — only where a passage **clearly** adds distinctive value versus a generic script (novel linkage, nuanced ethics angle, crisp application). Otherwise prefer `false` or neutral framing that **teaches** (add dimension X, tighten Y, quantify Z).
- Aim for diagnostic balance: substantive answers should skew toward **developmental** comments, not applause.

TEACHER INSTRUCTIONS / MODEL KEY ({subject}):
{teacher_instructions}

STUDENT'S EXTRACTED ANSWERS:
{student_json}

OUTPUT EXACTLY a JSON array of objects (one per question evaluated). You MUST output exactly ONE object for EVERY question defined in the TEACHER INSTRUCTIONS. Do not omit any questions.

[
  {{
    "question_id": 1,
    "question": "The question text or summary",
    "max_marks": 5,
    "marks_awarded": 3.5,
    "status": "correct",
    "student_answer_summary": "Full text or summary of the student's answer",
    "feedback": "Insightful specific feedback...",
    "start_page": 1,
    "start_y_position_percent": 10.0,
    "end_page": 2,
    "end_y_position_percent": 85.0,
    "marking_page": 1,
    "marking_x_position_percent": 50.0,
    "marking_y_position_percent": 45.0,
    "annotations": [
      {{
        "page_index": 0,
        "y_position_percent": 50.0,
        "x_start_percent": 20.0,
        "x_end_percent": 80.0,
        "comment": "short teacher remark",
        "is_positive": true
      }}
    ]
  }}
]

IMPORTANT — COORDINATES:
- If the student's OCR JSON contains a row for that ``question_id`` (even with empty ``student_answer``), copy that row's page bounds and marking positions into your output, and add annotations only across those pages.
- Honour the layout intent encoded in marking coordinates from OCR: annotations and examiner marks belong in the handwritten answer zone — vertically below printed stem/handwriting divider and above page-foot rules; spread ``marking_y`` and annotation ``y_position_percent`` to avoid stacking on identical Y as the student's first printed line label.
- If the student's OCR JSON has **no row at all** for that ``question_id`` (question not written on the copy), set ``start_page``, ``end_page``, ``marking_page``, all ``*_position_percent`` fields, and ``marking_*_position_percent`` to JSON ``null``, and set ``annotations`` to ``[]``. Do **not** guess page 1 or other placeholders.

The "annotations" array — place **specific** examiner remarks **on** the student's content (preferred over cramming detail only into ``feedback``):

When the student wrote a substantive answer (more than ~6–8 lines or multiple paragraphs across pages):
- Emit **at least 4 annotations** total across all spanned pages (more when the answer is long or richly structured).
- You **must cover three structural zones when each exists in the handwriting** — label each implicitly in wording, not brackets:
  1) **Opening / introduction** — framing, definitions, roadmap, relevance to the question quality (positive or corrective).
  2) **Body / core evidence** — main arguments, examples, logic, omissions vs marking scheme (one or two spots).
  3) **Conclusion / synthesis** — how they closed: summary worthiness, link back to stem, takeaway; if they stop abruptly without closure, ``is_positive``: false and suggest briefly what a closing line could do.
- If the introduction or conclusion zones are visibly missing from the handwritten answer, add an annotation in the nearest appropriate band (often lower on first / last answered page) saying what is lacking.

For **short** answers (about 3 lines or less): emit **at least 2 annotations** anchored to concrete phrases.

Across pages: distribute so the **first page with significant ink** tends to carry intro-related critique and the **last page with substantive writing** tends to carry conclusion-related critique unless structure clearly contradicts.

Per page density: roughly **2–4** annotations on a page that carries dense handwriting; thinner pages proportionally fewer. Never exceed what vertical spacing allows below.

Technical: ``page_index`` MUST be **0-based** page index matching the PDF raster (first page ``0``, second ``1``). It must lie between ``start_page - 1`` and ``end_page - 1`` when those are 1-based page numbers from OCR. Use the student's language (English or Hindi) for every ``comment`` — formal, succinct, mains-appropriate wording.
- For genuinely strong, non-generic merits only, set "is_positive": true (sparse use).
- For sharpening, omission, misconception, shallow example, weak conclusion, missing stakeholder/dimension — set "is_positive": false and give one actionable examiner line (not moralising fluff).

CRITICAL JSON FORMATTING RULES:
- You must ONLY output a valid JSON array.
- DO NOT use unescaped double quotes inside string values (use ' or escape them).
- DO NOT use literal newlines inside string values (use \\n).
- Make sure every object and array is properly closed.

CRITICAL PLACEMENT RULE: When placing annotations, ensure their "y_position_percent" values are well separated (at least 15% apart) from each other AND from the final "marking_y_position_percent" on the same page. This prevents text overlap in the UI.
Status must be one of: "correct", "partial", "wrong", "unattempted".
"""


# ---------------------------------------------------------------------------
# JSON parsing + regex fallback (Dart parity)
# ---------------------------------------------------------------------------


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    return m.group(1).strip() if m else t


def _repair_truncated_json(text: str) -> str:
    """
    Best-effort repair of malformed/truncated Gemini JSON.

    Handles two failure modes observed in production:
    1. Trailing commas before } or ] (→ strip them).
    2. Output truncated mid-string/mid-object (→ find last complete object
       and close the array at that point).
    """
    # Pass 1: strip trailing commas before closing delimiters
    t = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        pass

    # Pass 2: find the end of the last fully-closed top-level JSON object.
    # Walk character-by-character tracking brace depth, skipping strings.
    depth = 0
    in_str = False
    escape = False
    last_complete_end = -1

    for i, ch in enumerate(t):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete_end = i

    if last_complete_end > 0:
        # Close the array after the last complete object and re-validate
        repaired = t[: last_complete_end + 1] + "]"
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    return t


def _parse_evaluation_response(raw: str) -> list[dict[str, Any]]:
    cleaned = _strip_json_fence(raw)
    # First attempt: strict parse
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Second attempt: repair common truncation / trailing-comma issues
        repaired = _repair_truncated_json(cleaned)
        parsed = json.loads(repaired)
    if not isinstance(parsed, list):
        raise ValueError("Evaluation response is not a JSON array.")
    return [dict(item) for item in parsed]


def _regex_extract_evaluations(text: str) -> list[dict[str, Any]]:
    """Port of Dart's _regexExtractEvaluations — last-resort fallback."""
    results: list[dict[str, Any]] = []
    blocks = re.split(r'"question_id"\s*:', text)
    for block in blocks[1:]:
        item: dict[str, Any] = {}

        m = re.match(r"^\s*([0-9]+)", block)
        if m:
            item["question_id"] = int(m.group(1))

        for field in ("question", "status", "student_answer_summary", "feedback"):
            fm = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)', block, re.DOTALL)
            item[field] = (
                fm.group(1).replace(r"\"", '"').replace(r"\n", "\n").strip()
                if fm
                else ""
            )

        for field in ("max_marks", "start_page", "end_page", "marking_page"):
            fm = re.search(rf'"{field}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', block)
            item[field] = int(float(fm.group(1))) if fm else 0

        for field in (
            "marks_awarded",
            "start_y_position_percent",
            "end_y_position_percent",
            "marking_x_position_percent",
            "marking_y_position_percent",
        ):
            fm = re.search(rf'"{field}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', block)
            item[field] = float(fm.group(1)) if fm else 50.0

        ann_re = re.compile(
            r'\{[^{}]*"page_index"\s*:\s*(\d+)[^{}]*'
            r'"y_position_percent"\s*:\s*([0-9.]+)[^{}]*'
            r'"x_start_percent"\s*:\s*([0-9.]+)[^{}]*'
            r'"x_end_percent"\s*:\s*([0-9.]+)[^{}]*'
            r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*'
            r'"is_positive"\s*:\s*(true|false)',
            re.DOTALL,
        )
        item["annotations"] = [
            {
                "page_index": int(am.group(1)),
                "y_position_percent": float(am.group(2)),
                "x_start_percent": float(am.group(3)),
                "x_end_percent": float(am.group(4)),
                "comment": am.group(5).replace(r"\"", '"').replace(r"\n", "\n"),
                "is_positive": am.group(6) == "true",
            }
            for am in ann_re.finditer(block)
        ]
        results.append(item)
    return results


# Keys produced by Stage 3 grading (merged into each OCR item; avoid duplicating
# question text / coordinates already present from Stage 2).
_GRADING_KEYS = (
    "max_marks",
    "marks_awarded",
    "status",
    "student_answer_summary",
    "feedback",
    "annotations",
    "answer_span",
    "marking",
)

# OCR layout fields — absent when this question never appeared in the structure pass output.
_ABSENT_FROM_OCR_COORD_KEYS = (
    "start_page",
    "end_page",
    "marking_page",
    "start_y_position_percent",
    "end_y_position_percent",
    "marking_x_position_percent",
    "marking_y_position_percent",
)


def _nullify_coordinates_for_absent_student_row(row: dict[str, Any]) -> None:
    """Teacher-only evaluations for questions missing from OCR have no drawable bounds."""
    for k in _ABSENT_FROM_OCR_COORD_KEYS:
        row[k] = None
    row["annotations"] = []


def _norm_question_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def item_is_attempted_for_grading(item: dict[str, Any]) -> bool:
    """True if OCR has substantive answer text worth sending to Gemini for grading."""
    if item.get("is_attempted") is False:
        return False
    return bool(str(item.get("student_answer", "") or "").strip())


def student_items_for_grading(
    student_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Subset passed to Gemini. Blank / unattempted rows are graded as unattempted without LLM fabricating feedback."""
    return [it for it in student_items if item_is_attempted_for_grading(it)]


def _normalize_annotation_page_index(raw: Any, item: dict[str, Any]) -> int:
    """Map evaluator ``page_index`` to 0-based index for ``page_free_zones[page_idx]``.

    Gemini may emit physical (1-based) page numbers or 0-based raster indices. For answers that
    span multiple sheets, overlaps (e.g. ``p`` valid as both) follow the evaluator prompt's
    0-based convention; single-sheet rows still remap ``page_index === start_page`` from 1-based.

    When ``raw`` falls outside both 0-based and 1-based valid ranges, the result is clamped
    to the nearest 0-based page within ``[start_page-1, end_page-1]``. This prevents Gemini
    page-index slips from snapping annotations onto a previous question's writing space.
    """
    try:
        p = int(raw)
    except (TypeError, ValueError):
        p = 0
    sp, ep = item.get("start_page"), item.get("end_page")
    if sp is not None and ep is not None:
        try:
            sp_i, ep_i = int(sp), int(ep)
        except (TypeError, ValueError):
            pass
        else:
            zb_lo, zb_hi = sp_i - 1, ep_i - 1
            valid_zb = zb_lo <= p <= zb_hi
            valid_ob = sp_i <= p <= ep_i
            if valid_zb and valid_ob:
                # Multi-page spans: evaluation prompt uses 0-based page_index; trust zb when both
                # conventions fit the same integer (e.g. p=1 → second sheet, index 1).
                if sp_i < ep_i:
                    return p
                mp = item.get("marking_page")
                if mp is not None:
                    try:
                        m = int(mp)
                    except (TypeError, ValueError):
                        m = None
                    else:
                        if p == m:
                            return m - 1
                        if p == m - 1:
                            return p
                if p == sp_i:
                    return p - 1
                return p
            if valid_zb:
                return p
            if valid_ob:
                return p - 1
            # Out of both ranges → clamp to closest valid 0-based answer page.
            return max(zb_lo, min(zb_hi, p))
    mp = item.get("marking_page")
    if mp is not None:
        try:
            m = int(mp)
        except (TypeError, ValueError):
            pass
        else:
            if p == m:
                return m - 1
            if p == m - 1:
                return p
    return p


def _reseed_annotations_from_ocr_coords(item: dict[str, Any]) -> None:
    """Seed annotation placement from structure-pass OCR so the snapper starts near truth.

    - Normalizes ``page_index`` to 0-based (fixes 1-based model slips).
    - On the marking page, seeds Y from ``marking_y_position_percent`` with ±spread so multiple
      annotations don't share one identical snap target.
    - Seeds horizontal band from ``marking_x_position_percent`` (narrow band; snap assigns zone X).
    """
    anns = item.get("annotations")
    if not isinstance(anns, list) or not anns:
        return

    # Smart-OCR v2 / cell-overlay path — placement is already cell-native.
    if any(
        isinstance(a, dict)
        and (
            (isinstance(a.get("comment_rows"), list) and len(a.get("comment_rows") or []) > 0)
            or (
                isinstance((a.get("anchor") or {}).get("rows"), list)
                and len((a.get("anchor") or {}).get("rows") or []) > 0
            )
        )
        for a in anns
    ):
        return

    for ann in anns:
        if isinstance(ann, dict):
            ann["page_index"] = _normalize_annotation_page_index(ann.get("page_index"), item)

    m_page = item.get("marking_page")
    m_y = item.get("marking_y_position_percent")
    m_x = item.get("marking_x_position_percent")
    if m_page is None or m_y is None:
        return
    try:
        mp = int(m_page)
        my = float(m_y)
        mx = float(m_x) if m_x is not None else 50.0
    except (TypeError, ValueError):
        return

    target_pi = mp - 1
    half_w = 15.0
    x0 = max(5.0, min(100.0 - half_w * 2, mx - half_w))
    x1 = min(95.0, max(x0 + 5.0, mx + half_w))

    on_marking = [i for i, a in enumerate(anns) if isinstance(a, dict) and int(a.get("page_index", -1)) == target_pi]
    if not on_marking:
        return

    step = 16.0
    ordered = sorted(
        on_marking,
        key=lambda i: float(anns[i].get("y_position_percent") or my),
    )
    n = len(ordered)
    mid = (n - 1) / 2.0

    for rank, i in enumerate(ordered):
        offset = (rank - mid) * step
        ny = max(6.0, min(94.0, my + offset))
        a = anns[i]
        a["y_position_percent"] = round(ny, 2)
        a["x_start_percent"] = round(x0, 2)
        a["x_end_percent"] = round(x1, 2)


def merge_evaluations_into_items(
    items: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach Stage 3 grading fields to each OCR item by ``question_id``.

    Teacher-only questions (present in ``evaluations`` but not in ``items``) are
    appended so every graded row appears once.
    """
    by_qid: dict[int, dict[str, Any]] = {}
    for ev in evaluations:
        qid = _norm_question_id(ev.get("question_id"))
        if qid is not None:
            by_qid[qid] = ev

    merged: list[dict[str, Any]] = []
    used_qids: set[int] = set()

    for item in items:
        out = dict(item)
        qid = _norm_question_id(item.get("question_id"))
        ev = by_qid.get(qid) if qid is not None else None
        if ev is not None and qid is not None:
            used_qids.add(qid)
            for k in _GRADING_KEYS:
                if k in ev:
                    out[k] = ev[k]
            _reseed_annotations_from_ocr_coords(out)
        merged.append(out)

    for qid, ev in by_qid.items():
        if qid in used_qids:
            continue
        row = dict(ev)
        if "student_answer" not in row:
            row["student_answer"] = ""
        if "is_attempted" not in row:
            st = str(row.get("status") or "").lower()
            row["is_attempted"] = st not in ("", "unattempted")
        _nullify_coordinates_for_absent_student_row(row)
        merged.append(row)

    merged.sort(
        key=lambda r: (
            _norm_question_id(r.get("question_id")) or 10**9,
        )
    )
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_student_answers_against_model(
    api_key: str,
    subject: str,
    teacher_instructions: str,
    student_items: list[dict[str, Any]],
    *,
    request_id: str,
    check_level: str = "Moderate",
    overlay_images: list[bytes] | None = None,
    use_overlay_prompt: bool = False,
) -> list[dict[str, Any]]:
    """Call Gemini to grade student OCR items against the teacher model key.

    ``check_level`` is ``Moderate`` or ``Hard`` (``checkLevel`` parity).

    ``overlay_images`` + ``use_overlay_prompt``: when both are provided, the
    cell-overlay grading prompt is used and the JPEG bytes are attached to
    the multimodal payload (one image part per page, page-1 first). Gemini
    emits cell-ID-native placement (``comment_rows``, ``anchor.rows``,
    ``answer_span[].rows``) directly. Otherwise the legacy percent-coord
    prompt runs unchanged.

    Returns a list of evaluation dicts (one per teacher question).
    Raises ValueError if all attempts fail and regex fallback is also empty.
    """
    student_json = json.dumps(student_items, ensure_ascii=False)

    if use_overlay_prompt and overlay_images:
        prompt = _build_evaluation_prompt_with_overlay(
            subject, teacher_instructions, student_json, check_level=check_level
        )
        contents: list[types.Part] = [types.Part.from_text(text=prompt)]
        for img in overlay_images:
            contents.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
        log.info(
            "evaluate[%s] using cell-overlay prompt (pages=%s, payload=%s KB)",
            request_id, len(overlay_images),
            sum(len(b) for b in overlay_images) // 1024,
        )
    else:
        prompt = _build_evaluation_prompt(
            subject, teacher_instructions, student_json, check_level=check_level
        )
        contents = [types.Part.from_text(text=prompt)]

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=65536,
        response_mime_type="application/json",
    )

    last_raw: str = ""
    last_err: Exception | None = None

    for attempt in range(1, 4):  # 3 attempts, matching Dart
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=cfg,
            )
            raw = (getattr(resp, "text", None) or "").strip()
            if not raw:
                raise ValueError("Empty response from Gemini.")
            last_raw = raw
            result = _parse_evaluation_response(raw)
            log.info(
                "evaluate[%s] attempt=%s ok evaluations=%s check_level=%s",
                request_id,
                attempt,
                len(result),
                check_level,
            )
            return result
        except Exception as e:
            last_err = e
            log.warning(
                "evaluate[%s] attempt=%s failed: %s",
                request_id,
                attempt,
                e,
            )
            if attempt < 3:
                time.sleep(attempt)

    # Regex fallback
    log.warning(
        "evaluate[%s] all attempts failed; trying regex fallback. last_err=%s",
        request_id,
        last_err,
    )
    if last_raw:
        fallback = _regex_extract_evaluations(last_raw)
        if fallback:
            log.info(
                "evaluate[%s] regex fallback recovered %s items",
                request_id,
                len(fallback),
            )
            return fallback

    raise ValueError(
        f"Evaluation failed after 3 attempts and regex fallback: {last_err}"
    )
