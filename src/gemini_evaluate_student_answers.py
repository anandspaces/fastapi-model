"""Stage 3: grade student OCR items against a teacher answer model (Dart-parity)."""

from __future__ import annotations

import json
import logging
import os
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
      questionNo, title, desc, pageNum, marks, diagramDescriptions
    """
    lines: list[str] = [f"Subject / Paper: {title or 'General'}", ""]
    for q in questions:
        qno = q.get("questionNo") or q.get("question_no") or "?"
        qlabel = str(qno).strip()
        if qlabel != "?" and not qlabel.upper().startswith("Q"):
            qlabel = f"Q{qlabel}"
        qtitle = (q.get("title") or "").strip()
        desc = (q.get("desc") or "").strip()
        marks = q.get("marks") or 0
        diagrams = q.get("diagramDescriptions") or []

        lines.append(f"{qlabel}. {qtitle}")
        if desc:
            lines.append(f"   Details: {desc}")
        if diagrams:
            lines.append(f"   Diagrams/Key points: {'; '.join(str(d) for d in diagrams)}")
        lines.append(f"   MAX MARKS ALLOWED: {marks}")
        lines.append("")
    return "\n".join(lines)


def _norm_question_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def smart_ocr_auto_combined_review_enabled() -> bool:
    """Env: SMART_OCR_AUTO_COMBINED_REVIEW=1 runs paper-level combined review after grading."""
    raw = (os.getenv("SMART_OCR_AUTO_COMBINED_REVIEW", "") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def compact_student_rows_for_evaluation(
    student_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One compact row per OCR question for stable alignment (includes blank attempts).

    Keys separate stem vs response so the model does not treat the printed stem as the answer.
    """
    rows: list[dict[str, Any]] = []
    for it in student_items:
        qid = _norm_question_id(it.get("question_id"))
        if qid is None:
            continue
        ans = str(it.get("student_answer", "") or "")
        rows.append(
            {
                "question_id": qid,
                "question_stem_ocr": str(it.get("question", "") or ""),
                "student_answer_text": ans,
                "has_handwritten_substance": bool(ans.strip()),
                "start_page": it.get("start_page"),
                "end_page": it.get("end_page"),
                "marking_page": it.get("marking_page"),
                "start_y_position_percent": it.get("start_y_position_percent"),
                "end_y_position_percent": it.get("end_y_position_percent"),
                "marking_x_position_percent": it.get("marking_x_position_percent"),
                "marking_y_position_percent": it.get("marking_y_position_percent"),
            }
        )
    return rows


def graded_items_for_combined_review(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map merged graded items to the per-question shape used by ``generate_combined_review``."""
    out: list[dict[str, Any]] = []
    for it in items:
        qid = _norm_question_id(it.get("question_id"))
        if qid is None:
            continue
        if "marks_awarded" not in it:
            continue
        try:
            ma = float(it.get("marks_awarded") or 0)
        except (TypeError, ValueError):
            ma = 0.0
        try:
            mm = int(it.get("max_marks") or 0)
        except (TypeError, ValueError):
            mm = 0
        title = str(it.get("question", "")).strip() or f"Question {qid}"
        fb = str(it.get("feedback") or "").strip()
        fd = str(it.get("feedback_detail") or "").strip()
        st = str(it.get("status") or "").lower()
        if st == "correct" and ma > 0 and fb:
            good = fb if fb.startswith("•") else f"• {fb}"
        elif ma > 0:
            good = f"• Awarded {ma:g}/{mm} marks. {fb}"[:400]
        else:
            good = "• No substantive marks." if not fb else (fb if fb.startswith("•") else f"• {fb}")
        impr = fd if fd else fb
        if impr and not impr.startswith("•"):
            impr = f"• {impr}"
        elif not impr:
            impr = "• Review model answer and terminology."
        fr = (fb + " " + fd).strip()[:500]
        out.append(
            {
                "question_no": str(qid),
                "title": title[:200],
                "marks_awarded": ma,
                "marks_total": mm,
                "good_points": good,
                "improvements": impr,
                "final_review": fr or fb,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_evaluation_prompt(
    subject: str,
    teacher_instructions: str,
    student_json: str,
) -> str:
    return f"""You are a strict but fair {subject} teacher evaluating an exam paper. You have deep expertise in {subject} and will evaluate answers with subject-specific knowledge.

You are given:
1. TEACHER INSTRUCTIONS / MODEL KEY — questions, model answers, marks, or marking rules.
2. STUDENT OCR TABLE — JSON array of objects with:
   - ``question_id``, ``question_stem_ocr``, ``student_answer_text``, ``has_handwritten_substance`` (see CRITICAL — STEM VS ANSWER).
   - Layout fields when present from OCR: ``start_page``, ``end_page``, ``marking_page``, ``start_y_position_percent``, ``end_y_position_percent``, ``marking_x_position_percent``, ``marking_y_position_percent`` — copy these into your output for that question when the student row exists; use JSON ``null`` for any missing input field.

CRITICAL — STEM VS ANSWER:
- ``question_stem_ocr`` is NOT the student's answer. Never award marks for merely restating or copying the printed stem.
- Grade ONLY substantive response content in ``student_answer_text``. If the stem is merged into that field by OCR, mentally separate boilerplate stem from the candidate's own reasoning, facts, and calculations.
- ``student_answer_summary`` in your output must summarize **only** the student's response substance — not the stem alone.

Your job: Grade against the TEACHER INSTRUCTIONS. Match by ``question_id`` / content. If a teacher question has **no** student row in the JSON (missing ``question_id``), treat as unattempted. If the row exists but ``has_handwritten_substance`` is false, status ``unattempted`` and 0 marks.

MARKING RULES:
- The `max_marks` in your output MUST exactly match the "MAX MARKS ALLOWED" for each question in the TEACHER INSTRUCTIONS.
- Award `marks_awarded` as a DECIMAL in multiples of 0.5 (0, 0.5, 1, 1.5, ...).
- NEVER EXCEED the "MAX MARKS ALLOWED" for a question.
- CONCEPTUAL FLEXIBILITY: Award full marks when meaning and logic match the key; do not require verbatim match.

FEEDBACK:
- ``feedback``: one concise paragraph (max ~40 words) — headline verdict referencing concrete terms from the student's answer.
- ``feedback_detail``: 2–5 short sentences (or bullet lines separated by \\n) — specific gaps, errors, or strengths; **minute** diagnosis where relevant.
- ``priority_improvement``: one sentence — the single highest-impact improvement for this question.
- ``dimension_notes``: one string with lines or bullets covering what applies among: **argument**, **facts/content**, **structure/organisation**, **language/clarity** (skip lines that do not apply).
- ``example_suggestion``: one sentence — one concrete example, illustration, or reference the student could add (aligned with the teacher key); empty string if not applicable.

LONG / ESSAY ANSWERS:
- If the answer spans multiple pages or reads like an essay, comment in ``feedback_detail`` whether the **conclusion** closes the argument (or if it trails off / repeats intro).
- Prefer one annotation with ``y_position_percent`` in the **lower third** (roughly 66–95) of the **last page** of the answer when remarking on the conclusion or final paragraph.

TEACHER INSTRUCTIONS / MODEL KEY ({subject}):
{teacher_instructions}

STUDENT OCR TABLE (compact):
{student_json}

OUTPUT EXACTLY a JSON array — ONE object per question in the TEACHER INSTRUCTIONS (same order as listed there). Do not omit questions.

[
  {{
    "question_id": 1,
    "question": "Question summary for output",
    "max_marks": 5,
    "marks_awarded": 3.5,
    "status": "partial",
    "student_answer_summary": "Substantive response only — not stem alone",
    "feedback": "Concise headline feedback...",
    "feedback_detail": "Sentence one.\\nSentence two.",
    "priority_improvement": "One sentence.",
    "dimension_notes": "argument: ...\\nfacts: ...\\nstructure: ...",
    "example_suggestion": "e.g. cite X or define Y",
    "start_page": 1,
    "start_y_position_percent": 10.0,
    "end_page": 2,
    "end_y_position_percent": 85.0,
    "marking_page": 1,
    "marking_x_position_percent": 50.0,
    "marking_y_position_percent": 45.0,
    "annotations": [
      {{
        "page_index": 1,
        "y_position_percent": 50.0,
        "x_start_percent": 20.0,
        "x_end_percent": 80.0,
        "comment": "specific remark",
        "is_positive": true
      }}
    ]
  }}
]

IMPORTANT — COORDINATES:
- If the student OCR table contains a row for that ``question_id`` (including blank ``student_answer_text``), copy the layout fields from that row into your output (use nulls where the input had null).
- If there was **no** OCR row for that ``question_id`` in the table, set ``start_page``, ``end_page``, ``marking_page``, all ``*_position_percent`` fields to JSON ``null`` and ``annotations`` to ``[]``. Do not guess page 1.

Annotations when the student wrote something: 1–3 per page spanned; ``page_index`` between ``start_page`` and ``end_page``. Match student's language.

CRITICAL JSON FORMATTING RULES:
- Output ONLY a valid JSON array.
- Escape quotes inside strings; use \\n for newlines inside strings.

CRITICAL PLACEMENT RULE: Space ``y_position_percent`` on the same page at least 15% apart from each other and from ``marking_y_position_percent``.

Status must be one of: "correct", "partial", "wrong", "unattempted".
"""


# ---------------------------------------------------------------------------
# JSON parsing + regex fallback (Dart parity)
# ---------------------------------------------------------------------------


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    return m.group(1).strip() if m else t


def _parse_evaluation_response(raw: str) -> list[dict[str, Any]]:
    cleaned = _strip_json_fence(raw)
    parsed = json.loads(cleaned)
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

        for field in (
            "question",
            "status",
            "student_answer_summary",
            "feedback",
            "feedback_detail",
            "priority_improvement",
            "dimension_notes",
            "example_suggestion",
        ):
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
    "feedback_detail",
    "priority_improvement",
    "dimension_notes",
    "example_suggestion",
    "annotations",
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
) -> list[dict[str, Any]]:
    """Call Gemini to grade student OCR items against the teacher model key.

    Returns a list of evaluation dicts (one per teacher question).
    Raises ValueError if all attempts fail and regex fallback is also empty.
    """
    compact = compact_student_rows_for_evaluation(student_items)
    student_json = json.dumps(compact, ensure_ascii=False)
    prompt = _build_evaluation_prompt(subject, teacher_instructions, student_json)

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=12288,
        response_mime_type="application/json",
    )

    last_raw: str = ""
    last_err: Exception | None = None

    for attempt in range(1, 4):  # 3 attempts, matching Dart
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=[types.Part.from_text(text=prompt)],
                config=cfg,
            )
            raw = (getattr(resp, "text", None) or "").strip()
            if not raw:
                raise ValueError("Empty response from Gemini.")
            last_raw = raw
            result = _parse_evaluation_response(raw)
            log.info(
                "evaluate[%s] attempt=%s ok evaluations=%s",
                request_id,
                attempt,
                len(result),
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
