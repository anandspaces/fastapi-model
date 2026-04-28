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
1. TEACHER INSTRUCTIONS / MODEL KEY — the correct questions, answers, marks, or marking rules provided by the teacher.
2. A STUDENT's OCR-extracted answers (student_answer) from their handwritten copy.

Your job: Based on the TEACHER INSTRUCTIONS, grade the student's extracted answers. Find each question's matching student answer and award marks. Match the student's answer to the specific model key question by content or question number, even if the student answered them out of order (e.g. if the student answered Q5 first, match it to Q5 in the model key). If a question is present in the TEACHER INSTRUCTIONS but missing from the student's answers, it MUST be included in the output with a status of "unattempted" and 0 marks.

MARKING RULES:
- The `max_marks` in your output MUST exactly match the "MAX MARKS ALLOWED" for each question in the TEACHER INSTRUCTIONS.
- Award `marks_awarded` as a DECIMAL in multiples of 0.5 (0, 0.5, 1, 1.5, ...).
- NEVER EXCEED the "MAX MARKS ALLOWED" for a question. If a student's answer is perfect, give exactly the MAX MARKS ALLOWED.
- CONCEPTUAL FLEXIBILITY: Evaluate with deep human intelligence! Do not blindly string-match; award full marks if the underlying meaning and logic is identical.
- CONSTRUCTIVE FEEDBACK: The `feedback` string MUST explicitly reference the specific terms, concepts, or calculations in the student's answer. Limit it to maximum 20-35 words.

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
        "page_index": 1,
        "y_position_percent": 50.0,
        "x_start_percent": 20.0,
        "x_end_percent": 80.0,
        "comment": "short teacher remark",
        "is_positive": true
      }}
    ]
  }}
]

IMPORTANT: Copy ALL page bound and marking position_percent fields from the matching student OCR answer directly into your output. For unattempted questions, provide safe defaults for page fields (e.g. start_page=1, start_y_position_percent=10.0, etc.) so that the app doesn't crash.
The "annotations" array MUST contain at least 1-3 specific spots FOR EVERY SINGLE PAGE the student's answer spans. Do NOT put all annotations on one page. Distribute them across ALL pages between "start_page" and "end_page". "page_index" must be a valid 1-indexed page number within the bounds of "start_page" and "end_page". Use the student's language (English or Hindi) for the comment.
- For correct or good parts, set "is_positive": true.
- For mistakes, wrong answers, or areas needing improvement, set "is_positive": false and provide a constructive comment correcting the student.

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
    student_json = json.dumps(student_items, ensure_ascii=False)
    prompt = _build_evaluation_prompt(subject, teacher_instructions, student_json)

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
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
