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


def _grading_hint_fields(student_answer: str) -> dict[str, Any]:
    """Derived hints for the evaluator only (not authoritative layout)."""
    s = student_answer or ""
    parts = [p for p in s.split("\n\n") if str(p).strip()]
    lines = s.splitlines()
    return {
        "_grading_hint_paragraph_count": len(parts) if parts else (1 if s.strip() else 0),
        "_grading_hint_line_count": len(lines),
        "_grading_hint_char_count": len(s),
    }


def _build_evaluation_prompt(
    subject: str,
    teacher_instructions: str,
    student_json: str,
) -> str:
    return f"""You are a strict but fair {subject} examiner-mentor (UPSC Civil Services / mains-answer ethos). Your feedback reads like seasoned script evaluation — professional, restrained, clinically useful — never cheerleading.

You are given:
1. TEACHER INSTRUCTIONS / MODEL KEY — the correct questions, answers, marks, or marking rules provided by the teacher.
2. A STUDENT's OCR-extracted answers (student_answer) from their handwritten copy. Each row may include ``_grading_hint_*`` fields (paragraph/line/char counts) — use them only to scale annotation density, not as marks evidence.
   Rows with completely blank answers may be omitted from this JSON — you must still emit "unattempted" with 0 marks for every TEACHER question. Do not invent OCR text in ``student_answer_summary``.

Your job: Based on the TEACHER INSTRUCTIONS, grade the student's extracted answers. Find each question's matching student answer and award marks. Match the student's answer to the specific model key question by content or question number, even if the student answered them out of order (e.g. if the student answered Q5 first, match it to Q5 in the model key). If a question is present in the TEACHER INSTRUCTIONS but missing from the student's answers, it MUST be included in the output with a status of "unattempted" and 0 marks.

MARKING RULES:
- The `max_marks` in your output MUST exactly match the "MAX MARKS ALLOWED" for each question in the TEACHER INSTRUCTIONS.
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

IMPORTANT — COORDINATES:
- If the student's OCR JSON contains a row for that ``question_id`` (even with empty ``student_answer``), copy that row's page bounds and marking positions into your output, and add annotations only across those pages.
- **Drawable vertical band per page:** Treat ``start_y_position_percent`` / ``end_y_position_percent`` with ``start_page`` / ``end_page`` as the handwritten answer band. On the **first** page of the answer use Y from ``start_y_position_percent`` down to the bottom of the page (100%). On the **last** page use Y from the top (0%) to ``end_y_position_percent``. On **middle** pages (if any) use the full page (0%–100%). Place annotation centres in **whitespace between paragraph bands** — infer paragraph breaks from ``student_answer`` (``\\n\\n``): aim callout Y in the **gaps** between blocks, not centred on dense ink lines when a blank band exists between paragraphs.
- Honour the layout intent from OCR: annotations and examiner marks stay in the handwritten answer zone — below printed stem / divider and above page-foot rules; spread ``marking_y`` and annotation ``y_position_percent`` so callouts do not stack on the same Y as the first printed line label.
- If the student's OCR JSON has **no row at all** for that ``question_id`` (question not written on the copy), set ``start_page``, ``end_page``, ``marking_page``, all ``*_position_percent`` fields, and ``marking_*_position_percent`` to JSON ``null``, and set ``annotations`` to ``[]``. Do **not** guess page 1 or other placeholders.

The "annotations" array — **dense examiner marginalia** (specific remarks **on** the student's content; preferred over cramming detail only into ``feedback``). Diagnostic, non-repetitive — like a seasoned mains evaluator filling margins.

**Minimum annotation counts (scale up using ``_grading_hint_paragraph_count`` / line length):**
- **Substantive long answers** (roughly more than ~6–8 lines or multiple paragraphs / multi-page): emit **at least 8** annotations total when the script spans multiple dense pages or ``_grading_hint_paragraph_count`` ≥ 4; otherwise **at least 6**. For very long mains-style answers (many paragraphs or high line count), aim **10–14+** total across pages — **at least one substantive annotation per paragraph block** when ``student_answer`` has multiple ``\\n\\n`` blocks (each block addressed somewhere on the relevant page band).
- You **must still cover three structural zones** when they exist — label implicitly in wording only:
  1) **Opening / introduction** — framing, definitions, roadmap, relevance to the question.
  2) **Body / core evidence** — arguments, examples, logic, omissions vs marking scheme (several spots for long answers).
  3) **Conclusion / synthesis** — closure quality and link back to the stem; if missing, annotate in the appropriate band what is lacking.
- If intro or conclusion zones are missing from the handwriting, add an annotation in the nearest appropriate vertical band on the first/last answered page.

For **short** answers (about 3 lines or less): emit **at least 3 annotations** anchored to concrete phrases (not generic filler).

**Per-page density:** On each page with **dense** handwriting in the drawable band, aim **4–7** annotations where vertical space allows; thinner pages fewer. **Every page with substantial ink** should carry multiple remarks — not only the first or last page.

Across pages: the **first page with significant ink** tends toward intro-related critique; the **last page with substantive writing** toward conclusion-related critique unless structure contradicts.

Technical: ``page_index`` must lie between ``start_page`` and ``end_page`` (same 1-based convention as those fields). Use the student's language (English or Hindi) for every ``comment`` — formal, succinct, mains-appropriate.
- For genuinely strong, non-generic merits only, set "is_positive": true (sparse use).
- For sharpening, omission, misconception, shallow example, weak conclusion — set "is_positive": false and one actionable line.

**Horizontal stagger:** When multiple annotations fall on the **same** ``page_index``, alternate **left vs right** margin bands using ``x_start_percent`` / ``x_end_percent`` (e.g. left cluster ~5–40%, right cluster ~60–95%) so more remarks fit without overlap.

CRITICAL JSON FORMATTING RULES:
- You must ONLY output a valid JSON array.
- DO NOT use unescaped double quotes inside string values (use ' or escape them).
- DO NOT use literal newlines inside string values (use \\n).
- Make sure every object and array is properly closed.

CRITICAL PLACEMENT RULE: On the same ``page_index``, keep ``y_position_percent`` at least **12%** apart from each other **unless** you stagger horizontally (left vs right bands as above); if staggered, **8%** minimum vertical gap is acceptable. Stay at least **8%** away from ``marking_y_position_percent`` on that page when ``marking_page`` equals that ``page_index``. This prevents UI overlap while allowing dense examiner-style coverage.
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


# Keys produced by Stage 3 grading (merged into each OCR item; avoid duplicating
# question text / coordinates already present from Stage 2).
_GRADING_KEYS = (
    "max_marks",
    "marks_awarded",
    "status",
    "student_answer_summary",
    "feedback",
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


def _clamp_pct_val(x: Any, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = 0.0
    return max(lo, min(hi, v))


def _vertical_band_for_page(
    page: int,
    start_page: int,
    end_page: int,
    start_y: float,
    end_y: float,
) -> tuple[float, float]:
    """Drawable vertical band (y low→high percent from page top) for one 1-based page index."""
    if page < start_page or page > end_page:
        return (0.0, 100.0)
    sy = _clamp_pct_val(start_y)
    ey = _clamp_pct_val(end_y)
    if start_page == end_page:
        lo, hi = (sy, ey) if sy <= ey else (ey, sy)
        if hi - lo < 5.0:
            hi = min(100.0, lo + 10.0)
        return (lo, hi)
    if page == start_page:
        return (sy, 100.0)
    if page == end_page:
        return (0.0, ey)
    return (0.0, 100.0)


def _nudge_y_away_from_marking(
    y: float,
    marking_y: float,
    lo: float,
    hi: float,
    clearance: float = 10.0,
) -> float:
    if abs(y - marking_y) >= clearance:
        return y
    above = marking_y + clearance
    below = marking_y - clearance
    candidates = []
    if lo <= below <= hi:
        candidates.append(below)
    if lo <= above <= hi:
        candidates.append(above)
    if not candidates:
        return _clamp_pct_val((lo + hi) / 2.0)
    return min(candidates, key=lambda c: abs(c - y))


def normalize_evaluation_annotations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spread annotation Y (and stagger X) within OCR vertical bands; preserves comment order per page."""
    out: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        anns = row.get("annotations")
        if not isinstance(anns, list) or not anns:
            out.append(row)
            continue
        sp, ep = row.get("start_page"), row.get("end_page")
        if sp is None or ep is None:
            out.append(row)
            continue
        try:
            start_page = int(sp)
            end_page = int(ep)
        except (TypeError, ValueError):
            out.append(row)
            continue

        sy = _clamp_pct_val(row.get("start_y_position_percent"))
        ey = _clamp_pct_val(row.get("end_y_position_percent"), 0.0, 100.0)
        try:
            marking_page = int(row.get("marking_page") or start_page)
        except (TypeError, ValueError):
            marking_page = start_page
        marking_y = _clamp_pct_val(row.get("marking_y_position_percent"))

        new_anns: list[dict[str, Any]] = []
        for a in anns:
            new_anns.append(dict(a) if isinstance(a, dict) else {})

        by_page: dict[int, list[int]] = {}
        for idx, a in enumerate(new_anns):
            if not isinstance(a, dict):
                continue
            try:
                pi = int(a.get("page_index", start_page))
            except (TypeError, ValueError):
                pi = start_page
            pi = max(start_page, min(end_page, pi))
            by_page.setdefault(pi, []).append(idx)

        for page, indices in by_page.items():
            lo, hi = _vertical_band_for_page(page, start_page, end_page, sy, ey)
            pad = 3.0
            lo = _clamp_pct_val(lo + pad)
            hi = _clamp_pct_val(hi - pad)
            if hi - lo < 15.0:
                lo, hi = _vertical_band_for_page(page, start_page, end_page, sy, ey)

            ordered = sorted(
                indices,
                key=lambda i: _clamp_pct_val(new_anns[i].get("y_position_percent")),
            )
            n = len(ordered)
            for slot, ann_idx in enumerate(ordered):
                if n <= 0:
                    break
                y_new = lo + (slot + 1) * (hi - lo) / (n + 1)
                if marking_page == page:
                    y_new = _nudge_y_away_from_marking(y_new, marking_y, lo, hi)
                new_anns[ann_idx]["y_position_percent"] = round(_clamp_pct_val(y_new), 2)
                # Left/right stagger for same-page density
                if slot % 2 == 0:
                    new_anns[ann_idx]["x_start_percent"] = 5.0
                    new_anns[ann_idx]["x_end_percent"] = 42.0
                else:
                    new_anns[ann_idx]["x_start_percent"] = 58.0
                    new_anns[ann_idx]["x_end_percent"] = 95.0

        row["annotations"] = new_anns
        out.append(row)
    return out


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
    payload: list[dict[str, Any]] = []
    for it in student_items:
        row = dict(it)
        sa = str(row.get("student_answer") or "")
        row.update(_grading_hint_fields(sa))
        payload.append(row)
    student_json = json.dumps(payload, ensure_ascii=False)
    prompt = _build_evaluation_prompt(subject, teacher_instructions, student_json)

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=16384,
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
            err_s = str(e).lower()
            extra = ""
            if "json" in err_s or "parse" in err_s or "unterminated" in err_s:
                extra = " (JSON parse failure may indicate output truncation — check max_output_tokens)"
            log.warning(
                "evaluate[%s] attempt=%s failed: %s%s",
                request_id,
                attempt,
                e,
                extra,
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
