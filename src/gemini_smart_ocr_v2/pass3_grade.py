"""Pass 3: grade one question at a time with FREE/OCCUPIED spatial vocabulary."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from cell_grid_service_v4 import PageCellGrid

from src.gemini_evaluate_student_answers import (
    evaluation_strictness_instruction,
    format_answer_model_as_teacher_instructions,
)
from src.gemini_evaluate_student_answers import _repair_truncated_json, _strip_json_fence
from src.gemini_extract import MODEL_ID

from src.annotation_planner import assign_comment_rows_for_evaluation

from .free_cells import compute_free_cells, pass3_spatial_payload
from .validator import snap_anchor_rows_to_grid, validate_annotations_for_item

log = logging.getLogger(__name__)


def _teacher_block_for_question(
    questions: list[dict[str, Any]],
    title: str,
    question_id: int,
) -> str:
    for q in questions:
        qno = q.get("questionNo")
        if qno is None:
            qno = q.get("question_no")
        try:
            if int(qno) == int(question_id):
                return format_answer_model_as_teacher_instructions([q], title)
        except (TypeError, ValueError):
            continue
    return format_answer_model_as_teacher_instructions([], title)


def _pages_for_item(item: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    sp, ep = item.get("start_page"), item.get("end_page")
    try:
        if sp is not None:
            pages.add(int(sp))
        if ep is not None:
            pages.add(int(ep))
    except (TypeError, ValueError):
        pass
    acc = item.get("answer_span")
    if isinstance(acc, list):
        for span in acc:
            if isinstance(span, dict) and span.get("page") is not None:
                try:
                    pages.add(int(span["page"]))
                except (TypeError, ValueError):
                    continue
    return pages or {1}


def _build_pass3_prompt(
    *,
    subject: str,
    teacher_block: str,
    student_json: str,
    spatial_json: str,
    check_level: str,
) -> str:
    strict_line = evaluation_strictness_instruction(check_level)
    return f"""You are a strict but fair {subject} examiner-mentor (UPSC Civil Services mains ethos).

You receive ONE student's answer for ONE question. Spatial vocabulary:
- Row ranges like "E13:S13" mean columns E through S on row 13.
- FREE_PAGE_N lists writable cell ranges where COMMENT text may go.
- OCCUPIED_PAGE_N lists cells covered by this student's handwriting for this question — use these (plus adjacent FREE rows when the anchor type requires it, e.g. underline beneath the text) for anchor marks (underline/ellipse/tick…).

{strict_line}

MARKING RULES:
- max_marks MUST match MAX MARKS ALLOWED in the teacher block below.
- marks_awarded in 0.5 steps; never exceed max_marks.
- feedback: 25–40 words, examiner tone.

ANNOTATION RULES:
- Each annotation: page (1-based), comment, is_positive, comment_font_pts in [14–16] (default 14),
  preferred_side ("left" | "right" | "gap"), priority (1=high, 2=medium, 3=low),
  anchor: {{ "type": ..., "rows": [...] }}.
- Do NOT emit comment_rows — Python places comments into FREE cells using your preferred_side + anchor proximity.
- answer_span and marking are FIXED — do NOT emit them (Python merges resolved spans).

COMMENT PLACEMENT HINTS (semantic only — backend computes exact cells):
- preferred_side "right": margin notes in cols AV–BG (typical for UPSC body text).
- preferred_side "left": notes in cols A–K.
- preferred_side "gap": horizontal blank band between paragraphs (Python prefers wide gap rows).
- priority: use 1 for the most important feedback on a page, 2 default, 3 for minor notes — resolves overlapping placement order.
- Spread preferred_side across annotations on the same page (mix left/right/gap) when possible.

ANNOTATION COUNT (per-page floor):
- For EACH page that contains this student's handwriting, emit 3-4 annotations on that page.
  A 2-page answer therefore yields 6-8 annotations total; a 1-page answer 3-4.
- Distribute the annotations across the page (top, middle, bottom) so feedback covers the full answer, not only the start.

ANCHOR PLACEMENT (use the overlay JPEG to choose cells precisely):
- "ellipse"   (used to be "circle" — always emit "ellipse" now)
  Rows MUST be TIGHT to the specific word/phrase being ringed — only the columns whose cells contain that word's ink.
  Do NOT span the whole writing row. Read the overlay to find the exact leftmost and rightmost cells of the word and emit ONE single-row range that hugs it.
  Example: a single word "anger" sitting in cells M59-R59 -> "rows": ["M59:R59"].
  Wrong: "rows": ["M59:AT59"] (covers the entire row, not just the word).
- "underline"
  Underlines live in FREE rows directly UNDER the ink cells (never on the same row as the words).
  If the praised/criticised phrase fits on ONE OCCUPIED row R, anchor.rows contains ONE range on row R+1 with tight column span to that phrase's ink below.
  If handwriting is SLANTED or wraps across OCCUPIED rows R, R+1, … R+k as one phrase, anchor.rows MUST list ONE range string per OCCUPIED row, each describing the underline band immediately below THAT row — e.g. ["L{{R+1}}:AU{{R+1}}", "L{{R+2}}:AU{{R+2}}", ...] hugging ink under each respective line (no single spanning rectangle through the body's mid-air).
  Use empty cells beneath the handwriting even when FREE — that is intentional for underline placement.
- "tick"
  Rows are 1-3 cells either at the very end of the praised writing row or in the adjacent margin column on that row.
- "arrow"
  Margin arrow pointing at specific ink — rows cover the target phrase band (single-row range tight to ink).
- "curly_brace"
  Rows span multiple lines of ink (paragraph bracket); list one range per row from line_start to line_end columns.
- "none"
  Omit "rows" (or pass an empty array). Use sparingly — prefer a real anchor when feedback refers to a specific phrase.

DO NOT emit "circle" — use "ellipse".

TEACHER KEY (this question only):
{teacher_block}

STUDENT ROW (JSON — single object):
{student_json}

SPATIAL CONSTRAINTS (JSON):
{spatial_json}

Overlay JPEG images follow in page order for the pages listed above (page 1 first among those sent).

OUTPUT: a single JSON object with keys:
question_id, question, max_marks, marks_awarded, status, student_answer_summary,
feedback, annotations

Each annotation must include anchor.rows (except type "none") and preferred_side + priority — NOT comment_rows.
Do NOT include answer_span or marking."""

def _parse_one_question_eval(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = _repair_truncated_json(cleaned)
        data = json.loads(cleaned)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("Pass3: expected object")
    return dict(data)


def grade_items_pass3_v2(
    api_key: str,
    subject: str,
    title: str,
    teacher_questions: list[dict[str, Any]],
    items_to_grade: list[dict[str, Any]],
    grids: list[PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    overlay_images: list[bytes] | None,
    *,
    request_id: str,
    check_level: str = "Moderate",
) -> list[dict[str, Any]]:
    """Returns evaluation dicts (same merge shape as ``evaluate_student_answers_against_model``)."""
    client = genai.Client(api_key=api_key)
    grids_by_page = {g.page: g for g in grids}
    initial_free_full = compute_free_cells(grids, flat_blocks)
    evaluations: list[dict[str, Any]] = []

    for item in items_to_grade:
        if not isinstance(item, dict):
            continue
        try:
            qid = int(item.get("question_id"))
        except (TypeError, ValueError):
            continue

        teacher_block = _teacher_block_for_question(teacher_questions, title, qid)
        pages = _pages_for_item(item)
        answer_refs = [str(x) for x in (item.get("answer_block_refs") or [])]
        spatial = pass3_spatial_payload(grids, flat_blocks, answer_refs, pages)

        student_payload = {
            "question_id": qid,
            "question": item.get("question"),
            "student_answer": item.get("student_answer"),
            "is_attempted": item.get("is_attempted"),
            "answer_span": item.get("answer_span"),
            "marking": item.get("marking"),
            **spatial,
        }
        prompt = _build_pass3_prompt(
            subject=subject,
            teacher_block=teacher_block,
            student_json=json.dumps(student_payload, ensure_ascii=False),
            spatial_json=json.dumps(spatial, ensure_ascii=False),
            check_level=check_level,
        )

        parts: list[types.Part] = [types.Part.from_text(text=prompt)]
        if overlay_images:
            sorted_pages = sorted(pages)
            for p in sorted_pages:
                idx = p - 1
                if 0 <= idx < len(overlay_images):
                    parts.append(
                        types.Part.from_bytes(data=overlay_images[idx], mime_type="image/jpeg")
                    )

        cfg = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=65536,
            response_mime_type="application/json",
        )

        last_raw = ""
        ev: dict[str, Any] | None = None
        for attempt in range(1, 3):
            try:
                resp = client.models.generate_content(
                    model=MODEL_ID, contents=parts, config=cfg
                )
                last_raw = (getattr(resp, "text", None) or "").strip()
                if not last_raw:
                    time.sleep(0.2)
                    continue
                ev = _parse_one_question_eval(last_raw)
                break
            except Exception as e:
                log.warning("pass3[%s] qid=%s attempt err %s", request_id, qid, e)
                time.sleep(0.25)

        if ev is None:
            log.error("pass3[%s] failed qid=%s raw=%r", request_id, qid, last_raw[:200])
            ev = {
                "question_id": qid,
                "question": str(item.get("question") or ""),
                "max_marks": 0,
                "marks_awarded": 0,
                "status": "unattempted",
                "student_answer_summary": "",
                "feedback": "Grading pass failed.",
                "annotations": [],
            }

        ev["answer_span"] = item.get("answer_span")
        ev["marking"] = item.get("marking")

        anns = ev.get("annotations")
        if isinstance(anns, list):
            normed: list[dict[str, Any]] = []
            for a in anns:
                if not isinstance(a, dict):
                    continue
                ad = dict(a)
                try:
                    pg = int(ad.get("page", 1))
                except (TypeError, ValueError):
                    pg = 1
                g0 = grids_by_page.get(pg)
                if g0 is not None and ad.get("anchor"):
                    ad["anchor"] = snap_anchor_rows_to_grid(g0, ad["anchor"])
                normed.append(ad)
            assign_comment_rows_for_evaluation(normed, grids_by_page, initial_free_full)
            init_free = {p: set(s) for p, s in initial_free_full.items()}
            ev["annotations"] = validate_annotations_for_item(
                grids_by_page, normed, init_free
            )
        else:
            ev["annotations"] = []

        evaluations.append(ev)
        log.info(
            "smart_ocr_v2[%s] pass3 qid=%s annotations=%s",
            request_id,
            qid,
            len(ev.get("annotations") or []),
        )

    return evaluations
