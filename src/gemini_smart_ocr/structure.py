"""Stage 2 — flatten OCR text into question rows with coordinates.

Gemini receives the JSON-encoded per-page OCR payload (no images) and returns a
``{sections:[{section_name, questions:[…]}]}`` document. We then:

  - Normalize each question row (clamp page/y, infer ``question_id`` from
    printed label, fill blank answers without dropping their numbering).
  - Stable-sort by (start_page, start_y, question_id).
  - Assign gap IDs to any row that still has no ``question_id`` so downstream
    callers can index by it.
  - On undercount (got < 80% of expected), retry once with a half-PDF split.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from .config import (
    ANSWER_TYPES,
    RETRY_BACKOFF_S,
    STRUCTURE_MAX_OUTPUT_TOKENS,
    structure_model,
    thinking_off,
)
from .dedup import extract_page_body
from .parsing import clamp_pct, page_num, parse_json_candidates
from .prompts import build_structure_prompt
from .schemas import STRUCTURE_ROOT_SCHEMA

log = logging.getLogger(__name__)


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


def _normalize_answer_type(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in ANSWER_TYPES:
        return s
    if s in ("correction", "short", "line"):
        return "correction"
    if s in ("word_list", "wordlist", "list", "pairs", "tabular"):
        return "word_list"
    if s in ("paragraph", "prose", "long"):
        return "paragraph"
    return "paragraph"


def estimate_expected_questions(page_blocks: list[str]) -> int | None:
    """Heuristic: infer approximate question count from visible numbering labels."""
    nums: set[int] = set()
    patterns = (
        r"\bq(?:uestion)?\s*[:.\-]?\s*(\d{1,3})\b",
        r"(?:^|\n)\s*\(?(\d{1,3})\)?\s*[.)\-:]\s+",
        r"(?:प्रश्न|उ\.?\s*प्र\.?)\s*[:.\-]?\s*(\d{1,3})\b",
    )
    for block in page_blocks:
        body = extract_page_body(block)
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
    start_page = page_num(item.get("start_page"), total_pages, 1)
    end_page = page_num(item.get("end_page"), total_pages, start_page)
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
    start_sy = clamp_pct(item.get("start_y_position_percent", 0))
    end_sy = clamp_pct(item.get("end_y_position_percent", 100))

    # Score mark always lands on the start page at the y where the answer begins.
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
        "marking_x_position_percent": 85.0,  # right margin centerline
        "marking_y_position_percent": my,
    }


def _parse_structure_sections(raw: str, total_pages: int) -> list[dict[str, Any]]:
    parsed: Any = None
    for candidate in parse_json_candidates(raw):
        parsed = candidate
        break
    if parsed is None:
        log.warning("smart_ocr structure JSON parse failed: invalid JSON")
        raise ValueError("invalid JSON")

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
            out.append(_normalize_flat_item(row, total_pages, section_name))
    if not out:
        raise ValueError("No question-answer blocks detected.")
    out.sort(key=_structure_item_sort_key)

    used_ids: set[int] = {
        i for i in (item.get("question_id") for item in out)
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


def _run_structure_pass(
    client: genai.Client,
    pages_payload_json: str,
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    prompt = build_structure_prompt(language, total_pages, expected_questions)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=pages_payload_json),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=STRUCTURE_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=STRUCTURE_ROOT_SCHEMA,
        thinking_config=thinking_off(),
    )
    model = structure_model()
    last_err: Exception | None = None
    for attempt in range(1, 3):
        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw:
            raise ValueError("Structure pass returned an empty response.")
        try:
            return _parse_structure_sections(raw, total_pages)
        except ValueError as e:
            last_err = e
            log.warning("structure_qa parse attempt %s/2 failed: %s", attempt, e)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S * 0.8)
    raise ValueError(f"Structure pass failed after 2 attempts: {last_err}") from last_err


def structure_qa_with_fallback(
    client: genai.Client,
    pages_payload: dict[str, Any],
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    """Run Stage 2; on undercount, retry once with a halved payload and merge."""
    pages_payload_json = json.dumps(pages_payload, ensure_ascii=False)
    rows = _run_structure_pass(
        client, pages_payload_json, language, total_pages,
        expected_questions=expected_questions,
    )
    if expected_questions and len(rows) < max(1, int(expected_questions * 0.8)):
        pages = pages_payload.get("pages", [])
        if not isinstance(pages, list) or len(pages) < 2:
            return rows
        log.warning(
            "structure_qa undercount got=%s expected~%s; retry split",
            len(rows), expected_questions,
        )
        mid = len(pages) // 2
        first = {"pages": pages[:mid]}
        second = {"pages": pages[mid:]}
        rows1 = _run_structure_pass(
            client, json.dumps(first, ensure_ascii=False), language, total_pages,
        )
        rows2 = _run_structure_pass(
            client, json.dumps(second, ensure_ascii=False), language, total_pages,
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
