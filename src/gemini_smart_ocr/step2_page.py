"""Step 2 — per-page OCR + Q&A extraction (intro flag on page 1).

One Gemini call per page. Receives the grid-overlaid page image and returns:
  - ``text``: full OCR transcript
  - ``is_intro``: only meaningful when ``is_first_page=True``
  - ``questions``: list of per-page question records (label, continuation flag,
    student-answer-on-this-page, y-bounds, answer_type)

Two retries with ``RETRY_BACKOFF_S`` backoff and finish-reason logging, matching
the previous ocr.py retry pattern.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google import genai
from google.genai import types

from .config import (
    RETRY_BACKOFF_S,
    STEP2_HTTP_TIMEOUT_MS,
    STEP2_MAX_OUTPUT_TOKENS,
    afc_off,
    finish_reason_name,
    http_opts,
    step2_model,
    thinking_off,
)
from .parsing import parse_json_candidates
from .prompts import build_step2_page_prompt
from .schemas import STEP2_PAGE_SCHEMA

log = logging.getLogger(__name__)


def extract_page(
    api_key: str,
    grid_png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
    *,
    is_first_page: bool,
) -> dict[str, Any]:
    """One Gemini call. Returns ``{"text", "is_intro", "questions": [...]}``.

    On failure after both attempts, returns the safe-empty record
    ``{"text": "", "is_intro": False, "questions": []}`` and logs a warning —
    callers must treat such pages as "OCR failed" rather than "intro".
    """
    client = genai.Client(api_key=api_key)
    prompt = build_step2_page_prompt(
        page_num, total_pages, language, is_first_page=is_first_page,
    )
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=grid_png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=STEP2_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=STEP2_PAGE_SCHEMA,
        thinking_config=thinking_off(),
        automatic_function_calling=afc_off(),
        http_options=http_opts(STEP2_HTTP_TIMEOUT_MS),
    )
    model = step2_model()
    last_raw = ""
    last_fr = "UNKNOWN"
    for attempt in range(1, 3):
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        except Exception as e:
            log.warning("step2[p%s] attempt=%s call failed: %s", page_num, attempt, e)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        last_raw = (getattr(resp, "text", None) or "").strip()
        last_fr = finish_reason_name(resp)
        if not last_raw:
            log.warning("step2[p%s] empty response finish_reason=%s", page_num, last_fr)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        parsed: dict[str, Any] | None = None
        for cand in parse_json_candidates(last_raw):
            if isinstance(cand, dict) and "text" in cand:
                parsed = cand
                break
        if parsed is None:
            log.warning(
                "step2[p%s] parse fail attempt=%s finish_reason=%s prefix=%r",
                page_num, attempt, last_fr, last_raw[:200],
            )
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        return _normalize_step2_response(parsed, is_first_page=is_first_page)

    log.warning(
        "step2[p%s] giving up after 2 attempts finish_reason=%s tail=%r",
        page_num, last_fr, last_raw[-200:],
    )
    return {"text": "", "is_intro": False, "questions": []}


def _normalize_step2_response(
    data: dict[str, Any],
    *,
    is_first_page: bool,
) -> dict[str, Any]:
    text = str(data.get("text", "") or "").strip()
    is_intro = bool(data.get("is_intro", False)) if is_first_page else False
    raw_qs = data.get("questions") or []
    questions: list[dict[str, Any]] = []
    if isinstance(raw_qs, list):
        for q in raw_qs:
            if not isinstance(q, dict):
                continue
            questions.append({
                "question_label": str(q.get("question_label", "") or "").strip(),
                "question_text": str(q.get("question_text", "") or "").strip(),
                "student_answer": str(q.get("student_answer", "") or "").strip(),
                "start_y_position_percent": _to_pct(q.get("start_y_position_percent")),
                "end_y_position_percent": _to_pct(q.get("end_y_position_percent")),
                "answer_type": str(q.get("answer_type", "paragraph") or "paragraph").strip().lower(),
                "is_continuation": bool(q.get("is_continuation", False)),
            })
    return {"text": text, "is_intro": is_intro, "questions": questions}


def _to_pct(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, x))
