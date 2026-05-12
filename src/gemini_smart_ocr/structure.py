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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from google import genai
from google.genai import types

from .config import (
    ANSWER_TYPES,
    STRUCTURE_CHUNK_PAGES,
    STRUCTURE_CHUNK_THRESHOLD,
    STRUCTURE_HTTP_TIMEOUT_MS,
    STRUCTURE_MAX_OUTPUT_TOKENS,
    afc_off,
    finish_reason_name,
    http_opts,
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
    use_schema: bool = False,
) -> list[dict[str, Any]]:
    """Single-shot Stage-2 Gemini call. Schema-less by default (2–3× faster generation);
    callers fall back to ``use_schema=True`` on parse failure.
    """
    prompt = build_structure_prompt(language, total_pages, expected_questions)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=pages_payload_json),
    ]
    cfg_kwargs: dict[str, Any] = dict(
        temperature=0.1,
        max_output_tokens=STRUCTURE_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        thinking_config=thinking_off(),
        automatic_function_calling=afc_off(),
        http_options=http_opts(STRUCTURE_HTTP_TIMEOUT_MS),
    )
    if use_schema:
        cfg_kwargs["response_schema"] = STRUCTURE_ROOT_SCHEMA
    cfg = types.GenerateContentConfig(**cfg_kwargs)
    model = structure_model()
    try:
        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
    except Exception as e:
        raise ValueError(f"Structure pass HTTP failure: {e}") from e
    raw = (getattr(resp, "text", None) or "").strip()
    fr = finish_reason_name(resp)
    if not raw:
        raise ValueError(f"Structure pass empty response finish_reason={fr}")
    if fr == "MAX_TOKENS":
        log.warning(
            "structure_qa MAX_TOKENS truncation cap=%s schema=%s prefix=%r tail=%r",
            STRUCTURE_MAX_OUTPUT_TOKENS, use_schema, raw[:300], raw[-200:],
        )
        raise ValueError(f"structure truncated (MAX_TOKENS): cap={STRUCTURE_MAX_OUTPUT_TOKENS}")
    try:
        return _parse_structure_sections(raw, total_pages)
    except ValueError as e:
        log.warning(
            "structure_qa parse failure schema=%s finish_reason=%s len=%s prefix=%r tail=%r err=%s",
            use_schema, fr, len(raw), raw[:500], raw[-200:], e,
        )
        raise


def _merge_and_dedup(
    row_groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Concat row groups, sort by (start_page, start_y, qid), drop duplicate question_ids."""
    merged: list[dict[str, Any]] = []
    for g in row_groups:
        merged.extend(g)
    merged.sort(key=_structure_item_sort_key)
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for it in merged:
        qid = it.get("question_id")
        if qid not in seen:
            out.append(it)
            seen.add(qid)
    return out


def _run_structure_chunked(
    client: genai.Client,
    pages: list[dict[str, Any]],
    language: str,
    total_pages: int,
) -> list[dict[str, Any]]:
    """Split ``pages`` into ``STRUCTURE_CHUNK_PAGES``-sized chunks; run them in parallel.

    Each chunk tries schema-less first then schema-enabled retry on parse failure.
    Chunk-level failures are logged and skipped — survivors are merged + deduped.
    """
    chunk_size = max(2, STRUCTURE_CHUNK_PAGES)
    chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]
    log.info(
        "structure_qa chunked pages=%s chunks=%s chunk_size=%s",
        len(pages), len(chunks), chunk_size,
    )

    def _job(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = json.dumps({"pages": chunk}, ensure_ascii=False)
        try:
            return _run_structure_pass(client, payload, language, total_pages, use_schema=False)
        except Exception as e:
            log.warning("structure_qa chunk schema-less failed: %s — retry with schema", e)
            return _run_structure_pass(client, payload, language, total_pages, use_schema=True)

    groups: list[list[dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as pool:
        for fut in as_completed([pool.submit(_job, c) for c in chunks]):
            try:
                groups.append(fut.result())
            except Exception as e:
                log.warning("structure_qa chunk permanently failed: %s", e)
    out = _merge_and_dedup(groups)
    if not out:
        raise ValueError("Structure pass failed: all chunks returned no items.")
    return out


def structure_qa_with_fallback(
    client: genai.Client,
    pages_payload: dict[str, Any],
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    """Run Stage 2.

    Strategy: chunk-by-default for multi-page payloads (parallel Gemini calls);
    schema-less first, schema fallback on parse failure; if expected_questions
    is provided and the result undercounts, re-chunk as a final retry.
    """
    pages = pages_payload.get("pages", []) if isinstance(pages_payload, dict) else []

    if isinstance(pages, list) and len(pages) > STRUCTURE_CHUNK_THRESHOLD:
        return _run_structure_chunked(client, pages, language, total_pages)

    payload_json = json.dumps(pages_payload, ensure_ascii=False)
    try:
        rows = _run_structure_pass(
            client, payload_json, language, total_pages,
            expected_questions=expected_questions, use_schema=False,
        )
    except Exception as e:
        log.warning("structure_qa schema-less pass failed: %s — retry with schema", e)
        rows = _run_structure_pass(
            client, payload_json, language, total_pages,
            expected_questions=expected_questions, use_schema=True,
        )

    if (
        expected_questions
        and len(rows) < max(1, int(expected_questions * 0.8))
        and isinstance(pages, list)
        and len(pages) >= 2
    ):
        log.warning(
            "structure_qa undercount got=%s expected~%s — chunked retry",
            len(rows), expected_questions,
        )
        try:
            rows = _run_structure_chunked(client, pages, language, total_pages)
        except ValueError as e:
            log.warning("structure_qa chunked retry failed: %s (keeping primary rows)", e)
    return rows
