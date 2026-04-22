"""Smart OCR extraction of question-answer blocks from student copy PDFs."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from google import genai
from google.genai import types

from src.gemini_copy_ocr import (
    copy_ocr_max_pages,
    copy_ocr_parallel_workers,
    copy_ocr_raster_dpi,
    count_pdf_pages,
    rasterize_pdf_to_png_pages,
)
from src.gemini_extract import MODEL_ID

# ---------------------------------------------------------------------------
# Response schema (unchanged from v2)
# ---------------------------------------------------------------------------

_SMART_ITEM_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "question_id": types.Schema(type=types.Type.INTEGER),
        "question": types.Schema(type=types.Type.STRING),
        "student_answer": types.Schema(type=types.Type.STRING),
        "start_page": types.Schema(type=types.Type.INTEGER),
        "start_y_position_percent": types.Schema(type=types.Type.NUMBER),
        "end_page": types.Schema(type=types.Type.INTEGER),
        "end_y_position_percent": types.Schema(type=types.Type.NUMBER),
        "marking_page": types.Schema(type=types.Type.INTEGER),
        "marking_x_position_percent": types.Schema(type=types.Type.NUMBER),
        "marking_y_position_percent": types.Schema(type=types.Type.NUMBER),
    },
    required=[
        "question_id",
        "question",
        "student_answer",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
)

_SMART_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=_SMART_ITEM_SCHEMA,
)

# ---------------------------------------------------------------------------
# Pass 1 — per-page OCR prompt
# ---------------------------------------------------------------------------


def _build_ocr_prompt(page_num: int, total_pages: int, language: str) -> str:
    lang = (language or "en").strip().lower()
    script_note = (
        "The page may contain Hindi (Devanagari script) and/or English. "
        "Preserve both scripts exactly as written — do NOT transliterate."
        if lang == "hi"
        else (
            "The page may contain English and/or Hindi (Devanagari script). "
            "Preserve both scripts exactly as written — do NOT transliterate."
        )
    )

    return f"""You are an expert OCR engine transcribing a handwritten student exam answer sheet.

This is PAGE {page_num} of {total_pages}.

YOUR TASK:
Transcribe ALL visible text on this page with maximum accuracy and completeness.

RULES:
1. Preserve the original layout using blank lines to separate visual sections (question headers, answer paragraphs, diagrams descriptions, tables).
2. If a question number or label appears (e.g. "Q1", "Question 3(a)", "Ans:", "उत्तर:"), keep it on its own line exactly as written.
3. For diagrams or drawings that cannot be transcribed, write: [DIAGRAM]
4. For tables, transcribe cell content row by row separated by " | ".
5. Do NOT summarise, paraphrase, correct spelling, or add any commentary.
6. Do NOT wrap output in markdown code fences or add any preamble.
7. {script_note}

Output ONLY the raw transcribed text for this page and nothing else."""


# ---------------------------------------------------------------------------
# Pass 2 — structure extraction prompt
# ---------------------------------------------------------------------------


def _build_structure_prompt(language: str, total_pages: int) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Input may contain Hindi and English; preserve script exactly as written."
        if lang == "hi"
        else "Input may contain English and Hindi; preserve script exactly as written."
    )

    return f"""You are an expert AI assistant. You will receive the full OCR text of a student exam answer sheet, \
split into pages with the marker "=== PAGE N ===" at the start of each page's text.

YOUR TASK:
Identify every question attempted by the student and extract:
- The question text or identifier exactly as it appears.
- The student's complete answer exactly as transcribed (do not summarise or shorten).
- The precise page and vertical position (0–100 % from top) where the answer starts and ends.
- The best position to place a grading mark (center of the answer body).

RULES:
1. Total pages in this document: {total_pages}. All page numbers must be in range [1, {total_pages}].
2. start_y_position_percent = vertical % on start_page where the answer text begins (0 = very top, 100 = very bottom).
3. end_y_position_percent   = vertical % on end_page   where the answer text ends.
4. marking_page / marking_x_position_percent / marking_y_position_percent = center of the answer body; \
   marking_x_position_percent should be ~50 (horizontal center) unless the answer is in a clearly offset column.
5. {lang_note}
6. Include EVERY question found. Do not skip any, even if the answer is very short or blank.
7. If a question spans multiple pages, set start_page/end_page accordingly.

CRITICAL JSON FORMATTING (response must be valid JSON):
- Use \\n for line breaks inside string values; never embed raw newline characters inside a JSON string.
- Escape any double-quote inside a string value as \\".
- Do not truncate; every question object must have complete student_answer text.

Return ONLY a JSON array matching the schema — no preamble, no markdown fences, no trailing text."""


# ---------------------------------------------------------------------------
# Helpers (unchanged from v2)
# ---------------------------------------------------------------------------


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if m:
        return m.group(1).strip()
    return t


def _repair_json(text: str) -> str:
    """Best-effort repair for truncated or sloppy JSON."""
    t = _strip_json_fence(text)
    # Remove trailing commas before closing brackets
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # Close unterminated string literal
    stripped = re.sub(r"\\.", "", t)  # remove escape sequences
    if stripped.count('"') % 2 != 0:
        t += '"'
    # Close unbalanced braces / brackets
    open_b = t.count("{") - t.count("}")
    open_sq = t.count("[") - t.count("]")
    if open_b > 0:
        t += "}" * open_b
    if open_sq > 0:
        t += "]" * open_sq
    return t


def _clamp_pct(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(100.0, x))


def _normalize_item(item: dict[str, Any], total_pages: int) -> dict[str, Any]:
    def page_num(v: Any, fallback: int = 1) -> int:
        try:
            p = int(v)
        except (TypeError, ValueError):
            p = fallback
        return max(1, min(total_pages, p))

    start_page = page_num(item.get("start_page"), 1)
    end_page = page_num(item.get("end_page"), start_page)
    if end_page < start_page:
        end_page = start_page
    marking_page = page_num(item.get("marking_page"), start_page)

    try:
        qid = int(item.get("question_id", 0))
    except (TypeError, ValueError):
        qid = 0
    if qid < 1:
        qid = 1

    return {
        "question_id": qid,
        "question": str(item.get("question", "")).strip(),
        "student_answer": str(item.get("student_answer", "")).strip(),
        "start_page": start_page,
        "start_y_position_percent": _clamp_pct(item.get("start_y_position_percent", 0)),
        "end_page": end_page,
        "end_y_position_percent": _clamp_pct(item.get("end_y_position_percent", 100)),
        "marking_page": marking_page,
        "marking_x_position_percent": _clamp_pct(item.get("marking_x_position_percent", 50)),
        "marking_y_position_percent": _clamp_pct(item.get("marking_y_position_percent", 50)),
    }


def _parse_response(raw: str, total_pages: int) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        if not candidate.strip():
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        break
    else:
        msg = str(last_err) if last_err else "invalid JSON"
        log.warning(
            "smart_ocr JSON parse failed: %s raw_prefix=%r",
            msg,
            (raw[:800] + "…") if len(raw) > 800 else raw,
        )
        raise ValueError(msg) from last_err

    if not isinstance(parsed, list):
        raise ValueError("Smart OCR response must be a JSON array.")
    out: list[dict[str, Any]] = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        normalized = _normalize_item(row, total_pages)
        if normalized["student_answer"]:
            out.append(normalized)
    if not out:
        raise ValueError("No question-answer blocks detected.")
    for idx, item in enumerate(out, start=1):
        item["question_id"] = idx
    return out


# ---------------------------------------------------------------------------
# Pass 1 — OCR a single page (runs in thread pool)
# ---------------------------------------------------------------------------


def _ocr_single_page(
    api_key: str,
    png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
) -> str:
    """Return transcribed text for one page, prefixed with the page marker."""
    client = genai.Client(api_key=api_key)
    prompt = _build_ocr_prompt(page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
    )
    text = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            break
        if attempt < 2:
            time.sleep(0.5)

    if not text:
        raise RuntimeError(
            f"Empty OCR response for page {page_num} after 2 attempts."
        )

    return f"=== PAGE {page_num} ===\n{text}"


# ---------------------------------------------------------------------------
# Pass 2 — structure JSON from combined OCR text
# ---------------------------------------------------------------------------


def _structure_qa(
    client: genai.Client,
    page_texts: list[str],
    language: str,
    total_pages: int,
) -> list[dict[str, Any]]:
    """Send plain-text OCR to Gemini and get structured Q&A JSON back."""
    combined = "\n\n".join(page_texts)
    prompt = _build_structure_prompt(language, total_pages)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=combined),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_SMART_RESPONSE_SCHEMA,
    )
    last_err: Exception | None = None
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw:
            raise ValueError("Structure pass returned an empty response.")
        try:
            return _parse_response(raw, total_pages)
        except ValueError as e:
            last_err = e
            log.warning("structure_qa parse attempt %s/2 failed: %s", attempt, e)
            if attempt < 2:
                time.sleep(0.4)
    raise ValueError(f"Structure pass failed after 2 attempts: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Public entry point — signature identical to v1 / v2
# ---------------------------------------------------------------------------


def smart_ocr_extract_student_answers(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Extract structured question-answer spans and marking positions from PDF.

    Two-pass approach:
      Pass 1 — OCR each page independently (parallel, bounded token budget).
      Pass 2 — Structure the combined plain-text into a JSON Q&A array.

    This eliminates the JSON-truncation failures seen when vision + long-form
    JSON generation competed for the same token budget in a single call.
    """
    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    dpi = copy_ocr_raster_dpi()
    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=dpi, request_id=request_id)

    # ------------------------------------------------------------------
    # Pass 1: OCR every page concurrently (bounded workers).
    # Each call is a small vision task — plain text output (no JSON).
    # One client per thread for thread-safe HTTP.
    # ------------------------------------------------------------------
    page_texts: list[str] = [""] * total_pages
    max_workers = max(1, min(copy_ocr_parallel_workers(), total_pages))

    def _ocr_task(
        args: tuple[int, bytes, str, str, int, str],
    ) -> tuple[int, str]:
        idx, png, api_key_inner, lang, tp, rid = args
        page_num = idx + 1
        text = _ocr_single_page(api_key_inner, png, page_num, tp, lang)
        log.info(
            "smart_ocr[%s] ocr pass1 page=%s/%s done chars=%s",
            rid,
            page_num,
            tp,
            len(text),
        )
        return idx, text

    task_args = [
        (i, png_pages[i], api_key, language, total_pages, request_id)
        for i in range(total_pages)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_ocr_task, a): a[0] for a in task_args}
        for future in as_completed(futures):
            idx, text = future.result()
            page_texts[idx] = text

    log.info(
        "smart_ocr[%s] ocr pass1 complete total_chars=%s",
        request_id,
        sum(len(t) for t in page_texts),
    )

    # ------------------------------------------------------------------
    # Pass 2: Structure extraction from plain text only.
    # No images → output token budget goes entirely to the JSON payload.
    # ------------------------------------------------------------------
    structure_client = genai.Client(api_key=api_key)
    rows = _structure_qa(structure_client, page_texts, language, total_pages)

    log.info(
        "smart_ocr[%s] structure pass2 complete questions=%s",
        request_id,
        len(rows),
    )

    return {"items": rows, "page_count": total_pages}
