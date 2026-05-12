"""Stage 1B — per-page handwriting OCR.

Receives a non-grid PNG (text quality is best on the clean page) plus the page's
``page_type`` from Stage 1A to pick a focus-tuned prompt. Returns a string in
``=== PAGE n ===\\n<text>`` form so downstream stages can rely on stable headers.
"""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from .config import (
    OCR_HTTP_TIMEOUT_MS,
    OCR_MAX_OUTPUT_TOKENS,
    RETRY_BACKOFF_S,
    afc_off,
    finish_reason_name,
    http_opts,
    ocr_model,
    thinking_off,
)
from .parsing import parse_ocr_page_text
from .prompts import ocr_prompt_for_page_type
from .schemas import OCR_PAGE_SCHEMA

log = logging.getLogger(__name__)


def ocr_single_page(
    api_key: str,
    png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
    page_type: str,
) -> str:
    """One Gemini OCR call (with one retry). Returns ``=== PAGE n ===\\n<text>``."""
    client = genai.Client(api_key=api_key)
    prompt = ocr_prompt_for_page_type(page_type, page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=OCR_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=OCR_PAGE_SCHEMA,
        thinking_config=thinking_off(),
        automatic_function_calling=afc_off(),
        http_options=http_opts(OCR_HTTP_TIMEOUT_MS),
    )
    model = ocr_model()
    page_text = ""
    last_raw = ""
    last_fr = "UNKNOWN"
    for attempt in range(1, 3):
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        except Exception as e:
            log.warning("ocr[p%s] attempt=%s call failed: %s", page_num, attempt, e)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        last_raw = (getattr(resp, "text", None) or "").strip()
        last_fr = finish_reason_name(resp)
        if not last_raw:
            log.warning("ocr[p%s] empty response finish_reason=%s", page_num, last_fr)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        try:
            page_text = parse_ocr_page_text(last_raw, page_num)
            break
        except ValueError:
            log.warning(
                "ocr[p%s] parse fail finish_reason=%s prefix=%r",
                page_num, last_fr, last_raw[:200],
            )
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)

    if not page_text:
        raise RuntimeError(
            f"OCR failed for page {page_num} after 2 attempts; "
            f"finish_reason={last_fr} raw={last_raw[:200]!r}"
        )

    return f"=== PAGE {page_num} ===\n{page_text}"
