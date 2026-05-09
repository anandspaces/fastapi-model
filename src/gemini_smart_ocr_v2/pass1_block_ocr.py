"""Pass 1: per-page block OCR with cell-tagged ranges (overlay JPEG)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from cell_grid_service_v4 import PageCellGrid

from src.gemini_extract import MODEL_ID

from .snap import snap_blocks

log = logging.getLogger(__name__)

_PASS1_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "page": types.Schema(type=types.Type.INTEGER),
        "page_type": types.Schema(
            type=types.Type.STRING,
            description="PARAGRAPH | WORD_LIST | CORRECTION | DUPLICATE | UNKNOWN",
        ),
        "is_cover_page": types.Schema(
            type=types.Type.BOOLEAN,
            description=(
                "True if this page is purely a cover/intro/personal-detail sheet "
                "(Name/Roll/Date/Centre/Contact/Medium/Test number/Subject/Examiner-use) "
                "with NO printed exam questions and NO real handwritten answers."
            ),
        ),
        "blocks": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "block_id": types.Schema(type=types.Type.STRING),
                    "kind": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "printed_question | handwritten_answer | marking_box | "
                            "header_footer | instructions | boilerplate | crossed_out"
                        ),
                    ),
                    "question_label": types.Schema(type=types.Type.STRING),
                    "text": types.Schema(type=types.Type.STRING),
                    "cells": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    ),
                },
                required=["block_id", "kind", "text", "cells"],
            ),
        ),
    },
    required=["page", "page_type", "is_cover_page", "blocks"],
)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    return m.group(1).strip() if m else t


def _pass1_prompt(page_num: int, total_pages: int, language: str) -> str:
    lang = (language or "en").strip().lower()
    script = (
        "Hindi (Devanagari) and English may appear."
        if lang == "hi"
        else "English and/or Hindi may appear."
    )
    return f"""You see ONE overlay page (page {page_num} of {total_pages}) from a handwritten exam answer book.
The image shows the original sheet with a cell grid; each cell is labelled (A1, B1, …). GREEN cells are writable; WHITE cells contain ink.

{script}

TASK — emit ONE JSON object (schema enforced):
- ``page_type``: coarse layout guess (PARAGRAPH, WORD_LIST, CORRECTION, DUPLICATE, UNKNOWN).
- ``is_cover_page``: TRUE only when the entire page is personal/cover info (Name, Roll, Date, Centre, Contact, Medium, Test number, Subject, Examiner-use boxes) with NO printed exam questions AND NO real handwritten answers. Otherwise FALSE — even if the page also contains a small header strip with student details.
- ``blocks``: ordered TOP-TO-BOTTOM. Each block is ONE logical unit (paragraph, list block, printed stem, header strip, score box — NOT one sentence each).

For EACH block set:
- ``block_id``: stable id ``p{page_num}-b<seq>`` e.g. p3-b1, p3-b2 (seq starts at 1 per page).
- ``kind``:
  - printed_question — printed stem/instructions from the paper.
  - handwritten_answer — student's handwriting (cells MUST overlap GREEN/writable cells only after snapping).
  - marking_box — printed score / marks entry rectangle (often top-right).
  - header_footer — roll number, page instructions, session labels.
  - instructions | boilerplate | crossed_out — as appropriate.

- ``question_label``: short label like "Que 1" when ``kind`` is printed_question, else "".
- ``text``: exact transcription for handwritten/printed content; "" for empty marking_box.
- ``cells``: list of SINGLE-ROW ranges "COLstartROW:COLendROW" e.g. "E13:S13", "E14:T14".
  Cover exactly the cells the block occupies on THIS overlay. Multi-row = multiple strings.

Rules:
- Handwritten_answer ranges must lie on writable (green) cells in the overlay.
- Do not invent cell IDs outside the printed grid.
- Keep block count reasonable (merge micro-lines into paragraphs).

Return ONLY JSON matching the schema."""


def parse_pass1_response(raw: str, page_num: int) -> dict[str, Any]:
    text = _strip_json_fence(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Pass1 page {page_num}: invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"Pass1 page {page_num}: expected object")
    return data


def run_pass1_page(
    api_key: str,
    overlay_jpeg: bytes,
    grid: PageCellGrid,
    page_num: int,
    total_pages: int,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Gemini call for one page; returns snapped page dict with ``blocks``."""
    client = genai.Client(api_key=api_key)
    prompt = _pass1_prompt(page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=overlay_jpeg, mime_type="image/jpeg"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=32768,
        response_mime_type="application/json",
        response_schema=_PASS1_SCHEMA,
    )
    last_raw = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(model=MODEL_ID, contents=parts, config=cfg)
        last_raw = (getattr(resp, "text", None) or "").strip()
        if not last_raw:
            time.sleep(0.25)
            continue
        try:
            data = parse_pass1_response(last_raw, page_num)
        except ValueError:
            time.sleep(0.25)
            continue
        blocks_raw = data.get("blocks")
        if not isinstance(blocks_raw, list):
            data["blocks"] = []
        else:
            data["blocks"] = snap_blocks(blocks_raw, grid)
        data["page"] = page_num
        log.info(
            "smart_ocr_v2[%s] pass1 page=%s/%s blocks=%s",
            request_id,
            page_num,
            total_pages,
            len(data["blocks"]),
        )
        return data

    raise ValueError(f"Pass1 failed page {page_num}: {last_raw[:200]!r}")
