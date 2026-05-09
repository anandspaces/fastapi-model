"""Pass 2: single Gemini call — group Pass-1 blocks into questions via refs only."""

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

_PASS2_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sections": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "section_name": types.Schema(type=types.Type.STRING),
                    "questions": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "question_id": types.Schema(type=types.Type.INTEGER),
                                "question_block_refs": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.STRING),
                                ),
                                "answer_block_refs": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.STRING),
                                ),
                                "answer_type": types.Schema(type=types.Type.STRING),
                                "is_attempted": types.Schema(type=types.Type.BOOLEAN),
                            },
                            required=[
                                "question_id",
                                "question_block_refs",
                                "answer_block_refs",
                                "answer_type",
                                "is_attempted",
                            ],
                        ),
                    ),
                },
                required=["section_name", "questions"],
            ),
        ),
    },
    required=["sections"],
)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    return m.group(1).strip() if m else t


def _prompt(payload_json: str, language: str, total_pages: int) -> str:
    lang = (language or "en").strip().lower()
    script = "Use the student's language (Hindi or English) when citing fragments." if lang == "hi" else ""
    return f"""You group OCR blocks into examination questions. You receive ONLY JSON — block texts and cell ranges — no images.

TASK:
- Each question references blocks by ``block_id`` strings exactly as given.
- ``question_block_refs``: printed_question / instructions blocks for that stem.
- ``answer_block_refs``: handwritten_answer blocks composing the student's reply (may span multiple pages).
- ``answer_type``: paragraph | word_list | correction.
- ``is_attempted``: false only when there is no substantive handwriting.

RULES:
- Do NOT invent cell IDs — refs only.
- Order ``answer_block_refs`` in reading order.
- Match numbered stems to the intended handwritten continuation across pages.
- Cover pages 1..{total_pages}; headers that are not answers should not appear in answer_block_refs.
- COVER / INTRO / PERSONAL-DETAIL pages (Name, Roll, Date, Centre, Contact, Medium, Test number, Subject, Examiner-use boxes, etc.) must NOT become questions. Skip those blocks entirely — do not create a section, do not create a question_id, do not place them in answer_block_refs.
- Every emitted question MUST have at least one ``printed_question`` block in ``question_block_refs``. Do not fabricate questions from handwritten-only blocks.

{script}

INPUT JSON:
{payload_json}

Return ONLY JSON matching the schema (sections → questions)."""


def parse_pass2_response(raw: str) -> dict[str, Any]:
    text = _strip_json_fence(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Pass2: expected object")
    return data


def run_pass2_structure(
    api_key: str,
    blocks_payload: dict[str, Any],
    language: str,
    total_pages: int,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Single Gemini call returning ``sections``."""
    client = genai.Client(api_key=api_key)
    payload_json = json.dumps(blocks_payload, ensure_ascii=False)
    prompt = _prompt(payload_json, language, total_pages)
    parts = [types.Part.from_text(text=prompt)]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
        response_mime_type="application/json",
        response_schema=_PASS2_SCHEMA,
    )
    last_raw = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(model=MODEL_ID, contents=parts, config=cfg)
        last_raw = (getattr(resp, "text", None) or "").strip()
        if not last_raw:
            time.sleep(0.25)
            continue
        try:
            data = parse_pass2_response(last_raw)
            log.info(
                "smart_ocr_v2[%s] pass2 ok sections=%s",
                request_id,
                len(data.get("sections") or []),
            )
            return data
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("smart_ocr_v2[%s] pass2 parse retry: %s", request_id, e)
            time.sleep(0.25)

    raise ValueError(f"Pass2 structure failed: {last_raw[:300]!r}")
