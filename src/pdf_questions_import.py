"""Extract question list only from a PDF via Gemini (Files API).

This module is separate from ``gemini_extract.process_pdf_path``, which builds full
booklet rows (title, desc, marks, …). Here we only need question text for the
import + per-question answer pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from google import genai
from google.genai import types

from src.gemini_extract import (
    MODEL_ID,
    extract_json_array,
    wait_for_file_active,
)

log = logging.getLogger(__name__)

QUESTIONS_ONLY_PROMPT = """You are given a PDF (exam paper, worksheet, or similar). Read all pages.

Extract every distinct question as one array element. For each question output exactly these keys:
- "id": string, stable id e.g. "q-001", "q-002" (use order in the document).
- "questionNo": string, label as printed e.g. "Q1", "1", "2(a)".
- "questionText": string, the full wording of the question including all sub-parts in reading order.

Rules:
- Do not include model answers, marking schemes, or solutions in questionText — only what the student is asked.
- If the document has no exam-style questions (only cover pages, only answers, or not readable), return an empty JSON array [].
- Do not invent questions.
- Output ONLY a JSON array. No markdown."""

_QUESTIONS_ONLY_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "id": types.Schema(type=types.Type.STRING),
            "questionNo": types.Schema(type=types.Type.STRING),
            "questionText": types.Schema(type=types.Type.STRING),
        },
        required=["id", "questionNo", "questionText"],
        property_ordering=["id", "questionNo", "questionText"],
    ),
)


def _normalize_question_row(obj: object) -> dict[str, str] | None:
    if not isinstance(obj, dict):
        return None
    qid = str(obj.get("id", "")).strip()
    qno = str(obj.get("questionNo", "")).strip()
    text = str(obj.get("questionText", "")).strip()
    if not text:
        return None
    if not qid:
        qid = "q-unknown"
    if not qno:
        qno = "Q?"
    return {"id": qid, "questionNo": qno, "question_text": text}


def extract_questions_from_pdf(pdf_path: Path, api_key: str) -> list[dict[str, str]]:
    """Upload *pdf_path* to Gemini, return a list of ``{id, questionNo, question_text}``.

    Returns an empty list when the model finds no questions.
    """
    client = genai.Client(api_key=api_key)
    label = pdf_path.name

    log.info("[%s] Questions-only import: uploading…", label)
    uploaded = client.files.upload(file=str(pdf_path))
    state = str(getattr(uploaded, "state", "") or "").upper()
    if not uploaded.uri or state == "PROCESSING":
        log.info("[%s] Waiting for file to become active…", label)
        uploaded = wait_for_file_active(client, uploaded)

    if not uploaded.uri:
        raise RuntimeError("File has no URI after upload.")

    log.info("[%s] Questions-only import: calling %s…", label, MODEL_ID)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[uploaded, QUESTIONS_ONLY_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_QUESTIONS_ONLY_SCHEMA,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text.")

    text = text.strip()
    # Schema mode usually returns clean JSON; keep fence strip for safety
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    raw = extract_json_array(text)
    out: list[dict[str, str]] = []
    for o in raw:
        row = _normalize_question_row(o)
        if row:
            out.append(row)

    log.info("[%s] Questions-only import: %d question(s).", label, len(out))
    return out
