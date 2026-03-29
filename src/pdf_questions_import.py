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

_QUESTIONS_ONLY_PROMPT_DOCUMENT_ORDER = """You are given a PDF (exam paper, worksheet, or similar). Read all pages.

Extract every distinct question as one array element. For each question output exactly these keys:
- "id": string, stable id e.g. "q-001", "q-002" (use order in the document).
- "questionNo": string, label as printed e.g. "Q1", "1", "2(a)".
- "questionText": string, the full wording of the question including all sub-parts in reading order.

Rules:
- Do not include model answers, marking schemes, or solutions in questionText — only what the student is asked.
- If the document has no exam-style questions (only cover pages, only answers, or not readable), return an empty JSON array [].
- Do not invent questions.
- Output ONLY a JSON array. No markdown."""

_QUESTIONS_ONLY_PROMPT_BOOKLET_ORDERED = """You are given a PDF (exam paper, answer booklet, or worksheet). Read all pages.

Questions may appear in ANY order on the page (e.g. Q1, then Q5, then Q7 if the student or layout jumped around). Extract every distinct question as one array element.

For each question output exactly these keys:
- "id": string — assign "q-001", "q-002", … in ascending LOGICAL question order (see below), NOT in the order blocks appear on the page.
- "questionNo": string — the label EXACTLY as printed e.g. "Q1", "5", "2(a)"; do not renumber or rename.
- "questionText": string — the full wording of the question including all sub-parts in reading order.

Ordering rule for the JSON array:
- Sort questions by ascending exam order: primary number first (1 before 10), then sub-parts (e.g. 2, then 2(a), then 2(b), then 3, then 5, then 7).
- The FIRST item in the array after sorting must have id "q-001", the second "q-002", and so on.

Other rules:
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


def _question_no_sort_key(question_no: str) -> tuple:
    """Sort key for labels like Q1, 10, 2(a) — numeric chunks as int, letters lowercased."""
    s = re.sub(r"^Q\s*", "", str(question_no).strip(), flags=re.I)
    parts: list = []
    for m in re.finditer(r"\d+|[a-zA-Z]+", s):
        g = m.group()
        if g.isdigit():
            parts.append(int(g))
        else:
            parts.append(g.lower())
    return tuple(parts) if parts else (999_999, str(question_no).lower())


def _reorder_and_renumber_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stable logical order by questionNo; ids become q-001, q-002, …"""
    if not rows:
        return rows
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda ix_row: (_question_no_sort_key(ix_row[1]["questionNo"]), ix_row[0]))
    out: list[dict[str, str]] = []
    for i, (_, row) in enumerate(indexed, start=1):
        new_row = {**row, "id": f"q-{i:03d}"}
        out.append(new_row)
    return out


def extract_questions_from_pdf(
    pdf_path: Path,
    api_key: str,
    booklet_type: str = "custom",
) -> list[dict[str, str]]:
    """Upload *pdf_path* to Gemini, return a list of ``{id, questionNo, question_text}``.

    Returns an empty list when the model finds no questions.

    For ``booklet_type`` ``custom`` or ``essay``, the prompt asks Gemini to output
    questions in logical order and the server re-sorts by ``questionNo`` as a safety net.
    """
    bt = (booklet_type or "custom").strip().lower()
    use_booklet_order = bt in ("custom", "essay")
    prompt = (
        _QUESTIONS_ONLY_PROMPT_BOOKLET_ORDERED
        if use_booklet_order
        else _QUESTIONS_ONLY_PROMPT_DOCUMENT_ORDER
    )

    client = genai.Client(api_key=api_key)
    label = pdf_path.name

    log.info(
        "[%s] Questions-only import: uploading… booklet_type=%s booklet_order=%s",
        label,
        bt,
        use_booklet_order,
    )
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
        contents=[uploaded, prompt],
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

    if use_booklet_order and out:
        out = _reorder_and_renumber_ids(out)
        log.info(
            "[%s] Questions-only import: reordered %d question(s) by questionNo; ids set to q-001…",
            label,
            len(out),
        )

    log.info("[%s] Questions-only import: %d question(s).", label, len(out))
    return out
