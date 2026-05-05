"""Gemini PDF extraction used only by the API (not shared with read_pdf_to_json)."""

import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_ID = "gemini-3.1-pro-preview"
FILE_ACTIVE_TIMEOUT = 180

log = logging.getLogger(__name__)

SCHEMA_PROMPT = """You are given a PDF document (exam or study material). Read all pages carefully.

Extract every question/answer block into a structured list. For each block output exactly one JSON object with these keys (no extra keys):
- "id": string, e.g. "q-eng-001", "q-eng-002" (use question number to form id).
- "questionNo": string, e.g. "Q1", "Q2".
- "title": string, the question title or prompt (one line/sentence).
- "desc": string, the full description or answer text (can be multi-line).
- "instruction_name": string, marking guidance / examiner notes / value-add (objectives, scheme rubric, quotations, supplementary context) SEPARATE from the main answer body in desc — use "" if none.
- "pageNum": integer, 1-based page number where this question appears.
- "marks": integer, the number of marks for this question. Search thoroughly for patterns like "[8]", "(8 marks)", "8 marks", "8m", or any number near the question indicating marks. Never default to 0 — always look carefully. Use 0 only if the document genuinely has no mark indicators anywhere for that question.
- "diagramDescriptions": array of strings, any descriptions of diagrams/flowcharts/maps mentioned (empty array if none).

Output ONLY a single JSON array of such objects. No markdown, no explanation. If the document has no clear questions, return one synthetic row with the whole content in "desc", instruction_name "", pageNum 1, id "q-eng-001", questionNo "Q1", title "Content", marks 0 (no per-question marks in that case), diagramDescriptions [].
Valid JSON array example: [{"id":"q-eng-001","questionNo":"Q1","title":"...","desc":"...","instruction_name":"","pageNum":1,"marks":8,"diagramDescriptions":["..."]}]
"""

_EXTRACT_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "id": types.Schema(type=types.Type.STRING),
            "questionNo": types.Schema(type=types.Type.STRING),
            "title": types.Schema(type=types.Type.STRING),
            "desc": types.Schema(type=types.Type.STRING),
            "instruction_name": types.Schema(
                type=types.Type.STRING,
                description="Marking notes / value-add separate from desc; empty string if none.",
            ),
            "pageNum": types.Schema(type=types.Type.INTEGER),
            "marks": types.Schema(
                type=types.Type.INTEGER,
                description="Per-question mark value from the PDF (not 0 unless truly absent).",
            ),
            "diagramDescriptions": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
        },
        required=[
            "id",
            "questionNo",
            "title",
            "desc",
            "instruction_name",
            "pageNum",
            "marks",
            "diagramDescriptions",
        ],
        property_ordering=[
            "id",
            "questionNo",
            "title",
            "desc",
            "instruction_name",
            "pageNum",
            "marks",
            "diagramDescriptions",
        ],
    ),
)


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to a .env file or environment.")
    return key


def wait_for_file_active(
    client: genai.Client, file: types.File, timeout: int = FILE_ACTIVE_TIMEOUT
) -> types.File:
    name = file.name
    if not name:
        return file
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        f = client.files.get(name=name)
        state = str(getattr(f, "state", "") or "").upper()
        if state == "ACTIVE":
            return f
        if state == "FAILED":
            raise RuntimeError(f"File processing failed: {name}")
        time.sleep(2)
    raise TimeoutError(f"File {name} did not become ACTIVE within {timeout}s")


def extract_json_array(text: str) -> list:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def _parse_marks(raw: object) -> int:
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if raw is None:
        return 0
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else 0


def _assign_page_nums(questions: list[dict]) -> None:
    """First half of questions → pageNum 2, second half → 3 (e.g. 20: Q1–10→2, Q11–20→3)."""
    n = len(questions)
    mid = n // 2
    for i, q in enumerate(questions):
        q["pageNum"] = 2 if i < mid else 3


def _apply_default_marks_fallback(questions: list[dict]) -> None:
    """If Gemini returned marks=0, fallback to position-based defaults.

    Uses the same first/second half split as `_assign_page_nums` (indices [0, mid) vs [mid, n)):
    - first half -> 8 marks
    - second half -> 12 marks
    """
    n = len(questions)
    mid = n // 2
    for i, q in enumerate(questions):
        if int(q.get("marks", 0)) == 0:
            q["marks"] = 8 if i < mid else 12


def normalize(obj: dict) -> dict:
    marks = _parse_marks(obj.get("marks"))
    return {
        "id": str(obj.get("id", "q-eng-001")),
        "questionNo": str(obj.get("questionNo", "Q1")),
        "title": str(obj.get("title", "")),
        "desc": str(obj.get("desc", "")),
        "instruction_name": str(obj.get("instruction_name", "")),
        "pageNum": int(obj.get("pageNum", 1)),
        "marks": marks,
        "diagramDescriptions": (
            [str(x) for x in obj["diagramDescriptions"]]
            if isinstance(obj.get("diagramDescriptions"), list)
            else []
        ),
    }


def process_pdf_path(pdf_path: Path, api_key: str) -> list[dict]:
    client = genai.Client(api_key=api_key)
    label = pdf_path.name

    log.info("[%s] Uploading…", label)
    uploaded = client.files.upload(file=str(pdf_path))
    state = str(getattr(uploaded, "state", "") or "").upper()
    if not uploaded.uri or state == "PROCESSING":
        log.info("[%s] Waiting for file to become active…", label)
        uploaded = wait_for_file_active(client, uploaded)

    if not uploaded.uri:
        raise RuntimeError("File has no URI after upload.")

    log.info("[%s] Calling %s…", label, MODEL_ID)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[uploaded, SCHEMA_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_EXTRACT_RESPONSE_SCHEMA,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text.")

    raw = extract_json_array(text)
    questions = [normalize(o) for o in raw]
    _assign_page_nums(questions)
    _apply_default_marks_fallback(questions)
    log.info("[%s] Extracted %d question(s).", label, len(questions))
    return questions
