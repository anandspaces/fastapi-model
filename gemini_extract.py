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
- "pageNum": integer, 1-based page number where this question appears.
- "marks": integer, marks for the question (use 0 if not found).
- "diagramDescriptions": array of strings, any descriptions of diagrams/flowcharts/maps mentioned (empty array if none).

Output ONLY a single JSON array of such objects. No markdown, no explanation. If the document has no clear questions, return one object with the whole content in "desc" and pageNum 1, id "q-eng-001", questionNo "Q1", title "Content", marks 0, diagramDescriptions [].
Valid JSON array example: [{"id":"q-eng-001","questionNo":"Q1","title":"...","desc":"...","pageNum":1,"marks":8,"diagramDescriptions":["..."]}]
"""


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


def normalize(obj: dict) -> dict:
    return {
        "id": str(obj.get("id", "q-eng-001")),
        "questionNo": str(obj.get("questionNo", "Q1")),
        "title": str(obj.get("title", "")),
        "desc": str(obj.get("desc", "")),
        "pageNum": int(obj.get("pageNum", 1)),
        "marks": int(obj.get("marks", 0)),
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
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text.")

    raw = extract_json_array(text)
    questions = [normalize(o) for o in raw]
    log.info("[%s] Extracted %d question(s).", label, len(questions))
    return questions
