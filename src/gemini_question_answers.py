"""Generate a model/reference answer for each imported question (Gemini, text-only).

Used by the PDF import Q+A pipeline. No FastAPI dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from src.gemini_extract import MODEL_ID

log = logging.getLogger(__name__)

# Cap concurrent Gemini answer calls to limit rate spikes (custom booklet path).
_ANSWER_PARALLEL_WORKERS = 5

_ANSWER_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "answer": types.Schema(
            type=types.Type.STRING,
            description="Model or reference answer suitable for a marking key.",
        ),
    },
    required=["answer"],
    property_ordering=["answer"],
)


def _answer_prompt(
    question_text: str, language: str, booklet_type: str = "custom"
) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Write the answer in Hindi (Devanagari) where appropriate for the subject."
        if lang.startswith("hi")
        else "Write the answer in clear English."
    )
    bt = (booklet_type or "custom").strip().lower()
    if bt == "essay":
        length_note = (
            "Produce a full-mark extended response: aim for roughly 1000–1200 words. "
            "Use clear sections (introduction, developed argument or working, conclusion as appropriate). "
            "Include depth, nuance, and explicit reasoning; cite definitions and relationships where relevant."
        )
    else:
        length_note = (
            "Produce a concise but complete model answer that would earn full marks: "
            "correct reasoning, key terms, and structure."
        )
    return f"""You are an expert examiner. The following is an exam question (possibly with sub-parts).

{lang_note}
{length_note}
If the question asks for a diagram or sketch, describe what should appear in words (no image).

Question:
{question_text}

Respond ONLY with JSON matching the schema: one object with key "answer" (string)."""


def _parse_answer_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    data = json.loads(text)
    if isinstance(data, dict) and "answer" in data:
        return str(data["answer"]).strip()
    raise ValueError("Answer JSON missing 'answer' field")


def generate_answer_for_question(
    client: genai.Client,
    question_text: str,
    language: str = "en",
    booklet_type: str = "custom",
) -> str:
    """Single Gemini call: return the model answer string for *question_text*."""
    bt = (booklet_type or "custom").strip().lower()
    cfg_kw: dict = {
        "temperature": 0.3,
        "response_mime_type": "application/json",
        "response_schema": _ANSWER_SCHEMA,
    }
    if bt == "essay":
        cfg_kw["max_output_tokens"] = 8192
    cfg = types.GenerateContentConfig(**cfg_kw)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_text(
                text=_answer_prompt(question_text, language, booklet_type=bt)
            )
        ],
        config=cfg,
    )
    raw = getattr(response, "text", None) or ""
    if not raw.strip():
        raise RuntimeError("Gemini returned no answer text.")
    return _parse_answer_json(raw)


def _fetch_answer_row(
    q: dict[str, str], api_key: str, language: str, booklet_type: str = "custom"
) -> dict[str, str]:
    """One Gemini call per task; own Client (not shared across threads)."""
    qtext = q["question_text"]
    client = genai.Client(api_key=api_key)
    try:
        answer = generate_answer_for_question(
            client, qtext, language, booklet_type=booklet_type
        )
    except Exception:
        log.exception(
            "Answer generation failed for question %s",
            q.get("questionNo", "?"),
        )
        raise
    return {
        "question_id": q["id"],
        "question_no": q["questionNo"],
        "question": qtext,
        "answer": answer,
    }


def fill_answers_for_questions(
    questions: list[dict[str, str]],
    api_key: str,
    language: str = "en",
    booklet_type: str = "custom",
) -> list[dict[str, str]]:
    """For each imported row (id, questionNo, question_text), append Gemini ``answer``.

    *booklet_type* ``essay`` requests long-form model answers; ``custom`` (default) uses concise answers.

    Runs up to ``_ANSWER_PARALLEL_WORKERS`` answer requests in parallel (I/O-bound).
    Result order matches *questions*.
    """
    if not questions:
        return []

    n = len(questions)
    workers = min(_ANSWER_PARALLEL_WORKERS, n)
    log.info("Generating answers for %d question(s) with up to %d parallel workers", n, workers)

    bt = (booklet_type or "custom").strip().lower()

    if workers <= 1:
        return [_fetch_answer_row(q, api_key, language, bt) for q in questions]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda q: _fetch_answer_row(q, api_key, language, bt),
                questions,
            )
        )
