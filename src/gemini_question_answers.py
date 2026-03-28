"""Generate a model/reference answer for each imported question (Gemini, text-only).

Used by the PDF import Q+A pipeline. No FastAPI dependencies.
"""

from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types

from src.gemini_extract import MODEL_ID

log = logging.getLogger(__name__)

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


def _answer_prompt(question_text: str, language: str) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Write the answer in Hindi (Devanagari) where appropriate for the subject."
        if lang.startswith("hi")
        else "Write the answer in clear English."
    )
    return f"""You are an expert examiner. The following is an exam question (possibly with sub-parts).

{lang_note}
Produce a concise but complete model answer that would earn full marks: correct reasoning, key terms, and structure.
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
    client: genai.Client, question_text: str, language: str = "en"
) -> str:
    """Single Gemini call: return the model answer string for *question_text*."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[types.Part.from_text(text=_answer_prompt(question_text, language))],
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=_ANSWER_SCHEMA,
        ),
    )
    raw = getattr(response, "text", None) or ""
    if not raw.strip():
        raise RuntimeError("Gemini returned no answer text.")
    return _parse_answer_json(raw)


def fill_answers_for_questions(
    questions: list[dict[str, str]],
    api_key: str,
    language: str = "en",
) -> list[dict[str, str]]:
    """For each imported row (id, questionNo, question_text), append Gemini ``answer``."""
    client = genai.Client(api_key=api_key)
    out: list[dict[str, str]] = []
    for i, q in enumerate(questions):
        qtext = q["question_text"]
        log.info("Generating answer for question %s (%d/%d)", q.get("questionNo"), i + 1, len(questions))
        answer = generate_answer_for_question(client, qtext, language)
        out.append(
            {
                "question_id": q["id"],
                "question_no": q["questionNo"],
                "question": qtext,
                "answer": answer,
            }
        )
    return out
