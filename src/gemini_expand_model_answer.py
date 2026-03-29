"""Expand a draft answer into a full model answer (Gemini, text-only). Stateless; no FastAPI."""

from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types

from src.gemini_extract import MODEL_ID

log = logging.getLogger(__name__)

_EXPAND_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "answer": types.Schema(
            type=types.Type.STRING,
            description="Full model answer for a marking key, meeting the requested length.",
        ),
        "diagram_description": types.Schema(
            type=types.Type.STRING,
            description="Clear diagram guidance for markers; empty if no diagram applies.",
        ),
    },
    required=["answer", "diagram_description"],
    property_ordering=["answer", "diagram_description"],
)


def _word_target(answer_type: str) -> int:
    t = (answer_type or "").strip().lower()
    return 1200 if t == "essay" else 300


def _expand_prompt(
    *,
    answer_type: str,
    question: str,
    draft_answer: str,
    diagram_description: str,
    language: str,
) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Write entirely in Hindi (Devanagari) where appropriate for the subject."
        if lang.startswith("hi")
        else "Write in clear English."
    )
    n_words = _word_target(answer_type)
    diagram_block = (
        diagram_description.strip()
        if diagram_description.strip()
        else "(No diagram notes were provided by the user.)"
    )
    return f"""You are an expert examiner writing a model answer for a marking key.

{lang_note}
Booklet type label (for context only; standard and custom use the same length rules): {answer_type}

Target length for the "answer" field: approximately {n_words} words (within ~15%% is fine). Structure with clear paragraphs. Include correct reasoning, key terms, and what would earn full marks.

Use the exam question and the user's draft/reference answer as the factual base. Expand, polish, and complete the draft; do not invent unrelated content.

For "diagram_description": produce concise marking-key text describing what a correct diagram should show (labels, axes, key features). If the question does not require a diagram or the user gave no useful diagram notes, set "diagram_description" to an empty string.

Question:
{question.strip()}

User draft / reference answer:
{draft_answer.strip()}

User diagram notes:
{diagram_block}

Respond ONLY with JSON matching the schema: one object with keys "answer" (string) and "diagram_description" (string)."""


def _parse_expand_json(text: str) -> tuple[str, str]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    if "answer" not in data:
        raise ValueError("JSON missing 'answer' field")
    ans = str(data["answer"]).strip()
    diag = str(data.get("diagram_description", "")).strip()
    return ans, diag


def expand_model_answer(
    api_key: str,
    *,
    answer_type: str,
    question: str,
    draft_answer: str,
    diagram_description: str,
    language: str = "en",
) -> dict[str, str]:
    """Single Gemini call: return dict with answer and diagram_description."""
    target_words = _word_target(answer_type)
    log.info(
        "expand_model_answer: calling %s type=%s lang=%s target_words=%d q_chars=%d a_chars=%d diagram_chars=%d",
        MODEL_ID,
        answer_type,
        language,
        target_words,
        len(question or ""),
        len(draft_answer or ""),
        len(diagram_description or ""),
    )
    client = genai.Client(api_key=api_key)
    prompt = _expand_prompt(
        answer_type=answer_type,
        question=question,
        draft_answer=draft_answer,
        diagram_description=diagram_description,
        language=language,
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=_EXPAND_SCHEMA,
        ),
    )
    raw = getattr(response, "text", None) or ""
    if not raw.strip():
        log.warning("expand_model_answer: Gemini returned empty text")
        raise RuntimeError("Gemini returned no text.")
    try:
        answer, diagram_desc = _parse_expand_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.exception(
            "expand_model_answer: failed to parse model JSON (raw_len=%d)",
            len(raw),
        )
        raise
    log.info(
        "expand_model_answer: done answer_chars=%d diagram_chars=%d",
        len(answer),
        len(diagram_desc),
    )
    return {
        "answer": answer,
        "diagram_description": diagram_desc,
    }
