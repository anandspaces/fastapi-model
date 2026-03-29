"""Generate a model answer from the question only (Gemini, text-only). Stateless; no FastAPI."""

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
        "diagramDescriptions": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="One concise marking-key string per diagram or figure; empty if none.",
        ),
    },
    required=["answer", "diagramDescriptions"],
    property_ordering=["answer", "diagramDescriptions"],
)


def _word_target(answer_type: str) -> int:
    t = (answer_type or "").strip().lower()
    return 1200 if t == "essay" else 300


def _expand_prompt(
    *,
    answer_type: str,
    question: str,
    language: str,
) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Write entirely in Hindi (Devanagari) where appropriate for the subject."
        if lang.startswith("hi")
        else "Write in clear English."
    )
    n_words = _word_target(answer_type)
    return f"""You are an expert examiner writing a model answer for a marking key.

{lang_note}
Booklet type label (for context only; standard and custom use the same length rules): {answer_type}

Target length for the "answer" field: approximately {n_words} words (within ~15%% is fine). Structure with clear paragraphs. Produce a complete model answer that would earn full marks: correct reasoning, key terms, and structure, based only on the exam question below.

For "diagramDescriptions": return a JSON array of strings. If the question requires one or more diagrams, sketches, graphs, maps, flowcharts, or labeled figures, add one short marking-key description per distinct diagram (what to draw, labels, axes, key features). If no diagram is appropriate, return an empty array [].

Question:
{question.strip()}

Respond ONLY with JSON matching the schema: one object with keys "answer" (string) and "diagramDescriptions" (array of strings)."""


def _normalize_diagram_descriptions(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str):
            t = x.strip()
            if t:
                out.append(t)
    return out


def _parse_expand_json(text: str) -> tuple[str, list[str]]:
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
    diags = _normalize_diagram_descriptions(data.get("diagramDescriptions"))
    return ans, diags


def expand_model_answer(
    api_key: str,
    *,
    answer_type: str,
    question: str,
    language: str = "en",
) -> dict[str, str | list[str]]:
    """Single Gemini call: return answer and diagramDescriptions list."""
    target_words = _word_target(answer_type)
    log.info(
        "expand_model_answer: calling %s type=%s lang=%s target_words=%d q_chars=%d",
        MODEL_ID,
        answer_type,
        language,
        target_words,
        len(question or ""),
    )
    client = genai.Client(api_key=api_key)
    prompt = _expand_prompt(
        answer_type=answer_type,
        question=question,
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
        answer, diagram_descs = _parse_expand_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.exception(
            "expand_model_answer: failed to parse model JSON (raw_len=%d)",
            len(raw),
        )
        raise
    log.info(
        "expand_model_answer: done answer_chars=%d diagram_count=%d",
        len(answer),
        len(diagram_descs),
    )
    return {
        "answer": answer,
        "diagramDescriptions": diagram_descs,
    }
