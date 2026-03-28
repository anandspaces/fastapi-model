"""Orchestrate: PDF → import questions → Gemini answers per question.

No database, no HTTP. See route in ``main`` for the API surface.

FUTURE (not implemented): persist the returned payload under a UUID (e.g. answer_models +
key_uploads) and expose GET /models/{{id}} (or a dedicated route) to reload the same
questions + answers without re-running Gemini.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from src.gemini_question_answers import fill_answers_for_questions
from src.pdf_questions_import import extract_questions_from_pdf


class PdfQaNoQuestions(TypedDict):
    kind: Literal["no_questions"]


class PdfQaOk(TypedDict):
    kind: Literal["ok"]
    questions: list[dict[str, str]]


PdfQaPipelineResult = PdfQaNoQuestions | PdfQaOk


def run_pdf_questions_and_answers(
    pdf_path: Path,
    api_key: str,
    language: str = "en",
) -> PdfQaPipelineResult:
    """Run full pipeline on *pdf_path*.

    Returns ``{"kind": "no_questions"}`` when no questions are found in the PDF.
    Returns ``{"kind": "ok", "questions": [...]}`` with ``question_id``, ``question_no``,
    ``question``, ``answer`` per item.
    """
    imported = extract_questions_from_pdf(pdf_path, api_key)
    if not imported:
        return {"kind": "no_questions"}

    filled = fill_answers_for_questions(imported, api_key, language)
    return {"kind": "ok", "questions": filled}
