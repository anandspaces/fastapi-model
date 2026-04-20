"""Orchestrate: PDF → import questions → Gemini answers per question.

No HTTP. Persistence is handled by ``POST /models/answer-booklet`` when the model key was created with ``type=custom``, ``type=custom_with_model``, or ``type=essay`` (same PDF path; essay uses longer generated answers).
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
    booklet_type: str = "custom",
) -> PdfQaPipelineResult:
    """Run full pipeline on *pdf_path*.

    Returns ``{"kind": "no_questions"}`` when no questions are found in the PDF.
    Returns ``{"kind": "ok", "questions": [...]}`` with ``question_id``, ``question_no``,
    ``question``, ``answer`` per item.
    """
    imported = extract_questions_from_pdf(
        pdf_path, api_key, booklet_type=booklet_type
    )
    if not imported:
        return {"kind": "no_questions"}

    filled = fill_answers_for_questions(
        imported, api_key, language, booklet_type=booklet_type
    )
    return {"kind": "ok", "questions": filled}
