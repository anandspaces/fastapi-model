"""Per-question ``instructions`` field (schema + read normalization)."""

from __future__ import annotations

from src.gemini_evaluate_student_answers import format_answer_model_as_teacher_instructions
from src.schemas import QuestionPayload
from src.service import _questions_with_instructions_defaults


def test_question_payload_instructions_default_empty():
    p = QuestionPayload(
        questionNo="Q1",
        title="t",
        desc="d",
        pageNum=1,
        marks=5,
    )
    d = p.model_dump()
    assert d["instructions"] == ""


def test_question_payload_instructions_round_trip():
    p = QuestionPayload(
        questionNo="Q1",
        title="t",
        desc="d",
        instructions="  Use marking scheme section A. ",
        pageNum=1,
        marks=10,
        diagramDescriptions=[],
    )
    assert "marking scheme" in p.model_dump()["instructions"]


def test_normalize_legacy_questions_adds_instructions():
    raw = [{"id": "q-eng-001", "questionNo": "Q1", "title": "x", "desc": "y"}]
    out = _questions_with_instructions_defaults(raw)
    assert out[0]["instructions"] == ""


def test_normalize_preserves_instructions_string():
    raw = [{"id": "1", "instructions": "Notes here"}]
    out = _questions_with_instructions_defaults(raw)
    assert out[0]["instructions"] == "Notes here"


def test_normalize_none_instructions_becomes_empty():
    raw = [{"id": "1", "instructions": None}]
    out = _questions_with_instructions_defaults(raw)
    assert out[0]["instructions"] == ""


def test_teacher_prompt_pairs_booklet_and_instructions():
    block = format_answer_model_as_teacher_instructions(
        [
            {
                "questionNo": "Q1",
                "title": "Sample stem",
                "desc": "Ideal answer body from booklet.",
                "instructions": "Require three sub-indicators; quote Bery on federalism.",
                "marks": 10,
                "diagramDescriptions": [],
            }
        ],
        "GS Paper 3",
    )
    assert "Model booklet (ideal answer):" in block
    assert "Ideal answer body from booklet." in block
    assert "Instructions (examiner marking key" in block
    assert "Require three sub-indicators" in block
