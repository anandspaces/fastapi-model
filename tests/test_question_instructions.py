"""Per-question ``instruction_name`` field (schema + read normalization)."""

from __future__ import annotations

from src.gemini_evaluate_student_answers import format_answer_model_as_teacher_instructions
from src.schemas import QuestionPayload
from src.service import _questions_with_instruction_name_defaults


def test_question_payload_instruction_name_default_empty():
    p = QuestionPayload(
        questionNo="Q1",
        title="t",
        desc="d",
        pageNum=1,
        marks=5,
    )
    d = p.model_dump()
    assert d["instruction_name"] == ""


def test_question_payload_instruction_name_round_trip():
    p = QuestionPayload(
        questionNo="Q1",
        title="t",
        desc="d",
        instruction_name="  Use marking scheme section A. ",
        pageNum=1,
        marks=10,
        diagramDescriptions=[],
    )
    assert "marking scheme" in p.model_dump()["instruction_name"]


def test_normalize_legacy_questions_adds_instruction_name():
    raw = [{"id": "q-eng-001", "questionNo": "Q1", "title": "x", "desc": "y"}]
    out = _questions_with_instruction_name_defaults(raw)
    assert out[0]["instruction_name"] == ""


def test_normalize_preserves_instruction_name_string():
    raw = [{"id": "1", "instruction_name": "Notes here"}]
    out = _questions_with_instruction_name_defaults(raw)
    assert out[0]["instruction_name"] == "Notes here"


def test_normalize_none_instruction_name_becomes_empty():
    raw = [{"id": "1", "instruction_name": None}]
    out = _questions_with_instruction_name_defaults(raw)
    assert out[0]["instruction_name"] == ""


def test_teacher_prompt_pairs_booklet_and_instruction_name():
    block = format_answer_model_as_teacher_instructions(
        [
            {
                "questionNo": "Q1",
                "title": "Sample stem",
                "desc": "Ideal answer body from booklet.",
                "instruction_name": "Require three sub-indicators; quote Bery on federalism.",
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

