"""Smart-OCR evaluator checkLevel (Moderate | Hard), Dart GeminiService parity."""

from __future__ import annotations

from src.gemini_evaluate_student_answers import (
    evaluation_strictness_instruction,
    normalize_evaluation_check_level,
)
from src.gemini_evaluate_student_answers import (
    _build_evaluation_prompt as build_eval_prompt,
)


def test_normalize_check_level_accepted_variants():
    assert normalize_evaluation_check_level("Moderate") == "Moderate"
    assert normalize_evaluation_check_level("moderate") == "Moderate"
    assert normalize_evaluation_check_level("Hard") == "Hard"
    assert normalize_evaluation_check_level("HARD") == "Hard"


def test_normalize_check_level_rejects_unknown():
    assert normalize_evaluation_check_level("easy") == ""


def test_normalize_empty_defaults_to_moderate():
    assert normalize_evaluation_check_level("") == "Moderate"


def test_strictness_lines_match_mobile_semantics():
    mod = evaluation_strictness_instruction("Moderate")
    hard = evaluation_strictness_instruction("Hard")
    assert "MODERATE" in mod.upper() or "moderate" in mod.lower()
    assert "50%" in mod
    assert "HARD" in hard.upper()
    assert "50%" in hard
    assert "extremely strict" in hard.lower()


def test_evaluation_prompt_contains_strictness_when_hard():
    p = build_eval_prompt("Test", "Q1...", "[]", check_level="Hard")
    assert "EVALUATION STRICTNESS: HARD" in p


def test_evaluation_prompt_contains_moderate_by_default():
    p = build_eval_prompt("Test", "Q1...", "[]")
    assert "EVALUATION STRICTNESS: MODERATE" in p
