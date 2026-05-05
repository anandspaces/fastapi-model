"""get_question_for_owner helper."""

from src.service import get_question_for_owner


def test_get_question_for_owner_missing_model() -> None:
    q, err = get_question_for_owner("nonexistent-id", "q-eng-001", "user-1")
    assert q is None
    assert err == "Model not found."

