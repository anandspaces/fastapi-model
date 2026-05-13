"""get_question_for_owner helper.

DB-backed test — gated on TEST_DATABASE_URL via the ``db_session`` fixture
declared in tests/conftest.py. Skipped automatically when that env var is
absent.
"""

from src.service import get_question_for_owner


def test_get_question_for_owner_missing_model(db_session) -> None:
    q, err = get_question_for_owner("nonexistent-id", "q-eng-001", "user-1")
    assert q is None
    assert err == "Model not found."
