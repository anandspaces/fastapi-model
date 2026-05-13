"""Pytest configuration.

Loads ``.env`` early so non-DB env vars (Gemini API key, JWT secret, …) are
available to tests. Sets ``DATABASE_URL`` from ``TEST_DATABASE_URL`` when one
is provided, otherwise installs a stub URL so importing :mod:`src.db` succeeds
without a real Postgres reachable. Tests that need the DB request the
``db_session`` fixture, which skips automatically when ``TEST_DATABASE_URL``
is unset — that prevents any test from accidentally touching the developer's
real database configured in ``.env``.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

if _TEST_DB_URL:
    # Force every `from src.db ...` import inside the test run to use the test DB.
    os.environ["DATABASE_URL"] = _TEST_DB_URL
else:
    # Stub so ``src.db.base`` import succeeds. We never connect — DB-backed tests
    # are skipped below via the ``db_session`` fixture.
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql://stub:stub@localhost:9999/stub_unused_at_test_time",
    )


@pytest.fixture
def db_session():
    """Yields a SQLAlchemy session bound to ``TEST_DATABASE_URL``.

    Skipped automatically if ``TEST_DATABASE_URL`` is not set. Rolls back on
    teardown so tests don't leave residue in the test DB.
    """
    if not _TEST_DB_URL:
        pytest.skip("TEST_DATABASE_URL not set; DB-backed test skipped")
    from src.db import SessionLocal

    sess = SessionLocal()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()
