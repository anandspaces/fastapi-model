"""Filesystem locations for uploaded artifacts.

The actual DB has moved to Postgres (see :mod:`src.db`); these constants stay
because the app still writes raw PDFs to disk alongside the DB metadata row.
"""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
UPLOADS_DIR = DATA_ROOT / "uploads"

# Ensure the upload dir exists on import; cheap and idempotent.
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
