"""Session helpers: context manager for service-layer use, FastAPI dependency."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from .base import SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """Open a session, commit on success, rollback on error, always close.

    Drop-in replacement for the old ``with get_conn() as db:`` pattern in
    :mod:`src.service`.
    """
    sess = SessionLocal()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency form. Use as ``db: Session = Depends(get_db)``."""
    with get_session() as sess:
        yield sess
