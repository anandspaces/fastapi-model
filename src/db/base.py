"""SQLAlchemy engine, sessionmaker, and declarative base.

DATABASE_URL is read from the environment at import time. The bare
``postgresql://`` scheme is rewritten to ``postgresql+psycopg://`` so the user's
plain connection string from ``.env`` works without manual driver suffixing.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _normalize_url(raw: str) -> str:
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


DATABASE_URL = _normalize_url(os.environ.get("DATABASE_URL", "").strip())
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured; set it in .env "
        "(e.g. postgresql://user:pass@host:5432/dbname?sslmode=require)."
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in :mod:`src.db.models`."""
