"""ORM models — single source of truth for the Postgres schema.

These classes also drive Alembic ``--autogenerate``. Schema mirrors the prior
SQLite layout (post-ALTER) exactly to keep the SQLite → Postgres switch a pure
transport change. Native Postgres improvements (JSONB on ``questions_json``,
``TIMESTAMP WITH TIME ZONE`` on ``created_at``, real FK on ``owner_user_id``)
are intentionally deferred to a follow-up migration.
"""

from __future__ import annotations

from sqlalchemy import Column, Integer, String, Text

from .base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(String, nullable=False)  # ISO-8601 text, unchanged


class KeyUpload(Base):
    __tablename__ = "key_uploads"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    lang = Column(String, nullable=False)
    pdf_path = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    owner_user_id = Column(String, nullable=True, index=True)
    booklet_type = Column(String, nullable=False, server_default="standard")


class AnswerModel(Base):
    __tablename__ = "answer_models"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    lang = Column(String, nullable=False)
    questions_json = Column(Text, nullable=False)
    question_count = Column(Integer, nullable=False)
    booklet_pdf_path = Column(String, nullable=True)
    created_at = Column(String, nullable=False)
    owner_user_id = Column(String, nullable=True, index=True)
    booklet_type = Column(String, nullable=False, server_default="standard")
    intro_page = Column(Integer, nullable=False, server_default="2")
