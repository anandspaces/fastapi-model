"""
SQLite persistence for the API (pdf_json/api/data/).

Tables:
  - key_uploads: uploaded model-key PDFs (multipart POST /models/key)
  - answer_models: extracted answer booklets (POST /models/answer-booklet)
"""

import json
import sqlite3
import time
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parent / "data"
DB_PATH = _DATA_ROOT / "models.db"
UPLOADS_DIR = _DATA_ROOT / "uploads"


def _conn() -> sqlite3.Connection:
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create key_uploads and answer_models if missing."""
    with _conn() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS key_uploads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                lang TEXT NOT NULL,
                pdf_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answer_models (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                lang TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                question_count INTEGER NOT NULL,
                booklet_pdf_path TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        db.commit()


def insert_key_upload(key_id: str, title: str, lang: str, pdf_path: str) -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as db:
        db.execute(
            "INSERT INTO key_uploads (id, title, lang, pdf_path, created_at) VALUES (?,?,?,?,?)",
            (key_id, title, lang, pdf_path, created),
        )
        db.commit()


def get_key_upload(key_id: str) -> dict | None:
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM key_uploads WHERE id = ?", (key_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "title": row["title"],
        "lang": row["lang"],
        "pdf_path": row["pdf_path"],
    }


def insert_answer_model(
    model_id: str,
    title: str,
    lang: str,
    questions: list,
    booklet_pdf_path: str | None,
) -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    qjson = json.dumps(questions, ensure_ascii=False)
    n = len(questions)
    with _conn() as db:
        db.execute(
            """INSERT INTO answer_models
            (id, title, lang, questions_json, question_count, booklet_pdf_path, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (model_id, title, lang, qjson, n, booklet_pdf_path, created),
        )
        db.commit()


def get_answer_model(model_id: str) -> dict | None:
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM answer_models WHERE id = ?", (model_id,)
        ).fetchone()
    if not row:
        return None
    questions = json.loads(row["questions_json"])
    return {
        "id": row["id"],
        "title": row["title"],
        "lang": row["lang"],
        "question_count": row["question_count"],
        "questions": questions,
        "created_at": row["created_at"],
    }


def list_answer_models_page(page: int, page_size: int) -> tuple[list[dict], int]:
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    offset = (page - 1) * page_size
    with _conn() as db:
        total = db.execute("SELECT COUNT(*) FROM answer_models").fetchone()[0]
        rows = db.execute(
            """SELECT id, title, lang, question_count FROM answer_models
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (page_size, offset),
        ).fetchall()
    items = [
        {"id": r["id"], "title": r["title"], "lang": r["lang"], "question_count": r["question_count"]}
        for r in rows
    ]
    return items, total


def delete_answer_model(model_id: str) -> tuple[bool, str | None]:
    with _conn() as db:
        row = db.execute(
            "SELECT booklet_pdf_path FROM answer_models WHERE id = ?", (model_id,)
        ).fetchone()
        if not row:
            return False, None
        path = row["booklet_pdf_path"]
        db.execute("DELETE FROM answer_models WHERE id = ?", (model_id,))
        db.commit()
    return True, path
