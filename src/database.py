"""Database config, connection helpers, and schema initialization."""

import sqlite3
from pathlib import Path

# Keep DB under project root data/ so docker volume ./data:/app/data works.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_ROOT / "models.db"
UPLOADS_DIR = DATA_ROOT / "uploads"


def get_conn() -> sqlite3.Connection:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create key_uploads and answer_models if missing."""
    with get_conn() as db:
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
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        db.commit()
