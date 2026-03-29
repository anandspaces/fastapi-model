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
                created_at TEXT NOT NULL,
                owner_user_id TEXT
            );
            CREATE TABLE IF NOT EXISTS answer_models (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                lang TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                question_count INTEGER NOT NULL,
                booklet_pdf_path TEXT,
                created_at TEXT NOT NULL,
                owner_user_id TEXT
            );
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        # Ensure ownership columns exist for older databases.
        key_cols = {
            row["name"] for row in db.execute("PRAGMA table_info(key_uploads)").fetchall()
        }
        if "owner_user_id" not in key_cols:
            db.execute("ALTER TABLE key_uploads ADD COLUMN owner_user_id TEXT")

        key_cols = {
            row["name"] for row in db.execute("PRAGMA table_info(key_uploads)").fetchall()
        }
        if "booklet_type" not in key_cols:
            db.execute(
                "ALTER TABLE key_uploads ADD COLUMN booklet_type TEXT NOT NULL DEFAULT 'standard'"
            )

        answer_cols = {
            row["name"] for row in db.execute("PRAGMA table_info(answer_models)").fetchall()
        }
        if "owner_user_id" not in answer_cols:
            db.execute("ALTER TABLE answer_models ADD COLUMN owner_user_id TEXT")

        answer_cols = {
            row["name"] for row in db.execute("PRAGMA table_info(answer_models)").fetchall()
        }
        if "booklet_type" not in answer_cols:
            db.execute(
                "ALTER TABLE answer_models ADD COLUMN booklet_type TEXT NOT NULL DEFAULT 'standard'"
            )

        db.commit()
