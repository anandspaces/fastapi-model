"""SQLite persistence service for model keys and answer booklets."""

import json
import time
import uuid

from src.database import UPLOADS_DIR, get_conn


def insert_key_upload(key_id: str, title: str, lang: str, pdf_path: str) -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with get_conn() as db:
        db.execute(
            "INSERT INTO key_uploads (id, title, lang, pdf_path, created_at) VALUES (?,?,?,?,?)",
            (key_id, title, lang, pdf_path, created),
        )
        db.commit()


def get_key_upload(key_id: str) -> dict | None:
    with get_conn() as db:
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


def update_key_upload(key_id: str, title: str, lang: str) -> tuple[bool, str | None]:
    """Update title/lang in key_uploads and linked answer_models row (same id) if present."""
    with get_conn() as db:
        existing = db.execute(
            "SELECT 1 FROM key_uploads WHERE id = ?", (key_id,)
        ).fetchone()
        if not existing:
            return False, "Model key not found"

        db.execute(
            "UPDATE key_uploads SET title = ?, lang = ? WHERE id = ?",
            (title, lang, key_id),
        )
        db.execute(
            "UPDATE answer_models SET title = ?, lang = ? WHERE id = ?",
            (title, lang, key_id),
        )
        db.commit()
    return True, None


def delete_model_key(key_id: str) -> tuple[bool, str | None, str | None]:
    """Delete key_upload and linked answer_model row by id.

    Returns (deleted, key_pdf_path, booklet_pdf_path).
    """
    with get_conn() as db:
        key_row = db.execute(
            "SELECT pdf_path FROM key_uploads WHERE id = ?", (key_id,)
        ).fetchone()
        if not key_row:
            return False, None, None

        answer_row = db.execute(
            "SELECT booklet_pdf_path FROM answer_models WHERE id = ?", (key_id,)
        ).fetchone()
        booklet_path = answer_row["booklet_pdf_path"] if answer_row else None

        db.execute("DELETE FROM key_uploads WHERE id = ?", (key_id,))
        db.execute("DELETE FROM answer_models WHERE id = ?", (key_id,))
        db.commit()
    return True, key_row["pdf_path"], booklet_path


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
    with get_conn() as db:
        db.execute(
            """INSERT INTO answer_models
            (id, title, lang, questions_json, question_count, booklet_pdf_path, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (model_id, title, lang, qjson, n, booklet_pdf_path, created),
        )
        db.commit()


def get_answer_model(model_id: str) -> dict | None:
    with get_conn() as db:
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


def list_registered_models() -> list[dict]:
    """Every id from POST /models/key (key_uploads), with title/lang.

    Includes rows before answer booklet is posted. ``has_booklet`` is True once
    that id exists in answer_models.
    """
    with get_conn() as db:
        rows = db.execute(
            """
            SELECT k.id, k.title, k.lang,
                   CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END AS has_booklet
            FROM key_uploads k
            LEFT JOIN answer_models a ON a.id = k.id
            ORDER BY k.created_at DESC
            """
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "lang": r["lang"],
            "has_booklet": bool(r["has_booklet"]),
        }
        for r in rows
    ]


def delete_answer_model(model_id: str) -> tuple[bool, str | None]:
    with get_conn() as db:
        row = db.execute(
            "SELECT booklet_pdf_path FROM answer_models WHERE id = ?", (model_id,)
        ).fetchone()
        if not row:
            return False, None
        path = row["booklet_pdf_path"]
        db.execute("DELETE FROM answer_models WHERE id = ?", (model_id,))
        db.commit()
    return True, path


def update_answer_model_question(
    model_id: str, question_id: str, new_question: dict
) -> tuple[bool, str | None]:
    with get_conn() as db:
        row = db.execute(
            "SELECT questions_json FROM answer_models WHERE id = ?", (model_id,)
        ).fetchone()
        if not row:
            return False, "Model not found"

        try:
            questions = json.loads(row["questions_json"])
        except Exception:
            return False, "Invalid questions_json"

        if not isinstance(questions, list):
            return False, "Invalid questions_json"

        idx = None
        for i, q in enumerate(questions):
            if isinstance(q, dict) and q.get("id") == question_id:
                idx = i
                break

        if idx is None:
            return False, "Question not found"

        new_obj = dict(new_question)
        new_obj["id"] = question_id
        questions[idx] = new_obj

        qjson = json.dumps(questions, ensure_ascii=False)
        db.execute(
            "UPDATE answer_models SET questions_json = ?, question_count = ? WHERE id = ?",
            (qjson, len(questions), model_id),
        )
        db.commit()
    return True, None


def create_user(username: str, password_hash: str) -> tuple[bool, str | None]:
    user_id = str(uuid.uuid4())
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with get_conn() as db:
        existing = db.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            return False, "Username already exists"
        db.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?,?,?,?)",
            (user_id, username, password_hash, created),
        )
        db.commit()
    return True, None


def get_user_by_username(username: str) -> dict | None:
    with get_conn() as db:
        row = db.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
    }
