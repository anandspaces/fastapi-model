"""Persistence service for model keys, answer booklets, users.

All persistence goes through SQLAlchemy ORM sessions (see :mod:`src.db`).
Function signatures and return shapes are preserved verbatim from the prior
SQLite implementation so call sites in :mod:`main` need no changes beyond
import.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from sqlalchemy import case, func

from src.db import AnswerModel, KeyUpload, User, get_session

_Q_ENG_ID = re.compile(r"^q-eng-(\d+)$", re.I)

BOOKLET_TYPES = frozenset({"standard", "custom", "custom_with_model", "essay"})


def _questions_with_instruction_name_defaults(questions: list) -> list:
    """Ensure each question dict has string ``instruction_name`` (API/read parity)."""
    out: list = []
    for q in questions:
        if not isinstance(q, dict):
            out.append(q)
            continue
        d = dict(q)
        ins = d.get("instruction_name")
        d["instruction_name"] = "" if ins is None else str(ins)
        out.append(d)
    return out


def _normalize_booklet_type(raw: str | None) -> str:
    t = (raw or "standard").strip().lower()
    return t if t in BOOKLET_TYPES else "standard"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# key_uploads
# ---------------------------------------------------------------------------


def insert_key_upload(
    key_id: str,
    title: str,
    lang: str,
    pdf_path: str,
    owner_user_id: str,
    booklet_type: str = "standard",
) -> None:
    with get_session() as db:
        db.add(
            KeyUpload(
                id=key_id,
                title=title,
                lang=lang,
                pdf_path=pdf_path,
                created_at=_now_iso(),
                owner_user_id=owner_user_id,
                booklet_type=_normalize_booklet_type(booklet_type),
            )
        )


def get_key_upload(key_id: str, owner_user_id: str) -> dict | None:
    with get_session() as db:
        row = (
            db.query(KeyUpload)
            .filter_by(id=key_id, owner_user_id=owner_user_id)
            .first()
        )
    if row is None:
        return None
    return {
        "id": row.id,
        "title": row.title,
        "lang": row.lang,
        "pdf_path": row.pdf_path,
        "booklet_type": row.booklet_type or "standard",
    }


def update_key_upload(
    key_id: str,
    title: str,
    lang: str,
    owner_user_id: str,
    booklet_type: str | None = None,
) -> tuple[bool, str | None]:
    """Update title/lang (and optionally booklet_type) in key_uploads and answer_models if present."""
    with get_session() as db:
        row = (
            db.query(KeyUpload)
            .filter_by(id=key_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            return False, "Model key not found"

        bt = (
            _normalize_booklet_type(booklet_type)
            if booklet_type is not None
            else (row.booklet_type or "standard")
        )

        row.title = title
        row.lang = lang
        row.booklet_type = bt

        db.query(AnswerModel).filter_by(
            id=key_id, owner_user_id=owner_user_id
        ).update({"title": title, "lang": lang, "booklet_type": bt})
    return True, None


def delete_model_key(
    key_id: str, owner_user_id: str
) -> tuple[bool, str | None, str | None]:
    """Delete key_upload and linked answer_model row by id.

    Returns ``(deleted, key_pdf_path, booklet_pdf_path)``.
    """
    with get_session() as db:
        key_row = (
            db.query(KeyUpload)
            .filter_by(id=key_id, owner_user_id=owner_user_id)
            .first()
        )
        if key_row is None:
            return False, None, None
        key_pdf_path = key_row.pdf_path

        answer_row = (
            db.query(AnswerModel)
            .filter_by(id=key_id, owner_user_id=owner_user_id)
            .first()
        )
        booklet_path = answer_row.booklet_pdf_path if answer_row else None

        db.query(KeyUpload).filter_by(
            id=key_id, owner_user_id=owner_user_id
        ).delete()
        db.query(AnswerModel).filter_by(
            id=key_id, owner_user_id=owner_user_id
        ).delete()
    return True, key_pdf_path, booklet_path


# ---------------------------------------------------------------------------
# answer_models
# ---------------------------------------------------------------------------


def insert_answer_model(
    model_id: str,
    title: str,
    lang: str,
    questions: list,
    booklet_pdf_path: str | None,
    owner_user_id: str,
    booklet_type: str = "standard",
) -> None:
    with get_session() as db:
        db.add(
            AnswerModel(
                id=model_id,
                title=title,
                lang=lang,
                questions_json=json.dumps(questions, ensure_ascii=False),
                question_count=len(questions),
                booklet_pdf_path=booklet_pdf_path,
                created_at=_now_iso(),
                owner_user_id=owner_user_id,
                booklet_type=_normalize_booklet_type(booklet_type),
            )
        )


def upsert_answer_model_from_booklet(
    model_id: str,
    title: str,
    lang: str,
    questions: list,
    booklet_pdf_path: str,
    owner_user_id: str,
    booklet_type: str = "standard",
) -> None:
    """Insert or update answer_models after PDF booklet processing.

    Replaces ``questions_json`` if the row already exists.
    """
    qjson = json.dumps(questions, ensure_ascii=False)
    n = len(questions)
    bt = _normalize_booklet_type(booklet_type)
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is not None:
            row.title = title
            row.lang = lang
            row.questions_json = qjson
            row.question_count = n
            row.booklet_pdf_path = booklet_pdf_path
            row.booklet_type = bt
        else:
            db.add(
                AnswerModel(
                    id=model_id,
                    title=title,
                    lang=lang,
                    questions_json=qjson,
                    question_count=n,
                    booklet_pdf_path=booklet_pdf_path,
                    created_at=_now_iso(),
                    owner_user_id=owner_user_id,
                    booklet_type=bt,
                )
            )


def get_answer_model(model_id: str, owner_user_id: str) -> dict | None:
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
    if row is None:
        return None
    try:
        questions = json.loads(row.questions_json)
    except Exception:
        questions = []
    if isinstance(questions, list):
        questions = _questions_with_instruction_name_defaults(questions)
    return {
        "id": row.id,
        "title": row.title,
        "lang": row.lang,
        "question_count": row.question_count,
        "questions": questions,
        "created_at": row.created_at,
        "booklet_type": str(row.booklet_type or "standard"),
        "intro_page": int(row.intro_page or 2),
    }


def get_question_for_owner(
    model_id: str, question_id: str, owner_user_id: str
) -> tuple[dict | None, str | None]:
    """Return ``(question_dict, None)`` or ``(None, error_message)``."""
    model = get_answer_model(model_id, owner_user_id)
    if not model:
        return None, "Model not found."
    questions = model.get("questions")
    if not isinstance(questions, list):
        return None, "Model has no questions."
    for q in questions:
        if isinstance(q, dict) and q.get("id") == question_id:
            return q, None
    return None, "Question not found for this model."


def list_registered_models(owner_user_id: str) -> list[dict]:
    """Every id from POST /models/key, joined LEFT with answer_models for has_booklet."""
    with get_session() as db:
        rows = (
            db.query(
                KeyUpload.id,
                KeyUpload.title,
                KeyUpload.lang,
                case((AnswerModel.id.is_not(None), 1), else_=0).label("has_booklet"),
                func.coalesce(AnswerModel.booklet_type, KeyUpload.booklet_type).label(
                    "booklet_type"
                ),
            )
            .outerjoin(
                AnswerModel,
                (AnswerModel.id == KeyUpload.id)
                & (AnswerModel.owner_user_id == KeyUpload.owner_user_id),
            )
            .filter(KeyUpload.owner_user_id == owner_user_id)
            .order_by(KeyUpload.created_at.desc())
            .all()
        )
    return [
        {
            "id": r.id,
            "title": r.title,
            "lang": r.lang,
            "has_booklet": bool(r.has_booklet),
            "booklet_type": r.booklet_type,
        }
        for r in rows
    ]


def delete_answer_model(
    model_id: str, owner_user_id: str
) -> tuple[bool, str | None]:
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            return False, None
        path = row.booklet_pdf_path
        db.delete(row)
    return True, path


# ---------------------------------------------------------------------------
# questions inside answer_models (denormalized JSON column)
# ---------------------------------------------------------------------------


def _next_q_eng_id(questions: list) -> str:
    max_n = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        if not isinstance(qid, str):
            continue
        m = _Q_ENG_ID.match(qid.strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"q-eng-{max_n + 1:03d}"


def add_answer_model_question(
    model_id: str,
    owner_user_id: str,
    fields: dict,
) -> tuple[bool, str | None, str | None, int | None]:
    """Append one question; assign next q-eng-NNN id; renumber questionNo to Q1..Qn.

    If no answer_models row exists yet but the key_uploads row does, create the
    answer_models row with this single question and no booklet PDF.
    Returns ``(ok, error, new_question_id, question_count)``.
    """
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            key_row = (
                db.query(KeyUpload)
                .filter_by(id=model_id, owner_user_id=owner_user_id)
                .first()
            )
            if key_row is None:
                return False, "Model key not found for this user.", None, None

            questions: list = []
            new_id = _next_q_eng_id(questions)
            new_obj = {**fields, "id": new_id}
            questions = [new_obj]
            for i, q in enumerate(questions, 1):
                if isinstance(q, dict):
                    q["questionNo"] = f"Q{i}"

            db.add(
                AnswerModel(
                    id=model_id,
                    title=key_row.title,
                    lang=key_row.lang,
                    questions_json=json.dumps(questions, ensure_ascii=False),
                    question_count=1,
                    booklet_pdf_path=None,
                    created_at=_now_iso(),
                    owner_user_id=owner_user_id,
                    booklet_type=_normalize_booklet_type(key_row.booklet_type),
                )
            )
            return True, None, new_id, 1

        try:
            questions = json.loads(row.questions_json)
        except Exception:
            return False, "Invalid questions_json", None, None

        if not isinstance(questions, list):
            return False, "Invalid questions_json", None, None

        new_id = _next_q_eng_id(questions)
        new_obj = {**fields, "id": new_id}
        questions.append(new_obj)

        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                q["questionNo"] = f"Q{i}"

        row.questions_json = json.dumps(questions, ensure_ascii=False)
        row.question_count = len(questions)
    return True, None, new_id, len(questions)


def update_answer_model_question(
    model_id: str, question_id: str, new_question: dict, owner_user_id: str
) -> tuple[bool, str | None]:
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            return False, "Model not found"

        try:
            questions = json.loads(row.questions_json)
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

        row.questions_json = json.dumps(questions, ensure_ascii=False)
        row.question_count = len(questions)
    return True, None


def bulk_patch_answer_model_question_page_marks(
    model_id: str,
    owner_user_id: str,
    items: list[tuple[str, int, int]],
    intro_page: int | None = None,
) -> tuple[bool, str | None, list[str], list[str], int]:
    """Apply pageNum/marks for many question ids. Unknown ids reported in not_found.

    Duplicate question_ids in *items*: last occurrence wins. Returns
    ``(ok, error, updated_question_ids, not_found_question_ids, intro_page)``.
    """
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            return False, "Model not found", [], [], 2

        if not items:
            current_intro = int(row.intro_page or 2)
            if intro_page is not None:
                current_intro = int(intro_page)
                row.intro_page = current_intro
            return True, None, [], [], current_intro

        try:
            questions = json.loads(row.questions_json)
        except Exception:
            return False, "Invalid questions_json", [], [], 2

        if not isinstance(questions, list):
            return False, "Invalid questions_json", [], [], 2

        id_to_index: dict[str, int] = {}
        for i, q in enumerate(questions):
            if isinstance(q, dict) and isinstance(q.get("id"), str) and q["id"].strip():
                id_to_index[q["id"].strip()] = i

        updated_order: list[str] = []
        seen_updated: set[str] = set()
        not_found_order: list[str] = []
        seen_nf: set[str] = set()

        for qid_raw, page_num, marks in items:
            qid = (qid_raw or "").strip()
            if not qid:
                continue
            idx = id_to_index.get(qid)
            if idx is None:
                if qid not in seen_nf:
                    seen_nf.add(qid)
                    not_found_order.append(qid)
                continue
            questions[idx]["pageNum"] = int(page_num)
            questions[idx]["marks"] = int(marks)
            if qid not in seen_updated:
                seen_updated.add(qid)
                updated_order.append(qid)

        next_intro_page = int(row.intro_page or 2)
        if intro_page is not None:
            next_intro_page = int(intro_page)

        row.questions_json = json.dumps(questions, ensure_ascii=False)
        row.question_count = len(questions)
        row.intro_page = next_intro_page
    return True, None, updated_order, not_found_order, next_intro_page


def reorder_answer_model_questions(
    model_id: str, owner_user_id: str, order: list[str]
) -> tuple[bool, str | None, list[dict] | None]:
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            return False, "Model not found.", None

        try:
            questions = json.loads(row.questions_json)
        except Exception:
            return False, "Invalid questions_json.", None

        if not isinstance(questions, list):
            return False, "Invalid questions_json.", None

        ids = [q.get("id") for q in questions if isinstance(q, dict)]
        if len(ids) != len(questions) or any(
            not isinstance(i, str) or not i.strip() for i in ids
        ):
            return False, "Invalid questions_json.", None

        if len(set(order)) != len(order):
            return False, "Invalid order: duplicate ids.", None
        if len(order) != len(questions):
            return False, "Invalid order: missing question ids.", None

        existing = set(ids)
        requested = set(order)
        if requested - existing:
            return False, "Invalid order: unknown question ids.", None
        if existing - requested:
            return False, "Invalid order: missing question ids.", None

        by_id = {q["id"]: q for q in questions}
        arranged = [by_id[qid] for qid in order]

        for i, q in enumerate(arranged, 1):
            q["questionNo"] = f"Q{i}"

        row.questions_json = json.dumps(arranged, ensure_ascii=False)
        row.question_count = len(arranged)
    return True, None, arranged


def delete_answer_model_question(
    model_id: str, question_id: str, owner_user_id: str
) -> tuple[bool, str | None, list[dict] | None]:
    with get_session() as db:
        row = (
            db.query(AnswerModel)
            .filter_by(id=model_id, owner_user_id=owner_user_id)
            .first()
        )
        if row is None:
            return False, "Model not found.", None

        try:
            questions = json.loads(row.questions_json)
        except Exception:
            return False, "Invalid questions_json.", None

        if not isinstance(questions, list):
            return False, "Invalid questions_json.", None

        idx = None
        for i, q in enumerate(questions):
            if isinstance(q, dict) and q.get("id") == question_id:
                idx = i
                break

        if idx is None:
            return False, "Question not found.", None

        questions.pop(idx)
        for i, q in enumerate(questions, 1):
            q["questionNo"] = f"Q{i}"

        row.questions_json = json.dumps(questions, ensure_ascii=False)
        row.question_count = len(questions)
    return True, None, questions


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------


def create_user(username: str, password_hash: str) -> tuple[bool, str | None]:
    user_id = str(uuid.uuid4())
    with get_session() as db:
        existing = db.query(User).filter_by(username=username).first()
        if existing is not None:
            return False, "Username already exists"
        db.add(
            User(
                id=user_id,
                username=username,
                password_hash=password_hash,
                created_at=_now_iso(),
            )
        )
    return True, None


def get_user_by_username(username: str) -> dict | None:
    with get_session() as db:
        row = db.query(User).filter_by(username=username).first()
    if row is None:
        return None
    return {
        "id": row.id,
        "username": row.username,
        "password_hash": row.password_hash,
        "created_at": row.created_at,
    }
