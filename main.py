import asyncio
import json
from datetime import datetime, timedelta, timezone
import logging
import re
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from google import genai
from src.gemini_analyse import (
    analyse_cached_ocr,
    analyse_intro_page,
    analyse_pages,
    generate_combined_review,
)
from src.custom_booklet_storage import custom_qna_rows_to_canonical_questions
from src.gemini_copy_ocr import (
    copy_ocr_max_bytes,
    ocr_essay_copy_pdf,
    ocr_essay_copy_pdf_rasterized,
)
from src.gemini_smart_ocr import smart_ocr_extract_student_answers
from src.gemini_evaluate_student_answers import (
    evaluate_student_answers_against_model,
    format_answer_model_as_teacher_instructions,
    merge_evaluations_into_items,
    normalize_evaluation_check_level,
    student_items_for_grading,
)
from src.gemini_expand_model_answer import expand_model_answer
from src.gemini_extract import load_api_key, process_pdf_path
from src.free_space_service import (
    analyze_pdf_free_space,
    api_response_to_page_zones,
    page_zones_to_api_response,
    snap_items_annotations,
)
from src.pdf_qa_pipeline import run_pdf_questions_and_answers
from src.database import UPLOADS_DIR, init_db
from src.service import (
    BOOKLET_TYPES,
    _normalize_booklet_type,
    add_answer_model_question,
    bulk_patch_answer_model_question_page_marks,
    create_user,
    delete_answer_model,
    delete_answer_model_question,
    delete_model_key,
    get_answer_model,
    get_key_upload,
    get_user_by_username,
    insert_key_upload,
    list_registered_models,
    reorder_answer_model_questions,
    update_answer_model_question,
    update_key_upload,
    upsert_answer_model_from_booklet,
)
from dotenv import load_dotenv
from src.schemas import (
    AuthRequest,
    CachedOcrRequest,
    CombinedReviewRequest,
    ExpandModelAnswerRequest,
    BulkQuestionPageMarksPayload,
    QuestionPayload,
    ReorderQuestionsPayload,
    TokenData,
)


class ModelKeyPayload(BaseModel):
    title: str
    lang: str
    booklet_type: str | None = None


class SnapAnnotationsRequest(BaseModel):
    items: list[dict]
    pages: list[dict]


load_dotenv()

log = logging.getLogger(__name__)
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_ALGORITHM = "HS256"
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "60"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    init_db()
    yield


app = FastAPI(
    title="PDF model keys & answer booklets",
    version="0.1.0",
    lifespan=lifespan,
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
    },
)


_PUBLIC_OPENAPI_ROUTES = frozenset(
    {
        ("GET", "/"),
        ("POST", "/auth/signup"),
        ("POST", "/auth/login"),
    }
)


def custom_openapi() -> dict:
    """Expose JWT Bearer in OpenAPI so Swagger /docs Authorize sends the header."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=getattr(app, "version", "0.1.0"),
        openapi_version=app.openapi_version,
        description=getattr(app, "description", None),
        routes=app.routes,
    )
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "HTTPBearer"
    ] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "JWT from POST /auth/login response data.accessToken. "
            "Paste only the token string; Swagger adds the Bearer prefix."
        ),
    }
    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue
            if not isinstance(operation, dict):
                continue
            route_key = (method.upper(), path)
            if route_key in _PUBLIC_OPENAPI_ROUTES:
                continue
            operation["security"] = [{"HTTPBearer": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ok(message: str, **fields: object) -> dict:
    return {"status": 1, "message": message, "data": {**fields}}


def _err(message: str, **extra: object) -> dict:
    return {"status": 0, "message": message, "data":{**extra}}


def _auth_err(
    message: str = "Unauthorized.",
    http_status_code: int = 401,
    **extra: object,
) -> JSONResponse:
    return JSONResponse({"status": -1, "message": message, "data": {**extra}}, status_code=http_status_code)


def _is_pdf(filename: str | None, content_type: str | None) -> bool:
    if filename and re.search(r"\.pdf$", filename, re.I):
        return True
    if content_type and "pdf" in content_type.lower():
        return True
    return False


def _hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    return pwd_ctx.verify(password, password_hash)


def _create_access_token(username: str) -> TokenData:
    if not JWT_SECRET:
        raise ValueError("JWT_SECRET is not configured.")
    expires_delta = timedelta(minutes=JWT_EXPIRES_MINUTES)
    exp = datetime.now(timezone.utc) + expires_delta
    token = jwt.encode(
        {"sub": username, "exp": exp},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    return TokenData(accessToken=token, expiresIn=int(expires_delta.total_seconds()))



def _require_auth_username(request: Request) -> tuple[str | None, JSONResponse | None]:
    auth = request.headers.get("Authorization", "").strip()
    if not auth:
        return None, _auth_err("Authorization header is required.")
    if not auth.lower().startswith("bearer "):
        return None, _auth_err("Authorization header must be Bearer token.")
    token = auth[7:].strip()
    if not token:
        return None, _auth_err("Bearer token is required.")
    if not JWT_SECRET:
        return None, _auth_err("JWT_SECRET is not configured.")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except InvalidTokenError:
        return None, _auth_err("Invalid or expired token.")

    username = str(payload.get("sub", "")).strip()
    if not username:
        return None, _auth_err("Invalid token payload.")
    return username, None


def _require_auth_user(request: Request) -> tuple[dict | None, JSONResponse | None]:
    username, err = _require_auth_username(request)
    if err:
        return None, err
    user = get_user_by_username(username or "")
    if not user:
        return None, _auth_err("User not found.")
    return user, None


@app.get("/")
async def get_root() -> JSONResponse:
    """Health check endpoint that confirms the API service is running."""
    return JSONResponse("API is running successfully!")


@app.post("/auth/signup")
async def auth_signup(payload: AuthRequest) -> JSONResponse:
    """Creates a user account and returns an authentication token."""
    username = payload.username.strip()
    password = payload.password
    if not username or not password:
        return JSONResponse(_err("username and password are required."))
    if len(password) < 6:
        return JSONResponse(_err("password must be at least 6 characters."))

    ok, reason = create_user(username, _hash_password(password))
    if not ok:
        return JSONResponse(_err(reason or "Signup failed."))
    try:
        token = _create_access_token(username)
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    return JSONResponse(
        _ok(
            "Signup successful.",
            accessToken=token.accessToken,
            tokenType=token.tokenType,
            expiresIn=token.expiresIn,
            username=username,
        )
    )


@app.post("/auth/login")
async def auth_login(payload: AuthRequest) -> JSONResponse:
    """Validates credentials and issues an authentication token."""
    username = payload.username.strip()
    password = payload.password
    if not username or not password:
        return JSONResponse(_err("username and password are required."))

    user = get_user_by_username(username)
    if not user or not _verify_password(password, user["password_hash"]):
        return JSONResponse(_err("Invalid username or password."))

    try:
        token = _create_access_token(username)
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    return JSONResponse(
        _ok(
            "Login successful.",
            accessToken=token.accessToken,
            tokenType=token.tokenType,
            expiresIn=token.expiresIn,
            username=username,
        )
    )

@app.post("/models/key")
async def post_model_key(
    request: Request,
    title: str = Form(...),
    lang: str = Form(...),
    booklet_type_param: str = Form("standard", alias="type"),
) -> JSONResponse:
    """Creates a model key record and stores key metadata for the user."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    title, lang = title.strip(), lang.strip()
    if not title or not lang:
        return JSONResponse(_err("title and lang are required (non-empty)."))
    bt_raw = (booklet_type_param or "standard").strip().lower()
    if bt_raw not in BOOKLET_TYPES:
        return JSONResponse(
            _err(
                'type must be "standard", "custom", "custom_with_model", or "essay".'
            )
        )
    key_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"key_{key_id}.pdf"
    try:
        insert_key_upload(key_id, title, lang, str(dest), user["id"], booklet_type=bt_raw)
    except Exception as e:
        log.exception("key upload failed")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))
    return JSONResponse(
        _ok(
            "Key uploaded successfully",
            id=key_id,
            title=title,
            lang=lang,
            booklet_type=bt_raw,
        )
    )


@app.put("/models/key/{key_id}")
async def put_model_key(
    request: Request, key_id: str, payload: ModelKeyPayload
) -> JSONResponse:
    """Updates key metadata such as title, language, and booklet type."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    title, lang = payload.title.strip(), payload.lang.strip()

    if not title or not lang:
        return JSONResponse(_err("title and lang are required (non-empty)."))

    update_kw: dict = {}
    dumped = payload.model_dump(exclude_unset=True)
    if "booklet_type" in dumped:
        v = dumped["booklet_type"]
        if v is None:
            update_kw["booklet_type"] = None
        else:
            vv = str(v).strip().lower()
            if vv not in BOOKLET_TYPES:
                return JSONResponse(
                    _err(
                        'booklet_type must be "standard", "custom", "custom_with_model", or "essay".'
                    )
                )
            update_kw["booklet_type"] = vv

    ok, reason = update_key_upload(key_id, title, lang, user["id"], **update_kw)
    log.info(f"Updated model key: {ok}, {reason}")
    if not ok:
        return JSONResponse(_err(reason or "Model key not found."))
    row = get_key_upload(key_id, user["id"])
    bt_out = (row or {}).get("booklet_type", "standard")
    return JSONResponse(
        _ok(
            "Model key updated successfully",
            id=key_id,
            title=title,
            lang=lang,
            booklet_type=bt_out,
        )
    )


@app.delete("/models/key/{key_id}")
async def delete_key(request: Request, key_id: str) -> JSONResponse:
    """Deletes a model key record and removes related stored PDF files."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    deleted, key_pdf_path, booklet_pdf_path = delete_model_key(key_id, user["id"])
    if not deleted:
        return JSONResponse(_err("Model key not found."))
    if key_pdf_path:
        log.info(f"Deleting key PDF: {key_pdf_path}")
        Path(key_pdf_path).unlink(missing_ok=True)
    if booklet_pdf_path:
        log.info(f"Deleting booklet PDF: {booklet_pdf_path}")
        Path(booklet_pdf_path).unlink(missing_ok=True)
    log.info(f"Model key deleted successfully: {key_id}")
    return JSONResponse(_ok("Model key deleted successfully", id=key_id))


@app.post("/models/answer-booklet")
async def post_answer_booklet(
    request: Request,
    id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    """Uploads a booklet PDF, extracts questions, and saves model data."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    id = id.strip()
    if not id:
        return JSONResponse(_err("id is required (non-empty)."))

    key_record = get_key_upload(id, user["id"])
    if not key_record:
        return JSONResponse(_err(f"No key upload found for id {id!r}."))

    bt_raw = _normalize_booklet_type(str(key_record.get("booklet_type") or "standard"))

    dest = UPLOADS_DIR / f"booklet_{id}.pdf"
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        raw = await file.read()
        if not raw:
            return JSONResponse(_err("empty file."))
        dest.write_bytes(raw)
    except Exception as e:
        log.exception("booklet save failed")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))
    log.info("Booklet saved to %s id=%s booklet_type=%s", dest, id, bt_raw)

    try:
        api_key = load_api_key()
    except ValueError as e:
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))

    try:
        if bt_raw == "standard":
            questions = await asyncio.to_thread(process_pdf_path, dest, api_key)
        else:
            result = await asyncio.to_thread(
                run_pdf_questions_and_answers,
                dest,
                api_key,
                key_record["lang"],
                bt_raw,
            )
            if result["kind"] == "no_questions":
                dest.unlink(missing_ok=True)
                return JSONResponse(_err("No question found.", questions=[]))
            questions = custom_qna_rows_to_canonical_questions(result["questions"])
    except Exception as e:
        log.exception("answer booklet processing failed (booklet_type=%s)", bt_raw)
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))

    try:
        upsert_answer_model_from_booklet(
            id,
            key_record["title"],
            key_record["lang"],
            questions,
            str(dest),
            user["id"],
            booklet_type=bt_raw,
        )
    except Exception as e:
        log.exception("db insert failed")
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))

    row = get_answer_model(id, user["id"])
    intro_page = int((row or {}).get("intro_page") or 2)

    return JSONResponse(
        _ok(
            "Answer booklet uploaded successfully",
            id=id,
            questions=questions,
            booklet_type=bt_raw,
            intro_page=intro_page,
        )
    )


@app.get("/models/{model_id}")
async def get_model(request: Request, model_id: str) -> JSONResponse:
    """Returns a single stored model with all question details."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    row = get_answer_model(model_id, user["id"])
    if not row:
        return JSONResponse(_err("Model not found."))
    return JSONResponse(_ok("Model found", **row))


@app.get("/models")
async def list_models(request: Request) -> JSONResponse:
    """Lists all models that belong to the authenticated user."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    items = list_registered_models(user["id"])
    return JSONResponse(_ok("Models listed successfully", items=items))


@app.put("/models/{model_id}/questions/reorder")
async def reorder_model_questions(
    request: Request, model_id: str, payload: ReorderQuestionsPayload
) -> JSONResponse:
    """Reorders model questions according to the provided question ID order."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    ok, reason, arranged = reorder_answer_model_questions(model_id, user["id"], payload.order)
    if not ok:
        return JSONResponse(_err(reason or "Reorder failed."))

    return JSONResponse(
        _ok(
            "Questions reordered successfully",
            id=model_id,
            question_count=len(arranged or []),
            questions=arranged or [],
        )
    )


def _question_payload_log_dict(payload: QuestionPayload) -> dict[str, object]:
    d = payload.model_dump()
    preview_len = 300
    for key in ("desc", "instruction_name"):
        text = str(d.get(key) or "")
        prev = text[:preview_len] + (
            f"... (+{len(text) - preview_len} chars)" if len(text) > preview_len else ""
        )
        d[key] = prev.replace("\r", " ").replace("\n", " ")
    d["diagramDescriptions_count"] = len(d.get("diagramDescriptions") or [])
    if "diagramDescriptions" in d:
        del d["diagramDescriptions"]
    return d


@app.post("/models/{model_id}/create_question")
async def post_model_question(
    request: Request, model_id: str, payload: QuestionPayload
) -> JSONResponse:
    """Creates a new question and appends it to an existing model."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    log.info(
        "POST create_question incoming user_id=%s model_id=%s body=%s",
        user.get("id"),
        model_id,
        json.dumps(_question_payload_log_dict(payload), ensure_ascii=False),
    )

    ok, reason, new_id, q_count = add_answer_model_question(
        model_id, user["id"], payload.model_dump()
    )
    if not ok:
        log.warning(
            "POST create_question failed user_id=%s model_id=%s reason=%s",
            user.get("id"),
            model_id,
            reason,
        )
        return JSONResponse(_err(reason or "Create failed"))

    log.info(
        "POST create_question ok user_id=%s model_id=%s questionId=%s question_count=%s",
        user.get("id"),
        model_id,
        new_id,
        q_count,
    )
    return JSONResponse(
        _ok(
            "Question created successfully",
            id=model_id,
            questionId=new_id,
            question_count=q_count,
        )
    )


@app.put("/models/{model_id}/questions/{question_id}")
async def put_model_question(
    request: Request, model_id: str, question_id: str, payload: QuestionPayload
) -> JSONResponse:
    """Updates an existing question's content and scoring information."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    ok, reason = update_answer_model_question(
        model_id, question_id, {"id": question_id, **payload.model_dump()}, user["id"]
    )
    if not ok:
        if reason in ("Model not found", "Question not found"):
            return JSONResponse(_err(reason))
        return JSONResponse(_err(reason or "Update failed"))

    return JSONResponse(_ok("Question updated successfully", id=model_id, question_id=question_id))


@app.put("/models/questions/bulk-page-marks")
async def put_questions_bulk_page_marks(
    request: Request, payload: BulkQuestionPageMarksPayload
) -> JSONResponse:
    """Bulk updates page numbers and marks for multiple model questions."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    mk = payload.modelKey.strip()
    tuples = [
        (it.questionId.strip(), it.pageNum, it.marks) for it in payload.items
    ]
    log.info(
        "models/questions/bulk-page-marks start user_id=%s modelKey=%s item_count=%d",
        user.get("id"),
        mk,
        len(tuples),
    )
    ok, reason, updated_ids, not_found_ids, intro_page = bulk_patch_answer_model_question_page_marks(
        mk, user["id"], tuples, payload.intro_page
    )
    if not ok:
        log.warning(
            "models/questions/bulk-page-marks failed user_id=%s modelKey=%s reason=%s",
            user.get("id"),
            mk,
            reason,
        )
        return JSONResponse(_err(reason or "Update failed"))

    log.info(
        "models/questions/bulk-page-marks ok user_id=%s modelKey=%s updated=%d not_found=%d",
        user.get("id"),
        mk,
        len(updated_ids),
        len(not_found_ids),
    )
    return JSONResponse(
        _ok(
            "Question page and marks bulk update applied.",
            modelKey=mk,
            updatedQuestionIds=updated_ids,
            notFoundQuestionIds=not_found_ids,
            updatedCount=len(updated_ids),
            notFoundCount=len(not_found_ids),
            intro_page=intro_page,
        )
    )


@app.delete("/models/{model_id}/questions/{question_id}")
async def delete_model_question(
    request: Request, model_id: str, question_id: str
) -> JSONResponse:
    """Deletes one question from the specified model."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    ok, reason, questions = delete_answer_model_question(model_id, question_id, user["id"])
    if not ok:
        return JSONResponse(_err(reason or "Delete failed."))

    return JSONResponse(
        _ok(
            "Question deleted successfully",
            id=model_id,
            question_id=question_id,
            question_count=len(questions or []),
            questions=questions or [],
        )
    )


@app.delete("/models/{model_id}")
async def delete_model(request: Request, model_id: str) -> JSONResponse:
    """Deletes a model and removes its associated stored PDF file."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    deleted, pdf_path = delete_answer_model(model_id, user["id"])
    if not deleted:
        return JSONResponse(_err("Model not found."))
    if pdf_path:
        Path(pdf_path).unlink(missing_ok=True)
    return JSONResponse(_ok("Model deleted successfully"))


def _valid_lang(code: str) -> bool:
    return code.strip().lower() in ("en", "hi")


def _smart_ocr_skipped_pages(page_count: int, items: list) -> list[int]:
    """1-based page numbers with no structured answer content (intro/cover/orphan sheets).

    Uses items produced by the structure pass: any row with a non-empty ``section_name``
    contributes its ``start_page``..``end_page`` span. Rows without ``section_name``
    (e.g. evaluator-only placeholders for unattempted questions) are ignored so bogus
    default coordinates do not hide a skipped intro page."""
    covered: set[int] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        sec = raw.get("section_name")
        if sec is None or (isinstance(sec, str) and not sec.strip()):
            continue
        try:
            sp = int(raw.get("start_page", 1))
            ep = int(raw.get("end_page", sp))
        except (TypeError, ValueError):
            continue
        if ep < sp:
            ep = sp
        sp = max(1, min(sp, page_count))
        ep = max(1, min(ep, page_count))
        for p in range(sp, ep + 1):
            covered.add(p)
    return [p for p in range(1, page_count + 1) if p not in covered]


@app.post("/model/answer-expand")
async def post_model_answer_expand(
    request: Request, payload: ExpandModelAnswerRequest
) -> JSONResponse:
    """Expands a model answer using AI and returns enriched response text."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    lang = payload.language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    bt = payload.type.strip().lower()
    if bt not in ("standard", "custom", "custom_with_model", "essay"):
        log.info(
            "model-answer/expand rejected invalid type=%r user_id=%s",
            payload.type.strip(),
            user.get("id"),
        )
        return JSONResponse(
            _err(
                'type must be "standard", "custom", "custom_with_model", or "essay".'
            )
        )
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    log.info(
        "model-answer/expand start user_id=%s type=%s lang=%s",
        user.get("id"),
        bt,
        lang,
    )
    try:
        result = await asyncio.to_thread(
            expand_model_answer,
            api_key,
            answer_type=bt,
            question=payload.question,
            language=lang,
        )
    except Exception as e:
        log.exception(
            "model-answer/expand failed user_id=%s type=%s",
            user.get("id"),
            bt,
        )
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    diags = result["diagramDescriptions"]
    log.info(
        "model-answer/expand ok user_id=%s type=%s answer_chars=%d diagram_count=%d",
        user.get("id"),
        bt,
        len(result["answer"]),
        len(diags),
    )
    return JSONResponse(
        _ok(
            "Model answer expanded.",
            type=bt,
            answer=result["answer"],
            diagramDescriptions=diags,
        )
    )


@app.post("/analyse/pages")
async def post_analyse_pages(
    request: Request,
    pages: list[UploadFile] = File(...),
    question_title: str = Form(...),
    model_description: str = Form(...),
    total_marks: int = Form(...),
    language: str = Form(...),
) -> JSONResponse:
    """Analyses uploaded pages against model criteria and returns scoring."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    if not pages:
        return JSONResponse(_err("pages field is required"))
    if total_marks < 1:
        return JSONResponse(_err("total_marks must be a positive integer"))
    lang = language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    blobs: list[bytes] = []
    for p in pages:
        raw = await p.read()
        if raw:
            blobs.append(raw)
    if not blobs:
        return JSONResponse(_err("pages field is required"))
    try:
        client = genai.Client(api_key=api_key)
        result = await asyncio.to_thread(
            analyse_pages,
            client,
            blobs,
            question_title.strip(),
            model_description.strip(),
            total_marks,
            lang,
        )
    except Exception as e:
        log.exception("analyse/pages failed")
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    return JSONResponse(_ok("Analysis complete.", **result))


@app.post("/analyse/copy-ocr")
async def post_analyse_copy_ocr(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("en"),
) -> JSONResponse:
    """Extracts OCR text from a PDF copy using direct PDF processing."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    lang = language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    if not _is_pdf(file.filename, file.content_type):
        return JSONResponse(_err("file must be a PDF."))
    raw = await file.read()
    if not raw:
        return JSONResponse(_err("file is empty."))
    max_bytes = copy_ocr_max_bytes()
    if len(raw) > max_bytes:
        return JSONResponse(
            _err(
                f"PDF exceeds maximum size ({max_bytes} bytes). "
                "Reduce the file or raise COPY_OCR_MAX_BYTES."
            )
        )
    if not raw.startswith(b"%PDF"):
        return JSONResponse(_err("file does not look like a valid PDF."))
    rid = str(uuid.uuid4())
    log.info(
        "analyse/copy-ocr start request_id=%s user_id=%s filename=%r bytes=%s max_bytes=%s",
        rid,
        user.get("id"),
        file.filename,
        len(raw),
        max_bytes,
    )
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    tmp: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
        result = await asyncio.to_thread(
            ocr_essay_copy_pdf,
            tmp,
            api_key,
            lang,
            request_id=rid,
        )
    except ValueError as e:
        log.warning("analyse/copy-ocr rejected request_id=%s: %s", rid, e)
        return JSONResponse(_err(str(e)))
    except Exception as e:
        log.exception("analyse/copy-ocr failed request_id=%s", rid)
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                log.warning("analyse/copy-ocr temp unlink failed path=%s", tmp)
    log.info(
        "analyse/copy-ocr ok request_id=%s page_count=%s text_chars=%s",
        rid,
        result.get("page_count"),
        len(str(result.get("text", ""))),
    )
    return JSONResponse(
        _ok(
            "OCR complete.",
            text=result["text"],
            pageCount=result["page_count"],
        )
    )


@app.post("/analyse/copy-ocr-rasterization")
async def post_analyse_copy_ocr_rasterization(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("en"),
) -> JSONResponse:
    """Extracts OCR text from a PDF after rasterizing pages to images."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    lang = language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    if not _is_pdf(file.filename, file.content_type):
        return JSONResponse(_err("file must be a PDF."))
    raw = await file.read()
    if not raw:
        return JSONResponse(_err("file is empty."))
    max_bytes = copy_ocr_max_bytes()
    if len(raw) > max_bytes:
        return JSONResponse(
            _err(
                f"PDF exceeds maximum size ({max_bytes} bytes). "
                "Reduce the file or raise COPY_OCR_MAX_BYTES."
            )
        )
    if not raw.startswith(b"%PDF"):
        return JSONResponse(_err("file does not look like a valid PDF."))
    rid = str(uuid.uuid4())
    log.info(
        "analyse/copy-ocr-rasterization start request_id=%s user_id=%s filename=%r bytes=%s max_bytes=%s",
        rid,
        user.get("id"),
        file.filename,
        len(raw),
        max_bytes,
    )
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    tmp: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
        result = await asyncio.to_thread(
            ocr_essay_copy_pdf_rasterized,
            tmp,
            api_key,
            lang,
            request_id=rid,
        )
    except ValueError as e:
        log.warning("analyse/copy-ocr-rasterization rejected request_id=%s: %s", rid, e)
        return JSONResponse(_err(str(e)))
    except Exception as e:
        log.exception("analyse/copy-ocr-rasterization failed request_id=%s", rid)
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                log.warning(
                    "analyse/copy-ocr-rasterization temp unlink failed path=%s", tmp
                )
    log.info(
        "analyse/copy-ocr-rasterization ok request_id=%s page_count=%s text_chars=%s dpi=%s workers=%s",
        rid,
        result.get("page_count"),
        len(str(result.get("text", ""))),
        result.get("dpi"),
        result.get("workers"),
    )
    return JSONResponse(
        _ok(
            "OCR complete.",
            text=result["text"],
            pageCount=result["page_count"],
            rasterDpi=result["dpi"],
            parallelWorkers=result["workers"],
        )
    )


@app.post("/analyse/smart-ocr")
async def post_analyse_smart_ocr(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("en"),
    model_id: str = Form("", alias="modelId"),
    snap_annotations: bool = Form(True, alias="snapAnnotations"),
    check_level: str = Form("Moderate", alias="checkLevel"),
) -> JSONResponse:
    """Extracts question-wise answers and marking coordinates from a PDF.

    If modelId is provided, also runs Stage 3 grading against the stored answer
    model and merges marks, status, feedback, and annotations into each object
    in ``items`` (same ``question_id``). Intro/cover pages are segregated in the
    structure (phase 2) pass, not via extra API parameters.

    Optional Form field ``checkLevel`` (``Moderate`` | ``Hard``) controls evaluator
    strictness when grading with ``modelId`` — matches Flutter ``checkLevel``.

    Response includes ``skippedPages``: 1-based page indexes not covered by any
    structured question row (typically intro/cover sheets).
    """
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    check_canon = normalize_evaluation_check_level(check_level)
    if not check_canon:
        return JSONResponse(
            _err('checkLevel must be "Moderate" or "Hard" (camelCase form field checkLevel).')
        )

    lang = language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    if not _is_pdf(file.filename, file.content_type):
        return JSONResponse(_err("file must be a PDF."))

    raw = await file.read()
    if not raw:
        return JSONResponse(_err("file is empty."))
    max_bytes = copy_ocr_max_bytes()
    if len(raw) > max_bytes:
        return JSONResponse(
            _err(
                f"PDF exceeds maximum size ({max_bytes} bytes). "
                "Reduce the file or raise COPY_OCR_MAX_BYTES."
            )
        )
    if not raw.startswith(b"%PDF"):
        return JSONResponse(_err("file does not look like a valid PDF."))

    rid = str(uuid.uuid4())
    mid = model_id.strip()

    log.info(
        "analyse/smart-ocr start request_id=%s user_id=%s filename=%r bytes=%s "
        "model_id=%r check_level=%s",
        rid,
        user.get("id"),
        file.filename,
        len(raw),
        mid or "(none)",
        check_canon,
    )

    # --- Validate model early (before burning OCR tokens) ---
    answer_model: dict | None = None
    if mid:
        answer_model = get_answer_model(mid, user["id"])
        if not answer_model:
            return JSONResponse(_err("Model not found."))

    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))

    # --- Stage 1+2: OCR + structure ---
    tmp: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
        result = await asyncio.to_thread(
            smart_ocr_extract_student_answers,
            tmp,
            api_key,
            lang,
            request_id=rid,
        )
    except ValueError as e:
        log.warning("analyse/smart-ocr rejected request_id=%s: %s", rid, e)
        return JSONResponse(_err(str(e)))
    except Exception as e:
        log.exception("analyse/smart-ocr failed request_id=%s", rid)
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                log.warning("analyse/smart-ocr temp unlink failed path=%s", tmp)

    page_count = result.get("page_count")
    items = result.get("items", [])
    if not isinstance(page_count, int) or page_count < 1:
        log.error(
            "analyse/smart-ocr invalid result request_id=%s page_count=%r",
            rid,
            page_count,
        )
        return JSONResponse(_err("AI service error: invalid OCR page count."), status_code=500)
    if not isinstance(items, list):
        log.error(
            "analyse/smart-ocr invalid result request_id=%s items_type=%s",
            rid,
            type(items).__name__,
        )
        return JSONResponse(_err("AI service error: invalid OCR items."), status_code=500)

    log.info(
        "analyse/smart-ocr ocr+structure ok request_id=%s page_count=%s item_count=%s",
        rid,
        page_count,
        len(items),
    )

    # --- Stage 3: grading (only when modelId provided) ---
    if answer_model:
        try:
            questions = answer_model.get("questions") or []
            title = answer_model.get("title") or "General"
            teacher_instructions = format_answer_model_as_teacher_instructions(
                questions, title
            )
            items_to_grade = student_items_for_grading(items)
            ev_list = await asyncio.to_thread(
                evaluate_student_answers_against_model,
                api_key,
                title,
                teacher_instructions,
                items_to_grade,
                request_id=rid,
                check_level=check_canon,
            )
            items = merge_evaluations_into_items(items, ev_list)
            log.info(
                "analyse/smart-ocr eval ok request_id=%s model_id=%s merged_items=%s",
                rid,
                mid,
                len(items),
            )
            if snap_annotations:
                try:
                    page_zones = await asyncio.to_thread(
                        analyze_pdf_free_space,
                        raw,
                    )
                    items = snap_items_annotations(items, page_zones)
                    snapped = sum(
                        1
                        for it in items
                        for ann in (it.get("annotations") or [])
                        if ann.get("_snapped")
                    )
                    log.info(
                        "analyse/smart-ocr snap ok request_id=%s snapped_annotations=%s",
                        rid,
                        snapped,
                    )
                except Exception as snap_err:
                    log.warning(
                        "analyse/smart-ocr snap failed request_id=%s: %s",
                        rid,
                        snap_err,
                    )
        except Exception as e:
            log.exception(
                "analyse/smart-ocr eval failed request_id=%s model_id=%s", rid, mid
            )
            # Don't fail the whole response — return OCR items with error note
            return JSONResponse(
                _ok(
                    "Smart OCR complete. Grading failed.",
                    pageCount=page_count,
                    items=items,
                    modelId=mid,
                    checkLevel=check_canon,
                    gradingError=str(e),
                    skippedPages=_smart_ocr_skipped_pages(page_count, items),
                )
            )

    # --- Build response ---
    extra: dict = {"checkLevel": check_canon}
    if mid:
        extra["modelId"] = mid
    skipped_pages = _smart_ocr_skipped_pages(page_count, items)

    return JSONResponse(
        _ok(
            "Smart OCR complete.",
            pageCount=page_count,
            items=items,
            skippedPages=skipped_pages,
            **extra,
        )
    )


@app.post("/analyse/cached-ocr")
async def post_analyse_cached_ocr(
    request: Request, payload: CachedOcrRequest
) -> JSONResponse:
    """Analyses cached OCR text and returns scoring and feedback details."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    lang = payload.language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    try:
        client = genai.Client(api_key=api_key)
        result = await asyncio.to_thread(
            analyse_cached_ocr,
            client,
            payload.cached_student_text,
            payload.question_title.strip(),
            payload.model_description.strip(),
            payload.total_marks,
            payload.page_count,
            lang,
        )
    except Exception as e:
        log.exception("analyse/cached-ocr failed")
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    return JSONResponse(_ok("Analysis complete.", **result))


@app.post("/analyse/combined-review")
async def post_combined_review(
    request: Request, payload: CombinedReviewRequest
) -> JSONResponse:
    """Combines question-level analysis results into one final review."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    rows = [q.model_dump() for q in payload.question_results]
    try:
        client = genai.Client(api_key=api_key)
        result = await asyncio.to_thread(generate_combined_review, client, rows)
    except Exception as e:
        log.exception("analyse/combined-review failed")
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    return JSONResponse(_ok("Combined review generated.", **result))


@app.post("/analyse/intro-page")
async def post_analyse_intro_page(
    request: Request, page: UploadFile = File(...)
) -> JSONResponse:
    """Parses intro page content to extract student and exam metadata."""
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    raw = await page.read()
    if not raw:
        return JSONResponse(_err("page field is required"))
    try:
        api_key = load_api_key()
    except ValueError as e:
        return JSONResponse(_err(str(e)))
    try:
        client = genai.Client(api_key=api_key)
        result = await asyncio.to_thread(analyse_intro_page, client, raw)
    except ValueError as e:
        if "Could not detect" in str(e) or "no text" in str(e).lower():
            return JSONResponse(_err(str(e)), status_code=422)
        log.exception("analyse/intro-page failed")
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    except Exception as e:
        log.exception("analyse/intro-page failed")
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    return JSONResponse(_ok("Intro page analysed.", **result))


@app.post("/analyse/free-space")
async def post_analyse_free_space(
    request: Request,
    file: UploadFile = File(...),
    rows: int = Form(20),
    cols: int = Form(8),
    min_score: float = Form(0.65),
) -> JSONResponse:
    """Detect annotatable free zones on each page of a student copy PDF.

    Returns per-page lists of rectangular regions (percent coordinates) that are
    empty enough to safely place examiner remarks without overlapping handwriting.

    - rows/cols: grid resolution (higher = finer, slower). Default 20×8.
    - min_score: emptiness threshold 0–1. Default 0.65 suits exam booklets.
    """
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    if not _is_pdf(file.filename, file.content_type):
        return JSONResponse(_err("file must be a PDF."))

    raw = await file.read()
    if not raw:
        return JSONResponse(_err("file is empty."))
    if not raw.startswith(b"%PDF"):
        return JSONResponse(_err("file does not look like a valid PDF."))

    max_bytes = copy_ocr_max_bytes()
    if len(raw) > max_bytes:
        return JSONResponse(
            _err(
                f"PDF exceeds maximum size ({max_bytes} bytes). "
                "Reduce the file or raise COPY_OCR_MAX_BYTES."
            )
        )

    rows = max(4, min(rows, 40))
    cols = max(4, min(cols, 20))
    min_score = max(0.0, min(min_score, 1.0))

    rid = str(uuid.uuid4())
    log.info(
        "analyse/free-space start request_id=%s user_id=%s filename=%r bytes=%d rows=%d cols=%d min_score=%.2f",
        rid,
        user.get("id"),
        file.filename,
        len(raw),
        rows,
        cols,
        min_score,
    )

    try:
        page_zones = await asyncio.to_thread(
            analyze_pdf_free_space,
            raw,
            rows=rows,
            cols=cols,
            min_score=min_score,
        )
    except Exception as e:
        log.exception("analyse/free-space failed request_id=%s", rid)
        return JSONResponse(_err(f"Free-space analysis error: {e}"), status_code=500)

    pages_data = page_zones_to_api_response(page_zones)
    total_zones = sum(len(p["freeZones"]) for p in pages_data)
    log.info(
        "analyse/free-space ok request_id=%s page_count=%d total_zones=%d",
        rid,
        len(pages_data),
        total_zones,
    )
    return JSONResponse(
        _ok(
            "Free space analysis complete.",
            pageCount=len(pages_data),
            gridRows=rows,
            gridCols=cols,
            minScore=min_score,
            pages=pages_data,
        )
    )


@app.post("/analyse/snap-annotations")
async def post_snap_annotations(
    request: Request,
    payload: SnapAnnotationsRequest,
) -> JSONResponse:
    """Snap smart-ocr annotation coordinates onto actual free zones.

    Accepts the ``items`` array from /analyse/smart-ocr and the ``pages`` array
    from /analyse/free-space. Returns the same items with annotation y/x coordinates
    adjusted to land in pixel-verified empty regions, with capacity tracking so no
    two annotations stack on the same zone.

    Use this when you have cached smart-ocr results and want to re-snap against a
    fresh or different free-space analysis without re-running OCR.
    """
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err

    items = [dict(it) for it in payload.items if isinstance(it, dict)]
    pages = [dict(p) for p in payload.pages if isinstance(p, dict)]

    if not items:
        return JSONResponse(_err("items must be a non-empty array."))
    if not pages:
        return JSONResponse(_err("pages must be a non-empty array."))

    try:
        page_zones = api_response_to_page_zones(pages)
        snapped_items = snap_items_annotations(items, page_zones)
    except Exception as e:
        log.exception("analyse/snap-annotations failed")
        return JSONResponse(_err(f"Snap error: {e}"), status_code=500)

    snapped_count = sum(
        1
        for it in snapped_items
        for ann in (it.get("annotations") or [])
        if ann.get("_snapped")
    )
    total_annotations = sum(len(it.get("annotations") or []) for it in snapped_items)
    log.info(
        "analyse/snap-annotations ok item_count=%s snapped=%s/%s",
        len(snapped_items),
        snapped_count,
        total_annotations,
    )
    return JSONResponse(
        _ok(
            "Annotations snapped.",
            items=snapped_items,
            snappedCount=snapped_count,
            totalAnnotations=total_annotations,
        )
    )
