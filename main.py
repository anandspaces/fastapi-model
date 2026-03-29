import asyncio
from datetime import datetime, timedelta, timezone
import logging
import re
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
from src.gemini_expand_model_answer import expand_model_answer
from src.gemini_extract import load_api_key, process_pdf_path
from src.pdf_qa_pipeline import run_pdf_questions_and_answers
from src.database import UPLOADS_DIR, init_db
from src.service import (
    create_user,
    delete_answer_model_question,
    delete_model_key,
    delete_answer_model,
    get_answer_model,
    get_key_upload,
    get_user_by_username,
    insert_answer_model,
    insert_key_upload,
    list_registered_models,
    update_key_upload,
    update_answer_model_question,
    reorder_answer_model_questions,
)
from dotenv import load_dotenv
from src.schemas import (
    AuthRequest,
    CachedOcrRequest,
    CombinedReviewRequest,
    ExpandModelAnswerRequest,
    QuestionPayload,
    ReorderQuestionsPayload,
    TokenData,
)


class ModelKeyPayload(BaseModel):
    title: str
    lang: str
    booklet_type: str | None = None


load_dotenv()

log = logging.getLogger(__name__)
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_ALGORITHM = "HS256"
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "60"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    init_db()
    yield


app = FastAPI(
    title="PDF model keys & answer booklets",
    version="0.1.0",
    lifespan=lifespan,
)

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
    return JSONResponse("API is running successfully!")


@app.post("/auth/signup")
async def auth_signup(payload: AuthRequest) -> JSONResponse:
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
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    title, lang = title.strip(), lang.strip()
    if not title or not lang:
        return JSONResponse(_err("title and lang are required (non-empty)."))
    bt_raw = (booklet_type_param or "standard").strip().lower()
    if bt_raw not in ("standard", "custom"):
        return JSONResponse(_err('type must be "standard" or "custom".'))
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
            if vv not in ("standard", "custom"):
                return JSONResponse(
                    _err('booklet_type must be "standard" or "custom".')
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
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    id = id.strip()
    if not id:
        return JSONResponse(_err("id is required (non-empty)."))

    key_record = get_key_upload(id, user["id"])
    if not key_record:
        return JSONResponse(_err(f"No key upload found for id {id!r}."))

    bt_raw = str(key_record.get("booklet_type") or "standard").strip().lower()
    if bt_raw not in ("standard", "custom"):
        bt_raw = "standard"

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
        insert_answer_model(
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

    return JSONResponse(
        _ok(
            "Answer booklet uploaded successfully",
            id=id,
            questions=questions,
            booklet_type=bt_raw,
        )
    )


@app.get("/models/{model_id}")
async def get_model(request: Request, model_id: str) -> JSONResponse:
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    row = get_answer_model(model_id, user["id"])
    if not row:
        return JSONResponse(_err("Model not found."))
    return JSONResponse(_ok("Model found", **row))


@app.get("/models")
async def list_models(request: Request) -> JSONResponse:
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    items = list_registered_models(user["id"])
    return JSONResponse(_ok("Models listed successfully", items=items))


@app.put("/models/{model_id}/questions/reorder")
async def reorder_model_questions(
    request: Request, model_id: str, payload: ReorderQuestionsPayload
) -> JSONResponse:
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


@app.put("/models/{model_id}/questions/{question_id}")
async def put_model_question(
    request: Request, model_id: str, question_id: str, payload: QuestionPayload
) -> JSONResponse:
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


@app.delete("/models/{model_id}/questions/{question_id}")
async def delete_model_question(
    request: Request, model_id: str, question_id: str
) -> JSONResponse:
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


@app.post("/model/answer-expand")
async def post_model_answer_expand(
    request: Request, payload: ExpandModelAnswerRequest
) -> JSONResponse:
    user, auth_err = _require_auth_user(request)
    if auth_err:
        return auth_err
    lang = payload.language.strip().lower()
    if not _valid_lang(lang):
        return JSONResponse(_err("language must be en or hi"))
    bt = payload.type.strip().lower()
    if bt not in ("standard", "custom", "essay"):
        log.info(
            "model-answer/expand rejected invalid type=%r user_id=%s",
            payload.type.strip(),
            user.get("id"),
        )
        return JSONResponse(
            _err('type must be "standard", "custom", or "essay".')
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
            draft_answer=payload.answer,
            diagram_description=payload.diagram_description,
            language=lang,
        )
    except Exception as e:
        log.exception(
            "model-answer/expand failed user_id=%s type=%s",
            user.get("id"),
            bt,
        )
        return JSONResponse(_err(f"AI service error: {e}"), status_code=500)
    log.info(
        "model-answer/expand ok user_id=%s type=%s answer_chars=%d diagram_chars=%d",
        user.get("id"),
        bt,
        len(result["answer"]),
        len(result["diagram_description"]),
    )
    return JSONResponse(
        _ok(
            "Model answer expanded.",
            type=bt,
            answer=result["answer"],
            diagram_description=result["diagram_description"],
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


@app.post("/analyse/cached-ocr")
async def post_analyse_cached_ocr(
    request: Request, payload: CachedOcrRequest
) -> JSONResponse:
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
