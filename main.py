import asyncio
import logging
import re
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gemini_extract import load_api_key, process_pdf_path
from model_store import (
    UPLOADS_DIR,
    delete_answer_model,
    get_answer_model,
    get_key_upload,
    init_db,
    insert_answer_model,
    insert_key_upload,
    list_answer_models_page,
)
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


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


def _ok(**fields: object) -> dict:
    return {"data": {"status": 1, **fields}}


def _err(message: str, **extra: object) -> dict:
    return {"data": {"status": 0, "message": message, **extra}}


def _is_pdf(filename: str | None, content_type: str | None) -> bool:
    if filename and re.search(r"\.pdf$", filename, re.I):
        return True
    if content_type and "pdf" in content_type.lower():
        return True
    return False


@app.post("/models/key")
async def post_model_key(
    request: Request,
    title: str = Form(...),
    lang: str = Form(...),
) -> JSONResponse:
    log.info(f"Posting model key: {title}, {lang}, {request.headers.get('Authorization')}")
    req_token = request.headers.get("Authorization")
    if not req_token:
        return JSONResponse(_err("Authorization header is required."))
    if req_token != f"Bearer {os.getenv('API_TOKEN')}":
        return JSONResponse(_err("Invalid token."))
    title, lang = title.strip(), lang.strip()
    if not title or not lang:
        return JSONResponse(_err("title and lang are required (non-empty)."))
    key_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"key_{key_id}.pdf"
    try:
        insert_key_upload(key_id, title, lang, str(dest))
    except Exception as e:
        log.exception("key upload failed")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))
    return JSONResponse(_ok(id=key_id, title=title, lang=lang))


@app.post("/models/answer-booklet")
async def post_answer_booklet(
    request: Request,
    id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    req_token = request.headers.get("Authorization")
    if not req_token:
        return JSONResponse(_err("Authorization header is required."))
    if req_token != f"Bearer {os.getenv('API_TOKEN')}":
        return JSONResponse(_err("Invalid token."))
    id = id.strip()
    if not id:
        return JSONResponse(_err("id is required (non-empty)."))
    dest = UPLOADS_DIR / f"booklet_{id}.pdf"
    log.info(f"Saving booklet to {dest} with id {id}")
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
    log.info(f"File read completed from {dest}")
    try:
        api_key = load_api_key()
    except ValueError as e:
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))
    log.info(f"Extracting questions from {dest} with API key {api_key}")
    try:
        questions = await asyncio.to_thread(process_pdf_path, dest, api_key)
    except Exception as e:
        log.exception("Gemini extraction failed")
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))

    key_record = get_key_upload(id)
    if not key_record:
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(f"No key upload found for id {id!r}."))

    try:
        insert_answer_model(
            id,
            key_record["title"],
            key_record["lang"],
            questions,
            str(dest),
        )
    except Exception as e:
        log.exception("db insert failed")
        dest.unlink(missing_ok=True)
        return JSONResponse(_err(str(e)))

    return JSONResponse(
        _ok(id=id, questions=questions)
    )


@app.get("/models/{model_id}")
async def get_model(request: Request, model_id: str) -> JSONResponse:
    req_token = request.headers.get("Authorization")
    if not req_token:
        return JSONResponse(_err("Authorization header is required."))
    if req_token != f"Bearer {os.getenv('API_TOKEN')}":
        return JSONResponse(_err("Invalid token."))
    row = get_answer_model(model_id)
    if not row:
        return JSONResponse(_err("Model not found."))
    return JSONResponse(_ok(**row))


@app.get("/models")
async def list_models(request: Request, page: int = 1, page_size: int = 20) -> JSONResponse:
    req_token = request.headers.get("Authorization")
    if not req_token:
        return JSONResponse(_err("Authorization header is required."))
    if req_token != f"Bearer {os.getenv('API_TOKEN')}":
        return JSONResponse(_err("Invalid token."))
    items, total = list_answer_models_page(page, page_size)
    return JSONResponse(
        _ok(items=items, page=page, page_size=page_size, total=total)
    )


@app.delete("/models/{model_id}")
async def delete_model(request: Request, model_id: str) -> JSONResponse:
    req_token = request.headers.get("Authorization")
    if not req_token:
        return JSONResponse(_err("Authorization header is required."))
    if req_token != f"Bearer {os.getenv('API_TOKEN')}":
        return JSONResponse(_err("Invalid token."))
    deleted, pdf_path = delete_answer_model(model_id)
    if not deleted:
        return JSONResponse(_err("Model not found."))
    if pdf_path:
        Path(pdf_path).unlink(missing_ok=True)
    return JSONResponse(_ok())
