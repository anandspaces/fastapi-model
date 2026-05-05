# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run locally:**
```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Run with Docker:**
```bash
docker-compose up --build   # exposes port 7037
```

**Run pytest suite:**
```bash
python -m pytest tests/ -v
python -m pytest tests/test_free_space.py -v   # single test file
```

**Run integration tests** (requires the API to be running):
```bash
bash scripts/run_all.sh
# or individual scripts:
python scripts/01_auth.py
python scripts/02_models_happy_path.py
```

## Environment

Copy `.env.example` to `.env` and fill in:
- `GEMINI_API_KEY` — Google Gemini API key (required)
- `JWT_SECRET` — secret for signing JWTs
- `OCR_*` / parallel worker limits — tunable for performance

## Architecture

### Entry Point

`main.py` (~1,460 lines) contains the entire FastAPI app: app init, lifespan (`init_db()`), CORS, JWT middleware, and all 25 route handlers. There is no router split — all endpoints live directly in `main.py`.

### `src/` — Business Logic

| Module | Role |
|--------|------|
| `database.py` | SQLite connection, schema init (`init_db`) |
| `schemas.py` | Pydantic request/response models |
| `service.py` | CRUD helpers for `key_uploads`, `answer_models`, `users` (always scoped to `owner_user_id`) |
| `gemini_extract.py` | PDF → structured questions JSON via Gemini |
| `gemini_expand_model_answer.py` | Expand short model answers to detailed ones |
| `gemini_question_answers.py` | Process Q&A pairs from PDFs |
| `gemini_smart_ocr.py` | Extract student answers from handwritten booklet PDFs |
| `gemini_copy_ocr.py` | Essay copy extraction — whole-PDF and rasterized parallel |
| `gemini_analyse.py` | AI grading, copy-checking, combined reviews, intro-page marks |
| `gemini_evaluate_student_answers.py` | Evaluate student answers against model answers |
| `free_space_utils.py` | Detect writable zones in PDFs, grid scoring, annotation snapping |
| `free_space_service.py` | Orchestrate free-space analysis, map to API response |
| `pdf_questions_import.py` | Parse custom booklet PDFs into canonical Q&A format |
| `pdf_qa_pipeline.py` | High-level pipeline for PDF Q&A processing |
| `custom_booklet_storage.py` | Convert custom booklet rows to canonical questions |

### Data Layer

SQLite file at `data/db.sqlite` (created at runtime). Three tables:
- `users` — accounts with bcrypt-hashed passwords
- `key_uploads` — model answer key metadata + PDF path
- `answer_models` — extracted/generated model answers as JSON, linked to `key_uploads`

All service queries filter by `owner_user_id` for owner isolation.

Uploaded PDFs are stored under `data/uploads/`.

### Auth

JWT Bearer tokens issued at `POST /auth/login`. Protected endpoints verify the token via a `get_current_user` dependency. The Swagger UI is configured with JWT authorization support.

### API Response Envelope

All endpoints return:
```json
{ "status": 1, "data": {...} }   // success
{ "status": 0, "message": "..." } // error
{ "status": -1, "message": "..." } // auth/permission error
```

### Key AI Flows

1. **Model key upload** → `gemini_extract.py` extracts questions → stored in `answer_models`
2. **Student booklet grading** → `gemini_smart_ocr.py` extracts answers → `gemini_evaluate_student_answers.py` grades against model
3. **Essay copy-check** → `gemini_copy_ocr.py` (whole-PDF or rasterized parallel) → `gemini_analyse.py` copy review
4. **Free-space annotation** → `free_space_utils.py` detects zones → `free_space_service.py` returns snappable regions

All Gemini calls use `google-genai` SDK targeting Gemini models configured in each module.
