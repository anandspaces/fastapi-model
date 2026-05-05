# PDF models API

All API code lives under `pdf_json/api/`. The CLI script is [`read_pdf_to_json.py`](../read_pdf_to_json.py).

## Environment

Copy [`.env.example`](.env.example) to `.env` and set:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API (PDF → questions) |
| `JWT_SECRET` | Secret used to sign JWT access tokens |
| `JWT_EXPIRES_MINUTES` | JWT lifetime in minutes (default `60`) |
| `COPY_OCR_MAX_BYTES` | Max PDF upload for copy-OCR routes (default ~25 MiB) |
| `COPY_OCR_MAX_PAGES` | Max PDF pages for copy-OCR routes (default `30`) |
| `COPY_OCR_RASTER_DPI` | Render DPI for `POST /analyse/copy-ocr-rasterization` (default `150`) |
| `COPY_OCR_PARALLEL_WORKERS` | Max concurrent Gemini calls for rasterization OCR (default `5`) |

## Docker

Use a filled-in `.env` (see **Environment**), or `cp .env.example .env` and edit.

```bash
docker compose up --build
```

API: `http://localhost:8000` — SQLite and uploads persist in `./data` (mounted into the container).

## Run

From **repository root** (with `GEMINI_API_KEY` set). Module path uses **dots** (`.`),
not slashes — `pdf_json.api.main`, not `pdf_json/api/main`.

```bash
uv run uvicorn pdf_json.api.main:app --reload --host 0.0.0.0 --port 8000
```

From `pdf_json/api/`:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

(`pdf_json.api.app:app` is an alias of the same app.)

## Layout

| File | Role |
|------|------|
| `main.py` | FastAPI app & routes (entrypoint) |
| `database.py` | DB path constants, SQLite connection helper, and schema init |
| `service.py` | All helper methods for key/booklet CRUD |
| `gemini_extract.py` | Gemini PDF → questions JSON |
| `gemini_analyse.py` | AI copy-checker: grading, combined review, intro marks table |
| `gemini_copy_ocr.py` | Essay copy OCR: whole-PDF (`/analyse/copy-ocr`) and raster + parallel (`/analyse/copy-ocr-rasterization`) |

Data: `data/` at repo root (DB + uploads).

## Endpoints

- `POST /auth/signup` — JSON: `username`, `password`
- `POST /auth/login` — JSON: `username`, `password` -> token data (`accessToken`, `tokenType`, `expiresIn`)
- `POST /models/key` — multipart: `title`, `lang`, optional `type` (`standard` \| `custom` \| `custom_with_model` \| `essay`; drives `POST /models/answer-booklet` processing for that id; `custom_with_model` matches `custom`; `essay` uses the same PDF path as `custom` with longer generated answers)
- `PUT /models/key/{key_id}` — update model key metadata (`title`, `lang`, optional `booklet_type`: `standard` \| `custom` \| `custom_with_model` \| `essay`)
- `DELETE /models/key/{key_id}` — delete model key (and linked answer model if present)
- `POST /models/answer-booklet` — multipart: `id`, `file` (PDF); booklet mode comes from the key’s `type` set at `POST /models/key`
- `GET /models` — every key-registered id: `{ id, title, lang, has_booklet, booklet_type }[]` (`booklet_type` null until booklet exists)
- `GET /models/{model_id}` — full model payload includes `booklet_type`
- `POST /models/{model_id}/create_question` — JSON body same as PUT question (no `id`); `diagramDescriptions` optional (default `[]`); works after `POST /models/key` (bootstraps answer model if needed); `POST /models/answer-booklet` later replaces questions from the PDF
- `PUT /models/{model_id}/questions/{question_id}` — replace one question object (match by `question.id`)
- `DELETE /models/{model_id}/questions/{question_id}` — delete one question by `question.id` and renumber remaining `questionNo`
- `PUT /models/{model_id}/questions/reorder` — reorder questions by id and renumber `questionNo` as `Q1..Qn`
- `DELETE /models/{model_id}`

### AI Copy Checker (`/analyse/*`)

All require `Authorization: Bearer <accessToken>`. Success uses `{ "status": 1, "message": "...", "data": { ... } }` (same as other routes).

- `POST /analyse/full` — **`multipart/form-data` only** (not JSON): `modelId`, `questionId`, repeated `pageImages` file parts; optional `checkLevel`. Scheme/marks/language/instructions from DB. See **`AI_ANALYSE_API_INTEGRATION.md`**.
- `POST /analyse/cached-ocr` — **JSON**: `modelId`, `questionId`, `cachedStudentText`; optional `checkLevel`. **`data`** snake_case.
- `POST /analyse/combined-review` — **JSON**: `modelId` + compact `questionResults` (`questionId`, `marksAwarded`, `goodPoints`, `improvements`, `finalReview`); server fills titles/marks from DB.
- `POST /analyse/intro-page` — JSON: `pageImageBase64`; **`data.cells`** entries use `question_no`, `marks_text`, `x_percent`, `y_percent`.
- `POST /analyse/copy-ocr` — multipart: `file` (PDF), optional `language`, optional `modelId` (default language from model when language omitted)
- `POST /analyse/copy-ocr-rasterization` — same inputs; rasterizes each page (PyMuPDF) and OCRs pages in parallel (up to `COPY_OCR_PARALLEL_WORKERS`, default 5); `data` adds `rasterDpi`, `parallelWorkers`

Responses: `{ "status": 1 \| 0 \| -1, "message": ..., "data": { ... } }` (errors from validation use `status: 0`; auth failures `status: -1`).

All `/models*`, `/models/key*`, and `/analyse*` endpoints require:

`Authorization: Bearer <accessToken>`

## Frontend Reorder Flow

Suggested UI:

- Render `questions` from `GET /models/{model_id}` in a drag-and-drop list.
- Keep `originalQuestions` and `workingQuestions` in state.
- On drag, reorder `workingQuestions` only.
- Show `Unsaved changes` + `Save order` + `Reset`.

Save request:

```json
{
  "order": ["q-eng-003", "q-eng-001", "q-eng-002"]
}
```

Send to:

- `PUT /models/{model_id}/questions/reorder`

On success:

- Replace local list with `data.questions` from API response.
- `questionNo` is already normalized (`Q1..Qn`) by backend.

Validation errors returned by API:

- `Model not found.`
- `Invalid order: duplicate ids.`
- `Invalid order: missing question ids.`
- `Invalid order: unknown question ids.`
- `Invalid questions_json.`
