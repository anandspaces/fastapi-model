# PDF models API

All API code lives under `pdf_json/api/`. The CLI script is [`read_pdf_to_json.py`](../read_pdf_to_json.py).

## Environment

Copy [`.env.example`](.env.example) to `.env` and set:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API (PDF → questions) |
| `API_TOKEN` | Bearer token for protected routes |

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
| `model_store.py` | SQLite (`key_uploads`, `answer_models`) |
| `gemini_extract.py` | Gemini PDF → questions JSON |

Data: `pdf_json/api/data/` (DB + uploads).

## Endpoints

- `POST /models/key` — multipart: `title`, `lang`, `file` (PDF)
- `POST /models/answer-booklet` — multipart: `title`, `lang`, `file` (PDF)
- `GET /models` — every key-registered id: `{ id, title, lang, has_booklet }[]` (newest first; `has_booklet` false until answer booklet is posted)
- `GET /models/{model_id}`
- `PUT /models/{model_id}/questions/{question_id}` — replace one question object (match by `question.id`)
- `DELETE /models/{model_id}`

Responses: `{ "data": { "status": 1 \| 0, ... } }`.
