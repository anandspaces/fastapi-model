# PDF models API

All API code lives under `pdf_json/api/`. The CLI script is [`read_pdf_to_json.py`](../read_pdf_to_json.py).

## Environment

Copy [`.env.example`](.env.example) to `.env` and set:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API (PDF → questions) |
| `JWT_SECRET` | Secret used to sign JWT access tokens |
| `JWT_EXPIRES_MINUTES` | JWT lifetime in minutes (default `60`) |

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

Data: `pdf_json/api/data/` (DB + uploads).

## Endpoints

- `POST /auth/signup` — JSON: `username`, `password`
- `POST /auth/login` — JSON: `username`, `password` -> token data (`accessToken`, `tokenType`, `expiresIn`)
- `POST /models/key` — multipart: `title`, `lang`, `file` (PDF)
- `PUT /models/key/{key_id}` — update model key metadata (`title`, `lang`)
- `DELETE /models/key/{key_id}` — delete model key (and linked answer model if present)
- `POST /models/answer-booklet` — multipart: `title`, `lang`, `file` (PDF)
- `GET /models` — every key-registered id: `{ id, title, lang, has_booklet }[]` (newest first; `has_booklet` false until answer booklet is posted)
- `GET /models/{model_id}`
- `PUT /models/{model_id}/questions/{question_id}` — replace one question object (match by `question.id`)
- `PUT /models/{model_id}/questions/reorder` — reorder questions by id and renumber `questionNo` as `Q1..Qn`
- `DELETE /models/{model_id}`

Responses: `{ "data": { "status": 1 \| 0, ... } }`.

All `/models*` and `/models/key*` endpoints require:

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
