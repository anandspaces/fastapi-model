# API Endpoints + Integration Guide

This guide documents the current backend APIs section-wise, including request/response examples and client integration flow.

## 1) Overview

### Base URLs

- Local: `http://127.0.0.1:8000`
- Deployed (example): `https://checkermodel.service.dextora.org`

### Common Response Envelope

All business responses use this shape:

```json
{
  "status": 1,
  "message": "Human-readable message",
  "data": {}
}
```

- `status: 1` -> success
- `status: 0` -> business/validation/AI handling error
- `status: -1` -> authentication/authorization error (usually HTTP `401`)

### Authentication Header

Protected endpoints require:

```http
Authorization: Bearer <accessToken>
```

---

## 2) Authentication APIs

### POST `/auth/signup`

- Content-Type: `application/json`
- Auth required: No

#### Request

```json
{
  "username": "test123",
  "password": "Pass@123"
}
```

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Signup successful.",
  "data": {
    "accessToken": "<jwt>",
    "tokenType": "Bearer",
    "expiresIn": 3600,
    "username": "test123"
  }
}
```

#### Error Response (`200`)

```json
{
  "status": 0,
  "message": "Username already exists",
  "data": {}
}
```

---

### POST `/auth/login`

- Content-Type: `application/json`
- Auth required: No

#### Request

```json
{
  "username": "test123",
  "password": "Pass@123"
}
```

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Login successful.",
  "data": {
    "accessToken": "<jwt>",
    "tokenType": "Bearer",
    "expiresIn": 3600,
    "username": "test123"
  }
}
```

#### Error Response (`200`)

```json
{
  "status": 0,
  "message": "Invalid username or password.",
  "data": {}
}
```

---

## 3) Model Key + Booklet APIs

### POST `/models/key`

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Request form fields

- `title` (string, required)
- `lang` (string, required)
- `type` (string, optional; default `standard`) — `standard`, `custom`, `custom_with_model`, or `essay`. Stored on the key and used when you later call `POST /models/answer-booklet` for this `id` (no `type` field on answer-booklet). `custom_with_model` uses the same PDF pipeline and answer generation as `custom` (distinct label for clients). `essay` uses the same PDF pipeline as `custom` but generates longer model answers per question.

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Key uploaded successfully",
  "data": {
    "id": "uuid",
    "title": "Physics Unit Test",
    "lang": "en",
    "booklet_type": "standard"
  }
}
```

---

### PUT `/models/key/{key_id}`

- Content-Type: `application/json`
- Auth required: Yes

#### Request

```json
{
  "title": "Updated title",
  "lang": "en",
  "booklet_type": "custom"
}
```

`booklet_type` is optional. Allowed values: `standard`, `custom`, `custom_with_model`, or `essay`. If omitted, the existing value on the key is kept. If a booklet row already exists for this id, `answer_models.booklet_type` is updated to match.

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Model key updated successfully",
  "data": {
    "id": "uuid",
    "title": "Updated title",
    "lang": "en",
    "booklet_type": "custom"
  }
}
```

---

### DELETE `/models/key/{key_id}`

- Content-Type: none
- Auth required: Yes

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Model key deleted successfully",
  "data": {
    "id": "uuid"
  }
}
```

---

### POST `/models/answer-booklet`

Uploads a PDF for an existing model key. Processing mode (**standard** vs **custom** / **custom_with_model** / **essay**) is taken from the key’s stored `type` / `booklet_type` (set at `POST /models/key` or `PUT /models/key/{key_id}`), not from this request.

If an `answer_models` row already exists for this id (for example you used **`POST /models/{model_id}/create_question`** first), this endpoint **updates** it: **`questions` in the database are replaced** by whatever the PDF pipeline extracts, and the booklet file path is set.

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Request form fields

- `id` (string, required; key id from `POST /models/key`)
- `file` (PDF file, required)

#### Booklet modes (from the key)

- **`standard`** — Full booklet extraction via Gemini ([`process_pdf_path`](src/gemini_extract.py)): marks, page hints, diagrams metadata, etc.
- **`custom`** — Import **questions** from the PDF, then **one Gemini call per question** for a concise model answer. Answer language follows the key’s `lang`.
- **`custom_with_model`** — Same import and per-question flow as **`custom`** (stored `booklet_type` differs for client routing only).
- **`essay`** — Same import and per-question flow as **`custom`**, but each generated answer targets an extended response (roughly essay length; see [`src/gemini_question_answers.py`](src/gemini_question_answers.py)).

#### Success Response (`200`)

Same envelope and `data` keys for all booklet modes:

```json
{
  "status": 1,
  "message": "Answer booklet uploaded successfully",
  "data": {
    "id": "uuid",
    "booklet_type": "standard",
    "questions": [
      {
        "id": "q-eng-001",
        "questionNo": "Q1",
        "title": "Question title",
        "desc": "Answer or model-answer text",
        "pageNum": 2,
        "marks": 8,
        "diagramDescriptions": []
      }
    ]
  }
}
```

When the key is `custom`, `custom_with_model`, or `essay`, `data.booklet_type` is that value. Each question uses the same keys; typically `pageNum` is `1`, `marks` is `0`, and `desc` holds the generated model answer.

**Question order (`custom` / `custom_with_model` / `essay`):** Questions may appear on the PDF in any layout order (e.g. Q1, then Q5, then Q7). The server asks Gemini to return them in **ascending logical question order** and then **re-sorts** by printed `questionNo` as a safety net. Internal `id` values are `q-001`, `q-002`, … in that sorted order; `questionNo` stays as printed (e.g. `"Q5"` is not renamed).

#### No questions found (key `custom`, `custom_with_model`, or `essay`, `200` business failure)

When the PDF has no extractable exam-style questions, nothing is persisted and the uploaded file is removed:

```json
{
  "status": 0,
  "message": "No question found.",
  "data": {
    "questions": []
  }
}
```

---

### GET `/models`

- Content-Type: none
- Auth required: Yes

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Models listed successfully",
  "data": {
    "items": [
      {
        "id": "uuid",
        "title": "Model title",
        "lang": "en",
        "has_booklet": true,
        "booklet_type": "standard"
      }
    ]
  }
}
```

When `has_booklet` is false, `booklet_type` is `null`. Otherwise it is `"standard"`, `"custom"`, `"custom_with_model"`, or `"essay"`.

---

### GET `/models/{model_id}`

- Content-Type: none
- Auth required: Yes

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Model found",
  "data": {
    "id": "uuid",
    "title": "Model title",
    "lang": "en",
    "booklet_type": "standard",
    "question_count": 20,
    "questions": [],
    "created_at": "2026-03-23T00:00:00Z"
  }
}
```

Other APIs can branch on `booklet_type` when different behavior is needed for standard vs custom vs custom_with_model vs essay models.

---

### DELETE `/models/{model_id}`

- Content-Type: none
- Auth required: Yes

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Model deleted successfully",
  "data": {}
}
```

---

## 4) Question Management APIs

### POST `/models/{model_id}/create_question`

- Content-Type: `application/json`
- Auth required: Yes

Creates a new question for a model **key** you already registered (`POST /models/key`). If no answer booklet row exists yet, the server **creates** `answer_models` with your question and a null booklet PDF; you can upload a PDF later with **`POST /models/answer-booklet`** (which will replace questions with the extracted set from the file). If the `model_id` is unknown or not owned by you, you get an error.

#### Request

(no `id` in the body — the server assigns the next `q-eng-NNN` id):

```json
{
  "questionNo": "Q1",
  "title": "New question title",
  "desc": "Full question / model answer text",
  "instruction_name": "Optional examiner / scheme notes (value-add, rubric) separate from desc",
  "pageNum": 1,
  "marks": 5,
  "diagramDescriptions": []
}
```

- **`instruction_name`** is optional; omit it or send `""` when there are no extra marking notes (stored in DB inside `questions_json` with each question).
- **`diagramDescriptions`** is optional; omit it or send `[]` when there are no diagrams.
- **`questionNo` in the body is overwritten** after create: every question’s `questionNo` is renumbered to `Q1`, `Q2`, … in array order (same behaviour as after deleting a question).
- New **`id`**: next available `q-eng-001`, `q-eng-002`, … by scanning existing question ids matching `q-eng-` + three digits.

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Question created successfully",
  "data": {
    "id": "model_id",
    "questionId": "q-eng-003",
    "question_count": 3
  }
}
```

#### Error Response (`200`, `status: 0`)

- No **`POST /models/key`** row for this id / user: `Model key not found for this user.`
- Corrupt `questions_json`: `Invalid questions_json`.

---

### PUT `/models/{model_id}/questions/{question_id}`

- Content-Type: `application/json`
- Auth required: Yes

#### Request

```json
{
  "questionNo": "Q1",
  "title": "Updated title",
  "desc": "Updated description",
  "instruction_name": "",
  "pageNum": 2,
  "marks": 8,
  "diagramDescriptions": []
}
```

- **`instruction_name`** follows the same rules as on create (optional; defaults to `""`).

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Question updated successfully",
  "data": {
    "id": "model_id",
    "question_id": "q-eng-001"
  }
}
```

---

### PUT `/models/questions/bulk-page-marks`

- Content-Type: `application/json`
- Auth required: Yes

Updates `pageNum` and `marks` for **many** questions on one answer model in a single request. Unknown `questionId` values are **skipped** (listed in the response); the request still succeeds. If the same `questionId` appears more than once in `items`, **the last entry wins**.

JSON field names use **camelCase**, consistent with question objects elsewhere (`questionNo`, `pageNum`, `marks`).

#### Request

```json
{
  "modelKey": "uuid-same-as-model-id",
  "items": [
    { "questionId": "q-eng-001", "pageNum": 2, "marks": 8 },
    { "questionId": "q-eng-002", "pageNum": 3, "marks": 4 }
  ]
}
```

- `items` may be an empty array (no database write; model must exist).

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Question page and marks bulk update applied.",
  "data": {
    "modelKey": "uuid-same-as-model-id",
    "updatedQuestionIds": ["q-eng-001", "q-eng-002"],
    "notFoundQuestionIds": ["q-eng-999"],
    "updatedCount": 2,
    "notFoundCount": 1
  }
}
```

#### Error Response (`200`, `status: 0`)

- Model missing or not owned by the user: `message` is `Model not found`.
- Corrupt `questions_json`: `message` is `Invalid questions_json`.

---

### PUT `/models/{model_id}/questions/reorder`

- Content-Type: `application/json`
- Auth required: Yes

#### Request

```json
{
  "order": ["q-eng-002", "q-eng-001", "q-eng-003"]
}
```

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Questions reordered successfully",
  "data": {
    "id": "model_id",
    "question_count": 3,
    "questions": []
  }
}
```

---

### DELETE `/models/{model_id}/questions/{question_id}`

- Content-Type: none
- Auth required: Yes

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Question deleted successfully",
  "data": {
    "id": "model_id",
    "question_id": "q-eng-003",
    "question_count": 2,
    "questions": []
  }
}
```

---

## 5) Model answer expansion (stateless)

Generates a full examiner-style model answer from the **question text only** using Gemini (no user draft). **Nothing is stored** in the database. This endpoint’s `type` field is **only** for answer length (`standard` / `custom` / `custom_with_model` ≈ 300 words, `essay` ≈ 1200 words). Booklet keys use the same values on `POST /models/key` / `PUT /models/key/{key_id}` to drive **stored** models and `POST /models/answer-booklet` processing; `POST /model/answer-expand` does not read the database and only uses `type` for length.

Response `diagramDescriptions` uses the same camelCase array shape as question objects elsewhere (`diagramDescriptions` on each question).

**Breaking change:** Earlier versions required `answer` and optional `diagram_description` in the request and returned a single `diagram_description` string. Clients must send only `type` and `question` (plus optional `language`) and must read `diagramDescriptions` as an array.

Implementation reference: [`src/gemini_expand_model_answer.py`](src/gemini_expand_model_answer.py).

### POST `/model/answer-expand`

- Content-Type: `application/json`
- Auth required: Yes

#### Request (JSON body)

| Field | Type | Required | Notes |
|--------|------|----------|--------|
| `type` | string | Yes | `standard`, `custom`, `custom_with_model`, or `essay` (case-insensitive after trim). `standard`, `custom`, and `custom_with_model` use the same target length (~300 words). `essay` uses ~1200 words. |
| `question` | string | Yes | Exam question text (non-empty). |
| `language` | string | No | Default `en`. Must be `en` or `hi` when provided. |

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Model answer expanded.",
  "data": {
    "type": "essay",
    "answer": "Full model answer text...",
    "diagramDescriptions": [
      "Labeled sketch of the carbon cycle with reservoirs and arrows.",
      "Graph of temperature vs time with axis labels."
    ]
  }
}
```

If no diagram is appropriate, `diagramDescriptions` is `[]`.

#### Validation / business errors (`200`, `status: 0`)

- Invalid `type`:

```json
{
  "status": 0,
  "message": "type must be \"standard\", \"custom\", \"custom_with_model\", or \"essay\".",
  "data": {}
}
```

- Invalid `language`:

```json
{
  "status": 0,
  "message": "language must be en or hi",
  "data": {}
}
```

- Missing `GEMINI_API_KEY` (or similar config): `status: 0` with an explanatory `message`.

Pydantic may return **`422`** if required JSON fields are missing or `question` is empty (below `min_length`).

#### AI failures (`500`)

Same pattern as other Gemini routes: `status: 0`, `message` prefixed with `AI service error:`.

#### Client timeouts

Allow **60–180s** depending on `type` (essay responses are longer to generate).

---

## 6) AI Analyse APIs

> Focused HTTP reference (auth, errors, field tables): **`AI_ANALYSE_API_INTEGRATION.md`**.

> Intro-page flow is separate and optional. If full pages are provided for grading, intro flow can be handled independently.

### POST `/analyse/full`

Full handwritten-page grading (Flutter-aligned JSON). Same auth envelope as other routes.

- Content-Type: `application/json`
- Auth required: Yes

#### Request JSON fields (camelCase)

- `pageImagesBase64` (array of base64-encoded JPEG/PNG bytes, required, non-empty)
- `questionTitle` (string, required)
- `instructionName` (string, optional)
- `modelDescription` (string, required)
- `totalMarks` (integer > 0, required)
- `language` (`en` or `hi`, required)
- `checkLevel` (`Moderate` or `Hard`, optional; default `Moderate`)

#### Success Response (`200`)

All keys under `data` are **snake_case**.

```json
{
  "status": 1,
  "message": "Analysis complete.",
  "data": {
    "student_text": "OCR text...",
    "marks_awarded": 5.0,
    "confidence_percent": 78.5,
    "good_points": "• ...",
    "improvements": "• ...",
    "final_review": "Overall review...",
    "annotations": [
      {
        "page_index": 0,
        "y_position_percent": 25.0,
        "x_start_percent": 10.0,
        "x_end_percent": 50.0,
        "comment": "Feedback",
        "is_positive": true,
        "line_style": "straight"
      }
    ]
  }
}
```

---

### POST `/analyse/copy-ocr`

Whole-PDF OCR for a **single-question essay-style student answer** (handwriting or print). The server uploads the PDF once to Gemini (File API) and runs **one** `generate_content` call; nothing is stored in the database.

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Limits

- **Size:** Rejects uploads larger than **`COPY_OCR_MAX_BYTES`** (default `26214400` ≈ 25 MiB). Configurable in `.env`.
- **Pages:** Counts pages with **pypdf** before calling Gemini; rejects if **`COPY_OCR_MAX_PAGES`** (default `30`) is exceeded.
- The file must start with the PDF magic bytes `%PDF`.

#### Request form fields

- `file` (PDF file, required)
- `language` (string, optional; default `en`) — must be `en` or `hi` when sent

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "OCR complete.",
  "data": {
    "text": "Full plain-text transcript of the student copy…",
    "pageCount": 3
  }
}
```

#### Validation / business errors (`200`, `status: 0`)

- Not a PDF, empty file, over byte or page limit, unreadable PDF, or OCR returned empty text: explanatory `message`.
- Missing `GEMINI_API_KEY`: same pattern as other Gemini routes.

#### AI / parsing failures (`500`)

Same pattern as `/analyse/full`: `status: 0`, `message` prefixed with `AI service error:`.

#### Client timeouts

Allow **60–180s**; large or dense scans may need the upper end.

---

### POST `/analyse/copy-ocr-rasterization`

Same use case as **`/analyse/copy-ocr`** (single-question essay copy → plain text, **no DB**), but each PDF page is **rasterized** (PyMuPDF) to PNG and sent to Gemini in **parallel** with up to **`COPY_OCR_PARALLEL_WORKERS`** concurrent threads (default **5**). Full text is page sections joined as `--- Page N ---` blocks.

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Limits

- Same **`COPY_OCR_MAX_BYTES`** and **`COPY_OCR_MAX_PAGES`** as `/analyse/copy-ocr`.
- **`COPY_OCR_RASTER_DPI`** (default `150`) — render resolution for each page image.
- **`COPY_OCR_PARALLEL_WORKERS`** (default `5`) — max concurrent Gemini calls (capped at page count).

#### Request form fields

- `file` (PDF file, required)
- `language` (string, optional; default `en`) — `en` or `hi`

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "OCR complete.",
  "data": {
    "text": "--- Page 1 ---\n…\n\n--- Page 2 ---\n…",
    "pageCount": 2,
    "rasterDpi": 150,
    "parallelWorkers": 5
  }
}
```

#### Notes

- **More Gemini calls** than `/analyse/copy-ocr` (one per page); watch quotas and allow **longer client timeouts** when many pages are processed in parallel batches.

#### Errors

Same pattern as `/analyse/copy-ocr` (`status: 0` for validation, `500` + `AI service error:` for AI failures).

---

### POST `/analyse/cached-ocr`

- Content-Type: `application/json`
- Auth required: Yes

#### Request (camelCase)

```json
{
  "cachedStudentText": "OCR text from previous response...",
  "questionTitle": "Q3...",
  "instructionName": "optional examiner notes",
  "modelDescription": "Marking scheme...",
  "totalMarks": 8,
  "pageCount": 2,
  "language": "en",
  "checkLevel": "Moderate"
}
```

#### Success Response (`200`)

Same shape as **`/analyse/full`** (snake_case `data`). `data.student_text` is always the **request’s** `cachedStudentText`.

---

### POST `/analyse/combined-review`

- Content-Type: `application/json`
- Auth required: Yes

#### Request (camelCase)

```json
{
  "questionResults": [
    {
      "questionNo": "1",
      "title": "Q1 ...",
      "marksAwarded": 6.0,
      "marksTotal": 8,
      "goodPoints": "• ...",
      "improvements": "• ...",
      "finalReview": "..."
    }
  ]
}
```

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Combined review generated.",
  "data": {
    "overall_review": "Long consolidated review...",
    "overall_improvements": "Line1\nLine2\nLine3\nLine4",
    "one_thing_to_write": "Single top practice action."
  }
}
```

#### Possible AI Error (`500`)

```json
{
  "status": 0,
  "message": "AI service error: Could not parse combined review JSON",
  "data": {}
}
```

---

### POST `/analyse/intro-page`

- Content-Type: `application/json`
- Auth required: Yes

#### Request JSON

- `pageImageBase64` (single JPEG/PNG image as base64, required)

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Intro page analysed.",
  "data": {
    "cells": [
      {
        "question_no": 1,
        "marks_text": "4",
        "x_percent": 73.5,
        "y_percent": 30.2
      }
    ]
  }
}
```

`cells` may be an empty array if no table rows could be parsed.

---

## 7) Error Handling Matrix

| Case | HTTP | Body |
|---|---:|---|
| Missing/invalid token | 401 | `status: -1` |
| Validation/business rule failure | 200 (sometimes 422) | `status: 0` |
| AI service / parsing failure | 500 | `status: 0`, `message` starts with `AI service error:` |

### Auth Error Example

```json
{
  "status": -1,
  "message": "Invalid or expired token.",
  "data": {}
}
```

---

## 8) Client Integration Guide

### Recommended Flow

1. `POST /auth/signup` or `POST /auth/login`
2. Store `accessToken`
3. Attach bearer token for all protected routes
4. For model workflow:
   - `POST /models/key` (sets `title`, `lang`, and optional `type` / booklet mode: `standard`, `custom`, `custom_with_model`, or `essay`)
   - `POST /models/answer-booklet` with `id` and `file` only (mode comes from the key)
   - manage via `GET/PUT/DELETE /models*` endpoints
   - Optional: `POST /model/answer-expand` to generate a model answer from a question only (no DB; `type` `standard` / `custom` / `custom_with_model` / `essay` for length — see section 5)
5. For AI checking workflow:
   - `POST /analyse/full` (JSON base64 page images + grading)
   - optional `POST /analyse/copy-ocr` or `POST /analyse/copy-ocr-rasterization` (essay PDF → plain `text` + `pageCount`, no DB; rasterization = per-page parallel OCR)
   - optionally `POST /analyse/cached-ocr`
   - `POST /analyse/combined-review`
   - optionally `POST /analyse/intro-page` (separate intro-only use case; JSON `pageImageBase64`)

### Token Usage

```http
Authorization: Bearer <accessToken>
```

### Timeout / Retry Suggestions (client)

- Use longer timeout for AI endpoints (60-180s depending on image/page count). `POST /analyse/copy-ocr` for multi-page PDFs: prefer **60–180s**. `POST /analyse/copy-ocr-rasterization` with many pages: prefer the **upper end** or higher (multiple sequential Gemini waves).
- `POST /models/answer-booklet` for a key with `type=custom`, `type=custom_with_model`, or `type=essay` can take several minutes (one Gemini call per extracted question plus the import pass; `essay` answers are longer to generate).
- `POST /model/answer-expand`: allow at least **60–180s**; prefer the upper end when `type` is `essay`.
- Retry on network failures and `5xx` with exponential backoff.
---

## 9) Quick cURL Snippets

### Login

```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"test123","password":"Pass@123"}'
```

### Create key

```bash
curl -X POST "http://127.0.0.1:8000/models/key" \
  -H "Authorization: Bearer <token>" \
  -F "title=Physics Model" \
  -F "lang=en" \
  -F "type=standard"
```

### Create key (custom QnA booklet)

```bash
curl -X POST "http://127.0.0.1:8000/models/key" \
  -H "Authorization: Bearer <token>" \
  -F "title=Physics Model" \
  -F "lang=en" \
  -F "type=custom"
```

### Create key (essay-length answers from booklet PDF)

```bash
curl -X POST "http://127.0.0.1:8000/models/key" \
  -H "Authorization: Bearer <token>" \
  -F "title=History Essay Key" \
  -F "lang=en" \
  -F "type=essay"
```

### Upload booklet

```bash
curl -X POST "http://127.0.0.1:8000/models/answer-booklet" \
  -H "Authorization: Bearer <token>" \
  -F "id=<model_id>" \
  -F "file=@/absolute/path/booklet.pdf;type=application/pdf"
```

### Expand model answer (stateless)

```bash
curl -X POST "http://127.0.0.1:8000/model/answer-expand" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "essay",
    "question": "Discuss the main causes of climate change.",
    "language": "en"
  }'
```

### Essay copy OCR (PDF)

```bash
curl -X POST "http://127.0.0.1:8000/analyse/copy-ocr" \
  -H "Authorization: Bearer <token>" \
  -F "file=@/absolute/path/student_essay.pdf;type=application/pdf" \
  -F "language=en"
```

### Essay copy OCR — rasterization + parallel pages

```bash
curl -X POST "http://127.0.0.1:8000/analyse/copy-ocr-rasterization" \
  -H "Authorization: Bearer <token>" \
  -F "file=@/absolute/path/student_essay.pdf;type=application/pdf" \
  -F "language=en"
```

### Analyse full (JSON base64 images)

Build `PAGE_B64` with `base64 -w0 page1.jpg` (or your tooling), then:

```bash
curl -X POST "http://127.0.0.1:8000/analyse/full" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"pageImagesBase64\":[\"$PAGE_B64\"],\"questionTitle\":\"Q3...\",\"modelDescription\":\"Key points...\",\"totalMarks\":8,\"language\":\"en\",\"checkLevel\":\"Moderate\"}"
```

### Analyse intro page (JSON base64)

```bash
curl -X POST "http://127.0.0.1:8000/analyse/intro-page" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"pageImageBase64\":\"$PAGE_B64\"}"
```

