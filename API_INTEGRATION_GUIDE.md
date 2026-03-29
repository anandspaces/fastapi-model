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
- `type` (string, optional; default `standard`) — `standard` or `custom`. Stored on the key and used when you later call `POST /models/answer-booklet` for this `id` (no `type` field on answer-booklet).

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

`booklet_type` is optional. If omitted, the existing value on the key is kept. If a booklet row already exists for this id, `answer_models.booklet_type` is updated to match.

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

Uploads a PDF for an existing model key. Processing mode (**standard** vs **custom**) is taken from the key’s stored `type` / `booklet_type` (set at `POST /models/key` or `PUT /models/key/{key_id}`), not from this request.

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Request form fields

- `id` (string, required; key id from `POST /models/key`)
- `file` (PDF file, required)

#### Booklet modes (from the key)

- **`standard`** — Full booklet extraction via Gemini ([`process_pdf_path`](src/gemini_extract.py)): marks, page hints, diagrams metadata, etc.
- **`custom`** — Import **questions** from the PDF, then **one Gemini call per question** for a model answer. Answer language follows the key’s `lang`.

#### Success Response (`200`)

Same envelope and `data` keys for both types:

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

When the key is `custom`, `data.booklet_type` is `"custom"`. Each question uses the same keys; typically `pageNum` is `1`, `marks` is `0`, and `desc` holds the generated model answer.

#### No questions found (key `custom` only, `200` business failure)

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

`booklet_type` is `null` when `has_booklet` is false; otherwise `"standard"` or `"custom"`.

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

Other APIs can branch on `booklet_type` when different behavior is needed for custom vs standard models.

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

Creates a new question on an existing answer model. Requires **`POST /models/answer-booklet`** to have succeeded for this `model_id` first (otherwise `Model not found`).

#### Request

(no `id` in the body — the server assigns the next `q-eng-NNN` id):

```json
{
  "questionNo": "Q1",
  "title": "New question title",
  "desc": "Full question / model answer text",
  "pageNum": 1,
  "marks": 5,
  "diagramDescriptions": []
}
```

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

- No answer model for this id / user: `Model not found`.
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
  "pageNum": 2,
  "marks": 8,
  "diagramDescriptions": []
}
```

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

Generates a full examiner-style model answer from the **question text only** using Gemini (no user draft). **Nothing is stored** in the database. This endpoint’s `type` field is **only** for answer length (`standard` / `custom` ≈ 300 words, `essay` ≈ 1200 words). It is **not** the same as the booklet `type` on `POST /models/key` (which remains `standard` or `custom` only).

Response `diagramDescriptions` uses the same camelCase array shape as question objects elsewhere (`diagramDescriptions` on each question).

**Breaking change:** Earlier versions required `answer` and optional `diagram_description` in the request and returned a single `diagram_description` string. Clients must send only `type` and `question` (plus optional `language`) and must read `diagramDescriptions` as an array.

Implementation reference: [`src/gemini_expand_model_answer.py`](src/gemini_expand_model_answer.py).

### POST `/model/answer-expand`

- Content-Type: `application/json`
- Auth required: Yes

#### Request (JSON body)

| Field | Type | Required | Notes |
|--------|------|----------|--------|
| `type` | string | Yes | `standard`, `custom`, or `essay` (case-insensitive after trim). `standard` and `custom` use the same target length (~300 words). `essay` uses ~1200 words. |
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
  "message": "type must be \"standard\", \"custom\", or \"essay\".",
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

> Intro-page flow is separate and optional. If full pages are provided for grading, intro flow can be handled independently.

### POST `/analyse/pages`

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Request form fields

- `pages` (repeatable image files, required)
- `question_title` (string, required)
- `model_description` (string, required)
- `total_marks` (integer > 0, required)
- `language` (`en` or `hi`, required)

#### Success Response (`200`)

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

### POST `/analyse/cached-ocr`

- Content-Type: `application/json`
- Auth required: Yes

#### Request

```json
{
  "cached_student_text": "OCR text from previous response...",
  "question_title": "Q3...",
  "model_description": "Marking scheme...",
  "total_marks": 8,
  "page_count": 2,
  "language": "en"
}
```

#### Success Response (`200`)

Same structure as `/analyse/pages`.

---

### POST `/analyse/combined-review`

- Content-Type: `application/json`
- Auth required: Yes

#### Request

```json
{
  "question_results": [
    {
      "question_no": "1",
      "title": "Q1 ...",
      "marks_awarded": 6.0,
      "marks_total": 8,
      "good_points": "• ...",
      "improvements": "• ...",
      "final_review": "..."
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
    "final_review": "Long consolidated review...",
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

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Request form fields

- `page` (single image, required)

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

#### Controlled No-table Response (`422`)

```json
{
  "status": 0,
  "message": "Could not detect a marks table on this page",
  "data": {}
}
```

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
   - `POST /models/key` (sets `title`, `lang`, and optional `type` / booklet mode: `standard` or `custom`)
   - `POST /models/answer-booklet` with `id` and `file` only (mode comes from the key)
   - manage via `GET/PUT/DELETE /models*` endpoints
   - Optional: `POST /model/answer-expand` to generate a model answer from a question only (no DB; `type` `standard` / `custom` / `essay` for length — see section 5)
5. For AI checking workflow:
   - `POST /analyse/pages`
   - optionally `POST /analyse/cached-ocr`
   - `POST /analyse/combined-review`
   - optionally `POST /analyse/intro-page` (separate intro-only use case)

### Token Usage

```http
Authorization: Bearer <accessToken>
```

### Timeout / Retry Suggestions (client)

- Use longer timeout for AI endpoints (60-180s depending on image/page count). `POST /models/answer-booklet` for a key with `type=custom` can take several minutes (one Gemini call per extracted question plus the import pass).
- `POST /model/answer-expand`: allow at least **60–180s**; prefer the upper end when `type` is `essay`.
- Retry on network failures and `5xx` with exponential backoff.
- For `/analyse/intro-page`, treat `422` as a valid “not detected” business outcome.

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

### Analyse pages

```bash
curl -X POST "http://127.0.0.1:8000/analyse/pages" \
  -H "Authorization: Bearer <token>" \
  -F "pages=@/absolute/path/page1.jpg" \
  -F "pages=@/absolute/path/page2.jpg" \
  -F "question_title=Q3. Explain photosynthesis..." \
  -F "model_description=Key points..." \
  -F "total_marks=8" \
  -F "language=en"
```

### Analyse intro page

```bash
curl -X POST "http://127.0.0.1:8000/analyse/intro-page" \
  -H "Authorization: Bearer <token>" \
  -F "page=@/absolute/path/intro_page.jpg"
```

