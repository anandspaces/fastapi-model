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

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Key uploaded successfully",
  "data": {
    "id": "uuid",
    "title": "Physics Unit Test",
    "lang": "en"
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
  "lang": "en"
}
```

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Model key updated successfully",
  "data": {
    "id": "uuid",
    "title": "Updated title",
    "lang": "en"
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

- Content-Type: `multipart/form-data`
- Auth required: Yes

#### Request form fields

- `id` (string, required; key id)
- `file` (PDF file, required)

#### Success Response (`200`)

```json
{
  "status": 1,
  "message": "Answer booklet uploaded successfully",
  "data": {
    "id": "uuid",
    "questions": [
      {
        "id": "q-eng-001",
        "questionNo": "Q1",
        "title": "Question title",
        "desc": "Answer description",
        "pageNum": 2,
        "marks": 8,
        "diagramDescriptions": []
      }
    ]
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
        "has_booklet": true
      }
    ]
  }
}
```

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
    "question_count": 20,
    "questions": [],
    "created_at": "2026-03-23T00:00:00Z"
  }
}
```

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

## 5) AI Analyse APIs

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

## 6) Error Handling Matrix

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

## 7) Client Integration Guide

### Recommended Flow

1. `POST /auth/signup` or `POST /auth/login`
2. Store `accessToken`
3. Attach bearer token for all protected routes
4. For model workflow:
   - `POST /models/key`
   - `POST /models/answer-booklet`
   - manage via `GET/PUT/DELETE /models*` endpoints
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

- Use longer timeout for AI endpoints (60-180s depending on image/page count).
- Retry on network failures and `5xx` with exponential backoff.
- For `/analyse/intro-page`, treat `422` as a valid “not detected” business outcome.

---

## 8) Quick cURL Snippets

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
  -F "lang=en"
```

### Upload booklet

```bash
curl -X POST "http://127.0.0.1:8000/models/answer-booklet" \
  -H "Authorization: Bearer <token>" \
  -F "id=<model_id>" \
  -F "file=@/absolute/path/booklet.pdf;type=application/pdf"
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

