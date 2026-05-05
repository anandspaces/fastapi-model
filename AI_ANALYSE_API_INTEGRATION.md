# Analyse module — API integration

All routes require **`Authorization: Bearer <accessToken>`** unless noted. Success envelope:

```json
{ "status": 1, "message": "…", "data": { } }
```

Errors: **`status`** `0` (validation / business / AI) or **`-1`** (auth). AI failures often use HTTP **500** with `status: 0` and `message` starting with `AI service error:`.

**Base URL (local):** `http://127.0.0.1:8000`

---

## Route summary

| Method | Path | Content-Type | Purpose |
|--------|------|----------------|---------|
| POST | `/analyse/full` | `multipart/form-data` | Grade handwritten pages (images + model/question ids). |
| POST | `/analyse/cached-ocr` | `application/json` | Re-grade using cached OCR text + ids. |
| POST | `/analyse/combined-review` | `application/json` | End-of-paper combined comment from compact per-question results + ids. |
| POST | `/analyse/intro-page` | `application/json` | Extract intro/cover marks table from one image. |
| POST | `/analyse/copy-ocr` | `multipart/form-data` | Essay PDF → full-text OCR. |
| POST | `/analyse/copy-ocr-rasterization` | `multipart/form-data` | PDF → per-page raster OCR (parallel). |
| POST | `/analyse/smart-ocr` | `multipart/form-data` | Booklet PDF → structured answers; optional grading with `modelId`. |
| POST | `/analyse/free-space` | `multipart/form-data` | Detect empty zones for annotations. |
| POST | `/analyse/snap-annotations` | `application/json` | Snap annotations to free-space zones. |

---

## 1. `POST /analyse/full`

### Form data only (not JSON)

This endpoint accepts **`multipart/form-data` only**. Do **not** send `application/json`, `pageImagesBase64`, or any base64-encoded bodies—pages must be sent as **real image bytes** in repeated **`pageImages`** file parts (JPEG/PNG typical). That avoids base64 size overhead and matches mobile/web camera uploads.

Marking scheme, marks, and language are loaded from the stored answer model (`GET /models/{model_id}`) for the authenticated user.

Sending **`Content-Type: application/json`** to this URL will not parse as this handler expects (`422` / unparsed body).

### Request parts (`multipart/form-data`)

| Part | Name | Required | Description |
|------|------|----------|-------------|
| Field | `modelId` | Yes | Answer model id (same as key id). |
| Field | `questionId` | Yes | Question id from `data.questions[].id`. |
| Field | `checkLevel` | No | `Moderate` or `Hard` (default `Moderate`). |
| File | `pageImages` | Yes (≥ 1) | One JPEG/PNG per page, order = page order. |

### Success response (`data`, snake_case)

```json
{
  "status": 1,
  "message": "Analysis complete.",
  "data": {
    "student_text": "…",
    "marks_awarded": 5.0,
    "confidence_percent": 78.5,
    "good_points": "…",
    "improvements": "…",
    "final_review": "…",
    "annotations": [
      {
        "page_index": 0,
        "y_position_percent": 25.0,
        "x_start_percent": 10.0,
        "x_end_percent": 50.0,
        "comment": "…",
        "is_positive": true,
        "line_style": "straight"
      }
    ],
    "model_id": "<modelId>",
    "question_id": "<questionId>"
  }
}
```

### Server-derived fields

- **`questionTitle` / scheme / `totalMarks`** from stored question (`title`, `desc`, `marks`); **`marks`** must be ≥ 1.
- Extra examiner instructions for Gemini come only from the stored question’s **`instruction_name`** (no request field).
- **`language`** from model **`lang`** (`en` or `hi`).
- **`page_count`** for clamping annotations comes from uploaded image count.

### Typical errors

- **400** — Missing/empty `pageImages`; model/question not found; invalid `checkLevel`; stored question missing title/desc or marks &lt; 1; model `lang` not `en`/`hi`.

### cURL example

```bash
curl -s -X POST "$BASE/analyse/full" \
  -H "Authorization: Bearer $TOKEN" \
  -F "modelId=$MID" \
  -F "questionId=$QID" \
  -F "checkLevel=Moderate" \
  -F "pageImages=@page1.jpg;type=image/jpeg" \
  -F "pageImages=@page2.jpg;type=image/jpeg"
```

---

## 2. `POST /analyse/cached-ocr`

### Request body (JSON)

No **`instructionName`** field — examiner instructions come only from the stored question’s **`instruction_name`**.

```json
{
  "modelId": "<uuid>",
  "questionId": "q-eng-001",
  "cachedStudentText": "…",
  "checkLevel": "Moderate"
}
```

| Field | Required | Notes |
|-------|----------|--------|
| `modelId` | Yes | |
| `questionId` | Yes | |
| `cachedStudentText` | Yes | Non-empty. |
| `checkLevel` | No | Default `Moderate`. |

### Server-derived

Same title/desc/marks/language/instruction as **`/analyse/full`**. **`page_count`** = `max(1, question.pageNum)` from stored question.

### Success response

Same shape as **`/analyse/full`**. **`data.student_text`** is always the request **`cachedStudentText`**.

```json
{
  "status": 1,
  "message": "Analysis complete.",
  "data": {
    "student_text": "<same as cachedStudentText>",
    "marks_awarded": 4.5,
    "confidence_percent": 70.0,
    "good_points": "…",
    "improvements": "…",
    "final_review": "…",
    "annotations": [],
    "model_id": "<modelId>",
    "question_id": "<questionId>"
  }
}
```

---

## 3. `POST /analyse/combined-review`

### Request body (JSON)

**Compact rows only** — `questionNo`, `title`, `marksTotal` are filled from the database.

```json
{
  "modelId": "<uuid>",
  "questionResults": [
    {
      "questionId": "q-eng-001",
      "marksAwarded": 6.0,
      "goodPoints": "• …",
      "improvements": "• …",
      "finalReview": "…"
    }
  ]
}
```

### Success response

```json
{
  "status": 1,
  "message": "Combined review generated.",
  "data": {
    "overall_review": "…",
    "overall_improvements": "…",
    "one_thing_to_write": "…"
  }
}
```

### Typical errors

- **400** — Model not found; unknown `questionId` for that model.

---

## 4. `POST /analyse/intro-page`

### Request (JSON)

```json
{ "pageImageBase64": "<base64-jpeg-or-png>" }
```

### Success response

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

`question_no` **`0`** = total row. `cells` may be `[]`.

---

## 5. `POST /analyse/copy-ocr`

Whole-PDF OCR for a single-question essay PDF.

### Request (multipart)

| Field | Required | Description |
|-------|----------|-------------|
| `file` | Yes | PDF. |
| `language` | No | `en` or `hi`. If omitted/empty, **`modelId`** may supply language from the stored model. |
| `modelId` | No | If set and `language` not set, language defaults from model `lang`. |

### Success response

```json
{
  "status": 1,
  "message": "OCR complete.",
  "data": {
    "text": "…",
    "pageCount": 3
  }
}
```

---

## 6. `POST /analyse/copy-ocr-rasterization`

Same as copy-OCR but rasterizes each page; limits **`COPY_OCR_MAX_BYTES`**, **`COPY_OCR_MAX_PAGES`**, **`COPY_OCR_RASTER_DPI`**, **`COPY_OCR_PARALLEL_WORKERS`**.

### Request (multipart)

Same as **`/analyse/copy-ocr`**: `file`, optional `language`, optional `modelId`.

### Success response

```json
{
  "status": 1,
  "message": "OCR complete.",
  "data": {
    "text": "--- Page 1 ---\n…",
    "pageCount": 2,
    "rasterDpi": 220,
    "parallelWorkers": 5
  }
}
```

Exact keys (`rasterDpi`, `parallelWorkers`, etc.) match runtime env — see [`API_INTEGRATION_GUIDE.md`](API_INTEGRATION_GUIDE.md).

---

## 7. `POST /analyse/smart-ocr`

### Request (multipart)

| Field | Required | Description |
|-------|----------|-------------|
| `file` | Yes | Student booklet PDF. |
| `language` | No | Default `en`. |
| `modelId` | No | If set, runs grading vs stored model. |
| `snapAnnotations` | No | Default `true`. |
| `checkLevel` | No | `Moderate` or `Hard` when grading. |

### Success response (shape)

```json
{
  "status": 1,
  "message": "Smart OCR complete.",
  "data": {
    "pageCount": 10,
    "items": [],
    "skippedPages": [1],
    "modelId": "<optional>",
    "checkLevel": "Moderate"
  }
}
```

See OpenAPI `/docs` for full `items` structure.

---

## 8. `POST /analyse/free-space`

### Request (multipart)

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `file` | Yes | — | PDF |
| `rows` | No | 20 | Grid rows (4–40). |
| `cols` | No | 8 | Grid cols (4–20). |
| `min_score` | No | 0.65 | Emptiness threshold 0–1. |

### Success response

```json
{
  "status": 1,
  "message": "Free space analysis complete.",
  "data": {
    "pageCount": 3,
    "gridRows": 20,
    "gridCols": 8,
    "minScore": 0.65,
    "pages": [
      {
        "pageIndex": 0,
        "freeZones": [
          {
            "xPercent": 10.0,
            "yPercent": 20.0,
            "widthPercent": 80.0,
            "heightPercent": 15.0,
            "score": 0.72
          }
        ]
      }
    ]
  }
}
```

---

## 9. `POST /analyse/snap-annotations`

### Request (JSON)

Body matches **`SnapAnnotationsRequest`**: `items` (from smart-ocr), `pages` (from free-space).

```json
{
  "items": [],
  "pages": []
}
```

### Success response

```json
{
  "status": 1,
  "message": "Annotations snapped.",
  "data": {
    "items": [],
    "snappedCount": 3,
    "totalAnnotations": 5
  }
}
```

---

## Environment

| Variable | Role |
|----------|------|
| `GEMINI_API_KEY` | Required for Gemini routes. |
| `GEMINI_ANALYSE_MODEL` | Optional model id for analyse flows. |
| `JWT_SECRET` | Bearer auth. |
| `COPY_OCR_*` | Limits for copy-OCR routes (see `.env.example`). |

---

## See also

- **[`API_GUIDE.md`](API_GUIDE.md)** — Flutter-oriented prompt alignment (legacy naming notes).
- **[`API_INTEGRATION_GUIDE.md`](API_INTEGRATION_GUIDE.md)** — Broader API surface.
