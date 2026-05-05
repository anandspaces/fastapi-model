# AI Analyse APIs — Integration Guide

HTTP reference for the Gemini-backed **grading**, **cached re-analysis**, **combined paper review**, and **intro-page marks table** flows. These replace direct Flutter `GeminiService` calls.

## Authentication

All endpoints below require:

```http
Authorization: Bearer <accessToken>
```

Obtain `<accessToken>` from `POST /auth/login` or `POST /auth/signup`. Response shape:

```json
{
  "status": 1,
  "message": "Login successful.",
  "data": {
    "accessToken": "<jwt>",
    "tokenType": "Bearer",
    "expiresIn": 3600,
    "username": "..."
  }
}
```

| Situation | HTTP status | Body |
|-----------|--------------|------|
| Missing / invalid Bearer token | `401` | `"status": -1`, `"message": "..."` |
| Validation / business rule | Usually `200` | `"status": 0`, `"message": "..."` |
| AI or unexpected server failure | `500` | `"status": 0`, `"message": "AI service error: ..."` |

Pydantic may return **`422 Unprocessable Entity`** if required JSON fields are missing or fail constraints (e.g. empty arrays).

## Response envelope (success)

```json
{
  "status": 1,
  "message": "Human-readable message",
  "data": { }
}
```

All **`data` keys for these four endpoints are snake_case** (e.g. `student_text`, `marks_awarded`, `overall_review`). There is **no** token-usage or billing metadata in the response.

## Environment (server)

| Variable | Role |
|----------|------|
| `GEMINI_API_KEY` | Required for any Gemini call (loaded via `load_api_key()`). |
| `GEMINI_ANALYSE_MODEL` | Optional; model id passed to Gemini for these flows. Set explicitly in production (e.g. `gemini-2.5-flash`). See `.env.example`. |
| `JWT_SECRET` | Required for issuing and validating Bearer tokens. |

---

## 1. Full page-image analysis

**`POST /analyse/full`**

Grades handwritten answer pages sent as base64 images (JPEG/PNG). Content-Type: **`application/json`**.

### Request body

Clients may use **camelCase** names (Flutter-style); snake_case is also accepted.

| Field | JSON name(s) | Type | Notes |
|-------|----------------|------|--------|
| Page images | `pageImagesBase64` | `string[]` | Required, min length 1. Each element is standard base64 (padding allowed). Decoded bytes must be non-empty. |
| Question title | `questionTitle` | `string` | Required. |
| Instructions | `instructionName` | `string` \| omitted | Optional extra examiner instructions. |
| Marking scheme | `modelDescription` | `string` | Required. |
| Total marks | `totalMarks` | `int` | Required, ≥ 1. |
| Language | `language` | `string` | Required: `en` or `hi` (case-insensitive after trim). |
| Strictness | `checkLevel` | `string` | Optional; default `Moderate`. Must normalize to **`Moderate`** or **`Hard`** (see errors). |

### Success `data`

| Key | Type | Description |
|-----|------|-------------|
| `student_text` | `string` | Transcribed / synthesized student answer text. |
| `marks_awarded` | `number` | Decimal marks (0.5 steps), capped by `totalMarks`. |
| `confidence_percent` | `number` | 0–100. |
| `good_points` | `string` | Bullet-style strengths. |
| `improvements` | `string` | Bullet-style gaps. |
| `final_review` | `string` | Short overall remark. |
| `annotations` | `array` | See **Annotation object** below. |

### Annotation object

Each element:

| Key | Type |
|-----|------|
| `page_index` | `integer` — 0-based page index in the request image list. |
| `y_position_percent` | `number` |
| `x_start_percent` | `number` |
| `x_end_percent` | `number` |
| `comment` | `string` |
| `is_positive` | `boolean` |
| `line_style` | `string` (e.g. `straight`) |

### Typical errors

- **`400`** — Invalid base64 or empty decoded bytes for an entry in `pageImagesBase64` (`Invalid base64...` / `Empty image bytes...`).
- **`200`, `status: 0`** — `language must be en or hi`; `checkLevel must be "Moderate" or "Hard"`; missing API key message from config.
- **`500`** — Gemini failure or JSON parse failure after retries (`AI service error: ...`).

### Client timeouts

Allow **60–180 seconds** depending on image count and size.

---

## 2. Cached OCR re-analysis

**`POST /analyse/cached-ocr`**

Same grading logic as full analysis, but the model receives **text** instead of images. The returned **`student_text` is always overwritten** with the request’s `cachedStudentText` so clients can rely on stable OCR text. Content-Type: **`application/json`**.

### Request body

| Field | JSON name(s) | Type | Notes |
|-------|----------------|------|--------|
| Cached transcript | `cachedStudentText` | `string` | Required, non-empty. |
| Question title | `questionTitle` | `string` | Required. |
| Instructions | `instructionName` | optional | |
| Marking scheme | `modelDescription` | `string` | Required. |
| Total marks | `totalMarks` | `int` | ≥ 1. |
| Page count | `pageCount` | `int` | ≥ 1 (used for annotation placement hints). |
| Language | `language` | `string` | `en` or `hi`. |
| Strictness | `checkLevel` | `string` | Same as full analysis. |

### Success `data`

Same shape as **`POST /analyse/full`**. Guaranteed: `data.student_text === request.cachedStudentText`.

---

## 3. Combined end-of-paper review

**`POST /analyse/combined-review`**

Builds one consolidated teacher-style summary from per-question results. Content-Type: **`application/json`**.

### Request body

Top-level:

| Field | JSON name | Type |
|-------|-----------|------|
| Per-question rows | `questionResults` | array (min length 1) |

Each item in `questionResults`:

| Field | JSON name | Type |
|-------|-----------|------|
| Question id / label | `questionNo` | `string` |
| Title | `title` | `string` |
| Marks awarded | `marksAwarded` | `number` |
| Marks total | `marksTotal` | `integer` |
| Good points | `goodPoints` | `string` |
| Improvements | `improvements` | `string` |
| Final review | `finalReview` | `string` |

### Success `data`

| Key | Type |
|-----|------|
| `overall_review` | `string` — long consolidated paragraph (maps from model `final_review` / legacy keys internally). |
| `overall_improvements` | `string` — typically several lines. |
| `one_thing_to_write` | `string` — single actionable tip. |

### Typical errors

- **`500`** — Parse failure or Gemini error (`AI service error: ...`).

---

## 4. Intro / cover page marks table

**`POST /analyse/intro-page`**

Extracts handwritten marks from the **M.Obt.** column of a cover marks table. Content-Type: **`application/json`**.

### Request body

| Field | JSON name | Type |
|-------|-----------|------|
| Single page image | `pageImageBase64` | `string` — non-empty base64 (JPEG/PNG). |

### Success `data`

```json
{
  "cells": [
    {
      "question_no": 1,
      "marks_text": "4",
      "x_percent": 73.5,
      "y_percent": 30.2
    }
  ]
}
```

- `question_no`: use **`0`** for the Total / Grand Total row.
- `marks_text`: empty string if the cell has no visible mark.
- `cells` may be **`[]`** if nothing could be parsed.

### Typical errors

- **`400`** — `Invalid pageImageBase64.` or empty decoded bytes.

---

## Related endpoints (same `/analyse/` prefix)

These use the same auth envelope but different content types; see **`API_INTEGRATION_GUIDE.md`** section 6 for multipart PDF OCR (`/analyse/copy-ocr`, `/analyse/copy-ocr-rasterization`) and other flows.

---

## Example: login + full analysis

```bash
TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"you","password":"yourpassword"}' \
  | jq -r '.data.accessToken')

PAGE_B64=$(base64 -w0 page1.jpg)

curl -s -X POST "http://127.0.0.1:8000/analyse/full" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"pageImagesBase64\":[\"$PAGE_B64\"],\"questionTitle\":\"Q3\",\"modelDescription\":\"Marking scheme...\",\"totalMarks\":8,\"language\":\"en\",\"checkLevel\":\"Moderate\"}"
```

## Example: intro page

```bash
curl -s -X POST "http://127.0.0.1:8000/analyse/intro-page" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"pageImageBase64\":\"$PAGE_B64\"}"
```

---

## Spec alignment

Flutter-oriented field naming and prompt semantics are summarized in **`API_GUIDE.md`**. This document is the source of truth for **URLs**, **HTTP codes**, and **JSON envelopes** as implemented in `main.py` and `src/schemas.py`.
