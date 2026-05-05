# AI Copy Checker — Backend API Spec (GeminiService Replacement)

This spec is generated from current `lib/services/gemini_service.dart` so backend can implement exactly, and Flutter can switch with minimal edits.

## Goal

Move Gemini calls to backend. Flutter should call backend endpoints and receive already-normalized JSON.

## Base URL

`https://<your-domain>` — paths below are rooted at the API host (e.g. local `http://127.0.0.1:8000`). Authoritative URLs: **`AI_ANALYSE_API_INTEGRATION.md`**.

- **`POST /analyse/full`** is **`multipart/form-data` only** — form fields `modelId`, `questionId`, and repeated **`pageImages`** file parts (no JSON body, no base64). Other routes below use **`application/json`** unless noted.
- `Authorization: Bearer <token>` (recommended)
- Responses use JSON with stable schema

---

## Endpoints (Direct Mapping)

| Flutter method in `gemini_service.dart` | Backend endpoint |
|---|---|
| `analyse(...)` / `analyseInIsolate(...)` | `POST /analyse/full` |
| `_analyseFromCachedText(...)` / `analyseWithCachedOcr(...)` | `POST /analyse/cached-ocr` |
| `_doCombinedKeyReview(...)` / `generateCombinedKeyReview(...)` | `POST /analyse/combined-review` |
| `_doAnalyseIntroPage(...)` / `analyseIntroPageInIsolate(...)` | `POST /analyse/intro-page` |

---

## Shared Response Objects

### Response key casing

Success payloads under `data` use **snake_case** keys (e.g. `student_text`, `marks_awarded`). Token usage is not returned by this backend.

### Annotation Object (API shape)

```json
{
  "page_index": 0,
  "y_position_percent": 42.5,
  "x_start_percent": 20.0,
  "x_end_percent": 53.0,
  "comment": "Well articulated — this directly addresses the marking scheme.",
  "is_positive": true,
  "line_style": "straight"
}
```

---

## 1) Full Analysis (Images)

### `POST /analyse/full`

### Request (`multipart/form-data`)

Form fields: **`modelId`**, **`questionId`**, optional **`checkLevel`**.  
Repeated file field **`pageImages`** — one image per answer page (order preserved).

Question title, marking scheme (`desc`), total marks, language, and examiner **`instruction_name`** come from the stored model (`GET /models/{modelId}`). See **`AI_ANALYSE_API_INTEGRATION.md`**.

### Response (`data` fields, snake_case)

```json
{
  "student_text": "...",
  "marks_awarded": 3.5,
  "confidence_percent": 74.0,
  "good_points": "• ...",
  "improvements": "• ...",
  "final_review": "...",
  "annotations": [],
  "model_id": "...",
  "question_id": "..."
}
```

### Hard Requirements

- Clamp marks: `0 <= marks_awarded <= totalMarks`
- Clamp confidence: `0 <= confidence_percent <= 100`
- Clamp each annotation `page_index` to `[0, number_of_uploaded_page_images - 1]`
- Retry model call up to 3 times if JSON parse fails
- Output annotations should be positive-only in current behavior:
  - `isPositive: true`
  - `lineStyle: "straight"`

---

## 2) Cached OCR Re-analysis

### `POST /analyse/cached-ocr`

### Request (JSON)

```json
{
  "modelId": "<uuid>",
  "questionId": "q-eng-001",
  "cachedStudentText": "...",
  "checkLevel": "Moderate"
}
```

Metadata, **`instruction_name`** (for prompts), and page-count hint are resolved server-side from the stored question.

### Response

Same as full analysis response object.

### Hard Requirements

- Same normalization/rules as full analysis
- `student_text` in response should match the incoming `cachedStudentText`

---

## 3) Combined Key Review

### `POST /analyse/combined-review`

### Request (JSON)

```json
{
  "modelId": "<uuid>",
  "questionResults": [
    {
      "questionId": "q-eng-001",
      "marksAwarded": 3.5,
      "improvements": "• ...",
      "goodPoints": "• ...",
      "finalReview": "..."
    }
  ]
}
```

`questionNo`, `title`, and `marksTotal` are merged server-side from the database.

### Response (`data`, snake_case)

```json
{
  "overall_improvements": "line1\nline2\nline3\nline4",
  "one_thing_to_write": "single sentence",
  "overall_review": "long flowing paragraph"
}
```

### Hard Requirements

- Retry + parse/repair fallback exactly like analysis calls
- If model returns `finalReview`, prefer that; otherwise fallback to `overallReview`

---

## 4) Intro Page Marks Extraction

### `POST /analyse/intro-page`

### Request

```json
{
  "pageImageBase64": "<base64-jpeg>"
}
```

### Response (`data`, snake_case)

```json
{
  "cells": [
    {
      "question_no": 1,
      "marks_text": "4",
      "x_percent": 74.0,
      "y_percent": 30.0
    }
  ]
}
```

### Hard Requirements

- Keep all rows including blank marks cells
- If Gemini output is malformed, use regex fallback extraction
- If no candidates, return empty `cells`

---

## Prompt Templates (Copy to Backend)

Backend should use prompts equivalent to current Flutter file.
To avoid drift, backend should copy **verbatim** prompt bodies from:
- `GeminiService._doAnalyseIntroPage`
- `GeminiService._doCombinedKeyReview`
- `GeminiService._analyseFromCachedText`
- `GeminiService.analyse`

### A) Full Analysis Prompt Template

Use this template with variable substitutions:
- `${instructionBlock}` = empty OR `EXTRA ANSWER INSTRUCTIONS:\n{instructionName}\n\n`
- `${checkLevelInstruction}` = strictness line based on `checkLevel`
- `${languageBlock}` = Hindi or English block
- `${questionTitle}`, `${modelDescription}`, `${totalMarks}`, `${maxPageIndex}`

```text
You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

You will be provided with images of the student's answer pages in sequence. Some pages may appear mostly white or contain very little visible ink — the student may have written lightly or used a light pen. ALWAYS attempt to read all pages. If a page genuinely has no answer at all, note it but still grade the rest of the answer.

${instructionBlock}QUESTION TITLE:
${questionTitle}

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
${modelDescription}

TOTAL MARKS FOR THIS QUESTION: ${totalMarks}

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
${checkLevelInstruction}
- Award marksAwarded as a DECIMAL in multiples of 0.5 ...
- [same rules as current gemini_service.dart]

YOUR TASKS:
1. READ the handwritten text from all images ...
2. GRADE objectively ...
3. ANNOTATE ...

ANNOTATION PLACEMENT RULES ...
- IMPORTANT: ALL annotations MUST have "isPositive": true.

${languageBlock}

OUTPUT FORMAT:
Return ONLY valid JSON with keys:
{
  "studentText": "...",
  "marksAwarded": ...,
  "confidencePercent": ...,
  "goodPoints": "...",
  "improvements": "...",
  "finalReview": "...",
  "annotations": [
    {
      "pageIndex": <0..${maxPageIndex}>,
      "yPositionPercent": ...,
      "xStartPercent": ...,
      "xEndPercent": ...,
      "comment": "...",
      "isPositive": true,
      "lineStyle": "straight"
    }
  ]
}
```

Backend implementation note: use the exact body currently in `GeminiService.analyse` and only substitute variables; do not paraphrase.

### B) Cached OCR Prompt Template

Same structure as full-analysis but with:
- OCR text block (`cachedStudentText`) instead of images
- page count mention in annotation section
- same positive-only annotation rule
- backend should copy exact prompt from `GeminiService._analyseFromCachedText`

### C) Combined Review Prompt Template

Use current prompt style from Flutter:
- Input: text summary of each question (`questionResults`)
- Output JSON keys:
  - `finalReview`
  - `overallImprovements`
  - `oneThingToWrite`
- backend should copy exact prompt from `GeminiService._doCombinedKeyReview`

### D) Intro Page Prompt Template

Return strictly pipe-delimited lines:

`questionNo|marksText|xPercent|yPercent`

No markdown/no JSON in raw generation text.
- backend should copy exact prompt from `GeminiService._doAnalyseIntroPage`

---

## Full Prompt Appendix (Verbatim Source)

This section gives backend developers the exact prompt content used now in Flutter, so implementation can be done without opening `gemini_service.dart`.

### 1) Full Analysis Prompt (`GeminiService.analyse`)

```text
You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

You will be provided with images of the student's answer pages in sequence. Some pages may appear mostly white or contain very little visible ink — the student may have written lightly or used a light pen. ALWAYS attempt to read all pages. If a page genuinely has no answer at all, note it but still grade the rest of the answer.

${instructionName != null && instructionName.isNotEmpty ? 'EXTRA ANSWER INSTRUCTIONS:\n$instructionName\n\n' : ''}QUESTION TITLE:
$questionTitle

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
$modelDescription

TOTAL MARKS FOR THIS QUESTION: $totalMarks

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
$checkLevelInstruction
- Award marksAwarded as a DECIMAL in multiples of 0.5 (e.g. 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 …). NEVER exceed $totalMarks.
- Marking tiers by question size:
    8-mark question  → Bahot achha (excellent): 3.5  |  Moderate: 2.5  |  Low: 1.5
    12-mark question → Bahot achha (excellent): 5    |  Moderate: 3    |  Low: 2.5
    Other totals     → scale proportionally using 0.5-step decimals.
- You are a STRICT examiner. Default to FEWER marks, not more.
- FULL marks ONLY if answer nearly perfectly matches the scheme — all key points, correct terminology, clear reasoning. This is rare.
- Do NOT reward effort or length — reward accuracy and relevance only.
- A long but mostly irrelevant answer scores LOW. A short precise answer can outscore a long vague one.
- Do NOT give benefit of the doubt. If a key point is not clearly stated, do not assume it was implied.
- Deduct for: wrong facts, missing key terms, no examples when required, incorrect conclusions.
- When in doubt between two values, always choose the LOWER one.

YOUR TASKS:

1. READ the handwritten text from all images (including labels, captions, diagram text).
2. GRADE objectively against the marking scheme. If the scheme expects diagrams/figures, evaluate whether the student addressed those (sketches, descriptions, labels).
3. ANNOTATE the answer: mark 2–5 specific spots in the student's writing with short teacher-style comments. Make annotations PRECISE — annotate only the specific word/phrase, not the whole line.

ANNOTATION PLACEMENT RULES (CRITICAL — prevents overlapping comments):
- On any single page, every annotation MUST have a yPositionPercent that differs by AT LEAST 12 from every other annotation on that same page. Space them out evenly across the page height.
- If two annotations are naturally close together (within 12% of each other vertically), pick only the more important one; do NOT place both at nearly the same y position.
- Alternate comment placement: for annotations that are on the LEFT half of the text (xEndPercent < 55), keep xEndPercent ≤ 55. For annotations on the RIGHT half (xStartPercent > 45), keep xStartPercent ≥ 45. This ensures comments are anchored to different horizontal zones and do not collide.
- Prefer spreading annotations across different pages when the answer spans multiple pages — do not cluster all annotations on page 0.
- IMPORTANT: ALL annotations MUST have "isPositive": true. Only circle/annotate things done correctly. Do NOT mark errors with annotations — errors should be mentioned in the "improvements" field instead.

TONE & LANGUAGE GUIDELINES:

Write ALL feedback as a professional yet approachable teacher would — clear, constructive, and specific. Avoid robotic phrasing. Think of the tone an experienced senior examiner uses: direct, helpful, respectful.

GOOD annotation comments (professional but human):
  - "Well articulated — this directly addresses the marking scheme."
  - "Correct definition. Clear and concise."
  - "This definition is inaccurate — please revise from your textbook."
  - "You started well but left this point incomplete. Elaborate further next time."
  - "Good reasoning, but the correct formula is F = ma, not F = mv."
  - "Diagram is present but labels are missing — always label axes and units."

BAD (too robotic/generic — AVOID these):
  - "The student correctly identified the concept."
  - "Improvement needed in this area."
  - "Good point."
  - "Incorrect."

For goodPoints: Address the student directly and be specific about what was done well.
For improvements: Be specific — mention what was missed and why it matters for marks.
For finalReview: Write 2–3 sentences as a professional remark — constructive and encouraging.

${language == 'hi' ? '''LANGUAGE INSTRUCTION:
This is a Hindi-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in Hindi. No English feedback text at all.

HINDI VOCABULARY RULES (apply strictly):
- Use "saraahaniyan" or "Utkrisht" for praise — NOT just "achha".
- Use "Prayas" when noting effort.
- Nishkarsh (conclusion) must be future-oriented: "Aapka nishkarsh bhavishya unmukhi hona chahiye."
- Address the student as "Aap" / "Aapka" — NEVER "Tumne" or "Tu".
- "Introduction" → write "Prashtavana" or "Parichay" — never the English word.
- NEVER use the word "Shabash".
- "Utpatti" → use "Prashtuti".
- Strong conclusion: "Aapka nishkarsh prabhavshali hai."
- Line could be more specific: "Yah line aur vishishth ho sakti hai; udaharan ke sath samjhaya ja sakta tha."
- Replace "Sahi dhang se samjhaya hai" with "Sahi dhang se prastut kiya hai."''' : '''LANGUAGE INSTRUCTION:
This is an English-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in English. No Hindi feedback text at all.'''}

══════════════════════════════════════════
OUTPUT FORMAT:
══════════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no explanation outside JSON) with exactly these keys:
{
  "studentText": "<transcribe the exact handwritten text you see across all pages — skip printed question headers>",
  "marksAwarded": <decimal in multiples of 0.5, range 0 to $totalMarks; never exceed $totalMarks>,
  "confidencePercent": <float 0-100>,
  "goodPoints": "<bullet-point list — each point should sound like a real teacher praising the student>",
  "improvements": "<bullet-point list — each point should sound like a real teacher pointing out what to fix and why>",
  "finalReview": "<2-3 sentence handwritten-note-style overall review — warm, personal, constructive>",
  "annotations": [
    {
      "pageIndex": <int 0-indexed corresponding to the image sequence. Maximum is ${pageImages.length - 1}>,
      "yPositionPercent": <float 0-100 indicating the approximate vertical position of the specific text to underline>,
      "xStartPercent": <float 0-100 indicating the tight horizontal start of the specific word(s) to underline>,
      "xEndPercent": <float 0-100 indicating the tight horizontal end of the specific word(s) to underline>,
      "comment": "<short, warm, colloquial teacher remark — praise what's right, sound human>",
      "isPositive": true,
      "lineStyle": "straight"
    }
  ]
}
NOTE: Every annotation MUST have "isPositive": true. Do NOT produce any negative/cross annotations. Errors should be mentioned only in the "improvements" field.
```

### 2) Cached OCR Prompt (`GeminiService._analyseFromCachedText`)

```text
You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

The student's handwritten answer has already been transcribed for you (OCR result):
"""
$cachedStudentText
"""

${instructionName != null && instructionName.isNotEmpty ? 'EXTRA ANSWER INSTRUCTIONS:\n$instructionName\n\n' : ''}QUESTION TITLE:
$questionTitle

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
$modelDescription

TOTAL MARKS FOR THIS QUESTION: $totalMarks

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
$checkLevelInstruction
- Award marksAwarded as a DECIMAL in multiples of 0.5 (e.g. 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 …). NEVER exceed $totalMarks.
- Marking tiers by question size:
    8-mark question  → Bahot achha (excellent): 3.5  |  Moderate: 2.5  |  Low: 1.5
    12-mark question → Bahot achha (excellent): 5    |  Moderate: 3    |  Low: 2.5
    Other totals     → scale proportionally using 0.5-step decimals.
- You are a STRICT examiner. Default to FEWER marks, not more.
- FULL marks ONLY if answer nearly perfectly matches the scheme — all key points, correct terminology, clear reasoning. This is rare.
- Do NOT reward effort or length — reward accuracy and relevance only.
- A long but mostly irrelevant answer scores LOW. A short precise answer can outscore a long vague one.
- Do NOT give benefit of the doubt. If a key point is not clearly stated, do not assume it was implied.
- Deduct for: wrong facts, missing key terms, no examples when required, incorrect conclusions.
- When in doubt between two values, always choose the LOWER one.

YOUR TASKS:

1. GRADE objectively against the marking scheme based on the transcribed text above.
3. ANNOTATE the answer: mark 2–5 specific spots with short teacher-style comments. Estimate page/position from the text structure (the answer spans $pageCount page(s)).

ANNOTATION PLACEMENT RULES (CRITICAL — prevents overlapping comments):
- On any single page, every annotation MUST have a yPositionPercent that differs by AT LEAST 20 from every other annotation ON THE SAME SIDE (left or right). Space them out evenly across the page height (e.g. 10, 30, 50, 70, 90).
- Two annotations that would be on the SAME side and within 20% of each other vertically → keep only the more important one and DISCARD the other.
- Two annotations that are on OPPOSITE sides (one left half, one right half) CAN share a similar yPositionPercent — this is fine and encouraged to keep them visually spread.
- Decide left vs right using xStartPercent and xEndPercent: if midX = (xStart+xEnd)/2 < 50, place it on the LEFT half (xEndPercent ≤ 50); otherwise on the RIGHT half (xStartPercent ≥ 50). Strictly alternate left/right when possible.
- Prefer spreading annotations across different pages — do not cluster all on page 0.
- Aim for at most 2–3 annotations per page. If you have more, move the extras to other pages or drop the least important ones.
- IMPORTANT: ALL annotations MUST have "isPositive": true. Only mark things worth circling positively (correct facts, good phrases, relevant points). Do NOT mark errors with annotations — errors should be mentioned in the "improvements" field instead.

TONE & LANGUAGE GUIDELINES:
Write ALL feedback as a professional yet approachable teacher — clear, constructive, specific.

For goodPoints: Address the student directly and be specific about what was done well.
For improvements: Be specific — mention what was missed and why it matters for marks.
For finalReview: Write 2–3 sentences as a professional remark — warm, personal, constructive.

${language == 'hi' ? '''LANGUAGE INSTRUCTION:
This is a Hindi-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in Hindi. No English feedback text at all.

HINDI VOCABULARY RULES (apply strictly):
- Use "saraahaniyan" or "Utkrisht" for praise — NOT just "achha".
- Use "Prayas" when noting effort.
- Nishkarsh (conclusion) must be future-oriented: "Aapka nishkarsh bhavishya unmukhi hona chahiye."
- Address the student as "Aap" / "Aapka" — NEVER "Tumne" or "Tu".
- "Introduction" → write "Prashtavana" or "Parichay" — never the English word.
- NEVER use the word "Shabash".
- "Utpatti" → use "Prashtuti".
- Strong conclusion: "Aapka nishkarsh prabhavshali hai."
- Line could be more specific: "Yah line aur vishishth ho sakti hai; udaharan ke sath samjhaya ja sakta tha."
- Replace "Sahi dhang se samjhaya hai" with "Sahi dhang se prastut kiya hai."''' : '''LANGUAGE INSTRUCTION:
This is an English-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in English. No Hindi feedback text at all.'''}

══════════════════════════════════════════
OUTPUT FORMAT:
══════════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no explanation outside JSON) with exactly these keys:
{
  "studentText": "$cachedStudentText",
  "marksAwarded": <decimal in multiples of 0.5, range 0 to $totalMarks; never exceed $totalMarks>,
  "confidencePercent": <float 0-100>,
  "goodPoints": "<bullet-point list — each point should sound like a real teacher praising the student>",
  "improvements": "<bullet-point list — each point should sound like a real teacher pointing out what to fix and why>",
  "finalReview": "<2-3 sentence handwritten-note-style overall review — warm, personal, constructive>",
  "annotations": [
    {
      "pageIndex": <int 0-indexed. Maximum is ${pageCount - 1}>,
      "yPositionPercent": <float 0-100>,
      "xStartPercent": <float 0-100>,
      "xEndPercent": <float 0-100>,
      "comment": "<short, warm teacher remark>",
      "isPositive": true,
      "lineStyle": "straight"
    }
  ]
}
NOTE: Every annotation MUST have "isPositive": true. Do NOT produce negative/cross annotations. Mention errors only in the "improvements" field.
```

### 3) Combined Key Review Prompt (`GeminiService._doCombinedKeyReview`)

```text
You are an experienced school teacher writing a detailed end-of-paper comment for a student.
Below are the per-question analysis results:

$questionSummary

YOUR TASK:
Write a "finalReview" that is a flowing, natural paragraph-style teacher comment of AT LEAST 150 words (aim for 180-220 words). It should read exactly like a real teacher's handwritten remark at the end of a corrected answer sheet — warm, personal, specific, and professional.

Structure (all in one block of plain prose, no headings, no bullets, no numbering, no markdown):
1. Start with 2-3 sentences acknowledging what the student did well across the paper, mentioning specific questions or topics.
2. Write 3-4 sentences identifying the most important weaknesses, with concrete examples from the student's answers (e.g. "In Q3 you left the conclusion incomplete…").
3. Give 2-3 sentences of clear, actionable advice on how to improve — specific study tips or practice habits.
4. End with 1-2 encouraging sentences motivating the student to keep working hard.

Keep every sentence natural and conversational — the way a teacher actually writes, not like a report. Do NOT use any symbols, asterisks, or special characters.

Also return:
- "overallImprovements": 4 plain sentences of improvement points (one per line, no symbols).
- "oneThingToWrite": ONE sentence — the single most impactful practice tip.

LANGUAGE: Match the dominant language in the question improvements above (Hindi or English).

Return ONLY valid JSON, no markdown:
{
  "finalReview": "<flowing paragraph of 150-220 words, sentences separated by \n>",
  "overallImprovements": "<4 lines separated by \n>",
  "oneThingToWrite": "<one sentence>"
}
```

### 4) Intro Page Prompt (`GeminiService._doAnalyseIntroPage`)

```text
You are analysing the COVER / INTRO page of a student exam answer sheet.

The page contains a MARKS TABLE with columns: Q.No. | M.Mark | M.Obt.

TASK:
1. Find the M.Obt. column (where the teacher writes the marks obtained).
2. For EVERY row in that table — question rows AND the Total row — output one line in this exact format:
   questionNo|marksText|xPercent|yPercent
   - questionNo : integer (use 0 for the Total/Grand-Total row)
   - marksText  : the handwritten number if visible, otherwise leave it empty (nothing between the pipes)
   - xPercent   : horizontal centre of the M.Obt. cell as % of page width (0–100)
   - yPercent   : vertical centre of the M.Obt. cell as % of page height (0–100)
3. Include ALL rows, even if M.Obt. cell is empty.
4. Output ONLY the data lines. No headers, no explanation, no JSON, no markdown.

Example (values are illustrative only — use real values from the image):
1|4|74|30
2||74|33
3|3.5|74|36
0|91|74|88
```

---

## Gemini Generation Config

| Flow | temperature | maxOutputTokens | responseMimeType |
|---|---:|---:|---|
| Full analysis | 0.2 | 8192 | `application/json` |
| Cached OCR | 0.2 | 8192 | `application/json` |
| Combined review | 0.35 | 2048 | `application/json` |
| Intro page | 0.0 | 2048 | plain text (no strict JSON mime needed) |

Model: `gemini-2.5-flash`

---

## Flutter Replacement Notes (Minimal Changes in `gemini_service.dart`)

Backend dev can implement now; Flutter side will only replace Gemini HTTP calls:

1. Replace `_url` Gemini endpoint with backend base URL.
2. Replace `_callGeminiRaw(...)` with backend call helper(s), keeping retry + parse strategy if desired.
3. Map backend response JSON directly to:
   - `GeminiAnalysisOutput`
   - `CombinedKeyReviewOutput`
   - `IntroPageAnalysis`
4. Keep existing models (`TeacherAnnotation`, token usage model) unchanged to avoid UI breakage.

---

## Standard Error Shape

```json
{
  "error": "Human readable error",
  "detail": "Optional technical detail"
}
```

---

Last updated: aligned to current `lib/services/gemini_service.dart` (includes `instructionName`, `checkLevel`, positive-only annotations, and full-analysis behavior). Token counts are not returned by this backend.
