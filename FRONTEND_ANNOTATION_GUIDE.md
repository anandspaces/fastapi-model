# Frontend Annotation Guide — Copy Checker

How to call the smart-OCR API and render the annotations it returns.

---

## 1. API Call

```http
POST /analyse/smart-ocr
Authorization: Bearer <token>
Content-Type: multipart/form-data

Fields:
  file        (required)  PDF file
  language    en | hi     (default: en)
  modelId                 Answer model ID — enables grading + annotations
  checkLevel  Moderate | Hard   (default: Moderate)
```

**OCR-only** (no `modelId`): returns items with position data but no `annotations` or `marks`.
**With `modelId`**: returns marks, feedback, and per-annotation bboxes ready to render.

---

## 2. Full Response Shape

```jsonc
{
  "message": "Smart OCR complete.",
  "data": {
    "pageCount": 12,
    "skippedPages": [1],          // intro/cover pages — show differently
    "modelId": "abc123",
    "checkLevel": "Moderate",
    "items": [
      {
        // ── Identity ──────────────────────────────────────────
        "question_id": 1,
        "question": "Explain the significance of...",
        "student_answer": "The student wrote...",
        "is_attempted": true,
        "section_name": "General Studies I",
        "answer_type": "paragraph",      // paragraph | word_list | correction

        // ── Page span ─────────────────────────────────────────
        "start_page": 2,                 // 1-indexed
        "start_y_position_percent": 12.5,
        "end_page": 3,
        "end_y_position_percent": 88.0,

        // ── Score box ─────────────────────────────────────────
        // Draw a marks bubble here.
        "marking_page": 2,
        "marking_x_position_percent": 78.3,   // centre of the bubble
        "marking_y_position_percent": 18.0,
        "marking_box_page": 2,
        "marking_box_x1_percent": 72.0,
        "marking_box_y1_percent": 10.5,
        "marking_box_x2_percent": 96.0,
        "marking_box_y2_percent": 24.0,

        // ── Grading (only when modelId provided) ──────────────
        "marks": 6.5,
        "max_marks": 10,
        "status": "partial",             // correct | partial | wrong | unattempted
        "feedback": "Good coverage of…",

        // ── Annotations ───────────────────────────────────────
        "annotations": [
          {
            "page_index": 1,             // 0-based page index
            "y_position_percent": 35.0,  // hint Y anchor on the page
            "x_start_percent": 14.0,     // hint X start
            "x_end_percent": 78.0,       // hint X end
            "comment": "Assertion unsupported — cite policy or statute.",
            "is_positive": false,
            // bbox = the ACTUAL writable zone to render in:
            "bbox": {
              "page": 2,                 // 1-based
              "x1_percent": 72.5,
              "y1_percent": 34.0,
              "x2_percent": 96.0,
              "y2_percent": 38.5
            }
          }
        ]
      }
    ]
  }
}
```

---

## 3. Coordinate System

| Field | Origin | Axis | Range |
|---|---|---|---|
| `*_percent` | top-left of the PDF page | x → right, y → down | 0–100 |
| `page` / `marking_page` / `bbox.page` | 1-indexed | — | 1…N |
| `page_index` (inside annotation) | 0-indexed | — | 0…N-1 |

To convert a percent coordinate to CSS pixels inside a PDF page container:

```js
// pageContainer is the DOM element rendering that page (e.g. a PDF.js canvas wrapper)
const left   = (x1_percent / 100) * pageContainer.offsetWidth;
const top    = (y1_percent / 100) * pageContainer.offsetHeight;
const width  = ((x2_percent - x1_percent) / 100) * pageContainer.offsetWidth;
const height = ((y2_percent - y1_percent) / 100) * pageContainer.offsetHeight;
```

---

## 4. Rendering the Score Box (`marking_box_*`)

Use `marking_box_*_percent` for the bounding rect, `marking_*_position_percent` for the
bubble centre within that rect.

```jsx
// React example
function ScoreBox({ item, pageContainer }) {
  if (!item.marking_box_page || item.marks == null) return null;

  const W = pageContainer.offsetWidth;
  const H = pageContainer.offsetHeight;
  const style = {
    position: "absolute",
    left:   `${item.marking_box_x1_percent}%`,
    top:    `${item.marking_box_y1_percent}%`,
    width:  `${item.marking_box_x2_percent - item.marking_box_x1_percent}%`,
    height: `${item.marking_box_y2_percent - item.marking_box_y1_percent}%`,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    pointerEvents: "none",
  };
  const color = item.status === "correct" ? "#16a34a" :
                item.status === "partial" ? "#d97706" : "#dc2626";

  return (
    <div style={style}>
      <svg viewBox="0 0 60 60" style={{ width: "100%", height: "100%" }}>
        <ellipse cx="30" cy="30" rx="28" ry="28"
                 stroke={color} strokeWidth="2" fill="white" fillOpacity="0.85" />
        <text x="30" y="35" textAnchor="middle"
              fontFamily="cursive" fontSize="16" fill={color}>
          {item.marks}/{item.max_marks}
        </text>
      </svg>
    </div>
  );
}
```

---

## 5. Rendering Annotation Comments (`annotations[].bbox`)

Each annotation has a `bbox` giving the writable zone where the comment fits without
overlapping existing handwriting. Render a callout/badge inside that rect.

`is_positive: true` → green (the student got something right)
`is_positive: false` → red/amber (corrective feedback)
`is_positive` absent → neutral

```jsx
function AnnotationLayer({ item, pageNo }) {
  return item.annotations
    .filter(ann => ann.bbox && ann.bbox.page === pageNo)
    .map((ann, i) => {
      const { x1_percent, y1_percent, x2_percent, y2_percent } = ann.bbox;
      const color = ann.is_positive === true  ? "#16a34a"
                  : ann.is_positive === false ? "#dc2626"
                  : "#6b7280";
      return (
        <div
          key={i}
          style={{
            position: "absolute",
            left:   `${x1_percent}%`,
            top:    `${y1_percent}%`,
            width:  `${x2_percent - x1_percent}%`,
            height: `${y2_percent - y1_percent}%`,
            border: `1.5px solid ${color}`,
            borderRadius: "3px",
            backgroundColor: `${color}15`,   // 8% opacity fill
            padding: "2px 4px",
            fontSize: "clamp(7px, 1.2vw, 11px)",
            fontFamily: "cursive",
            color,
            overflowY: "auto",
            boxSizing: "border-box",
            pointerEvents: "auto",
            lineHeight: 1.3,
          }}
          title={ann.comment}
        >
          {ann.comment}
        </div>
      );
    });
}
```

---

## 6. Full Page Overlay Pattern

Structure your PDF viewer so annotations sit in an absolute-positioned layer that
exactly matches the rendered page canvas:

```html
<div class="page-wrapper" style="position:relative; display:inline-block;">
  <!-- PDF.js or <embed> renders the actual page -->
  <canvas id="pdf-page-2"></canvas>

  <!-- Annotation overlay — same size, absolutely positioned on top -->
  <div class="annotation-overlay"
       style="position:absolute; inset:0; pointer-events:none;">
    <!-- Score box -->
    <ScoreBox item={item} />
    <!-- Per-annotation comments -->
    <AnnotationLayer item={item} pageNo={2} />
  </div>
</div>
```

**Key rule:** the overlay `div` must have the same `width` and `height` as the canvas so
percent coordinates map 1-to-1. Use `ResizeObserver` or CSS to keep them in sync.

---

## 7. Edge Cases

| Situation | What the API returns | How to handle |
|---|---|---|
| No writable space found for a comment | `ann.bbox` is absent | Show comment in a tooltip on hover, or skip the overlay |
| Question not attempted | `is_attempted: false`, `marks: 0`, `annotations: []` | Grey out the marking box |
| Intro / cover page | In `skippedPages` list; no item spans it | Render the page without any overlay |
| Multi-page answer | `start_page ≠ end_page`; annotations have individual `bbox.page` | Render each annotation on its own page |
| `marking_box_*` fields absent | Can happen if cell-grid found no blank zone near the answer | Fall back to `marking_x/y_position_percent` for bubble centre, skip the box |

---

## 8. Testing Locally Without a Server

```bash
# Run the pipeline directly on any PDF (reads GEMINI_API_KEY from .env)
python3 scripts/test_smart_ocr.py test_pdf3.pdf

# With grading (needs a valid model ID from your DB)
python3 scripts/test_smart_ocr.py test_pdf3.pdf --model-id <MODEL_ID>

# Output files
#   scripts/out_test_pdf3_response.json      — full API response
#   scripts/out_test_pdf3_annotations.json   — compact annotation list

# Verify bbox geometry on an existing response JSON
python3 scripts/verify_smart_ocr_bboxes.py \
    --pdf test_pdf3.pdf \
    --response scripts/out_test_pdf3_response.json
```
