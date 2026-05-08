# Frontend Annotation Guide — Copy Checker (Cell-ID Edition)

How to call the smart-OCR API and render the annotations it returns. The API
now uses a **cell-ID grid** (Excel-style: A1, B5, AC12, …) as the primary
placement contract. Every page in the response is partitioned into a fixed
grid; every annotation tells you which cells to render in. Convert cell IDs
→ percent rectangles via the `cellGridMeta` block included on every response.

> **TL;DR for migrating frontends:** read `annotation.cell_ids` /
> `annotation.range_id` first. The legacy `annotation.bbox` is still emitted
> (computed from the cell IDs) for backwards compatibility, so existing
> bbox-based renderers keep working without changes.

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

**OCR-only** (no `modelId`): returns items with position data, no `annotations`.
**With `modelId`**: returns marks, feedback, and per-annotation cell-IDs +
bbox.

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

    // ── Cell-grid meta (NEW, primary) ──────────────────────────
    // One block per page in the analysed PDF. The frontend uses these to
    // turn any cell ID anywhere in the response into a percent rectangle.
    "cellGridMeta": [
      {
        "page": 1,                 // 1-based
        "rows": 70,                // count of cell rows on the page
        "cols": 49,                // count of cell columns
        "cell_size_pts": 12.0,     // square cell side, PDF points
        "left_margin_pts": 3.66,   // centred — cell A1 starts here in pts
        "top_margin_pts": 0.96,
        "page_w_pts": 595.0,
        "page_h_pts": 842.0
      }
      // …
    ],

    "items": [
      {
        "question_id": 1,
        "question": "Explain the significance of...",
        "student_answer": "The student wrote...",
        "is_attempted": true,
        "section_name": "General Studies I",
        "answer_type": "paragraph",

        "start_page": 2,
        "start_y_position_percent": 12.5,
        "end_page": 3,
        "end_y_position_percent": 88.0,

        "marking_page": 2,
        "marking_x_position_percent": 78.3,
        "marking_y_position_percent": 18.0,
        "marking_box_page": 2,
        "marking_box_x1_percent": 72.0,
        "marking_box_y1_percent": 10.5,
        "marking_box_x2_percent": 96.0,
        "marking_box_y2_percent": 24.0,

        "marks": 6.5,
        "max_marks": 10,
        "status": "partial",
        "feedback": "Good coverage of…",

        "annotations": [
          {
            "page_index": 1,             // 0-based
            "y_position_percent": 35.0,
            "x_start_percent": 14.0,
            "x_end_percent": 78.0,
            "comment": "Assertion unsupported — cite policy or statute.",
            "is_positive": false,

            // ── NEW (primary placement contract) ───────────────
            "cell_ids": [
              "C12", "D12", "E12", "F12", "G12", "H12"
            ],
            "range_id": "C12:H12",       // compact span; convenient for logs / Gemini round-trips
            "placement_tier": 1,         // 1=region, 2=run, 3=synthesised (see §7)

            // ── Legacy bbox (computed from cell_ids) ───────────
            // Identical shape to the previous contract — keep using if you
            // have not migrated to cell IDs yet. Always present.
            "bbox": {
              "page": 2,
              "x1_percent": 7.5,
              "y1_percent": 16.0,
              "x2_percent": 18.0,
              "y2_percent": 17.7
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
| `cell_size_pts`, `*_margin_pts`, `page_w_pts`, `page_h_pts` | PDF points (1 pt = 1/72 inch) | — | float |
| `cell_ids[]` / `range_id` | cell A1 = top-left of grid | row down, col right | "A1" … "AW70" (typical A4) |
| `page` / `marking_page` / `bbox.page` | 1-indexed | — | 1…N |
| `page_index` (inside annotation) | 0-indexed | — | 0…N-1 |

---

## 4. Cell-ID Conventions

```
col 1..26   →  A,B,…,Z
col 27..52  →  AA,AB,…,AZ
col 53..78  →  BA,BB,…,BZ
…
row 1..N    →  numeric suffix  (no zero padding)
```

**Examples**

| Cell | row | col |
|---|---|---|
| `A1`   | 1  | 1  |
| `Z1`   | 1  | 26 |
| `AA1`  | 1  | 27 |
| `AC12` | 12 | 29 |
| `AW70` | 70 | 49 |

**Range syntax**

* Single cell: `"A1"`.
* Rectangle (start ≤ end, inclusive): `"C12:H12"`, `"D5:K9"`.
* `cell_ids[]` (the list) is enumerated **left → right, top → bottom** within
  the laid-out rectangle. For a 2 × 3 placement starting at C12 you'll get
  `["C12","D12","E12","C13","D13","E13"]`.

---

## 5. Resolving cell IDs → percent rectangles

Drop these helpers into your renderer. They are pure functions; no API
round-trip needed once you have `cellGridMeta` for the page.

```js
// ─── Cell-ID → percent rectangle ────────────────────────────────────────
//   meta = cellGridMeta entry for the page
//   cellId = "AC12"
function cellIdToBbox(cellId, meta) {
  const m = cellId.match(/^([A-Z]+)(\d+)$/);
  if (!m) throw new Error(`bad cell id: ${cellId}`);
  let col = 0;
  for (const ch of m[1]) col = col * 26 + (ch.charCodeAt(0) - 64);
  const row = parseInt(m[2], 10);

  const x1_pts = meta.left_margin_pts + (col - 1) * meta.cell_size_pts;
  const y1_pts = meta.top_margin_pts  + (row - 1) * meta.cell_size_pts;
  const x2_pts = x1_pts + meta.cell_size_pts;
  const y2_pts = y1_pts + meta.cell_size_pts;

  return {
    x1_percent: (x1_pts / meta.page_w_pts) * 100,
    y1_percent: (y1_pts / meta.page_h_pts) * 100,
    x2_percent: (x2_pts / meta.page_w_pts) * 100,
    y2_percent: (y2_pts / meta.page_h_pts) * 100,
  };
}

// ─── Range-ID → percent rectangle (union of endpoints) ──────────────────
//   rangeId = "C12:H12" or "D5:K9" or "A1"
function rangeIdToBbox(rangeId, meta) {
  const [start, end] = rangeId.includes(":") ? rangeId.split(":") : [rangeId, rangeId];
  const a = cellIdToBbox(start, meta);
  const b = cellIdToBbox(end,   meta);
  return {
    x1_percent: Math.min(a.x1_percent, b.x1_percent),
    y1_percent: Math.min(a.y1_percent, b.y1_percent),
    x2_percent: Math.max(a.x2_percent, b.x2_percent),
    y2_percent: Math.max(a.y2_percent, b.y2_percent),
  };
}

// ─── Cell-IDs list → tight union rectangle ──────────────────────────────
function cellIdsToBbox(cellIds, meta) {
  if (!cellIds?.length) return null;
  let x1 = +Infinity, y1 = +Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const cid of cellIds) {
    const r = cellIdToBbox(cid, meta);
    x1 = Math.min(x1, r.x1_percent);
    y1 = Math.min(y1, r.y1_percent);
    x2 = Math.max(x2, r.x2_percent);
    y2 = Math.max(y2, r.y2_percent);
  }
  return { x1_percent: x1, y1_percent: y1, x2_percent: x2, y2_percent: y2 };
}
```

The percent → CSS-pixel conversion (unchanged from the previous contract):

```js
const left   = (rect.x1_percent / 100) * pageContainer.offsetWidth;
const top    = (rect.y1_percent / 100) * pageContainer.offsetHeight;
const width  = ((rect.x2_percent - rect.x1_percent) / 100) * pageContainer.offsetWidth;
const height = ((rect.y2_percent - rect.y1_percent) / 100) * pageContainer.offsetHeight;
```

---

## 6. Rendering Recipes

### 6.1 Score Box (`marking_box_*`)

Unchanged — still uses percent fields:

```jsx
function ScoreBox({ item, pageContainer }) {
  if (!item.marking_box_page || item.marks == null) return null;
  const style = {
    position: "absolute",
    left:   `${item.marking_box_x1_percent}%`,
    top:    `${item.marking_box_y1_percent}%`,
    width:  `${item.marking_box_x2_percent - item.marking_box_x1_percent}%`,
    height: `${item.marking_box_y2_percent - item.marking_box_y1_percent}%`,
    display: "flex", alignItems: "center", justifyContent: "center",
    pointerEvents: "none",
  };
  const color = item.status === "correct" ? "#16a34a"
              : item.status === "partial" ? "#d97706" : "#dc2626";
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

### 6.2 Annotation comment (cell-ID first, bbox fallback)

```jsx
function AnnotationLayer({ item, pageNo, cellGridMeta }) {
  const meta = cellGridMeta.find(m => m.page === pageNo);

  return item.annotations
    .filter(ann => (ann.bbox?.page ?? null) === pageNo)
    .map((ann, i) => {
      // Prefer the cell-ID contract; fall back to legacy bbox
      const rect = (ann.range_id && meta)
        ? rangeIdToBbox(ann.range_id, meta)
        : ann.bbox;
      const color = ann.is_positive === true  ? "#16a34a"
                  : ann.is_positive === false ? "#dc2626"
                  : "#6b7280";
      return (
        <div
          key={i}
          style={{
            position: "absolute",
            left:   `${rect.x1_percent}%`,
            top:    `${rect.y1_percent}%`,
            width:  `${rect.x2_percent - rect.x1_percent}%`,
            height: `${rect.y2_percent - rect.y1_percent}%`,
            border: `1.5px solid ${color}`,
            borderRadius: "3px",
            backgroundColor: `${color}15`,
            padding: "2px 4px",
            fontSize: "clamp(7px, 1.2vw, 11px)",
            fontFamily: "cursive",
            color,
            overflowY: "auto",
            boxSizing: "border-box",
            pointerEvents: "auto",
            lineHeight: 1.3,
          }}
          title={`${ann.range_id ?? ""} • ${ann.comment}`}
        >
          {ann.comment}
        </div>
      );
    });
}
```

### 6.3 Multi-cell run highlight (e.g., underlining a phrase)

If you want to highlight every cell individually rather than as one
rectangle (useful for non-contiguous lists), iterate `cell_ids[]` and call
`cellIdToBbox` for each.

```jsx
{ann.cell_ids.map(cid => {
  const r = cellIdToBbox(cid, meta);
  return <div className="cell-hl" style={{ left:`${r.x1_percent}%`, /*…*/ }} />;
})}
```

### 6.4 Full Page Overlay Pattern

Same as before — annotation layer absolutely positioned over the page canvas:

```html
<div class="page-wrapper" style="position:relative; display:inline-block;">
  <canvas id="pdf-page-2"></canvas>
  <div class="annotation-overlay"
       style="position:absolute; inset:0; pointer-events:none;">
    <ScoreBox item={item} />
    <AnnotationLayer item={item} pageNo={2} cellGridMeta={cellGridMeta} />
  </div>
</div>
```

The overlay `div` MUST have the same `width` and `height` as the canvas so
percent coordinates map 1-to-1 — use `ResizeObserver` or matching CSS.

---

## 7. Placement Tiers (`placement_tier`)

The backend resolves each annotation's cells using a 3-tier strategy.
`placement_tier` tells you which tier won.

| Tier | Meaning | When you'd see it |
|---|---|---|
| **1 — region** | Annotation placed inside a 2-D writable region — a coherent rectangle of blank cells (margin column, gap band, half-page diagram void). Best quality. | Most common on normal answer sheets. |
| **2 — run** | No region was big enough; fell back to a single horizontal run of writable cells on the row nearest the anchor (with optional wrap). | Dense pages or very long comments. |
| **3 — synth** | No grid available for the page (e.g., raster failed) **or** no run/region could fit the comment — bbox synthesised from the annotation's hint x/y ± 1.5%. May visually overlap handwriting. | Edge case; show with extra styling if you want. |

`cell_ids` and `range_id` are always populated for tiers 1–2, and empty for
tier 3 (`bbox` is still set — derived from raw coordinates).

---

## 8. Edge Cases

| Situation | What the API returns | How to handle |
|---|---|---|
| Dense page — no large free zone | `placement_tier: 2`, cells in narrow row run | Render normally — bbox is valid. |
| Completely full page — nothing fits | `placement_tier: 3`, empty `cell_ids`, synthesised `bbox` | Render normally; callout may overlap handwriting. |
| Question not attempted | `is_attempted: false`, `marks: 0`, `annotations: []` | Grey out the marking box. |
| Intro / cover page | Listed in `skippedPages`; no item spans it. `cellGridMeta` still includes the page (for completeness). | Render the page without any overlay. |
| Multi-page answer | `start_page ≠ end_page`; each annotation has its own `bbox.page` and `cell_ids` for that specific page. | Render each annotation on its own page. |
| `marking_box_*` fields absent | Cell-grid found no blank zone near the answer | Fall back to `marking_x/y_position_percent` for the bubble centre, skip the box. |
| Cell ID outside grid (shouldn't happen) | `range_id` referencing a non-existent row/col | Validate against `meta.rows` / `meta.cols`; log + fall back to `bbox`. |

---

## 9. Migration from the bbox-only contract

| Field | Status | Notes |
|---|---|---|
| `cellGridMeta` | **NEW** (top-level data block) | Use to convert any cell ID → percent. |
| `annotation.cell_ids` | **NEW** | Primary placement; ordered left→right, top→bottom. |
| `annotation.range_id` | **NEW** | Compact "A1:B5" form for logs / cross-system traffic. |
| `annotation.placement_tier` | **NEW** | 1 region / 2 run / 3 synth. |
| `annotation.bbox` | **KEPT** | Unchanged shape; computed server-side from `cell_ids`. |
| `marking_box_*_percent` | **KEPT** | Unchanged. (A future release may add `marking_box_range_id`.) |

**Recommended migration**

1. Read `cellGridMeta` and stash by page.
2. For each annotation: try `range_id`/`cell_ids` first via
   `rangeIdToBbox`. If absent (very old responses), fall back to
   `annotation.bbox`.
3. Existing renderers that only know `bbox` keep working — no breaking
   change.
4. To take advantage of higher-fidelity placement, render multi-cell runs
   per cell rather than as a single rectangle.

---

## 10. Local Testing

```bash
# Run the pipeline directly on any PDF (reads GEMINI_API_KEY from .env)
python3 scripts/test_smart_ocr.py test_pdf3.pdf

# With grading (needs a valid model ID from your DB)
python3 scripts/test_smart_ocr.py test_pdf3.pdf --model-id <MODEL_ID>

# Render the v4 grid as a vector overlay PDF (the artefact Gemini reads)
python3 cell_grid_service_v4.py test_original_pdf1.pdf overlay_pdf1_v4.pdf

# Output files
#   scripts/out_test_pdf3_response.json      — full API response
#   scripts/out_test_pdf3_annotations.json   — compact annotation list

# Verify bbox geometry on an existing response JSON
python3 scripts/verify_smart_ocr_bboxes.py \
    --pdf test_pdf3.pdf \
    --response scripts/out_test_pdf3_response.json
```

---

## 11. FAQ

**Why cells instead of just bboxes?**
The grid is the same coordinate system Gemini sees on the overlay PDF, so
the model can return cell IDs directly. The frontend resolves those IDs to
exact rectangles using a one-line formula — round-trip with zero loss.

**What if Gemini returns a cell ID outside the grid?**
The backend validates against `meta.rows`/`meta.cols` and either snaps to
the nearest valid cell or drops to tier 2/3. By the time the response
reaches the frontend, every emitted `cell_id` is guaranteed in-bounds.

**How do I render a multi-cell run as one box?**
Use `rangeIdToBbox(ann.range_id, meta)` — it returns the union of the
start/end cells, which equals the layout rectangle for any contiguous run.
For non-contiguous cell lists, iterate `cell_ids[]`.

**Can I ignore `cellGridMeta` and just use `bbox`?**
Yes. The legacy contract still works. You only need the meta block when you
want to render at cell granularity or accept new annotation sources keyed
by cell ID.
