# CLAUDE.md — Teacher Annotation Engine
## `teacher_annotate.py` — Unified UPSC Answer-Sheet Annotation System

---

## Project Purpose

Single-script PDF annotation engine that rasterises answer-sheet PDF pages and
renders teacher-authentic ink overlays — wavy underlines, organic circles,
natural tick marks, curved arrows, curly braces, exponent carets — with remark
text in Homemade Apple handwriting font. Output looks like a human teacher
checked the paper with a red pen.

**Status: BUILT** — `teacher_annotate.py` (1100 LOC) + `test_teacher_annotate.py` (regression suite, 14 tests, all passing).

**Predecessor scripts (kept on disk for reference, not used by the new pipeline):**

| Old File | What It Did | Status |
|---|---|---|
| `annotate_pdf_final.py` | Drawing primitives + CLI | 📚 Reference (kept) |
| `free_space_detector.py` | CV free-space analysis (3 techniques) | 📚 Reference (kept) |
| `pdf_region_annotator.py` | Connected-component writable regions | 📚 Reference (kept) |
| `cell_grid_service.py` | Rigid 25pt cell grid (deprecated) | ❌ Don't use |

**Everything now lives in one file: `teacher_annotate.py`**

### Currently Implemented vs Deferred

| Feature | Status |
|---|---|
| 3 CV free-space techniques (strips, word gaps, free rects) | ✅ Done |
| All 6 drawing primitives | ✅ Done |
| ProximityLayout with **pre-seeded** occupied slots | ✅ Done (BUG FIX applied) |
| `wavy_underline` band ceiling clamp | ✅ Done (BUG FIX applied) |
| `organic_arrow` control-point free-band routing | ✅ Done (BUG FIX applied) |
| `place_in_gap` text wrapping | ✅ Done (BUG FIX applied) |
| `RemarkResolver` with auto / auto_line / auto_gap | ✅ Done |
| `remark_zone="gap"` → `right_margin` fallback | ✅ Done |
| `--scaffold-dense` (auto-fill placement stress test) | ✅ Done |
| `--analyze-only` (layout JSON without rendering) | ✅ Done |
| `WritableRegion` connected-component labeling | ⏸ Deferred (current 3 techniques + pre-seeded ProximityLayout proved sufficient) |
| Multi-line underline (`x_start2`/`x_end2`) | ✅ Done |
| Frontend JSON-coordinate output | ⏸ Deferred (current pipeline writes annotated PDF; structured coord export is a future flag) |

---

## Repository Layout

```
teacher_annotate.py               ← ONLY file you ever run or edit
test_teacher_annotate.py          ← regression test suite
fonts/
  HomemadeApple-Regular.ttf       ← only font used (must be present)
test_original_pdf1.pdf            ← 2-page test fixture (fast iteration)
test_original_pdf2.pdf            ← 2-page test fixture (denser content)
test_pdf3.pdf                     ← 52-page stress-test fixture
test_remarks_pdf1.json            ← remarks spec for pdf1
test_remarks_pdf2.json            ← remarks spec for pdf2
test_remarks.json                 ← remarks spec for pdf3
data/                             ← any additional test assets
```

---

## CLI Usage

```bash
# Annotate a PDF (real flow)
python3 teacher_annotate.py input.pdf remarks.json output.pdf --dpi 250

# Analyse free space only — prints layout JSON to stdout
python3 teacher_annotate.py input.pdf --analyze-only

# Generate dense auto-filled remarks JSON (placement stress test)
python3 teacher_annotate.py input.pdf --scaffold-dense > dense.json
python3 teacher_annotate.py input.pdf dense.json out_dense.pdf --dpi 250
```

All progress logs go to stderr; stdout is reserved for JSON output of the
`--analyze-only` and `--scaffold-dense` modes so they pipe cleanly.

---

## Function Index — `teacher_annotate.py`

Quick lookup of where each piece lives in the actual file (line numbers approximate, search for the section header):

| Function / Class | Section | Purpose |
|---|---|---|
| `_color_for(severity)` | §1 | severity → RGB |
| `GapInfo`, `FreeStrip`, `WordGap`, `FreeRect`, `LayoutMap`, `ResolvedRemark` | §2 | dataclasses |
| `LayoutMap.line_y(n)` | §2 | safe 1-indexed line lookup |
| `LayoutMap.best_gap_for_line(line)` | §2 | gap within ±5 lines |
| `LayoutMap.best_score_position()` | §2 | top-right free rect |
| `load_pdf_pages_pil(pdf, dpi)` | §3 | PyMuPDF rasterise |
| `detect_lines(gray, dpi)` | §4 | CC centroid clustering |
| `detect_gaps(line_ys, H, lh, min_gap)` | §4 | inter-line bands |
| `detect_strips(...)` | §4 | Technique 1 — morph dilation |
| `detect_word_gaps(...)` | §4 | Technique 2 — column projection |
| `detect_free_rects(...)` | §4 | Technique 3 — Canny + contours |
| `build_layout_map(pil_img, page, dpi)` | §4 | runs ALL detectors once |
| `find_ink_top` / `find_ink_bottom` | §5 | row-projection ink edges |
| `snap_to_word_boundaries(...)` | §5 | expand x range to word edges |
| `snap_exponent_gap(layout, line, x)` | §5 | nearest word gap |
| `wavy_underline(arr, gray, ...)` | §6 | ink-bottom-following wave (BUG FIX: y_bot ceiling) |
| `organic_ellipse(arr, cx, cy, rx, ry, ...)` | §6 | wobbly hand-drawn oval |
| `natural_tick(arr, cx, cy, size, ...)` | §6 | two Bezier strokes |
| `_free_band_centre(bands, lo, hi)` | §6 | overlap-best free band |
| `organic_arrow(arr, sx, sy, dx, dy, ..., free_bands)` | §6 | Bezier arrow (BUG FIX: snaps cpy through free band) |
| `curly_brace(arr, y_top, y_bot, bx, side, ...)` | §6 | cubic Bezier { or } |
| `exponent_caret(arr, cx, caret_y, ...)` | §6 | ^ insertion mark |
| `paste_hw(layer, text, x, y, sz, color)` | §7 | render Homemade Apple |
| `ProximityLayout(layout)` | §7 | non-overlap text slots (BUG FIX: pre-seeded with occupied ranges) |
| `place_in_gap(layer, text, gap, body_l, body_r, sz, color)` | §7 | wrap + centre in gap (BUG FIX: now wraps) |
| `RemarkResolver(layout).resolve_all(raw)` | §8 | JSON → ResolvedRemark |
| `_free_bands_for_arrows(layout)` | §9 | union of strips + gaps for routing |
| `draw_page(pil_img, resolved, layout)` | §9 | dispatch loop |
| `_DENSE_GAP_REMARKS`, `_DENSE_MARGIN_REMARKS` | §10 | pre-canned text pool |
| `scaffold_dense(layout)` | §10 | auto-generates dense remark spec |
| `process_pdf(pdf_in, remarks, pdf_out, dpi, *, dense, analyze_only)` | §10 | top-level pipeline |
| `_cli()` | §11 | argparse, dispatches to process_pdf |

---

## Internal Architecture

```
teacher_annotate.py is divided into 11 sections:

SECTION 1   Constants & Config
SECTION 2   Data Classes          (LayoutMap, ResolvedRemark, FreeStrip,
                                   WordGap, FreeRect, WritableRegion, GapInfo)
SECTION 3   PDF Ingestion         (_load_pdf_pages_pil, fitz fallback)
SECTION 4   Layout Intelligence   (build_layout_map — runs once per page)
              4a  Line detection   character-centroid clustering
              4b  Gap detection    inter-line blank bands
              4c  Free strips      Technique 1: morphological dilation
              4d  Word gaps        Technique 2: column projection
              4e  Free rects       Technique 3: Canny + contour bounding boxes
              4f  Writable regions connected-component labeling (scipy.ndimage)
SECTION 5   Precision Snapping
              5a  snap_to_word_boundaries
              5b  snap_exponent_gap
              5c  find_ink_top / find_ink_bottom
SECTION 6   Drawing Primitives    (all teacher-authentic)
              6a  wavy_underline
              6b  organic_ellipse
              6c  natural_tick
              6d  organic_arrow
              6e  curly_brace
              6f  exponent_caret
SECTION 7   Text Rendering        (HomemadeApple font, ProximityLayout)
SECTION 8   RemarkResolver        (JSON spec → ResolvedRemark pixel coords)
SECTION 9   Page Annotator        (draw_page — composites OpenCV + PIL layers)
SECTION 10  Pipeline Orchestrator (process_pdf — top-level entry)
SECTION 11  CLI                   (argparse, --analyze-only, --scaffold)
```

---

## Data Flow

```
input.pdf  +  remarks.json
       │
       ▼
SECTION 3: _load_pdf_pages_pil()
  → list[PIL.Image]  (RGB, at target DPI)
       │
       ▼
SECTION 4: build_layout_map(gray, page_no, dpi)
  → LayoutMap  (built ONCE per page, passed everywhere)
  Contains:
    line_ys          — pixel y of each text baseline
    inter_line_gaps  — blank vertical bands between lines
    left_strips      — free bands in left margin
    right_strips     — free bands in right margin
    word_gaps        — inter-word x-positions per line
    free_rects       — large 2D blank areas
    writable_regions — connected blobs of writable cells
       │
       ▼
SECTION 8: RemarkResolver(layout, gray).resolve_all(raw_remarks)
  → list[ResolvedRemark]  (all coords in pixel space, sorted by priority)
       │
       ▼
SECTION 9: draw_page(pil_img, resolved_remarks, layout, dpi)
  → PIL.Image  (annotated RGB)
  Two layers:
    arr      — OpenCV BGR array  (underlines, circles, arrows, ticks, braces)
    text_l   — PIL RGBA layer    (HomemadeApple remark text)
  alpha_composite(arr_as_RGBA, text_l) → final RGB
       │
       ▼
SECTION 10: img2pdf.convert(png_paths) → output.pdf
```

---

## LayoutMap Reference

```python
@dataclass
class LayoutMap:
    page: int
    W: int; H: int; dpi: int
    scale: float                    # dpi / 150.0

    line_ys: list[int]              # pixel y, 0-indexed internally
    line_half_height: int           # LH = max(12, H // 55)

    inter_line_gaps: list[GapInfo]  # blank bands between lines
    left_strips: list[FreeStrip]    # free bands in left margin
    right_strips: list[FreeStrip]   # free bands in right margin
    word_gaps: list[WordGap]        # inter-word gaps per line
    free_rects: list[FreeRect]      # large 2D blank areas
    writable_regions: list[WritableRegion]

    lmx: int   # left margin boundary  = int(W * 0.135)
    rmx: int   # right margin boundary = int(W * 0.875)

    # Key methods
    def line_y(self, n: int) -> int          # safe 1-indexed lookup
    def best_gap_for_line(self, line, prefer="below") -> GapInfo | None
    def word_gaps_on_line(self, line: int) -> list[WordGap]
    def best_score_position(self) -> tuple[float, int]  # (cx_pct, cy_line)
```

**Page zones (hard-coded, derived from page width W):**

| Zone | x range | Pixel range |
|---|---|---|
| Left margin | 0 – 13.5% | `0 → lmx` |
| Body | 13.5% – 87.5% | `lmx → rmx` |
| Right margin | 87.5% – 100% | `rmx → W` |

---

## remarks.json Schema

```jsonc
{
  "pages": [
    {
      "page": 1,         // 1-indexed page number
      "remarks": [ ... ] // array of remark objects (see types below)
    }
  ]
}
```

### Coordinate System

| Field | Type | Meaning |
|---|---|---|
| `cx`, `x_start`, `x_end`, `x_pct`, `from_x` | float % or `"auto"` / `"auto_line"` / `"auto_gap"` | Percent of page width (0–100) |
| `cy`, `anchor_y` | float % | Percent of page height (0–100) |
| `line`, `cy_line`, `from_line`, `line_start`, `line_end` | int | 1-indexed detected text line number |
| `zone`, `remark_zone` | string | `"left_margin"` \| `"right_margin"` \| `"gap"` |

**Auto-resolution rules:**
- `"auto"` on `cx`/`cy_line` in `score_circle` → uses `layout.best_score_position()`
- `"auto_line"` on `x_start`/`x_end` → expands to full body width (`lmx` → `rmx`)
- `"auto_gap"` on `x_pct` in `exponent_insert` → nearest word gap to line midpoint

### Remark Types

#### `score_circle` — Score in an organic ellipse
```jsonc
{
  "type": "score_circle",
  "text": "14/20",        // supports plain numbers and fractions
  "cx": "auto",           // or explicit % e.g. 85.0
  "cy_line": "auto",      // or explicit line number e.g. 1
  "severity": "positive"
}
```

#### `tick` — Natural curved ✓ checkmark
```jsonc
{
  "type": "tick",
  "line": 3,
  "zone": "left_margin",  // or "right_margin"
  "severity": "positive"
}
```

#### `underline_remark` — Wavy underline + remark text
```jsonc
{
  "type": "underline_remark",
  "line": 5,
  "line_end": 6,           // optional — for multi-line underline
  "x_start": "auto_line",  // or explicit % e.g. 22.5
  "x_end": "auto_line",    // or explicit % e.g. 68.0
  "x_start2": 20.0,        // x_start for line_end (multi-line only)
  "x_end2": 80.0,          // x_end for line_end (multi-line only)
  "remark": "Needs elaboration",
  "remark_zone": "gap",    // "gap" | "right_margin" | "left_margin"
  "severity": "suggestion"
}
```

#### `circle_remark` — Organic oval snapped to word bounds
```jsonc
{
  "type": "circle_remark",
  "line": 7,
  "x_start": 22.5,         // rough % — snapped to actual word boundaries
  "x_end": 48.0,
  "remark": "Incorrect term — use 'sovereignty'",
  "remark_zone": "right_margin",
  "severity": "critical"
}
```

#### `exponent_insert` — Caret ^ for missing word
```jsonc
{
  "type": "exponent_insert",
  "line": 9,
  "x_pct": "auto_gap",    // or explicit % snapped to nearest word gap
  "text": "not",           // word(s) to insert above caret
  "severity": "critical"
}
```

#### `arrow_remark` — Bezier arrow pointing to issue
```jsonc
{
  "type": "arrow_remark",
  "from_line": 11,
  "from_x": 35.0,
  "remark": "Contradicts line 5 argument",
  "remark_zone": "right_margin",
  "severity": "critical"
}
```

#### `curly_brace` — Spans a paragraph
```jsonc
{
  "type": "curly_brace",
  "line_start": 13,
  "line_end": 16,
  "zone": "left_margin",
  "remark": "Entire para off-topic",
  "severity": "critical"
}
```

#### `text` — Plain margin note
```jsonc
{
  "type": "text",
  "line": 18,
  "zone": "right_margin",
  "text": "See model answer\npara 3",
  "severity": "suggestion"
}
```

**All types accept:** `"severity": "positive"` (green) | `"suggestion"` (soft red) | `"critical"` (red)

---

## Remark Placement Decision Tree

```
1. Determine line number(s)
     → use detect_lines() output (1-indexed)
     → verify against KNOWN_LAYOUTS for pdf1/pdf2

2. Choose x coordinates
     → "auto_line"  : full body width (underlines spanning whole line)
     → explicit %   : use word_gap_by_line from layout for precision
     → "auto_gap"   : let engine snap to nearest word gap (exponent only)

3. Choose remark_zone
     "gap"          needs: gap with after_line within ±5 of annotation line
                           gap height ≥ S_REM + 10 px (S_REM ≈ H // 65)
                           → text placed vertically centred in gap
                           → arrow drawn from annotation to text
     "right_margin" needs: free strip near that line's y%
                           → ProximityLayout auto-slots, no overlap
     "left_margin"  needs: free strip near that line's y%
                           → ProximityLayout auto-slots, no overlap
     fallback rule  : if gap requested but none found → auto-falls back
                      to "right_margin" (RemarkResolver handles this)

4. score_circle
     → always use "auto" unless you need a specific position
     → engine picks largest free rect in top-right quadrant (x>60%, y<20%)

5. tick
     → line number of the correct paragraph's first or last line
     → zone: "left_margin" (default) or "right_margin"

6. curly_brace
     → spans line_start to line_end
     → pick the margin (left/right) with the most free strips in that y-range
```

---

## Draw Priority Order

Remarks are drawn in this order regardless of JSON order.
**Do not change this — it determines correct visual layering.**

```
0  score_circle      (always topmost, drawn first on OpenCV layer)
1  tick              (before any underlines)
2  curly_brace       (before the lines it brackets)
3  underline_remark  (before circles — circles draw over underlines)
4  circle_remark     (over underlines)
5  exponent_insert   (above all line content)
6  arrow_remark      (connects marks — drawn after all marks exist)
7  text              (margin text drawn absolutely last)
```

---

## Drawing Primitives Reference

### `wavy_underline(arr, gray, line_y, lh, x1, x2, color, thickness, dpi)`
- Extracts per-column ink-bottom profile from the line band
- Fills gaps (word spaces) with forward+backward interpolation
- Applies 82nd-percentile sliding window (ignores descenders: g, p, y, q)
- Heavy Gaussian smooth (σ ∝ dpi) → gentle flowing wave
- Enforces: underline never goes above actual ink bottom
- Draws as OpenCV polyline (subsampled every 4px)
- **Band ceiling fix:** `y_bot` clamped to `next_line_y - lh - 4` to prevent bleed

### `organic_ellipse(arr, cx, cy, rx, ry, color, thickness, dpi)`
- Three Fourier harmonics on the radius: h1=sin(5t), h2=sin(11t), h3=sin(17t)
- Amplitude ≈ 4.5–6.5% wobble, random per harmonic
- Random tilt ±8°
- Start angle slightly off 0°; total angle = 2π + 0.05–0.18 rad overrun
- `_seed(cx,cy)` ensures same position always gets same wobble

### `organic_ellipse_fitted(arr, gray, line_y, lh, x1, x2, color, thickness, dpi, layout)`
- Wrapper used by `circle_remark`
- Calls `snap_to_word_boundaries` + `find_ink_top` + `find_ink_bottom`
- Auto-computes `rx`, `ry` from actual ink bounding box + padding
- Returns `(cx, cy, rx, ry)` for arrow attachment

### `natural_tick(arr, cx, cy, size, color, thickness)`
- Two quadratic Bezier strokes: short foot + long ascending arm
- Random jitter ±2–3px on control points per `_seed(cx,cy)`
- Not a V-shape — curves organically like a pen stroke

### `organic_arrow(arr, sx, sy, dx, dy, color, th, free_bands, H)`
- Quadratic Bezier, control point computed from direction:
  - Right-going → bow upward
  - Down-going → bow outward
  - Other → perpendicular bow
- Control point snapped to centre of a free band when available
- Arrowhead: two lines at ±0.45 rad from final tangent direction

### `curly_brace(arr, y_top, y_bot, bx, side, color, LN_TH, TIP_W)`
- Cubic Bezier (upper half) + cubic Bezier (lower half)
- Both halves share the tip point at horizontal protrusion `TIP_W`
- `side="left"` → tip points right; `side="right"` → tip points left

### `exponent_caret(arr, cx, caret_y, s, color, LN_TH)`
- Two `cv2.line` calls forming a ^ shape
- Text rendered above via `paste_hw()` on text layer

---

## ProximityLayout

Manages non-overlapping placement of remark text in margins.

```python
ProximityLayout(layout: LayoutMap)
  # Pre-seeded with occupied margin strips (inverted from free strips)
  # so text never lands on top of existing content

.find_y(zone, preferred_y, height, margin=6) → int
  # Tries preferred_y first; if overlapping, searches outward in steps

.render(layer, text_lines, zone, preferred_y, font_size, color) → (x, y, total_h)
  # Word-wraps to margin width
  # Calls find_y for non-overlapping slot
  # Pastes each line with paste_hw()
```

**Gap placement alternative:** `place_in_gap(layer, text_lines, gap, body_l, font_size, color)`
- Used when `remark_zone == "gap"` and gap is tall enough
- Centres text block vertically within the gap

---

## Free Space Detection — Three CV Techniques

All three run inside `build_layout_map()` and populate `LayoutMap`.

### Technique 1 — Morphological Dilation (Strips)
- Ink mask → horizontal dilation (merges words) → vertical dilation (merges lines)
- Per-row occupancy check in left margin, right margin, body zones
- Free runs ≥ min_h px → `FreeStrip` objects
- Inter-line gaps tagged with `after_line` / `before_line` numbers

### Technique 2 — Column Projection (Word Gaps)
- Per detected line: extract band ± LH, threshold, horizontal dilation
- Column-wise sum → projection profile
- Ink→gap→ink transitions → `WordGap` objects with exact pixel x
- Refined using raw (un-dilated) projection for precise ink edge detection

### Technique 3 — Canny + Contours (Free Rects)
- Canny edge detection → dilate edges (clearance buffer ≈ 1.5mm)
- Union with full ink mask
- Invert → free-space mask
- Morphological OPEN → remove tiny free pockets
- `cv2.findContours` → bounding rects
- Filter: area ≥ min_area_px AND fill_ratio ≥ 0.35 (no slivers)
- Result: `FreeRect` list, sorted by area descending

---

## Known Layouts — Ground Truth for Regression

### test_original_pdf1.pdf (250 DPI, W=2068, H=2924)

**Page 1** — 22 lines
```
Gaps:
  after_line=13  y=54.5–59.8%  height=154px
  after_line=21  y=92.7–95.9%  height=92px
Right margin strips (y%):
  0.0–3.5,  14.9–18.1,  23.0–25.6,  26.4–29.8
Score position: cx≈73%  cy_line=1
```

**Page 2** — 19 lines
```
Gaps:
  after_line=3   y=20.2–25.0%  height=140px
  after_line=17  y=79.5–83.6%  height=118px
  after_line=18  y=87.5–95.9%  height=245px
Left margin strips (y%):
  16.0–20.2,  21.8–28.7,  31.3–34.0,  34.6–38.5,  95.9–100.0
Score position: cx≈88%  cy_line=1
```

### test_original_pdf2.pdf (250 DPI)

**Page 1** — 18 lines
```
Gaps:
  after_line=4   y=22.6–26.3%  height=107px
  after_line=6   y=36.1–40.8%  height=138px
  after_line=9   y=57.6–60.7%  height=92px
Right margin: no free strips (dense content)
```

**Page 2** — 20 lines
```
Gaps:
  after_line=19  y=90.6–97.5%  height=201px  (only gap, very bottom)
Left margin strips (y%):
  28.3–30.6,  32.6–35.0,  36.8–40.6,  47.0–50.6,
  57.1–59.4,  61.5–63.5,  65.6–69.1
Score position: cx≈64%  cy_line=1
```

---

## Gap Placement Rules (Critical)

```
remark_zone = "gap"
  → calls find_gap_near(line, gaps, prefer='below', max_dist=5)
  → looks for gap where: line ≤ gap.after_line ≤ line + 5
  → gap.height_px must be ≥ S_REM + 10   (S_REM = max(26, H // 65))
  → at 250 DPI on pdf1 (H=2924): S_REM ≈ 44px → min gap ≈ 54px
  → fallback: if no valid gap found → auto-switch to "right_margin"

Gap text placement:
  → vertically centred within the gap
  → horizontally: body_l + 30px (slight indent from body left)
  → arrow drawn from annotation anchor → gap text block

Margin text placement:
  → ProximityLayout.find_y() searches outward from preferred_y
  → Step size: max(4, font_height // 6)
  → Never overlaps existing slots (pre-seeded from occupied strips)
```

---

## Regression Test Commands

```bash
# Fast — run after every edit
python3 teacher_annotate.py test_original_pdf1.pdf test_remarks_pdf1.json \
    annotated_pdf1_test.pdf --dpi 250

python3 teacher_annotate.py test_original_pdf2.pdf test_remarks_pdf2.json \
    annotated_pdf2_test.pdf --dpi 250

# Layout regression only (no output PDF)
python3 test_teacher_annotate.py

# Stress test (slow — 52 pages)
python3 teacher_annotate.py test_pdf3.pdf test_remarks.json \
    test_pdf3_annotated.pdf --dpi 250

# Full debug output
python3 teacher_annotate.py test_original_pdf1.pdf test_remarks_pdf1.json \
    annotated_pdf1_test.pdf --dpi 250 --debug-layout --debug-freespace
```

---

## Regression Test Assertions (`test_teacher_annotate.py`)

```python
KNOWN_LAYOUTS = {
    "test_original_pdf1.pdf": {
        1: {"lines": 22, "gaps": [(13, 154), (21, 92)]},
        2: {"lines": 19, "gaps": [(3, 140), (17, 118), (18, 245)]},
    },
    "test_original_pdf2.pdf": {
        1: {"lines": 18, "gaps": [(4, 107), (6, 138), (9, 92)]},
        2: {"lines": 20, "gaps": [(19, 201)]},
    },
}
# Tolerance: line count ±0, gap height ±15%
# Output file: must exist and be > 100KB
```

---

## Implementation Sprints

```
Sprint 1 — Foundation (test: layout detection matches KNOWN_LAYOUTS)
  • PDF ingestion + fitz fallback
  • build_layout_map() — lines, gaps, free strips
  • LayoutMap dataclass + all helper methods
  • Basic CLI (annotate + --analyze-only)

Sprint 2 — Snapping (test: circles/underlines land on words, not spaces)
  • Technique 2: word gap detection in build_layout_map
  • snap_to_word_boundaries()
  • snap_exponent_gap()
  • find_ink_top() / find_ink_bottom()
  • RemarkResolver: score, tick, underline, circle, exponent

Sprint 3 — Drawing (test: annotate pdf1 + pdf2, visual inspection)
  • All 6 primitives ported from annotate_pdf_final + fixes:
      - wavy_underline band ceiling fix
      - organic_ellipse_fitted wrapper
      - organic_arrow routed through free bands
  • ProximityLayout with pre-seeded occupied strips
  • draw_page() full implementation
  • Gap text placement + arrow routing

Sprint 4 — Polish (test: full regression suite green on pdf1, pdf2, pdf3)
  • "auto" / "auto_line" / "auto_gap" coordinate resolution
  • Fraction scores ("14/20") in score_circle
  • Multi-line underline (line + line_end + x_start2/x_end2)
  • --scaffold CLI flag (generates remarks template from layout)
  • --debug-freespace overlay PDF
  • Technique 3 free rects in build_layout_map
  • WritableRegion connected components
```

---

## Critical Bug Fixes Applied (vs. annotate_pdf_final.py)

| Bug | Old Behaviour | Fixed Behaviour |
|---|---|---|
| Underline bleeds into next line | `y_bot = line_y + lh + lh*0.15` | `y_bot` clamped to `next_line_y - lh - 4` |
| Score formula instability | `mean * (1 - std)` collapses at high std | `clip(mean - std, 0, 1)` — stable |
| Grid left-aligned | Remainder piled on left edge | Centred: `left_margin = remainder / 2` |
| Margin text hits existing content | Slots not pre-seeded | ProximityLayout pre-seeded from occupied strips |
| Arrow ignores free bands | Control point = geometric midpoint | Control point snapped to centre of free band |
| Empty slice penalised | `ink_ratio = 1.0` for empty patch | `return 0.0, 0.0` — not writable but not penalised |

---

## Dependencies

```
pip install pymupdf pdf2image img2pdf Pillow opencv-python numpy scipy
```

Font (mandatory):
```
fonts/HomemadeApple-Regular.ttf
→ https://fonts.google.com/specimen/Homemade+Apple
```

Optional (for pdf2image):
```
poppler-utils   # apt install poppler-utils  (Ubuntu)
                # brew install poppler       (macOS)
# If missing, _load_pdf_pages_pil() auto-falls back to PyMuPDF
```

---

## Environment

- Python 3.10+
- DPI default: 250 (fast + accurate; use 300 for high-res output)
- At 250 DPI: 1 PDF point = 250/72 ≈ 3.47 pixels
- At 250 DPI, A4 page (595×842 pts) → ≈ 2068×2924 px
- `scale = dpi / 150.0` used throughout for DPI-relative sizing
- `_seed(cx, cy)` → `random.seed(int(cx)*31337 + int(cy))` — deterministic per annotation position