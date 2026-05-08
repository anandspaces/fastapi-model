# Cell Grid v4 — Internal & Gemini-Facing Reference

`cell_grid_service_v4.py` is the precise writable-space analyser that powers
annotation placement in the smart-OCR pipeline. This doc covers what the
algorithm produces, the cell-ID coordinate system, and how to drive it from
Gemini.

For the **frontend-facing** contract (response shape, JS resolution
helpers, rendering recipes) read `FRONTEND_ANNOTATION_GUIDE.md`.

---

## 1. What v4 produces

Per page:

| Field | Meaning |
|---|---|
| `rows`, `cols` | Grid size. A4 @ 12 pt cells → 70 × 49. |
| `cell_size_pts` | Square cell side in PDF points. Default `12.0` (~4.2 mm). |
| `left_margin_pts`, `top_margin_pts` | Centred slack so cells align with content. |
| `page_w_pts`, `page_h_pts` | Original page dimensions. |
| `cells[]` | Every cell with `cell_id`, `score`, `ink_ratio`, `writable` bool, percent + PDF rect. |
| `runs[]` | Contiguous horizontal runs of writable cells per row (`cell_count ≥ 2`). |
| `regions[]` | Up to 32 maximal writable rectangles, sorted by area desc. Each has `region_id` (R1, R2, …), `bbox_range_id` (e.g. `D5:K9`), and full `cell_ids[]`. |

> **Why both runs and regions?** Runs are 1-D and best for in-line marks
> (underlines, exponent inserts). Regions are 2-D and best for block
> placement (multi-line comments, score boxes). The placement layer
> (`assign_cell_ids_v4`) tries regions first, falls back to runs.

---

## 2. Cell-ID grammar

Excel-style. Columns: `A..Z, AA..AZ, BA..BZ, ...`. Rows: 1-indexed integer.

* `A1` — top-left cell.
* `AC12` — col 29, row 12.
* `AW70` — col 49, row 70 (typical A4 bottom-right at 12 pt).
* Range: `start:end` — `C12:H12` (single row), `D5:K9` (rectangle). Ranges
  always have `start_row ≤ end_row` and `start_col ≤ end_col`.

Helpers in `cell_grid_service_v4.py`:

```python
cell_id_from_rc(row, col) -> str
rc_from_cell_id(cell_id)  -> (row, col)
range_id_from_cells(start_id, end_id) -> str
```

---

## 3. Coordinate math (server side)

```python
x1_pts = left_margin_pts + (col - 1) * cell_size_pts
y1_pts = top_margin_pts  + (row - 1) * cell_size_pts
x2_pts = x1_pts + cell_size_pts
y2_pts = y1_pts + cell_size_pts

x1_percent = (x1_pts / page_w_pts) * 100   # … and so on
```

The same formulas live in `FRONTEND_ANNOTATION_GUIDE.md` as JS so the
client resolves cells without an extra round-trip.

---

## 4. Algorithm pipeline (per page)

```
PDF page
  → rasterise at 300 DPI (PyMuPDF)            → gray float32 [0,1]
  → ink mask: gray < 0.45                     → binary
  → close 3×3 (1 iter)                        → fill single-pixel specks
  → erode 3×3 (1 iter)                        → shrink ink edge by ~1 px
  → integral images (gray, gray², ink)
  → vectorised per-cell metrics:
        score      = clip(mean - std, 0, 1)
        ink_ratio  = mean(ink_mask) over cell
  → writable mask = (score ≥ 0.70) AND (ink_ratio ≤ 0.04)
  → row runs: contiguous writable spans, ≥ 2 cells
  → regions: maximal all-writable rectangles via histogram DP
              (cap 32, sorted by area, dropped if strictly contained)
```

### Why these defaults

| Knob | Value | Why |
|---|---|---|
| `cell_size_pts` | 12 (~4.2 mm) | Fine enough for handwriting placement; ~3500 cells/A4 keeps response size sane. |
| `dpi` | 300 | Picks up thin rules + faint ink. |
| `score = clip(mean - std, 0, 1)` | — | Stable on edge cells; replaces v1's `mean*(1-std)` which collapses at high std. (See `CLAUDE.md` Critical Bug Fixes.) |
| `min_score` | 0.70 | Relaxed (vs v3's 0.85) because new score formula has tighter spread. |
| `max_ink_ratio` | 0.04 | Allows occasional faint stray dots. Morph-close already removed specks. |
| `min_run_length` | 2 | Singletons rarely useful for placement. |
| `min_region_cells` | 2; `min_dim` 2 | Excludes 1-cell or 1×N strips that are not really placement zones. |
| `max_regions` | 32/page | Covers typical density without ballooning the response. |

### Margin centring

```python
left_margin = (page_w − cols·cell_size) / 2
top_margin  = (page_h − rows·cell_size) / 2
```

Both >= 0; cells are symmetric around the page midline.

---

## 5. CLI

```bash
# Basic — write overlay PDF, summary to stderr
python3 cell_grid_service_v4.py test_original_pdf1.pdf overlay_pdf1_v4.pdf

# With JSON dump to stdout (no per-cell list)
python3 cell_grid_service_v4.py test.pdf out.pdf --json > grid.json

# Include per-cell list (verbose)
python3 cell_grid_service_v4.py test.pdf out.pdf --json --include-cells > full.json

# Tune grid
python3 cell_grid_service_v4.py test.pdf out.pdf --cell-size-pts 10 --dpi 300

# Hide layers in the overlay
python3 cell_grid_service_v4.py test.pdf out.pdf --no-full-grid --no-axis-labels
```

The overlay PDF has these visual layers (bottom → top):

| Layer | Style |
|---|---|
| Grid lines | Faded gray 0.25-pt strokes — every cell border. |
| Writable cells | Translucent green fill + thin green stroke. |
| Region rectangles | Blue 1.3-pt outline + small `Rk D5:K9` label above. |
| Axis labels | Column letters above row 1; row numbers left of col 1. |

Cell-by-cell labels are off by default (slow on dense grids). Pass
`--cell-labels` for a debug overlay.

---

## 6. Driving Gemini

The flow we want Gemini to perform on the overlay PDF:

1. Receive the overlay PDF (`draw_cell_grid_overlay_v4` output).
2. For each annotation it wants to add (underline / circle / margin remark
   / score), pick a cell ID or range ID that lives in a writable area —
   green fill or blue-bordered region.
3. Return JSON like:

```jsonc
{
  "page": 2,
  "annotations": [
    { "id": "underline_line5",     "type": "underline_remark",
      "range_id": "C12:H12",
      "remark_range_id": "AT12:AW14",
      "comment": "Needs elaboration", "is_positive": false },
    { "id": "score_q1", "type": "score_circle",
      "range_id": "AT3:AW6", "value": "14/20" }
  ]
}
```

Backend then converts those into pixel coordinates via the meta block — same
math used by the frontend. No extra round-trip.

### Gemini system-prompt scaffold

> You will receive a PDF where every page has been annotated with a faint
> grid. Each grid cell has an Excel-style ID (column letters A..AW, row
> numbers 1..70). Column letters are labelled above the top row and row
> numbers along the left edge. Cells with translucent green fill are
> WRITABLE (free of student ink). Blue-bordered rectangles labelled `Rn`
> are coherent writable regions — prefer these for multi-line comments.
>
> For each annotation you produce, reply with a `range_id` of the form
> `<startCellId>:<endCellId>` (or just `<cellId>` for a single cell)
> entirely contained in writable cells. Do not place any annotation in a
> non-writable (ungreen) cell. Reply only in JSON.

---

## 7. Performance

Measured on this repo:

| Fixture | Pages | Time |
|---|---|---|
| `test_original_pdf1.pdf` | 2 | ~10 s |
| `test_original_pdf2.pdf` | 2 | ~8 s |
| `test_pdf3.pdf` | 52 | ~3 min |

Bottlenecks: image rasterisation (PyMuPDF) and the maximal-rectangle
histogram DP. Per-cell metrics are vectorised (integral images).

If you need it faster:

* Lower `dpi` to 200 (still fine at 12-pt cells; metrics survive).
* Increase `cell_size_pts` to 15 (≈1.5× speed-up at the cost of fidelity).
* Cap `max_regions` lower if regions are not needed downstream.

---

## 8. Tests

`test_cell_grid_service_v4.py` covers:

* Cell-ID round-trips (Z, AA, AC, AW boundaries).
* Score formula stability under high variance.
* Ink-mask morphology (specks closed, blocks survive).
* Grid shape on `test_original_pdf{1,2}.pdf`.
* Centred-margin invariant: `|left_margin − slack/2| < 0.05`.
* Run invariants: contiguous, ≥ 2 cells, IDs match row.
* Region invariants: every cell in `cell_ids[]` is writable, sorted by area.
* `bbox_range_id` matches `(start_corner):(end_corner)`.
* Dense-ink regression: full-black page → 0 writable cells.
* Frontend-resolution invariant: cell-id → percent matches stored
  `Cell.x*_percent` to <0.05 % drift.

```bash
python3 -m pytest test_cell_grid_service_v4.py -v
```

---

## 9. Files

| File | Role |
|---|---|
| `cell_grid_service_v4.py` | Algorithm + CLI + dataclasses. |
| `src/cell_grid_service.py` | FastAPI-facing wrapper (`build_cell_grid`, `cell_grid_meta_payload`, `build_overlay_pdf`). |
| `remark_cell_layout_service.py` | `assign_cell_ids_v4(items, grids)` — Tier 1 region / Tier 2 run / Tier 3 synth. |
| `test_cell_grid_service_v4.py` | Regression suite. |
| `cell_grid_service.py`, `_v3.py` | Legacy (v1, v3). Kept until v4 settles in production. |

---

## 10. Glossary

* **Writable cell** — gray score ≥ 0.70 AND ink_ratio ≤ 0.04 in the
  morphology-cleaned ink mask.
* **Run** — 1-D horizontal contiguous span of writable cells in one row.
* **Region** — 2-D maximal axis-aligned rectangle whose every cell is
  writable.
* **Range ID** — `start:end` form (e.g. `D5:K9`); `start` may equal `end`.
* **Placement tier** — region (1) > run (2) > synth (3); see
  `assign_cell_ids_v4`.
* **Meta block** — minimal per-page payload (`rows`, `cols`,
  `cell_size_pts`, margins, page dims) the frontend uses to resolve any
  cell ID into a percent rectangle.
