"""Tests for cell_overlay_renderer.

Coverage:
  - Smoke (PNGs produced, count matches page count, bytes are valid PNG)
  - Perf budget at the two cell densities we ship
  - Adaptive labelling threshold (per-cell labels off on dense grids)
  - Round-trip stability (idempotent on the input PDF)
"""

from __future__ import annotations

import io
import struct
import time
from pathlib import Path

import pytest

from cell_grid_service_v4 import analyze_pdf_cell_grid_v4
from cell_overlay_renderer import (
    MIN_CELL_LABEL_SIZE_PTS,
    _label_font_size_pts,
    _should_label_each_cell,
    render_overlay_pngs,
)

PDF_FIXTURE = Path("test_pdf4.pdf")
FONT_FIXTURE = Path("fonts/HomemadeApple-Regular.ttf")

# Per-page CI budgets (ms). These guard against regression, not against any
# specific architectural target — the design doc commits to ~600 ms/page on
# the dev machine for 24-pt cells; CI runs with shared CPU + warm caches from
# preceding tests, so the ceiling here is wider. Standalone benchmarks
# (cell_overlay_renderer.py CLI) report the actual perf.
PERF_BUDGET_MS_12PT = 900.0
PERF_BUDGET_MS_24PT = 750.0


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    if not PDF_FIXTURE.exists():
        pytest.skip(f"missing fixture {PDF_FIXTURE}")
    return PDF_FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def grids_24pt(pdf_bytes):
    return analyze_pdf_cell_grid_v4(pdf_bytes, cell_size_pts=24.0)


@pytest.fixture(scope="module")
def grids_12pt(pdf_bytes):
    return analyze_pdf_cell_grid_v4(pdf_bytes, cell_size_pts=12.0)


def _looks_like_png(buf: bytes) -> bool:
    return len(buf) > 24 and buf[:8] == b"\x89PNG\r\n\x1a\n"


def _looks_like_jpeg(buf: bytes) -> bool:
    return len(buf) > 4 and buf[:3] == b"\xff\xd8\xff"


def test_label_size_threshold():
    # 12-pt cells → 3.6pt label → below threshold → no per-cell labels
    assert _label_font_size_pts(12.0) < MIN_CELL_LABEL_SIZE_PTS
    assert not _should_label_each_cell(12.0)
    # 18-pt cells → 5.4pt label → above threshold → per-cell labels on
    assert _label_font_size_pts(18.0) >= MIN_CELL_LABEL_SIZE_PTS
    assert _should_label_each_cell(18.0)
    # 24-pt cells → 6pt label (capped) → on
    assert _should_label_each_cell(24.0)


def test_smoke_24pt(pdf_bytes, grids_24pt):
    pngs = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=200)
    assert len(pngs) == len(grids_24pt)
    for p in pngs:
        assert _looks_like_png(p)


def test_smoke_12pt(pdf_bytes, grids_12pt):
    pngs = render_overlay_pngs(pdf_bytes, grids_12pt, dpi=200)
    assert len(pngs) == len(grids_12pt)
    for p in pngs:
        assert _looks_like_png(p)


def test_perf_budget_24pt(pdf_bytes, grids_24pt):
    font_path = str(FONT_FIXTURE) if FONT_FIXTURE.exists() else None
    t0 = time.perf_counter()
    pngs = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=200, label_font_path=font_path)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    per_page = elapsed_ms / max(1, len(pngs))
    assert per_page < PERF_BUDGET_MS_24PT, (
        f"24-pt cells: {per_page:.0f} ms/page exceeds budget {PERF_BUDGET_MS_24PT:.0f}"
    )


def test_perf_budget_12pt(pdf_bytes, grids_12pt):
    font_path = str(FONT_FIXTURE) if FONT_FIXTURE.exists() else None
    t0 = time.perf_counter()
    pngs = render_overlay_pngs(pdf_bytes, grids_12pt, dpi=200, label_font_path=font_path)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    per_page = elapsed_ms / max(1, len(pngs))
    assert per_page < PERF_BUDGET_MS_12PT, (
        f"12-pt cells: {per_page:.0f} ms/page exceeds budget {PERF_BUDGET_MS_12PT:.0f}"
    )


def test_payload_size_reasonable(pdf_bytes, grids_12pt):
    """At 200 DPI on an 18-page PDF the total payload should comfortably fit
    inside Gemini's request limit; we cap at 30 MB as a sanity ceiling."""
    pngs = render_overlay_pngs(pdf_bytes, grids_12pt, dpi=200)
    total_mb = sum(len(p) for p in pngs) / (1024 * 1024)
    assert total_mb < 30.0, f"overlay payload {total_mb:.1f} MB > 30 MB"


def test_idempotent(pdf_bytes, grids_24pt):
    """Two renders on the same input produce byte-identical PNGs."""
    a = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=200)
    b = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=200)
    assert a == b


# ── label_every_cell flag ─────────────────────────────────────────────────


def test_label_every_cell_default_off(pdf_bytes, grids_24pt):
    """Default render and label_every_cell=False produce the same bytes."""
    default = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150)
    explicit = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                                    label_every_cell=False)
    assert default == explicit


def test_label_every_cell_grows_payload(pdf_bytes, grids_24pt):
    """Labelling non-writable cells too should produce visibly larger PNGs
    (more glyphs in the raster). At 24-pt cells the increase is modest but
    measurable on dense pages."""
    base = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150)
    full = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                                label_every_cell=True)
    base_total = sum(len(p) for p in base)
    full_total = sum(len(p) for p in full)
    assert full_total > base_total, (
        f"label_every_cell payload {full_total} not larger than base {base_total}"
    )


def test_label_every_cell_dense_grid_still_labels(pdf_bytes, grids_12pt):
    """At 12-pt cells, base render skips per-cell labels (font < 4pt).
    With label_every_cell=True, we override the threshold and label anyway —
    payload must grow vs base."""
    base = render_overlay_pngs(pdf_bytes, grids_12pt, dpi=150)
    full = render_overlay_pngs(pdf_bytes, grids_12pt, dpi=150,
                                label_every_cell=True)
    base_total = sum(len(p) for p in base)
    full_total = sum(len(p) for p in full)
    assert full_total > base_total


# ── JPEG output ────────────────────────────────────────────────────────────


def test_jpeg_format_valid_bytes(pdf_bytes, grids_24pt):
    pngs = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                                image_format="jpeg")
    assert len(pngs) == len(grids_24pt)
    for buf in pngs:
        assert _looks_like_jpeg(buf)


def test_jpeg_smaller_than_png(pdf_bytes, grids_24pt):
    """JPEG q-85 must be materially smaller than PNG on grid overlays."""
    png = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                               image_format="png")
    jpg = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                               image_format="jpeg", jpeg_quality=85)
    png_total = sum(len(b) for b in png)
    jpg_total = sum(len(b) for b in jpg)
    ratio = png_total / max(1, jpg_total)
    # Conservative ceiling — base render compresses well as PNG, so the gap
    # is narrowest in the no-label-every-cell case. The label_every_cell
    # case drops payload 3-5× (covered by the next test).
    assert ratio >= 1.5, f"JPEG/PNG ratio {ratio:.2f} — JPEG should be ≥1.5× smaller"


def test_jpeg_compression_wins_with_label_every_cell(pdf_bytes, grids_24pt):
    """The actual production use case (cell-overlay grading prompt) renders
    label_every_cell=True. JPEG should cut payload ≥3× there because every
    extra glyph blows up PNG's lossless cost."""
    png = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                               image_format="png", label_every_cell=True)
    jpg = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                               image_format="jpeg", label_every_cell=True,
                               jpeg_quality=85)
    ratio = sum(len(b) for b in png) / max(1, sum(len(b) for b in jpg))
    assert ratio >= 3.0, f"label_every_cell JPEG/PNG ratio {ratio:.2f} — expected ≥3×"


def test_jpeg_with_label_every_cell(pdf_bytes, grids_24pt):
    """Combined: label_every_cell + JPEG renders cleanly."""
    bufs = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=150,
                                image_format="jpeg", label_every_cell=True)
    for buf in bufs:
        assert _looks_like_jpeg(buf)


def test_invalid_image_format_rejected(pdf_bytes, grids_24pt):
    import pytest as _p
    with _p.raises(ValueError):
        render_overlay_pngs(pdf_bytes, grids_24pt, image_format="webp")


def test_label_every_cell_perf_24pt(pdf_bytes, grids_24pt):
    """Per-page budget under label_every_cell=True at 24-pt cells. About 840
    cells/page total → ~2× the writable-only label work, plus PNG encode.
    Allow CI variance."""
    font_path = str(FONT_FIXTURE) if FONT_FIXTURE.exists() else None
    t0 = time.perf_counter()
    pngs = render_overlay_pngs(pdf_bytes, grids_24pt, dpi=200,
                                label_font_path=font_path,
                                label_every_cell=True)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    per_page = elapsed_ms / max(1, len(pngs))
    # 24-pt every-cell at 200 DPI: standalone ~600 ms; CI ceiling 1100 ms.
    assert per_page < 1100.0, (
        f"label_every_cell 24-pt cells: {per_page:.0f} ms/page exceeds budget"
    )
