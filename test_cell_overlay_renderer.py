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
