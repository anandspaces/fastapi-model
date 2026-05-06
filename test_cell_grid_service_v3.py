from pathlib import Path

import pytest

from cell_grid_service_v3 import (
    analyze_pdf_writable_runs,
    draw_writable_runs_overlay,
    range_id_from_cells,
)


PDF_PATH = Path("test_original_pdf2.pdf")


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return PDF_PATH.read_bytes()


class TestRangeId:
    def test_single_cell(self):
        assert range_id_from_cells("A1", "A1") == "A1"

    def test_multi_cell(self):
        assert range_id_from_cells("A1", "D1") == "A1:D1"


class TestWritableRunsV3:
    def test_returns_pages_and_runs(self, pdf_bytes):
        pages = analyze_pdf_writable_runs(pdf_bytes)
        assert len(pages) >= 1
        assert any(len(p.runs) > 0 for p in pages)

    def test_all_runs_have_neighbors(self, pdf_bytes):
        pages = analyze_pdf_writable_runs(pdf_bytes)
        for page in pages:
            for run in page.runs:
                assert run.cell_count >= 2
                assert run.start_col < run.end_col
                assert run.start_cell_id.endswith(str(run.row))
                assert run.end_cell_id.endswith(str(run.row))

    def test_percent_and_pdf_ranges(self, pdf_bytes):
        pages = analyze_pdf_writable_runs(pdf_bytes)
        for page in pages:
            for run in page.runs:
                assert 0 <= run.x_start_percent <= run.x_end_percent <= 100
                assert 0 <= run.y_start_percent <= run.y_end_percent <= 100
                assert run.pdf_x1 < run.pdf_x2
                assert run.pdf_y1 < run.pdf_y2

    def test_min_run_length_one_keeps_singletons(self, pdf_bytes):
        strict = analyze_pdf_writable_runs(pdf_bytes, min_run_length=2)
        loose = analyze_pdf_writable_runs(pdf_bytes, min_run_length=1)
        strict_total = sum(len(p.runs) for p in strict)
        loose_total = sum(len(p.runs) for p in loose)
        assert loose_total >= strict_total

    def test_dense_ink_has_zero_runs(self):
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(0, 0, 595, 842), color=(0, 0, 0), fill=(0, 0, 0))
        raw = doc.tobytes()
        doc.close()

        pages = analyze_pdf_writable_runs(raw)
        assert len(pages) == 1
        assert len(pages[0].runs) == 0


class TestOverlayV3:
    def test_overlay_written(self, pdf_bytes, tmp_path):
        pages = analyze_pdf_writable_runs(pdf_bytes)
        out = tmp_path / "v3_overlay.pdf"
        result = draw_writable_runs_overlay(pdf_bytes, pages, out, show_labels=True)
        assert result.exists()
        assert result.stat().st_size > 0
