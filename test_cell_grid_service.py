from pathlib import Path

import pytest

from cell_grid_service import (
    CELL_SIZE_PTS,
    analyze_pdf_cell_grid,
    cell_id_from_rc,
    draw_cell_grid_overlay,
    find_writable_runs,
    rc_from_cell_id,
    _cell_metrics,
)


PDF_PATH = Path("test_original_pdf2.pdf")


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return PDF_PATH.read_bytes()


@pytest.fixture(scope="module")
def page_grids(pdf_bytes):
    return analyze_pdf_cell_grid(pdf_bytes, cell_size_pts=CELL_SIZE_PTS)


class TestCellId:
    def test_cell_id_basic(self):
        assert cell_id_from_rc(1, 1) == "A1"
        assert cell_id_from_rc(1, 26) == "Z1"
        assert cell_id_from_rc(1, 27) == "AA1"
        assert cell_id_from_rc(10, 28) == "AB10"

    def test_roundtrip(self):
        cases = ["A1", "Z1", "AA1", "AB10", "AZ99", "BA7"]
        for cid in cases:
            r, c = rc_from_cell_id(cid)
            assert cell_id_from_rc(r, c) == cid


class TestGridShape:
    def test_returns_all_pages(self, page_grids):
        assert len(page_grids) >= 1

    def test_cell_count_matches_rows_cols(self, page_grids):
        for g in page_grids:
            assert len(g.cells) == g.rows * g.cols

    def test_margins_non_negative(self, page_grids):
        for g in page_grids:
            assert g.left_margin_pts >= 0
            assert g.top_margin_pts >= 0


class TestCellRanges:
    def test_percent_ranges(self, page_grids):
        for g in page_grids:
            for c in g.cells:
                assert 0 <= c.x_start_percent <= c.x_end_percent <= 100
                assert 0 <= c.y_start_percent <= c.y_end_percent <= 100

    def test_pdf_ranges_ordered(self, page_grids):
        for g in page_grids:
            for c in g.cells:
                assert c.pdf_x1 < c.pdf_x2
                assert c.pdf_y1 < c.pdf_y2

    def test_a1_is_near_top_in_pymupdf_coords(self, page_grids):
        g = page_grids[0]
        a1 = next(c for c in g.cells if c.cell_id == "A1")
        assert a1.pdf_y1 >= 0.0
        assert a1.pdf_y1 < g.cell_size_pts + g.top_margin_pts + 0.1
        assert a1.pdf_y2 <= g.top_margin_pts + g.cell_size_pts + 0.1

    def test_writable_non_empty(self, page_grids):
        writable = [c for g in page_grids for c in g.cells if c.writable]
        assert len(writable) > 0

    def test_ink_ratio_present_and_range(self, page_grids):
        for g in page_grids:
            for c in g.cells:
                assert hasattr(c, "ink_ratio")
                assert 0.0 <= c.ink_ratio <= 1.0


class TestRuns:
    def test_runs_are_contiguous_and_writable(self, page_grids):
        for g in page_grids:
            runs = find_writable_runs(g, min_length=2)
            for run in runs:
                assert all(cell.writable for cell in run)
                for a, b in zip(run, run[1:]):
                    assert a.row == b.row
                    assert b.col == a.col + 1

    def test_runs_sorted_by_length_desc(self, page_grids):
        runs = find_writable_runs(page_grids[0], min_length=1)
        lengths = [len(r) for r in runs]
        assert lengths == sorted(lengths, reverse=True)


class TestOverlay:
    def test_overlay_pdf_created(self, pdf_bytes, page_grids, tmp_path):
        out = tmp_path / "grid_overlay.pdf"
        result = draw_cell_grid_overlay(pdf_bytes, page_grids, out)
        assert result.exists()
        assert result.stat().st_size > 0


class TestInkRegression:
    def test_dense_ink_cell_is_not_writable_under_strict_rules(self):
        import numpy as np

        dense = np.full((50, 50), 0.2, dtype=np.float32)
        score, ink_ratio = _cell_metrics(dense, ink_brightness_max=0.45)
        writable = (score >= 0.85) and (ink_ratio <= 0.03)
        assert writable is False
