"""Regression tests for cell_grid_service_v4 — fine-grained writable grid.

Run:
    python3 -m pytest test_cell_grid_service_v4.py -v
"""

from pathlib import Path

import fitz
import numpy as np
import pytest

from cell_grid_service_v4 import (
    Cell,
    PageCellGrid,
    WritableRegion,
    WritableRun,
    _build_clean_ink_mask,
    _cell_metrics,
    _maximal_rectangles,
    analyze_pdf_cell_grid_v4,
    cell_id_from_rc,
    draw_cell_grid_overlay_v4,
    range_id_from_cells,
    rc_from_cell_id,
)

PDF1 = Path("test_original_pdf1.pdf")
PDF2 = Path("test_original_pdf2.pdf")


@pytest.fixture(scope="module")
def pdf1_bytes() -> bytes:
    return PDF1.read_bytes()


@pytest.fixture(scope="module")
def pdf2_bytes() -> bytes:
    return PDF2.read_bytes()


@pytest.fixture(scope="module")
def grids_pdf1(pdf1_bytes) -> list[PageCellGrid]:
    return analyze_pdf_cell_grid_v4(pdf1_bytes)


@pytest.fixture(scope="module")
def grids_pdf2(pdf2_bytes) -> list[PageCellGrid]:
    return analyze_pdf_cell_grid_v4(pdf2_bytes)


# ── ID round-trips ─────────────────────────────────────────────────────────


class TestCellId:
    @pytest.mark.parametrize(
        "row,col,cid",
        [
            (1, 1, "A1"),
            (1, 26, "Z1"),
            (1, 27, "AA1"),
            (12, 29, "AC12"),
            (70, 49, "AW70"),
        ],
    )
    def test_roundtrip(self, row, col, cid):
        assert cell_id_from_rc(row, col) == cid
        assert rc_from_cell_id(cid) == (row, col)

    def test_range_format(self):
        assert range_id_from_cells("A1", "A1") == "A1"
        assert range_id_from_cells("C12", "H12") == "C12:H12"
        assert range_id_from_cells("D5", "K9") == "D5:K9"


# ── Algorithm correctness on synthetic input ───────────────────────────────


class TestScoreFormula:
    def test_blank_cell_high_score(self):
        cell = np.full((40, 40), 0.98, dtype=np.float32)
        ink = np.zeros_like(cell, dtype=bool)
        score, ratio = _cell_metrics(cell, ink)
        assert score >= 0.9
        assert ratio == 0.0

    def test_dense_ink_low_score(self):
        cell = np.full((40, 40), 0.05, dtype=np.float32)
        ink = np.ones_like(cell, dtype=bool)
        score, ratio = _cell_metrics(cell, ink)
        assert score <= 0.1
        assert ratio == 1.0

    def test_high_variance_no_collapse(self):
        cell = np.tile([0.1, 0.95], (40, 20)).astype(np.float32)
        ink = cell < 0.45
        score, ratio = _cell_metrics(cell, ink)
        assert 0.0 <= score <= 1.0
        assert score >= 0.0


class TestInkMask:
    def test_speck_is_closed(self):
        gray = np.full((20, 20), 0.95, dtype=np.float32)
        gray[5, 5] = 0.0
        mask = _build_clean_ink_mask(gray, ink_brightness_max=0.45)
        assert mask.dtype == np.bool_
        assert mask.shape == gray.shape

    def test_block_survives(self):
        gray = np.full((20, 20), 0.95, dtype=np.float32)
        gray[5:15, 5:15] = 0.0
        mask = _build_clean_ink_mask(gray, ink_brightness_max=0.45)
        assert mask.sum() > 0


class TestMaximalRectangles:
    def test_single_block(self):
        w = np.zeros((6, 6), dtype=bool)
        w[1:4, 1:4] = True
        rects = _maximal_rectangles(w, min_cells=2, min_dim=2)
        assert (1, 3, 1, 3) in rects

    def test_no_writable_returns_empty(self):
        rects = _maximal_rectangles(np.zeros((5, 5), dtype=bool), min_cells=2, min_dim=2)
        assert rects == []

    def test_min_dim_filters(self):
        w = np.zeros((5, 5), dtype=bool)
        w[2, :] = True
        rects = _maximal_rectangles(w, min_cells=2, min_dim=2)
        assert rects == []


# ── End-to-end on real fixtures ────────────────────────────────────────────


class TestGridShape:
    def test_pdf1_shape(self, grids_pdf1):
        assert len(grids_pdf1) == 2
        for g in grids_pdf1:
            assert g.rows == 70
            assert g.cols == 49
            assert g.cell_size_pts == 12.0
            assert g.left_margin_pts >= 0
            assert g.top_margin_pts >= 0
            assert len(g.cells) == g.rows * g.cols

    def test_pdf2_shape(self, grids_pdf2):
        assert len(grids_pdf2) == 2
        for g in grids_pdf2:
            assert g.rows == 70
            assert g.cols == 49

    def test_centered_margins(self, grids_pdf1):
        for g in grids_pdf1:
            slack_x = g.page_w_pts - g.cols * g.cell_size_pts
            slack_y = g.page_h_pts - g.rows * g.cell_size_pts
            assert abs(g.left_margin_pts - slack_x / 2.0) < 0.05
            assert abs(g.top_margin_pts - slack_y / 2.0) < 0.05


class TestWritableCells:
    def test_writable_count_in_range(self, grids_pdf1):
        for g in grids_pdf1:
            writable = sum(1 for c in g.cells if c.writable)
            total = len(g.cells)
            ratio = writable / total
            assert 0.3 <= ratio <= 0.95

    def test_cell_coords_consistent(self, grids_pdf1):
        for g in grids_pdf1:
            for cell in g.cells:
                assert cell.pdf_x1 < cell.pdf_x2
                assert cell.pdf_y1 < cell.pdf_y2
                assert 0 <= cell.x_start_percent < cell.x_end_percent <= 100.001
                assert 0 <= cell.y_start_percent < cell.y_end_percent <= 100.001

    def test_cell_id_resolves_to_position(self, grids_pdf1):
        g = grids_pdf1[0]
        cs = g.cell_size_pts
        for cell in g.cells[:200]:
            row, col = rc_from_cell_id(cell.cell_id)
            expected_x1 = g.left_margin_pts + (col - 1) * cs
            expected_y1 = g.top_margin_pts + (row - 1) * cs
            assert abs(cell.pdf_x1 - expected_x1) < 0.05
            assert abs(cell.pdf_y1 - expected_y1) < 0.05


class TestRowRuns:
    def test_runs_have_neighbours(self, grids_pdf1):
        for g in grids_pdf1:
            for run in g.runs:
                assert run.cell_count >= 2
                assert run.start_col < run.end_col
                assert run.cell_count == run.end_col - run.start_col + 1

    def test_run_endpoint_ids_match_row(self, grids_pdf1):
        for g in grids_pdf1:
            for run in g.runs:
                assert run.start_cell_id.endswith(str(run.row))
                assert run.end_cell_id.endswith(str(run.row))

    def test_runs_within_writable(self, grids_pdf1):
        for g in grids_pdf1:
            cells_by_id = {c.cell_id: c for c in g.cells}
            for run in g.runs:
                for col in range(run.start_col, run.end_col + 1):
                    cid = cell_id_from_rc(run.row, col)
                    assert cells_by_id[cid].writable


class TestRegions:
    def test_regions_are_writable_rects(self, grids_pdf1):
        for g in grids_pdf1:
            cells_by_id = {c.cell_id: c for c in g.cells}
            for rg in g.regions:
                assert rg.cell_count == (
                    (rg.row_end - rg.row_start + 1) * (rg.col_end - rg.col_start + 1)
                )
                for cid in rg.cell_ids:
                    assert cells_by_id[cid].writable, (
                        f"region {rg.region_id} contains non-writable {cid}"
                    )

    def test_regions_sorted_by_area_desc(self, grids_pdf1):
        for g in grids_pdf1:
            areas = [rg.cell_count for rg in g.regions]
            assert areas == sorted(areas, reverse=True)

    def test_region_bbox_range_id_matches_corners(self, grids_pdf1):
        for g in grids_pdf1:
            for rg in g.regions:
                expected = range_id_from_cells(
                    cell_id_from_rc(rg.row_start, rg.col_start),
                    cell_id_from_rc(rg.row_end, rg.col_end),
                )
                assert rg.bbox_range_id == expected

    def test_at_least_one_region_pdf2(self, grids_pdf2):
        for g in grids_pdf2:
            assert len(g.regions) >= 1


class TestDenseInk:
    def test_full_black_page_yields_no_writable(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(0, 0, 595, 842), color=(0, 0, 0), fill=(0, 0, 0))
        raw = doc.tobytes()
        doc.close()

        grids = analyze_pdf_cell_grid_v4(raw)
        assert len(grids) == 1
        g = grids[0]
        writable = sum(1 for c in g.cells if c.writable)
        assert writable == 0
        assert g.runs == []
        assert g.regions == []


class TestOverlay:
    def test_overlay_written(self, pdf1_bytes, grids_pdf1, tmp_path):
        out = tmp_path / "v4_overlay.pdf"
        result = draw_cell_grid_overlay_v4(pdf1_bytes, grids_pdf1, out)
        assert result.exists()
        assert result.stat().st_size > 50_000


# ── Frontend resolution invariant ──────────────────────────────────────────


class TestFrontendResolution:
    """The frontend resolves cell_id → percent bbox using only the meta block.
    This must agree with the bbox stored on each Cell to <0.05% drift.
    """

    def test_resolution_matches_stored(self, grids_pdf1):
        for g in grids_pdf1:
            for cell in g.cells[::97]:
                row, col = rc_from_cell_id(cell.cell_id)
                x1 = g.left_margin_pts + (col - 1) * g.cell_size_pts
                y1 = g.top_margin_pts + (row - 1) * g.cell_size_pts
                x2 = x1 + g.cell_size_pts
                y2 = y1 + g.cell_size_pts
                x1p = (x1 / g.page_w_pts) * 100.0
                y1p = (y1 / g.page_h_pts) * 100.0
                x2p = (x2 / g.page_w_pts) * 100.0
                y2p = (y2 / g.page_h_pts) * 100.0
                assert abs(x1p - cell.x_start_percent) < 0.05
                assert abs(y1p - cell.y_start_percent) < 0.05
                assert abs(x2p - cell.x_end_percent) < 0.05
                assert abs(y2p - cell.y_end_percent) < 0.05
