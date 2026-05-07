"""Tests for cell-grid marking-box injection in smart OCR."""

from __future__ import annotations

from pathlib import Path

import pytest

from cell_grid_service_v3 import PageWritableRuns, WritableRun, analyze_pdf_writable_runs
from src.gemini_smart_ocr import (
    CELL_GRID_MIN_RUN_LENGTH,
    _inject_marking_boxes,
    _pick_best_run,
)

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "test_pdf3.pdf"


def _dummy_run(
    *,
    page: int,
    row: int,
    cell_count: int,
    y_start_percent: float,
    y_end_percent: float,
    x_start_percent: float = 10.0,
    x_end_percent: float = 50.0,
) -> WritableRun:
    return WritableRun(
        page=page,
        row=row,
        start_col=1,
        end_col=cell_count,
        start_cell_id="A1",
        end_cell_id="Z9",
        range_id="A1:A1",
        cell_count=cell_count,
        x_start_percent=x_start_percent,
        x_end_percent=x_end_percent,
        y_start_percent=y_start_percent,
        y_end_percent=y_end_percent,
        pdf_x1=0.0,
        pdf_y1=0.0,
        pdf_x2=10.0,
        pdf_y2=10.0,
        mean_score=0.9,
        mean_ink_ratio=0.01,
    )


class TestPickBestRun:
    def test_picks_longest_inside_range(self):
        runs = [
            _dummy_run(page=1, row=1, cell_count=3, y_start_percent=10.0, y_end_percent=15.0),
            _dummy_run(page=1, row=2, cell_count=5, y_start_percent=12.0, y_end_percent=18.0),
            _dummy_run(page=1, row=3, cell_count=4, y_start_percent=50.0, y_end_percent=60.0),
        ]
        chosen = _pick_best_run(runs, 11.0, 17.0)
        assert chosen is not None
        assert chosen.cell_count == 5

    def test_tiebreak_lower_on_page(self):
        runs = [
            _dummy_run(page=1, row=1, cell_count=4, y_start_percent=20.0, y_end_percent=25.0),
            _dummy_run(page=1, row=2, cell_count=4, y_start_percent=22.0, y_end_percent=27.0),
        ]
        chosen = _pick_best_run(runs, 19.0, 26.0)
        assert chosen is not None
        assert chosen.y_start_percent == 22.0

    def test_returns_none_when_below_min_length(self):
        runs = [
            _dummy_run(page=1, row=1, cell_count=2, y_start_percent=10.0, y_end_percent=15.0),
        ]
        assert _pick_best_run(runs, 10.0, 15.0) is None


@pytest.fixture(scope="module")
def pdf3_runs() -> list[PageWritableRuns]:
    if not PDF_PATH.is_file():
        pytest.skip("test_pdf3.pdf required in project root")
    return analyze_pdf_writable_runs(
        PDF_PATH.read_bytes(),
        min_run_length=CELL_GRID_MIN_RUN_LENGTH,
        dpi=150,
    )


class TestRealPdfFixture:
    def test_real_pdf_yields_runs(self, pdf3_runs):
        assert any(
            any(r.cell_count >= CELL_GRID_MIN_RUN_LENGTH for r in p.runs)
            for p in pdf3_runs
        )

    def test_inject_marking_boxes_adds_box_fields(self, pdf3_runs):
        best_run = None
        for p in pdf3_runs:
            for r in p.runs:
                if r.cell_count >= CELL_GRID_MIN_RUN_LENGTH:
                    if best_run is None or r.cell_count > best_run.cell_count:
                        best_run = r
        assert best_run is not None

        item = {
            "start_page": best_run.page,
            "end_page": best_run.page,
            "start_y_position_percent": best_run.y_start_percent + 0.01,
            "end_y_position_percent": best_run.y_end_percent - 0.01,
        }
        _inject_marking_boxes([item], pdf3_runs)

        assert item["marking_box_page"] == best_run.page
        assert "marking_box_x1_percent" in item
        x1, x2 = item["marking_box_x1_percent"], item["marking_box_x2_percent"]
        y1, y2 = item["marking_box_y1_percent"], item["marking_box_y2_percent"]
        mx, my = item["marking_x_position_percent"], item["marking_y_position_percent"]
        assert x1 < mx < x2
        assert y1 < my < y2

    def test_inject_centroid_matches_box(self, pdf3_runs):
        best_run = None
        for p in pdf3_runs:
            for r in p.runs:
                if r.cell_count >= CELL_GRID_MIN_RUN_LENGTH:
                    if best_run is None or r.cell_count > best_run.cell_count:
                        best_run = r
        assert best_run is not None

        item = {
            "start_page": best_run.page,
            "end_page": best_run.page,
            "start_y_position_percent": best_run.y_start_percent,
            "end_y_position_percent": best_run.y_end_percent,
        }
        _inject_marking_boxes([item], pdf3_runs)

        assert item["marking_x_position_percent"] == round(
            (best_run.x_start_percent + best_run.x_end_percent) / 2.0, 2
        )
        assert item["marking_y_position_percent"] == round(
            (best_run.y_start_percent + best_run.y_end_percent) / 2.0, 2
        )

    def test_skips_when_no_overlap(self, pdf3_runs):
        page1 = next(p for p in pdf3_runs if p.page == 1)
        qualifying = [
            r
            for r in page1.runs
            if r.cell_count >= CELL_GRID_MIN_RUN_LENGTH
        ]
        # Pick a thin horizontal band that does not intersect any qualifying run on page 1.
        occupied: list[tuple[float, float]] = [
            (r.y_start_percent, r.y_end_percent) for r in qualifying
        ]
        sy, ey = 0.05, 0.15
        for _ in range(200):
            overlaps = any(
                sy <= hi and ey >= lo for lo, hi in occupied
            )
            if not overlaps:
                break
            sy += 0.5
            ey += 0.5
        else:
            pytest.skip("could not find a non-overlapping y-band on page 1")

        item = {
            "start_page": 1,
            "end_page": 1,
            "start_y_position_percent": sy,
            "end_y_position_percent": ey,
        }
        before_keys = set(item.keys())
        _inject_marking_boxes([item], pdf3_runs)
        assert set(item.keys()) == before_keys
        assert "marking_box_x1_percent" not in item

    def test_falls_back_to_end_page(self, pdf3_runs):
        source_run = None
        for p in pdf3_runs:
            for r in p.runs:
                if r.cell_count >= CELL_GRID_MIN_RUN_LENGTH:
                    if source_run is None or r.cell_count > source_run.cell_count:
                        source_run = r
        assert source_run is not None

        sy = source_run.y_start_percent + 0.05
        ey = source_run.y_end_percent - 0.05
        runs_by_page = {p.page: p.runs for p in pdf3_runs}

        other_page = None
        for q, qr in runs_by_page.items():
            if q == source_run.page:
                continue
            if _pick_best_run(qr, sy, ey) is None:
                other_page = q
                break
        if other_page is None:
            pytest.skip(
                "no second page without an overlapping run for this vertical band"
            )

        item = {
            "start_page": other_page,
            "end_page": source_run.page,
            "start_y_position_percent": sy,
            "end_y_position_percent": ey,
        }
        _inject_marking_boxes([item], pdf3_runs)
        assert item.get("marking_box_page") == source_run.page
        assert item["marking_y_position_percent"] == round(
            (source_run.y_start_percent + source_run.y_end_percent) / 2.0, 2
        )


class TestInjectSyntheticFallback:
    """Guaranteed end_page fallback without depending on pdf3 layout."""

    def test_end_page_used_when_start_has_no_runs(self):
        good = _dummy_run(
            page=2,
            row=5,
            cell_count=4,
            y_start_percent=30.0,
            y_end_percent=35.0,
            x_start_percent=5.0,
            x_end_percent=80.0,
        )
        page_runs = [
            PageWritableRuns(
                page=1,
                rows=10,
                cols=10,
                cell_size_pts=25.0,
                left_margin_pts=0.0,
                top_margin_pts=0.0,
                runs=[],
            ),
            PageWritableRuns(
                page=2,
                rows=10,
                cols=10,
                cell_size_pts=25.0,
                left_margin_pts=0.0,
                top_margin_pts=0.0,
                runs=[good],
            ),
        ]
        item = {
            "start_page": 1,
            "end_page": 2,
            "start_y_position_percent": 31.0,
            "end_y_position_percent": 34.0,
        }
        _inject_marking_boxes([item], page_runs)
        assert item["marking_box_page"] == 2
        assert item["marking_box_y1_percent"] == 30.0
