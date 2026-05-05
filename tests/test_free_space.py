"""Tests for free-space detection using test_original_pdf2.pdf.

The PDF is a 2-page handwritten UPSC answer booklet:
  Page 1: Dense handwriting + mind-map diagram.
          Key free gaps: row 7 (35–40%, between intro and diagram) and
          right half of row 8 (40–45%).
  Page 2: Dense top half, empty bottom (~90–95%).

Run from the project root:
    python -m pytest tests/test_free_space.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.free_space_utils import (
    FreeZone,
    find_free_zones,
    get_grid_scores,
    merge_adjacent_zones,
    score_cell,
    snap_annotations_to_free_zones,
    snap_y_to_nearest_free_zone,
)
from src.free_space_service import (
    _DEFAULT_DPI,
    analyze_page_free_space,
    analyze_pdf_free_space,
    page_zones_to_api_response,
    snap_items_annotations,
    _rasterize_pdf_to_gray_arrays,
)

PDF_PATH = Path(__file__).parent.parent / "test_original_pdf2.pdf"
ROWS = 20
COLS = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return PDF_PATH.read_bytes()


@pytest.fixture(scope="module")
def gray_pages(pdf_bytes: bytes) -> list[np.ndarray]:
    return _rasterize_pdf_to_gray_arrays(pdf_bytes, dpi=_DEFAULT_DPI)


@pytest.fixture(scope="module")
def page1_gray(gray_pages: list[np.ndarray]) -> np.ndarray:
    return gray_pages[0]


@pytest.fixture(scope="module")
def page2_gray(gray_pages: list[np.ndarray]) -> np.ndarray:
    return gray_pages[1]


@pytest.fixture(scope="module")
def page1_scores(page1_gray: np.ndarray) -> list[list[float]]:
    return get_grid_scores(page1_gray, ROWS, COLS)


@pytest.fixture(scope="module")
def page2_scores(page2_gray: np.ndarray) -> list[list[float]]:
    return get_grid_scores(page2_gray, ROWS, COLS)


@pytest.fixture(scope="module")
def page1_zones(page1_gray: np.ndarray) -> list[FreeZone]:
    return analyze_page_free_space(page1_gray, rows=ROWS, cols=COLS, merge=True)


@pytest.fixture(scope="module")
def page2_zones(page2_gray: np.ndarray) -> list[FreeZone]:
    return analyze_page_free_space(page2_gray, rows=ROWS, cols=COLS, merge=True)


# ---------------------------------------------------------------------------
# 1. Unit tests: score_cell
# ---------------------------------------------------------------------------


class TestScoreCell:
    def test_pure_white_scores_near_one(self):
        cell = np.ones((50, 50), dtype=np.float32)
        assert score_cell(cell) >= 0.95

    def test_pure_black_scores_near_zero(self):
        cell = np.zeros((50, 50), dtype=np.float32)
        assert score_cell(cell) < 0.05

    def test_checkerboard_scores_low(self):
        # alternating 0/1 → high std → low score
        cell = np.zeros((50, 50), dtype=np.float32)
        cell[::2, ::2] = 1.0
        cell[1::2, 1::2] = 1.0
        assert score_cell(cell) < 0.4

    def test_mostly_white_with_few_dark_pixels_scores_high(self):
        cell = np.ones((50, 50), dtype=np.float32)
        cell[24, 10:40] = 0.0  # single horizontal line (printed rule)
        assert score_cell(cell) >= 0.60

    def test_empty_cell_returns_zero(self):
        assert score_cell(np.array([], dtype=np.float32).reshape(0, 10)) == 0.0

    def test_score_monotonic_with_density(self):
        """More ink → lower score."""
        h, w = 60, 60
        base = np.ones((h, w), dtype=np.float32)
        scores = []
        for density in (0, 10, 30, 60, 100):
            cell = base.copy()
            n = (h * w * density) // 100
            if n:
                idx = np.unravel_index(np.arange(n), (h, w))
                cell[idx] = 0.0
            scores.append(score_cell(cell))
        assert scores == sorted(scores, reverse=True), f"Not monotonic: {scores}"


# ---------------------------------------------------------------------------
# 2. Unit tests: get_grid_scores shape & range
# ---------------------------------------------------------------------------


class TestGridScores:
    def test_shape(self, page1_scores: list[list[float]]):
        assert len(page1_scores) == ROWS
        assert all(len(row) == COLS for row in page1_scores)

    def test_all_values_in_range(self, page1_scores: list[list[float]]):
        for row in page1_scores:
            for s in row:
                assert 0.0 <= s <= 1.0, f"score {s} out of [0,1]"

    def test_page1_row7_is_mostly_free(self, page1_scores: list[list[float]]):
        # Row 7 (35–40%) contains the blank gap between intro paragraph and mind map.
        # Expected: most columns have score > 0.65 (confirmed by manual inspection).
        row7 = page1_scores[7]
        high_count = sum(1 for s in row7 if s > 0.65)
        assert high_count >= 4, f"Expected ≥4 free cols in row7, got {high_count}: {row7}"

    def test_page1_row0_is_mostly_occupied(self, page1_scores: list[list[float]]):
        # Row 0 (0–5%) is the question stem — handwriting (score_cell prefers mean>std).
        row0 = page1_scores[0]
        occupied_count = sum(1 for s in row0 if s < 0.55)
        assert occupied_count >= 4, f"Expected ≥4 occupied cols in row0: {row0}"

    def test_page2_row18_is_mostly_free(self, page2_scores: list[list[float]]):
        # Row 18 (90–95%) on page 2 is the empty area below the conclusion paragraph.
        row18 = page2_scores[18]
        high_count = sum(1 for s in row18 if s > 0.65)
        assert high_count >= 4, f"Expected ≥4 free cols in page2 row18: {row18}"

    def test_page2_row0_to_5_mostly_occupied(self, page2_scores: list[list[float]]):
        # Top rows have dense handwriting (Aristotle quote, relationship section).
        top_occupied = sum(
            1
            for r in range(5)
            for s in page2_scores[r]
            if s < 0.55
        )
        assert top_occupied >= 9, (
            f"Expected many handwriting-heavy cells in top rows, got {top_occupied}"
        )


# ---------------------------------------------------------------------------
# 3. Unit tests: find_free_zones
# ---------------------------------------------------------------------------


class TestFindFreeZones:
    def test_returns_list_of_free_zones(self, page1_scores: list[list[float]]):
        zones = find_free_zones(
            page1_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.65
        )
        assert isinstance(zones, list)
        assert all(isinstance(z, FreeZone) for z in zones)

    def test_zones_sorted_by_score_desc(self, page1_scores: list[list[float]]):
        zones = find_free_zones(
            page1_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.65
        )
        if len(zones) >= 2:
            for a, b in zip(zones, zones[1:]):
                assert a.score >= b.score, "Zones not sorted by score desc"

    def test_page1_has_free_zones(self, page1_scores: list[list[float]]):
        zones = find_free_zones(
            page1_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.65
        )
        assert len(zones) >= 3, f"Expected ≥3 free zones on page1, got {len(zones)}"

    def test_page2_has_free_zones(self, page2_scores: list[list[float]]):
        zones = find_free_zones(
            page2_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.65
        )
        assert len(zones) >= 3, f"Expected ≥3 free zones on page2, got {len(zones)}"

    def test_percent_coords_in_range(self, page1_scores: list[list[float]]):
        zones = find_free_zones(
            page1_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.0
        )
        for z in zones:
            assert 0.0 <= z.x_start_percent <= z.x_end_percent <= 100.0
            assert 0.0 <= z.y_start_percent <= z.y_end_percent <= 100.0

    def test_top_n_respected(self, page1_scores: list[list[float]]):
        zones = find_free_zones(
            page1_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.0, top_n=5
        )
        assert len(zones) <= 5

    def test_left_binding_excluded(self, page1_scores: list[list[float]]):
        # Cols 0–1 excluded (binding strip) — analyze_page_free_space uses exclude_left_cols=2.
        zones = find_free_zones(
            page1_scores,
            ROWS,
            COLS,
            img_w=1191,
            img_h=1684,
            min_score=0.0,
            exclude_left_cols=2,
        )
        for z in zones:
            assert z.col >= 2, f"Left binding cols should be excluded, got col={z.col}"

    def test_bottom_footer_excluded(self, page1_scores: list[list[float]]):
        # Bottom two rows excluded (footer rules) — matches analyze_page_free_space.
        zones = find_free_zones(
            page1_scores,
            ROWS,
            COLS,
            img_w=1191,
            img_h=1684,
            min_score=0.0,
            exclude_bottom_rows=2,
        )
        for z in zones:
            assert z.row < ROWS - 2, f"Footer rows should be excluded, got row={z.row}"

    def test_page1_gap_row7_appears_in_zones(self, page1_scores: list[list[float]]):
        # The blank row at 35–40% should appear in results.
        zones = find_free_zones(
            page1_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.65
        )
        rows_found = {z.row for z in zones}
        assert 7 in rows_found, f"Row 7 (blank gap) missing from zones. Found rows: {rows_found}"

    def test_page2_bottom_empty_row_appears(self, page2_scores: list[list[float]]):
        zones = find_free_zones(
            page2_scores, ROWS, COLS, img_w=1191, img_h=1684, min_score=0.65
        )
        rows_found = {z.row for z in zones}
        assert 18 in rows_found, f"Row 18 (bottom empty area) missing. Found rows: {rows_found}"


# ---------------------------------------------------------------------------
# 4. Unit tests: merge_adjacent_zones
# ---------------------------------------------------------------------------


class TestMergeAdjacentZones:
    def test_merges_adjacent_cols_in_same_row(self):
        zones = [
            FreeZone(x_start_percent=12.5, x_end_percent=25.0, y_start_percent=35.0, y_end_percent=40.0, score=0.9, row=7, col=1),
            FreeZone(x_start_percent=25.0, x_end_percent=37.5, y_start_percent=35.0, y_end_percent=40.0, score=0.95, row=7, col=2),
            FreeZone(x_start_percent=37.5, x_end_percent=50.0, y_start_percent=35.0, y_end_percent=40.0, score=0.85, row=7, col=3),
        ]
        merged = merge_adjacent_zones(zones, ROWS, COLS)
        assert len(merged) == 1
        assert merged[0].x_start_percent == pytest.approx(12.5)
        assert merged[0].x_end_percent == pytest.approx(50.0)
        assert merged[0].score == pytest.approx(0.9, abs=0.05)

    def test_does_not_merge_non_adjacent(self):
        zones = [
            FreeZone(x_start_percent=12.5, x_end_percent=25.0, y_start_percent=35.0, y_end_percent=40.0, score=0.9, row=7, col=1),
            FreeZone(x_start_percent=37.5, x_end_percent=50.0, y_start_percent=35.0, y_end_percent=40.0, score=0.85, row=7, col=3),
        ]
        merged = merge_adjacent_zones(zones, ROWS, COLS)
        assert len(merged) == 2

    def test_merges_vertical_adjacent_same_col(self):
        """Same column + consecutive rows collapse to one tall strip (right margin case)."""
        zones = [
            FreeZone(x_start_percent=12.5, x_end_percent=25.0, y_start_percent=35.0, y_end_percent=40.0, score=0.9, row=7, col=1),
            FreeZone(x_start_percent=12.5, x_end_percent=25.0, y_start_percent=40.0, y_end_percent=45.0, score=0.9, row=8, col=1),
        ]
        merged = merge_adjacent_zones(zones, ROWS, COLS)
        assert len(merged) == 1
        assert merged[0].y_start_percent == pytest.approx(35.0)
        assert merged[0].y_end_percent == pytest.approx(45.0)

    def test_empty_input(self):
        assert merge_adjacent_zones([], ROWS, COLS) == []


# ---------------------------------------------------------------------------
# 5. Unit tests: snap_y_to_nearest_free_zone
# ---------------------------------------------------------------------------


class TestSnapYToNearestFreeZone:
    def _zone(self, y_start: float, y_end: float, score: float = 0.8) -> FreeZone:
        return FreeZone(
            x_start_percent=15.0, x_end_percent=90.0,
            y_start_percent=y_start, y_end_percent=y_end,
            score=score, row=0, col=1,
        )

    def test_snaps_to_nearest_zone(self):
        zones = [self._zone(30, 35), self._zone(60, 65), self._zone(80, 85)]
        result = snap_y_to_nearest_free_zone(32.0, zones)
        assert result is not None
        assert result.y_center() == pytest.approx(32.5)

    def test_returns_none_when_all_beyond_max_shift(self):
        zones = [self._zone(80, 85), self._zone(90, 95)]
        result = snap_y_to_nearest_free_zone(10.0, zones, max_shift_pct=15.0)
        assert result is None

    def test_returns_none_for_empty_zones(self):
        result = snap_y_to_nearest_free_zone(50.0, [])
        assert result is None

    def test_prefers_closest_zone(self):
        zones = [self._zone(30, 35), self._zone(55, 60)]
        result = snap_y_to_nearest_free_zone(40.0, zones, max_shift_pct=30.0)
        assert result is not None
        # 40 - 32.5 = 7.5, 40 - 57.5 = 17.5 → first is closer
        assert result.y_center() == pytest.approx(32.5)


# ---------------------------------------------------------------------------
# 6. Unit tests: snap_annotations_to_free_zones
# ---------------------------------------------------------------------------


class TestSnapAnnotationsToFreeZones:
    def _make_zone(self, y_start: float, y_end: float, col: int = 3) -> FreeZone:
        return FreeZone(
            x_start_percent=col * 12.5,
            x_end_percent=(col + 1) * 12.5,
            y_start_percent=y_start,
            y_end_percent=y_end,
            score=0.85,
            row=int(y_start * ROWS // 100),
            col=col,
        )

    def test_adjusts_annotation_coordinates(self):
        ann = [{"page_index": 0, "y_position_percent": 42.0, "x_start_percent": 10.0, "x_end_percent": 90.0, "comment": "test", "is_positive": True}]
        zones = [[self._make_zone(37.5, 42.5)]]
        result = snap_annotations_to_free_zones(ann, zones)
        assert result[0]["y_position_percent"] == pytest.approx(40.0)
        assert result[0]["_snapped"] is True

    def test_does_not_snap_when_beyond_max_shift(self):
        ann = [{"page_index": 0, "y_position_percent": 10.0, "x_start_percent": 10.0, "x_end_percent": 90.0, "comment": "test", "is_positive": False}]
        zones = [[self._make_zone(80, 85)]]
        result = snap_annotations_to_free_zones(ann, zones, max_shift_pct=15.0)
        assert result[0]["y_position_percent"] == pytest.approx(10.0)
        assert result[0]["_snapped"] is False

    def test_preserves_original_comment_and_flags(self):
        ann = [{"page_index": 0, "y_position_percent": 50.0, "x_start_percent": 5.0, "x_end_percent": 95.0, "comment": "My remark", "is_positive": True, "line_style": "zigzag"}]
        zones = [[self._make_zone(47.5, 52.5)]]
        result = snap_annotations_to_free_zones(ann, zones)
        assert result[0]["comment"] == "My remark"
        assert result[0]["is_positive"] is True
        assert result[0]["line_style"] == "zigzag"

    def test_handles_missing_page_index(self):
        ann = [{"page_index": 5, "y_position_percent": 50.0, "x_start_percent": 10.0, "x_end_percent": 90.0, "comment": "c", "is_positive": False}]
        zones = [[self._make_zone(47.5, 52.5)]]  # only page 0 available
        result = snap_annotations_to_free_zones(ann, zones)
        assert result[0]["_snapped"] is False
        assert result[0]["y_position_percent"] == pytest.approx(50.0)

    def test_does_not_mutate_original(self):
        ann = [{"page_index": 0, "y_position_percent": 50.0, "x_start_percent": 5.0, "x_end_percent": 95.0, "comment": "c", "is_positive": False}]
        zones = [[self._make_zone(47.5, 52.5)]]
        snap_annotations_to_free_zones(ann, zones)
        assert ann[0]["y_position_percent"] == pytest.approx(50.0)  # original unchanged


# ---------------------------------------------------------------------------
# 7. Integration tests: analyze_page_free_space on real PDF
# ---------------------------------------------------------------------------


class TestAnalyzePageFreeSpaceReal:
    def test_page1_returns_zones(self, page1_zones: list[FreeZone]):
        assert len(page1_zones) >= 2, "Page 1 should have at least 2 free zones"

    def test_page2_returns_zones(self, page2_zones: list[FreeZone]):
        assert len(page2_zones) >= 2, "Page 2 should have at least 2 free zones"

    def test_page1_zones_have_valid_percent_coords(self, page1_zones: list[FreeZone]):
        for z in page1_zones:
            assert 0.0 <= z.x_start_percent < z.x_end_percent <= 100.0
            assert 0.0 <= z.y_start_percent < z.y_end_percent <= 100.0

    def test_page1_zones_all_above_min_score(self, page1_zones: list[FreeZone]):
        for z in page1_zones:
            assert z.score >= 0.70, f"Zone score {z.score} below threshold"

    def test_page1_blank_row_gap_is_in_zones(self, page1_zones: list[FreeZone]):
        # The visible blank row between intro and mind map is at ~35–40% (row 7).
        # At least one zone should exist in the 30–45% y-band.
        band_zones = [z for z in page1_zones if 30.0 <= z.y_start_percent <= 45.0]
        assert len(band_zones) >= 1, (
            f"Expected a free zone in y=30–45% on page 1 (blank gap). "
            f"Zones found: {[(z.y_start_percent, z.y_end_percent) for z in page1_zones]}"
        )

    def test_page2_bottom_empty_zone_is_found(self, page2_zones: list[FreeZone]):
        # Page 2 emptiness sits in the lowest band — merged strips may span 75–90%.
        bottom_zones = [
            z for z in page2_zones
            if z.y_end_percent > 85.0 and z.y_start_percent < 96.0
        ]
        assert len(bottom_zones) >= 1, (
            f"Expected free zone below y=85% on page 2. "
            f"Zones: {[(z.y_start_percent, z.y_end_percent) for z in page2_zones]}"
        )

    def test_no_zones_in_binding_strip(self, page1_zones: list[FreeZone]):
        # Leftmost two columns (binding) excluded — first zone x_start ≥ 25%.
        for z in page1_zones:
            assert z.x_start_percent >= 2 * (100.0 / COLS), (
                f"Zone at x_start={z.x_start_percent} is inside excluded binding cols"
            )


# ---------------------------------------------------------------------------
# 8. Integration: analyze_pdf_free_space (full PDF)
# ---------------------------------------------------------------------------


class TestAnalyzePdfFreeSpaceReal:
    def test_returns_two_pages(self, pdf_bytes: bytes):
        result = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)
        assert len(result) == 2

    def test_each_page_has_zones(self, pdf_bytes: bytes):
        result = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)
        for page_idx, zones in enumerate(result):
            assert len(zones) >= 1, f"Page {page_idx + 1} returned no free zones"

    def test_custom_min_score_strict_reduces_zones(self, pdf_bytes: bytes):
        default = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS, min_score=0.70)
        strict = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS, min_score=0.90)
        total_default = sum(len(p) for p in default)
        total_strict = sum(len(p) for p in strict)
        assert total_strict <= total_default, (
            "Stricter threshold should yield ≤ zones"
        )

    def test_custom_min_score_loose_increases_zones(self, pdf_bytes: bytes):
        default = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS, min_score=0.70)
        loose = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS, min_score=0.30)
        total_default = sum(len(p) for p in default)
        total_loose = sum(len(p) for p in loose)
        assert total_loose >= total_default, (
            "Looser threshold should yield ≥ zones"
        )

    def test_api_response_shape(self, pdf_bytes: bytes):
        zones = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)
        data = page_zones_to_api_response(zones)
        assert len(data) == 2
        for entry in data:
            assert "pageIndex" in entry
            assert "freeZones" in entry
            for z in entry["freeZones"]:
                assert set(z.keys()) == {
                    "x_start_percent", "x_end_percent",
                    "y_start_percent", "y_end_percent",
                    "score",
                }


# ---------------------------------------------------------------------------
# 9. Integration: snap_items_annotations with real page zones
# ---------------------------------------------------------------------------


class TestSnapItemsAnnotations:
    def test_annotations_are_snapped_into_free_zones(self, pdf_bytes: bytes):
        page_zones = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)

        # Simulate LLM-generated items with annotations on both pages
        items = [
            {
                "question_id": 1,
                "student_answer": "Emotional intelligence is the ability...",
                "annotations": [
                    # Page 1, y=10% — dense area, should snap to nearest free zone
                    {
                        "page_index": 0,
                        "y_position_percent": 10.0,
                        "x_start_percent": 20.0,
                        "x_end_percent": 80.0,
                        "comment": "Good introduction framing.",
                        "is_positive": True,
                        "line_style": "straight",
                    },
                    # Page 2, y=92% — should land in/near the bottom empty area
                    {
                        "page_index": 1,
                        "y_position_percent": 92.0,
                        "x_start_percent": 10.0,
                        "x_end_percent": 90.0,
                        "comment": "Conclusion is abrupt; add a synthesis line.",
                        "is_positive": False,
                        "line_style": "zigzag",
                    },
                ],
            }
        ]

        updated = snap_items_annotations(items, page_zones)
        assert len(updated) == 1
        anns = updated[0]["annotations"]
        assert len(anns) == 2

        # First annotation: should be snapped
        ann0 = anns[0]
        assert "_snapped" in ann0
        assert ann0["comment"] == "Good introduction framing."

        # Second annotation at y=92% — should be snapped to the bottom free zone (page 2)
        ann1 = anns[1]
        assert "_snapped" in ann1
        assert ann1["comment"] == "Conclusion is abrupt; add a synthesis line."
        # If snapped, the y should be near one of the free zones (not wildly different)
        if ann1["_snapped"]:
            assert abs(ann1["y_position_percent"] - 92.0) <= 30.0

    def test_items_without_annotations_unchanged(self, pdf_bytes: bytes):
        page_zones = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)
        items = [{"question_id": 2, "student_answer": "...", "annotations": []}]
        updated = snap_items_annotations(items, page_zones)
        assert updated[0]["annotations"] == []

    def test_items_without_annotation_key_unchanged(self, pdf_bytes: bytes):
        page_zones = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)
        items = [{"question_id": 3, "student_answer": "..."}]
        updated = snap_items_annotations(items, page_zones)
        assert "annotations" not in updated[0]


# ---------------------------------------------------------------------------
# 10. Diagnostic: print a visual summary of free zones (not a test assertion)
# ---------------------------------------------------------------------------


def test_print_free_zone_summary(pdf_bytes: bytes):
    """Human-readable output showing where remarks would be placed on each page."""
    page_zones = analyze_pdf_free_space(pdf_bytes, rows=ROWS, cols=COLS)
    for page_idx, zones in enumerate(page_zones):
        print(f"\n{'='*60}")
        print(f"Page {page_idx + 1}: {len(zones)} free zone(s)")
        print(f"{'='*60}")
        if not zones:
            print("  (no free zones found above threshold)")
            continue
        for z in zones:
            bar_start = int(z.x_start_percent / 100 * 40)
            bar_end = int(z.x_end_percent / 100 * 40)
            bar = " " * bar_start + "█" * (bar_end - bar_start) + " " * (40 - bar_end)
            print(
                f"  y={z.y_start_percent:5.1f}%–{z.y_end_percent:5.1f}%"
                f"  x={z.x_start_percent:5.1f}%–{z.x_end_percent:5.1f}%"
                f"  score={z.score:.3f}"
                f"  [{bar}]"
            )
