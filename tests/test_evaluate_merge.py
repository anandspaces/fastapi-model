"""Tests for Stage-3 merge: OCR layout + evaluator annotations."""

from __future__ import annotations

import pytest

from src.gemini_evaluate_student_answers import (
    _normalize_annotation_page_index,
    merge_evaluations_into_items,
)


def test_normalize_page_index_multipage_prefers_zero_based_on_overlap():
    """When 1-based and 0-based collide (e.g. p=1,2), multi-page spans trust 0-based."""
    item = {"start_page": 1, "end_page": 3, "marking_page": 2}
    assert _normalize_annotation_page_index(0, item) == 0
    assert _normalize_annotation_page_index(2, item) == 2
    assert _normalize_annotation_page_index(1, item) == 1
    assert _normalize_annotation_page_index(3, item) == 2


def test_merge_reseeds_marking_page_coords_and_spreads_y():
    items = [
        {
            "question_id": 1,
            "start_page": 1,
            "end_page": 1,
            "marking_page": 1,
            "marking_y_position_percent": 40.0,
            "marking_x_position_percent": 60.0,
        }
    ]
    evaluations = [
        {
            "question_id": 1,
            "max_marks": 10,
            "marks_awarded": 5,
            "status": "partial",
            "student_answer_summary": "x",
            "feedback": "y",
            "annotations": [
                {
                    "page_index": 1,
                    "y_position_percent": 5.0,
                    "x_start_percent": 0.0,
                    "x_end_percent": 100.0,
                    "comment": "a",
                    "is_positive": False,
                },
                {
                    "page_index": 1,
                    "y_position_percent": 7.0,
                    "x_start_percent": 0.0,
                    "x_end_percent": 100.0,
                    "comment": "b",
                    "is_positive": True,
                },
            ],
        }
    ]
    out = merge_evaluations_into_items(items, evaluations)
    anns = out[0]["annotations"]
    assert len(anns) == 2
    assert anns[0]["page_index"] == 0
    assert anns[1]["page_index"] == 0
    ys = sorted(float(a["y_position_percent"]) for a in anns)
    assert ys[0] < ys[1], "Spread Y so snapper gets distinct seeds"
    assert abs(sum(ys) / 2 - 40.0) < 12.0, "Cluster around OCR marking_y"
    assert 5.0 <= anns[0]["x_start_percent"] < anns[0]["x_end_percent"] <= 95.0
    assert 5.0 <= anns[1]["x_start_percent"] < anns[1]["x_end_percent"] <= 95.0


def test_merge_does_not_touch_other_pages_for_reseed():
    items = [
        {
            "question_id": 2,
            "start_page": 1,
            "end_page": 2,
            "marking_page": 1,
            "marking_y_position_percent": 40.0,
            "marking_x_position_percent": 50.0,
        }
    ]
    evaluations = [
        {
            "question_id": 2,
            "max_marks": 5,
            "marks_awarded": 3,
            "status": "partial",
            "student_answer_summary": "x",
            "feedback": "y",
            "annotations": [
                {
                    "page_index": 1,
                    "y_position_percent": 88.0,
                    "x_start_percent": 10.0,
                    "x_end_percent": 90.0,
                    "comment": "on page 2",
                    "is_positive": False,
                },
            ],
        }
    ]
    out = merge_evaluations_into_items(items, evaluations)
    ann = out[0]["annotations"][0]
    assert ann["page_index"] == 1
    assert pytest.approx(float(ann["y_position_percent"]), rel=0, abs=1e-6) == 88.0
