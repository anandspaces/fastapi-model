"""Merge grading-stage annotations onto each item's remark boxes.

Called after the evaluator has produced ``annotations`` per item (with ``comment``
plus ``page_index`` / ``y_position_percent``). For each annotation:

  1. Find the nearest empty-comment remark on the same page within 20 pct units
     of the annotation's y center; copy ``comment`` onto it.
  2. If no match is close enough, create a new right-margin remark clamped to
     the item's effective answer-body y-range (so injected remarks never land on
     the printed question-text zone).

After enrichment, all remarks are re-spread to eliminate any new overlap.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import REM_MIN_H, REM_RIGHT_X1, REM_RIGHT_X2, REM_Y_MAX, REM_Y_MIN
from .layout import deoverlap_item_remarks

log = logging.getLogger(__name__)

_GUARD = 8.0  # % skipped past printed question-text zone from start_y


def enrich_remarks_from_annotations(items: list[dict[str, Any]]) -> None:
    """Fill remarks[].comment from each item's grading ``annotations`` array.

    Mutates items in place; returns None.
    """
    for item in items:
        remarks: list[dict[str, Any]] = item.get("remarks") or []
        annotations: list[dict[str, Any]] = item.get("annotations") or []
        if not annotations:
            continue

        item_start_page = int(item.get("start_page", 1))
        item_end_page   = int(item.get("end_page", item_start_page))
        item_sy         = float(item.get("start_y_position_percent", 0.0))
        item_ey         = float(item.get("end_y_position_percent", 100.0))

        matched_remark_indices: set[int] = set()

        for ann in annotations:
            try:
                target_page = int(ann.get("page_index", 0)) + 1   # 0-based → 1-based
                ann_y: float = float(ann.get("y_position_percent", 50))
            except (TypeError, ValueError):
                continue

            best_idx: int | None = None
            best_dist = float("inf")
            for i, rem in enumerate(remarks):
                if i in matched_remark_indices:
                    continue
                if rem.get("comment"):   # already filled by a previous annotation
                    continue
                if rem.get("page") != target_page:
                    continue
                y_centre = (float(rem.get("y1_pct", 0)) + float(rem.get("y2_pct", 0))) / 2.0
                dist = abs(ann_y - y_centre)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist <= 20.0:
                remarks[best_idx]["comment"] = str(ann.get("comment", ""))
                matched_remark_indices.add(best_idx)
            else:
                _inject_remark_for_annotation(
                    remarks, ann, ann_y, target_page,
                    item_start_page, item_end_page, item_sy, item_ey,
                )

        item["remarks"] = deoverlap_item_remarks(remarks)


def _inject_remark_for_annotation(
    remarks: list[dict[str, Any]],
    ann: dict[str, Any],
    ann_y: float,
    target_page: int,
    item_start_page: int,
    item_end_page: int,
    item_sy: float,
    item_ey: float,
) -> None:
    """Insert a brand-new right-margin remark clamped to the item's answer-body range."""
    if target_page == item_start_page == item_end_page:
        body_y_min = item_sy + _GUARD
        body_y_max = item_ey
    elif target_page == item_start_page:
        body_y_min = item_sy + _GUARD
        body_y_max = 100.0
    elif target_page == item_end_page:
        body_y_min = REM_Y_MIN
        body_y_max = item_ey
    else:
        body_y_min = REM_Y_MIN
        body_y_max = REM_Y_MAX

    if body_y_min >= body_y_max:
        log.info(
            "enrich_remarks: skip ann y=%.0f page=%s — no body room",
            ann_y, target_page,
        )
        return

    ann_y_clamped = max(body_y_min, min(body_y_max, ann_y))
    log.info(
        "enrich_remarks: new remark ann_y=%.0f → clamped=%.0f page=%s body[%.0f-%.0f]",
        ann_y, ann_y_clamped, target_page, body_y_min, body_y_max,
    )

    try:
        y1 = max(REM_Y_MIN, ann_y_clamped - 5.0)
        y2 = min(REM_Y_MAX, ann_y_clamped + 5.0)
        if y2 - y1 < REM_MIN_H:
            y2 = min(REM_Y_MAX, y1 + REM_MIN_H)
        remarks.append({
            "page": target_page,
            "x1_pct": REM_RIGHT_X1,
            "y1_pct": round(y1, 2),
            "x2_pct": REM_RIGHT_X2,
            "y2_pct": round(y2, 2),
            "comment": str(ann.get("comment", "")),
            "text_y1_pct": round(ann_y_clamped - 2.0, 2),
            "text_y2_pct": round(ann_y_clamped + 2.0, 2),
            "connector_type": "arrow",
            "connector_x_pct": 73.0,
            "connector_cy_pct": round(ann_y_clamped, 2),
        })
    except (TypeError, ValueError):
        pass
