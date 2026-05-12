"""Layout post-processing for smart-OCR annotations.

Three responsibilities:
  1. ``content_bands`` parsing / merging / extending.
  2. Remark slot derivation (from anchor-mark clusters) + spread / deoverlap into
     left/right margin columns + height computation for the rendered comment text.
  3. Coordinate conversion from grid (1..50) to page-percent (0..100) plus connector
     fields (arrow / brace) used by the renderer.

All public helpers are pure functions that mutate only their direct inputs (or
return new lists). The Gemini-call layer in ``classify.py`` / ``pipeline.py``
composes them without owning any geometry.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.grid_overlay import grid_to_pct

from .config import (
    ANCHOR_CLUSTER_GAP_GRID,
    AVG_CHAR_WIDTH_FACTOR,
    BOX_PADDING_PT,
    CONTENT_X_MAX,
    CONTENT_X_MIN,
    FONT_SIZE_PT,
    INTER_REMARK_GAP_PCT,
    LINE_HEIGHT_FACTOR,
    PAGE_HEIGHT_PT,
    PAGE_WIDTH_PT,
    REM_LEFT_X1,
    REM_LEFT_X2,
    REM_MIN_H,
    REM_RIGHT_X1,
    REM_RIGHT_X2,
    REM_Y_MAX,
    REM_Y_MIN,
)

log = logging.getLogger(__name__)


# --- content_bands --------------------------------------------------------------------

def parse_content_bands(raw_bands: Any) -> list[dict[str, float]]:
    """Validate ``content_bands`` from Gemini and clamp to grid range."""
    if not isinstance(raw_bands, list):
        return []
    bands: list[dict[str, float]] = []
    for b in raw_bands:
        if not isinstance(b, dict):
            continue
        try:
            y1 = max(1.0, min(50.0, float(b.get("y1", 1))))
            y2 = max(1.0, min(50.0, float(b.get("y2", 50))))
            if y2 > y1:
                bands.append({"y1": y1, "y2": y2})
        except (TypeError, ValueError):
            pass
    return bands


def merge_content_bands(
    bands: list[dict[str, float]], gap_threshold: float = 4.0
) -> list[dict[str, float]]:
    """Merge consecutive bands separated by ≤ gap_threshold grid rows.

    Eliminates over-segmentation caused by inter-line whitespace within a single
    paragraph being reported as separate bands.
    """
    if not bands:
        return bands
    sorted_b = sorted(bands, key=lambda b: b["y1"])
    merged = [dict(sorted_b[0])]
    for b in sorted_b[1:]:
        if b["y1"] - merged[-1]["y2"] <= gap_threshold:
            merged[-1]["y2"] = max(merged[-1]["y2"], b["y2"])
        else:
            merged.append(dict(b))
    return merged


def extend_last_band(
    bands: list[dict[str, float]],
    anchor_cys_grid: list[float],
    page_type: str,
) -> list[dict[str, float]]:
    """Extend the last content band to cover text below the final anchor mark.

    PARAGRAPH and CORRECTION pages often have a conclusion paragraph that Gemini
    stops short of. Extends to ``max(last_y2, max_anchor_cy) + 7``, capped at grid 44.
    """
    if not bands or page_type not in ("PARAGRAPH", "CORRECTION"):
        return bands
    bands = [dict(b) for b in bands]
    last = bands[-1]
    max_anchor = max(anchor_cys_grid, default=last["y2"])
    new_y2 = min(44.0, max(last["y2"], max_anchor) + 7.0)
    if new_y2 > last["y2"]:
        log.info(
            "  extend_last_band  y2 %.0f → %.0f  (max_anchor=%.1f type=%s)",
            last["y2"], new_y2, max_anchor, page_type,
        )
        last["y2"] = new_y2
    return bands


# --- Remark slot derivation from anchor clusters --------------------------------------


def remarks_from_anchor_marks(
    anchor_marks_grid: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive right-margin remark slots from anchor mark y-positions.

    Clusters anchor marks within ``ANCHOR_CLUSTER_GAP_GRID`` rows and places one slot
    per cluster at the cluster centroid. Every remark is vertically aligned with
    something the teacher actually annotated.

    Fallback: 3 evenly spaced slots when no anchor marks are present.
    """
    if not anchor_marks_grid:
        return [
            {"x1": 39.0, "y1": 10.0, "x2": 49.0, "y2": 14.0, "comment": "",
             "_text_y1_grid": 10.0, "_text_y2_grid": 14.0, "_cluster_size": 0},
            {"x1": 39.0, "y1": 24.0, "x2": 49.0, "y2": 28.0, "comment": "",
             "_text_y1_grid": 24.0, "_text_y2_grid": 28.0, "_cluster_size": 0},
            {"x1": 39.0, "y1": 38.0, "x2": 49.0, "y2": 42.0, "comment": "",
             "_text_y1_grid": 38.0, "_text_y2_grid": 42.0, "_cluster_size": 0},
        ]

    cys = sorted(float(a.get("cy", 25)) for a in anchor_marks_grid)

    clusters: list[list[float]] = []
    for cy in cys:
        if clusters and cy - clusters[-1][-1] <= ANCHOR_CLUSTER_GAP_GRID:
            clusters[-1].append(cy)
        else:
            clusters.append([cy])

    slots: list[dict[str, Any]] = []
    for cluster in clusters:
        cy = sum(cluster) / len(cluster)
        y1 = max(6.0, cy - 2.0)
        y2 = min(45.0, y1 + 4.0)
        slots.append({
            "x1": 39.0, "y1": y1, "x2": 49.0, "y2": y2, "comment": "",
            "_text_y1_grid": min(cluster),
            "_text_y2_grid": max(cluster),
            "_cluster_size": len(cluster),
        })

    _MIN_SLOTS = 3
    if len(slots) < _MIN_SLOTS:
        occupied = {round(s["y1"]) for s in slots}
        candidates = [10.0, 20.0, 30.0, 38.0, 14.0, 26.0]
        for cy in candidates:
            if len(slots) >= _MIN_SLOTS:
                break
            y1 = max(6.0, cy - 2.0)
            if round(y1) in occupied:
                continue
            y2 = min(45.0, y1 + 4.0)
            slots.append({
                "x1": 39.0, "y1": y1, "x2": 49.0, "y2": y2, "comment": "",
                "_text_y1_grid": y1, "_text_y2_grid": y2, "_cluster_size": 0,
            })
            occupied.add(round(y1))
        slots.sort(key=lambda s: s["y1"])

    return slots


def filter_overlapping_remarks(
    remarks: list[dict[str, Any]],
    content_bands: list[dict[str, float]],
    page_num: int = 0,
) -> list[dict[str, Any]]:
    """Remove remark boxes that overlap with identified handwriting zones.

    Right-margin remarks (x1 >= 38) are always kept — student writing stays in
    columns 1-37, so the right margin is free at any y position.
    Horizontal-gap remarks are dropped if their y-span intersects any content band.
    """
    if not content_bands:
        return remarks
    result: list[dict[str, Any]] = []
    for r in remarks:
        x1_grid = float(r.get("x1", 5))
        x2_grid = float(r.get("x2", 50))
        y1_grid = float(r.get("y1", 1))
        y2_grid = float(r.get("y2", 50))
        if x1_grid >= 38:
            log.info(
                "  remark_filter[p%s] KEPT  right_margin x[%.0f-%.0f] y[%.0f-%.0f]",
                page_num, x1_grid, x2_grid, y1_grid, y2_grid,
            )
            result.append(r)
            continue
        overlapping_band = next(
            (b for b in content_bands if y1_grid < b["y2"] and y2_grid > b["y1"]), None
        )
        if overlapping_band is None:
            log.info(
                "  remark_filter[p%s] KEPT  gap         x[%.0f-%.0f] y[%.0f-%.0f]",
                page_num, x1_grid, x2_grid, y1_grid, y2_grid,
            )
            result.append(r)
        else:
            log.info(
                "  remark_filter[p%s] DROP  overlap     x[%.0f-%.0f] y[%.0f-%.0f]"
                "  hits_band y[%.0f-%.0f]",
                page_num, x1_grid, x2_grid, y1_grid, y2_grid,
                overlapping_band["y1"], overlapping_band["y2"],
            )
    return result


# --- Font-aware remark height + two-column spread -------------------------------------


def remark_height_pct(comment: str, col_x1_pct: float, col_x2_pct: float) -> float:
    """Compute the remark box height (%) needed to fit `comment` at FONT_SIZE_PT.

    Uses A4 dimensions, the column's percentage width, and char-wrap arithmetic.
    Always returns at least REM_MIN_H.
    """
    col_width_pt   = (col_x2_pct - col_x1_pct) / 100.0 * PAGE_WIDTH_PT
    usable_pt      = max(20.0, col_width_pt - BOX_PADDING_PT)
    chars_per_line = max(8, int(usable_pt / (FONT_SIZE_PT * AVG_CHAR_WIDTH_FACTOR)))
    n_lines        = math.ceil(len(comment) / chars_per_line) if comment else 1
    h_pt           = n_lines * FONT_SIZE_PT * LINE_HEIGHT_FACTOR + BOX_PADDING_PT
    return max(REM_MIN_H, h_pt / PAGE_HEIGHT_PT * 100.0)


def _first_free_slot(
    placed: list[tuple[float, float]], ideal_y1: float, height: float
) -> float | None:
    """Return the first y1 ≥ ideal_y1 where [y1, y1+height] doesn't overlap any placed interval.

    Scans placed intervals (sorted) and jumps past each conflict until a free gap is
    found or REM_Y_MAX is exceeded. Returns None when no slot fits.
    """
    y1 = ideal_y1
    for _ in range(20):
        y2 = y1 + height
        if y2 > REM_Y_MAX:
            return None
        conflict = next((p for p in placed if p[0] < y2 and p[1] > y1), None)
        if conflict is None:
            return y1
        y1 = conflict[1]
    return None


def spread_remarks_two_column(
    remarks_pct: list[dict[str, Any]],
    page_num: int = 0,
) -> list[dict[str, Any]]:
    """Re-layout a page's remarks into right + left margin with no overlaps.

    Uses interval-based free-slot search: each remark independently finds the first
    non-overlapping slot in right and left columns, then places on whichever side is
    closer to its ideal y. True bidirectional — naturally alternates when one side is
    densely packed. Remarks that can't fit in either column are dropped silently.
    """
    if not remarks_pct:
        return []

    sorted_r = sorted(remarks_pct, key=lambda r: r.get("y1_pct", 0.0))

    log.info(
        "spread[p%s] input  %s", page_num,
        " ".join(
            f"x[{r.get('x1_pct',0):.0f}-{r.get('x2_pct',0):.0f}]"
            f"y[{r.get('y1_pct',0):.0f}-{r.get('y2_pct',0):.0f}]"
            for r in sorted_r
        ),
    )

    right_placed: list[tuple[float, float]] = []
    left_placed:  list[tuple[float, float]] = []
    result: list[dict[str, Any]] = []

    for orig in sorted_r:
        r       = dict(orig)
        comment = r.get("comment", "") or ""
        h_right = remark_height_pct(comment, REM_RIGHT_X1, REM_RIGHT_X2)
        h_left  = remark_height_pct(comment, REM_LEFT_X1,  REM_LEFT_X2)
        ideal   = max(REM_Y_MIN, r.get("y1_pct", REM_Y_MIN))

        ry1 = _first_free_slot(right_placed, ideal, h_right)
        ly1 = _first_free_slot(left_placed,  ideal, h_left)

        right_fits = ry1 is not None
        left_fits  = ly1 is not None

        # Fallback: when forward scan overflows, try bottom-clamped position in each column.
        if not right_fits:
            bc = REM_Y_MAX - h_right
            if bc >= REM_Y_MIN and not any(p[0] < REM_Y_MAX and p[1] > bc for p in right_placed):
                ry1 = bc
                right_fits = True
                log.info("  spread[p%s] R_bottom_clamp ideal_y=%.0f → y1=%.0f", page_num, ideal, bc)

        if not left_fits:
            bc = REM_Y_MAX - h_left
            if bc >= REM_Y_MIN and not any(p[0] < REM_Y_MAX and p[1] > bc for p in left_placed):
                ly1 = bc
                left_fits = True
                log.info("  spread[p%s] L_bottom_clamp ideal_y=%.0f → y1=%.0f", page_num, ideal, bc)

        if not right_fits and not left_fits:
            log.info("  spread[p%s] DROP  truly_no_space ideal_y=%.0f", page_num, ideal)
            continue

        # Pick whichever side's free slot starts closer to ideal y; tie → right.
        if right_fits and (not left_fits or ry1 <= ly1):
            col = "R"
            y1, y2 = ry1, ry1 + h_right
            right_placed.append((y1, y2 + INTER_REMARK_GAP_PCT))
            right_placed.sort()
            r.update({"x1_pct": REM_RIGHT_X1, "x2_pct": REM_RIGHT_X2,
                       "y1_pct": round(y1, 2), "y2_pct": round(y2, 2)})
        else:
            col = "L"
            y1, y2 = ly1, ly1 + h_left
            left_placed.append((y1, y2 + INTER_REMARK_GAP_PCT))
            left_placed.sort()
            r.update({"x1_pct": REM_LEFT_X1, "x2_pct": REM_LEFT_X2,
                       "y1_pct": round(y1, 2), "y2_pct": round(y2, 2)})

        log.info(
            "  spread[p%s] %s  ideal_y=%.0f → y[%.0f-%.0f](h=%.1f)  push=%.0f  comment_len=%s",
            page_num, col, ideal, y1, y2,
            h_right if col == "R" else h_left,
            y1 - ideal, len(comment),
        )
        result.append(r)

    log.info(
        "spread[p%s] output R=%s L=%s", page_num,
        " ".join(f"y[{y1:.0f}-{y2:.0f}]" for y1, y2 in right_placed),
        " ".join(f"y[{y1:.0f}-{y2:.0f}]" for y1, y2 in left_placed),
    )
    return result


# --- Connectors -----------------------------------------------------------------------


def recalc_connector(r: dict[str, Any]) -> None:
    """Recompute connector fields after spread may have changed the remark's column."""
    x1 = float(r.get("x1_pct", 76.0))
    x2 = float(r.get("x2_pct", 96.0))
    if x2 < 74.0:
        r["connector_type"] = "none"
        r.pop("connector_cy_pct", None)
        return
    text_y1 = float(r.get("text_y1_pct", 0.0))
    text_y2 = float(r.get("text_y2_pct", 0.0))
    span = text_y2 - text_y1
    if span >= 4.0:
        r["connector_type"] = "brace"
        r.pop("connector_cy_pct", None)
    else:
        r["connector_type"] = "arrow"
        r["connector_cy_pct"] = round((text_y1 + text_y2) / 2.0, 2)
    r["connector_x_pct"] = 73.0 if x1 >= 70.0 else 23.0


def deoverlap_item_remarks(remarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-spread all remarks for one item after enrichment to eliminate any new overlaps.

    Groups by page, then runs ``spread_remarks_two_column`` on each page's complete
    remark set. Inline gap remarks (x2_pct < 74%) bypass spread — they already have a
    precise full-width placement inside a handwriting gap. After spread, every remark's
    connector fields are recalculated to match the final column assignment.
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for r in remarks:
        pg = int(r.get("page", 0))
        by_page.setdefault(pg, []).append(r)

    result: list[dict[str, Any]] = []
    for pg in sorted(by_page.keys()):
        inline = [r for r in by_page[pg] if float(r.get("x2_pct", 96)) < 74.0]
        margin = [r for r in by_page[pg] if float(r.get("x2_pct", 96)) >= 74.0]
        spread = spread_remarks_two_column(margin, page_num=pg)
        for r in spread:
            recalc_connector(r)
        result.extend(spread)
        result.extend(inline)
    return result


# --- Anchor mark validation -----------------------------------------------------------


def normalize_anchor_marks(raw_anchors: Any) -> list[dict[str, Any]]:
    """Coerce raw anchor mark dicts into the canonical grid-space shape.

    - Type whitelist (``ellipse``, ``underline``, ``tick``).
    - Underline: ``ry`` forced to 0, ``rx`` clamped to fit within content columns.
    - Ellipse: ``cx`` clamped to content columns.
    - Tick: ``cx`` clamped to content columns, radii unchanged.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(raw_anchors, list):
        return out
    for a in raw_anchors:
        if not isinstance(a, dict):
            continue
        t = str(a.get("type", "")).strip().lower()
        if t not in ("ellipse", "underline", "tick"):
            continue
        try:
            cx = float(a.get("cx", 25))
            cy = float(a.get("cy", 25))
            rx = float(a.get("rx", 2))
            ry = float(a.get("ry", 2))
            if t == "underline":
                ry = 0.0
                rx = min(rx, cx - CONTENT_X_MIN, CONTENT_X_MAX - cx)
                rx = max(0.5, rx)
                cx = max(CONTENT_X_MIN + rx, min(CONTENT_X_MAX - rx, cx))
            elif t in ("circle", "ellipse"):
                cx = max(CONTENT_X_MIN + rx, min(CONTENT_X_MAX - rx, cx))
            else:  # tick
                cx = max(CONTENT_X_MIN, min(CONTENT_X_MAX, cx))
            out.append({"type": t, "cx": cx, "cy": cy, "rx": rx, "ry": ry})
        except (TypeError, ValueError):
            pass
    return out


def normalize_remark_boxes(raw_remarks: Any) -> list[dict[str, Any]]:
    """Coerce raw remark-box dicts into clamped grid-space rectangles with min size."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw_remarks, list):
        return out
    for r in raw_remarks:
        if not isinstance(r, dict):
            continue
        try:
            _gs = 50.0
            x1 = max(1.0, min(_gs, float(r.get("x1", 5))))
            y1 = max(1.0, min(_gs, float(r.get("y1", 80))))
            x2 = max(1.0, min(_gs, float(r.get("x2", 45))))
            y2 = max(1.0, min(_gs, float(r.get("y2", 90))))
            if x2 - x1 < 6:
                x2 = min(_gs, x1 + 6)
            if y2 - y1 < 3:
                y2 = min(_gs, y1 + 3)
            out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "comment": str(r.get("comment", ""))})
        except (TypeError, ValueError):
            pass
    return out


# --- Grid → percent conversion (with connector synthesis) -----------------------------


def annotations_grid_to_pct(
    anchor_marks: list[dict[str, Any]],
    remarks: list[dict[str, Any]],
    page: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert grid coords (1-50) to percentage coords (0-100) and attach page number."""
    out_anchors: list[dict[str, Any]] = []
    for a in anchor_marks:
        cx_pct, cy_pct = grid_to_pct(a["cx"], a["cy"])
        rx_pct, _ = grid_to_pct(a["rx"] + 1, 1)   # radius: shift from origin
        _, ry_pct = grid_to_pct(1, a["ry"] + 1)
        out_anchors.append({
            "type": a["type"],
            "page": page,
            "cx_pct": round(cx_pct, 2),
            "cy_pct": round(cy_pct, 2),
            "rx_pct": round(rx_pct, 2),
            "ry_pct": round(ry_pct, 2),
        })

    out_remarks: list[dict[str, Any]] = []
    for r in remarks:
        x1_pct, y1_pct = grid_to_pct(r["x1"], r["y1"])
        x2_pct, y2_pct = grid_to_pct(r["x2"], r["y2"])

        tg1 = r.get("_text_y1_grid", r["y1"])
        tg2 = r.get("_text_y2_grid", r["y2"])
        _, text_y1_pct = grid_to_pct(1, tg1)
        _, text_y2_pct = grid_to_pct(1, tg2)
        cluster_size = r.get("_cluster_size", 0)
        span_pct = text_y2_pct - text_y1_pct

        # Inline gap remark spans the content area (not a margin) — no connector needed.
        is_inline = x2_pct < 74.0
        conn_cy: float | None = None
        if is_inline:
            conn_type = "none"
            is_right = False
        elif cluster_size >= 2 or span_pct >= 4.0:
            conn_type = "brace"
            is_right = x1_pct >= 70.0
        else:
            conn_type = "arrow"
            conn_cy = round((text_y1_pct + text_y2_pct) / 2.0, 2)
            is_right = x1_pct >= 70.0
        conn_x = 73.0 if is_right else 23.0

        rem_out: dict[str, Any] = {
            "page": page,
            "x1_pct": round(x1_pct, 2),
            "y1_pct": round(y1_pct, 2),
            "x2_pct": round(x2_pct, 2),
            "y2_pct": round(y2_pct, 2),
            "comment": r.get("comment", ""),
            "text_y1_pct": round(text_y1_pct, 2),
            "text_y2_pct": round(text_y2_pct, 2),
            "connector_type": conn_type,
            "connector_x_pct": conn_x,
        }
        if conn_cy is not None:
            rem_out["connector_cy_pct"] = conn_cy
        out_remarks.append(rem_out)

    return out_anchors, out_remarks


# --- Clipping marks to an item's answer range -----------------------------------------


def clip_marks_to_answer(
    marks: list[dict[str, Any]],
    start_page: int,
    end_page: int,
    start_y: float,
    end_y: float,
    y_key: str = "cy_pct",
) -> list[dict[str, Any]]:
    """Keep only marks whose page+y fall within the student's answer range.

    Y-axis semantics depend on ``y_key``:
      - ``cy_pct`` (anchor marks): compare the mark's center y directly.
      - any other value (remarks): compare the *center* of the [y1_pct, y2_pct] box.
    """
    result: list[dict[str, Any]] = []
    for m in marks:
        pg = m.get("page", 0)
        if pg < start_page or pg > end_page:
            continue
        if start_page < pg < end_page:
            result.append(m)
            continue
        if y_key == "cy_pct":
            y = float(m.get("cy_pct", 50.0))
        else:
            y1_r = float(m.get("y1_pct", 0.0))
            y2_r = float(m.get("y2_pct", y1_r))
            y = (y1_r + y2_r) / 2.0
        if pg == start_page and pg == end_page:
            if start_y <= y <= end_y:
                result.append(m)
        elif pg == start_page:
            if y >= start_y:
                result.append(m)
        else:
            if y <= end_y:
                result.append(m)
    return result


def nudge_mark_past_remarks(item: dict[str, Any]) -> None:
    """If the score mark at marking_y overlaps a right-margin remark, push it below."""
    my = float(item.get("marking_y_position_percent", 0.0))
    mp = int(item.get("marking_page", item.get("start_page", 1)))
    remarks = item.get("remarks") or []
    right = sorted(
        [r for r in remarks
         if r.get("page") == mp and float(r.get("x1_pct", 0)) >= 70.0],
        key=lambda r: r["y1_pct"],
    )
    for r in right:
        y1, y2 = float(r["y1_pct"]), float(r["y2_pct"])
        if y1 <= my < y2:
            my = y2 + 1.0
    item["marking_y_position_percent"] = round(min(my, 95.0), 2)
