"""Pure-function utilities for pixel-level free-space detection on exam page images.

No I/O, no Gemini, no FastAPI. Accepts numpy arrays, returns typed data.

Score formula: score = mean_brightness * (1 - min(1, std_brightness * 6))
  - High mean  → mostly white background → good for annotation
  - Low std    → uniform (no ink variation) → good for annotation
  - Score 1.0  = pure white cell
  - Score ~0.0 = dense handwriting
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FreeZone:
    """Rectangular region (percent-coordinates) with an emptiness score."""

    x_start_percent: float
    x_end_percent: float
    y_start_percent: float
    y_end_percent: float
    score: float
    row: int
    col: int

    def y_center(self) -> float:
        return (self.y_start_percent + self.y_end_percent) / 2.0

    def x_center(self) -> float:
        return (self.x_start_percent + self.x_end_percent) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_start_percent": self.x_start_percent,
            "x_end_percent": self.x_end_percent,
            "y_start_percent": self.y_start_percent,
            "y_end_percent": self.y_end_percent,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def score_cell(cell: np.ndarray) -> float:
    """Return 0.0–1.0 emptiness score for a grayscale cell (0=black, 1=white).

    Scores near 1.0 = clean white paper, safe to annotate.
    Scores near 0.0 = dense handwriting / drawing.
    """
    if cell.size == 0:
        return 0.0
    mean = float(np.mean(cell))
    std = float(np.std(cell))
    std_factor = min(1.0, std * 6.0)
    return max(0.0, min(1.0, mean * (1.0 - std_factor)))


def get_grid_scores(gray: np.ndarray, rows: int, cols: int) -> list[list[float]]:
    """Score every cell in a rows×cols grid over a 2-D float32 grayscale array."""
    h, w = gray.shape
    result: list[list[float]] = []
    for r in range(rows):
        y0 = r * h // rows
        y1 = (r + 1) * h // rows if r < rows - 1 else h
        row_scores: list[float] = []
        for c in range(cols):
            x0 = c * w // cols
            x1 = (c + 1) * w // cols if c < cols - 1 else w
            row_scores.append(score_cell(gray[y0:y1, x0:x1]))
        result.append(row_scores)
    return result


# ---------------------------------------------------------------------------
# Zone extraction
# ---------------------------------------------------------------------------


def find_free_zones(
    scores: list[list[float]],
    rows: int,
    cols: int,
    img_w: int,
    img_h: int,
    *,
    min_score: float = 0.65,
    top_n: int = 15,
    exclude_left_cols: int = 1,
    exclude_right_cols: int = 0,
    exclude_top_rows: int = 0,
    exclude_bottom_rows: int = 1,
) -> list[FreeZone]:
    """Return up to top_n cells with score ≥ min_score as FreeZone objects.

    Sorted by score descending, then by y_start ascending (top-first) for ties.

    Args:
        exclude_left_cols:   Skip leftmost N cols (binding strip / holes).
        exclude_right_cols:  Skip rightmost N cols.
        exclude_top_rows:    Skip topmost N rows (header).
        exclude_bottom_rows: Skip bottommost N rows (footer / contact line).
    """
    candidates: list[FreeZone] = []
    for r in range(rows):
        if r < exclude_top_rows or r >= rows - exclude_bottom_rows:
            continue
        for c in range(cols):
            if c < exclude_left_cols:
                continue
            if exclude_right_cols > 0 and c >= cols - exclude_right_cols:
                continue
            s = scores[r][c]
            if s < min_score:
                continue
            x0_pct = c * 100.0 / cols
            x1_pct = (c + 1) * 100.0 / cols
            y0_pct = r * 100.0 / rows
            y1_pct = (r + 1) * 100.0 / rows
            candidates.append(
                FreeZone(
                    x_start_percent=round(x0_pct, 2),
                    x_end_percent=round(min(x1_pct, 100.0), 2),
                    y_start_percent=round(y0_pct, 2),
                    y_end_percent=round(min(y1_pct, 100.0), 2),
                    score=round(s, 4),
                    row=r,
                    col=c,
                )
            )

    candidates.sort(key=lambda z: (-z.score, z.y_start_percent))
    return candidates[:top_n]


def merge_adjacent_zones(zones: list[FreeZone], rows: int, cols: int) -> list[FreeZone]:
    """Merge horizontally adjacent cells in the same row into wider zones.

    Only merges cells that are contiguous (same row, consecutive columns).
    Returns a new list of FreeZone objects (merged where possible).
    """
    if not zones:
        return []

    by_row: dict[int, list[FreeZone]] = {}
    for z in zones:
        by_row.setdefault(z.row, []).append(z)

    merged: list[FreeZone] = []
    for row_zones in by_row.values():
        row_zones_sorted = sorted(row_zones, key=lambda z: z.col)
        current = row_zones_sorted[0]
        for nxt in row_zones_sorted[1:]:
            if nxt.col == current.col + 1:
                # extend current zone to include nxt
                current = FreeZone(
                    x_start_percent=current.x_start_percent,
                    x_end_percent=nxt.x_end_percent,
                    y_start_percent=current.y_start_percent,
                    y_end_percent=current.y_end_percent,
                    score=round((current.score + nxt.score) / 2.0, 4),
                    row=current.row,
                    col=current.col,
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    merged.sort(key=lambda z: (-z.score, z.y_start_percent))
    return merged


# ---------------------------------------------------------------------------
# Annotation snapping
# ---------------------------------------------------------------------------


def snap_y_to_nearest_free_zone(
    y_pct: float,
    free_zones: list[FreeZone],
    *,
    max_shift_pct: float = 30.0,
) -> FreeZone | None:
    """Find the free zone whose y-center is closest to y_pct within max_shift_pct."""
    best: FreeZone | None = None
    best_dist = float("inf")
    for zone in free_zones:
        dist = abs(zone.y_center() - y_pct)
        if dist < best_dist and dist <= max_shift_pct:
            best_dist = dist
            best = zone
    return best


def snap_annotations_to_free_zones(
    annotations: list[dict[str, Any]],
    page_free_zones: list[list[FreeZone]],
    *,
    max_shift_pct: float = 30.0,
) -> list[dict[str, Any]]:
    """Adjust each annotation's y/x coordinates to the nearest free zone.

    Args:
        annotations:      LLM-generated annotation dicts with page_index,
                          y_position_percent, x_start_percent, x_end_percent.
        page_free_zones:  Per-page list of FreeZone objects (0-indexed).
        max_shift_pct:    Max y-distance (%) to accept a snap; keep original if exceeded.

    Returns a new list; original dicts are not mutated.
    """
    result: list[dict[str, Any]] = []
    for ann in annotations:
        out = dict(ann)
        page_idx = int(ann.get("page_index", 0))
        y_pct = float(ann.get("y_position_percent", 50.0))

        if page_idx < len(page_free_zones):
            zones = page_free_zones[page_idx]
            best = snap_y_to_nearest_free_zone(y_pct, zones, max_shift_pct=max_shift_pct)
            if best is not None:
                out["y_position_percent"] = round(best.y_center(), 2)
                out["x_start_percent"] = best.x_start_percent
                out["x_end_percent"] = best.x_end_percent
                out["_snapped"] = True
                out["_snap_score"] = best.score
            else:
                out["_snapped"] = False
        else:
            out["_snapped"] = False

        result.append(out)
    return result
