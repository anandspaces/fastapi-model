"""Grid overlay utilities for smart-OCR page images.

Draws a 50×50 reference grid on rasterized PDF page images so that Gemini can report
precise visual coordinates (circle, ellipse, underline, tick, remark box) using integer
grid indices instead of estimated percentages.

Grid layout
-----------
- GRID_SIZE = 50 points labeled 1..50 on both axes.
- x_step = image_width  / GRID_SIZE
- y_step = image_height / GRID_SIZE
- Grid point i (1-indexed) sits at pixel offset (i-1) * step.
  Point 1 → pixel 0 (left/top edge).
  Point 50 → pixel 49 * step = 98 % of the dimension.
- The remaining ~2 % strip at the right and bottom is the natural margin.
  All pages of the same booklet share the same pixel dimensions at a fixed DPI,
  so grid coordinates are identical across every page.

Coordinate conversion
---------------------
  grid → percent :  pct = (g - 1) / GRID_SIZE * 100
  g=1  → 0 %
  g=50 → 98 %
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

GRID_SIZE: int = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_grid_overlay(width: int, height: int) -> Image.Image:
    """Build a transparent RGBA grid layer sized width×height.

    Called once per unique (width, height) pair; results can be cached and
    composited cheaply onto any page of the same dimensions.
    """
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x_step = width / GRID_SIZE
    y_step = height / GRID_SIZE

    line_color = (140, 140, 140, 80)   # faint gray, semi-transparent

    for i in range(GRID_SIZE):
        x = round(i * x_step)
        y = round(i * y_step)
        draw.line([(x, 0), (x, height - 1)], fill=line_color, width=1)
        draw.line([(0, y), (width - 1, y)], fill=line_color, width=1)

    # Numeric labels 1–50 along the top edge (columns) and left edge (rows).
    try:
        font = ImageFont.load_default(size=max(8, min(14, int(x_step * 0.7))))
    except TypeError:
        font = ImageFont.load_default()

    label_color = (80, 80, 200, 200)   # blue-ish, readable

    for i in range(1, GRID_SIZE + 1):
        x = round((i - 1) * x_step)
        y = round((i - 1) * y_step)
        label = str(i)
        # top edge label (column numbers)
        draw.text((x + 1, 1), label, fill=label_color, font=font)
        # left edge label (row numbers)
        draw.text((1, y + 1), label, fill=label_color, font=font)

    return overlay


# Reuse the same overlay for pages with identical dimensions.
_overlay_cache: dict[tuple[int, int], Image.Image] = {}


def _get_overlay(width: int, height: int) -> Image.Image:
    key = (width, height)
    if key not in _overlay_cache:
        _overlay_cache[key] = _make_grid_overlay(width, height)
    return _overlay_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draw_grid(png_bytes: bytes) -> bytes:
    """Composite a 50×50 reference grid onto a PNG image.

    Returns PNG bytes with faint grid lines and numeric labels at every 5th
    point along the bottom and right edges. The original image content is
    preserved; the grid is rendered on a transparent overlay so handwriting
    remains fully readable.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = _get_overlay(img.width, img.height)
    composited = Image.alpha_composite(img, overlay).convert("RGB")

    buf = io.BytesIO()
    composited.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def batch_draw_grid(pages_png: Sequence[bytes], max_workers: int | None = None) -> list[bytes]:
    """Draw the grid on multiple pages in parallel using a thread pool.

    Parameters
    ----------
    pages_png:
        Sequence of PNG bytes, one entry per page.
    max_workers:
        Thread pool size. Defaults to len(pages_png).

    Returns
    -------
    list[bytes]
        Same-length list of grid-annotated PNG bytes in page order.
    """
    n = len(pages_png)
    if n == 0:
        return []
    workers = max_workers if max_workers is not None else n

    results: list[bytes] = [b""] * n

    def _job(idx: int) -> tuple[int, bytes]:
        return idx, draw_grid(pages_png[idx])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for idx, data in pool.map(_job, range(n)):
            results[idx] = data

    return results


def grid_to_pct(gx: float, gy: float) -> tuple[float, float]:
    """Convert 1..50 grid coordinates to 0..100 page-percentage coordinates.

    Grid point 1 maps to 0 %, grid point 50 maps to 98 % (= 49/50 * 100).
    The remaining ~2 % strip at the right/bottom is the natural page margin.

    Parameters
    ----------
    gx, gy:
        Horizontal and vertical grid indices in the range [1, 50]. Values
        outside this range are clamped before conversion.

    Returns
    -------
    (x_pct, y_pct) in [0.0, 100.0].
    """
    gx = max(1.0, min(float(GRID_SIZE), float(gx)))
    gy = max(1.0, min(float(GRID_SIZE), float(gy)))
    return (gx - 1.0) / GRID_SIZE * 100.0, (gy - 1.0) / GRID_SIZE * 100.0


def pct_to_grid(px: float, py: float) -> tuple[float, float]:
    """Convert 0..100 page-percentage coordinates to 1..50 grid coordinates."""
    gx = px / 100.0 * GRID_SIZE + 1.0
    gy = py / 100.0 * GRID_SIZE + 1.0
    return max(1.0, min(float(GRID_SIZE), gx)), max(1.0, min(float(GRID_SIZE), gy))


def clamp_grid(g: float) -> float:
    """Clamp a single grid coordinate to [1, GRID_SIZE]."""
    return max(1.0, min(float(GRID_SIZE), float(g)))
