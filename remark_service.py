"""Pure business-logic service for PDF remark coordinate extraction."""

from __future__ import annotations

import base64
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import fitz  # PyMuPDF
import numpy as np
from scipy.ndimage import gaussian_filter

BASE_PDF_RELATIVE_PATH    = "test_original_pdf2.pdf"
_DEFAULT_DPI              = 150
_DEFAULT_TOP_N            = 10
_DEFAULT_CLEARANCE_PT     = 20.0   # gaussian sigma in PDF points (20pt ≈ 7mm clearance)
_DEFAULT_PROXIMITY_THRESH = 0.04   # max allowed proximity (0=totally clear, 1=on ink)
_DEFAULT_SAMPLE_ROWS      = 30     # grid rows for candidate sampling
_DEFAULT_SAMPLE_COLS      = 15     # grid cols for candidate sampling
_DEFAULT_MIN_SPACING_PCT  = 12.0   # minimum spacing between selected points (% of content)
_LEFT_MARGIN_FRACTION     = 0.18   # skip spiral binding / hole-punch strip
_RIGHT_MARGIN_FRACTION    = 0.97   # skip right bleed
_TOP_MARGIN_FRACTION      = 0.04   # skip top header
_BOTTOM_MARGIN_FRACTION   = 0.93   # skip footer


@dataclass
class Remark:
    key: str
    text: str
    category: Optional[str] = None
    color: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Placement:
    key: str
    text: str
    page: int
    x: int
    y: int
    canvas_width: int
    canvas_height: int
    pct_x: float
    pct_y: float
    pdf_x: float
    pdf_y: float
    category: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Session:
    session_id: str
    filename: str
    pdf_bytes: bytes
    remarks: list[Remark]
    total_pages: int
    page_sizes: list[dict[str, float]]
    dpi: int
    placements: dict[str, Placement] = field(default_factory=dict)


@dataclass
class FreeSpaceBBox:
    page: int
    score: float
    x_start_percent: float
    x_end_percent: float
    y_start_percent: float
    y_end_percent: float
    pdf_x1: float
    pdf_y1: float
    pdf_x2: float
    pdf_y2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageFreeSpaceResult:
    page: int
    bboxes: list[FreeSpaceBBox]

    def to_dict(self) -> dict[str, Any]:
        return {"page": self.page, "bboxes": [b.to_dict() for b in self.bboxes]}


@dataclass
class _FreeZone:
    x_start_percent: float
    x_end_percent: float
    y_start_percent: float
    y_end_percent: float
    score: float
    row: int   # kept for backward compat; always 0 in new algorithm
    col: int   # kept for backward compat; always 0 in new algorithm


class SessionNotFound(KeyError):
    """Raised when a session_id is not found in the store."""


class RemarkNotFound(KeyError):
    """Raised when a remark key does not exist in the session."""


class PageOutOfRange(ValueError):
    """Raised when a page number is outside the valid range."""


class InvalidRemarksFormat(ValueError):
    """Raised when the remarks payload cannot be normalised."""


class SessionStore:
    """Simple in-memory session store."""

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def create(
        self,
        pdf_bytes: bytes,
        remarks_raw: Any,
        filename: str = "upload.pdf",
        dpi: int = 120,
    ) -> str:
        remarks = _parse_remarks(remarks_raw) if remarks_raw else []
        total_pages, page_sizes = _pdf_meta(pdf_bytes)
        sid = str(uuid.uuid4())
        self._store[sid] = Session(
            session_id=sid,
            filename=filename,
            pdf_bytes=pdf_bytes,
            remarks=remarks,
            total_pages=total_pages,
            page_sizes=page_sizes,
            dpi=max(72, min(300, dpi)),
        )
        return sid

    def get(self, session_id: str) -> Session:
        try:
            return self._store[session_id]
        except KeyError as exc:
            raise SessionNotFound(session_id) from exc

    def delete(self, session_id: str) -> None:
        if session_id not in self._store:
            raise SessionNotFound(session_id)
        del self._store[session_id]

    def list_all(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "filename": s.filename,
                "total_pages": s.total_pages,
                "remarks": len(s.remarks),
                "placed": len(s.placements),
            }
            for s in self._store.values()
        ]


def load_base_pdf_bytes(base_pdf_relative_path: str = BASE_PDF_RELATIVE_PATH) -> bytes:
    """Load base PDF bytes from a configurable path relative to cwd."""
    path = Path(base_pdf_relative_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.read_bytes()


class RemarkService:
    """Stateless remark placement service."""

    def render_page(
        self,
        pdf_bytes: bytes,
        page: int,
        dpi: int = 120,
    ) -> tuple[str, int, int]:
        _assert_positive_page(page)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        _assert_page_in_range(page, doc.page_count)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page - 1].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode(), pix.width, pix.height

    def render_page_stream(
        self,
        pdf_bytes: bytes,
        page: int,
        dpi: int = 120,
    ) -> tuple[bytes, int, int]:
        _assert_positive_page(page)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        _assert_page_in_range(page, doc.page_count)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page - 1].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes, pix.width, pix.height

    def place(
        self,
        store: SessionStore,
        session_id: str,
        *,
        page: int,
        key: str,
        x: int,
        y: int,
        canvas_width: int,
        canvas_height: int,
    ) -> Placement:
        s = store.get(session_id)
        _assert_page_in_range(page, s.total_pages)
        remark = _find_remark(s.remarks, key)

        pdf_x, pdf_y = _canvas_to_pdf(x, y, canvas_width, canvas_height, s.pdf_bytes, page)
        pct_x = round((x / canvas_width) * 100, 2)
        pct_y = round((y / canvas_height) * 100, 2)

        placement = Placement(
            key=key,
            text=remark.text,
            page=page,
            x=x,
            y=y,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            pct_x=pct_x,
            pct_y=pct_y,
            pdf_x=pdf_x,
            pdf_y=pdf_y,
            category=remark.category,
        )
        s.placements[key] = placement
        return placement

    def remove_placement(self, store: SessionStore, session_id: str, key: str) -> None:
        s = store.get(session_id)
        if key not in s.placements:
            raise RemarkNotFound(key)
        del s.placements[key]

    def auto_place(
        self,
        store: SessionStore,
        session_id: str,
        *,
        page: int = 1,
        strategy: Literal["grid", "column", "row", "free_space"] = "free_space",
        only_unplaced: bool = True,
        use_base_pdf_for_free_space: bool = True,
        base_pdf_relative_path: str = BASE_PDF_RELATIVE_PATH,
    ) -> list[Placement]:
        s = store.get(session_id)
        _assert_page_in_range(page, s.total_pages)

        candidates = (
            [r for r in s.remarks if r.key not in s.placements]
            if only_unplaced
            else s.remarks
        )
        if not candidates:
            return []

        # Compute canvas dimensions from cached page_sizes + dpi
        # (eliminates a redundant full-page rasterize just to get width/height)
        w_pts = s.page_sizes[page - 1]["width"]
        h_pts = s.page_sizes[page - 1]["height"]
        cw    = round(w_pts * s.dpi / 72)
        ch    = round(h_pts * s.dpi / 72)

        if strategy == "free_space":
            target_pdf_bytes = s.pdf_bytes
            try:
                if use_base_pdf_for_free_space:
                    target_pdf_bytes = load_base_pdf_bytes(base_pdf_relative_path)
                pages_result = self.detect_free_space_bboxes(target_pdf_bytes)
                positions = _positions_from_bbox_pages(pages_result, page, len(candidates), cw, ch)
            except Exception:
                positions = []
            if len(positions) < len(candidates):
                fallback_positions = _compute_positions("grid", len(candidates), cw, ch)
                positions = _merge_unique_positions(positions, fallback_positions, len(candidates))
        else:
            positions = _compute_positions(strategy, len(candidates), cw, ch)

        placed: list[Placement] = []
        for remark, (x, y) in zip(candidates, positions):
            p = self.place(
                store,
                session_id,
                page=page,
                key=remark.key,
                x=x,
                y=y,
                canvas_width=cw,
                canvas_height=ch,
            )
            placed.append(p)
        return placed

    def export(
        self,
        store: SessionStore,
        session_id: str,
        fmt: Literal["full", "minimal", "pdf", "normalized"] = "full",
    ) -> dict[str, Any]:
        s = store.get(session_id)

        def _shape(v: Placement) -> dict[str, Any]:
            if fmt == "minimal":
                return {"page": v.page, "x": v.x, "y": v.y, "pct_x": v.pct_x, "pct_y": v.pct_y}
            if fmt == "pdf":
                return {"page": v.page, "pdf_x": v.pdf_x, "pdf_y": v.pdf_y, "text": v.text}
            if fmt == "normalized":
                return {"page": v.page, "pct_x": v.pct_x, "pct_y": v.pct_y, "text": v.text}
            return v.to_dict()

        return {
            "session_id": session_id,
            "filename": s.filename,
            "format": fmt,
            "total_placed": len(s.placements),
            "unplaced_keys": [r.key for r in s.remarks if r.key not in s.placements],
            "coordinates": {k: _shape(v) for k, v in s.placements.items()},
        }

    def session_summary(self, store: SessionStore, session_id: str) -> dict[str, Any]:
        s = store.get(session_id)
        return {
            "session_id": session_id,
            "filename": s.filename,
            "total_pages": s.total_pages,
            "page_sizes": s.page_sizes,
            "dpi": s.dpi,
            "remarks": [r.to_dict() for r in s.remarks],
            "placements_count": len(s.placements),
            "placed_keys": list(s.placements.keys()),
            "unplaced_keys": [r.key for r in s.remarks if r.key not in s.placements],
        }

    def detect_free_space_bboxes(
        self,
        pdf_bytes: bytes,
        *,
        top_n: int              = _DEFAULT_TOP_N,
        dpi: int                = _DEFAULT_DPI,
        clearance_pt: float     = _DEFAULT_CLEARANCE_PT,
        proximity_thresh: float = _DEFAULT_PROXIMITY_THRESH,
        sample_rows: int        = _DEFAULT_SAMPLE_ROWS,
        sample_cols: int        = _DEFAULT_SAMPLE_COLS,
        min_spacing_pct: float  = _DEFAULT_MIN_SPACING_PCT,
        left_margin: float      = _LEFT_MARGIN_FRACTION,
        right_margin: float     = _RIGHT_MARGIN_FRACTION,
        top_margin: float       = _TOP_MARGIN_FRACTION,
        bottom_margin: float    = _BOTTOM_MARGIN_FRACTION,
        # Legacy params accepted for backward compat but ignored:
        rows: int = 40,
        cols: int = 12,
        min_score: float = 0.92,
        merge: bool = True,
    ) -> list[PageFreeSpaceResult]:
        total_pages, page_sizes = _pdf_meta(pdf_bytes)
        zones_per_page = analyze_pdf_free_space(
            pdf_bytes,
            top_n=top_n,
            dpi=dpi,
            clearance_pt=clearance_pt,
            proximity_thresh=proximity_thresh,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
            min_spacing_pct=min_spacing_pct,
            left_margin=left_margin,
            right_margin=right_margin,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
        )
        pages: list[PageFreeSpaceResult] = []
        for page_idx in range(total_pages):
            size = page_sizes[page_idx]
            page_no = page_idx + 1
            page_boxes = [
                _zone_to_bbox(
                    zone,
                    page_no,
                    size["width"],
                    size["height"],
                    left_margin=left_margin,
                    right_margin=right_margin,
                    top_margin=top_margin,
                    bottom_margin=bottom_margin,
                )
                for zone in zones_per_page[page_idx]
            ]
            pages.append(PageFreeSpaceResult(page=page_no, bboxes=page_boxes))
        return pages


def _parse_remarks(raw: Any) -> list[Remark]:
    if isinstance(raw, str):
        raw = json.loads(raw)

    if isinstance(raw, dict):
        out: list[Remark] = []
        for k, v in raw.items():
            if isinstance(v, str):
                out.append(Remark(key=str(k), text=v))
            elif isinstance(v, dict):
                out.append(
                    Remark(
                        key=str(k),
                        text=v.get("text") or v.get("remark") or str(v),
                        category=v.get("category"),
                        color=v.get("color"),
                    )
                )
        return out

    if isinstance(raw, list):
        out = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                out.append(Remark(key=str(i + 1), text=item))
            elif isinstance(item, dict):
                key = item.get("key") or item.get("id") or item.get("label") or str(i + 1)
                text = item.get("text") or item.get("remark") or item.get("comment") or str(item)
                out.append(
                    Remark(
                        key=str(key),
                        text=str(text),
                        category=item.get("category"),
                        color=item.get("color"),
                    )
                )
        return out

    raise InvalidRemarksFormat(f"Cannot parse remarks from type {type(raw).__name__}")


def _pdf_meta(pdf_bytes: bytes) -> tuple[int, list[dict[str, float]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = doc.page_count
    sizes = [{"width": round(p.rect.width, 2), "height": round(p.rect.height, 2)} for p in doc]
    doc.close()
    return total, sizes


def _canvas_to_pdf(
    x: int,
    y: int,
    cw: int,
    ch: int,
    pdf_bytes: bytes,
    page: int,
) -> tuple[float, float]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rect = doc[page - 1].rect
    doc.close()
    pdf_x = round((x / cw) * rect.width, 2)
    pdf_y = round(rect.height - (y / ch) * rect.height, 2)
    return pdf_x, pdf_y


def _compute_positions(strategy: str, n: int, cw: int, ch: int) -> list[tuple[int, int]]:
    if strategy == "column":
        x = int(cw * 0.85)
        return [(x, int(ch * (i + 1) / (n + 1))) for i in range(n)]
    if strategy == "row":
        y = int(ch * 0.08)
        return [(int(cw * (i + 1) / (n + 1)), y) for i in range(n)]

    cols = max(1, round(math.sqrt(n)))
    rows = math.ceil(n / cols)
    return [
        (
            int(cw * (i % cols + 1) / (cols + 1)),
            int(ch * (i // cols + 1) / (rows + 1)),
        )
        for i in range(n)
    ]


def _positions_from_bbox_pages(
    pages: list[PageFreeSpaceResult],
    page: int,
    n: int,
    cw: int,
    ch: int,
    left_margin: float = _LEFT_MARGIN_FRACTION,
    right_margin: float = _RIGHT_MARGIN_FRACTION,
    top_margin: float = _TOP_MARGIN_FRACTION,
    bottom_margin: float = _BOTTOM_MARGIN_FRACTION,
) -> list[tuple[int, int]]:
    if page < 1 or page > len(pages):
        return []
    page_boxes = pages[page - 1].bboxes
    if not page_boxes:
        return []

    cx0 = int(cw * left_margin)
    cx1 = int(cw * right_margin)
    cy0 = int(ch * top_margin)
    cy1 = int(ch * bottom_margin)
    content_cw = max(1, cx1 - cx0)
    content_ch = max(1, cy1 - cy0)

    sorted_boxes = sorted(
        page_boxes,
        key=lambda b: -(
            (b.x_end_percent - b.x_start_percent)
            * (b.y_end_percent - b.y_start_percent)
            * b.score
        ),
    )
    positions: list[tuple[int, int]] = []
    for box in sorted_boxes:
        if len(positions) >= n:
            break
        x_pct = (box.x_start_percent + box.x_end_percent) / 2
        y_pct = (box.y_start_percent + box.y_end_percent) / 2
        canvas_x = cx0 + int(content_cw * x_pct / 100)
        canvas_y = cy0 + int(content_ch * y_pct / 100)
        positions.append((canvas_x, canvas_y))
    return positions


def _zone_to_bbox(
    zone: Any,
    page: int,
    page_width: float,
    page_height: float,
    left_margin: float = _LEFT_MARGIN_FRACTION,
    right_margin: float = _RIGHT_MARGIN_FRACTION,
    top_margin: float = _TOP_MARGIN_FRACTION,
    bottom_margin: float = _BOTTOM_MARGIN_FRACTION,
) -> FreeSpaceBBox:
    pdf_x1, pdf_x2, pdf_y1, pdf_y2 = _percent_to_pdf_bbox(
        zone.x_start_percent,
        zone.x_end_percent,
        zone.y_start_percent,
        zone.y_end_percent,
        page_width,
        page_height,
        left_margin,
        right_margin,
        top_margin,
        bottom_margin,
    )
    return FreeSpaceBBox(
        page=page,
        score=round(float(zone.score), 4),
        x_start_percent=round(float(zone.x_start_percent), 2),
        x_end_percent=round(float(zone.x_end_percent), 2),
        y_start_percent=round(float(zone.y_start_percent), 2),
        y_end_percent=round(float(zone.y_end_percent), 2),
        pdf_x1=pdf_x1,
        pdf_y1=pdf_y1,
        pdf_x2=pdf_x2,
        pdf_y2=pdf_y2,
    )


def _percent_to_pdf_bbox(
    x_start_percent: float,
    x_end_percent: float,
    y_start_percent: float,
    y_end_percent: float,
    page_width: float,
    page_height: float,
    left_margin: float = _LEFT_MARGIN_FRACTION,
    right_margin: float = _RIGHT_MARGIN_FRACTION,
    top_margin: float = _TOP_MARGIN_FRACTION,
    bottom_margin: float = _BOTTOM_MARGIN_FRACTION,
) -> tuple[float, float, float, float]:
    content_x0 = page_width * left_margin
    content_x1 = page_width * right_margin
    content_y0 = page_height * top_margin
    content_y1 = page_height * bottom_margin
    content_w = content_x1 - content_x0
    content_h = content_y1 - content_y0

    abs_x1 = content_x0 + (x_start_percent / 100.0) * content_w
    abs_x2 = content_x0 + (x_end_percent / 100.0) * content_w
    abs_y_top = page_height - (content_y0 + (y_start_percent / 100.0) * content_h)
    abs_y_bottom = page_height - (content_y0 + (y_end_percent / 100.0) * content_h)

    pdf_y1 = round(min(abs_y_top, abs_y_bottom), 2)
    pdf_y2 = round(max(abs_y_top, abs_y_bottom), 2)
    return round(abs_x1, 2), round(abs_x2, 2), pdf_y1, pdf_y2


def _otsu_binarise(gray_u8: np.ndarray) -> np.ndarray:
    """
    Return boolean mask: True = background (light), False = ink (dark).

    Numpy-only Otsu threshold. Replaces _otsu_background_mask; the dilation
    step has been removed — experiments showed it reduced paragraph-gap score
    from 0.999 → 0.909, making genuine blank zones indistinguishable from
    text rows. The Gaussian blur in _ink_proximity_map provides clearance instead.
    """
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
    total    = gray_u8.size
    sum_all  = float(np.dot(np.arange(256), hist))
    sum_b    = 0.0
    w_b      = 0.0
    max_var  = -1.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b  += t * hist[t]
        mean_b  = sum_b / w_b
        mean_f  = (sum_all - sum_b) / w_f
        between = w_b * w_f * (mean_b - mean_f) ** 2
        if between > max_var:
            max_var   = between
            threshold = t
    return gray_u8 > threshold   # True = background


def _rasterize_page_to_gray_u8(
    pdf_bytes: bytes,
    page_index: int,
    dpi: int = _DEFAULT_DPI,
) -> np.ndarray:
    """
    Rasterise one PDF page to a uint8 grayscale numpy array (shape: height × width).

    Single-page design avoids holding all pages in memory simultaneously.
    try/finally guarantees doc.close() even on exception, fixing the resource
    leak present in the old _rasterize_pdf_to_gray_arrays.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = doc[page_index].get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    finally:
        doc.close()


def _ink_proximity_map(
    ink_mask: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """
    Compute per-pixel proximity to nearest ink via Gaussian blur.

    Returns float32 array: 0.0 = far from ink (safe), ~1.0 = on or adjacent to ink.
    sigma_px = clearance_pt × dpi / 72.  At 20pt and 150 DPI, sigma ≈ 41 px ≈ 7 mm.

    Gaussian blur creates a soft gradient so the threshold is robust to exact
    choice of value. Hard dilation (old approach) created a binary boundary that
    experimentally shrank genuine blank zones into the text-score range.
    """
    return gaussian_filter(ink_mask.astype(np.float32), sigma=sigma_px)


def _sample_proximity_grid(
    proximity: np.ndarray,
    sample_rows: int,
    sample_cols: int,
) -> list[tuple[float, float, float]]:
    """
    Sample the proximity map on a regular grid.

    Returns list of (proximity_value, y_pct, x_pct) tuples (unsorted).
    y_pct and x_pct are cell centres as percentages of the content area (0–100).
    """
    ch, cw = proximity.shape
    candidates: list[tuple[float, float, float]] = []
    for r in range(sample_rows):
        py = int(ch * (r + 0.5) / sample_rows)
        y_pct = (r + 0.5) / sample_rows * 100.0
        for c in range(sample_cols):
            px = int(cw * (c + 0.5) / sample_cols)
            x_pct = (c + 0.5) / sample_cols * 100.0
            candidates.append((float(proximity[py, px]), y_pct, x_pct))
    return candidates


def _greedy_diverse_select(
    candidates: list[tuple[float, float, float]],
    top_n: int,
    min_spacing_pct: float,
    max_proximity: float,
) -> list[tuple[float, float, float]]:
    """
    Select up to top_n candidates that are (a) below max_proximity and
    (b) at least min_spacing_pct apart from each other (Euclidean in % space).

    Sort ascending by proximity (best first); accept each candidate only if it
    is farther than min_spacing_pct from every already-accepted candidate.
    Early-exit on the first candidate that exceeds max_proximity (O(N × k)).
    """
    sorted_candidates = sorted(candidates, key=lambda t: t[0])
    selected: list[tuple[float, float, float]] = []

    for prox, y_pct, x_pct in sorted_candidates:
        if prox > max_proximity:
            break   # sorted: all remaining candidates are worse
        too_close = any(
            ((y_pct - sy) ** 2 + (x_pct - sx) ** 2) ** 0.5 < min_spacing_pct
            for _, sy, sx in selected
        )
        if not too_close:
            selected.append((prox, y_pct, x_pct))
        if len(selected) >= top_n:
            break

    return selected


def _candidates_to_free_zones(
    selected: list[tuple[float, float, float]],
    sample_rows: int,
    sample_cols: int,
) -> list[_FreeZone]:
    """
    Convert greedy-selected anchor points to _FreeZone objects.

    Each anchor is expanded to a bounding box the size of its grid cell.
    score = 1.0 - proximity (higher = better, matching the old convention).
    row and col are set to 0; they were internal merge bookkeeping in the old
    algorithm and carry no meaning in the new one.
    """
    cell_h_pct = 100.0 / sample_rows
    cell_w_pct = 100.0 / sample_cols
    zones: list[_FreeZone] = []

    for prox, y_pct, x_pct in selected:
        zones.append(
            _FreeZone(
                x_start_percent=round(max(0.0,   x_pct - cell_w_pct / 2), 2),
                x_end_percent  =round(min(100.0, x_pct + cell_w_pct / 2), 2),
                y_start_percent=round(max(0.0,   y_pct - cell_h_pct / 2), 2),
                y_end_percent  =round(min(100.0, y_pct + cell_h_pct / 2), 2),
                score          =round(1.0 - prox, 4),
                row            =0,
                col            =0,
            )
        )

    return zones


def analyze_pdf_free_space(
    pdf_bytes: bytes,
    *,
    top_n: int              = _DEFAULT_TOP_N,
    dpi: int                = _DEFAULT_DPI,
    clearance_pt: float     = _DEFAULT_CLEARANCE_PT,
    proximity_thresh: float = _DEFAULT_PROXIMITY_THRESH,
    sample_rows: int        = _DEFAULT_SAMPLE_ROWS,
    sample_cols: int        = _DEFAULT_SAMPLE_COLS,
    min_spacing_pct: float  = _DEFAULT_MIN_SPACING_PCT,
    left_margin: float      = _LEFT_MARGIN_FRACTION,
    right_margin: float     = _RIGHT_MARGIN_FRACTION,
    top_margin: float       = _TOP_MARGIN_FRACTION,
    bottom_margin: float    = _BOTTOM_MARGIN_FRACTION,
    # Legacy parameters kept for backwards compatibility but ignored:
    rows: int        = 40,
    cols: int        = 12,
    min_score: float = 0.92,
    merge: bool      = True,
) -> list[list[_FreeZone]]:
    """
    Detect free-space anchor zones in every page of a PDF.

    Algorithm (per page):
      1. Rasterise to uint8 grayscale at `dpi` resolution.
      2. Otsu-binarise → boolean ink mask (True=background, False=ink).
      3. Gaussian-blur the ink mask with sigma = clearance_pt × dpi / 72.
         Produces a proximity-to-ink map (0=clear, ~1=on ink).
      4. Clip to the content area defined by the margin fractions.
      5. Sample the proximity map on a sample_rows × sample_cols grid.
      6. Greedy-diverse selection: accept candidates with proximity below
         proximity_thresh, enforcing min_spacing_pct between selections.
      7. Convert selected anchor points to _FreeZone objects.

    The old `rows`, `cols`, `min_score`, `merge` parameters are accepted for
    backwards compatibility but have no effect. Remove from call sites when
    convenient.
    """
    total_pages, _ = _pdf_meta(pdf_bytes)
    sigma_px = (dpi / 72.0) * clearance_pt
    page_zones: list[list[_FreeZone]] = []

    for page_index in range(total_pages):
        # Step 1: rasterise
        gray_u8 = _rasterize_page_to_gray_u8(pdf_bytes, page_index, dpi)
        h, w = gray_u8.shape

        # Step 2: binarise
        ink_bool = ~_otsu_binarise(gray_u8)   # True = ink

        # Step 3: proximity map
        proximity_full = _ink_proximity_map(ink_bool, sigma_px)

        # Step 4: clip to content area
        x0 = int(w * left_margin);  x1 = int(w * right_margin)
        y0 = int(h * top_margin);   y1 = int(h * bottom_margin)
        proximity_content = proximity_full[y0:y1, x0:x1]

        # Step 5: sample
        candidates = _sample_proximity_grid(proximity_content, sample_rows, sample_cols)

        # Step 6: greedy diverse selection
        selected = _greedy_diverse_select(
            candidates,
            top_n=top_n,
            min_spacing_pct=min_spacing_pct,
            max_proximity=proximity_thresh,
        )

        # Step 7: convert
        zones = _candidates_to_free_zones(selected, sample_rows, sample_cols)
        page_zones.append(zones)

    return page_zones


def _merge_unique_positions(
    primary: list[tuple[int, int]],
    secondary: list[tuple[int, int]],
    n: int,
) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    merged: list[tuple[int, int]] = []
    for pos in primary + secondary:
        if pos in seen:
            continue
        seen.add(pos)
        merged.append(pos)
        if len(merged) >= n:
            break
    return merged


def _find_remark(remarks: list[Remark], key: str) -> Remark:
    for r in remarks:
        if r.key == key:
            return r
    raise RemarkNotFound(key)


def _assert_positive_page(page: int) -> None:
    if page < 1:
        raise PageOutOfRange(f"Page must be ≥ 1, got {page}")


def _assert_page_in_range(page: int, total: int) -> None:
    _assert_positive_page(page)
    if page > total:
        raise PageOutOfRange(f"Page {page} out of range (1–{total})")
