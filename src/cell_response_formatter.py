"""cell_response_formatter.py — shape the ``/analyse/smart-ocr`` items payload.

Single source of truth for the response shape the frontend reads. Takes the
internal placement output (``cell_ids`` + percent fields the placer attaches)
and emits the cell-ID-native item shape:

  {
    "question_id": ..., "question": ..., "student_answer": ...,
    "is_attempted": ..., "section_name": ..., "answer_type": ...,
    "max_marks": ..., "marks_awarded": ..., "status": ..., "feedback": ...,
    "student_answer_summary": ...,

    "answer_span":  [{ "page": N, "top_cell": "...", "bottom_cell": "..." }, ...],
    "marking":      { "page": N, "score_range": "...", "score_box_range": "..." } | null,
    "annotations":  [{
        "page": N,
        "is_positive": bool,
        "comment": "...",
        "comment_font_pts": float,                 // [14.0, 16.0]
        "comment_rows": ["A5:K5", "A6:K6", ...],
        "anchor": {
          "type": "ellipse"|"tick"|"underline"|"curly_brace"|"exponent_caret"|"none",
          "rows": ["..."],                         // preferred; optional anchor.range legacy
          "extra": { ... }                         // type-specific
        }
    }]
  }

Legacy spatial fields (percent coords, page_index, bbox, placement_tier,
cell_ids array) are not echoed into the response.
"""

from __future__ import annotations

from typing import Any

from cell_grid_service_v4 import PageCellGrid, cell_id_from_rc, rc_from_cell_id

from src.gemini_smart_ocr_v2.snap import cells_from_range_strings

# Comment font is rendered in the [14, 16] pt range when annotations are
# painted onto the answer sheet. Default applies when the placer does not
# choose explicitly.
DEFAULT_COMMENT_FONT_PTS: float = 14.0

# Item-level fields preserved verbatim. No spatial fields here — those are
# rebuilt as ``answer_span`` and ``marking`` below.
_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "question_id", "question",
    "student_answer", "student_answer_summary",
    "is_attempted", "section_name", "answer_type",
    "max_marks", "marks_awarded", "status",
    "feedback",
)


# ── Cell-coordinate helpers ─────────────────────────────────────────────────


def _cell_at_xy(grid: PageCellGrid, x_pct: float, y_pct: float) -> str | None:
    cs = float(grid.cell_size_pts)
    if cs <= 0:
        return None
    x_pts = float(grid.page_w_pts) * (x_pct / 100.0)
    y_pts = float(grid.page_h_pts) * (y_pct / 100.0)
    col = int((x_pts - float(grid.left_margin_pts)) / cs) + 1
    row = int((y_pts - float(grid.top_margin_pts)) / cs) + 1
    col = max(1, min(int(grid.cols), col))
    row = max(1, min(int(grid.rows), row))
    return cell_id_from_rc(row, col)


def _cell_at_y(grid: PageCellGrid, y_pct: float, *, side: str) -> str | None:
    """``side='top'`` → leftmost column at row(y); ``side='bottom'`` → rightmost."""
    cs = float(grid.cell_size_pts)
    if cs <= 0:
        return None
    y_pts = float(grid.page_h_pts) * (y_pct / 100.0)
    row = int((y_pts - float(grid.top_margin_pts)) / cs) + 1
    row = max(1, min(int(grid.rows), row))
    col = 1 if side == "top" else int(grid.cols)
    return cell_id_from_rc(row, col)


def _comment_rows_from_bbox(r_lo: int, c_lo: int, r_hi: int, c_hi: int) -> list[str]:
    """One Excel-style token per horizontal row spanning [c_lo, c_hi]."""
    out: list[str] = []
    for row in range(r_lo, r_hi + 1):
        a = cell_id_from_rc(row, c_lo)
        b = cell_id_from_rc(row, c_hi)
        out.append(a if a == b else f"{a}:{b}")
    return out


def _bounds_from_range_id_str(range_id: str) -> tuple[int, int, int, int]:
    rid = range_id.strip()
    if ":" in rid:
        a, b = rid.split(":", 1)
    else:
        a = b = rid
    r1, c1 = rc_from_cell_id(a.strip())
    r2, c2 = rc_from_cell_id(b.strip())
    return min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2)


def _bounds_from_cell_id_list(ids: list[str]) -> tuple[int, int, int, int] | None:
    rcs: list[tuple[int, int]] = []
    for x in ids:
        try:
            rcs.append(rc_from_cell_id(str(x).strip()))
        except ValueError:
            return None
    if not rcs:
        return None
    rs = [r for r, _ in rcs]
    cs = [c for _, c in rcs]
    return min(rs), min(cs), max(rs), max(cs)


# ── Per-section builders ────────────────────────────────────────────────────


def _sanitize_row_tokens(rows: list[str], grid: PageCellGrid) -> list[str]:
    """Drop range tokens that do not resolve to any in-grid cell."""
    out: list[str] = []
    for r in rows:
        s = str(r).strip()
        if s and cells_from_range_strings(grid, [s]):
            out.append(s)
    return out


def _build_answer_span(
    item: dict[str, Any], grids_by_page: dict[int, PageCellGrid]
) -> list[dict[str, Any]]:
    """Per-page answer span. Two source shapes accepted:

    1. Gemini cell-overlay path emits ``answer_span: [{"page": N, "rows":
       ["E13:S13", "E14:T14", ...]}]`` directly — passed through as-is
       (after light validation).
    2. Legacy placer path emits ``start_page`` / ``start_y_position_percent``
       / ``end_page`` / ``end_y_position_percent`` — derived to a per-page
       top/bottom cell pair using the grid.
    """
    direct = item.get("answer_span")
    if isinstance(direct, list) and direct:
        out: list[dict[str, Any]] = []
        for span in direct:
            if not isinstance(span, dict):
                continue
            page = span.get("page")
            rows = span.get("rows")
            if isinstance(rows, list) and rows:
                try:
                    pg = int(page)
                except (TypeError, ValueError):
                    pg = 1
                g = grids_by_page.get(pg)
                row_strs = [str(r) for r in rows]
                if g is not None:
                    row_strs = _sanitize_row_tokens(row_strs, g)
                if row_strs:
                    out.append({"page": pg, "rows": row_strs})
                continue
            top_cell = span.get("top_cell")
            bot_cell = span.get("bottom_cell")
            if page is not None and top_cell and bot_cell:
                out.append({"page": int(page),
                             "top_cell": str(top_cell),
                             "bottom_cell": str(bot_cell)})
        if out:
            return out

    sp = item.get("start_page")
    ep = item.get("end_page")
    if sp is None or ep is None:
        return []
    try:
        sp_i, ep_i = int(sp), int(ep)
    except (TypeError, ValueError):
        return []
    sy = float(item.get("start_y_position_percent") or 0.0)
    ey = float(item.get("end_y_position_percent") or 100.0)
    spans: list[dict[str, Any]] = []
    for page in range(sp_i, ep_i + 1):
        grid = grids_by_page.get(page)
        if grid is None:
            continue
        # On the first page the answer ink starts at sy; on subsequent pages
        # it starts at the page top. Mirrored for end page.
        top_y = sy if page == sp_i else 0.0
        bot_y = ey if page == ep_i else 100.0
        top_cell = _cell_at_y(grid, top_y, side="top")
        bot_cell = _cell_at_y(grid, bot_y, side="bottom")
        if top_cell and bot_cell:
            spans.append({"page": page, "top_cell": top_cell, "bottom_cell": bot_cell})
    return spans


def _build_marking(
    item: dict[str, Any], grids_by_page: dict[int, PageCellGrid]
) -> dict[str, Any] | None:
    """Marking block. Two source shapes accepted:

    1. Gemini cell-overlay path emits ``marking: {page, score_range,
       score_box_range}`` — passed through as-is.
    2. Legacy placer path emits ``marking_page`` + ``marking_*_percent`` —
       derived to cell IDs using the grid.
    """
    direct = item.get("marking")
    if isinstance(direct, dict):
        page = direct.get("page")
        if page is not None:
            block: dict[str, Any] = {"page": int(page)}
            score_range = direct.get("score_range")
            if isinstance(score_range, str) and score_range:
                block["score_range"] = score_range
            score_box_range = direct.get("score_box_range")
            if isinstance(score_box_range, str) and score_box_range:
                block["score_box_range"] = score_box_range
            return block

    mp = item.get("marking_page")
    if mp is None:
        return None
    try:
        mp_i = int(mp)
    except (TypeError, ValueError):
        return None
    grid = grids_by_page.get(mp_i)
    if grid is None:
        return None

    block: dict[str, Any] = {"page": mp_i}

    sx = item.get("marking_x_position_percent")
    sy = item.get("marking_y_position_percent")
    if sx is not None and sy is not None:
        cell = _cell_at_xy(grid, float(sx), float(sy))
        if cell:
            block["score_range"] = cell

    mb = item.get("marking_box_page")
    if mb is not None:
        try:
            mb_i = int(mb)
        except (TypeError, ValueError):
            mb_i = None
        if mb_i is not None:
            gb = grids_by_page.get(mb_i)
            if gb is not None:
                x1 = item.get("marking_box_x1_percent")
                y1 = item.get("marking_box_y1_percent")
                x2 = item.get("marking_box_x2_percent")
                y2 = item.get("marking_box_y2_percent")
                if all(v is not None for v in (x1, y1, x2, y2)):
                    a = _cell_at_xy(gb, float(x1), float(y1))
                    b = _cell_at_xy(gb, float(x2), float(y2))
                    if a and b:
                        block["score_box_range"] = a if a == b else f"{a}:{b}"
    return block


_VALID_ANCHOR_TYPES = (
    "ellipse", "tick", "underline", "arrow", "curly_brace", "exponent_caret", "none",
)


def _normalise_anchor(raw: Any) -> dict[str, Any]:
    """Coerce Gemini-emitted anchor (or absent value) into the wire shape.

    Default: ``{"type": "none"}``. Pass-through when valid. ``rows[]`` entries
    are stringified; ``extra`` is preserved if present.
    """
    if not isinstance(raw, dict):
        return {"type": "none"}
    atype = raw.get("type")
    if atype == "circle":
        atype = "ellipse"
    if atype not in _VALID_ANCHOR_TYPES:
        return {"type": "none"}
    out: dict[str, Any] = {"type": atype}
    rows = raw.get("rows")
    if isinstance(rows, list) and rows:
        out["rows"] = [str(r) for r in rows]
    extra = raw.get("extra")
    if isinstance(extra, dict) and extra:
        out["extra"] = dict(extra)
    return out


def _build_annotation(
    ann: dict[str, Any],
    grids_by_page: dict[int, PageCellGrid] | None = None,
) -> dict[str, Any]:
    """Per-annotation wire shape. Two source shapes accepted:

    1. Gemini cell-overlay path: ``page`` (1-indexed), ``comment_rows[]``
       and ``anchor`` emitted directly — passed through (no ``comment_range``).
    2. Legacy placer path: ``page_index`` (0-indexed), ``range_id`` /
       ``cell_ids[]`` converted to ``comment_rows``.
    """
    # Page — prefer Gemini's 1-indexed `page`, fall back to 0-indexed `page_index`.
    if "page" in ann:
        try:
            page = int(ann["page"])
        except (TypeError, ValueError):
            page = 1
    else:
        try:
            page = int(ann.get("page_index", 0)) + 1
        except (TypeError, ValueError):
            page = 1

    out: dict[str, Any] = {
        "page": page,
        "is_positive": bool(ann.get("is_positive", False)),
        "comment": str(ann.get("comment") or ""),
        "comment_font_pts": float(ann.get("comment_font_pts") or DEFAULT_COMMENT_FONT_PTS),
        "anchor": _normalise_anchor(ann.get("anchor")),
    }

    # comment_rows — Gemini overlay path row-wise list; wire format exposes only comment_rows.
    comment_rows_raw = ann.get("comment_rows")
    if isinstance(comment_rows_raw, list) and comment_rows_raw:
        rows = [str(r) for r in comment_rows_raw]
        if grids_by_page:
            g = grids_by_page.get(page)
            if g is not None:
                rows = _sanitize_row_tokens(rows, g)
        if rows:
            out["comment_rows"] = rows
        return out

    # Legacy placer path — derive comment_rows from rectangular range_id or cell_ids.
    if grids_by_page:
        g = grids_by_page.get(page)
        bounds: tuple[int, int, int, int] | None = None
        rid_raw = ann.get("range_id")
        if isinstance(rid_raw, str) and rid_raw.strip():
            try:
                bounds = _bounds_from_range_id_str(rid_raw)
            except ValueError:
                bounds = None
        if bounds is None:
            cid_list = ann.get("cell_ids")
            if isinstance(cid_list, list) and cid_list:
                bounds = _bounds_from_cell_id_list([str(x) for x in cid_list])
        if g is not None and bounds is not None:
            rl, cl, rh, ch = bounds
            rows = _comment_rows_from_bbox(rl, cl, rh, ch)
            rows = _sanitize_row_tokens(rows, g)
            if rows:
                out["comment_rows"] = rows
    return out


# ── Public entry point ──────────────────────────────────────────────────────


def build_response_items(
    items: list[dict[str, Any]],
    page_grids: list[PageCellGrid],
) -> list[dict[str, Any]]:
    """Shape internal placer output into the wire format the frontend reads."""
    grids_by_page: dict[int, PageCellGrid] = {g.page: g for g in page_grids}
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        new = {k: item.get(k) for k in _PASSTHROUGH_KEYS if k in item}
        new["answer_span"] = _build_answer_span(item, grids_by_page)
        new["marking"] = _build_marking(item, grids_by_page)
        anns = item.get("annotations") or []
        new["annotations"] = [
            _build_annotation(a, grids_by_page) for a in anns if isinstance(a, dict)
        ]
        out.append(new)
    return out
