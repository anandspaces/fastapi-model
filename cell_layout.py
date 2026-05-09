"""cell_layout.py — invariant validator for the v2 cell-ID annotation contract.

Observe-only in this milestone. Every check returns a count of violations into
``Telemetry``; no response field is mutated. The repair codepath lands in P3
once we have baseline numbers on Gemini's clean-output rate.

Invariants (mirror the design doc):
    I1  every cell ID is within page grid bounds
    I2  comment_range cells are all writable
    I3  anchor.range cells are within grid (writability not required)
    I4  marking.score_range ⊆ marking.score_box_range
    I5  comment_font_pts ∈ [14.0, 16.0]
    I6  comment text fits comment_range at comment_font_pts (with 5% margin)
    I7  no two annotations on the same page have overlapping comment_range
    I8  exponent_caret anchors have non-empty ``extra.missing_word``
    I9  curly_brace anchors have ``extra.side`` ∈ {"left", "right"}

Public API:
    validate(response, grids, *, font=None) -> Telemetry

The response payload is expected to match the v2 ``/analyse/smart-ocr/v2``
shape — the data wrapper layer is optional, so both ``{"data": {...}}`` and
the inner dict are accepted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz

from cell_grid_service_v4 import PageCellGrid, rc_from_cell_id

FONT_PTS_MIN: float = 14.0
FONT_PTS_MAX: float = 16.0
DEFAULT_FONT_PTS: float = 14.0
COMMENT_FIT_MARGIN: float = 0.05  # 5% breathing room inside cell rectangle
LINE_HEIGHT_MULT: float = 1.4


@dataclass
class ViolationRecord:
    invariant: str                       # "I1" .. "I9"
    page: int | None
    item_index: int | None
    annotation_index: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "page": self.page,
            "item_index": self.item_index,
            "annotation_index": self.annotation_index,
            "detail": self.detail,
        }


@dataclass
class Telemetry:
    counts: Counter = field(default_factory=Counter)
    violations: list[ViolationRecord] = field(default_factory=list)
    items_total: int = 0
    annotations_total: int = 0

    def add(self, rec: ViolationRecord, *, max_records: int = 200) -> None:
        self.counts[rec.invariant] += 1
        if len(self.violations) < max_records:
            self.violations.append(rec)

    def violation_rate(self) -> float:
        if self.annotations_total == 0:
            return 0.0
        return sum(self.counts.values()) / self.annotations_total

    def summary(self) -> dict[str, Any]:
        return {
            "items_total": self.items_total,
            "annotations_total": self.annotations_total,
            "counts": dict(self.counts),
            "violation_rate": round(self.violation_rate(), 4),
        }


# ── Range helpers ───────────────────────────────────────────────────────────


def _expand_range(range_id: str) -> list[tuple[int, int]]:
    """Expand ``"A1:C3"`` (or single ``"A1"``) to list of (row, col) tuples."""
    if ":" in range_id:
        a, b = range_id.split(":", 1)
    else:
        a = b = range_id
    r1, c1 = rc_from_cell_id(a.strip())
    r2, c2 = rc_from_cell_id(b.strip())
    r_lo, r_hi = min(r1, r2), max(r1, r2)
    c_lo, c_hi = min(c1, c2), max(c1, c2)
    return [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]


def _range_bounds(range_id: str) -> tuple[int, int, int, int]:
    """Return (row_lo, col_lo, row_hi, col_hi) for a range ID."""
    if ":" in range_id:
        a, b = range_id.split(":", 1)
    else:
        a = b = range_id
    r1, c1 = rc_from_cell_id(a.strip())
    r2, c2 = rc_from_cell_id(b.strip())
    return min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ── Capacity check (I6) ─────────────────────────────────────────────────────


def _comment_fits(
    comment: str,
    font: fitz.Font,
    font_size: float,
    *,
    cols: int,
    rows: int,
    cell_size_pts: float,
) -> bool:
    """Word-wrap ``comment`` and check it fits inside the cell rectangle.

    Conservative: any single word wider than the available width (after
    margin) fails the check; line height is ``font_size × LINE_HEIGHT_MULT``.
    """
    if cell_size_pts <= 0 or cols <= 0 or rows <= 0:
        return False
    avail_w = cols * cell_size_pts * (1.0 - COMMENT_FIT_MARGIN)
    avail_h = rows * cell_size_pts * (1.0 - COMMENT_FIT_MARGIN)
    line_h = font_size * LINE_HEIGHT_MULT
    if line_h > avail_h:
        return False

    words = comment.split()
    if not words:
        return True
    space_w = font.text_length(" ", fontsize=font_size)
    lines = 1
    cur_w = 0.0
    for w in words:
        ww = font.text_length(w, fontsize=font_size)
        if ww > avail_w:
            return False  # single word too wide; we don't hyphenate
        if cur_w == 0.0:
            cur_w = ww
        elif cur_w + space_w + ww <= avail_w:
            cur_w += space_w + ww
        else:
            lines += 1
            cur_w = ww
            if lines * line_h > avail_h:
                return False
    return lines * line_h <= avail_h


# ── Per-page invariant checks ───────────────────────────────────────────────


def _check_cells_in_grid(
    cells: Iterable[tuple[int, int]], grid: PageCellGrid
) -> list[tuple[int, int]]:
    return [(r, c) for r, c in cells if r < 1 or r > grid.rows or c < 1 or c > grid.cols]


def _check_cells_writable(
    cells: Iterable[tuple[int, int]], writable_set: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [rc for rc in cells if rc not in writable_set]


def _writable_set(grid: PageCellGrid) -> set[tuple[int, int]]:
    return {(c.row, c.col) for c in grid.cells if c.writable}


def _safe_get_grid(
    grids_by_page: dict[int, PageCellGrid], page: Any
) -> PageCellGrid | None:
    try:
        return grids_by_page.get(int(page))
    except (TypeError, ValueError):
        return None


# ── Validator ───────────────────────────────────────────────────────────────


def _resolve_data(response: dict[str, Any]) -> dict[str, Any]:
    """Accept either a full HTTP envelope ``{"data": {...}}`` or just the inner
    dict directly."""
    if isinstance(response.get("data"), dict):
        return response["data"]
    return response


# ── v1 → v2 shape adapter ───────────────────────────────────────────────────


def _percent_to_cell_id(
    x_pct: float, y_pct: float, meta: dict[str, Any]
) -> str | None:
    """Convert (x%, y%) into a single cell ID using cellGridMeta."""
    try:
        pw = float(meta["page_w_pts"])
        ph = float(meta["page_h_pts"])
        cs = float(meta["cell_size_pts"])
        lm = float(meta["left_margin_pts"])
        tm = float(meta["top_margin_pts"])
        rows = int(meta["rows"])
        cols = int(meta["cols"])
    except (KeyError, TypeError, ValueError):
        return None
    if cs <= 0:
        return None
    x_pts = pw * (x_pct / 100.0)
    y_pts = ph * (y_pct / 100.0)
    col = int((x_pts - lm) / cs) + 1
    row = int((y_pts - tm) / cs) + 1
    col = max(1, min(cols, col))
    row = max(1, min(rows, row))
    from cell_grid_service_v4 import cell_id_from_rc
    return cell_id_from_rc(row, col)


def _percent_box_to_range(
    x1_pct: float, y1_pct: float, x2_pct: float, y2_pct: float,
    meta: dict[str, Any],
) -> str | None:
    a = _percent_to_cell_id(x1_pct, y1_pct, meta)
    b = _percent_to_cell_id(x2_pct, y2_pct, meta)
    if not a or not b:
        return None
    return f"{a}:{b}" if a != b else a


# NOTE: the live response transformer used by /analyse/smart-ocr lives in
# src/cell_response_formatter.py (build_response_items). The functions below
# (prompt-side cell layout, legacy adapter, invariant validator) are
# observation / prompt-feeding tooling — they don't transform the response.


# ── Prompt-side spatial vocabulary ─────────────────────────────────────────


def cell_layout_for_prompt(
    grids: list[PageCellGrid],
    *,
    include_runs: bool = True,
    include_regions: bool = True,
) -> list[dict[str, Any]]:
    """Compact per-page cell-layout description Gemini reads as the spatial
    vocabulary for cell-ID-native annotation placement.

    For each page emits:
        { "page": N, "rows": R, "cols": C,
          "writable_runs": ["A5:X5", "A6:X6", ...],   // truth source for comment_range
          "regions":       ["A5:X9", "A22:X29"]       // largest free rectangles, hint only
        }

    Token budget. Runs are O(rows × density) per page; regions are O(32) max.
    Measured on real PDFs:

        cell_size_pts   pages   runs    regions   total chars   ~tokens
        24              18       550      221       9 KB         2.4k
        12              18      3479      576      48 KB        12.0k

    For dense 12-pt grids the runs payload alone can rival the rest of the
    grading call's input. Caller passes ``include_runs=False`` (regions only)
    when the budget is tight; comments still place correctly because a
    region IS a writable rectangle by construction.
    """
    out: list[dict[str, Any]] = []
    for g in grids:
        page_entry: dict[str, Any] = {
            "page": int(g.page),
            "rows": int(g.rows),
            "cols": int(g.cols),
        }
        if include_runs:
            page_entry["writable_runs"] = [
                r.range_id for r in (getattr(g, "runs", None) or [])
            ]
        if include_regions:
            page_entry["regions"] = [
                rg.bbox_range_id for rg in (getattr(g, "regions", None) or [])
            ]
        out.append(page_entry)
    return out


# Heuristic budget for when runs become noise: if the per-call serialized
# cell-layout would exceed this many characters, drop the per-row runs and
# rely on regions alone. Roughly 1/4 the chars give the token estimate.
CELL_LAYOUT_MAX_CHARS_BUDGET: int = 16000


def cell_layout_for_prompt_budgeted(
    grids: list[PageCellGrid],
    *,
    max_chars: int = CELL_LAYOUT_MAX_CHARS_BUDGET,
) -> list[dict[str, Any]]:
    """Auto-decide whether to include ``writable_runs`` based on budget.

    First tries the full payload (runs + regions). If serialized > ``max_chars``,
    falls back to regions only. Always returns a usable payload — never None.
    """
    import json
    full = cell_layout_for_prompt(grids, include_runs=True, include_regions=True)
    if len(json.dumps(full)) <= max_chars:
        return full
    return cell_layout_for_prompt(grids, include_runs=False, include_regions=True)


def adapt_v1_to_v2(response: dict[str, Any]) -> dict[str, Any]:
    """Convert a today-shape response into the v2 contract shape.

    Used by the baseline eval to run v2 invariants over today's responses.
    Best-effort — fields the v1 response does not have are filled with v2
    defaults (``comment_font_pts=12``, ``anchor.type="none"``).
    """
    src = _resolve_data(response)
    meta_by_page: dict[int, dict[str, Any]] = {
        m["page"]: m for m in (src.get("cellGridMeta") or [])
        if isinstance(m, dict) and "page" in m
    }

    out_items: list[dict[str, Any]] = []
    for it in src.get("items") or []:
        if not isinstance(it, dict):
            continue
        new_item = {
            k: it[k] for k in (
                "question_id", "max_marks", "marks_awarded", "status",
                "feedback", "student_answer_summary",
            ) if k in it
        }
        # marking
        mp = it.get("marking_page")
        if mp is not None:
            meta = meta_by_page.get(int(mp), {})
            score_cell = _percent_to_cell_id(
                float(it.get("marking_x_position_percent") or 0.0),
                float(it.get("marking_y_position_percent") or 0.0),
                meta,
            )
            mb = it.get("marking_box_page")
            score_box_range = None
            if mb is not None:
                meta_b = meta_by_page.get(int(mb), {})
                score_box_range = _percent_box_to_range(
                    float(it.get("marking_box_x1_percent") or 0.0),
                    float(it.get("marking_box_y1_percent") or 0.0),
                    float(it.get("marking_box_x2_percent") or 0.0),
                    float(it.get("marking_box_y2_percent") or 0.0),
                    meta_b,
                )
            marking_block: dict[str, Any] = {"page": int(mp)}
            if score_cell:
                marking_block["score_range"] = score_cell
            if score_box_range:
                marking_block["score_box_range"] = score_box_range
            new_item["marking"] = marking_block

        # annotations
        new_anns: list[dict[str, Any]] = []
        for ann in it.get("annotations") or []:
            if not isinstance(ann, dict):
                continue
            page_index = ann.get("page_index", 0)
            try:
                page = int(page_index) + 1
            except (TypeError, ValueError):
                page = 1
            new_ann: dict[str, Any] = {
                "page": page,
                "is_positive": bool(ann.get("is_positive", False)),
                "comment": str(ann.get("comment") or ""),
                "comment_font_pts": float(ann.get("comment_font_pts") or DEFAULT_FONT_PTS),
                "anchor": {"type": "none"},
            }
            range_id = ann.get("range_id")
            if range_id:
                new_ann["comment_range"] = str(range_id)
            new_anns.append(new_ann)
        new_item["annotations"] = new_anns
        out_items.append(new_item)

    return {
        "data": {
            "items": out_items,
            "cellGridMeta": src.get("cellGridMeta") or [],
            "pageCount": src.get("pageCount"),
        }
    }


def validate(
    response: dict[str, Any],
    grids: list[PageCellGrid],
    *,
    font: fitz.Font | None = None,
    font_path: str | None = "fonts/HomemadeApple-Regular.ttf",
) -> Telemetry:
    """Observe-only invariant check; returns Telemetry, never mutates response."""
    if font is None:
        if font_path and Path(font_path).exists():
            font = fitz.Font(fontfile=font_path)
        else:
            font = fitz.Font("helv")

    data = _resolve_data(response)
    items = data.get("items") or []
    grids_by_page = {g.page: g for g in grids}
    writable_by_page: dict[int, set[tuple[int, int]]] = {}

    tel = Telemetry()
    tel.items_total = len(items)

    # Per-page comment-range bounds for I7 (overlap detection)
    per_page_comment_rects: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}

    for ii, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        anns = item.get("annotations") or []
        tel.annotations_total += len(anns)

        # ── I4: marking.score_range ⊆ score_box_range ────────────────────
        marking = item.get("marking")
        if isinstance(marking, dict):
            score_range = marking.get("score_range")
            box_range = marking.get("score_box_range")
            if score_range and box_range:
                try:
                    s_lo_r, s_lo_c, s_hi_r, s_hi_c = _range_bounds(str(score_range))
                    b_lo_r, b_lo_c, b_hi_r, b_hi_c = _range_bounds(str(box_range))
                    contained = (
                        s_lo_r >= b_lo_r and s_hi_r <= b_hi_r
                        and s_lo_c >= b_lo_c and s_hi_c <= b_hi_c
                    )
                    if not contained:
                        tel.add(ViolationRecord(
                            invariant="I4",
                            page=marking.get("page"),
                            item_index=ii,
                            annotation_index=None,
                            detail=f"score_range {score_range} not inside score_box_range {box_range}",
                        ))
                except ValueError as exc:
                    tel.add(ViolationRecord(
                        invariant="I4",
                        page=marking.get("page"),
                        item_index=ii,
                        annotation_index=None,
                        detail=f"unparseable range: {exc}",
                    ))

        # ── per-annotation checks ─────────────────────────────────────────
        for ai, ann in enumerate(anns):
            if not isinstance(ann, dict):
                continue
            page = ann.get("page")
            grid = _safe_get_grid(grids_by_page, page)
            if grid is None:
                tel.add(ViolationRecord(
                    invariant="I1",
                    page=page,
                    item_index=ii,
                    annotation_index=ai,
                    detail=f"no grid for page {page}",
                ))
                continue
            if grid.page not in writable_by_page:
                writable_by_page[grid.page] = _writable_set(grid)
            wset = writable_by_page[grid.page]

            # ── I5: comment_font_pts in range ────────────────────────────
            font_pts_raw = ann.get("comment_font_pts", DEFAULT_FONT_PTS)
            try:
                font_pts = float(font_pts_raw)
            except (TypeError, ValueError):
                font_pts = -1.0
            if not (FONT_PTS_MIN <= font_pts <= FONT_PTS_MAX):
                tel.add(ViolationRecord(
                    invariant="I5",
                    page=grid.page,
                    item_index=ii,
                    annotation_index=ai,
                    detail=f"comment_font_pts={font_pts_raw} not in [{FONT_PTS_MIN}, {FONT_PTS_MAX}]",
                ))
                font_pts = max(FONT_PTS_MIN, min(FONT_PTS_MAX, font_pts)) if font_pts > 0 else DEFAULT_FONT_PTS

            # ── comment_range: I1 + I2 + I6 + I7 ─────────────────────────
            comment_range = ann.get("comment_range")
            if isinstance(comment_range, str) and comment_range:
                try:
                    cells = _expand_range(comment_range)
                    bounds = _range_bounds(comment_range)
                except ValueError as exc:
                    tel.add(ViolationRecord(
                        invariant="I1",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail=f"unparseable comment_range: {exc}",
                    ))
                    continue

                # I1: bounds
                oob = _check_cells_in_grid(cells, grid)
                if oob:
                    tel.add(ViolationRecord(
                        invariant="I1",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail=f"comment_range cells out of grid: {oob[:3]} (+{max(0,len(oob)-3)} more)",
                    ))

                # I2: writable
                non_writable = _check_cells_writable(cells, wset)
                if non_writable:
                    tel.add(ViolationRecord(
                        invariant="I2",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail=f"{len(non_writable)} non-writable cells in comment_range "
                               f"(e.g. {non_writable[:3]})",
                    ))

                # I6: capacity
                comment_text = str(ann.get("comment") or "")
                rows = bounds[2] - bounds[0] + 1
                cols = bounds[3] - bounds[1] + 1
                if comment_text and not _comment_fits(
                    comment_text, font, font_pts,
                    cols=cols, rows=rows, cell_size_pts=grid.cell_size_pts,
                ):
                    tel.add(ViolationRecord(
                        invariant="I6",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail=f"comment ({len(comment_text)} chars) overflows {rows}x{cols} cells "
                               f"at {font_pts:.1f}pt",
                    ))

                # Stash for I7
                per_page_comment_rects.setdefault(grid.page, []).append((ai, bounds))

            # ── anchor: I3 + I8 + I9 ──────────────────────────────────────
            anchor = ann.get("anchor") or {}
            if not isinstance(anchor, dict):
                continue
            atype = anchor.get("type") or "none"
            arange = anchor.get("range")
            arows = anchor.get("rows")
            extra = anchor.get("extra") or {}

            # Two source shapes for anchor cells:
            #  - new wire shape: anchor.rows = ["B25:G25", "B26:G26"]
            #  - legacy:        anchor.range = "B25:G25"
            anchor_ranges: list[str] = []
            if isinstance(arows, list) and arows:
                anchor_ranges = [str(r) for r in arows]
            elif isinstance(arange, str) and arange:
                anchor_ranges = [arange]

            if atype != "none" and anchor_ranges:
                a_cells: list[tuple[int, int]] = []
                bad_parse = False
                for r in anchor_ranges:
                    try:
                        a_cells.extend(_expand_range(r))
                    except ValueError as exc:
                        bad_parse = True
                        tel.add(ViolationRecord(
                            invariant="I3",
                            page=grid.page,
                            item_index=ii,
                            annotation_index=ai,
                            detail=f"unparseable anchor range {r!r}: {exc}",
                        ))
                a_oob = _check_cells_in_grid(a_cells, grid) if not bad_parse else []
                if a_oob:
                    tel.add(ViolationRecord(
                        invariant="I3",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail=f"anchor cells out of grid: {a_oob[:3]}",
                    ))

            if atype == "exponent_caret":
                missing = extra.get("missing_word") if isinstance(extra, dict) else None
                if not missing or not str(missing).strip():
                    tel.add(ViolationRecord(
                        invariant="I8",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail="exponent_caret anchor with empty/missing extra.missing_word",
                    ))

            if atype == "curly_brace":
                side = extra.get("side") if isinstance(extra, dict) else None
                if side not in ("left", "right"):
                    tel.add(ViolationRecord(
                        invariant="I9",
                        page=grid.page,
                        item_index=ii,
                        annotation_index=ai,
                        detail=f"curly_brace anchor with extra.side={side!r} (expected left/right)",
                    ))

    # ── I7: same-page comment overlap (across items + within item) ──────
    for page, entries in per_page_comment_rects.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                ai_a, rect_a = entries[i]
                ai_b, rect_b = entries[j]
                if _rects_overlap(rect_a, rect_b):
                    tel.add(ViolationRecord(
                        invariant="I7",
                        page=page,
                        item_index=None,
                        annotation_index=ai_a,
                        detail=f"comment_range overlaps annotation #{ai_b} on page {page}",
                    ))

    return tel
