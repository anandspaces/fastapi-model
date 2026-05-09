"""Resolve block refs → flat student items with ``answer_span`` and ``marking``."""

from __future__ import annotations

import logging
import re
from typing import Any

from cell_grid_service_v4 import PageCellGrid, cell_id_from_rc, detect_marking_boxes, rc_from_cell_id

from .snap import bounding_range_string, cells_from_range_strings, leftmost_writable_cell_in_ranges


log = logging.getLogger(__name__)

_BLOCK_PAGE_RE = re.compile(r"^p(\d+)-", re.I)

_COVER_SECTION_RE = re.compile(
    r"\b(student\s*detail|personal\s*detail|cover|intro|preamble|"
    r"examiner\s*use|roll\s*info|booklet\s*detail)\b",
    re.IGNORECASE,
)


def _page_from_block_id(block_id: str) -> int | None:
    m = _BLOCK_PAGE_RE.match(block_id.strip())
    if not m:
        return None
    return int(m.group(1))


def _collect_answer_span_pages(
    grids_by_page: dict[int, PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    answer_refs: list[str],
) -> list[dict[str, Any]]:
    by_page: dict[int, list[str]] = {}
    for ref in answer_refs:
        blk = flat_blocks.get(ref)
        if not blk or blk.get("kind") != "handwritten_answer":
            continue
        bid = str(blk.get("block_id") or ref)
        p = _page_from_block_id(bid)
        if p is None:
            continue
        cells = blk.get("cells")
        if not isinstance(cells, list):
            continue
        rs = [str(x) for x in cells]
        by_page.setdefault(p, []).extend(rs)
    out: list[dict[str, Any]] = []
    for p in sorted(by_page.keys()):
        grid = grids_by_page.get(p)
        if grid is None:
            continue
        rows = by_page[p]
        # merge overlapping row strings visually — keep list as aligned segments
        out.append({"page": p, "rows": rows})
    return out


def _find_marking_box_ranges(
    flat_blocks: dict[str, dict[str, Any]],
    pages: list[int],
) -> list[str] | None:
    ranges: list[str] = []
    for ref, blk in flat_blocks.items():
        if blk.get("kind") != "marking_box":
            continue
        bid = str(blk.get("block_id") or ref)
        p = _page_from_block_id(bid)
        if p is None or p not in pages:
            continue
        cells = blk.get("cells")
        if isinstance(cells, list) and cells:
            ranges.extend(str(x) for x in cells)
    return ranges or None


def _build_marking(
    grids_by_page: dict[int, PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    marking_hints: dict[int, str],
    start_page: int,
    end_page: int,
) -> dict[str, Any] | None:
    pages_try = [start_page, end_page]
    mb_ranges = _find_marking_box_ranges(flat_blocks, pages_try)
    score_page = start_page
    if mb_ranges is None:
        hint = marking_hints.get(start_page) or marking_hints.get(end_page)
        if not hint:
            return None
        grid = grids_by_page.get(start_page) or grids_by_page.get(end_page)
        if grid is None:
            return None
        score_page = start_page if marking_hints.get(start_page) else end_page
        mb_ranges = [hint]

    grid = grids_by_page.get(score_page)
    if grid is None or not mb_ranges:
        return None

    box_union = bounding_range_string(grid, mb_ranges)
    if not box_union:
        return None
    score_cell = leftmost_writable_cell_in_ranges(grid, mb_ranges)
    if not score_cell:
        # fall back: first cell in bounding rect's top-left
        score_cell = box_union.split(":")[0]

    return {
        "page": score_page,
        "score_range": score_cell,
        "score_box_range": box_union,
    }


def _rough_percent_from_cell(grid: PageCellGrid, cell_id: str) -> tuple[float, float]:
    """Centre of cell as percent for legacy merge helpers."""
    try:
        row, col = rc_from_cell_id(cell_id)
    except ValueError:
        return 50.0, 50.0
    cell = next((c for c in grid.cells if c.row == row and c.col == col), None)
    if cell is None:
        return 50.0, 50.0
    cx = (cell.pdf_x1 + cell.pdf_x2) / 2.0 / float(grid.page_w_pts) * 100.0
    cy = (cell.pdf_y1 + cell.pdf_y2) / 2.0 / float(grid.page_h_pts) * 100.0
    return round(cx, 2), round(cy, 2)


def resolve_sections_to_items(
    sections_raw: dict[str, Any],
    flat_blocks: dict[str, dict[str, Any]],
    grids: list[PageCellGrid],
    pdf_bytes: bytes,
) -> list[dict[str, Any]]:
    """Turn pass-2 ``sections`` into items compatible with grading / formatter."""
    grids_by_page = {g.page: g for g in grids}
    marking_hints = detect_marking_boxes(pdf_bytes, grids)

    sections = sections_raw.get("sections")
    if not isinstance(sections, list):
        return []

    out: list[dict[str, Any]] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        section_name = str(sec.get("section_name") or "General").strip() or "General"
        if _COVER_SECTION_RE.search(section_name):
            log.info("resolve: dropping cover-like section %r", section_name)
            continue
        questions = sec.get("questions")
        if not isinstance(questions, list):
            continue
        for row in questions:
            if not isinstance(row, dict):
                continue
            try:
                qid = int(row.get("question_id"))
            except (TypeError, ValueError):
                continue
            qrefs = row.get("question_block_refs") or []
            arefs = row.get("answer_block_refs") or []
            if not isinstance(qrefs, list):
                qrefs = []
            if not isinstance(arefs, list):
                arefs = []

            has_printed_question = False
            q_text_parts: list[str] = []
            for ref in qrefs:
                b = flat_blocks.get(str(ref))
                if not b:
                    continue
                if str(b.get("kind")) == "printed_question":
                    has_printed_question = True
                q_text_parts.append(str(b.get("text") or "").strip())
            question_text = "\n\n".join(x for x in q_text_parts if x)
            stem_hint = str(row.get("question") or "").strip()
            if stem_hint:
                question_text = stem_hint if not question_text else f"{question_text}\n\n{stem_hint}"

            if not has_printed_question and not stem_hint:
                log.info(
                    "resolve: dropping fabricated item qid=%s section=%r "
                    "(no printed_question ref and no stem hint — likely cover/intro)",
                    qid,
                    section_name,
                )
                continue

            ans_parts: list[str] = []
            for ref in arefs:
                b = flat_blocks.get(str(ref))
                if b and str(b.get("kind")) == "handwritten_answer":
                    t = str(b.get("text") or "").strip()
                    if t:
                        ans_parts.append(t)
            student_answer = "\n\n".join(ans_parts)
            is_attempted = bool(student_answer) and bool(row.get("is_attempted", True))

            pages_with_answers: list[int] = []
            for ref in arefs:
                b = flat_blocks.get(str(ref))
                if not b:
                    continue
                bid = str(b.get("block_id") or ref)
                p = _page_from_block_id(bid)
                if p is not None:
                    pages_with_answers.append(p)
            start_page = min(pages_with_answers) if pages_with_answers else 1
            end_page = max(pages_with_answers) if pages_with_answers else start_page

            answer_span = _collect_answer_span_pages(grids_by_page, flat_blocks, [str(x) for x in arefs])

            marking = _build_marking(
                grids_by_page, flat_blocks, marking_hints, start_page, end_page
            )

            # Legacy percent coords for merge_evaluations / assign_cell_ids fallbacks
            grid_sp = grids_by_page.get(start_page)
            cx, cy = (50.0, 50.0)
            if grid_sp and answer_span:
                first_rows = answer_span[0].get("rows") or []
                if first_rows:
                    cells = cells_from_range_strings(grid_sp, [str(first_rows[0])])
                    if cells:
                        r0, c0 = min(cells)
                        cid = cell_id_from_rc(r0, c0)
                        cx, cy = _rough_percent_from_cell(grid_sp, cid)

            mx, my = cx, cy
            mp = start_page
            if marking and grids_by_page.get(int(marking["page"])):
                gmk = grids_by_page[int(marking["page"])]
                sr = str(marking["score_range"])
                anchor_cell = sr.split(":")[0].strip() if ":" in sr else sr.strip()
                mx, my = _rough_percent_from_cell(gmk, anchor_cell)

            item: dict[str, Any] = {
                "question_id": qid,
                "question": question_text or f"Question {qid}",
                "student_answer": student_answer,
                "is_attempted": is_attempted,
                "section_name": section_name,
                "answer_type": str(row.get("answer_type") or "paragraph").strip()
                or "paragraph",
                "start_page": start_page,
                "end_page": end_page,
                "start_y_position_percent": max(0.0, cy - 5.0),
                "end_y_position_percent": min(100.0, cy + 5.0),
                "marking_page": mp,
                "marking_x_position_percent": mx,
                "marking_y_position_percent": my,
                "answer_span": answer_span,
                "marking": marking,
                "question_block_refs": [str(x) for x in qrefs],
                "answer_block_refs": [str(x) for x in arefs],
            }
            out.append(item)

    out.sort(key=lambda r: int(r.get("question_id") or 0))
    # Fill missing question_ids
    used = {int(i["question_id"]) for i in out if i.get("question_id") is not None}
    nxt = 1
    for it in out:
        if it.get("question_id") is None:
            while nxt in used:
                nxt += 1
            it["question_id"] = nxt
            used.add(nxt)
            nxt += 1

    return out
