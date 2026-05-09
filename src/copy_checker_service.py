"""Copy-checker pipeline service — importable without FastAPI.

Orchestrates: smart-OCR → optional grading → cell-grid → annotation bbox placement.
Mirrors the logic in ``POST /analyse/smart-ocr`` so the pipeline can be called directly
from scripts and tests without spinning up the HTTP server.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from cell_overlay_renderer import render_overlay_pngs

from src.cell_grid_service import build_cell_grid
from remark_cell_layout_service import (
    REMARK_FONTNAME_EN,
    REMARK_FONTNAME_HI,
    REMARK_FONT_SIZE_PTS,
    REMARK_MAX_WRAP_ROWS,
    assign_bboxes_to_annotations,
)
from src.gemini_evaluate_student_answers import (
    evaluate_student_answers_against_model,
    format_answer_model_as_teacher_instructions,
    merge_evaluations_into_items,
    student_items_for_grading,
)
from src.gemini_smart_ocr_v2 import grade_items_pass3_v2, smart_ocr_extract_student_answers_v2

log = logging.getLogger(__name__)

_CELL_GRID_DPI = int(os.getenv("COPY_CHECKER_CELL_DPI", "150"))


def _skipped_pages(page_count: int, items: list[dict[str, Any]]) -> list[int]:
    covered: set[int] = set()
    for row in items:
        if not isinstance(row, dict):
            continue
        sec = row.get("section_name")
        if sec is None or (isinstance(sec, str) and not sec.strip()):
            continue
        try:
            sp = int(row.get("start_page", 1))
            ep = int(row.get("end_page", sp))
        except (TypeError, ValueError):
            continue
        if ep < sp:
            ep = sp
        sp = max(1, min(sp, page_count))
        ep = max(1, min(ep, page_count))
        for p in range(sp, ep + 1):
            covered.add(p)
    return [p for p in range(1, page_count + 1) if p not in covered]


def run_copy_check(
    pdf_bytes: bytes,
    api_key: str,
    language: str,
    *,
    answer_model: dict[str, Any] | None = None,
    check_level: str = "Moderate",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Run OCR → optional grading → cell-grid → annotation bbox placement.

    Args:
        pdf_bytes:    Raw PDF file bytes.
        api_key:      Gemini API key.
        language:     ``"en"`` or ``"hi"``.
        answer_model: Dict from ``get_answer_model()``; ``None`` → OCR-only (no grading).
        check_level:  ``"Moderate"`` or ``"Hard"`` (grading strictness).
        request_id:   Caller-supplied trace ID; auto-generated if omitted.

    Returns:
        ``{"items": [...], "page_count": int, "skipped_pages": [int, ...]}``
        Each item mirrors the shape returned by ``POST /analyse/smart-ocr``.
    """
    rid = request_id or str(uuid.uuid4())
    lang = language.strip().lower()
    remark_font = REMARK_FONTNAME_HI if lang == "hi" else REMARK_FONTNAME_EN

    tmp: Path | None = None
    page_grids: list[Any] = []
    overlay_jpegs: list[bytes] | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "wb") as fh:
            fh.write(pdf_bytes)

        log.info("copy_checker[%s] building cell grid + overlays", rid)
        try:
            page_grids = build_cell_grid(pdf_bytes)
            overlay_jpegs = render_overlay_pngs(
                pdf_bytes,
                page_grids,
                dpi=_CELL_GRID_DPI,
                image_format="jpeg",
                jpeg_quality=85,
                label_every_cell=True,
            )
        except Exception:
            log.exception("copy_checker[%s] grid/overlay failed — OCR may rebuild internally", rid)
            page_grids = []
            overlay_jpegs = None

        log.info("copy_checker[%s] starting smart-ocr v2", rid)
        ocr_result = smart_ocr_extract_student_answers_v2(
            tmp,
            api_key,
            lang,
            request_id=rid,
            overlay_images=overlay_jpegs,
            grids=page_grids if page_grids else None,
        )
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    flat_blocks = ocr_result.pop("_flat_blocks", None)
    v2_ov = ocr_result.pop("_overlay_jpegs", None)
    page_count = int(ocr_result.get("page_count") or 0)
    items = list(ocr_result.get("items") or [])
    log.info("copy_checker[%s] ocr ok pages=%s items=%s", rid, page_count, len(items))

    if not page_grids:
        try:
            page_grids = build_cell_grid(pdf_bytes)
        except Exception:
            log.warning("copy_checker[%s] late grid build failed", rid)
            page_grids = []

    # Stage 3: optional grading (same strategy as POST /analyse/smart-ocr)
    if answer_model:
        try:
            questions = answer_model.get("questions") or []
            title = answer_model.get("title") or "General"
            teacher_instructions = format_answer_model_as_teacher_instructions(questions, title)
            items_to_grade = student_items_for_grading(items)
            overlay_for_pass3 = overlay_jpegs if overlay_jpegs is not None else v2_ov
            if flat_blocks and overlay_for_pass3:
                ev_list = grade_items_pass3_v2(
                    api_key,
                    title,
                    title,
                    questions,
                    items_to_grade,
                    page_grids,
                    flat_blocks,
                    overlay_for_pass3,
                    request_id=rid,
                    check_level=check_level,
                )
            else:
                ev_list = evaluate_student_answers_against_model(
                    api_key,
                    title,
                    teacher_instructions,
                    items_to_grade,
                    request_id=rid,
                    check_level=check_level,
                    overlay_images=overlay_for_pass3,
                    use_overlay_prompt=bool(overlay_for_pass3),
                )
            items = merge_evaluations_into_items(items, ev_list)
            log.info("copy_checker[%s] grading ok merged=%s", rid, len(items))
        except Exception:
            log.exception("copy_checker[%s] grading failed — returning OCR-only result", rid)

    # Annotation bbox placement
    try:
        assign_bboxes_to_annotations(
            items,
            page_grids,
            font_size_pts=REMARK_FONT_SIZE_PTS,
            fontname=remark_font,
            max_wrap_rows=REMARK_MAX_WRAP_ROWS,
        )
        log.info("copy_checker[%s] bbox assignment done", rid)
    except Exception:
        log.exception("copy_checker[%s] bbox assignment failed", rid)

    return {
        "items": items,
        "page_count": page_count,
        "skipped_pages": _skipped_pages(page_count, items),
    }
