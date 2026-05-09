"""Cell-native Smart OCR v2 — block OCR → structure → deterministic resolve."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cell_overlay_renderer import render_overlay_pngs

from cell_grid_service_v4 import PageCellGrid

from src.cell_grid_service import build_cell_grid
from src.gemini_copy_ocr import copy_ocr_max_pages, count_pdf_pages

from .pass1_block_ocr import run_pass1_page
from .pass2_structure import run_pass2_structure
from .pass3_grade import grade_items_pass3_v2
from .resolve import resolve_sections_to_items

log = logging.getLogger(__name__)

__all__ = [
    "smart_ocr_extract_student_answers_v2",
    "grade_items_pass3_v2",
]


def smart_ocr_extract_student_answers_v2(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
    overlay_images: list[bytes] | None = None,
    grids: list[PageCellGrid] | None = None,
) -> dict[str, Any]:
    """Pass-1/2/resolve Smart OCR — same top-level keys as v1 plus ``_flat_blocks``.

    ``overlay_images``: one JPEG per page (same contract as v1). When omitted,
    grids are analysed and overlays rendered here.

    ``grids``: optional pre-built v4 grids (e.g. from FastAPI pre-pass). When
    provided without overlays, JPEGs are rendered from these grids.

    Internal consumers may read ``_flat_blocks`` (block_id → block) until the
    HTTP layer pops it.
    """
    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    pdf_bytes = pdf_path.read_bytes()

    page_grids: list[PageCellGrid]
    if grids is not None:
        page_grids = grids
    else:
        page_grids = build_cell_grid(pdf_bytes)

    if overlay_images is None:
        overlay_images = render_overlay_pngs(
            pdf_bytes,
            page_grids,
            dpi=150,
            image_format="jpeg",
            jpeg_quality=85,
            label_every_cell=True,
        )

    if len(overlay_images) < total_pages:
        raise ValueError(
            f"overlay_images has {len(overlay_images)} page(s), PDF has {total_pages}"
        )

    grids_by_page = {g.page: g for g in page_grids}

    def _job(idx: int) -> tuple[int, dict[str, Any]]:
        pno = idx + 1
        grid = grids_by_page[pno]
        jpeg = overlay_images[idx]
        page_obj = run_pass1_page(
            api_key,
            jpeg,
            grid,
            pno,
            total_pages,
            language,
            request_id=request_id,
        )
        return idx, page_obj

    page_results: list[dict[str, Any] | None] = [None] * total_pages
    workers = min(8, total_pages)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, i) for i in range(total_pages)]
        for fut in as_completed(futs):
            idx, obj = fut.result()
            page_results[idx] = obj

    flat_blocks: dict[str, dict[str, Any]] = {}
    pages_payload: list[dict[str, Any]] = []
    cover_pages: list[int] = []
    for i, pg in enumerate(page_results):
        if pg is None:
            raise RuntimeError(f"Pass1 missing page index {i}")
        page_no = int(pg.get("page") or i + 1)
        if bool(pg.get("is_cover_page")):
            cover_pages.append(page_no)
            log.info(
                "smart_ocr_v2[%s] page=%s flagged is_cover_page=True — dropping blocks",
                request_id,
                page_no,
            )
            continue
        blocks = pg.get("blocks") or []
        pages_payload.append(
            {
                "page": page_no,
                "page_type": pg.get("page_type") or "UNKNOWN",
                "blocks": blocks,
            }
        )
        for b in blocks:
            if not isinstance(b, dict):
                continue
            bid = str(b.get("block_id") or "")
            if bid:
                flat_blocks[bid] = b

    blocks_document = {"pages": pages_payload}

    sections_raw = run_pass2_structure(
        api_key,
        blocks_document,
        language,
        total_pages,
        request_id=request_id,
    )

    items = resolve_sections_to_items(
        sections_raw,
        flat_blocks,
        page_grids,
        pdf_bytes,
    )

    log.info(
        "smart_ocr_v2[%s] complete items=%s blocks=%s cover_pages=%s",
        request_id,
        len(items),
        len(flat_blocks),
        cover_pages,
    )

    return {
        "items": items,
        "page_count": total_pages,
        "_flat_blocks": flat_blocks,
        "_overlay_jpegs": overlay_images,
        "_cover_pages": cover_pages,
    }
