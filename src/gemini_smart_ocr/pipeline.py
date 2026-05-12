"""Top-level smart-OCR orchestrator.

End-to-end flow per request:

  1. Rasterize the PDF (300+ DPI) and composite the 50×50 reference grid.
  2. Stage 1A (classify + annotate)  — parallel, grid-PNG, one Gemini call/page.
     Stage 1B (per-page OCR)         — parallel, clean PNG, one Gemini call/page.
     Both stages live in the same thread pool so OCR can start the moment its
     page's classification is known; ``DUPLICATE`` pages skip OCR entirely.
  3. Page-similarity dedup collapses near-identical neighbouring pages.
  4. Stage 2 (structure): a single Gemini call turns the OCR JSON payload into
     ``items`` (sections → flat question rows). Falls back to a halved-payload
     retry on undercount.
  5. Anchor marks + remark boxes are clipped to each item's answer range and
     attached. The marking_y position is nudged past any right-margin remark
     so the score doesn't land on top of teacher feedback.

Public API (unchanged for HTTP layer):
  ``smart_ocr_extract_student_answers(pdf_path, api_key, language, *, request_id)``
  returns ``{"items": [...], "page_count": N}``.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai

from src.gemini_copy_ocr import (
    copy_ocr_max_pages,
    copy_ocr_parallel_workers,
    copy_ocr_raster_dpi,
    count_pdf_pages,
    rasterize_pdf_to_png_pages,
)
from src.grid_overlay import batch_draw_grid

from .classify import classify_and_annotate_page
from .config import DEFAULT_CLASSIFY_WAIT_TIMEOUT_S, REMARK_ANSWER_GUARD_PCT
from .dedup import deduplicate_page_texts, extract_page_body
from .layout import (
    annotations_grid_to_pct,
    clip_marks_to_answer,
    nudge_mark_past_remarks,
    spread_remarks_two_column,
)
from .ocr import ocr_single_page
from .structure import estimate_expected_questions, structure_qa_with_fallback

log = logging.getLogger(__name__)


def smart_ocr_extract_student_answers(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Classify pages → type-aware OCR → dedupe → structure into items.

    Cover / intro segregation is performed in the Stage 2 structure pass — not here.
    """
    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    # OCR quality needs 300 DPI; grid overlay works at any DPI.
    dpi = max(300, copy_ocr_raster_dpi())
    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=dpi, request_id=request_id)
    max_workers = max(1, min(copy_ocr_parallel_workers(), total_pages))

    grid_pages = batch_draw_grid(png_pages, max_workers=max_workers)
    log.info("smart_ocr[%s] grid overlay drawn dpi=%s pages=%s", request_id, dpi, total_pages)

    # --- Stage 1 (parallel) ---------------------------------------------------------
    page_blocks: list[str] = [""] * total_pages
    page_anchor_marks: list[list[dict[str, Any]]] = [[] for _ in range(total_pages)]
    page_remarks: list[list[dict[str, Any]]] = [[] for _ in range(total_pages)]
    classifications: list[str] = ["UNKNOWN"] * total_pages
    classify_events: list[threading.Event] = [threading.Event() for _ in range(total_pages)]

    def _classify_annotate_job(idx: int) -> tuple[int, dict[str, Any]]:
        pno = idx + 1
        try:
            result = classify_and_annotate_page(
                api_key, grid_pages[idx], pno, total_pages, language
            )
            classifications[idx] = result["page_type"]
            log.info(
                "smart_ocr[%s] classify+annotate page=%s/%s type=%s anchors=%s remarks=%s",
                request_id, pno, total_pages, result["page_type"],
                len(result["anchor_marks"]), len(result["remarks"]),
            )
        except Exception as e:
            log.warning("smart_ocr[%s] classify+annotate failed page=%s: %s", request_id, pno, e)
            result = {"page_type": "UNKNOWN", "anchor_marks": [], "remarks": []}
            classifications[idx] = "UNKNOWN"
        finally:
            classify_events[idx].set()
        return idx, result

    def _ocr_job(idx: int) -> tuple[int, str]:
        pno = idx + 1
        # Wait for classification so DUPLICATE pages can skip the OCR Gemini call.
        classify_events[idx].wait(timeout=DEFAULT_CLASSIFY_WAIT_TIMEOUT_S)
        pt = classifications[idx]
        if pt == "DUPLICATE" and idx > 0:
            header = f"=== PAGE {pno} ==="
            block = (
                f"{header}\n[OMITTED: page classified DUPLICATE — duplicate of earlier sheet; "
                f"OCR skipped to save tokens.]\n"
            )
            log.info("smart_ocr[%s] ocr page=%s DUPLICATE skipped", request_id, pno)
            return idx, block
        block = ocr_single_page(api_key, png_pages[idx], pno, total_pages, language, pt)
        log.info(
            "smart_ocr[%s] ocr page=%s/%s class=%s chars=%s",
            request_id, pno, total_pages, pt, len(block),
        )
        return idx, block

    # 2× workers: N classify+annotate + N OCR jobs share one pool.
    with ThreadPoolExecutor(max_workers=max_workers * 2) as pool:
        cls_ann_futs = [pool.submit(_classify_annotate_job, i) for i in range(total_pages)]
        ocr_futs     = [pool.submit(_ocr_job, i) for i in range(total_pages)]
        for fut in as_completed(cls_ann_futs):
            idx, result = fut.result()
            anchors_pct, remarks_pct = annotations_grid_to_pct(
                result["anchor_marks"], result["remarks"], idx + 1
            )
            page_anchor_marks[idx] = anchors_pct
            page_remarks[idx] = spread_remarks_two_column(remarks_pct, page_num=idx + 1)
        for fut in as_completed(ocr_futs):
            idx, block = fut.result()
            page_blocks[idx] = block

    # Page-similarity dedup (text-only).
    page_blocks = deduplicate_page_texts(page_blocks, classifications, request_id)
    log.info(
        "smart_ocr[%s] after ocr+dedup total_chars=%s",
        request_id, sum(len(b) for b in page_blocks),
    )

    # --- Stage 2 (structure) --------------------------------------------------------
    pages_payload = {"pages": []}
    for i, block in enumerate(page_blocks):
        pages_payload["pages"].append({"page": i + 1, "text": extract_page_body(block)})
    pages_payload_json = json.dumps(pages_payload, ensure_ascii=False)
    payload_tokens_est = len(pages_payload_json) // 4
    expected_questions = estimate_expected_questions(page_blocks)
    log.info(
        "smart_ocr[%s] structure input_tokens_est=%s output_cap=%s expected_questions=%s",
        request_id, payload_tokens_est, 65536, expected_questions,
    )

    structure_client = genai.Client(api_key=api_key)
    rows = structure_qa_with_fallback(
        structure_client,
        pages_payload,
        language,
        total_pages,
        expected_questions=expected_questions,
    )
    log.info(
        "smart_ocr[%s] stage3 structure complete questions=%s",
        request_id, len(rows),
    )

    # --- Merge per-page anchor_marks + remarks into items ---------------------------
    all_anchors: list[dict[str, Any]] = []
    all_remarks: list[dict[str, Any]] = []
    for idx in range(total_pages):
        all_anchors.extend(page_anchor_marks[idx])
        all_remarks.extend(page_remarks[idx])

    for item in rows:
        sp  = int(item.get("start_page", 1))
        ep  = int(item.get("end_page", sp))
        sy  = float(item.get("start_y_position_percent", 0.0))
        ey  = float(item.get("end_y_position_percent", 100.0))
        qid = item.get("question_id", "?")

        page_spans: list[str] = []
        for pg in range(sp, ep + 1):
            if sp == ep:
                page_spans.append(f"p{pg}[y{sy:.0f}-{ey:.0f}%]")
            elif pg == sp:
                page_spans.append(f"p{pg}[y{sy:.0f}-100%]")
            elif pg == ep:
                page_spans.append(f"p{pg}[y0-{ey:.0f}%]")
            else:
                page_spans.append(f"p{pg}[y0-100%]")
        log.info("item[q%s] bbox  %s", qid, " ".join(page_spans))

        item["anchor_marks"] = clip_marks_to_answer(
            all_anchors, sp, ep, sy, ey, y_key="cy_pct"
        )
        # Push effective start_y past the printed question-text zone so remarks
        # within REMARK_ANSWER_GUARD_PCT of start_y don't land on the header.
        remark_sy = min(sy + REMARK_ANSWER_GUARD_PCT, ey)
        item["remarks"] = clip_marks_to_answer(
            all_remarks, sp, ep, remark_sy, ey, y_key="center"
        )

        log.info(
            "item[q%s] anchors=%s",
            qid,
            " ".join(
                f"p{a['page']}({a['type']}cy={a.get('cy_pct',0):.0f}%)"
                for a in item["anchor_marks"]
            ) or "(none)",
        )
        log.info(
            "item[q%s] remarks=%s",
            qid,
            " ".join(
                f"p{r['page']}y[{r.get('y1_pct',0):.0f}-{r.get('y2_pct',0):.0f}%]"
                for r in item["remarks"]
            ) or "(none)",
        )

        nudge_mark_past_remarks(item)

    return {"items": rows, "page_count": total_pages}
