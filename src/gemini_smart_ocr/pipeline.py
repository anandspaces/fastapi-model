"""Smart-OCR orchestrator — three explicit steps.

Step 1 — grid overlay. Rasterize the PDF at 300+ DPI, composite the 50×50
reference grid on each page. No Gemini calls.

Step 2 — parallel per-page OCR + Q&A extraction with intro skip on page 1.
One Gemini call per page (``extract_page`` from ``step2_page.py``). Each call
returns ``text``, ``is_intro`` (only meaningful on page 1), and the page's
per-page question records. A deterministic merger (``merge_per_page_questions``)
stitches the records into global ``items[]``.

Step 3 — per-question-id parallel marking. For each item, one Gemini call
(``mark_item`` from ``step3_mark.py``) that consumes the item, the grid-overlaid
pages spanning the item, and the matching model-answer-key entry, and returns
the item enriched with grading + remarks + anchor marks + annotations.

Public API:
  ``smart_ocr_run(pdf_path, api_key, language, answer_model, check_level, *, request_id)``
  returns ``{"items", "page_count", "skipped_pages"}``.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.gemini_copy_ocr import (
    copy_ocr_max_pages,
    copy_ocr_parallel_workers,
    copy_ocr_raster_dpi,
    count_pdf_pages,
    rasterize_pdf_to_png_pages,
)
from src.gemini_evaluate_student_answers import (
    evaluation_strictness_instruction,
    format_answer_model_as_teacher_instructions,
)
from src.grid_overlay import batch_draw_grid

from .config import SMART_OCR_TOTAL_TIMEOUT_S
from .dedup import deduplicate_page_texts
from .step2_page import extract_page
from .step3_mark import mark_item
from .structure import merge_per_page_questions

log = logging.getLogger(__name__)


def smart_ocr_run(
    pdf_path: Path,
    api_key: str | list[str],
    language: str,
    answer_model: dict[str, Any],
    check_level: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """End-to-end smart OCR. Returns ``{items, page_count, skipped_pages}``.

    ``api_key`` can be a single string (legacy) or a list of strings. When a list
    is passed, per-page (step 2) and per-question (step 3) submissions
    round-robin across the keys to split the per-key rate budget.
    """
    api_keys: list[str] = (
        [k for k in api_key if k] if isinstance(api_key, list) else [api_key]
    )
    if not api_keys:
        raise ValueError("smart_ocr_run requires at least one API key.")
    log.info("smart_ocr[%s] api_keys=%s", request_id, len(api_keys))
    run_started = time.monotonic()
    log.info("smart_ocr[%s] START", request_id)

    # ---------- Step 1: grid overlay ----------
    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    dpi = max(300, copy_ocr_raster_dpi())
    raster_started = time.monotonic()
    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=dpi, request_id=request_id)
    raster_elapsed = time.monotonic() - raster_started

    workers = max(1, min(copy_ocr_parallel_workers(), total_pages))

    grid_started = time.monotonic()
    grid_pages = batch_draw_grid(png_pages, max_workers=workers)
    grid_elapsed = time.monotonic() - grid_started
    log.info(
        "smart_ocr[%s] step1 done dpi=%s pages=%s raster=%.2fs grid=%.2fs",
        request_id, dpi, total_pages, raster_elapsed, grid_elapsed,
    )

    deadline = run_started + SMART_OCR_TOTAL_TIMEOUT_S

    # ---------- Step 2: parallel per-page OCR + Q&A ----------
    step2_started = time.monotonic()
    per_page: list[dict[str, Any]] = [
        {"text": "", "is_intro": False, "questions": []} for _ in range(total_pages)
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                extract_page,
                api_keys[i % len(api_keys)],
                grid_pages[i], i + 1, total_pages, language,
                is_first_page=(i == 0),
            ): i
            for i in range(total_pages)
        }
        for fut, i in futs.items():
            remaining = max(1.0, deadline - time.monotonic())
            try:
                per_page[i] = fut.result(timeout=remaining)
            except Exception as e:
                log.warning(
                    "smart_ocr[%s] step2 page=%s failed: %s",
                    request_id, i + 1, e,
                )

    skipped_pages: list[int] = []
    if per_page and per_page[0].get("is_intro"):
        log.info("smart_ocr[%s] page 1 classified intro/cover — skipping", request_id)
        skipped_pages.append(1)
        per_page[0] = {"text": "", "is_intro": True, "questions": []}

    # Optional similarity-dedup of OCR text (cheap, drops near-identical pages).
    page_blocks = [
        f"=== PAGE {i + 1} ===\n{p.get('text', '')}" for i, p in enumerate(per_page)
    ]
    page_blocks = deduplicate_page_texts(
        page_blocks, ["UNKNOWN"] * total_pages, request_id,
    )
    # Reflect dedup back onto per_page text (empties duplicates so step 3 won't see them).
    for i, block in enumerate(page_blocks):
        if "[OMITTED:" in block:
            per_page[i] = {"text": "", "is_intro": per_page[i].get("is_intro", False), "questions": []}

    items = merge_per_page_questions(per_page, total_pages)
    step2_elapsed = time.monotonic() - step2_started
    log.info(
        "smart_ocr[%s] step2 END pages=%s items=%s skipped=%s took=%.2fs",
        request_id, total_pages, len(items), skipped_pages, step2_elapsed,
    )

    if not items:
        raise ValueError("No answerable content detected.")

    # ---------- Step 3: per-question-id parallel marking ----------
    step3_started = time.monotonic()
    subject = (answer_model.get("title") or "General").strip() or "General"
    model_questions = answer_model.get("questions") or []
    teacher_instructions = format_answer_model_as_teacher_instructions(
        model_questions, subject,
    )
    strictness_line = evaluation_strictness_instruction(check_level)
    model_by_qid = _build_model_lookup(model_questions)

    item_qids = [it.get("question_id") for it in items]
    log.info(
        "smart_ocr[%s] step3 START model_questions=%s model_qids=%s item_qids=%s",
        request_id,
        len(model_questions),
        sorted(model_by_qid.keys()),
        item_qids,
    )

    def _job(item: dict[str, Any], job_index: int) -> tuple[Any, dict[str, Any]]:
        qid = item.get("question_id")
        start_page = int(item.get("start_page", 1))
        end_page = int(item.get("end_page", start_page))
        item_pages = list(range(start_page, end_page + 1))
        grid_subset = [grid_pages[p - 1] for p in item_pages]
        entry = model_by_qid.get(qid)
        # Rotate the key list so this item's per-page fallback (if any) starts on
        # a different key than the item's primary call — keeps load balanced
        # across keys when one item is a long essay.
        rotated_keys = api_keys[job_index % len(api_keys):] + api_keys[:job_index % len(api_keys)]
        marked = mark_item(
            rotated_keys, item, grid_subset, entry,
            subject=subject,
            teacher_instructions=teacher_instructions,
            check_level=check_level,
            strictness_line=strictness_line,
            language=language,
            request_id=request_id,
        )
        return qid, marked

    marked_by_qid: dict[Any, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, it, idx): it.get("question_id") for idx, it in enumerate(items)}
        for fut, qid in futs.items():
            remaining = max(1.0, deadline - time.monotonic())
            try:
                returned_qid, marked = fut.result(timeout=remaining)
                marked_by_qid[returned_qid] = marked
            except Exception as e:
                log.warning(
                    "smart_ocr[%s] step3 q=%s failed: %s",
                    request_id, qid, e,
                )

    def _ensure_shell(it: dict[str, Any]) -> dict[str, Any]:
        marked = marked_by_qid.get(it.get("question_id"))
        if marked is not None:
            return marked
        # Step-3 future failed (timeout / exception). Fill an empty shell so the
        # response is structurally consistent across items.
        entry = model_by_qid.get(it.get("question_id"))
        try:
            cap = float(entry.get("marks", 0)) if isinstance(entry, dict) else 0.0
        except (TypeError, ValueError):
            cap = 0.0
        it.setdefault("max_marks", cap)
        it.setdefault("marks_awarded", 0.0)
        it.setdefault("status", "partial")
        it.setdefault("feedback", "Grading failed (timeout or transient error); please re-run.")
        it.setdefault("student_answer_summary", "")
        it.setdefault("anchor_marks", [])
        it.setdefault("remarks", [])
        it.setdefault("annotations", [])
        return it

    items = [_ensure_shell(it) for it in items]
    step3_elapsed = time.monotonic() - step3_started
    log.info(
        "smart_ocr[%s] step3 END items=%s took=%.2fs",
        request_id, len(items), step3_elapsed,
    )

    total_elapsed = time.monotonic() - run_started
    log.info(
        "smart_ocr[%s] DONE step1=%.2fs step2=%.2fs step3=%.2fs total=%.2fs items=%s pages=%s skipped=%s",
        request_id,
        raster_elapsed + grid_elapsed, step2_elapsed, step3_elapsed, total_elapsed,
        len(items), total_pages, skipped_pages,
    )

    _truncate_student_answers_for_response(items)

    return {
        "items": items,
        "page_count": total_pages,
        "skipped_pages": skipped_pages,
    }


_RESPONSE_STUDENT_ANSWER_MAX_CHARS_DEFAULT = 4000


def _response_student_answer_max_chars() -> int:
    import os
    try:
        return max(0, int(os.environ.get(
            "SMART_OCR_RESPONSE_STUDENT_ANSWER_MAX_CHARS",
            _RESPONSE_STUDENT_ANSWER_MAX_CHARS_DEFAULT,
        )))
    except ValueError:
        return _RESPONSE_STUDENT_ANSWER_MAX_CHARS_DEFAULT


def _truncate_student_answers_for_response(items: list[dict[str, Any]]) -> None:
    """Cap ``student_answer`` on the outbound response only.

    The full text was already sent to Gemini in step 2 → step 3 (so grading
    context is preserved); this only shortens the JSON we hand back to the
    client. ``0`` disables truncation. Truncation keeps the head + tail with a
    middle ellipsis so first paragraph and conclusion are both visible.
    """
    cap = _response_student_answer_max_chars()
    if cap <= 0:
        return
    for it in items:
        text = it.get("student_answer") or ""
        if len(text) <= cap:
            continue
        head_n = cap // 2
        tail_n = cap - head_n
        head = text[:head_n].rstrip()
        tail = text[-tail_n:].lstrip()
        it["student_answer"] = (
            f"{head}\n\n[…truncated for response: {len(text)} chars total, "
            f"showing first {head_n} + last {tail_n}…]\n\n{tail}"
        )


_DIGIT_RE = re.compile(r"\d+")


def _build_model_lookup(model_questions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index model questions by the integer parsed out of ``questionNo``.

    Model storage writes ``questionNo`` as ``"Q1"``, ``"Q2"``, … (via service.py).
    We strip the ``Q`` prefix and any whitespace before parsing so the lookup
    matches step-2 / step-3 integer ``question_id`` values.
    """
    out: dict[int, dict[str, Any]] = {}
    for idx, q in enumerate(model_questions or [], start=1):
        if not isinstance(q, dict):
            continue
        # Try the explicit number fields first, in order of canonicity.
        raw_candidates = (
            q.get("questionNo"),
            q.get("question_no"),
            q.get("question_id"),
        )
        qid_int: int | None = None
        for raw in raw_candidates:
            if raw is None:
                continue
            m = _DIGIT_RE.search(str(raw))
            if m:
                try:
                    qid_int = int(m.group(0))
                    break
                except ValueError:
                    continue
        # Fallback: use the position (1-based) when the entry has no parseable label.
        if qid_int is None:
            qid_int = idx
        if qid_int not in out:
            out[qid_int] = q
    return out
