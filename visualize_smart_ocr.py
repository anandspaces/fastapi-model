#!/usr/bin/env python3
"""
Intercept every Gemini call in the smart-OCR pipeline and dump it to disk.

Run from repo root:
    python visualize_smart_ocr.py --pdf test_original_pdf1.pdf
    python visualize_smart_ocr.py --validate test-smart-ocr/foo/stage4_final/final_response.json

Uses the same cell-native Smart-OCR pipeline as ``POST /analyse/smart-ocr`` (block OCR → structure → resolve → optional pass-3 grading).

Creates test-smart-ocr/ with one folder per stage:
  model_answer/stage0_extract/    — model PDF question extraction
  {pdf_stem}/stage1a_classify/page{N}/  — page classification
  {pdf_stem}/stage1b_ocr/page{N}/       — type-aware OCR
  {pdf_stem}/stage1c_dedup/             — deduplication summary
  {pdf_stem}/stage2_structure/          — structure pass
  {pdf_stem}/stage2p5_cell_grid/        — v4 cell grid, overlay PDF, JPEGs
  {pdf_stem}/stage3_evaluate/           — grading (+ overlay_prompt.txt when --overlay)
  {pdf_stem}/stage3p5_placement/       — items after assign_cell_ids_v4
  {pdf_stem}/stage4_final/              — final_response.json (API-shaped)

Each stage folder may contain stage_status.json for robustness tracing.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ── repo root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("viz")

OUT = ROOT / "test-smart-ocr"
OUT.mkdir(exist_ok=True)

# ── thread-local stage context (set in wrapper fns, read by tracing client) ─
_tls = threading.local()
CURRENT_PDF: str = ""  # set before each top-level call; threads only read it


def _stage() -> str:
    return getattr(_tls, "stage", "unknown")


def _page() -> int | None:
    return getattr(_tls, "page", None)


def _set(stage: str, page: int | None = None) -> None:
    _tls.stage = stage
    _tls.page = page


def _w(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _j(path: Path, obj: object) -> None:
    _w(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _write_stage_status(
    path: Path,
    status: str,
    *,
    elapsed_ms: int = 0,
    error_msg: str | None = None,
    skipped_reason: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "elapsed_ms": elapsed_ms,
    }
    if error_msg:
        payload["error_msg"] = error_msg
    if skipped_reason:
        payload["skipped_reason"] = skipped_reason
    _j(path, payload)


# ── patch genai.Client BEFORE importing any src module ─────────────────────
from google import genai
from google.genai import types as gtypes

_RealClient = genai.Client


def _record(contents: object, raw: str, model: str) -> None:
    """Save one Gemini call to the current stage folder."""
    stage = _stage()
    page = _page()
    if stage == "unknown":
        return

    base = OUT / CURRENT_PDF / stage / (f"page{page:02d}" if page is not None else "")
    base.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    img_idx = 0
    for part in (contents if isinstance(contents, list) else [contents]):
        if isinstance(part, str):
            texts.append(part)
        elif hasattr(part, "text") and part.text:
            texts.append(part.text)
        elif hasattr(part, "inline_data") and part.inline_data:
            mime = getattr(part.inline_data, "mime_type", "image/png")
            ext = "jpg" if "jpeg" in mime else "png"
            img_path = base / f"image_{img_idx:02d}.{ext}"
            _w(img_path, part.inline_data.data)
            texts.append(f"[IMAGE saved → {img_path.name}  ({len(part.inline_data.data):,} bytes)]")
            img_idx += 1
        else:
            texts.append(f"[{type(part).__name__}: {getattr(part, 'uri', getattr(part, 'name', '?'))}]")

    _w(base / "prompt.txt", "\n\n────────────── next part ──────────────\n\n".join(texts))
    _w(base / "response_raw.json", raw)
    try:
        _j(base / "response_parsed.json", json.loads(raw))
    except Exception:
        pass
    _j(
        base / "meta.json",
        {
            "model": model,
            "stage": stage,
            "page": page,
            "prompt_parts": len(texts),
            "images": img_idx,
            "response_chars": len(raw),
        },
    )


class _TracingClient(_RealClient):
    """genai.Client subclass that records every generate_content call."""

    def __init__(self, api_key: str | None = None, **kw: object) -> None:
        super().__init__(api_key=api_key, **kw)
        _orig = self.models.generate_content

        def _traced(model: str, contents: object, config: object = None, **kwargs: object):
            resp = _orig(model=model, contents=contents, config=config, **kwargs)
            raw = (getattr(resp, "text", None) or "").strip()
            _record(contents, raw, model)
            return resp

        self.models.generate_content = _traced


genai.Client = _TracingClient  # type: ignore[assignment]

# ── import src modules (they now get the patched Client) ────────────────────
from src.gemini_extract import (
    MODEL_ID,
    SCHEMA_PROMPT,
    _EXTRACT_RESPONSE_SCHEMA,
    wait_for_file_active,
    extract_json_array,
    normalize,
    _assign_page_nums,
    _apply_default_marks_fallback,
)
import src.gemini_smart_ocr as _sm
from src.gemini_smart_ocr import _extract_page_body
from src.gemini_smart_ocr_v2 import (
    grade_items_pass3_v2,
    smart_ocr_extract_student_answers_v2,
)
from src.gemini_smart_ocr_v2.occupancy_map import write_occupancy_map_md
import src.gemini_evaluate_student_answers as _ev
from src.gemini_evaluate_student_answers import (
    format_answer_model_as_teacher_instructions,
    student_items_for_grading,
    merge_evaluations_into_items,
    _build_evaluation_prompt_with_overlay,
)

from cell_overlay_renderer import render_overlay_pngs
from src.cell_grid_service import (
    build_cell_grid,
    build_overlay_pdf,
    cell_grid_meta_payload,
    cell_grid_response_payload,
)
from src.cell_response_formatter import build_response_items
from remark_cell_layout_service import (
    REMARK_FONTNAME_EN,
    REMARK_FONTNAME_HI,
    REMARK_FONT_SIZE_PTS,
    REMARK_MAX_WRAP_ROWS,
    assign_cell_ids_v4,
)
from cell_grid_service_v4 import PageCellGrid


# ── wrap module functions to set stage context before each Gemini call ───────

_orig_classify = _sm._classify_page
_orig_ocr = _sm._ocr_single_page
_orig_struct = _sm._structure_qa_with_fallback
_orig_dedup = _sm._deduplicate_page_texts
_orig_eval = _ev.evaluate_student_answers_against_model


def _wp_classify(api_key: str, png: bytes, page_num: int, total_pages: int, language: str) -> str:
    _set("stage1a_classify", page_num)
    return _orig_classify(api_key, png, page_num, total_pages, language)


def _wp_ocr(
    api_key: str,
    png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
    page_type: str,
    *,
    overlay_jpeg: bytes | None = None,
) -> str:
    _set("stage1b_ocr", page_num)
    block = _orig_ocr(api_key, png, page_num, total_pages, language, page_type, overlay_jpeg=overlay_jpeg)
    text = _extract_page_body(block)
    _w(OUT / CURRENT_PDF / "stage1b_ocr" / f"page{page_num:02d}" / "extracted_text.txt", text)
    return block


def _wp_dedup(page_blocks: list[str], classifications: list[str], request_id: str, *, sim_threshold: float = 0.92) -> list[str]:
    result = _orig_dedup(page_blocks, classifications, request_id, sim_threshold=sim_threshold)
    d = OUT / CURRENT_PDF / "stage1c_dedup"
    d.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}
    for i, (orig, deduped) in enumerate(zip(page_blocks, result), 1):
        omitted = "[OMITTED:" in deduped and "[OMITTED:" not in orig
        summary[f"page{i:02d}"] = {
            "kept": not omitted,
            "classification": classifications[i - 1] if i - 1 < len(classifications) else "?",
            "orig_chars": len(orig),
            "after_chars": len(deduped),
            "note": deduped.split("\n")[1].strip() if omitted else "",
        }
    _j(d / "dedup_summary.json", summary)
    _j(
        d / "page_texts_after.json",
        {f"page{i:02d}": _extract_page_body(b) for i, b in enumerate(result, 1)},
    )
    return result


def _wp_struct(
    client: object,
    pages_payload: dict,
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
    overlay_images: list[bytes] | None = None,
) -> list[dict]:
    _set("stage2_structure")
    d = OUT / CURRENT_PDF / "stage2_structure"
    d.mkdir(parents=True, exist_ok=True)

    _j(d / "pages_payload_input.json", pages_payload)
    _j(
        d / "meta.json",
        {
            "language": language,
            "total_pages": total_pages,
            "expected_questions": expected_questions,
            "payload_chars": len(json.dumps(pages_payload)),
        },
    )

    result = _orig_struct(
        client, pages_payload, language, total_pages,
        expected_questions=expected_questions,
        overlay_images=overlay_images,
    )

    _j(d / "parsed_items.json", result)
    _j(d / "summary.json", {"questions_found": len(result), "has_overlay": bool(overlay_images)})
    return result


def _wp_eval(
    api_key: str,
    subject: str,
    teacher_instructions: str,
    student_items: list[dict],
    *,
    request_id: str,
    check_level: str = "Moderate",
    overlay_images: list[bytes] | None = None,
    use_overlay_prompt: bool = False,
) -> list[dict]:
    _set("stage3_evaluate")
    d = OUT / CURRENT_PDF / "stage3_evaluate"
    d.mkdir(parents=True, exist_ok=True)

    student_json = json.dumps(student_items, ensure_ascii=False)
    if use_overlay_prompt and overlay_images:
        try:
            overlay_txt = _build_evaluation_prompt_with_overlay(
                subject, teacher_instructions, student_json, check_level=check_level
            )
            _w(d / "overlay_prompt.txt", overlay_txt)
        except Exception as exc:
            log.warning("Could not save overlay_prompt.txt: %s", exc)

    _w(d / "teacher_instructions.txt", teacher_instructions)
    _j(d / "student_input.json", student_items)
    _j(
        d / "meta.json",
        {
            "subject": subject,
            "check_level": check_level,
            "n_student_items": len(student_items),
            "use_overlay": use_overlay_prompt and bool(overlay_images),
        },
    )

    result = _orig_eval(
        api_key,
        subject,
        teacher_instructions,
        student_items,
        request_id=request_id,
        check_level=check_level,
        overlay_images=overlay_images,
        use_overlay_prompt=use_overlay_prompt and bool(overlay_images),
    )

    _j(d / "parsed_evaluations.json", result)
    _write_stage_status(d / "stage_status.json", "ok", elapsed_ms=0)
    return result


# inject all patches into the modules they live in
_sm._classify_page = _wp_classify  # type: ignore[assignment]
_sm._ocr_single_page = _wp_ocr  # type: ignore[assignment]
_sm._deduplicate_page_texts = _wp_dedup  # type: ignore[assignment]
_sm._structure_qa_with_fallback = _wp_struct  # type: ignore[assignment]
_ev.evaluate_student_answers_against_model = _wp_eval  # type: ignore[assignment]


def _smart_ocr_skipped_pages(page_count: int, items: list) -> list[int]:
    """Mirror main.py — pages not covered by any structured row with section_name."""
    covered: set[int] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        sec = raw.get("section_name")
        if sec is None or (isinstance(sec, str) and not sec.strip()):
            continue
        try:
            sp = int(raw.get("start_page", 1))
            ep = int(raw.get("end_page", sp))
        except (TypeError, ValueError):
            continue
        if ep < sp:
            ep = sp
        sp = max(1, min(sp, page_count))
        ep = max(1, min(ep, page_count))
        for p in range(sp, ep + 1):
            covered.add(p)
    return [p for p in range(1, page_count + 1) if p not in covered]


def _wp_cell_grid(pdf_bytes: bytes, pdf_stem: str) -> tuple[list[PageCellGrid], list[bytes]]:
    """Stage 2.5 — build v4 grid, save artifacts.

    Returns ``(page_grids, overlay_jpegs)``. On failure returns ``([], [])``.
    """
    d = OUT / pdf_stem / "stage2p5_cell_grid"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        page_grids = build_cell_grid(pdf_bytes)
        _j(d / "cell_grid_meta.json", cell_grid_meta_payload(page_grids))
        _j(
            d / "cell_grid_full.json",
            cell_grid_response_payload(page_grids, include_cells=False),
        )
        overlay_pdf_path = d / "grid_overlay.pdf"
        build_overlay_pdf(pdf_bytes, page_grids, overlay_pdf_path)

        overlay_images = render_overlay_pngs(
            pdf_bytes,
            page_grids,
            dpi=150,
            image_format="jpeg",
            jpeg_quality=85,
            label_every_cell=True,
        )
        for i, img in enumerate(overlay_images, 1):
            _w(d / f"overlay_page{i:02d}.jpg", img)

        lines = ["# Writable summary", "", "| page | runs | regions | sample cells |", "| --- | --- | --- | --- |"]
        for g in page_grids:
            sample: list[str] = []
            for r in g.runs[:5]:
                sample.extend([r.start_cell_id, r.end_cell_id])
            for reg in g.regions[:2]:
                sample.extend(reg.cell_ids[:6])
            sample = list(dict.fromkeys(sample))[:10]
            lines.append(
                f"| {g.page} | {len(g.runs)} | {len(g.regions)} | {', '.join(sample) or '—'} |"
            )
        _w(d / "writable_summary.md", "\n".join(lines) + "\n")

        elapsed = int((time.perf_counter() - t0) * 1000)
        _write_stage_status(d / "stage_status.json", "ok", elapsed_ms=elapsed)
        log.info("[%s] stage2p5_cell_grid ok pages=%s", pdf_stem, len(page_grids))
        return page_grids, overlay_images
    except Exception as exc:
        log.exception("[%s] stage2p5_cell_grid failed: %s", pdf_stem, exc)
        elapsed = int((time.perf_counter() - t0) * 1000)
        _write_stage_status(
            d / "stage_status.json",
            "error",
            elapsed_ms=elapsed,
            error_msg=str(exc),
        )
        return [], []


def _wp_placement_save(pdf_stem: str, items: list[dict]) -> None:
    """Save deep copy of items after assign_cell_ids_v4 (internal fields retained)."""
    d = OUT / pdf_stem / "stage3p5_placement"
    d.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = copy.deepcopy(items)
        _j(d / "items_after_placement.json", snapshot)

        lines = ["# Placement summary", ""]
        for it in items:
            if not isinstance(it, dict):
                continue
            qid = it.get("question_id")
            anns = it.get("annotations") or []
            if not isinstance(anns, list):
                continue
            lines.append(f"## question_id={qid}")
            for a in anns:
                if not isinstance(a, dict):
                    continue
                comm = str(a.get("comment") or "")[:60]
                tier = a.get("placement_tier", "?")
                rid = a.get("range_id", "")
                cids = a.get("cell_ids", [])
                pg = a.get("page", a.get("page_index", 0))
                lines.append(f"- page={pg} tier={tier} range={rid} cells={cids}")
                lines.append(f"  `{comm}`")
            lines.append("")
        _w(d / "placement_summary.md", "\n".join(lines))
        _write_stage_status(d / "stage_status.json", "ok")
    except Exception as exc:
        log.exception("[%s] placement save failed: %s", pdf_stem, exc)
        _write_stage_status(
            d / "stage_status.json",
            "error",
            error_msg=str(exc),
        )


def _normalize_final_response(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Ensure API-shaped payload; returns (fixed_dict, warnings)."""
    warnings: list[str] = []
    if not isinstance(data, dict):
        warnings.append("root not object — wrapping")
        data = {"status": 1, "message": "Smart OCR complete.", "data": {}}

    if "status" not in data:
        data["status"] = 1
    if "message" not in data:
        data["message"] = "Smart OCR complete."
    inner = data.get("data")
    if not isinstance(inner, dict):
        inner = {}
        data["data"] = inner
        warnings.append("missing data object")

    if "pageCount" not in inner:
        inner["pageCount"] = max(1, int(inner.get("page_count") or 1))
        warnings.append("filled pageCount")
    if "items" not in inner:
        inner["items"] = []
        warnings.append("filled items")
    if "skippedPages" not in inner:
        inner["skippedPages"] = []
    if "cellGridMeta" not in inner:
        inner["cellGridMeta"] = []

    for it in inner["items"]:
        if not isinstance(it, dict):
            continue
        if "question_id" not in it:
            it["question_id"] = 0
            warnings.append("item missing question_id")
        if "answer_span" not in it:
            it["answer_span"] = []
        if "marking" not in it:
            it["marking"] = None
        if "annotations" not in it:
            it["annotations"] = []
        for ann in it["annotations"]:
            if not isinstance(ann, dict):
                continue
            if "page" not in ann:
                ann["page"] = 1
            if "comment" not in ann:
                ann["comment"] = ""
            if "comment_rows" not in ann:
                ann["comment_rows"] = []
            if "anchor" not in ann:
                ann["anchor"] = {"type": "none"}
            elif isinstance(ann["anchor"], dict) and "type" not in ann["anchor"]:
                ann["anchor"]["type"] = "none"

    return data, warnings


def validate_final_response_file(path: Path) -> bool:
    """Validate final_response.json shape; print warnings. Returns True if readable."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Cannot read JSON %s: %s", path, exc)
        return False
    _, warnings = _normalize_final_response(raw)
    if warnings:
        for w in warnings:
            log.warning("[validate] %s", w)
    log.info("[validate] OK %s", path)
    return True


def _wp_format(
    pdf_stem: str,
    page_count: int,
    items: list[dict],
    page_grids: list[PageCellGrid],
    *,
    check_level: str = "Moderate",
    model_id: str | None = None,
) -> dict[str, Any]:
    """Stage 4 — build wire items + full API payload."""
    d = OUT / pdf_stem / "stage4_final"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        skipped = _smart_ocr_skipped_pages(page_count, items)
        response_items = build_response_items(items, page_grids)
        meta = cell_grid_meta_payload(page_grids)

        payload: dict[str, Any] = {
            "status": 1,
            "message": "Smart OCR complete.",
            "data": {
                "pageCount": page_count,
                "items": response_items,
                "skippedPages": skipped,
                "cellGridMeta": meta,
                "checkLevel": check_level,
            },
        }
        if model_id:
            payload["data"]["modelId"] = model_id

        fixed, warnings = _normalize_final_response(payload)
        for w in warnings:
            log.warning("[%s] format normalize: %s", pdf_stem, w)

        _j(d / "final_response.json", fixed)

        map_lines = ["# Cell / comment map", ""]
        for it in fixed["data"]["items"]:
            qid = it.get("question_id")
            for ann in it.get("annotations") or []:
                if not isinstance(ann, dict):
                    continue
                cr = ann.get("comment_range", "")
                cr_rows = ann.get("comment_rows")
                anchor = ann.get("anchor") or {}
                pos = ann.get("is_positive")
                p = ann.get("page")
                line = f"- Q{qid} p{p} pos={pos} range={cr or cr_rows} anchor={anchor.get('type')}"
                map_lines.append(line)
            map_lines.append("")
        _w(d / "cell_id_map.md", "\n".join(map_lines))

        elapsed = int((time.perf_counter() - t0) * 1000)
        _write_stage_status(d / "stage_status.json", "ok", elapsed_ms=elapsed)
        return fixed
    except Exception as exc:
        log.exception("[%s] stage4_final failed: %s", pdf_stem, exc)
        elapsed = int((time.perf_counter() - t0) * 1000)
        _write_stage_status(
            d / "stage_status.json",
            "error",
            elapsed_ms=elapsed,
            error_msg=str(exc),
        )
        # minimal fallback file
        fallback = {
            "status": 1,
            "message": "Smart OCR complete.",
            "data": {
                "pageCount": max(1, page_count),
                "items": [],
                "skippedPages": [],
                "cellGridMeta": cell_grid_meta_payload(page_grids),
                "checkLevel": check_level,
            },
        }
        _j(d / "final_response.json", fallback)
        return fallback


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_ok_ocr(pdf_stem: str) -> dict[str, Any] | None:
    p = OUT / pdf_stem / "ocr_result.json"
    data = _load_json(p)
    if not isinstance(data, dict):
        return None
    if "items" not in data or "page_count" not in data:
        return None
    return data


def _checkpoint_ok_eval(pdf_stem: str) -> list[dict] | None:
    p = OUT / pdf_stem / "stage3_evaluate" / "parsed_evaluations.json"
    data = _load_json(p)
    if not isinstance(data, list):
        return None
    return data


def _checkpoint_ok_place(pdf_stem: str) -> list[dict] | None:
    p = OUT / pdf_stem / "stage3p5_placement" / "items_after_placement.json"
    data = _load_json(p)
    if not isinstance(data, list):
        return None
    return data


# ── Stage 0: model answer PDF → extract questions ───────────────────────────


def run_model_extraction(pdf_path: Path) -> list[dict]:
    global CURRENT_PDF
    CURRENT_PDF = "model_answer"
    _set("stage0_extract")

    d = OUT / "model_answer" / "stage0_extract"
    d.mkdir(parents=True, exist_ok=True)
    _w(d / "extraction_prompt.txt", SCHEMA_PROMPT)

    client = genai.Client(api_key=API_KEY)
    log.info("[Stage 0] Uploading %s to Gemini File API …", pdf_path.name)
    up = client.files.upload(file=str(pdf_path))
    if not up.uri or str(getattr(up, "state", "")).upper() == "PROCESSING":
        up = wait_for_file_active(client, up)

    _j(
        d / "file_upload_info.json",
        {
            "uri": str(getattr(up, "uri", "")),
            "name": str(getattr(up, "name", "")),
            "state": str(getattr(up, "state", "")),
            "mime_type": str(getattr(up, "mime_type", "")),
        },
    )
    log.info("[Stage 0] Calling %s for question extraction …", MODEL_ID)
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=[up, SCHEMA_PROMPT],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_EXTRACT_RESPONSE_SCHEMA,
        ),
    )
    raw = (getattr(resp, "text", None) or "").strip()

    qs = [normalize(o) for o in extract_json_array(raw)]
    _assign_page_nums(qs)
    _apply_default_marks_fallback(qs)
    _j(d / "parsed_questions.json", qs)
    log.info("[Stage 0] Extracted %d questions from %s", len(qs), pdf_path.name)
    return qs


def run_pdf_pipeline(
    pdf_path: Path,
    model_questions: list[dict],
    *,
    title: str = "UPSC",
    language: str = "en",
    use_overlay: bool = True,
    from_stage: str | None = None,
    check_level: str = "Moderate",
) -> None:
    global CURRENT_PDF
    CURRENT_PDF = pdf_path.stem
    pdf_out = OUT / CURRENT_PDF
    pdf_out.mkdir(parents=True, exist_ok=True)
    pdf_bytes = pdf_path.read_bytes()

    remark_font = REMARK_FONTNAME_HI if language.strip().lower() == "hi" else REMARK_FONTNAME_EN

    log.info("═══ Pipeline: %s (lang=%s overlay=%s from_stage=%s) ═══", pdf_path.name, language, use_overlay, from_stage)

    items: list[dict] = []
    page_count = 1
    evals: list[dict] = []
    skip_ocr = False
    skip_eval = False

    if from_stage == "place":
        loaded = _checkpoint_ok_place(CURRENT_PDF)
        if loaded is None:
            log.warning("WARN: checkpoint missing/invalid — need stage3p5_placement/items_after_placement.json")
            sys.exit(1)
        items = loaded
        ocr_fb = _checkpoint_ok_ocr(CURRENT_PDF) or {}
        page_count = max(1, int(ocr_fb.get("page_count") or 1))
        page_grids, _ = _wp_cell_grid(pdf_bytes, CURRENT_PDF)
        _wp_format(
            CURRENT_PDF,
            page_count,
            items,
            page_grids,
            check_level=check_level,
            model_id=None,
        )
        log.info("[%s] Done (from_stage=place).", CURRENT_PDF)
        return

    if from_stage == "eval":
        ocr_data = _checkpoint_ok_ocr(CURRENT_PDF)
        evals_file = _checkpoint_ok_eval(CURRENT_PDF)
        if ocr_data is None or evals_file is None:
            log.warning("WARN: checkpoint missing — need ocr_result.json + stage3_evaluate/parsed_evaluations.json")
            sys.exit(1)
        items = list(ocr_data.get("items") or [])
        page_count = int(ocr_data.get("page_count") or 1)
        merge_evaluations_into_items(items, evals_file)
        skip_ocr = True
        skip_eval = True
    elif from_stage == "grid":
        ocr_data = _checkpoint_ok_ocr(CURRENT_PDF)
        if ocr_data is None:
            log.warning("WARN: ocr checkpoint missing — running full OCR pipeline")
        else:
            items = list(ocr_data.get("items") or [])
            page_count = int(ocr_data.get("page_count") or 1)
            skip_ocr = True
    elif from_stage == "ocr":
        ocr_data = _checkpoint_ok_ocr(CURRENT_PDF)
        if ocr_data is None:
            log.warning("WARN: ocr checkpoint missing/invalid — running full OCR pipeline")
        else:
            items = list(ocr_data.get("items") or [])
            page_count = int(ocr_data.get("page_count") or 1)
            skip_ocr = True

    # ── Stage 2.5: cell grid (always built first so overlays can be sent to OCR)
    page_grids, overlay_jpegs = _wp_cell_grid(pdf_bytes, CURRENT_PDF)

    # Determine whether overlay images should be forwarded to Gemini.
    overlay_images: list[bytes] | None = None
    use_overlay_flag = False
    if use_overlay and page_grids and overlay_jpegs:
        overlay_images = overlay_jpegs
        use_overlay_flag = True
    elif use_overlay and not page_grids:
        log.warning("[%s] No cell grid — overlay prompt skipped", CURRENT_PDF)

    ocr_result: dict[str, Any] | None = None

    if not skip_ocr:
        lang = language.strip().lower() if language else "en"
        _set("stage1_block_ocr", None)
        ocr_result = smart_ocr_extract_student_answers_v2(
            pdf_path,
            API_KEY,
            lang,
            request_id=pdf_path.stem,
            overlay_images=overlay_images,
            grids=page_grids,
        )
        _set("stage2_block_structure", None)
        _j(
            pdf_out / "stage2_block_structure" / "ocr_bundle.json",
            {
                "items": ocr_result.get("items"),
                "page_count": ocr_result.get("page_count"),
                "_flat_blocks": ocr_result.get("_flat_blocks"),
            },
        )
        items = ocr_result["items"]
        page_count = int(ocr_result.get("page_count") or 1)
        log.info("[%s] OCR complete: %d questions, %d pages", CURRENT_PDF, len(items), page_count)
        save_ocr = dict(ocr_result)
        save_ocr.pop("_overlay_jpegs", None)
        _j(pdf_out / "ocr_result.json", save_ocr)
    else:
        _j(
            pdf_out / "ocr_result.json",
            {"items": items, "page_count": page_count},
        )
        ocr_result = _checkpoint_ok_ocr(CURRENT_PDF)

    if model_questions and not skip_eval:
        teacher_instr = format_answer_model_as_teacher_instructions(model_questions, title)
        graded_items = student_items_for_grading(items)
        log.info("[%s] Evaluating %d attempted items …", CURRENT_PDF, len(graded_items))
        try:
            flat_fb = (
                ocr_result.get("_flat_blocks") if isinstance(ocr_result, dict) else None
            )
            overlay_pass3 = overlay_images or (
                (ocr_result.get("_overlay_jpegs") if isinstance(ocr_result, dict) else None)
            )
            if flat_fb and overlay_pass3:
                _set("stage3_block_grade", None)
                evals = grade_items_pass3_v2(
                    API_KEY,
                    title,
                    title,
                    model_questions,
                    graded_items,
                    page_grids,
                    flat_fb,
                    overlay_pass3,
                    request_id=CURRENT_PDF,
                    check_level=check_level,
                )
                items = merge_evaluations_into_items(items, evals)
                write_occupancy_map_md(
                    pdf_out / "stage3_block_grade" / "occupancy_map.md",
                    page_grids,
                    flat_fb,
                    items,
                )
                log.info("[%s] Evaluation done (v2): %d graded rows", CURRENT_PDF, len(evals))
            else:
                evals = _ev.evaluate_student_answers_against_model(
                    API_KEY,
                    title,
                    teacher_instr,
                    graded_items,
                    request_id=CURRENT_PDF,
                    check_level=check_level,
                    overlay_images=overlay_images,
                    use_overlay_prompt=use_overlay_flag,
                )
                merged = merge_evaluations_into_items(items, evals)
                items = merged
                log.info("[%s] Evaluation done: %d graded rows", CURRENT_PDF, len(evals))
        except Exception as exc:
            log.exception("[%s] Evaluation failed — continuing without grades: %s", CURRENT_PDF, exc)
            stage3 = OUT / CURRENT_PDF / "stage3_evaluate"
            stage3.mkdir(parents=True, exist_ok=True)
            _write_stage_status(
                stage3 / "stage_status.json",
                "error",
                error_msg=str(exc),
            )
    elif model_questions and skip_eval:
        pass

    try:
        assign_cell_ids_v4(
            items,
            page_grids,
            font_size_pts=REMARK_FONT_SIZE_PTS,
            fontname=remark_font,
            max_wrap_rows=REMARK_MAX_WRAP_ROWS,
        )
    except Exception as exc:
        log.exception("[%s] assign_cell_ids_v4 failed: %s", CURRENT_PDF, exc)

    _wp_placement_save(CURRENT_PDF, items)

    inner = _wp_format(
        CURRENT_PDF,
        page_count,
        items,
        page_grids,
        check_level=check_level,
        model_id=None,
    )

    _j(
        pdf_out / "final_result.json",
        {"items": inner["data"]["items"], "page_count": page_count},
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart-OCR visualization & regression harness.")
    p.add_argument(
        "--pdf",
        dest="pdfs",
        action="append",
        metavar="PATH",
        help="PDF to process (repeatable). Default: test_original_pdf2.pdf",
    )
    p.add_argument(
        "pdf_positional",
        nargs="*",
        metavar="PDF",
        help="PDF paths (alternative to --pdf).",
    )
    p.add_argument(
        "--from-stage",
        choices=("ocr", "grid", "eval", "place"),
        default=None,
        help="Resume from saved checkpoint under test-smart-ocr/<stem>/",
    )
    p.add_argument("--lang", default="en", help="OCR language: en or hi (default: en).")
    p.add_argument(
        "--validate",
        metavar="PATH",
        help="Validate a final_response.json and exit (no Gemini).",
    )
    p.add_argument("--title", default="UPSC", help="Subject title for teacher instructions.")
    p.add_argument(
        "--check-level",
        default="Moderate",
        choices=("Moderate", "Hard"),
        help="Evaluator strictness.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.validate:
        ok = validate_final_response_file(Path(args.validate))
        sys.exit(0 if ok else 1)

    need_gemini = args.from_stage not in ("place", "eval")
    if need_gemini and not API_KEY:
        sys.exit("ERROR: GEMINI_API_KEY not set in .env or environment")

    cached = OUT / "model_answer" / "stage0_extract" / "parsed_questions.json"
    if cached.exists():
        model_questions: list[dict] = json.loads(cached.read_text(encoding="utf-8"))
        log.info("[Stage 0] Loaded %d questions from cache", len(model_questions))
    else:
        model_pdf = ROOT / "model_answer.pdf"
        model_questions = []
        if model_pdf.exists():
            model_questions = run_model_extraction(model_pdf)
        else:
            log.warning("model_answer.pdf not found — skipping Stage 0 (no grading key)")

    pdfs = list(args.pdfs or [])
    pdfs.extend(str(p) for p in (args.pdf_positional or []))
    if not pdfs:
        pdfs = [str(ROOT / "test_original_pdf2.pdf")]

    for pdf_s in pdfs:
        pdf_path = Path(pdf_s)
        if not pdf_path.is_file():
            log.warning("%s not found — skipping", pdf_path)
            continue
        run_pdf_pipeline(
            pdf_path,
            model_questions,
            title=args.title,
            language=args.lang,
            use_overlay=True,
            from_stage=args.from_stage,
            check_level=args.check_level,
        )

    log.info("All done. Open test-smart-ocr/ to inspect every Gemini call.")
    _print_tree()


def _print_tree() -> None:
    print("\n── test-smart-ocr/ contents ──")
    for p in sorted(OUT.rglob("*")):
        rel = p.relative_to(OUT)
        depth = len(rel.parts) - 1
        icon = "📁" if p.is_dir() else "📄"
        print("  " + "  " * depth + icon + " " + p.name)


if __name__ == "__main__":
    main()
