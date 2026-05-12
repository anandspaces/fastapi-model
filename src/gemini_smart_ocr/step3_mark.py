"""Step 3 — per-question-id grading + remarking + anchor remarking.

For each item the orchestrator passes:
  * the item dict (question_id, question, student_answer, start_page, end_page,
    start_y/end_y_position_percent, …),
  * the grid-overlaid PNG bytes for each page in the item's range,
  * the matching ``answer_model.questions[]`` entry (subject title + question entry
    with desc/instruction_name/marks),
  * the canonical ``check_level`` ("Moderate" | "Hard") and language.

We call Gemini once per item (or once-per-page on a fallback path for very long
items) and return the same item dict enriched with ``marks_awarded``, ``max_marks``,
``status``, ``feedback``, ``student_answer_summary``, ``anchor_marks`` (in %
coords), ``remarks`` (in % coords with model-key-aware comments and connector
fields ready for the frontend), and ``annotations``.

Grid-coordinate correctness is enforced by:
  * page-labelled image parts in ``contents``;
  * required ``page`` field in the response schema;
  * post-process clamp / free-zone / page-range / content-zone validation in
    ``_validate_step3_response`` before any grid→% conversion;
  * per-page grid→% conversion via ``annotations_grid_to_pct`` so the page tag
    survives onto every output mark;
  * a final ``clip_marks_to_answer`` pass keyed off the item's y-bounds.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from .config import (
    CONTENT_X_MAX,
    CONTENT_X_MIN,
    REMARK_ANSWER_GUARD_PCT,
    RETRY_BACKOFF_S,
    STEP3_HTTP_TIMEOUT_MS,
    STEP3_MAX_OUTPUT_TOKENS,
    STEP3_MULTI_PAGE_THRESHOLD,
    afc_off,
    finish_reason_name,
    http_opts,
    step3_model,
    thinking_off,
)
from .layout import (
    annotations_grid_to_pct,
    clip_marks_to_answer,
    nudge_mark_past_remarks,
    recalc_connector,
    spread_remarks_two_column,
)
from .parsing import parse_json_candidates
from .prompts import build_step3_item_prompt
from .schemas import STEP3_ITEM_SCHEMA

log = logging.getLogger(__name__)

# Anchor content band — same bounds the deleted classify prompt used.
_ANCHOR_CX_MIN = CONTENT_X_MIN   # 5
_ANCHOR_CX_MAX = CONTENT_X_MAX   # 37
_ANCHOR_CY_MIN = 6.0
_ANCHOR_CY_MAX = 45.0

# Free zones for remark boxes (grid 1–50).
_REM_RIGHT_X_MIN = 38.0
_REM_RIGHT_X_MAX = 50.0
_REM_LEFT_X_MIN = 1.0
_REM_LEFT_X_MAX = 12.0
_REM_INLINE_X_MIN = CONTENT_X_MIN   # 5
_REM_INLINE_X_MAX = CONTENT_X_MAX   # 37
_REM_Y_MIN_GRID = 6.0
_REM_Y_MAX_GRID = 46.0


def mark_item(
    api_key: str,
    item: dict[str, Any],
    grid_pages_for_item: list[bytes],
    model_entry: dict[str, Any] | None,
    *,
    subject: str,
    teacher_instructions: str,
    check_level: str,
    strictness_line: str,
    language: str,
    request_id: str,
) -> dict[str, Any]:
    """Run step 3 on a single item. Returns a new dict (input is not mutated)."""
    out = copy.deepcopy(item)
    qid = out.get("question_id", "?")
    start_page = int(out.get("start_page", 1))
    end_page = int(out.get("end_page", start_page))
    page_span = max(1, end_page - start_page + 1)
    valid_pages = list(range(start_page, end_page + 1))

    # Empty student answer: short-circuit, do not waste a Gemini call.
    if not (out.get("student_answer") or "").strip():
        out.update({
            "max_marks": float(model_entry.get("marks", 0)) if isinstance(model_entry, dict) else 0.0,
            "marks_awarded": 0.0,
            "status": "unattempted",
            "feedback": "Question left unattempted.",
            "student_answer_summary": "",
            "anchor_marks": [],
            "remarks": [],
            "annotations": [],
        })
        return out

    if model_entry is None:
        log.warning(
            "step3[%s] q=%s no matching model entry — emitting empty grading shell",
            request_id, qid,
        )
        out.update({
            "max_marks": 0.0,
            "marks_awarded": 0.0,
            "status": "unattempted",
            "feedback": "No model answer entry for this question.",
            "student_answer_summary": "",
            "anchor_marks": [],
            "remarks": [],
            "annotations": [],
        })
        return out

    # Long items: fall back to per-page calls (each call sees exactly one image so
    # grid-coordinate context is unambiguous).
    if page_span > STEP3_MULTI_PAGE_THRESHOLD:
        return _mark_item_per_page_fallback(
            api_key, out, grid_pages_for_item, model_entry,
            subject=subject, teacher_instructions=teacher_instructions,
            check_level=check_level, strictness_line=strictness_line,
            language=language, request_id=request_id,
        )

    raw_response = _call_step3(
        api_key=api_key, item=out, grid_pages_for_item=grid_pages_for_item,
        valid_pages=valid_pages, subject=subject,
        teacher_instructions=teacher_instructions, check_level=check_level,
        strictness_line=strictness_line, language=language, request_id=request_id,
    )
    if raw_response is None:
        return _empty_grading_shell(out, model_entry)

    validated = _validate_step3_response(raw_response, item=out, valid_pages=valid_pages)
    _backfill_max_marks(validated, model_entry, qid=qid, request_id=request_id)
    _attach_to_item(out, validated)
    return out


# ---------------------------------------------------------------------------
# Multi-page fallback
# ---------------------------------------------------------------------------


def _mark_item_per_page_fallback(
    api_key: str,
    item: dict[str, Any],
    grid_pages_for_item: list[bytes],
    model_entry: dict[str, Any],
    *,
    subject: str,
    teacher_instructions: str,
    check_level: str,
    strictness_line: str,
    language: str,
    request_id: str,
) -> dict[str, Any]:
    """One Gemini call per page of the item; aggregate marks from the first page only."""
    qid = item.get("question_id", "?")
    start_page = int(item.get("start_page", 1))
    end_page = int(item.get("end_page", start_page))
    pages = list(range(start_page, end_page + 1))

    log.info(
        "step3[%s] q=%s long item span=%s pages — per-page fallback",
        request_id, qid, len(pages),
    )

    aggregate_marks: dict[str, Any] | None = None
    all_anchor_grid: list[dict[str, Any]] = []
    all_remark_grid: list[dict[str, Any]] = []
    all_annotations: list[dict[str, Any]] = []

    for offset, page in enumerate(pages):
        single_image = grid_pages_for_item[offset:offset + 1]
        raw = _call_step3(
            api_key=api_key, item=item, grid_pages_for_item=single_image,
            valid_pages=[page], subject=subject,
            teacher_instructions=teacher_instructions, check_level=check_level,
            strictness_line=strictness_line, language=language, request_id=request_id,
        )
        if raw is None:
            continue
        validated = _validate_step3_response(raw, item=item, valid_pages=[page])
        if aggregate_marks is None:
            aggregate_marks = {
                "marks_awarded": validated["marks_awarded"],
                "max_marks": validated["max_marks"],
                "status": validated["status"],
                "feedback": validated["feedback"],
                "student_answer_summary": validated["student_answer_summary"],
            }
        all_anchor_grid.extend(validated["anchor_grid_by_page"].get(page, []))
        all_remark_grid.extend(validated["remark_grid_by_page"].get(page, []))
        all_annotations.extend(validated["annotations"])

    if aggregate_marks is None:
        return _empty_grading_shell(item, model_entry)

    composed = {
        **aggregate_marks,
        "anchor_grid_by_page": {p: [] for p in pages},
        "remark_grid_by_page": {p: [] for p in pages},
        "annotations": all_annotations,
    }
    _backfill_max_marks(composed, model_entry, qid=item.get("question_id", "?"), request_id=request_id)
    for a in all_anchor_grid:
        composed["anchor_grid_by_page"].setdefault(int(a["_page"]), []).append(a)
    for r in all_remark_grid:
        composed["remark_grid_by_page"].setdefault(int(r["_page"]), []).append(r)

    _attach_to_item(item, composed)
    return item


# ---------------------------------------------------------------------------
# Gemini call + parsing
# ---------------------------------------------------------------------------


def _call_step3(
    *,
    api_key: str,
    item: dict[str, Any],
    grid_pages_for_item: list[bytes],
    valid_pages: list[int],
    subject: str,
    teacher_instructions: str,
    check_level: str,
    strictness_line: str,
    language: str,
    request_id: str,
) -> dict[str, Any] | None:
    if len(grid_pages_for_item) != len(valid_pages):
        log.warning(
            "step3[%s] q=%s page/image count mismatch (%s vs %s) — aborting call",
            request_id, item.get("question_id", "?"),
            len(valid_pages), len(grid_pages_for_item),
        )
        return None

    client = genai.Client(api_key=api_key)
    prompt = build_step3_item_prompt(
        language=language, subject=subject,
        teacher_instructions=teacher_instructions, item=item,
        valid_pages=valid_pages, check_level=check_level,
        strictness_line=strictness_line,
    )
    parts: list[Any] = [types.Part.from_text(text=prompt)]
    for p, png in zip(valid_pages, grid_pages_for_item):
        parts.append(types.Part.from_text(
            text=f"--- PAGE {p} (grid 1–50 left→right, 1–50 top→bottom) ---"
        ))
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=STEP3_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=STEP3_ITEM_SCHEMA,
        thinking_config=thinking_off(),
        automatic_function_calling=afc_off(),
        http_options=http_opts(STEP3_HTTP_TIMEOUT_MS),
    )
    model = step3_model()
    qid = item.get("question_id", "?")
    last_raw = ""
    last_fr = "UNKNOWN"
    for attempt in range(1, 3):
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        except Exception as e:
            log.warning(
                "step3[%s] q=%s attempt=%s call failed: %s",
                request_id, qid, attempt, e,
            )
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        last_raw = (getattr(resp, "text", None) or "").strip()
        last_fr = finish_reason_name(resp)
        if not last_raw:
            log.warning(
                "step3[%s] q=%s empty response attempt=%s finish_reason=%s",
                request_id, qid, attempt, last_fr,
            )
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        for cand in parse_json_candidates(last_raw):
            if isinstance(cand, dict) and "marks_awarded" in cand:
                return cand
        log.warning(
            "step3[%s] q=%s parse fail attempt=%s finish_reason=%s prefix=%r",
            request_id, qid, attempt, last_fr, last_raw[:200],
        )
        if attempt < 2:
            time.sleep(RETRY_BACKOFF_S)

    log.warning(
        "step3[%s] q=%s gave up after 2 attempts finish_reason=%s tail=%r",
        request_id, qid, last_fr, last_raw[-200:],
    )
    return None


# ---------------------------------------------------------------------------
# Validation + coordinate post-processing
# ---------------------------------------------------------------------------


def _validate_step3_response(
    raw: dict[str, Any],
    *,
    item: dict[str, Any],
    valid_pages: list[int],
) -> dict[str, Any]:
    """Strictly validate Gemini output and bucket grid-space marks per page.

    Returns::

        {
          "marks_awarded", "max_marks", "status", "feedback",
          "student_answer_summary",
          "anchor_grid_by_page": {page: [grid_anchor, ...]},
          "remark_grid_by_page": {page: [grid_remark, ...]},
          "annotations": [annotation, ...],
        }

    Out-of-bounds / out-of-page marks are dropped with a debug log.
    """
    qid = item.get("question_id", "?")
    allowed = set(int(p) for p in valid_pages)

    anchors_by_page: dict[int, list[dict[str, Any]]] = {p: [] for p in allowed}
    for a in raw.get("anchor_marks") or []:
        if not isinstance(a, dict):
            continue
        t = str(a.get("type", "")).strip().lower()
        if t not in ("ellipse", "underline", "tick"):
            log.debug("step3 q=%s drop anchor bad type=%r", qid, t)
            continue
        try:
            page = int(a["page"])
        except (TypeError, ValueError, KeyError):
            log.debug("step3 q=%s drop anchor missing/bad page", qid)
            continue
        if page not in allowed:
            log.debug("step3 q=%s drop anchor out-of-range page=%s allowed=%s", qid, page, allowed)
            continue
        try:
            cx = _clamp_grid(a["cx"])
            cy = _clamp_grid(a["cy"])
            rx = _clamp_grid(a.get("rx", 0.0))
            ry = _clamp_grid(a.get("ry", 0.0))
        except (TypeError, ValueError, KeyError):
            log.debug("step3 q=%s drop anchor non-numeric coords", qid)
            continue

        if t == "underline":
            ry = 0.0
            rx = min(rx, cx - _ANCHOR_CX_MIN, _ANCHOR_CX_MAX - cx)
            rx = max(0.5, rx)
            cx = max(_ANCHOR_CX_MIN + rx, min(_ANCHOR_CX_MAX - rx, cx))
        elif t == "ellipse":
            cx = max(_ANCHOR_CX_MIN + rx, min(_ANCHOR_CX_MAX - rx, cx))
        else:  # tick
            cx = max(_ANCHOR_CX_MIN, min(_ANCHOR_CX_MAX, cx))

        if not (_ANCHOR_CY_MIN <= cy <= _ANCHOR_CY_MAX):
            log.debug("step3 q=%s drop anchor cy=%.1f outside [%s,%s]",
                      qid, cy, _ANCHOR_CY_MIN, _ANCHOR_CY_MAX)
            continue
        anchors_by_page[page].append(
            {"type": t, "cx": cx, "cy": cy, "rx": rx, "ry": ry, "_page": page}
        )

    remarks_by_page: dict[int, list[dict[str, Any]]] = {p: [] for p in allowed}
    for r in raw.get("remarks") or []:
        if not isinstance(r, dict):
            continue
        try:
            page = int(r["page"])
        except (TypeError, ValueError, KeyError):
            log.debug("step3 q=%s drop remark missing/bad page", qid)
            continue
        if page not in allowed:
            log.debug("step3 q=%s drop remark out-of-range page=%s allowed=%s", qid, page, allowed)
            continue
        try:
            x1 = _clamp_grid(r["x1"])
            y1 = _clamp_grid(r["y1"])
            x2 = _clamp_grid(r["x2"])
            y2 = _clamp_grid(r["y2"])
        except (TypeError, ValueError, KeyError):
            log.debug("step3 q=%s drop remark non-numeric coords", qid)
            continue
        if x2 - x1 < 6:
            x2 = min(50.0, x1 + 6.0)
        if y2 - y1 < 3:
            y2 = min(50.0, y1 + 3.0)
        if not (_REM_Y_MIN_GRID <= y1 <= y2 <= _REM_Y_MAX_GRID + 1):
            log.debug("step3 q=%s drop remark y outside band y=[%.1f,%.1f]", qid, y1, y2)
            continue
        if not _is_free_zone(x1, x2):
            log.debug("step3 q=%s drop remark x outside free zones x=[%.1f,%.1f]", qid, x1, x2)
            continue
        remarks_by_page[page].append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "comment": str(r.get("comment", "") or ""),
            "_page": page,
        })

    # Annotations: pass through with light clamping; drop entries pointing at pages
    # outside the item's range.
    annotations: list[dict[str, Any]] = []
    for ann in raw.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        try:
            page_index = int(ann["page_index"])
        except (TypeError, ValueError, KeyError):
            continue
        if (page_index + 1) not in allowed:
            log.debug(
                "step3 q=%s drop annotation page_index=%s outside allowed pages",
                qid, page_index,
            )
            continue
        annotations.append({
            "page_index": page_index,
            "y_position_percent": _clamp_pct(ann.get("y_position_percent", 0.0)),
            "x_start_percent": _clamp_pct(ann.get("x_start_percent", 0.0)),
            "x_end_percent": _clamp_pct(ann.get("x_end_percent", 100.0)),
            "comment": str(ann.get("comment", "") or ""),
            "is_positive": bool(ann.get("is_positive", False)),
        })

    status = str(raw.get("status", "unattempted") or "unattempted").strip().lower()
    if status not in ("correct", "partial", "wrong", "unattempted"):
        status = "partial"

    return {
        "marks_awarded": _to_number(raw.get("marks_awarded", 0)),
        "max_marks": _to_number(raw.get("max_marks", 0)),
        "status": status,
        "feedback": str(raw.get("feedback", "") or ""),
        "student_answer_summary": str(raw.get("student_answer_summary", "") or ""),
        "anchor_grid_by_page": anchors_by_page,
        "remark_grid_by_page": remarks_by_page,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Grid → % conversion + free-zone placement + clip to item bounds
# ---------------------------------------------------------------------------


def _attach_to_item(item: dict[str, Any], validated: dict[str, Any]) -> None:
    sp = int(item.get("start_page", 1))
    ep = int(item.get("end_page", sp))
    sy = float(item.get("start_y_position_percent", 0.0))
    ey = float(item.get("end_y_position_percent", 100.0))

    anchor_pct: list[dict[str, Any]] = []
    remark_pct: list[dict[str, Any]] = []
    for page, anchors_grid in validated["anchor_grid_by_page"].items():
        a_pct, _ = annotations_grid_to_pct(anchors_grid, [], int(page))
        anchor_pct.extend(a_pct)
    for page, remarks_grid in validated["remark_grid_by_page"].items():
        _, r_pct = annotations_grid_to_pct([], remarks_grid, int(page))
        remark_pct.extend(spread_remarks_two_column(r_pct, page_num=int(page)))

    # Clip to the item's answer y-range — defense in depth even if Gemini placed marks
    # outside the y bounds of the actual student handwriting.
    item["anchor_marks"] = clip_marks_to_answer(
        anchor_pct, sp, ep, sy, ey, y_key="cy_pct",
    )
    remark_sy = min(sy + REMARK_ANSWER_GUARD_PCT, ey)
    item["remarks"] = clip_marks_to_answer(
        remark_pct, sp, ep, remark_sy, ey, y_key="center",
    )
    for r in item["remarks"]:
        recalc_connector(r)

    item["max_marks"] = validated["max_marks"]
    item["marks_awarded"] = validated["marks_awarded"]
    item["status"] = validated["status"]
    item["feedback"] = validated["feedback"]
    item["student_answer_summary"] = validated["student_answer_summary"]
    item["annotations"] = validated["annotations"]

    nudge_mark_past_remarks(item)


def _empty_grading_shell(item: dict[str, Any], model_entry: dict[str, Any] | None) -> dict[str, Any]:
    item["max_marks"] = float(model_entry.get("marks", 0)) if isinstance(model_entry, dict) else 0.0
    item["marks_awarded"] = 0.0
    item["status"] = "partial"
    item["feedback"] = "Grading failed; please re-run."
    item["student_answer_summary"] = ""
    item["anchor_marks"] = []
    item["remarks"] = []
    item["annotations"] = []
    return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_grid(v: Any) -> float:
    x = float(v)
    return max(1.0, min(50.0, x))


def _clamp_pct(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, x))


def _to_number(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _is_free_zone(x1: float, x2: float) -> bool:
    right = _REM_RIGHT_X_MIN <= x1 and x2 <= _REM_RIGHT_X_MAX
    left = _REM_LEFT_X_MIN <= x1 and x2 <= _REM_LEFT_X_MAX
    inline = _REM_INLINE_X_MIN <= x1 and x2 <= _REM_INLINE_X_MAX
    return right or left or inline


def _backfill_max_marks(
    validated: dict[str, Any],
    model_entry: dict[str, Any] | None,
    *,
    qid: Any,
    request_id: str,
) -> None:
    """Fall back to ``model_entry["marks"]`` when Gemini returned 0 / missing max_marks.

    Gemini occasionally drops the ``max_marks`` cap on a per-item call. The model
    entry's stored ``marks`` is the authoritative cap, so use it when Gemini
    returns 0 (and log the discrepancy so it's visible).
    """
    if not isinstance(model_entry, dict):
        return
    try:
        model_max = float(model_entry.get("marks", 0) or 0)
    except (TypeError, ValueError):
        model_max = 0.0
    if model_max <= 0:
        return
    current = float(validated.get("max_marks", 0) or 0)
    if current > 0:
        return
    log.info(
        "step3[%s] q=%s backfilling max_marks from model_entry: gemini=%s model=%s",
        request_id, qid, current, model_max,
    )
    validated["max_marks"] = model_max
