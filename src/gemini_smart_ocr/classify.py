"""Stage 1A — classify the page + place anchor marks and remark boxes.

Single Gemini call per page: returns ``page_type``, ``content_bands``,
``content_lines``, ``anchor_marks`` and ``remarks`` (in grid space).

This module owns the response → typed-dict normalization, but defers layout
geometry (band merging, anchor clustering, overlap filtering) to ``layout.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google import genai
from google.genai import types

from .config import (
    CLASSIFY_HTTP_TIMEOUT_MS,
    CLASSIFY_MAX_OUTPUT_TOKENS,
    PAGE_TYPES,
    RETRY_BACKOFF_S,
    afc_off,
    classify_model,
    finish_reason_name,
    http_opts,
    thinking_off,
)
from .layout import (
    extend_last_band,
    filter_overlapping_remarks,
    merge_content_bands,
    normalize_anchor_marks,
    normalize_remark_boxes,
    parse_content_bands,
    remarks_from_anchor_marks,
)
from .parsing import parse_json_candidates
from .prompts import build_classify_annotation_prompt
from .schemas import CLASSIFY_ANNOTATION_SCHEMA

log = logging.getLogger(__name__)


def _coerce_page_type(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "_")
    if t in PAGE_TYPES:
        return t
    return "UNKNOWN"


def _parse_annotation_only(raw: str, page_num: int) -> dict[str, Any]:
    """Parse just anchor_marks + remarks from one of the JSON candidates."""
    for data in parse_json_candidates(raw):
        if not isinstance(data, dict):
            continue
        return {
            "anchor_marks": normalize_anchor_marks(data.get("anchor_marks") or []),
            "remarks": normalize_remark_boxes(data.get("remarks") or []),
        }
    log.warning("parse_annotation_only failed page=%s raw=%r", page_num, (raw or "")[:200])
    return {"anchor_marks": [], "remarks": []}


def _parse_classify_response(raw: str, page_num: int) -> dict[str, Any]:
    """Parse the full classify+annotation JSON; resolve page_type and emit final remarks.

    Returns a dict with keys ``page_type``, ``anchor_marks``, ``remarks``.
    """
    base = _parse_annotation_only(raw, page_num)

    page_type = "UNKNOWN"
    content_bands: list[dict[str, float]] = []
    content_lines: list[dict[str, int]] = []

    for data in parse_json_candidates(raw):
        if not (isinstance(data, dict) and data.get("page_type") is not None):
            continue
        page_type = _coerce_page_type(str(data["page_type"]))
        raw_anchor_cys = [
            float(a.get("cy", 25))
            for a in data.get("anchor_marks", [])
            if isinstance(a, dict)
        ]
        content_bands = merge_content_bands(parse_content_bands(data.get("content_bands", [])))
        content_bands = extend_last_band(content_bands, raw_anchor_cys, page_type)

        # Parse content_lines for diagnostic logging.
        raw_lines = data.get("content_lines", [])
        if isinstance(raw_lines, list):
            for ln in raw_lines:
                if not isinstance(ln, dict):
                    continue
                try:
                    content_lines.append({
                        "y":  max(1, min(50, int(ln.get("y",  25)))),
                        "x1": max(1, min(50, int(ln.get("x1",  1)))),
                        "x2": max(1, min(50, int(ln.get("x2", 37)))),
                    })
                except (TypeError, ValueError):
                    pass
            content_lines.sort(key=lambda l: l["y"])

        # Drop ellipses sitting in blank inter-paragraph gaps (not on any word).
        if content_bands:
            def _cy_in_band(cy_g: float) -> bool:
                return any(b["y1"] <= cy_g <= b["y2"] for b in content_bands)
            base["anchor_marks"] = [
                a for a in base["anchor_marks"]
                if a["type"] != "ellipse" or _cy_in_band(float(a.get("cy", 25)))
            ]

        break

    _log_layout(page_num, page_type, content_bands, content_lines)

    if page_type == "DUPLICATE":
        return {"page_type": "DUPLICATE", "anchor_marks": [], "remarks": []}

    derived = remarks_from_anchor_marks(base["anchor_marks"])
    log.info(
        "layout[p%s] remarks_derived=%s",
        page_num,
        " ".join(
            f"y[{r.get('y1',0):.0f}-{r.get('y2',0):.0f}]" for r in derived
        ) or "(none)",
    )

    # Inline gap remarks: Gemini suggestions inside content-area x range (1..37).
    inline_gaps = [r for r in base["remarks"] if float(r.get("x1", 50)) < 38]
    if inline_gaps:
        log.info(
            "layout[p%s] inline_gaps=%s",
            page_num,
            " ".join(
                f"x[{r.get('x1',0):.0f}-{r.get('x2',0):.0f}]y[{r.get('y1',0):.0f}-{r.get('y2',0):.0f}]"
                for r in inline_gaps
            ),
        )

    all_remarks = derived + inline_gaps
    base["remarks"] = filter_overlapping_remarks(all_remarks, content_bands, page_num)
    return {"page_type": page_type, **base}


def _log_layout(
    page_num: int,
    page_type: str,
    content_bands: list[dict[str, float]],
    content_lines: list[dict[str, int]],
) -> None:
    """Diagnostic logging — band + free-zone summary, per-line bounding boxes."""
    sorted_bands = sorted(content_bands, key=lambda b: b["y1"])
    free_zones: list[str] = []
    prev = 6.0
    for band in sorted_bands:
        if band["y1"] > prev + 0.5:
            free_zones.append(f"gap[{prev:.0f}-{band['y1']:.0f}]")
        prev = max(prev, band["y2"])
    if prev < 45.0:
        free_zones.append(f"gap[{prev:.0f}-45]")
    free_zones.insert(0, "right_margin[38-49,y6-45]")

    log.info(
        "layout[p%s] type=%-10s bands=%s",
        page_num, page_type,
        " ".join(f"y[{b['y1']:.0f}-{b['y2']:.0f}]" for b in sorted_bands) or "(none)",
    )
    log.info("layout[p%s] free_zones=%s", page_num, " ".join(free_zones))
    if content_lines:
        log.info(
            "layout[p%s] lines(%s)= %s",
            page_num, len(content_lines),
            "  ".join(
                f"row{l['y']}:col[{l['x1']}-{l['x2']}]" for l in content_lines
            ),
        )


def classify_and_annotate_page(
    api_key: str,
    grid_png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
) -> dict[str, Any]:
    """Combined classify + annotation call using the grid-overlaid PNG.

    Falls back to ``{page_type="UNKNOWN", anchor_marks=[], remarks=[]}`` on failure.
    Two attempts with a short backoff; ``thinking_budget=0`` disables CoT tokens.
    """
    client = genai.Client(api_key=api_key)
    prompt = build_classify_annotation_prompt(page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=grid_png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=CLASSIFY_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=CLASSIFY_ANNOTATION_SCHEMA,
        thinking_config=thinking_off(),
        automatic_function_calling=afc_off(),
        http_options=http_opts(CLASSIFY_HTTP_TIMEOUT_MS),
    )
    model = classify_model()
    last_raw = ""
    for attempt in range(1, 3):
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        except Exception as e:
            log.warning("classify_annotate[p%s] attempt=%s call failed: %s", page_num, attempt, e)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        last_raw = (getattr(resp, "text", None) or "").strip()
        fr = finish_reason_name(resp)
        if not last_raw:
            log.warning("classify_annotate[p%s] empty response finish_reason=%s", page_num, fr)
            if attempt < 2:
                time.sleep(RETRY_BACKOFF_S)
            continue
        log.info("classify_annotate[p%s] finish_reason=%s raw_response=%s", page_num, fr, last_raw)
        result = _parse_classify_response(last_raw, page_num)
        if result["page_type"] != "UNKNOWN" or result["anchor_marks"] or result["remarks"]:
            return result
        if attempt < 2:
            time.sleep(RETRY_BACKOFF_S)
    log.warning("classify_and_annotate: fallback page=%s raw=%r", page_num, last_raw[:200])
    return {"page_type": "UNKNOWN", "anchor_marks": [], "remarks": []}
