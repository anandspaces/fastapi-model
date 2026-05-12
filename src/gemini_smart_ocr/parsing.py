"""JSON repair / extraction helpers shared by every smart-OCR Gemini stage."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def strip_json_fence(text: str) -> str:
    """Drop ```json ... ``` fences if Gemini wraps its response in markdown."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if m:
        return m.group(1).strip()
    return t


def repair_json(text: str) -> str:
    """Best-effort closer for truncated JSON: drops trailing commas, balances brackets."""
    t = strip_json_fence(text)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    stripped = re.sub(r"\\.", "", t)
    if stripped.count('"') % 2 != 0:
        t += '"'
    open_b = t.count("{") - t.count("}")
    open_sq = t.count("[") - t.count("]")
    if open_b > 0:
        t += "}" * open_b
    if open_sq > 0:
        t += "]" * open_sq
    return t


def parse_json_candidates(raw: str) -> list[Any]:
    """Return up to two parse attempts (stripped + repaired). Both may fail; caller filters."""
    out: list[Any] = []
    for candidate in (strip_json_fence(raw), repair_json(raw)):
        if not candidate.strip():
            continue
        try:
            out.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return out


def parse_ocr_page_text(raw: str, page_num: int) -> str:
    for parsed in parse_json_candidates(raw):
        if isinstance(parsed, dict) and parsed.get("text") is not None:
            text = str(parsed.get("text", "")).strip()
            if text:
                return text
    raise ValueError(f"Could not parse OCR text JSON for page {page_num}.")


def clamp_pct(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(100.0, x))


def page_num(v: Any, total_pages: int, fallback: int = 1) -> int:
    try:
        p = int(v)
    except (TypeError, ValueError):
        p = fallback
    return max(1, min(total_pages, p))
