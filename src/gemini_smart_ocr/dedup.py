"""Near-duplicate page detection for smart-OCR transcripts.

Two collapse paths:
  1. ``DUPLICATE`` page type from Stage 1A → mark omitted immediately.
  2. Sequence-matcher similarity between adjacent page bodies above
     ``DEFAULT_DEDUP_SIM_THRESHOLD`` → mark the later page omitted.

Outputs keep ``=== PAGE n ===`` headers stable so downstream stages can rely on
1-based numbering even when bodies are blanked.
"""

from __future__ import annotations

import difflib
import logging
import re

from .config import DEFAULT_DEDUP_SIM_THRESHOLD

log = logging.getLogger(__name__)

_PAGE_HEADER_RE = re.compile(r"^===\s*PAGE\s+(\d+)\s*===\s*$", re.MULTILINE | re.IGNORECASE)


def extract_page_body(block: str) -> str:
    """Body text after the ``=== PAGE n ===`` header line."""
    m = _PAGE_HEADER_RE.search(block)
    if not m:
        return block.strip()
    return block[m.end():].strip()


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(
        None, _normalize_for_compare(a), _normalize_for_compare(b)
    ).ratio()


def deduplicate_page_texts(
    page_blocks: list[str],
    classifications: list[str],
    request_id: str,
    *,
    sim_threshold: float = DEFAULT_DEDUP_SIM_THRESHOLD,
) -> list[str]:
    """Drop near-duplicate page bodies; keep ``=== PAGE n ===`` headers for numbering."""
    out = list(page_blocks)
    for i in range(len(out)):
        page_no = i + 1
        header = f"=== PAGE {page_no} ==="
        body = extract_page_body(out[i])

        if classifications[i] == "DUPLICATE" and i > 0:
            out[i] = (
                f"{header}\n"
                f"[OMITTED: page classified DUPLICATE — duplicate of earlier sheet; "
                f"OCR skipped to save tokens.]\n"
            )
            log.info("smart_ocr[%s] dedup: DUPLICATE class page=%s", request_id, page_no)
            continue

        if i == 0 or "[OMITTED:" in body:
            continue

        prev_body = extract_page_body(out[i - 1])
        if "[OMITTED:" in prev_body:
            continue

        if _similarity(prev_body, body) >= sim_threshold:
            out[i] = (
                f"{header}\n"
                f"[OMITTED: near-duplicate of page {page_no - 1} (similarity >= {sim_threshold}).]\n"
            )
            log.info(
                "smart_ocr[%s] dedup: similarity collapse page=%s vs %s",
                request_id, page_no, page_no - 1,
            )
    return out
