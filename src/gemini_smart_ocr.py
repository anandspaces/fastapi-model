"""Smart OCR: classify → type-aware OCR → dedupe → structure (sections → flat items)."""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from google import genai
from google.genai import types

from src.gemini_copy_ocr import (
    copy_ocr_max_pages,
    copy_ocr_parallel_workers,
    copy_ocr_raster_dpi,
    count_pdf_pages,
    rasterize_pdf_to_png_pages,
)
from src.gemini_extract import MODEL_ID

# --- Stage 1: page classification (tiny JSON per page) ---------------------------------

_CLASSIFY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "page_type": types.Schema(
            type=types.Type.STRING,
            description=(
                "Exactly one of: DUPLICATE, CORRECTION, PARAGRAPH, WORD_LIST, UNKNOWN. "
                "DUPLICATE = second sheet visually same as question+answer duplicate; "
                "CORRECTION = short line/sentence fixes; PARAGRAPH = long prose / अर्थ+प्रयोग; "
                "WORD_LIST = pairs, विलोम, उपसर्ग/प्रत्यय tables."
            ),
        ),
    },
    required=["page_type"],
    property_ordering=["page_type"],
)

_PAGE_TYPES = frozenset(
    {"DUPLICATE", "CORRECTION", "PARAGRAPH", "WORD_LIST", "UNKNOWN"}
)


def _coerce_page_type(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "_")
    if t in _PAGE_TYPES:
        return t
    return "UNKNOWN"


def _parse_classify_response(raw: str) -> str | None:
    """Parse page_type from classify response; JSON first, then regex (handles preamble / truncation)."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(_strip_json_fence(text))
        if isinstance(data, dict) and data.get("page_type") is not None:
            return _coerce_page_type(str(data["page_type"]))
    except json.JSONDecodeError:
        pass
    m = re.search(r'"page_type"\s*:\s*"([^"]*)"', text, re.I)
    if m:
        val = (m.group(1) or "").strip()
        if val:
            return _coerce_page_type(val)
    return None


def _classify_page(
    api_key: str,
    png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
) -> str:
    client = genai.Client(api_key=api_key)
    lang = (language or "en").strip().lower()
    script = "Hindi (Devanagari) and/or English may appear." if lang == "hi" else "English and/or Hindi may appear."
    prompt = f"""You see ONE page (page {page_num} of {total_pages}) from a handwritten exam answer book (e.g. UPSC Hindi).
{script}

Choose exactly one page_type (JSON only, no markdown):

- DUPLICATE — same layout as a typical "question paper + answer" duplicate of another page (repeated sheet).
- CORRECTION — mostly short sentence/line corrections (e.g. वाक्य शुद्धि), numbered one-line fixes.
- PARAGRAPH — long answers, paragraphs, अर्थ / प्रयोग style blocks.
- WORD_LIST — word pairs, opposites (विलोम), उपसर्ग/प्रत्यय breakdowns, tabular lists.
- UNKNOWN — none of the above clearly fits.

Return: {{"page_type":"..."}} with one of the five strings above."""

    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_schema=_CLASSIFY_SCHEMA,
    )
    last_raw = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        last_raw = (getattr(resp, "text", None) or "").strip()
        if not last_raw:
            if attempt < 2:
                time.sleep(0.25)
            continue
        parsed = _parse_classify_response(last_raw)
        if parsed is not None:
            return parsed
        if attempt < 2:
            time.sleep(0.25)
    log.warning("classify_page: parse failed page=%s raw=%r", page_num, last_raw[:200])
    return "UNKNOWN"


# --- Stage 2: type-aware OCR prompts ---------------------------------------------------

def _build_ocr_prompt_generic(page_num: int, total_pages: int, language: str) -> str:
    lang = (language or "en").strip().lower()
    script_note = (
        "Preserve Hindi (Devanagari) and English exactly — do NOT transliterate."
        if lang == "hi"
        else "Preserve English and Hindi exactly — do NOT transliterate."
    )
    return f"""Expert OCR for handwritten exam page {page_num} of {total_pages}.

Transcribe ALL visible text. Preserve layout with blank lines between sections.
Question labels (Q1, (1), उत्तर:, etc.) on their own lines. [DIAGRAM] for unreadable figures.
Tables: use " | " between cells. No markdown fences. {script_note}

Output ONLY plain text for this page."""


def _build_ocr_prompt_correction(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (CORRECTION style): Prefer one line per numbered fix where possible.
Use format when obvious: Q{{n}}: <corrected sentence or line>
Otherwise keep the student's layout."""


def _build_ocr_prompt_paragraph(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (PARAGRAPH style): Keep question numbers visible, then अर्थ / प्रयोग blocks if present.
Separate distinct questions with a line containing only: ==="""


def _build_ocr_prompt_word_list(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (WORD_LIST style): One pair or row per line where possible, e.g.
<word> — <opposite / gloss / breakdown>
Keep columns as single lines separated by " | " if needed."""


def _ocr_prompt_for_page_type(
    page_type: str, page_num: int, total_pages: int, language: str
) -> str:
    if page_type == "CORRECTION":
        return _build_ocr_prompt_correction(page_num, total_pages, language)
    if page_type == "PARAGRAPH":
        return _build_ocr_prompt_paragraph(page_num, total_pages, language)
    if page_type == "WORD_LIST":
        return _build_ocr_prompt_word_list(page_num, total_pages, language)
    return _build_ocr_prompt_generic(page_num, total_pages, language)


# --- Stage 3: structure schema (sections) then flatten to legacy items -----------------

_QUESTION_BLOCK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "question_id": types.Schema(type=types.Type.INTEGER),
        "question": types.Schema(type=types.Type.STRING),
        "student_answer": types.Schema(type=types.Type.STRING),
        "answer_type": types.Schema(
            type=types.Type.STRING,
            description="One of: correction, paragraph, word_list",
        ),
        "start_page": types.Schema(type=types.Type.INTEGER),
        "start_y_position_percent": types.Schema(type=types.Type.NUMBER),
        "end_page": types.Schema(type=types.Type.INTEGER),
        "end_y_position_percent": types.Schema(type=types.Type.NUMBER),
        "marking_page": types.Schema(type=types.Type.INTEGER),
        "marking_x_position_percent": types.Schema(type=types.Type.NUMBER),
        "marking_y_position_percent": types.Schema(type=types.Type.NUMBER),
    },
    required=[
        "question_id",
        "question",
        "student_answer",
        "answer_type",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
    property_ordering=[
        "question_id",
        "question",
        "student_answer",
        "answer_type",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
)

_SECTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "section_name": types.Schema(type=types.Type.STRING),
        "questions": types.Schema(
            type=types.Type.ARRAY,
            items=_QUESTION_BLOCK_SCHEMA,
        ),
    },
    required=["section_name", "questions"],
    property_ordering=["section_name", "questions"],
)

_STRUCTURE_ROOT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sections": types.Schema(
            type=types.Type.ARRAY,
            items=_SECTION_SCHEMA,
        ),
    },
    required=["sections"],
    property_ordering=["sections"],
)

_ANSWER_TYPES = frozenset({"correction", "paragraph", "word_list"})


# --- JSON / text helpers ---------------------------------------------------------------

def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if m:
        return m.group(1).strip()
    return t


def _repair_json(text: str) -> str:
    t = _strip_json_fence(text)
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


def _clamp_pct(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(100.0, x))


def _page_num(v: Any, total_pages: int, fallback: int = 1) -> int:
    try:
        p = int(v)
    except (TypeError, ValueError):
        p = fallback
    return max(1, min(total_pages, p))


def _normalize_answer_type(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in _ANSWER_TYPES:
        return s
    if s in ("correction", "short", "line"):
        return "correction"
    if s in ("word_list", "wordlist", "list", "pairs", "tabular"):
        return "word_list"
    if s in ("paragraph", "prose", "long"):
        return "paragraph"
    return "paragraph"


def _normalize_flat_item(
    item: dict[str, Any],
    total_pages: int,
    section_name: str,
) -> dict[str, Any]:
    start_page = _page_num(item.get("start_page"), total_pages, 1)
    end_page = _page_num(item.get("end_page"), total_pages, start_page)
    if end_page < start_page:
        end_page = start_page
    marking_page = _page_num(item.get("marking_page"), total_pages, start_page)
    try:
        qid = int(item.get("question_id", 0))
    except (TypeError, ValueError):
        qid = 0
    if qid < 1:
        qid = 1
    return {
        "question_id": qid,
        "question": str(item.get("question", "")).strip(),
        "student_answer": str(item.get("student_answer", "")).strip(),
        "section_name": section_name.strip(),
        "answer_type": _normalize_answer_type(item.get("answer_type")),
        "start_page": start_page,
        "start_y_position_percent": _clamp_pct(item.get("start_y_position_percent", 0)),
        "end_page": end_page,
        "end_y_position_percent": _clamp_pct(item.get("end_y_position_percent", 100)),
        "marking_page": marking_page,
        "marking_x_position_percent": _clamp_pct(item.get("marking_x_position_percent", 50)),
        "marking_y_position_percent": _clamp_pct(item.get("marking_y_position_percent", 50)),
    }


def _parse_structure_sections(raw: str, total_pages: int) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    parsed: Any = None
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        if not candidate.strip():
            continue
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as e:
            last_err = e
    else:
        msg = str(last_err) if last_err else "invalid JSON"
        log.warning("smart_ocr structure JSON parse failed: %s", msg)
        raise ValueError(msg) from last_err

    if not isinstance(parsed, dict):
        raise ValueError("Structure response must be a JSON object.")
    sections = parsed.get("sections")
    if not isinstance(sections, list):
        raise ValueError("Structure response missing sections array.")

    out: list[dict[str, Any]] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        section_name = str(sec.get("section_name", "")).strip() or "अज्ञात अनुभाग"
        questions = sec.get("questions")
        if not isinstance(questions, list):
            continue
        for row in questions:
            if not isinstance(row, dict):
                continue
            norm = _normalize_flat_item(row, total_pages, section_name)
            if norm["student_answer"]:
                out.append(norm)
    if not out:
        raise ValueError("No question-answer blocks detected.")
    for idx, item in enumerate(out, start=1):
        item["question_id"] = idx
    return out


# --- Dedup -----------------------------------------------------------------------------

_PAGE_HEADER_RE = re.compile(r"^===\s*PAGE\s+(\d+)\s*===\s*$", re.MULTILINE | re.IGNORECASE)


def _extract_page_body(block: str) -> str:
    """Body text after === PAGE n === line."""
    m = _PAGE_HEADER_RE.search(block)
    if not m:
        return block.strip()
    return block[m.end() :].strip()


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(
        None, _normalize_for_compare(a), _normalize_for_compare(b)
    ).ratio()


def _deduplicate_page_texts(
    page_blocks: list[str],
    classifications: list[str],
    request_id: str,
    *,
    sim_threshold: float = 0.92,
) -> list[str]:
    """Drop near-duplicate page bodies; keep === PAGE n === headers for stable numbering."""
    out = list(page_blocks)
    for i in range(len(out)):
        page_no = i + 1
        header = f"=== PAGE {page_no} ==="
        body = _extract_page_body(out[i])

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

        prev_body = _extract_page_body(out[i - 1])
        if "[OMITTED:" in prev_body:
            continue

        if _similarity(prev_body, body) >= sim_threshold:
            out[i] = (
                f"{header}\n"
                f"[OMITTED: near-duplicate of page {page_no - 1} (similarity >= {sim_threshold}).]\n"
            )
            log.info(
                "smart_ocr[%s] dedup: similarity collapse page=%s vs %s",
                request_id,
                page_no,
                page_no - 1,
            )
    return out


# --- Gemini calls: OCR ----------------------------------------------------------------


def _ocr_single_page(
    api_key: str,
    png: bytes,
    page_num: int,
    total_pages: int,
    language: str,
    page_type: str,
) -> str:
    client = genai.Client(api_key=api_key)
    prompt = _ocr_prompt_for_page_type(page_type, page_num, total_pages, language)
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=png, mime_type="image/png"),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
    )
    text = ""
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            break
        if attempt < 2:
            time.sleep(0.5)

    if not text:
        raise RuntimeError(
            f"Empty OCR response for page {page_num} after 2 attempts."
        )

    return f"=== PAGE {page_num} ===\n{text}"


def _placeholder_ocr_block(page_num: int, reason: str) -> str:
    return f"=== PAGE {page_num} ===\n[{reason}]\n"


def _structure_qa(
    client: genai.Client,
    page_texts: list[str],
    language: str,
    total_pages: int,
    page_types_summary: str,
) -> list[dict[str, Any]]:
    combined = "\n\n".join(page_texts)
    lang_note = (
        "Text is Hindi/English mixed; preserve wording in student_answer."
        if (language or "en").strip().lower() == "hi"
        else "Text may be Hindi/English mixed; preserve wording in student_answer."
    )
    prompt = f"""You are structuring a handwritten exam answer book from OCR text only.

Pages are marked "=== PAGE N ===" for physical page numbers 1..{total_pages}.
Some pages may contain "[OMITTED: ...]" — those are intentional duplicate skips; do not invent answers for them.

Per-page layout hints (from a prior vision classifier — use as weak priors only):
{page_types_summary}

TASK:
1. Group questions into sections (e.g. वाक्य शुद्धि, वाक्य प्रयोग, विलोम, उपसर्ग/प्रत्यय) using section_name in Hindi or short English when needed.
2. For each student question: question text/label, full student_answer from OCR (do not shorten), answer_type: correction | paragraph | word_list,
   and layout fields: start_page, start_y_position_percent, end_page, end_y_position_percent, marking_page, marking_x_position_percent, marking_y_position_percent (0–100, page numbers 1..{total_pages}).

{lang_note}

CRITICAL JSON:
- Valid JSON only. Use \\n inside strings, never raw newlines inside JSON strings.
- Escape " as \\" inside strings.
- Return one object: {{"sections":[{{"section_name":"...","questions":[...]}}]}}
"""

    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=combined),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_schema=_STRUCTURE_ROOT_SCHEMA,
    )
    last_err: Exception | None = None
    for attempt in range(1, 3):
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=cfg,
        )
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw:
            raise ValueError("Structure pass returned an empty response.")
        try:
            return _parse_structure_sections(raw, total_pages)
        except ValueError as e:
            last_err = e
            log.warning("structure_qa parse attempt %s/2 failed: %s", attempt, e)
            if attempt < 2:
                time.sleep(0.4)
    raise ValueError(f"Structure pass failed after 2 attempts: {last_err}") from last_err


# --- Public API ------------------------------------------------------------------------


def smart_ocr_extract_student_answers(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Classify pages → type-aware OCR (parallel) → dedupe → structure into items.

    Response shape unchanged for HTTP layer: ``{{"items": [...], "page_count": N}}``.
    Each item includes legacy position fields plus ``section_name`` and ``answer_type``.
    """
    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    dpi = copy_ocr_raster_dpi()
    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=dpi, request_id=request_id)
    max_workers = max(1, min(copy_ocr_parallel_workers(), total_pages))

    # --- Stage 1: classify each page (parallel) ---
    classifications: list[str] = ["UNKNOWN"] * total_pages

    def _classify_task(
        args: tuple[int, bytes, str, str, int, str],
    ) -> tuple[int, str]:
        idx, png, key, lang, tp, rid = args
        pt = _classify_page(key, png, idx + 1, tp, lang)
        log.info(
            "smart_ocr[%s] stage1 classify page=%s/%s -> %s",
            rid,
            idx + 1,
            tp,
            pt,
        )
        return idx, pt

    classify_args = [
        (i, png_pages[i], api_key, language, total_pages, request_id)
        for i in range(total_pages)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_classify_task, a): a[0] for a in classify_args}
        for fut in as_completed(futs):
            idx, pt = fut.result()
            classifications[idx] = pt

    # --- Stage 2: type-aware OCR (skip body OCR for DUPLICATE after page 1) ---
    page_blocks: list[str] = [""] * total_pages

    def _ocr_task(
        args: tuple[int, bytes, str, str, int, str, str],
    ) -> tuple[int, str]:
        idx, png, key, lang, tp, rid, page_type = args
        pno = idx + 1
        if page_type == "DUPLICATE" and idx > 0:
            block = _placeholder_ocr_block(
                pno,
                "OMITTED: DUPLICATE class — OCR skipped; compare/dedup uses earlier pages.",
            )
            log.info("smart_ocr[%s] stage2 skip OCR page=%s (DUPLICATE)", rid, pno)
            return idx, block
        block = _ocr_single_page(key, png, pno, tp, lang, page_type)
        log.info(
            "smart_ocr[%s] stage2 ocr page=%s/%s type=%s chars=%s",
            rid,
            pno,
            tp,
            page_type,
            len(block),
        )
        return idx, block

    ocr_args = [
        (
            i,
            png_pages[i],
            api_key,
            language,
            total_pages,
            request_id,
            classifications[i],
        )
        for i in range(total_pages)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_ocr_task, a): a[0] for a in ocr_args}
        for fut in as_completed(futs):
            idx, block = fut.result()
            page_blocks[idx] = block

    # --- Stage 3: dedupe near-identical bodies (after DUPLICATE placeholders) ---
    page_blocks = _deduplicate_page_texts(
        page_blocks, classifications, request_id
    )

    log.info(
        "smart_ocr[%s] after stage2+dedup total_chars=%s",
        request_id,
        sum(len(b) for b in page_blocks),
    )

    # --- Stage 4: structure (text only) ---
    page_types_summary = "\n".join(
        f"PAGE {i + 1}: {classifications[i]}" for i in range(total_pages)
    )
    structure_client = genai.Client(api_key=api_key)
    rows = _structure_qa(
        structure_client,
        page_blocks,
        language,
        total_pages,
        page_types_summary,
    )

    log.info(
        "smart_ocr[%s] stage4 structure complete questions=%s",
        request_id,
        len(rows),
    )

    return {"items": rows, "page_count": total_pages}
