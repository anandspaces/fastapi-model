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
    return f"""You are performing OCR for page {page_num} of {total_pages} from a handwritten exam copy.

Transcribe ALL visible text on this page exactly as written.
- Preserve line breaks and paragraph breaks.
- Keep question labels like Q1, (1), प्रश्न 1, उत्तर exactly.
- Do not translate or transliterate text.
- For tables/lists, keep one row per line; use " | " between cells when needed.
- If a word is unreadable, use [illegible] only for that span.
- Do not invent, summarize, or correct content.

{script_note}

Return ONLY valid JSON:
{{"text":"full page transcript with \\n for line breaks"}}"""


def _build_ocr_prompt_correction(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (CORRECTION style): Prefer one line per numbered response where possible.
If numbering is visible, preserve it exactly."""


def _build_ocr_prompt_paragraph(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (PARAGRAPH style): Keep question numbers/labels visible before each answer block."""


def _build_ocr_prompt_word_list(page_num: int, total_pages: int, language: str) -> str:
    base = _build_ocr_prompt_generic(page_num, total_pages, language)
    return f"""{base}

FOCUS (WORD_LIST style): Keep one pair/row per line.
Preserve column order and separators."""


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

_QUESTION_NUM_RE = re.compile(
    r"""
    (?:
        \b(?:que(?:stion)?|q)\s*[:\.\-]?\s*(\d{1,3})\b       # Que:7 / Q.3 / Question 10
      | \bप्रश्न\s*[:\.\-]?\s*(\d{1,3})\b                     # प्रश्न 5
      | ^\s*\((\d{1,3})\)\s*[:\.\-]                          # (3):
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def _extract_question_number(question_text: str) -> int | None:
    """Parse the original question number from its leading label text."""
    m = _QUESTION_NUM_RE.search(question_text or "")
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            try:
                n = int(g, 10)
                return n if 1 <= n <= 500 else None
            except (TypeError, ValueError):
                return None
    return None


def _structure_item_sort_key(item: dict[str, Any]) -> tuple[int | float, ...]:
    sp = int(item.get("start_page", 1))
    sy = float(item.get("start_y_position_percent", 0.0))
    qi = item.get("question_id")
    if qi is None:
        return (sp, sy, 999999)
    try:
        return (sp, sy, int(qi))
    except (TypeError, ValueError):
        return (sp, sy, 999998)


_OCR_PAGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"text": types.Schema(type=types.Type.STRING)},
    required=["text"],
    property_ordering=["text"],
)


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


def _parse_ocr_page_text(raw: str, page_num: int) -> str:
    for candidate in (_strip_json_fence(raw), _repair_json(raw)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("text") is not None:
            text = str(parsed.get("text", "")).strip()
            if text:
                return text
    raise ValueError(f"Could not parse OCR text JSON for page {page_num}.")


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


def _estimate_expected_questions(page_blocks: list[str]) -> int | None:
    """Heuristic: infer approximate question count from visible numbering labels."""
    nums: set[int] = set()
    patterns = (
        r"\bq(?:uestion)?\s*[:.\-]?\s*(\d{1,3})\b",
        r"(?:^|\n)\s*\(?(\d{1,3})\)?\s*[.)\-:]\s+",
        r"(?:प्रश्न|उ\.?\s*प्र\.?)\s*[:.\-]?\s*(\d{1,3})\b",
    )
    for block in page_blocks:
        body = _extract_page_body(block)
        for pat in patterns:
            for m in re.finditer(pat, body, flags=re.IGNORECASE):
                try:
                    n = int(m.group(1))
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= 200:
                    nums.add(n)
    if not nums:
        return None
    return max(nums)


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

    question_text = str(item.get("question", "")).strip()
    label_num = _extract_question_number(question_text)
    try:
        model_qid = int(item.get("question_id", 0))
    except (TypeError, ValueError):
        model_qid = 0
    if label_num is not None:
        qid: int | None = label_num
    elif model_qid >= 1:
        qid = model_qid
    else:
        qid = None

    ans = str(item.get("student_answer", "")).strip()
    return {
        "question_id": qid,
        "question": question_text,
        "student_answer": ans,
        "is_attempted": bool(ans),
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
            out.append(norm)
    if not out:
        raise ValueError("No question-answer blocks detected.")
    out.sort(key=_structure_item_sort_key)

    used_ids: set[int] = {
        i
        for i in (item.get("question_id") for item in out)
        if isinstance(i, int)
    }
    fallback_counter = 1
    for item in out:
        if item.get("question_id") is not None:
            continue
        while fallback_counter in used_ids:
            fallback_counter += 1
        item["question_id"] = fallback_counter
        used_ids.add(fallback_counter)
        fallback_counter += 1
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
        response_mime_type="application/json",
        response_schema=_OCR_PAGE_SCHEMA,
    )
    page_text = ""
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
                time.sleep(0.5)
            continue
        try:
            page_text = _parse_ocr_page_text(last_raw, page_num)
            break
        except ValueError:
            if attempt < 2:
                time.sleep(0.5)

    if not page_text:
        raise RuntimeError(
            f"OCR failed for page {page_num} after 2 attempts; raw={last_raw[:200]!r}"
        )

    return f"=== PAGE {page_num} ===\n{page_text}"


def _placeholder_ocr_block(page_num: int, reason: str) -> str:
    return f"=== PAGE {page_num} ===\n[{reason}]\n"


def _structure_qa(
    client: genai.Client,
    pages_payload_json: str,
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    lang_note = (
        "Text is Hindi/English mixed; preserve wording in student_answer when present."
        if (language or "en").strip().lower() == "hi"
        else "Text may be Hindi/English mixed; preserve wording in student_answer when present."
    )
    expected_hint = (
        f"The exam has approximately {expected_questions} questions. "
        "You MUST extract all question-answer pairs and not stop early.\n"
        if expected_questions
        else ""
    )
    prompt = f"""You are structuring a handwritten exam answer book from OCR text only.

Input is JSON with this shape:
{{"pages":[{{"page":1,"text":"..."}}, ...]}}
where page numbers are physical page numbers 1..{total_pages}.

TASK:
1. Read pages in order and identify TOP-LEVEL questions only.
   A top-level question is introduced by a label like:
     Q1, Q:2, Que:3, Que 4, Q.5, प्रश्न 1, (1), etc.

   CRITICAL — DO NOT split these into separate items:
   - Sub-parts of a case study (labeled 1. 2. 3. or (a)(b)(c) under the same Que: N)
   - Continuation of an answer on the next page
   - Numbered points within an answer

   A new top-level question starts ONLY when a new Que/Q/प्रश्न label appears
   that is at the SAME level as the main question headers, NOT when a
   numbered sub-part (1. / 2. / 3.) appears inside a question or its answer.

2. For case study questions with sub-parts (e.g. "Questions:- 1. ... 2. ... 3. ..."):
   - Treat the entire case study as ONE item.
   - student_answer must include ALL sub-part answers concatenated in order.
   - question must include the full case study stem AND all sub-part labels.

3. Group questions into sections using section_name in Hindi or short English.

4. For each question emit:
   - question_id: the integer taken from that question stem's label in the OCR.
     Examples: Que:7 → 7, Q.3 → 3, प्रश्न 5 → 5.
     Do NOT substitute sequential 1,2,3 … when the paper shows Que:10, Que:11, etc. Use the printed numbering.
     If no number is visible in the stem, guess from context; the server may assign a gap id when still unknown.
     NEVER invent a row for a booklet question that never appears in the OCR (if Q7 is absent everywhere, omit question_id 7 entirely).
   - question: full question text including any sub-part labels
   - student_answer: complete answer text, joining sub-part answers with \\n\\n
   - answer_type: correction | paragraph | word_list
   - start_page, start_y_position_percent: where the question label appears
   - end_page, end_y_position_percent: where the answer ends
   - marking_page, marking_x_position_percent, marking_y_position_percent (0–100, page numbers 1..{total_pages})

5. Include questions with no answer written (blank page, "Ans:-" with nothing after):
   emit one row with student_answer exactly "" — do NOT omit unanswered items.

{lang_note}
{expected_hint}

CRITICAL: Extract every top-level question from the OCR in order through the last page, including unanswered ones. Do not emit extra items for sub-parts of a single case study.

WORKED EXAMPLE of case study grouping:
  OCR contains:
    "Que:9 You are a young IAS officer... public image.
     Questions:-
     1. Should there be ethical boundaries...?
     2. How will you handle the situation?
     3. What things should public servant keep in mind?"

  CORRECT — emit ONE item:
    question_id: 9
    question: "Que:9 You are a young IAS officer... Questions:- 1. Should there be... 2. How will you... 3. What things..."
    student_answer: "<answer to 1>\\n\\n<answer to 2>\\n\\n<answer to 3>"

  WRONG — do NOT emit three separate items with question_ids 9, 10, 11.

CRITICAL JSON:
- Valid JSON only. Use \\n inside strings, never raw newlines inside JSON strings.
- Escape " as \\" inside strings.
- Return one object: {{"sections":[{{"section_name":"...","questions":[...]}}]}}
"""

    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=pages_payload_json),
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=65536,
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


def _structure_qa_with_fallback(
    client: genai.Client,
    pages_payload: dict[str, Any],
    language: str,
    total_pages: int,
    *,
    expected_questions: int | None = None,
) -> list[dict[str, Any]]:
    pages_payload_json = json.dumps(pages_payload, ensure_ascii=False)
    rows = _structure_qa(
        client,
        pages_payload_json,
        language,
        total_pages,
        expected_questions=expected_questions,
    )
    if expected_questions and len(rows) < max(1, int(expected_questions * 0.8)):
        pages = pages_payload.get("pages", [])
        if not isinstance(pages, list) or len(pages) < 2:
            return rows
        log.warning(
            "structure_qa undercount got=%s expected~%s; retry split",
            len(rows),
            expected_questions,
        )
        mid = len(pages) // 2
        first = {"pages": pages[:mid]}
        second = {"pages": pages[mid:]}
        rows1 = _structure_qa(
            client,
            json.dumps(first, ensure_ascii=False),
            language,
            total_pages,
            expected_questions=None,
        )
        rows2 = _structure_qa(
            client,
            json.dumps(second, ensure_ascii=False),
            language,
            total_pages,
            expected_questions=None,
        )
        rows = rows1 + rows2
        rows.sort(key=_structure_item_sort_key)
        seen_ids: set[Any] = set()
        deduped: list[dict[str, Any]] = []
        for item in rows:
            qid = item.get("question_id")
            if qid not in seen_ids:
                deduped.append(item)
                seen_ids.add(qid)
        rows = deduped
    return rows


# --- Public API ------------------------------------------------------------------------


def smart_ocr_extract_student_answers(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """OCR each page (parallel) then structure into items.

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

    # Smart OCR needs better handwritten fidelity on long copies; enforce a higher floor.
    dpi = max(220, copy_ocr_raster_dpi())
    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=dpi, request_id=request_id)
    max_workers = max(1, min(copy_ocr_parallel_workers(), total_pages))

    # --- Stage 1: OCR each page (parallel) ---
    page_blocks: list[str] = [""] * total_pages

    def _ocr_task(
        args: tuple[int, bytes, str, str, int, str],
    ) -> tuple[int, str]:
        idx, png, key, lang, tp, rid = args
        pno = idx + 1
        block = _ocr_single_page(key, png, pno, tp, lang, "UNKNOWN")
        log.info(
            "smart_ocr[%s] stage1 ocr page=%s/%s chars=%s",
            rid,
            pno,
            tp,
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
        )
        for i in range(total_pages)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_ocr_task, a): a[0] for a in ocr_args}
        for fut in as_completed(futs):
            idx, block = fut.result()
            page_blocks[idx] = block

    # Deduplicate near-identical neighboring pages to reduce structure token load.
    page_blocks = _deduplicate_page_texts(
        page_blocks,
        ["UNKNOWN"] * total_pages,
        request_id,
    )

    log.info(
        "smart_ocr[%s] after stage1 ocr total_chars=%s",
        request_id,
        sum(len(b) for b in page_blocks),
    )

    # --- Stage 2: build per-page OCR JSON payload for structure pass ---
    pages_payload = {"pages": []}
    for i, block in enumerate(page_blocks):
        pages_payload["pages"].append(
            {
                "page": i + 1,
                "text": _extract_page_body(block),
            }
        )
    pages_payload_json = json.dumps(pages_payload, ensure_ascii=False)
    payload_tokens_est = len(pages_payload_json) // 4
    expected_questions = _estimate_expected_questions(page_blocks)
    log.info(
        "smart_ocr[%s] structure input_tokens_est=%s output_cap=%s expected_questions=%s",
        request_id,
        payload_tokens_est,
        65536,
        expected_questions,
    )

    # --- Stage 3: structure from OCR JSON payload ---
    structure_client = genai.Client(api_key=api_key)
    rows = _structure_qa_with_fallback(
        structure_client,
        pages_payload,
        language,
        total_pages,
        expected_questions=expected_questions,
    )

    log.info(
        "smart_ocr[%s] stage3 structure complete questions=%s",
        request_id,
        len(rows),
    )

    return {"items": rows, "page_count": total_pages}
