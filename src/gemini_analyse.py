"""Gemini-powered copy-checker analysis (grading, combined review, intro marks table)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

ANALYSE_MODEL = "gemini-3.1-pro-preview"

# Generation configs per endpoint (spec)
CFG_PAGES = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=8192,
    response_mime_type="application/json",
)

CFG_CACHED_OCR = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=8192,
    response_mime_type="application/json",
)

CFG_COMBINED = types.GenerateContentConfig(
    temperature=0.35,
    max_output_tokens=2048,
    response_mime_type="application/json",
)

CFG_INTRO = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=2048,
    # omit response_mime_type — plain pipe-delimited text
)

_ANALYSIS_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "student_text": types.Schema(type=types.Type.STRING),
        "marks_awarded": types.Schema(type=types.Type.NUMBER),
        "confidence_percent": types.Schema(type=types.Type.NUMBER),
        "good_points": types.Schema(type=types.Type.STRING),
        "improvements": types.Schema(type=types.Type.STRING),
        "final_review": types.Schema(type=types.Type.STRING),
        "annotations": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "page_index": types.Schema(type=types.Type.INTEGER),
                    "y_position_percent": types.Schema(type=types.Type.NUMBER),
                    "x_start_percent": types.Schema(type=types.Type.NUMBER),
                    "x_end_percent": types.Schema(type=types.Type.NUMBER),
                    "comment": types.Schema(type=types.Type.STRING),
                    "is_positive": types.Schema(type=types.Type.BOOLEAN),
                    "line_style": types.Schema(type=types.Type.STRING),
                },
                required=[
                    "page_index",
                    "y_position_percent",
                    "x_start_percent",
                    "x_end_percent",
                    "comment",
                    "is_positive",
                    "line_style",
                ],
            ),
        ),
    },
    required=[
        "student_text",
        "marks_awarded",
        "confidence_percent",
        "good_points",
        "improvements",
        "final_review",
        "annotations",
    ],
)

_COMBINED_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "final_review": types.Schema(type=types.Type.STRING),
        "overall_improvements": types.Schema(type=types.Type.STRING),
        "one_thing_to_write": types.Schema(type=types.Type.STRING),
    },
    required=["final_review", "overall_improvements", "one_thing_to_write"],
)


FEEDBACK_TONE_BLOCK = """
FEEDBACK VOICE — UPSC MAINS SCRIPT MENTOR (NOT COACH CHEERLEADING):
- Comments and bullet feedback read like examiner + senior tutor: restrained, rigorous, usefulness-first.
- Ban empty superlatives and blanket praise ('Excellent answer', 'Outstanding', 'Very well written' without diagnosis).
- good_points: only **specific**, mark-earning virtues (with a terse why they matter versus a generic script).
- improvements plus annotation comments: lead with gaps, sharper structure, absent dimension, factual precision, illustrative depth — show how to climb the marking ladder next.
- Maintain respect; warmth is spare — encouragement comes through clarity of next steps.
"""

MARKING_RULES_BLOCK = """
MARKING RULES (NON-NEGOTIABLE):
- NEVER exceed total_marks.
- Default to FEWER marks, not more.
- FULL marks: answer nearly perfectly matches scheme, all key points, correct terminology.
- MID-RANGE: covers some points but misses important ones or lacks depth.
- LOW marks: superficial, off-topic, or barely touches the scheme.
- ZERO: blank, irrelevant, or copied without understanding.
- Do NOT reward length — reward accuracy and relevance only.
- When in doubt between two values, always choose the LOWER one.

DECIMAL MARKING:
- marks_awarded must be a decimal in multiples of 0.5 only (e.g. 1.5, 3.5), never above total_marks.
- Scale proportionally for other totals; reference tiers:
  - For an 8-mark question: excellent ≈ 3.5, moderate ≈ 2.5, low ≈ 1.5 (relative to scheme quality).
  - For a 12-mark question: excellent ≈ 5, moderate ≈ 3, low ≈ 2.5 (relative to scheme quality).
"""

ANNOTATION_RULES_BLOCK = """
ANNOTATION PLACEMENT RULES:
- Same page, same side: y_position_percent must differ by at least 20 between annotations.
- Opposite sides on same page can share similar y positions.
- Left half: x_end_percent ≤ 55. Right half: x_start_percent ≥ 45.
- Spread across pages — do not cluster on page 0.
- Density: roughly 2–4 annotations per handwritten page when the answer is substantive; proportionally fewer on sparse pages.
- Cover the answer structure in comments: aim for annotations on opening/introduction framing, middle argument or evidence, and closing/conclusion synthesis (adapt to whatever structure the student actually wrote).
- Comment wording: UPSC script-mentor tone — spare praise, high diagnostic yield; use is_positive true only for genuinely distinctive merits.
"""


def _hi_language_block() -> str:
    return """
LANGUAGE (Hindi, language=hi):
- All feedback strings (good_points, improvements, final_review, annotation comments) MUST be in Hindi.
- Use formal address: "Aap/Aapka"; avoid "Tum/Tumne".
- Prefer formal terms (e.g. Prastavana/Parichay for introduction where appropriate).
- Avoid casual praise like "Shabash" and exaggerated appreciation; tone = गंभीर मार्गदर्शक परीक्षक (संयत, विश्लेषणात्मक).
"""


def _en_language_block() -> str:
    return """
LANGUAGE (English, language=en):
- All feedback strings must be in English only — formal, mains-exam tone; no slangy hype or empty praise clusters.
"""


def build_analyse_prompt(
    question_title: str,
    model_description: str,
    total_marks: int,
    language: str,
    *,
    page_count: int | None,
    mode: str,
) -> str:
    """mode: 'pages' | 'cached_ocr'"""
    lang = language.strip().lower()
    lang_rules = _hi_language_block() if lang == "hi" else _en_language_block()
    pc = (
        f"The student's answer spans {page_count} page(s). Use page_index 0..{page_count - 1} for annotations."
        if page_count is not None
        else ""
    )
    ocr_note = (
        "You are given the student's handwritten answer as one or more page images in order. "
        "Perform full OCR. Attempt to read light or sparse handwriting; do not skip pages."
        if mode == "pages"
        else "You are given cached OCR text of the student's answer (no images). "
        "Use it as the student_text basis and grade accordingly."
    )
    return f"""You are a senior examiner acting as UPSC mains script mentor — grade against the marking scheme with professional, restrained feedback.

{ocr_note}

QUESTION TITLE:
{question_title}

TEACHER MARKING SCHEME / MODEL ANSWER:
{model_description}

TOTAL MARKS FOR THIS QUESTION: {total_marks}
{pc}

{MARKING_RULES_BLOCK}

{FEEDBACK_TONE_BLOCK}

{ANNOTATION_RULES_BLOCK}

{lang_rules}

OUTPUT: Return ONE JSON object only (no markdown) with keys:
- student_text: string, full transcription of the student's answer (from OCR for images, or echo/normalize cached text for cached mode).
- marks_awarded: number, 0 to {total_marks}, steps of 0.5 only.
- confidence_percent: number, 0–100, your confidence in this grading.
- good_points: string, bullet points with leading "• " lines, teacher-style strengths.
- improvements: string, bullet points with "• ", teacher-style improvements.
- final_review: string, 2–4 sentences overall remark — examiner summary: measured verdict + priority fixes; avoid gushing praise.
- annotations: array of objects, each with:
  page_index (0-based), y_position_percent, x_start_percent, x_end_percent, comment, is_positive (boolean), line_style ("straight" or "zigzag").

Produce **meaningful granularity**: for multi-paragraph answers use **at least 5 annotations** when page_count ≥ 2, otherwise **at least 3**. Put concrete strengths and fixes in annotation ``comment`` strings (not only in good_points/improvements). Prefer one annotation each for intro quality, central argument/example quality, and conclusion quality whenever those regions exist.

Ensure annotations follow placement rules and match the answer content.
"""


def build_combined_prompt(question_results_json: str) -> str:
    return f"""You are a UPSC mains mentor drafting an integrated paper critique — examiner-like, restrained, diagnostically dense. Do not fluff or over-praise; synthesise patterns and actionable priorities.

PER-QUESTION RESULTS (JSON):
{question_results_json}

TASK:
1. Write final_review: one long flowing paragraph (~150–220 words). Match the dominant language of the improvements/good_points text below (Hindi vs English). Use \\n between sentences if helpful. Acknowledge competence briefly where merited; weight the paragraph toward systematic gaps and how to tighten answer-writing for marks.
2. Write overall_improvements: exactly 4 plain improvement sentences, one per line, separated by \\n.
3. Write one_thing_to_write: one sentence — the single most impactful practice tip.

Return ONE JSON object with keys: final_review, overall_improvements, one_thing_to_write only.
"""


INTRO_PAGE_PROMPT = """You are reading the first (cover/intro) page of an exam answer booklet.
Find the printed marks table. Extract every cell in the M.Obt. (marks obtained) column.

Output ONLY plain text lines, one per row, in this exact format (pipe-separated, no header):
questionNo|marksText|xPercent|yPercent

Rules:
- questionNo: integer. Use 1,2,3,... for each question row. Use 0 for the Total / Grand Total row if present.
- marksText: the value exactly as written (e.g. "4", "3.5", "3-75"). Use empty string if blank.
- xPercent, yPercent: horizontal and vertical centre of that M.Obt. cell as percentages of page width/height (0–100).

Include every row even when marksText is blank so positions are known.
Do not output JSON or markdown — only lines of text.
"""


def _image_mime(data: bytes) -> str:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/jpeg"


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text


def _repair_json(text: str) -> str:
    """Best-effort repair for truncated / sloppy JSON."""
    t = _strip_json_fence(text)
    # remove trailing commas before } or ]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # balance braces/brackets (simple)
    open_b = t.count("{") - t.count("}")
    open_sq = t.count("[") - t.count("]")
    if open_b > 0:
        t += "}" * open_b
    if open_sq > 0:
        t += "]" * open_sq
    return t


def _round_half(x: float) -> float:
    return round(float(x) * 2.0) / 2.0


def _clamp_marks(marks: float, total_marks: int) -> float:
    m = _round_half(marks)
    if m < 0:
        m = 0.0
    if m > total_marks:
        m = float(total_marks)
    return m


def _clamp_annotations(annotations: list[dict], page_count: int) -> list[dict]:
    if page_count <= 0:
        return []
    out: list[dict] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        try:
            pi = int(a.get("page_index", 0))
        except (TypeError, ValueError):
            pi = 0
        pi = max(0, min(page_count - 1, pi))
        line_style = str(a.get("line_style", "straight")).lower()
        if line_style not in ("straight", "zigzag"):
            line_style = "straight"
        out.append(
            {
                "page_index": pi,
                "y_position_percent": float(a.get("y_position_percent", 0)),
                "x_start_percent": float(a.get("x_start_percent", 0)),
                "x_end_percent": float(a.get("x_end_percent", 0)),
                "comment": str(a.get("comment", "")),
                "is_positive": bool(a.get("is_positive", False)),
                "line_style": line_style,
            }
        )
    return out


def _parse_analysis_obj(
    data: dict[str, Any], total_marks: int, page_count: int
) -> dict[str, Any]:
    raw_marks = data.get("marks_awarded", 0)
    try:
        marks = float(raw_marks)
    except (TypeError, ValueError):
        marks = 0.0
    marks = _clamp_marks(marks, total_marks)

    try:
        conf = float(data.get("confidence_percent", 0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(100.0, conf))

    ann = data.get("annotations")
    if not isinstance(ann, list):
        ann = []

    return {
        "student_text": str(data.get("student_text", "")),
        "marks_awarded": marks,
        "confidence_percent": conf,
        "good_points": str(data.get("good_points", "")),
        "improvements": str(data.get("improvements", "")),
        "final_review": str(data.get("final_review", "")),
        "annotations": _clamp_annotations(ann, page_count),
    }


def _extract_analysis_fallback(text: str, total_marks: int, page_count: int) -> dict[str, Any] | None:
    """Regex salvage when JSON is broken."""
    t = _strip_json_fence(text)

    def grab_str(key: str) -> str:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.DOTALL)
        if not m:
            return ""
        s = m.group(1)
        try:
            return json.loads(f'"{s}"')
        except json.JSONDecodeError:
            return s.replace("\\n", "\n").replace("\\t", "\t")

    def grab_num(key: str) -> float | None:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*([-0-9.]+)', t)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    st = grab_str("student_text")
    ma = grab_num("marks_awarded")
    cp = grab_num("confidence_percent")
    gp = grab_str("good_points")
    imp = grab_str("improvements")
    fr = grab_str("final_review")

    if ma is None:
        ma = 0.0
    if cp is None:
        cp = 0.0

    annotations: list[dict] = []
    for block in re.finditer(
        r"\{[^{}]*page_index[^{}]*\}", t, re.DOTALL
    ):
        try:
            obj = json.loads(block.group(0))
            if isinstance(obj, dict) and "page_index" in obj:
                annotations.append(obj)
        except json.JSONDecodeError:
            continue

    data = {
        "student_text": st,
        "marks_awarded": ma,
        "confidence_percent": cp,
        "good_points": gp,
        "improvements": imp,
        "final_review": fr,
        "annotations": annotations,
    }
    return _parse_analysis_obj(data, total_marks, page_count)


def _parse_analysis_json(
    text: str, total_marks: int, page_count: int
) -> dict[str, Any]:
    for candidate in (text, _repair_json(text)):
        try:
            raw = json.loads(_strip_json_fence(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            return _parse_analysis_obj(raw, total_marks, page_count)
    fb = _extract_analysis_fallback(text, total_marks, page_count)
    if fb is not None:
        return fb
    raise ValueError("Could not parse analysis JSON")


def _parse_combined_json(text: str) -> dict[str, Any]:
    for candidate in (text, _repair_json(text)):
        try:
            raw = json.loads(_strip_json_fence(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            return {
                "final_review": str(raw.get("final_review", "")),
                "overall_improvements": str(raw.get("overall_improvements", "")),
                "one_thing_to_write": str(raw.get("one_thing_to_write", "")),
            }
    t = _strip_json_fence(text)

    def grab(key: str) -> str:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.DOTALL)
        if not m:
            return ""
        s = m.group(1)
        try:
            return json.loads(f'"{s}"')
        except json.JSONDecodeError:
            return s.replace("\\n", "\n").replace("\\t", "\t")

    fr = grab("final_review")
    oi = grab("overall_improvements")
    ot = grab("one_thing_to_write")
    if fr or oi or ot:
        return {
            "final_review": fr,
            "overall_improvements": oi,
            "one_thing_to_write": ot,
        }
    raise ValueError("Could not parse combined review JSON")


def _call_gemini_analysis(
    client: genai.Client,
    contents: list[Any],
    config: types.GenerateContentConfig,
    total_marks: int,
    page_count: int,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            cfg = config.model_copy(update={"response_schema": _ANALYSIS_SCHEMA})
            response = client.models.generate_content(
                model=ANALYSE_MODEL,
                contents=contents,
                config=cfg,
            )
            text = getattr(response, "text", None) or ""
            if not text:
                raise RuntimeError("Gemini returned no text")
            return _parse_analysis_json(text, total_marks, page_count)
        except Exception as e:
            last_err = e
            log.warning("Analysis attempt %s failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.3 * attempt)
    assert last_err is not None
    raise last_err


def _call_gemini_combined(
    client: genai.Client, contents: list[Any]
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            cfg = CFG_COMBINED.model_copy(update={"response_schema": _COMBINED_SCHEMA})
            response = client.models.generate_content(
                model=ANALYSE_MODEL,
                contents=contents,
                config=cfg,
            )
            text = getattr(response, "text", None) or ""
            if not text:
                raise RuntimeError("Gemini returned no text")
            return _parse_combined_json(text)
        except Exception as e:
            last_err = e
            log.warning("Combined review attempt %s failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.3 * attempt)
    assert last_err is not None
    raise last_err


def _call_gemini_intro(client: genai.Client, contents: list[Any]) -> str:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=ANALYSE_MODEL,
                contents=contents,
                config=CFG_INTRO,
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ValueError("Could not detect a marks table on this page")
            return text
        except Exception as e:
            last_err = e
            log.warning("Intro-page attempt %s failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.3 * attempt)
    assert last_err is not None
    if isinstance(last_err, ValueError):
        raise last_err
    raise RuntimeError(f"AI service error: {last_err}")


def analyse_pages(
    client: genai.Client,
    images: list[bytes],
    question_title: str,
    model_description: str,
    total_marks: int,
    language: str,
) -> dict[str, Any]:
    page_count = len(images)
    prompt = build_analyse_prompt(
        question_title,
        model_description,
        total_marks,
        language,
        page_count=page_count,
        mode="pages",
    )
    parts: list[Any] = [types.Part.from_text(text=prompt)]
    for img in images:
        parts.append(
            types.Part.from_bytes(data=img, mime_type=_image_mime(img))
        )
    return _call_gemini_analysis(
        client,
        parts,
        CFG_PAGES,
        total_marks,
        page_count,
    )


def analyse_cached_ocr(
    client: genai.Client,
    cached_student_text: str,
    question_title: str,
    model_description: str,
    total_marks: int,
    page_count: int,
    language: str,
) -> dict[str, Any]:
    prompt = build_analyse_prompt(
        question_title,
        model_description,
        total_marks,
        language,
        page_count=page_count,
        mode="cached_ocr",
    )
    body = f"""CACHED STUDENT TEXT:
{cached_student_text}

{prompt}"""
    parts = [types.Part.from_text(text=body)]
    return _call_gemini_analysis(
        client,
        parts,
        CFG_CACHED_OCR,
        total_marks,
        page_count,
    )


def generate_combined_review(
    client: genai.Client, question_results: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = json.dumps(question_results, ensure_ascii=False)
    prompt = build_combined_prompt(payload)
    return _call_gemini_combined(client, [types.Part.from_text(text=prompt)])


def _parse_intro_lines(text: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        try:
            qno = int(parts[0].strip())
            marks_text = parts[1].strip()
            x_pct = float(parts[2].strip())
            y_pct = float(parts[3].strip())
        except (ValueError, IndexError):
            continue
        cells.append(
            {
                "question_no": qno,
                "marks_text": marks_text,
                "x_percent": x_pct,
                "y_percent": y_pct,
            }
        )
    return cells


def analyse_intro_page(client: genai.Client, image: bytes) -> dict[str, Any]:
    parts = [
        types.Part.from_text(text=INTRO_PAGE_PROMPT),
        types.Part.from_bytes(data=image, mime_type=_image_mime(image)),
    ]
    raw = _call_gemini_intro(client, parts)
    cells = _parse_intro_lines(raw)
    if not cells:
        raise ValueError("Could not detect a marks table on this page")
    return {"cells": cells}
