"""Gemini analysis flows for copy-check grading; HTTP ``data`` uses snake_case keys."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

ANALYSE_MODEL = os.environ.get("GEMINI_ANALYSE_MODEL")

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


INTRO_PAGE_PROMPT = """You are analysing the COVER / INTRO page of a student exam answer sheet.

The page contains a MARKS TABLE with columns: Q.No. | M.Mark | M.Obt.

TASK:
1. Find the M.Obt. column (where the teacher writes the marks obtained).
2. For EVERY row in that table — question rows AND the Total row — output one line in this exact format:
   questionNo|marksText|xPercent|yPercent
   - questionNo : integer (use 0 for the Total/Grand-Total row)
   - marksText  : the handwritten number if visible, otherwise leave it empty (nothing between the pipes)
   - xPercent   : horizontal centre of the M.Obt. cell as % of page width (0–100)
   - yPercent   : vertical centre of the M.Obt. cell as % of page height (0–100)
3. Include ALL rows, even if M.Obt. cell is empty.
4. Output ONLY the data lines. No headers, no explanation, no JSON, no markdown.

Example (values are illustrative only — use real values from the image):
1|4|74|30
2||74|33
3|3.5|74|36
0|91|74|88
"""


def _check_level_instruction(check_level: str) -> str:
    return (
        "- EVALUATION STRICTNESS: HARD. Be extremely strict. All answers must be strictly evaluated and normally score less than 50% of the total marks unless they are absolutely perfect without any flaws."
        if check_level.strip().lower() == "hard"
        else "- EVALUATION STRICTNESS: MODERATE. Grade normally, but keep medium or average answers around or below 50% of the total marks."
    )


def _instruction_block(instruction_name: str | None) -> str:
    if instruction_name and instruction_name.strip():
        return f"EXTRA ANSWER INSTRUCTIONS:\n{instruction_name.strip()}\n\n"
    return ""


def _language_block_teacher(language: str) -> str:
    lang = language.strip().lower()
    if lang == "hi":
        return """LANGUAGE INSTRUCTION:
This is a Hindi-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in Hindi. No English feedback text at all.

HINDI VOCABULARY RULES (apply strictly):
- Use "saraahaniyan" or "Utkrisht" for praise — NOT just "achha".
- Use "Prayas" when noting effort.
- Nishkarsh (conclusion) must be future-oriented: "Aapka nishkarsh bhavishya unmukhi hona chahiye."
- Address the student as "Aap" / "Aapka" — NEVER "Tumne" or "Tu".
- "Introduction" → write "Prashtavana" or "Parichay" — never the English word.
- NEVER use the word "Shabash".
- "Utpatti" → use "Prashtuti".
- Strong conclusion: "Aapka nishkarsh prabhavshali hai."
- Line could be more specific: "Yah line aur vishishth ho sakti hai; udaharan ke sath samjhaya ja sakta tha."
- Replace "Sahi dhang se samjhaya hai" with "Sahi dhang se prastut kiya hai." """
    return """LANGUAGE INSTRUCTION:
This is an English-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in English. No Hindi feedback text at all."""


def build_full_analysis_prompt(
    *,
    question_title: str,
    model_description: str,
    total_marks: int,
    language: str,
    instruction_name: str | None,
    check_level: str,
    max_page_index: int,
) -> str:
    check_level_instruction = _check_level_instruction(check_level)
    ib = _instruction_block(instruction_name)
    lb = _language_block_teacher(language)
    return f"""You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

You will be provided with images of the student's answer pages in sequence. Some pages may appear mostly white or contain very little visible ink — the student may have written lightly or used a light pen. ALWAYS attempt to read all pages. If a page genuinely has no answer at all, note it but still grade the rest of the answer.

{ib}QUESTION TITLE:
{question_title}

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
{model_description}

TOTAL MARKS FOR THIS QUESTION: {total_marks}

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
{check_level_instruction}
- Award marksAwarded as a DECIMAL in multiples of 0.5 (e.g. 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 …). NEVER exceed {total_marks}.
- Marking tiers by question size:
    8-mark question  → Bahot achha (excellent): 3.5  |  Moderate: 2.5  |  Low: 1.5
    12-mark question → Bahot achha (excellent): 5    |  Moderate: 3    |  Low: 2.5
    Other totals     → scale proportionally using 0.5-step decimals.
- You are a STRICT examiner. Default to FEWER marks, not more.
- FULL marks ONLY if answer nearly perfectly matches the scheme — all key points, correct terminology, clear reasoning. This is rare.
- Do NOT reward effort or length — reward accuracy and relevance only.
- A long but mostly irrelevant answer scores LOW. A short precise answer can outscore a long vague one.
- Do NOT give benefit of the doubt. If a key point is not clearly stated, do not assume it was implied.
- Deduct for: wrong facts, missing key terms, no examples when required, incorrect conclusions.
- When in doubt between two values, always choose the LOWER one.

YOUR TASKS:

1. READ the handwritten text from all images (including labels, captions, diagram text).
2. GRADE objectively against the marking scheme. If the scheme expects diagrams/figures, evaluate whether the student addressed those (sketches, descriptions, labels).
3. ANNOTATE the answer: mark 2–5 specific spots in the student's writing with short teacher-style comments. Make annotations PRECISE — annotate only the specific word/phrase, not the whole line.

ANNOTATION PLACEMENT RULES (CRITICAL — prevents overlapping comments):
- On any single page, every annotation MUST have a yPositionPercent that differs by AT LEAST 12 from every other annotation on that same page. Space them out evenly across the page height.
- If two annotations are naturally close together (within 12% of each other vertically), pick only the more important one; do NOT place both at nearly the same y position.
- Alternate comment placement: for annotations that are on the LEFT half of the text (xEndPercent < 55), keep xEndPercent ≤ 55. For annotations on the RIGHT half (xStartPercent > 45), keep xStartPercent ≥ 45. This ensures comments are anchored to different horizontal zones and do not collide.
- Prefer spreading annotations across different pages when the answer spans multiple pages — do not cluster all annotations on page 0.
- IMPORTANT: ALL annotations MUST have "isPositive": true. Only circle/annotate things done correctly. Do NOT mark errors with annotations — errors should be mentioned in the "improvements" field instead.

TONE & LANGUAGE GUIDELINES:

Write ALL feedback as a professional yet approachable teacher would — clear, constructive, and specific. Avoid robotic phrasing. Think of the tone an experienced senior examiner uses: direct, helpful, respectful.

GOOD annotation comments (professional but human):
  - "Well articulated — this directly addresses the marking scheme."
  - "Correct definition. Clear and concise."
  - "This definition is inaccurate — please revise from your textbook."
  - "You started well but left this point incomplete. Elaborate further next time."
  - "Good reasoning, but the correct formula is F = ma, not F = mv."
  - "Diagram is present but labels are missing — always label axes and units."

BAD (too robotic/generic — AVOID these):
  - "The student correctly identified the concept."
  - "Improvement needed in this area."
  - "Good point."
  - "Incorrect."

For goodPoints: Address the student directly and be specific about what was done well.
For improvements: Be specific — mention what was missed and why it matters for marks.
For finalReview: Write 2–3 sentences as a professional remark — constructive and encouraging.

{lb}

══════════════════════════════════════════
OUTPUT FORMAT:
══════════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no explanation outside JSON). Use these exact **snake_case** keys:
{{
  "student_text": "<transcribe the exact handwritten text you see across all pages — skip printed question headers>",
  "marks_awarded": <decimal in multiples of 0.5, range 0 to {total_marks}; never exceed {total_marks}>,
  "confidence_percent": <float 0-100>,
  "good_points": "<bullet-point list — each point should sound like a real teacher praising the student>",
  "improvements": "<bullet-point list — each point should sound like a real teacher pointing out what to fix and why>",
  "final_review": "<2-3 sentence handwritten-note-style overall review — warm, personal, constructive>",
  "annotations": [
    {{
      "page_index": <int 0-indexed corresponding to the image sequence. Maximum is {max_page_index}>,
      "y_position_percent": <float 0-100 indicating the approximate vertical position of the specific text to underline>,
      "x_start_percent": <float 0-100 indicating the tight horizontal start of the specific word(s) to underline>,
      "x_end_percent": <float 0-100 indicating the tight horizontal end of the specific word(s) to underline>,
      "comment": "<short, warm, colloquial teacher remark — praise what's right, sound human>",
      "is_positive": true,
      "line_style": "straight"
    }}
  ]
}}
NOTE: Every annotation MUST have "is_positive": true. Do NOT produce any negative/cross annotations. Errors should be mentioned only in the "improvements" field.
"""


def build_cached_ocr_prompt(
    *,
    cached_student_text: str,
    question_title: str,
    model_description: str,
    total_marks: int,
    language: str,
    instruction_name: str | None,
    check_level: str,
    page_count: int,
) -> str:
    check_level_instruction = _check_level_instruction(check_level)
    ib = _instruction_block(instruction_name)
    lb = _language_block_teacher(language)
    max_pi = max(0, page_count - 1)
    return f"""You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

The student's handwritten answer has already been transcribed for you (OCR result):
\"\"\"
{cached_student_text}
\"\"\"

{ib}QUESTION TITLE:
{question_title}

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
{model_description}

TOTAL MARKS FOR THIS QUESTION: {total_marks}

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
{check_level_instruction}
- Award marksAwarded as a DECIMAL in multiples of 0.5 (e.g. 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 …). NEVER exceed {total_marks}.
- Marking tiers by question size:
    8-mark question  → Bahot achha (excellent): 3.5  |  Moderate: 2.5  |  Low: 1.5
    12-mark question → Bahot achha (excellent): 5    |  Moderate: 3    |  Low: 2.5
    Other totals     → scale proportionally using 0.5-step decimals.
- You are a STRICT examiner. Default to FEWER marks, not more.
- FULL marks ONLY if answer nearly perfectly matches the scheme — all key points, correct terminology, clear reasoning. This is rare.
- Do NOT reward effort or length — reward accuracy and relevance only.
- A long but mostly irrelevant answer scores LOW. A short precise answer can outscore a long vague one.
- Do NOT give benefit of the doubt. If a key point is not clearly stated, do not assume it was implied.
- Deduct for: wrong facts, missing key terms, no examples when required, incorrect conclusions.
- When in doubt between two values, always choose the LOWER one.

YOUR TASKS:

1. GRADE objectively against the marking scheme based on the transcribed text above.
3. ANNOTATE the answer: mark 2–5 specific spots with short teacher-style comments. Estimate page/position from the text structure (the answer spans {page_count} page(s)).

ANNOTATION PLACEMENT RULES (CRITICAL — prevents overlapping comments):
- On any single page, every annotation MUST have a yPositionPercent that differs by AT LEAST 20 from every other annotation ON THE SAME SIDE (left or right). Space them out evenly across the page height (e.g. 10, 30, 50, 70, 90).
- Two annotations that would be on the SAME side and within 20% of each other vertically → keep only the more important one and DISCARD the other.
- Two annotations that are on OPPOSITE sides (one left half, one right half) CAN share a similar yPositionPercent — this is fine and encouraged to keep them visually spread.
- Decide left vs right using xStartPercent and xEndPercent: if midX = (xStart+xEnd)/2 < 50, place it on the LEFT half (xEndPercent ≤ 50); otherwise on the RIGHT half (xStartPercent ≥ 50). Strictly alternate left/right when possible.
- Prefer spreading annotations across different pages — do not cluster all on page 0.
- Aim for at most 2–3 annotations per page. If you have more, move the extras to other pages or drop the least important ones.
- IMPORTANT: ALL annotations MUST have "isPositive": true. Only mark things worth circling positively (correct facts, good phrases, relevant points). Do NOT mark errors with annotations — errors should be mentioned in the "improvements" field instead.

TONE & LANGUAGE GUIDELINES:
Write ALL feedback as a professional yet approachable teacher — clear, constructive, specific.

For goodPoints: Address the student directly and be specific about what was done well.
For improvements: Be specific — mention what was missed and why it matters for marks.
For finalReview: Write 2–3 sentences as a professional remark — warm, personal, constructive.

{lb}

══════════════════════════════════════════
OUTPUT FORMAT:
══════════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no explanation outside JSON). Use these exact **snake_case** keys:
{{
  "student_text": "<exactly the OCR text shown in the quoted block above — escape properly as one JSON string>",
  "marks_awarded": <decimal in multiples of 0.5, range 0 to {total_marks}; never exceed {total_marks}>,
  "confidence_percent": <float 0-100>,
  "good_points": "<bullet-point list — each point should sound like a real teacher praising the student>",
  "improvements": "<bullet-point list — each point should sound like a real teacher pointing out what to fix and why>",
  "final_review": "<2-3 sentence handwritten-note-style overall review — warm, personal, constructive>",
  "annotations": [
    {{
      "page_index": <int 0-indexed. Maximum is {max_pi}>,
      "y_position_percent": <float 0-100>,
      "x_start_percent": <float 0-100>,
      "x_end_percent": <float 0-100>,
      "comment": "<short, warm teacher remark>",
      "is_positive": true,
      "line_style": "straight"
    }}
  ]
}}
NOTE: Every annotation MUST have "is_positive": true. Do NOT produce negative/cross annotations. Mention errors only in the "improvements" field.
"""


def build_combined_prompt(question_summary: str) -> str:
    return f"""You are an experienced school teacher writing a detailed end-of-paper comment for a student.
Below are the per-question analysis results:

{question_summary}

YOUR TASK:
Write a "final_review" JSON field that is a flowing, natural paragraph-style teacher comment of AT LEAST 150 words (aim for 180-220 words). It should read exactly like a real teacher's handwritten remark at the end of a corrected answer sheet — warm, personal, specific, and professional.

Structure (all in one block of plain prose, no headings, no bullets, no numbering, no markdown):
1. Start with 2-3 sentences acknowledging what the student did well across the paper, mentioning specific questions or topics.
2. Write 3-4 sentences identifying the most important weaknesses, with concrete examples from the student's answers (e.g. "In Q3 you left the conclusion incomplete…").
3. Give 2-3 sentences of clear, actionable advice on how to improve — specific study tips or practice habits.
4. End with 1-2 encouraging sentences motivating the student to keep working hard.

Keep every sentence natural and conversational — the way a teacher actually writes, not like a report. Do NOT use any symbols, asterisks, or special characters.

Also return:
- "overall_improvements": 4 plain sentences of improvement points (one per line, no symbols).
- "one_thing_to_write": ONE sentence — the single most impactful practice tip.

LANGUAGE: Match the dominant language in the question improvements above (Hindi or English).

Return ONLY valid JSON (snake_case keys), no markdown:
{{
  "final_review": "<flowing paragraph of 150-220 words, sentences separated by \\n>",
  "overall_improvements": "<4 lines separated by \\n>",
  "one_thing_to_write": "<one sentence>"
}}
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
    t = _strip_json_fence(text)
    t = re.sub(r",\s*([}\]])", r"\1", t)
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


def _first_key(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _clamp_annotations_response(
    annotations: list[Any], page_count: int
) -> list[dict[str, Any]]:
    """Normalize annotations to API snake_case keys (Gemini may return camelCase)."""
    if page_count <= 0:
        return []
    out: list[dict[str, Any]] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        raw_pi = _first_key(a, "pageIndex", "page_index", default=0)
        try:
            pi = int(raw_pi)
        except (TypeError, ValueError):
            pi = 0
        pi = max(0, min(page_count - 1, pi))
        ls = str(_first_key(a, "lineStyle", "line_style", default="straight")).lower()
        if ls not in ("straight", "zigzag"):
            ls = "straight"
        out.append(
            {
                "page_index": pi,
                "y_position_percent": float(
                    _first_key(a, "yPositionPercent", "y_position_percent", default=0)
                ),
                "x_start_percent": float(
                    _first_key(a, "xStartPercent", "x_start_percent", default=0)
                ),
                "x_end_percent": float(
                    _first_key(a, "xEndPercent", "x_end_percent", default=0)
                ),
                "comment": str(_first_key(a, "comment", default="")),
                "is_positive": True,
                "line_style": "straight",
            }
        )
    return out


def _parse_string_or_list(val: Any) -> str:
    if isinstance(val, list):
        return "\n".join(f"• {x}" for x in val)
    return str(val or "")


def _parse_analysis_obj(
    data: dict[str, Any], total_marks: int, page_count: int
) -> dict[str, Any]:
    raw_marks = _first_key(data, "marksAwarded", "marks_awarded", default=0)
    try:
        marks = float(raw_marks)
    except (TypeError, ValueError):
        marks = 0.0
    marks = _clamp_marks(marks, total_marks)

    try:
        conf = float(
            _first_key(data, "confidencePercent", "confidence_percent", default=0)
        )
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(100.0, conf))

    ann = data.get("annotations")
    if not isinstance(ann, list):
        ann = []

    gp = _parse_string_or_list(_first_key(data, "goodPoints", "good_points", default=""))
    imp = _parse_string_or_list(
        _first_key(data, "improvements", "improvements", default="")
    )

    return {
        "student_text": str(_first_key(data, "studentText", "student_text", default="")),
        "marks_awarded": marks,
        "confidence_percent": conf,
        "good_points": gp,
        "improvements": imp,
        "final_review": str(_first_key(data, "finalReview", "final_review", default="")),
        "annotations": _clamp_annotations_response(ann, page_count),
    }


def _extract_analysis_fallback(text: str, total_marks: int, page_count: int) -> dict[str, Any] | None:
    t = _strip_json_fence(text)

    def grab_str(*keys: str) -> str:
        for key in keys:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.DOTALL)
            if not m:
                continue
            s = m.group(1)
            try:
                return json.loads(f'"{s}"')
            except json.JSONDecodeError:
                return s.replace("\\n", "\n").replace("\\t", "\t")
        return ""

    def grab_num(*keys: str) -> float | None:
        for key in keys:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*([-0-9.]+)', t)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    st = grab_str("studentText", "student_text")
    ma = grab_num("marksAwarded", "marks_awarded")
    cp = grab_num("confidencePercent", "confidence_percent")
    gp = grab_str("goodPoints", "good_points")
    imp = grab_str("improvements", "improvements")
    fr = grab_str("finalReview", "final_review")

    if ma is None:
        ma = 0.0
    if cp is None:
        cp = 0.0

    annotations: list[dict[str, Any]] = []
    for block in re.finditer(r"\{[^{}]*(?:pageIndex|page_index)[^{}]*\}", t, re.DOTALL):
        try:
            obj = json.loads(block.group(0))
            if isinstance(obj, dict) and (
                "pageIndex" in obj or "page_index" in obj
            ):
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
            fr = str(_first_key(raw, "final_review", "finalReview", default=""))
            oi = str(
                _first_key(
                    raw, "overall_improvements", "overallImprovements", default=""
                )
            )
            ot = str(
                _first_key(raw, "one_thing_to_write", "oneThingToWrite", default="")
            )
            return {
                "final_review": fr,
                "overall_improvements": oi,
                "one_thing_to_write": ot,
            }
    t = _strip_json_fence(text)

    def grab(*keys: str) -> str:
        for key in keys:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.DOTALL)
            if not m:
                continue
            s = m.group(1)
            try:
                return json.loads(f'"{s}"')
            except json.JSONDecodeError:
                return s.replace("\\n", "\n").replace("\\t", "\t")
        return ""

    fr = grab("final_review", "finalReview")
    oi = grab("overall_improvements", "overallImprovements")
    ot = grab("one_thing_to_write", "oneThingToWrite")
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
            parsed = _parse_analysis_json(text, total_marks, page_count)
            return parsed
        except Exception as e:
            last_err = e
            log.warning("Analysis attempt %s failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.3 * attempt)
    assert last_err is not None
    raise last_err


def _call_gemini_combined(client: genai.Client, contents: list[Any]) -> dict[str, Any]:
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
            parsed = _parse_combined_json(text)
            overall_review = str(
                _first_key(parsed, "final_review", "finalReview", default="")
            ) or str(_first_key(parsed, "overall_review", "overallReview", default=""))
            out = {
                "overall_improvements": str(
                    _first_key(
                        parsed, "overall_improvements", "overallImprovements", default=""
                    )
                ),
                "one_thing_to_write": str(
                    _first_key(
                        parsed, "one_thing_to_write", "oneThingToWrite", default=""
                    )
                ),
                "overall_review": overall_review,
            }
            return out
        except Exception as e:
            last_err = e
            log.warning("Combined review attempt %s failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.3 * attempt)
    assert last_err is not None
    raise last_err


def _call_gemini_intro_plain(client: genai.Client, contents: list[Any]) -> str:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=ANALYSE_MODEL,
                contents=contents,
                config=CFG_INTRO,
            )
            return (getattr(response, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            log.warning("Intro-page attempt %s failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.3 * attempt)
    assert last_err is not None
    raise last_err


def analyse_full_images(
    client: genai.Client,
    images: list[bytes],
    question_title: str,
    model_description: str,
    total_marks: int,
    language: str,
    *,
    instruction_name: str | None = None,
    check_level: str = "Moderate",
) -> dict[str, Any]:
    page_count = len(images)
    max_pi = max(0, page_count - 1)
    prompt = build_full_analysis_prompt(
        question_title=question_title.strip(),
        model_description=model_description.strip(),
        total_marks=total_marks,
        language=language,
        instruction_name=instruction_name,
        check_level=check_level,
        max_page_index=max_pi,
    )
    parts: list[Any] = [types.Part.from_text(text=prompt)]
    for img in images:
        parts.append(types.Part.from_bytes(data=img, mime_type=_image_mime(img)))
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
    *,
    instruction_name: str | None = None,
    check_level: str = "Moderate",
) -> dict[str, Any]:
    prompt = build_cached_ocr_prompt(
        cached_student_text=cached_student_text,
        question_title=question_title.strip(),
        model_description=model_description.strip(),
        total_marks=total_marks,
        language=language,
        instruction_name=instruction_name,
        check_level=check_level,
        page_count=page_count,
    )
    parts = [types.Part.from_text(text=prompt)]
    result = _call_gemini_analysis(
        client,
        parts,
        CFG_CACHED_OCR,
        total_marks,
        page_count,
    )
    result["student_text"] = cached_student_text
    return result


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
    raw = _call_gemini_intro_plain(client, parts)
    cells = _parse_intro_lines(raw)
    return {"cells": cells}
