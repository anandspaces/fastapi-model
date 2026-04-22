"""Smart OCR extraction of question-answer blocks from student copy PDFs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from src.gemini_copy_ocr import copy_ocr_max_pages, rasterize_pdf_to_png_pages
from src.gemini_extract import MODEL_ID

_SMART_ITEM_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "question_id": types.Schema(type=types.Type.INTEGER),
        "question": types.Schema(type=types.Type.STRING),
        "student_answer": types.Schema(type=types.Type.STRING),
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
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
)

_SMART_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=_SMART_ITEM_SCHEMA,
)


def _build_prompt(language: str, total_pages: int) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Input may contain Hindi and English; preserve script exactly as written."
        if lang == "hi"
        else "Input may contain English and Hindi; preserve script exactly as written."
    )
    return f"""You are an expert AI assistant tasked with extracting all Questions and the student's handwritten Answers from the provided student copy pages.

Extract EACH question and its corresponding full answer (as written by the student).

CRITICAL:
- Pages provided are 1-indexed from 1..{total_pages}.
- Record start_page/end_page where the answer starts and ends.
- Compute start_y_position_percent/end_y_position_percent in [0,100] relative to the page.
- Compute marking_page/marking_x_position_percent/marking_y_position_percent for placing grading marks at the center of the answer body.
- {lang_note}

Return ONLY JSON array with this exact shape:
[
  {{
    "question_id": 1,
    "question": "Question text or short identifier",
    "student_answer": "Full extracted answer text",
    "start_page": 1,
    "start_y_position_percent": 10.0,
    "end_page": 2,
    "end_y_position_percent": 85.0,
    "marking_page": 1,
    "marking_x_position_percent": 50.0,
    "marking_y_position_percent": 45.0
  }}
]"""


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if m:
        return m.group(1).strip()
    return t


def _clamp_pct(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(100.0, x))


def _normalize_item(item: dict[str, Any], total_pages: int) -> dict[str, Any]:
    def page_num(v: Any, fallback: int = 1) -> int:
        try:
            p = int(v)
        except (TypeError, ValueError):
            p = fallback
        return max(1, min(total_pages, p))

    start_page = page_num(item.get("start_page"), 1)
    end_page = page_num(item.get("end_page"), start_page)
    if end_page < start_page:
        end_page = start_page
    marking_page = page_num(item.get("marking_page"), start_page)

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
        "start_page": start_page,
        "start_y_position_percent": _clamp_pct(item.get("start_y_position_percent", 0)),
        "end_page": end_page,
        "end_y_position_percent": _clamp_pct(item.get("end_y_position_percent", 100)),
        "marking_page": marking_page,
        "marking_x_position_percent": _clamp_pct(item.get("marking_x_position_percent", 50)),
        "marking_y_position_percent": _clamp_pct(item.get("marking_y_position_percent", 50)),
    }


def _parse_response(raw: str, total_pages: int) -> list[dict[str, Any]]:
    parsed = json.loads(_strip_json_fence(raw))
    if not isinstance(parsed, list):
        raise ValueError("Smart OCR response must be a JSON array.")
    out: list[dict[str, Any]] = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        normalized = _normalize_item(row, total_pages)
        if normalized["student_answer"]:
            out.append(normalized)
    if not out:
        raise ValueError("No question-answer blocks detected.")
    # Keep question IDs ordered and contiguous for downstream stability.
    for idx, item in enumerate(out, start=1):
        item["question_id"] = idx
    return out


def smart_ocr_extract_student_answers(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Extract structured question-answer spans and marking positions from PDF."""
    from src.gemini_copy_ocr import count_pdf_pages

    total_pages = count_pdf_pages(pdf_path)
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    max_pages = copy_ocr_max_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"PDF has {total_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi=150, request_id=request_id)
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(language, total_pages)
    parts: list[Any] = [types.Part.from_text(text=prompt)]
    for i, png in enumerate(png_pages):
        parts.append(types.Part.from_text(text=f"--- PAGE {i + 1} ---"))
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_SMART_RESPONSE_SCHEMA,
    )
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=parts,
        config=cfg,
    )
    raw = getattr(resp, "text", None) or ""
    if not raw.strip():
        raise ValueError("Smart OCR returned an empty response.")
    rows = _parse_response(raw, total_pages)
    return {"items": rows, "page_count": total_pages}
