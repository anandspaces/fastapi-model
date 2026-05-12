"""Smart OCR pipeline — public API.

Three Gemini stages composed into one orchestrator:

  1. ``classify_and_annotate_page`` — classify page type + place anchor/remark marks.
  2. ``ocr_single_page``            — type-aware OCR of the handwriting.
  3. ``structure_qa_with_fallback`` — flatten OCR text into question rows.

External callers should only import:
  - :func:`smart_ocr_extract_student_answers` — full PDF → items + page_count.
  - :func:`enrich_remarks_from_annotations`   — merge grading feedback into remarks.
"""

from .enrich import enrich_remarks_from_annotations
from .pipeline import smart_ocr_extract_student_answers

__all__ = [
    "smart_ocr_extract_student_answers",
    "enrich_remarks_from_annotations",
]
