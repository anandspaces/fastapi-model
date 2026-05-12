"""Smart-OCR pipeline — public API.

Three-step pipeline:

  * Step 1 — grid overlay (image-only, no Gemini).
  * Step 2 — parallel per-page OCR + Q&A extraction (intro flag on page 1) → items[].
  * Step 3 — per-question-id parallel marking against the model answer key.

External callers should only import :func:`smart_ocr_run`.
"""

from .pipeline import smart_ocr_run
from .step2_page import extract_page
from .step3_mark import mark_item

__all__ = ["smart_ocr_run", "extract_page", "mark_item"]
