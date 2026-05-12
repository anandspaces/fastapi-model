"""Environment-driven configuration for the smart-OCR pipeline.

All tunables for layout geometry, Gemini call shape, and per-stage model selection
live here so the rest of the package can stay free of magic numbers and env reads.
"""

from __future__ import annotations

import os

from google.genai import types

from src.gemini_extract import MODEL_ID

# --- Layout geometry constants (percent coordinates, top-left origin) -----------------

REM_RIGHT_X1 = 76.0     # right margin left edge   (~grid 39 → pct)
REM_RIGHT_X2 = 96.0     # right margin right edge  (~grid 49 → pct)
REM_LEFT_X1 = 2.0       # left margin left edge    (~grid 2  → pct)
REM_LEFT_X2 = 20.0      # left margin right edge   (~grid 11 → pct)
REM_Y_MIN = 16.0        # top guard: below printed header
REM_Y_MAX = 94.0        # bottom guard
REM_MIN_H = 4.0         # minimum remark box height in %
ANCHOR_CLUSTER_GAP_GRID = 3   # anchor marks within N grid rows → one remark slot
CONTENT_X_MIN = 5.0     # leftmost grid column of student answer area
CONTENT_X_MAX = 37.0    # rightmost grid column of student answer area (38+ is right margin)

# Font-aware remark height calculation (A4 page)
FONT_SIZE_PT = 14.0
LINE_HEIGHT_FACTOR = 1.25
BOX_PADDING_PT = 4.0
AVG_CHAR_WIDTH_FACTOR = 0.55
PAGE_HEIGHT_PT = 842.0
PAGE_WIDTH_PT = 595.0
INTER_REMARK_GAP_PCT = 1.5

# Stage-2 / annotation Y-guards
REMARK_ANSWER_GUARD_PCT = 8.0   # skip past printed question-text zone from start_y

# Gemini call shape
CLASSIFY_MAX_OUTPUT_TOKENS = 8192
OCR_MAX_OUTPUT_TOKENS = 8192
STRUCTURE_MAX_OUTPUT_TOKENS = 65536
DEFAULT_CLASSIFY_WAIT_TIMEOUT_S = 12.0
RETRY_BACKOFF_S = 0.5

# Page-dedup similarity threshold
DEFAULT_DEDUP_SIM_THRESHOLD = 0.92


# --- Page-type vocabulary -------------------------------------------------------------

PAGE_TYPES = frozenset({"DUPLICATE", "CORRECTION", "PARAGRAPH", "WORD_LIST", "UNKNOWN"})
ANSWER_TYPES = frozenset({"correction", "paragraph", "word_list"})


# --- Gemini model selection -----------------------------------------------------------


def classify_model() -> str:
    return (os.environ.get("SMART_OCR_CLASSIFY_MODEL") or MODEL_ID or "").strip()


def ocr_model() -> str:
    return (os.environ.get("SMART_OCR_OCR_MODEL") or MODEL_ID or "").strip()


def structure_model() -> str:
    return (os.environ.get("SMART_OCR_STRUCTURE_MODEL") or MODEL_ID or "").strip()


# --- Thinking budget shim -------------------------------------------------------------


def thinking_off() -> types.ThinkingConfig:
    """Disable extended-thinking tokens (biggest single latency / cost drop on 2.5 series)."""
    return types.ThinkingConfig(thinking_budget=0)
