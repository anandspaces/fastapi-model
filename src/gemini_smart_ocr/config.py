"""Environment-driven configuration for the smart-OCR pipeline.

All tunables for layout geometry, Gemini call shape, and per-step model
selection live here so the rest of the package can stay free of magic numbers.
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

# Y-guard: skip past printed question-text zone when placing remarks.
REMARK_ANSWER_GUARD_PCT = 8.0

# --- Gemini call shape ----------------------------------------------------------------

STEP2_MAX_OUTPUT_TOKENS = int(os.environ.get("SMART_OCR_STEP2_MAX_TOKENS", "12000"))
STEP3_MAX_OUTPUT_TOKENS = int(os.environ.get("SMART_OCR_STEP3_MAX_TOKENS", "16000"))
RETRY_BACKOFF_S = 0.5

# Hard per-call HTTP timeouts (milliseconds — google-genai 1.68 HttpOptions.timeout is ms).
STEP2_HTTP_TIMEOUT_MS = int(os.environ.get("SMART_OCR_STEP2_TIMEOUT_MS", "60000"))
STEP3_HTTP_TIMEOUT_MS = int(os.environ.get("SMART_OCR_STEP3_TIMEOUT_MS", "60000"))
# Whole-request deadline (seconds). Pipeline raises concurrent.futures.TimeoutError if exceeded.
SMART_OCR_TOTAL_TIMEOUT_S = float(os.environ.get("SMART_OCR_TOTAL_TIMEOUT_S", "180"))

# When an item spans more than this many pages, step 3 falls back to per-page calls
# (one Gemini call per page) to keep grid-coordinate context unambiguous.
STEP3_MULTI_PAGE_THRESHOLD = int(os.environ.get("SMART_OCR_STEP3_MULTI_PAGE_THRESHOLD", "4"))

# Page-dedup similarity threshold
DEFAULT_DEDUP_SIM_THRESHOLD = 0.92


# --- Answer-type vocabulary -----------------------------------------------------------

ANSWER_TYPES = frozenset({"correction", "paragraph", "word_list"})


# --- Gemini model selection -----------------------------------------------------------


def step2_model() -> str:
    return (os.environ.get("SMART_OCR_STEP2_MODEL") or MODEL_ID or "").strip()


def step3_model() -> str:
    return (os.environ.get("SMART_OCR_STEP3_MODEL") or MODEL_ID or "").strip()


# --- Thinking budget shim -------------------------------------------------------------


def thinking_off() -> types.ThinkingConfig:
    """Disable extended-thinking tokens (biggest single latency / cost drop on 2.5 series)."""
    return types.ThinkingConfig(thinking_budget=0)


def afc_off() -> types.AutomaticFunctionCallingConfig:
    """Skip the SDK's Automatic Function Calling planner — no tools are configured."""
    return types.AutomaticFunctionCallingConfig(disable=True)


def http_opts(timeout_ms: int) -> types.HttpOptions:
    """Build an HttpOptions with a hard per-call timeout. ``timeout`` is in milliseconds."""
    return types.HttpOptions(timeout=timeout_ms)


def finish_reason_name(resp) -> str:
    """Best-effort string name of ``resp.candidates[0].finish_reason`` (e.g. STOP, MAX_TOKENS)."""
    try:
        fr = resp.candidates[0].finish_reason
        return getattr(fr, "name", str(fr)) if fr is not None else "NONE"
    except (AttributeError, IndexError, TypeError):
        return "UNKNOWN"
