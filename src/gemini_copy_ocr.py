"""Essay copy OCR: whole-PDF (File API) or per-page raster + parallel Gemini calls."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.gemini_extract import MODEL_ID, wait_for_file_active

log = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 25 * 1024 * 1024
_DEFAULT_MAX_PAGES = 30
_DEFAULT_RASTER_DPI = 150
_DEFAULT_PARALLEL_WORKERS = 5

_OCR_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "text": types.Schema(
            type=types.Type.STRING,
            description="Full verbatim transcript of the student copy (handwriting and print).",
        ),
    },
    required=["text"],
    property_ordering=["text"],
)


def _env_int(name: str, default: int) -> int:
    load_dotenv()
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw, 10)
        return v if v > 0 else default
    except ValueError:
        log.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def copy_ocr_max_bytes() -> int:
    return _env_int("COPY_OCR_MAX_BYTES", _DEFAULT_MAX_BYTES)


def copy_ocr_max_pages() -> int:
    return _env_int("COPY_OCR_MAX_PAGES", _DEFAULT_MAX_PAGES)


def copy_ocr_raster_dpi() -> int:
    return _env_int("COPY_OCR_RASTER_DPI", _DEFAULT_RASTER_DPI)


def copy_ocr_parallel_workers() -> int:
    return _env_int("COPY_OCR_PARALLEL_WORKERS", _DEFAULT_PARALLEL_WORKERS)


def count_pdf_pages(pdf_path: Path) -> int:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path), strict=False)
    return len(reader.pages)


def _ocr_prompt(language: str) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Transcribe Devanagari Hindi faithfully where it appears; keep English words in Latin script."
        if lang.startswith("hi")
        else "Transcribe in English; keep any Hindi/other script as written."
    )
    return f"""You are given a PDF of a student's handwritten or printed exam answer (typically one essay-style response).

Task: OCR — produce a single plain-text transcript of everything the student wrote. No grading, no commentary, no summary.

Rules:
- {lang_note}
- Preserve paragraph breaks and approximate line breaks where they matter for readability.
- Transcribe formulas, equations, and symbols as plain text or Unicode as best you can.
- If a word is illegible, write [illegible] for that span.
- Do not invent content not visible in the document.
- Read all pages in order and concatenate in reading order.

Respond ONLY with JSON matching the schema: one object with key "text" (string, the full transcript)."""


def _parse_ocr_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    data = json.loads(text)
    if isinstance(data, dict) and "text" in data:
        return str(data["text"]).strip()
    raise ValueError("OCR JSON missing 'text' field")


def ocr_essay_copy_pdf(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str | None = None,
) -> dict[str, object]:
    """Upload PDF, one Gemini OCR call. Returns ``{{"text": str, "page_count": int}}``.

    Raises ``ValueError`` for page limits or empty transcript. Raises on Gemini/network errors.
    """
    rid = request_id or str(uuid.uuid4())
    label = pdf_path.name
    max_pages = copy_ocr_max_pages()
    t0 = time.monotonic()

    log.info(
        "copy_ocr[%s] start path=%s max_pages=%s",
        rid,
        label,
        max_pages,
    )

    try:
        n_pages = count_pdf_pages(pdf_path)
    except Exception as e:
        log.exception(
            "copy_ocr[%s] pypdf failed path=%s err=%s",
            rid,
            label,
            e,
        )
        raise ValueError(f"Could not read PDF page count: {e}") from e

    elapsed_count = time.monotonic() - t0
    log.info(
        "copy_ocr[%s] page_count=%s count_phase_s=%.3f",
        rid,
        n_pages,
        elapsed_count,
    )

    if n_pages < 1:
        raise ValueError("PDF has no pages.")
    if n_pages > max_pages:
        raise ValueError(
            f"PDF has {n_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    client = genai.Client(api_key=api_key)
    prompt = _ocr_prompt(language)
    log.debug(
        "copy_ocr[%s] prompt_preview=%s",
        rid,
        (prompt[:400] + "…") if len(prompt) > 400 else prompt,
    )

    t_upload = time.monotonic()
    log.info("copy_ocr[%s] uploading file…", rid)
    uploaded = client.files.upload(file=str(pdf_path))
    state = str(getattr(uploaded, "state", "") or "").upper()
    if not uploaded.uri or state == "PROCESSING":
        log.info(
            "copy_ocr[%s] file state=%s waiting for ACTIVE…",
            rid,
            state or "?",
        )
        uploaded = wait_for_file_active(client, uploaded)
        log.info(
            "copy_ocr[%s] file active name=%s uri_set=%s",
            rid,
            getattr(uploaded, "name", ""),
            bool(getattr(uploaded, "uri", None)),
        )

    if not uploaded.uri:
        raise RuntimeError("File has no URI after upload.")

    log.info(
        "copy_ocr[%s] upload_wait_s=%.3f",
        rid,
        time.monotonic() - t_upload,
    )

    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_OCR_SCHEMA,
    )

    t_gen = time.monotonic()
    log.info(
        "copy_ocr[%s] generate_content model=%s pages=%s",
        rid,
        MODEL_ID,
        n_pages,
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[uploaded, types.Part.from_text(text=prompt)],
        config=cfg,
    )
    raw = getattr(response, "text", None) or ""
    gen_s = time.monotonic() - t_gen
    log.info(
        "copy_ocr[%s] generate_done raw_chars=%s gen_s=%.3f",
        rid,
        len(raw),
        gen_s,
    )

    if not raw.strip():
        raise RuntimeError("Gemini returned no OCR text.")

    try:
        text = _parse_ocr_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log.exception(
            "copy_ocr[%s] parse OCR JSON failed raw_prefix=%r",
            rid,
            raw[:500],
        )
        raise ValueError(f"Could not parse OCR response: {e}") from e

    if not text:
        raise ValueError("OCR produced empty text.")

    total_s = time.monotonic() - t0
    log.info(
        "copy_ocr[%s] ok transcript_chars=%s total_s=%.3f",
        rid,
        len(text),
        total_s,
    )
    return {"text": text, "page_count": n_pages}


def _page_ocr_prompt(language: str, page_1based: int, total_pages: int) -> str:
    lang = (language or "en").strip().lower()
    lang_note = (
        "Transcribe Devanagari Hindi faithfully where it appears; keep English words in Latin script."
        if lang.startswith("hi")
        else "Transcribe in English; keep any Hindi/other script as written."
    )
    return f"""You are given a single-page image (page {page_1based} of {total_pages}) from a student's handwritten or printed exam answer.

Task: OCR — transcribe only the text visible on THIS page. No grading, no commentary, no summary.

Rules:
- {lang_note}
- Preserve line breaks where they aid readability.
- Transcribe formulas as plain text or Unicode as best you can.
- If a word is illegible, write [illegible] for that span.
- Do not invent content not visible on this page.

Respond ONLY with JSON matching the schema: one object with key "text" (string)."""


def _rasterize_pdf_page_task(args: tuple[Path, int, float, str]) -> tuple[int, bytes]:
    """Render one PDF page to PNG in its own process thread (opens its own document handle)."""
    import fitz

    pdf_path, page_index, scale, rid = args
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png = pix.tobytes("png")
        log.debug(
            "copy_ocr_raster[%s] raster page %s png_bytes=%s",
            rid,
            page_index + 1,
            len(png),
        )
        return page_index, png
    finally:
        doc.close()


def rasterize_pdf_to_png_pages(pdf_path: Path, dpi: int, request_id: str) -> list[bytes]:
    """Render each PDF page to PNG bytes (RGB, no alpha).

    Uses :func:`copy_ocr_parallel_workers` threads to rasterize pages in parallel when
    there is more than one page (single-page PDFs stay sequential to avoid pool overhead).
    """
    import fitz

    rid = request_id
    log.info(
        "copy_ocr_raster[%s] rasterize start path=%s dpi=%s",
        rid,
        pdf_path.name,
        dpi,
    )
    t0 = time.monotonic()
    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        if n < 1:
            raise ValueError("PDF has no pages.")
    finally:
        doc.close()

    scale = dpi / 72.0
    workers_cfg = copy_ocr_parallel_workers()
    pool_workers = max(1, min(workers_cfg, n))

    if n == 1 or pool_workers == 1:
        doc2 = fitz.open(str(pdf_path))
        try:
            mat = fitz.Matrix(scale, scale)
            out: list[bytes] = []
            for i in range(n):
                page = doc2.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out.append(pix.tobytes("png"))
                log.debug(
                    "copy_ocr_raster[%s] raster page %s png_bytes=%s",
                    rid,
                    i + 1,
                    len(out[-1]),
                )
        finally:
            doc2.close()
    else:
        tasks = [(pdf_path, i, scale, rid) for i in range(n)]
        log.info(
            "copy_ocr_raster[%s] parallel raster workers=%s pages=%s",
            rid,
            pool_workers,
            n,
        )
        with ThreadPoolExecutor(max_workers=pool_workers) as executor:
            indexed = list(executor.map(_rasterize_pdf_page_task, tasks))
        indexed.sort(key=lambda x: x[0])
        out = [png for _, png in indexed]

    log.info(
        "copy_ocr_raster[%s] rasterize done pages=%s workers=%s elapsed_s=%.3f",
        rid,
        len(out),
        pool_workers,
        time.monotonic() - t0,
    )
    return out


def _ocr_single_raster_page(args: tuple) -> str:
    (
        page_1based,
        png_bytes,
        api_key,
        language,
        total_pages,
        request_id,
    ) = args
    rid = request_id
    client = genai.Client(api_key=api_key)
    prompt = _page_ocr_prompt(language, page_1based, total_pages)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_OCR_SCHEMA,
    )
    t0 = time.monotonic()
    log.info(
        "copy_ocr_raster[%s] gemini page %s/%s start",
        rid,
        page_1based,
        total_pages,
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
        config=cfg,
    )
    raw = getattr(response, "text", None) or ""
    elapsed = time.monotonic() - t0
    log.info(
        "copy_ocr_raster[%s] gemini page %s/%s done raw_chars=%s elapsed_s=%.3f",
        rid,
        page_1based,
        total_pages,
        len(raw),
        elapsed,
    )
    if not raw.strip():
        raise RuntimeError(f"Gemini returned no OCR text for page {page_1based}.")
    try:
        text = _parse_ocr_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log.exception(
            "copy_ocr_raster[%s] parse failed page=%s raw_prefix=%r",
            rid,
            page_1based,
            raw[:400],
        )
        raise ValueError(
            f"Could not parse OCR JSON for page {page_1based}: {e}"
        ) from e
    return text


def ocr_essay_copy_pdf_rasterized(
    pdf_path: Path,
    api_key: str,
    language: str,
    *,
    request_id: str | None = None,
) -> dict[str, object]:
    """Rasterize each page to PNG, OCR with up to *COPY_OCR_PARALLEL_WORKERS* threads.

    Returns ``{{"text": str, "page_count": int, "dpi": int, "workers": int}}``.
    """
    rid = request_id or str(uuid.uuid4())
    label = pdf_path.name
    max_pages = copy_ocr_max_pages()
    dpi = copy_ocr_raster_dpi()
    workers = copy_ocr_parallel_workers()
    t0 = time.monotonic()

    log.info(
        "copy_ocr_raster[%s] start path=%s max_pages=%s dpi=%s workers=%s",
        rid,
        label,
        max_pages,
        dpi,
        workers,
    )

    try:
        n_pages = count_pdf_pages(pdf_path)
    except Exception as e:
        log.exception("copy_ocr_raster[%s] pypdf count failed path=%s", rid, label)
        raise ValueError(f"Could not read PDF page count: {e}") from e

    if n_pages < 1:
        raise ValueError("PDF has no pages.")
    if n_pages > max_pages:
        raise ValueError(
            f"PDF has {n_pages} page(s); maximum allowed is {max_pages} (COPY_OCR_MAX_PAGES)."
        )

    png_pages = rasterize_pdf_to_png_pages(pdf_path, dpi, rid)
    if len(png_pages) != n_pages:
        log.warning(
            "copy_ocr_raster[%s] page count mismatch pypdf=%s fitz=%s",
            rid,
            n_pages,
            len(png_pages),
        )

    tasks = [
        (
            i + 1,
            png_pages[i],
            api_key,
            language,
            n_pages,
            rid,
        )
        for i in range(len(png_pages))
    ]
    pool_workers = max(1, min(workers, len(tasks)))
    log.info(
        "copy_ocr_raster[%s] thread_pool workers=%s tasks=%s",
        rid,
        pool_workers,
        len(tasks),
    )
    t_pool = time.monotonic()
    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        page_texts = list(executor.map(_ocr_single_raster_page, tasks))
    log.info(
        "copy_ocr_raster[%s] thread_pool done elapsed_s=%.3f",
        rid,
        time.monotonic() - t_pool,
    )

    parts: list[str] = []
    for i, txt in enumerate(page_texts):
        t = (txt or "").strip()
        if not t:
            raise ValueError(f"OCR produced empty text for page {i + 1}.")
        parts.append(f"--- Page {i + 1} ---\n{t}")
    full_text = "\n\n".join(parts)

    total_s = time.monotonic() - t0
    log.info(
        "copy_ocr_raster[%s] ok pages=%s transcript_chars=%s total_s=%.3f",
        rid,
        n_pages,
        len(full_text),
        total_s,
    )
    return {
        "text": full_text,
        "page_count": n_pages,
        "dpi": dpi,
        "workers": pool_workers,
    }
