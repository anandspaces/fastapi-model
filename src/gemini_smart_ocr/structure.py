"""Step-2 deterministic merger.

Each page's step-2 Gemini call returns its own question records (label, continuation
flag, partial student answer, y-bounds). This file stitches those per-page records
into global ``items[]`` matching the legacy ``smart_ocr`` item shape — no Gemini
call required.

Merge rules:
  * A record with ``is_continuation=true`` attaches to the most-recent labelled
    record that came from a prior page (same booklet stream).
  * Otherwise the record opens a new logical question keyed by ``question_id``
    (parsed from ``question_label`` via the same regex used by the legacy
    pipeline) or by a synthetic gap-id when no label is detectable.
  * ``student_answer`` across pages is joined with ``"\\n\\n"``.
  * ``start_page`` is the first appearance; ``end_page`` is the last (including
    continuations).
  * ``start_y_position_percent`` is the y from the first appearance;
    ``end_y_position_percent`` is the y from the last.
  * ``marking_page`` = ``start_page``; ``marking_x_position_percent`` = 85.0
    (right-margin centerline); ``marking_y_position_percent`` = ``start_y``.

Section assignment is best-effort: items keep ``section_name`` from the model
answer (filled by step 3) or the default ``"General"`` when unknown.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .config import ANSWER_TYPES
from .parsing import clamp_pct, page_num

log = logging.getLogger(__name__)


_QUESTION_NUM_RE = re.compile(
    r"""
    (?:
        \b(?:que(?:stion|l)?|q)\s*[:\.\->]*\s*(\d{1,3})\b      # Que:7 / Q.3 / Question 10 / Quel>1 / Quel:1
      | \bप्र(?:श्न)?\.?\s*[:\.\-]?\s*(\d{1,3})\b              # प्रश्न 5 / प्र.5
      | ^\s*\((\d{1,3})\)\s*[:\.\-]?                           # (3): or (3)
      | ^\s*(\d{1,3})\s*[\.)]\s+                               # 3. or 3) at line start
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def _extract_question_number(question_label: str, question_text: str = "") -> int | None:
    """Parse the printed question number out of the verbatim label (or fallback text).

    Accepted forms: Q1 / Q.1 / Q:1 / Que1 / Que:1 / Que.1 / Quel:1 / Quel>1 /
    प्रश्न 1 / प्र.1 / (1) / "1." at line start. Returns ``None`` when no
    recognisable label is present so the merger can decide whether to treat the
    record as a continuation or as the first item.
    """
    for src in (question_label or "", question_text or ""):
        m = _QUESTION_NUM_RE.search(src)
        if not m:
            continue
        for g in m.groups():
            if g is not None:
                try:
                    n = int(g, 10)
                    return n if 1 <= n <= 500 else None
                except (TypeError, ValueError):
                    return None
    return None


def _normalize_answer_type(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in ANSWER_TYPES:
        return s
    if s in ("short", "line"):
        return "correction"
    if s in ("wordlist", "list", "pairs", "tabular"):
        return "word_list"
    if s in ("prose", "long"):
        return "paragraph"
    return "paragraph"


def merge_per_page_questions(
    per_page: list[dict[str, Any]],
    total_pages: int,
    *,
    section_name: str = "General",
) -> list[dict[str, Any]]:
    """Stitch per-page question records into global items[].

    ``per_page[i]`` is the step-2 output for page ``i+1``::

        {"text": str, "is_intro": bool, "questions": [{
            "question_label", "question_text", "student_answer",
            "start_y_position_percent", "end_y_position_percent",
            "answer_type", "is_continuation",
        }, ...]}

    Merger policy for missing labels (Gemini sometimes drops or mangles the
    printed ``Que:N`` label):

      * If a record has no parseable question number AND there is a prior
        ``last_seen_qid``, treat it as a continuation of that question — this
        prevents sub-section headings like "Today's society :-" or "Conclusion"
        from being mistaken for new questions.
      * If a record has no parseable number AND no prior ``last_seen_qid`` (it's
        the very first item in the booklet), assume it's question 1 — booklets
        always start at Q1, so the first unlabelled item maps there.
    """
    items_by_qid: dict[Any, dict[str, Any]] = {}
    last_seen_qid: Any = None
    ordered_keys: list[Any] = []
    gap_counter = 1
    first_item_seen = False

    for page_idx, page_data in enumerate(per_page):
        page_number = page_idx + 1
        for q in page_data.get("questions", []) or []:
            label = (q.get("question_label") or "").strip()
            qtext_page = (q.get("question_text") or "").strip()
            stu_partial = (q.get("student_answer") or "").strip()
            sy = clamp_pct(q.get("start_y_position_percent", 0.0))
            ey = clamp_pct(q.get("end_y_position_percent", 100.0))
            atype = _normalize_answer_type(q.get("answer_type"))
            is_cont = bool(q.get("is_continuation"))

            qnum = _extract_question_number(label, qtext_page)

            if qnum is not None:
                # Numbered question — definitive new slot (or merge if already seen).
                key: Any = qnum
                if key not in items_by_qid:
                    items_by_qid[key] = _new_item_slot(
                        question_id=qnum, question_text=qtext_page,
                        section_name=section_name, answer_type=atype,
                        start_page=page_number, start_y=sy,
                    )
                    ordered_keys.append(key)
            elif is_cont or (first_item_seen and last_seen_qid is not None):
                # Unlabelled record after a labelled one → continuation. This catches
                # both explicit is_continuation=true and the common case where
                # Gemini incorrectly emits is_continuation=false for what is
                # actually a sub-section heading or a continuation paragraph.
                key = last_seen_qid
                if key is None or key not in items_by_qid:
                    key = f"_gap_{gap_counter}"
                    gap_counter += 1
                    items_by_qid[key] = _new_item_slot(
                        question_id=None, question_text="",
                        section_name=section_name, answer_type=atype,
                        start_page=page_number, start_y=sy,
                    )
                    ordered_keys.append(key)
            else:
                # First record in the booklet AND no label parsed.
                # Booklets always start at Q1, so assume this is Q1.
                key = 1
                items_by_qid[key] = _new_item_slot(
                    question_id=1, question_text=qtext_page,
                    section_name=section_name, answer_type=atype,
                    start_page=page_number, start_y=sy,
                )
                ordered_keys.append(key)

            slot = items_by_qid[key]
            if stu_partial:
                if slot["student_answer"]:
                    slot["student_answer"] = slot["student_answer"] + "\n\n" + stu_partial
                else:
                    slot["student_answer"] = stu_partial
            if not slot["question"] and qtext_page:
                slot["question"] = qtext_page
            slot["end_page"] = page_number
            slot["end_y_position_percent"] = ey
            if slot["answer_type"] == "paragraph" and atype != "paragraph":
                slot["answer_type"] = atype
            last_seen_qid = key
            first_item_seen = True

    items = list(items_by_qid.values())
    for it in items:
        it["start_page"] = page_num(it.get("start_page"), total_pages, 1)
        it["end_page"] = page_num(it.get("end_page"), total_pages, it["start_page"])
        if it["end_page"] < it["start_page"]:
            it["end_page"] = it["start_page"]
        it["is_attempted"] = bool(it["student_answer"])
        # marking position lives on the start page, right-margin centerline.
        it["marking_page"] = it["start_page"]
        it["marking_x_position_percent"] = 85.0
        it["marking_y_position_percent"] = it["start_y_position_percent"]

    items.sort(key=_sort_key)

    # Final question_id assignment: stable ints for items with parsed numbers,
    # gap-filled sequentially for the rest (after the parsed max).
    used_ids: set[int] = {
        i["question_id"] for i in items
        if isinstance(i.get("question_id"), int)
    }
    next_gap = max(used_ids, default=0) + 1
    for it in items:
        if not isinstance(it.get("question_id"), int):
            while next_gap in used_ids:
                next_gap += 1
            it["question_id"] = next_gap
            used_ids.add(next_gap)
            next_gap += 1

    return items


def _new_item_slot(
    *, question_id: int | None, question_text: str, section_name: str,
    answer_type: str, start_page: int, start_y: float,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question_text,
        "student_answer": "",
        "is_attempted": False,
        "section_name": section_name,
        "answer_type": answer_type,
        "start_page": start_page,
        "start_y_position_percent": start_y,
        "end_page": start_page,
        "end_y_position_percent": start_y,
        "marking_page": start_page,
        "marking_x_position_percent": 85.0,
        "marking_y_position_percent": start_y,
    }


def _sort_key(item: dict[str, Any]) -> tuple[int | float, ...]:
    sp = int(item.get("start_page", 1))
    sy = float(item.get("start_y_position_percent", 0.0))
    qi = item.get("question_id")
    if qi is None:
        return (sp, sy, 999999)
    try:
        return (sp, sy, int(qi))
    except (TypeError, ValueError):
        return (sp, sy, 999998)
