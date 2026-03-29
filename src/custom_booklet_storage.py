"""Map custom/essay QnA pipeline rows to the same question dict shape as ``process_pdf_path``."""

from __future__ import annotations


def custom_qna_rows_to_canonical_questions(
    filled: list[dict[str, str]],
) -> list[dict]:
    """Convert ``fill_answers_for_questions`` output into standard booklet question objects.

    Each input row uses ``question_id``, ``question_no``, ``question``, ``answer``.
    Output matches ``gemini_extract.normalize`` keys for API/DB parity with ``type=standard``.
    """
    out: list[dict] = []
    for i, row in enumerate(filled):
        qid = str(row.get("question_id") or f"q-{i + 1:03d}")
        qno = str(row.get("question_no") or f"Q{i + 1}")
        out.append(
            {
                "id": qid,
                "questionNo": qno,
                "title": str(row.get("question", "")),
                "desc": str(row.get("answer", "")),
                "pageNum": 1,
                "marks": 0,
                "diagramDescriptions": [],
            }
        )
    return out
