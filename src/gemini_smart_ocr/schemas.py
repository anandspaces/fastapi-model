"""Gemini ``response_schema`` definitions for the smart-OCR steps."""

from __future__ import annotations

from google.genai import types

# --- Step 2: per-page OCR + Q&A extraction ------------------------------------------

_STEP2_QUESTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "question_label": types.Schema(
            type=types.Type.STRING,
            description=(
                "Verbatim printed question label as it appears on this page "
                '(e.g. "Q1", "Que:2", "Q.3", "(4):", "प्रश्न 5"). Empty string '
                "when this record is a continuation of a question that started on "
                "an earlier page (set is_continuation=true)."
            ),
        ),
        "question_text": types.Schema(
            type=types.Type.STRING,
            description=(
                "Full printed question text on this page (excluding the student's "
                "handwriting). Empty string for continuations."
            ),
        ),
        "student_answer": types.Schema(
            type=types.Type.STRING,
            description=(
                "Transcribed student handwriting for THIS question on THIS page only. "
                "Empty string if the student left the question blank."
            ),
        ),
        "start_y_position_percent": types.Schema(
            type=types.Type.NUMBER,
            description="Top of the question (or its continuation) on this page, 0–100 from page top.",
        ),
        "end_y_position_percent": types.Schema(
            type=types.Type.NUMBER,
            description="Bottom of the student handwriting for this question on this page, 0–100.",
        ),
        "answer_type": types.Schema(
            type=types.Type.STRING,
            description="One of: correction, paragraph, word_list",
        ),
        "is_continuation": types.Schema(
            type=types.Type.BOOLEAN,
            description="True iff this record continues a question whose label appeared on a previous page.",
        ),
    },
    required=[
        "question_label",
        "question_text",
        "student_answer",
        "start_y_position_percent",
        "end_y_position_percent",
        "answer_type",
        "is_continuation",
    ],
    property_ordering=[
        "question_label",
        "question_text",
        "student_answer",
        "start_y_position_percent",
        "end_y_position_percent",
        "answer_type",
        "is_continuation",
    ],
)

STEP2_PAGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "text": types.Schema(
            type=types.Type.STRING,
            description="Full OCR transcript for this page, with \\n line breaks preserved.",
        ),
        "is_intro": types.Schema(
            type=types.Type.BOOLEAN,
            description=(
                "True iff this page is an intro/cover/instructions/roll-number/blank "
                "rough-work sheet with NO student handwriting. Only meaningful when "
                "the prompt told the model this is page 1."
            ),
        ),
        "questions": types.Schema(
            type=types.Type.ARRAY,
            items=_STEP2_QUESTION_SCHEMA,
            description="Question records found on THIS page (one entry per top-level question or continuation).",
        ),
    },
    required=["text", "is_intro", "questions"],
    property_ordering=["text", "is_intro", "questions"],
)


# --- Step 3: per-question-id grading + remarking + anchor remarking ------------------

_ANCHOR_MARK_GRID_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "type": types.Schema(
            type=types.Type.STRING,
            description="One of: ellipse, underline, tick",
        ),
        "page": types.Schema(
            type=types.Type.INTEGER,
            description="1-based page number the mark belongs to (must be in the allowed page list).",
        ),
        "cx": types.Schema(type=types.Type.NUMBER, description="Grid x center (1-50)"),
        "cy": types.Schema(type=types.Type.NUMBER, description="Grid y center (1-50)"),
        "rx": types.Schema(
            type=types.Type.NUMBER,
            description="x-radius in grid units; underline = half-width; tick = 0",
        ),
        "ry": types.Schema(
            type=types.Type.NUMBER,
            description="y-radius; underline = 0; tick = 0",
        ),
    },
    required=["type", "page", "cx", "cy", "rx", "ry"],
    property_ordering=["type", "page", "cx", "cy", "rx", "ry"],
)

_REMARK_BOX_GRID_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "page": types.Schema(
            type=types.Type.INTEGER,
            description="1-based page number the remark belongs to (must be in the allowed page list).",
        ),
        "x1": types.Schema(type=types.Type.NUMBER, description="Top-left grid x (1-50) — must lie in a free zone"),
        "y1": types.Schema(type=types.Type.NUMBER, description="Top-left grid y (1-50)"),
        "x2": types.Schema(type=types.Type.NUMBER, description="Bottom-right grid x (1-50)"),
        "y2": types.Schema(type=types.Type.NUMBER, description="Bottom-right grid y (1-50)"),
        "comment": types.Schema(
            type=types.Type.STRING,
            description=(
                "Teacher comment referring to the model answer — what is missing, wrong, "
                "or well done. Empty string only for purely visual underline-arrow boxes."
            ),
        ),
    },
    required=["page", "x1", "y1", "x2", "y2", "comment"],
    property_ordering=["page", "x1", "y1", "x2", "y2", "comment"],
)

_ANNOTATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "page_index": types.Schema(
            type=types.Type.INTEGER,
            description="0-based page index (page_number - 1).",
        ),
        "y_position_percent": types.Schema(type=types.Type.NUMBER),
        "x_start_percent": types.Schema(type=types.Type.NUMBER),
        "x_end_percent": types.Schema(type=types.Type.NUMBER),
        "comment": types.Schema(type=types.Type.STRING),
        "is_positive": types.Schema(type=types.Type.BOOLEAN),
    },
    required=["page_index", "y_position_percent", "x_start_percent", "x_end_percent", "comment", "is_positive"],
    property_ordering=["page_index", "y_position_percent", "x_start_percent", "x_end_percent", "comment", "is_positive"],
)

STEP3_ITEM_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "marks_awarded": types.Schema(type=types.Type.NUMBER),
        "max_marks": types.Schema(type=types.Type.NUMBER),
        "status": types.Schema(
            type=types.Type.STRING,
            description="One of: correct, partial, wrong, unattempted",
        ),
        "feedback": types.Schema(
            type=types.Type.STRING,
            description="Specific feedback grounded in the model answer (what is missing/wrong/correct).",
        ),
        "student_answer_summary": types.Schema(
            type=types.Type.STRING,
            description="1–3 sentence summary of what the student wrote.",
        ),
        "anchor_marks": types.Schema(
            type=types.Type.ARRAY,
            items=_ANCHOR_MARK_GRID_SCHEMA,
            description="4-10 markup spots on student handwriting across the item's pages.",
        ),
        "remarks": types.Schema(
            type=types.Type.ARRAY,
            items=_REMARK_BOX_GRID_SCHEMA,
            description="Free-zone bounding boxes (right margin / left margin / inline gap) with teacher comments.",
        ),
        "annotations": types.Schema(
            type=types.Type.ARRAY,
            items=_ANNOTATION_SCHEMA,
            description="Frontend mirror of remarks: 0-based page_index with y/x percent coords + comment + is_positive.",
        ),
    },
    required=[
        "marks_awarded",
        "max_marks",
        "status",
        "feedback",
        "student_answer_summary",
        "anchor_marks",
        "remarks",
        "annotations",
    ],
    property_ordering=[
        "marks_awarded",
        "max_marks",
        "status",
        "feedback",
        "student_answer_summary",
        "anchor_marks",
        "remarks",
        "annotations",
    ],
)
