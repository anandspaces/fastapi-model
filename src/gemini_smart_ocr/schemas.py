"""Gemini ``response_schema`` definitions for the three smart-OCR stages."""

from __future__ import annotations

from google.genai import types

# --- Stage 1A: classify + annotate ---------------------------------------------------

ANCHOR_MARK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "type": types.Schema(type=types.Type.STRING, description="One of: ellipse, underline, tick"),
        "cx": types.Schema(type=types.Type.NUMBER, description="Grid x center (1-50)"),
        "cy": types.Schema(type=types.Type.NUMBER, description="Grid y center (1-50)"),
        "rx": types.Schema(
            type=types.Type.NUMBER,
            description="x-radius in grid units; for underline = half horizontal span; for tick = 0",
        ),
        "ry": types.Schema(
            type=types.Type.NUMBER,
            description="y-radius; for underline = 0; for tick = 0",
        ),
    },
    required=["type", "cx", "cy", "rx", "ry"],
    property_ordering=["type", "cx", "cy", "rx", "ry"],
)

REMARK_BOX_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "x1": types.Schema(type=types.Type.NUMBER, description="Top-left grid x (1-50)"),
        "y1": types.Schema(type=types.Type.NUMBER, description="Top-left grid y (1-50)"),
        "x2": types.Schema(type=types.Type.NUMBER, description="Bottom-right grid x (1-50)"),
        "y2": types.Schema(type=types.Type.NUMBER, description="Bottom-right grid y (1-50)"),
        "comment": types.Schema(type=types.Type.STRING, description="Teacher comment text; empty string if none"),
    },
    required=["x1", "y1", "x2", "y2", "comment"],
    property_ordering=["x1", "y1", "x2", "y2", "comment"],
)

LAYOUT_ZONE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "y1": types.Schema(type=types.Type.INTEGER, description="First grid row of handwriting band (6-45)"),
        "y2": types.Schema(type=types.Type.INTEGER, description="Last grid row of handwriting band (6-45)"),
    },
    required=["y1", "y2"],
    property_ordering=["y1", "y2"],
)

CONTENT_LINE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "y":  types.Schema(type=types.Type.INTEGER, description="Baseline grid row of this handwriting line (6-45)"),
        "x1": types.Schema(type=types.Type.INTEGER, description="Leftmost grid column with ink on this line (1-37)"),
        "x2": types.Schema(type=types.Type.INTEGER, description="Rightmost grid column with ink on this line (1-37)"),
    },
    required=["y", "x1", "x2"],
    property_ordering=["y", "x1", "x2"],
)

CLASSIFY_ANNOTATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "page_type": types.Schema(
            type=types.Type.STRING,
            description=(
                "Exactly one of: DUPLICATE, CORRECTION, PARAGRAPH, WORD_LIST, UNKNOWN. "
                "DUPLICATE = visually identical layout to another sheet in this booklet. "
                "CORRECTION = short line/sentence fixes, numbered one-line corrections. "
                "PARAGRAPH = long prose answers, अर्थ/प्रयोग style blocks. "
                "WORD_LIST = word pairs, विलोम, उपसर्ग/प्रत्यय tables. "
                "UNKNOWN = none of the above clearly fits."
            ),
        ),
        "content_bands": types.Schema(
            type=types.Type.ARRAY,
            items=LAYOUT_ZONE_SCHEMA,
            description=(
                "Contiguous y-row bands (rows 6–45 only) that contain student handwriting. "
                "Top rows 1-5 and bottom rows 46-50 are always margin — never include them. "
                "Right margin columns 38-50 are always structurally free and are NOT content."
            ),
        ),
        "content_lines": types.Schema(
            type=types.Type.ARRAY,
            items=CONTENT_LINE_SCHEMA,
            description=(
                "Every individual handwriting line on the page. "
                "y = baseline row of the line, x1/x2 = leftmost/rightmost column with ink. "
                "List ALL lines top-to-bottom — one entry per written line."
            ),
        ),
        "anchor_marks": types.Schema(
            type=types.Type.ARRAY,
            items=ANCHOR_MARK_SCHEMA,
            description="4-7 SUGGESTED teacher markup positions on student handwriting.",
        ),
        "remarks": types.Schema(
            type=types.Type.ARRAY,
            items=REMARK_BOX_SCHEMA,
            description=(
                "3-5 free-whitespace bounding boxes in FREE ZONES only (see content_bands). "
                "Spread them across the full vertical range of the page — early, middle, and late y positions."
            ),
        ),
    },
    required=["page_type", "content_bands", "anchor_marks", "remarks"],
    property_ordering=["page_type", "content_bands", "content_lines", "anchor_marks", "remarks"],
)

# --- Stage 1B: per-page OCR ----------------------------------------------------------

OCR_PAGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"text": types.Schema(type=types.Type.STRING)},
    required=["text"],
    property_ordering=["text"],
)

# --- Stage 2: structure (sections → questions) ---------------------------------------

QUESTION_BLOCK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "question_id": types.Schema(type=types.Type.INTEGER),
        "question": types.Schema(type=types.Type.STRING),
        "student_answer": types.Schema(type=types.Type.STRING),
        "answer_type": types.Schema(
            type=types.Type.STRING,
            description="One of: correction, paragraph, word_list",
        ),
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
        "answer_type",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
    property_ordering=[
        "question_id",
        "question",
        "student_answer",
        "answer_type",
        "start_page",
        "start_y_position_percent",
        "end_page",
        "end_y_position_percent",
        "marking_page",
        "marking_x_position_percent",
        "marking_y_position_percent",
    ],
)

SECTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "section_name": types.Schema(type=types.Type.STRING),
        "questions": types.Schema(type=types.Type.ARRAY, items=QUESTION_BLOCK_SCHEMA),
    },
    required=["section_name", "questions"],
    property_ordering=["section_name", "questions"],
)

STRUCTURE_ROOT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sections": types.Schema(type=types.Type.ARRAY, items=SECTION_SCHEMA),
    },
    required=["sections"],
    property_ordering=["sections"],
)
