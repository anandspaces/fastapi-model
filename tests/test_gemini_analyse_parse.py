"""Unit tests for ``gemini_analyse`` JSON parsing (no live Gemini calls)."""

from src import gemini_analyse as ga


def test_parse_analysis_json_camel_case() -> None:
    text = """
    {
      "studentText": "hello",
      "marksAwarded": 3.5,
      "confidencePercent": 80,
      "goodPoints": "• ok",
      "improvements": "• fix",
      "finalReview": "done",
      "annotations": [
        {
          "pageIndex": 2,
          "yPositionPercent": 40,
          "xStartPercent": 10,
          "xEndPercent": 90,
          "comment": "nice",
          "isPositive": false,
          "lineStyle": "zigzag"
        }
      ]
    }
    """
    out = ga._parse_analysis_json(text, total_marks=8, page_count=2)
    assert out["student_text"] == "hello"
    assert out["marks_awarded"] == 3.5
    assert out["confidence_percent"] == 80.0
    assert len(out["annotations"]) == 1
    ann = out["annotations"][0]
    assert ann["page_index"] == 1
    assert ann["is_positive"] is True
    assert ann["line_style"] == "straight"


def test_parse_combined_json_accepts_snake_or_camel() -> None:
    snake = '{"final_review":"long","overall_improvements":"a\\nb","one_thing_to_write":"practice"}'
    out_s = ga._parse_combined_json(snake)
    assert out_s["final_review"] == "long"
    assert out_s["overall_improvements"] == "a\nb"
    camel = '{"finalReview":"long","overallImprovements":"a\\nb","oneThingToWrite":"practice"}'
    out_c = ga._parse_combined_json(camel)
    assert out_c["final_review"] == "long"
    assert out_c["one_thing_to_write"] == "practice"


def test_parse_intro_lines_pipe_format() -> None:
    raw = "1|4|74|30\n2||74|33\n"
    cells = ga._parse_intro_lines(raw)
    assert len(cells) == 2
    assert cells[0] == {
        "question_no": 1,
        "marks_text": "4",
        "x_percent": 74.0,
        "y_percent": 30.0,
    }
    assert cells[1]["marks_text"] == ""
