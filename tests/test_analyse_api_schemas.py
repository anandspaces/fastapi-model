"""Pydantic schemas for optimized analyse endpoints."""

from src.schemas import CachedOcrRequest, CombinedReviewCompactItem, CombinedReviewRequest


def test_cached_ocr_request_accepts_camel_case() -> None:
    r = CachedOcrRequest.model_validate(
        {
            "modelId": "mid",
            "questionId": "q-eng-001",
            "cachedStudentText": "hello",
            "checkLevel": "Hard",
        }
    )
    assert r.model_id == "mid"
    assert r.question_id == "q-eng-001"
    assert r.cached_student_text == "hello"


def test_combined_review_compact_request() -> None:
    r = CombinedReviewRequest.model_validate(
        {
            "modelId": "uuid-here",
            "questionResults": [
                {
                    "questionId": "q-eng-001",
                    "marksAwarded": 5.5,
                    "goodPoints": "a",
                    "improvements": "b",
                    "finalReview": "c",
                }
            ],
        }
    )
    assert r.model_id == "uuid-here"
    assert len(r.question_results) == 1
    assert isinstance(r.question_results[0], CombinedReviewCompactItem)

