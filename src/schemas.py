from pydantic import BaseModel, Field


class QuestionPayload(BaseModel):
    questionNo: str
    title: str
    desc: str
    pageNum: int = Field(ge=1)
    marks: int = Field(ge=0)
    diagramDescriptions: list[str]


class AuthRequest(BaseModel):
    username: str
    password: str


class TokenData(BaseModel):
    accessToken: str
    tokenType: str = "Bearer"
    expiresIn: int


class ReorderQuestionsPayload(BaseModel):
    order: list[str]


class QuestionPageMarksItem(BaseModel):
    """CamelCase field names match JSON for other question-shaped payloads (e.g. QuestionPayload)."""

    questionId: str = Field(min_length=1)
    pageNum: int = Field(ge=1)
    marks: int = Field(ge=0)


class BulkQuestionPageMarksPayload(BaseModel):
    modelKey: str = Field(min_length=1)
    items: list[QuestionPageMarksItem]


class CachedOcrRequest(BaseModel):
    cached_student_text: str = Field(min_length=1)
    question_title: str
    model_description: str
    total_marks: int = Field(ge=1)
    page_count: int = Field(ge=1)
    language: str


class QuestionResultItem(BaseModel):
    question_no: str
    title: str
    marks_awarded: float
    marks_total: int
    good_points: str
    improvements: str
    final_review: str


class CombinedReviewRequest(BaseModel):
    question_results: list[QuestionResultItem] = Field(min_length=1)


class ExpandModelAnswerRequest(BaseModel):
    type: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = "en"
