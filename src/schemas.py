from pydantic import BaseModel, ConfigDict, Field


class QuestionPayload(BaseModel):
    questionNo: str
    title: str
    desc: str
    instruction_name: str = ""
    pageNum: int = Field(ge=1)
    marks: int = Field(ge=0)
    diagramDescriptions: list[str] = Field(default_factory=list)


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
    """Client sends snake_case ``intro_page``. ``items`` may be omitted when only updating intro page."""

    modelKey: str = Field(min_length=1)
    items: list[QuestionPageMarksItem] = Field(default_factory=list)
    intro_page: int | None = Field(default=None, ge=1)


class CachedOcrRequest(BaseModel):
    """JSON body for ``POST /analyse/cached-ocr`` — ids + cached text; scheme loaded from DB."""

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(min_length=1, alias="modelId")
    question_id: str = Field(min_length=1, alias="questionId")
    cached_student_text: str = Field(min_length=1, alias="cachedStudentText")
    check_level: str = Field(default="Moderate", alias="checkLevel")


class CombinedReviewCompactItem(BaseModel):
    """Per-question grading summary; ``questionNo`` / ``title`` / ``marksTotal`` filled server-side."""

    model_config = ConfigDict(populate_by_name=True)

    question_id: str = Field(min_length=1, alias="questionId")
    marks_awarded: float = Field(alias="marksAwarded")
    good_points: str = Field(alias="goodPoints")
    improvements: str
    final_review: str = Field(alias="finalReview")


class CombinedReviewRequest(BaseModel):
    """JSON body for ``POST /analyse/combined-review`` — compact rows merged with stored questions."""

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(min_length=1, alias="modelId")
    question_results: list[CombinedReviewCompactItem] = Field(
        min_length=1, alias="questionResults"
    )


class IntroPageJsonRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    page_image_base64: str = Field(min_length=1, alias="pageImageBase64")


class ExpandModelAnswerRequest(BaseModel):
    type: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = "en"
