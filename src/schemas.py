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


class FullAnalysisRequest(BaseModel):
    """JSON body for ``POST /analyse/full`` (Flutter-aligned field names)."""

    model_config = ConfigDict(populate_by_name=True)

    page_images_base64: list[str] = Field(min_length=1, alias="pageImagesBase64")
    question_title: str = Field(alias="questionTitle")
    instruction_name: str | None = Field(default=None, alias="instructionName")
    model_description: str = Field(alias="modelDescription")
    total_marks: int = Field(ge=1, alias="totalMarks")
    language: str
    check_level: str = Field(default="Moderate", alias="checkLevel")


class CachedOcrRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cached_student_text: str = Field(min_length=1, alias="cachedStudentText")
    question_title: str = Field(alias="questionTitle")
    instruction_name: str | None = Field(default=None, alias="instructionName")
    model_description: str = Field(alias="modelDescription")
    total_marks: int = Field(ge=1, alias="totalMarks")
    page_count: int = Field(ge=1, alias="pageCount")
    language: str
    check_level: str = Field(default="Moderate", alias="checkLevel")


class QuestionResultItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question_no: str = Field(alias="questionNo")
    title: str
    marks_awarded: float = Field(alias="marksAwarded")
    marks_total: int = Field(alias="marksTotal")
    good_points: str = Field(alias="goodPoints")
    improvements: str
    final_review: str = Field(alias="finalReview")


class CombinedReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question_results: list[QuestionResultItem] = Field(min_length=1, alias="questionResults")


class IntroPageJsonRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    page_image_base64: str = Field(min_length=1, alias="pageImageBase64")


class ExpandModelAnswerRequest(BaseModel):
    type: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = "en"
