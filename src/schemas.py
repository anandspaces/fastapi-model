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
