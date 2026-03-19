from pydantic import BaseModel, Field

class QuestionPayload(BaseModel):
    questionNo: str
    title: str
    desc: str
    pageNum: int = Field(ge=1)
    marks: int = Field(ge=0)
    diagramDescriptions: list[str]
