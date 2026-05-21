from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    max_papers: int = Field(default=5, ge=1, le=10)


class Paper(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    published: str | None = None
    url: str


class Evaluation(BaseModel):
    clarity: int = Field(..., ge=1, le=5)
    logic: int = Field(..., ge=1, le=5)
    novelty: int = Field(..., ge=1, le=5)
    feasibility: int = Field(..., ge=1, le=5)
    literature_alignment: int = Field(..., ge=1, le=5)
    comments: str = ""


class BaselineResult(BaseModel):
    proposal: str | None = None
    evaluation: Evaluation | None = None


class ResearchResult(BaseModel):
    topic: str
    papers: list[Paper]
    summary: str
    gap: str
    question: str
    hypothesis: str
    methodology: str
    proposal: str
    evaluation: Evaluation
    baseline: BaselineResult | None = None
