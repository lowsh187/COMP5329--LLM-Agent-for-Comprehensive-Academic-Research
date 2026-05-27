from typing import Any, Literal

from pydantic import BaseModel, Field

ExperimentCondition = Literal[
    "baseline_with_retrieval",
    "multi_stage_with_retrieval",
    "multi_stage_without_retrieval",
    "multi_stage_group_selection",
    "main_comparison",
]
IndependentExperimentCondition = Literal[
    "baseline_with_retrieval",
    "multi_stage_with_retrieval",
    "multi_stage_without_retrieval",
    "multi_stage_group_selection",
]


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    max_papers: int = Field(default=5, ge=1, le=10)
    condition: ExperimentCondition = "multi_stage_with_retrieval"
    experiment_id: str | None = None
    run_index: int | None = Field(default=None, ge=1)


class Paper(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    published: str | None = None
    url: str
    source: str = "Unknown"


class LiteratureTheme(BaseModel):
    theme: str
    paper_indices: list[int]
    summary: str
    limitations: str = ""
    gap_relevance: str = ""


class EvidenceNote(BaseModel):
    paper_id: str
    title: str
    problem: str = ""
    method: str = ""
    evidence: str = ""
    limitation: str = ""
    gap_relevance: str = ""


class RubricScore(BaseModel):
    criterion: str
    weight: int
    score: float = Field(..., ge=0)
    level: str = ""
    justification: str = ""


class Evaluation(BaseModel):
    clarity: int = Field(..., ge=1, le=5)
    logic: int = Field(..., ge=1, le=5)
    novelty: int = Field(..., ge=1, le=5)
    feasibility: int = Field(..., ge=1, le=5)
    literature_alignment: int = Field(..., ge=1, le=5)
    phd_level_quality: int | None = Field(default=None, ge=1, le=5)
    presentation: int | None = Field(default=None, ge=1, le=5)
    weighted_total: float | None = Field(default=None, ge=0, le=100)
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    comments: str = ""


class BaselineResult(BaseModel):
    proposal: str | None = None
    evaluation: Evaluation | None = None


class RunMetrics(BaseModel):
    condition: ExperimentCondition
    model: str
    temperature: float
    retrieval_query: str | None = None
    retrieval_snapshot_id: str | None = None
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    json_valid: bool = True
    required_field_completeness: float = Field(default=1.0, ge=0, le=1)
    failed_requests: int = 0
    retry_count: int = 0
    fallback_reason: str | None = None


class ResearchResult(BaseModel):
    experiment_id: str | None = None
    run_index: int | None = None
    topic: str
    condition: ExperimentCondition = "multi_stage_with_retrieval"
    retrieval_query: str | None = None
    retrieval_snapshot_id: str | None = None
    papers: list[Paper]
    literature_themes: list[LiteratureTheme] = Field(default_factory=list)
    evidence_notes: list[EvidenceNote] = Field(default_factory=list)
    summary: str | None = None
    scholarly_positioning: str | None = None
    domain_router: str | None = None
    research_scope: str | None = None
    concept_definition: str | None = None
    technical_feasibility: str | None = None
    gap: str | None = None
    question: str | None = None
    theoretical_mechanism: str | None = None
    hypothesis: str | None = None
    operationalization_causal_check: str | None = None
    methodology: str | None = None
    novelty_contribution: str | None = None
    failure_analysis: str | None = None
    proposal_logic_graph: dict[str, Any] | None = None
    proposal: str
    draft_proposal: str | None = None
    proposal_critique: dict[str, Any] | None = None
    evaluation: Evaluation
    baseline: BaselineResult | None = None
    group_selection: dict[str, Any] | None = None
    embedding_analysis: dict[str, Any] | None = None
    metrics: RunMetrics | None = None


class ExperimentRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    max_papers: int = Field(default=5, ge=1, le=10)
    conditions: list[IndependentExperimentCondition] = Field(
        default_factory=lambda: ["baseline_with_retrieval", "multi_stage_with_retrieval"],
        min_length=1,
        max_length=4,
    )
    repeats: int = Field(default=1, ge=1, le=5)


class ExperimentResult(BaseModel):
    experiment_id: str
    topic: str
    retrieval_query: str
    max_papers: int
    conditions: list[IndependentExperimentCondition]
    repeats: int
    runs: list[ResearchResult]
    quality_comparison: list[dict[str, Any]]
    stability_cost: list[dict[str, Any]]
