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
    prompt_profile: str = "default"
    pipeline_mode: str = "full"
    attention_guidance: str | None = None
    target_revision: bool = False
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
    literature_tension_graph: str | None = None
    scholarly_positioning: str | None = None
    domain_router: str | None = None
    research_scope: str | None = None
    concept_definition: str | None = None
    technical_feasibility: str | None = None
    gap: str | None = None
    question: str | None = None
    research_problem_validity_check: str | None = None
    research_problem_blueprint: dict[str, Any] | None = None
    blueprint_compliance_check: dict[str, Any] | None = None
    theoretical_mechanism: str | None = None
    hypothesis: str | None = None
    operationalization_causal_check: str | None = None
    methodology: str | None = None
    novelty_contribution: str | None = None
    contribution_type_router: str | None = None
    phd_contribution_design: str | None = None
    pre_proposal_critique: str | None = None
    failure_analysis: str | None = None
    bad_proposal_pattern_detector: str | None = None
    evidence_claim_alignment: str | None = None
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


class AgentActionStats(BaseModel):
    count: int = Field(default=0, ge=0)
    mean_reward: float = 0.0
    total_reward: float = 0.0
    last_reward: float | None = None
    best_reward: float | None = None
    mean_quality: float = 0.0
    total_quality: float = 0.0
    last_quality: float | None = None
    best_quality: float = 0.0
    quality_improvement_count: int = Field(default=0, ge=0)
    mean_tokens: float = 0.0
    total_tokens: float = 0.0
    mean_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fallback_count: int = Field(default=0, ge=0)
    fallback_rate: float = 0.0


class AgentStrategy(BaseModel):
    strategy_id: str
    condition: IndependentExperimentCondition
    max_papers: int = Field(default=5, ge=0, le=10)
    description: str = ""
    prompt_profile: str = "default"
    pipeline_mode: str = "full"
    cost_profile: str = "balanced"
    target_revision: bool = False


class AgentPolicyState(BaseModel):
    algorithm: str = "epsilon_greedy_quality_max_bandit"
    epsilon: float = Field(default=0.2, ge=0, le=1)
    episodes: int = Field(default=0, ge=0)
    strategies: dict[str, AgentStrategy] = Field(default_factory=dict)
    actions: dict[str, AgentActionStats]
    attention_weights: dict[str, float] = Field(default_factory=dict)
    best_action: str | None = None
    updated_at: str | None = None
    warm_started_from_runs: int = Field(default=0, ge=0)


class AgentRewardBreakdown(BaseModel):
    quality_score: float
    quality_reward: float
    high_score_bonus: float = 0.0
    target_score_bonus: float = 0.0
    quality_record_bonus: float = 0.0
    near_target_bonus: float = 0.0
    target_gap_penalty: float = 0.0
    token_penalty: float = 0.0
    latency_penalty: float = 0.0
    completeness_bonus: float = 0.0
    fallback_penalty: float = 0.0
    reward: float


class AgentTrainingRequest(BaseModel):
    topics: list[str] = Field(..., min_length=1, max_length=50)
    episodes: int = Field(default=4, ge=1, le=100)
    max_papers: int = Field(default=5, ge=1, le=10)
    epsilon: float | None = Field(default=None, ge=0, le=1)
    warm_start: bool = True
    warm_start_max_runs: int | None = Field(default=60, ge=1, le=500)
    strategy_ids: list[str] | None = Field(default=None, min_length=1, max_length=20)


class AgentTrainingEpisode(BaseModel):
    episode: int
    topic: str
    selected_strategy: AgentStrategy
    selected_action: IndependentExperimentCondition
    generated_strategies: list[AgentStrategy] = Field(default_factory=list)
    weakness_tags: list[str] = Field(default_factory=list)
    attention_focus: dict[str, float] = Field(default_factory=dict)
    reward: AgentRewardBreakdown
    weighted_total: float | None = None
    latency_ms: int | None = None
    total_tokens: int | None = None
    fallback_reason: str | None = None


class AgentTrainingResult(BaseModel):
    training_id: str
    episodes: list[AgentTrainingEpisode]
    policy: AgentPolicyState
