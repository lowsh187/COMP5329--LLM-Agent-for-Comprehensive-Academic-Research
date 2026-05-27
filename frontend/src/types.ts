export type Paper = {
  title: string;
  authors: string[];
  abstract: string;
  published?: string | null;
  url: string;
  source?: string;
};

export type Evaluation = {
  clarity: number;
  logic: number;
  novelty: number;
  feasibility: number;
  literature_alignment: number;
  phd_level_quality?: number | null;
  presentation?: number | null;
  weighted_total?: number | null;
  rubric_scores?: RubricScore[];
  comments: string;
};

export type LiteratureTheme = {
  theme: string;
  paper_indices: number[];
  summary: string;
  limitations?: string;
  gap_relevance?: string;
};

export type EvidenceNote = {
  paper_id: string;
  title: string;
  problem?: string;
  method?: string;
  evidence?: string;
  limitation?: string;
  gap_relevance?: string;
};

export type RubricScore = {
  criterion: string;
  weight: number;
  score: number;
  level: string;
  justification: string;
};

export type ExperimentCondition =
  | "baseline_with_retrieval"
  | "multi_stage_with_retrieval"
  | "multi_stage_without_retrieval"
  | "multi_stage_group_selection"
  | "main_comparison";

export type IndependentExperimentCondition =
  | "baseline_with_retrieval"
  | "multi_stage_with_retrieval"
  | "multi_stage_without_retrieval"
  | "multi_stage_group_selection";

export type RunMetrics = {
  condition: ExperimentCondition;
  model: string;
  temperature: number;
  retrieval_query?: string | null;
  retrieval_snapshot_id?: string | null;
  latency_ms: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  json_valid: boolean;
  required_field_completeness: number;
  failed_requests: number;
  retry_count: number;
  fallback_reason?: string | null;
};

export type ResearchResult = {
  experiment_id?: string | null;
  run_index?: number | null;
  topic: string;
  condition?: ExperimentCondition;
  retrieval_query?: string | null;
  retrieval_snapshot_id?: string | null;
  papers: Paper[];
  literature_themes?: LiteratureTheme[];
  evidence_notes?: EvidenceNote[];
  summary?: string | null;
  scholarly_positioning?: string | null;
  domain_router?: string | null;
  research_scope?: string | null;
  concept_definition?: string | null;
  technical_feasibility?: string | null;
  gap?: string | null;
  question?: string | null;
  theoretical_mechanism?: string | null;
  hypothesis?: string | null;
  operationalization_causal_check?: string | null;
  methodology?: string | null;
  novelty_contribution?: string | null;
  failure_analysis?: string | null;
  proposal_logic_graph?: Record<string, unknown> | null;
  proposal: string;
  draft_proposal?: string | null;
  proposal_critique?: Record<string, unknown> | null;
  evaluation: Evaluation;
  baseline?: {
    proposal?: string;
    evaluation?: Partial<Evaluation>;
  };
  group_selection?: GroupSelectionRecord | null;
  embedding_analysis?: Record<string, unknown> | null;
  metrics?: RunMetrics | null;
};

export type GroupCandidate = {
  id?: string;
  text?: string;
  supporting_paper_ids?: string[];
  scores?: Record<string, number>;
  total?: number | null;
  reason?: string;
};

export type GroupSelectionStage = {
  candidates?: GroupCandidate[];
  selected_id?: string | null;
  selected_text?: string | null;
  criteria?: string[];
};

export type GroupSelectionRecord = {
  gap?: GroupSelectionStage;
  hypothesis?: GroupSelectionStage;
  methodology?: GroupSelectionStage;
};

export type ResearchRequest = {
  topic: string;
  max_papers: number;
  condition: ExperimentCondition;
};

export type ExperimentRequest = {
  topic: string;
  max_papers: number;
  conditions: IndependentExperimentCondition[];
  repeats: number;
};

export type ExperimentResult = {
  experiment_id: string;
  topic: string;
  retrieval_query: string;
  max_papers: number;
  conditions: IndependentExperimentCondition[];
  repeats: number;
  runs: ResearchResult[];
  quality_comparison: Array<Record<string, string | number | null>>;
  stability_cost: Array<Record<string, string | number | null>>;
};
