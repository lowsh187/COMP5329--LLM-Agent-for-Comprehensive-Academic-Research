import { demoResult } from "./mockData";
import type { ExperimentRequest, ExperimentResult, ResearchRequest, ResearchResult } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function runResearch(request: ResearchRequest, demoMode: boolean): Promise<ResearchResult> {
  if (demoMode) {
    await delay(500);
    const isBaseline = request.condition === "baseline_with_retrieval";
    return {
      ...demoResult,
      topic: request.topic,
      retrieval_query: request.topic,
      condition: request.condition,
      summary: isBaseline ? null : demoResult.summary,
      gap: isBaseline ? null : demoResult.gap,
      question: isBaseline ? null : demoResult.question,
      hypothesis: isBaseline ? null : demoResult.hypothesis,
      methodology: isBaseline ? null : demoResult.methodology,
      proposal: isBaseline ? demoResult.baseline?.proposal || demoResult.proposal : demoResult.proposal,
      evaluation:
        isBaseline && demoResult.baseline?.evaluation
          ? { ...demoResult.evaluation, ...demoResult.baseline.evaluation }
          : demoResult.evaluation,
      baseline: request.condition === "main_comparison" ? demoResult.baseline : undefined,
      metrics: demoResult.metrics
        ? {
            ...demoResult.metrics,
            condition: request.condition,
            retrieval_query: request.topic
          }
        : null,
      group_selection: request.condition === "multi_stage_group_selection" ? demoGroupSelection() : null
    };
  }

  const response = await postResearch(request);

  if (response.ok) {
    return response.json();
  }

  if (response.status === 422 && needsLegacyConditionFallback(request.condition)) {
    const fallbackResponse = await postResearch({ ...request, condition: "main_comparison" });
    if (!fallbackResponse.ok) {
      const message = await fallbackResponse.text();
      throw new Error(message || "Backend request failed");
    }
    const fallbackResult = (await fallbackResponse.json()) as ResearchResult;
    return adaptLegacyResearchResult(fallbackResult, request.condition);
  }

  const message = await response.text();
  throw new Error(message || "Backend request failed");
}

async function postResearch(request: ResearchRequest): Promise<Response> {
  return fetch(`${API_BASE_URL}/api/research/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request)
  });
}

export async function runExperiment(request: ExperimentRequest, demoMode: boolean): Promise<ExperimentResult> {
  if (demoMode) {
    await delay(700);
    const runs = request.conditions.flatMap((condition) =>
      Array.from({ length: request.repeats }, (_, index) => ({
        ...demoResult,
        experiment_id: "demo-experiment",
        run_index: index + 1,
        topic: request.topic,
        retrieval_query: request.topic,
        condition,
        summary: condition === "baseline_with_retrieval" ? null : demoResult.summary,
        gap: condition === "baseline_with_retrieval" ? null : demoResult.gap,
        question: condition === "baseline_with_retrieval" ? null : demoResult.question,
        hypothesis: condition === "baseline_with_retrieval" ? null : demoResult.hypothesis,
        methodology: condition === "baseline_with_retrieval" ? null : demoResult.methodology,
        proposal:
          condition === "baseline_with_retrieval"
            ? demoResult.baseline?.proposal || demoResult.proposal
            : demoResult.proposal,
        evaluation:
          condition === "baseline_with_retrieval" && demoResult.baseline?.evaluation
            ? { ...demoResult.evaluation, ...demoResult.baseline.evaluation }
            : demoResult.evaluation,
        baseline: undefined,
        metrics: demoResult.metrics
          ? {
              ...demoResult.metrics,
              condition,
              retrieval_query: request.topic,
              latency_ms:
                condition === "baseline_with_retrieval"
                  ? 1600
                  : condition === "multi_stage_group_selection"
                    ? 9800
                    : 5200,
              total_tokens:
                condition === "baseline_with_retrieval"
                  ? 1400
                  : condition === "multi_stage_group_selection"
                    ? 7600
                    : 4300
            }
          : null,
        group_selection: condition === "multi_stage_group_selection" ? demoGroupSelection() : null
      }))
    );

    return {
      experiment_id: "demo-experiment",
      topic: request.topic,
      retrieval_query: request.topic,
      max_papers: request.max_papers,
      conditions: request.conditions,
      repeats: request.repeats,
      runs,
      quality_comparison: buildDemoQuality(runs),
      stability_cost: buildDemoStability(runs)
    };
  }

  const response = await fetch(`${API_BASE_URL}/api/experiment/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Experiment request failed");
  }

  return response.json();
}

function buildDemoQuality(runs: ResearchResult[]): Array<Record<string, string | number | null>> {
  const metrics = ["clarity", "logic", "novelty", "feasibility", "literature_alignment"] as const;
  return metrics.map((metric) => {
    const row: Record<string, string | number | null> = { metric };
    for (const run of runs) {
      row[run.condition || "unknown"] = run.evaluation[metric];
    }
    const baseline = row.baseline_with_retrieval;
    const multi = row.multi_stage_with_retrieval;
    const group = row.multi_stage_group_selection;
    row.multi_stage_minus_baseline =
      typeof baseline === "number" && typeof multi === "number" ? multi - baseline : null;
    row.group_selection_minus_multi_stage =
      typeof group === "number" && typeof multi === "number" ? group - multi : null;
    return row;
  });
}

function buildDemoStability(runs: ResearchResult[]): Array<Record<string, string | number | null>> {
  const conditions = Array.from(new Set(runs.map((run) => run.condition).filter(Boolean))) as string[];
  return ["run_count", "average_latency_ms", "average_total_tokens", "average_paper_count"].map((metric) => {
    const row: Record<string, string | number | null> = { metric };
    for (const condition of conditions) {
      const conditionRuns = runs.filter((run) => run.condition === condition);
      if (metric === "run_count") row[condition] = conditionRuns.length;
      if (metric === "average_latency_ms") row[condition] = average(conditionRuns.map((run) => run.metrics?.latency_ms));
      if (metric === "average_total_tokens") row[condition] = average(conditionRuns.map((run) => run.metrics?.total_tokens));
      if (metric === "average_paper_count") row[condition] = average(conditionRuns.map((run) => run.papers.length));
    }
    return row;
  });
}

function demoGroupSelection() {
  return {
    gap: {
      selected_id: "gap_2",
      selected_text:
        "Existing feedback systems rarely compare literature-grounded feedback with generic LLM feedback across debugging outcomes.",
      criteria: ["specificity", "literature_grounding", "novelty", "feasibility"],
      candidates: [
        { id: "gap_1", text: "LLM feedback quality varies across contexts.", scores: { specificity: 2 }, total: 9, reason: "Too broad." },
        {
          id: "gap_2",
          text:
            "Existing feedback systems rarely compare literature-grounded feedback with generic LLM feedback across debugging outcomes.",
          scores: { specificity: 5, literature_grounding: 4, novelty: 4, feasibility: 5 },
          total: 18,
          reason: "Most specific and directly testable."
        },
        { id: "gap_3", text: "Students may need better explanations.", scores: { specificity: 3 }, total: 11, reason: "Feasible but less grounded." }
      ]
    },
    hypothesis: {
      selected_id: "hypothesis_1",
      selected_text:
        "Students receiving literature-grounded feedback will improve debugging accuracy more than students receiving generic LLM feedback.",
      criteria: ["testability", "alignment_with_gap", "clarity", "methodological_implication"],
      candidates: []
    },
    methodology: {
      selected_id: "methodology_3",
      selected_text:
        "Run a controlled study comparing generic and literature-grounded feedback, measuring debugging accuracy, explanation quality, and perceived usefulness.",
      criteria: ["feasibility", "baseline_design", "evaluation_metrics", "scope_control", "alignment_with_hypothesis"],
      candidates: []
    }
  };
}

function average(values: Array<number | null | undefined>): number | null {
  const realValues = values.filter((value): value is number => typeof value === "number");
  if (!realValues.length) return null;
  return Math.round((realValues.reduce((sum, value) => sum + value, 0) / realValues.length) * 1000) / 1000;
}

function needsLegacyConditionFallback(condition: ResearchRequest["condition"]): boolean {
  return condition === "baseline_with_retrieval" || condition === "multi_stage_with_retrieval";
}

function adaptLegacyResearchResult(result: ResearchResult, requestedCondition: ResearchRequest["condition"]): ResearchResult {
  if (requestedCondition === "baseline_with_retrieval" && result.baseline) {
    return {
      ...result,
      condition: "baseline_with_retrieval",
      summary: null,
      gap: null,
      question: null,
      hypothesis: null,
      methodology: null,
      proposal: result.baseline.proposal || result.proposal,
      evaluation: result.baseline.evaluation
        ? { ...result.evaluation, ...result.baseline.evaluation }
        : result.evaluation,
      baseline: undefined,
      metrics: result.metrics
        ? {
            ...result.metrics,
            condition: "baseline_with_retrieval"
          }
        : result.metrics
    };
  }

  return {
    ...result,
    condition: requestedCondition,
    baseline: undefined,
    metrics: result.metrics
      ? {
          ...result.metrics,
          condition: requestedCondition
        }
      : result.metrics
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
