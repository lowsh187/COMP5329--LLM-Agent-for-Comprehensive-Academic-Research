from datetime import datetime, timezone

from app.literature_clustering import build_embeddings, cosine
from app.literature_client import clean_retrieval_query, search_literature
from app.llm_client import LLMClient
from app.models import Evaluation, ExperimentRequest, ExperimentResult, ResearchRequest, ResearchResult
from app.pipeline import run_research_pipeline
from app.storage import save_retrieval_snapshot

QUALITY_METRICS = [
    "clarity",
    "logic",
    "novelty",
    "feasibility",
    "literature_alignment",
    "phd_level_quality",
    "presentation",
    "weighted_total",
]


async def run_single_topic_experiment(request: ExperimentRequest) -> ExperimentResult:
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retrieval_query = clean_retrieval_query(request.topic)
    papers = await search_literature(request.topic, request.max_papers)
    retrieval_snapshot_id = save_retrieval_snapshot(
        request.topic,
        request.max_papers,
        papers,
        retrieval_query,
    )
    print(f"[experiment] shared retrieval snapshot: {retrieval_snapshot_id}", flush=True)
    runs: list[ResearchResult] = []

    for condition in request.conditions:
        for run_index in range(1, request.repeats + 1):
            print(
                f"[experiment] {experiment_id} condition={condition} run={run_index}/{request.repeats}",
                flush=True,
            )
            result = await run_research_pipeline(
                ResearchRequest(
                    topic=request.topic,
                    max_papers=request.max_papers,
                    condition=condition,
                    experiment_id=experiment_id,
                    run_index=run_index,
                ),
                papers_override=papers,
                retrieval_snapshot_id_override=retrieval_snapshot_id,
            )
            runs.append(result)

    stability_cost = build_stability_cost(runs)
    stability_cost.extend(await build_embedding_stability(runs))

    return ExperimentResult(
        experiment_id=experiment_id,
        topic=request.topic,
        retrieval_query=retrieval_query,
        max_papers=request.max_papers,
        conditions=request.conditions,
        repeats=request.repeats,
        runs=runs,
        quality_comparison=build_quality_comparison(runs),
        stability_cost=stability_cost,
    )


def build_quality_comparison(runs: list[ResearchResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = unique_conditions(runs)

    for metric in QUALITY_METRICS:
        row: dict[str, object] = {"metric": metric}
        values_by_condition: dict[str, float | None] = {}
        for condition in conditions:
            values = [
                get_score(run.evaluation, metric)
                for run in runs
                if run.condition == condition and run.evaluation is not None
            ]
            values = [value for value in values if value is not None]
            mean_value = mean(values)
            row[condition] = mean_value
            values_by_condition[condition] = mean_value

        baseline = values_by_condition.get("baseline_with_retrieval")
        multi_stage = values_by_condition.get("multi_stage_with_retrieval")
        group_selection = values_by_condition.get("multi_stage_group_selection")
        row["multi_stage_minus_baseline"] = (
            round(multi_stage - baseline, 3)
            if baseline is not None and multi_stage is not None
            else None
        )
        row["group_selection_minus_multi_stage"] = (
            round(group_selection - multi_stage, 3)
            if group_selection is not None and multi_stage is not None
            else None
        )
        rows.append(row)

    return rows


def build_stability_cost(runs: list[ResearchResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = unique_conditions(runs)
    metric_builders = {
        "run_count": lambda condition_runs: len(condition_runs),
        "json_valid_rate": lambda condition_runs: mean(
            [1 if run.metrics and run.metrics.json_valid else 0 for run in condition_runs]
        ),
        "field_completeness": lambda condition_runs: mean(
            [
                run.metrics.required_field_completeness
                for run in condition_runs
                if run.metrics is not None
            ]
        ),
        "average_latency_ms": lambda condition_runs: mean(
            [run.metrics.latency_ms for run in condition_runs if run.metrics is not None]
        ),
        "average_total_tokens": lambda condition_runs: mean(
            [
                run.metrics.total_tokens
                for run in condition_runs
                if run.metrics is not None and run.metrics.total_tokens is not None
            ]
        ),
        "average_paper_count": lambda condition_runs: mean([len(run.papers) for run in condition_runs]),
    }

    for metric, builder in metric_builders.items():
        row: dict[str, object] = {"metric": metric}
        for condition in conditions:
            condition_runs = [run for run in runs if run.condition == condition]
            row[condition] = builder(condition_runs)
        rows.append(row)

    return rows


async def build_embedding_stability(runs: list[ResearchResult]) -> list[dict[str, object]]:
    row: dict[str, object] = {"metric": "proposal_embedding_similarity"}
    llm = LLMClient()
    for condition in unique_conditions(runs):
        proposals = [
            run.proposal
            for run in runs
            if run.condition == condition and run.proposal and not (run.metrics and run.metrics.fallback_reason)
        ]
        if len(proposals) < 2:
            row[condition] = None
            continue
        vectors = await build_embeddings(llm, proposals)
        similarities = [
            cosine(vectors[left], vectors[right])
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
        ]
        row[condition] = round(sum(similarities) / len(similarities), 4) if similarities else None
    return [row]


def unique_conditions(runs: list[ResearchResult]) -> list[str]:
    return list(dict.fromkeys(run.condition for run in runs))


def get_score(evaluation: Evaluation, metric: str) -> int | float | None:
    value = getattr(evaluation, metric, None)
    return value if isinstance(value, int | float) else None


def mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)
