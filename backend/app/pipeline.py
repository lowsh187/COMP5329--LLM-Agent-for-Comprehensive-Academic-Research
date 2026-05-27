import json
from time import perf_counter
from typing import Any

from app.config import settings
from app.fallback import fallback_evaluation, fallback_stage_outputs
from app.literature_clustering import (
    add_candidate_embedding_alignment,
    analyze_text_against_papers,
    cluster_literature,
    format_embedding_evidence,
    format_literature_themes,
    rerank_papers_by_query,
)
from app.literature_client import clean_retrieval_query, search_literature
from app.llm_client import LLMClient
from app.models import BaselineResult, Evaluation, EvidenceNote, Paper, ResearchRequest, ResearchResult, RunMetrics
from app.prompts import format_papers_context, load_prompt
from app.rubric import build_evaluation_prompt, evaluation_from_rubric_response
from app.storage import save_retrieval_snapshot, save_run

SYSTEM_PROMPT = (
    "You are an academic research assistant. Always return valid JSON only. "
    "Keep outputs concise, specific, feasible, and aligned with the retrieved literature."
)


async def run_research_pipeline(
    request: ResearchRequest,
    papers_override: list[Paper] | None = None,
    retrieval_snapshot_id_override: str | None = None,
) -> ResearchResult:
    started_at = perf_counter()
    print(f"[pipeline] start topic={request.topic!r}", flush=True)
    print(f"[pipeline] condition={request.condition}", flush=True)
    print("[pipeline] searching literature sources", flush=True)
    retrieval_query = clean_retrieval_query(request.topic)
    if retrieval_query != request.topic.strip():
        print(f"[pipeline] retrieval query={retrieval_query!r}", flush=True)
    llm = LLMClient()
    embedding_analysis: dict[str, Any] = {}
    should_rerank = request.condition != "baseline_with_retrieval"
    if papers_override is not None:
        papers = papers_override
        retrieval_snapshot_id = retrieval_snapshot_id_override
        print(f"[pipeline] reusing retrieval snapshot: {retrieval_snapshot_id}", flush=True)
    else:
        papers = await search_literature(request.topic, request.max_papers)
        if should_rerank:
            papers, rerank_scores = await rerank_papers_by_query(llm, retrieval_query, papers)
            embedding_analysis["retrieval_reranking"] = rerank_scores
        else:
            embedding_analysis["retrieval_mode"] = "raw_source_order"
        retrieval_snapshot_id = save_retrieval_snapshot(
            request.topic,
            request.max_papers,
            papers,
            retrieval_query,
        )
        print(f"[pipeline] saved retrieval snapshot: {retrieval_snapshot_id}", flush=True)
    if papers_override is not None and should_rerank:
        papers, rerank_scores = await rerank_papers_by_query(llm, retrieval_query, papers)
        embedding_analysis["retrieval_reranking"] = rerank_scores
    elif papers_override is not None:
        embedding_analysis["retrieval_mode"] = "raw_source_order"
    print(f"[pipeline] literature search returned {len(papers)} paper(s)", flush=True)
    literature_themes = await cluster_literature(llm, papers)
    paper_dicts = [paper.model_dump() for paper in papers]
    raw_papers_context = format_papers_context(paper_dicts)
    evidence_notes = (
        []
        if request.condition in {"baseline_with_retrieval", "multi_stage_without_retrieval"}
        else await generate_evidence_notes(llm, request.topic, raw_papers_context, len(papers))
    )
    evidence_notes_context = format_evidence_notes(evidence_notes)
    themes_context = format_literature_themes(literature_themes)
    retrieval_context = (
        raw_papers_context
        if request.condition == "baseline_with_retrieval"
        else "\n\n".join(part for part in [themes_context, evidence_notes_context] if part)
    )
    context = (
        ""
        if request.condition == "multi_stage_without_retrieval"
        else retrieval_context
    )
    group_selection = None
    draft_proposal = None
    proposal_critique = None
    proposal_logic_graph = None
    fallback_reason = None

    try:
        baseline = None
        if request.condition == "baseline_with_retrieval":
            baseline = await run_baseline(llm, request.topic, context)
            stage_data = {
                "summary": None,
                "scholarly_positioning": None,
                "domain_router": None,
                "research_scope": None,
                "concept_definition": None,
                "technical_feasibility": None,
                "gap": None,
                "question": None,
                "theoretical_mechanism": None,
                "hypothesis": None,
                "operationalization_causal_check": None,
                "methodology": None,
                "novelty_contribution": None,
                "failure_analysis": None,
                "proposal": baseline.proposal or "",
            }
            evaluation = await run_evaluation(llm, request.topic, context, stage_data)
            baseline.evaluation = evaluation
        elif request.condition == "multi_stage_group_selection":
            stage_data, group_selection = await run_group_selection_stages(llm, request.topic, context, papers)
            embedding_analysis["group_selection"] = group_selection
            draft_proposal = stage_data["proposal"]
            stage_data, proposal_critique = await run_proposal_refinement(llm, request.topic, context, stage_data)
            proposal_logic_graph = await run_logic_graph_check(llm, request.topic, context, stage_data)
            evaluation = await run_evaluation(llm, request.topic, context, stage_data)
        else:
            stage_data, stage_embedding_analysis = await run_llm_stages(llm, request.topic, context, papers)
            embedding_analysis.update(stage_embedding_analysis)
            if request.condition in {"multi_stage_with_retrieval", "multi_stage_without_retrieval", "main_comparison"}:
                draft_proposal = stage_data["proposal"]
                stage_data, proposal_critique = await run_proposal_refinement(llm, request.topic, context, stage_data)
            if request.condition != "baseline_with_retrieval":
                proposal_logic_graph = await run_logic_graph_check(llm, request.topic, context, stage_data)
            evaluation = await run_evaluation(llm, request.topic, context, stage_data)
        if request.condition == "main_comparison":
            baseline = await run_baseline(llm, request.topic, context)
        print("[pipeline] completed with LLM output", flush=True)
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        print(f"[pipeline] falling back because: {fallback_reason}", flush=True)
        baseline = None
        evaluation = fallback_evaluation()
        if request.condition == "baseline_with_retrieval":
            baseline = BaselineResult(
                proposal=(
                    "Single-prompt baseline fallback: generate the full research proposal directly from the topic "
                    "without explicit literature summary, gap, question, hypothesis, and methodology stages."
                ),
                evaluation=evaluation,
            )
            stage_data = {
                "summary": None,
                "scholarly_positioning": None,
                "domain_router": None,
                "research_scope": None,
                "concept_definition": None,
                "technical_feasibility": None,
                "gap": None,
                "question": None,
                "theoretical_mechanism": None,
                "hypothesis": None,
                "operationalization_causal_check": None,
                "methodology": None,
                "novelty_contribution": None,
                "failure_analysis": None,
                "proposal": baseline.proposal,
            }
        else:
            stage_data = fallback_stage_outputs(request.topic)
            draft_proposal = stage_data.get("proposal")
            if request.condition == "multi_stage_group_selection":
                group_selection = fallback_group_selection(stage_data)
            if request.condition == "main_comparison":
                baseline = BaselineResult(
                    proposal=(
                        "Single-prompt baseline fallback: generate the full research proposal directly from the topic "
                        "without explicit literature summary, gap, question, hypothesis, and methodology stages."
                    ),
                    evaluation=Evaluation(
                        clarity=3,
                        logic=3,
                        novelty=3,
                        feasibility=4,
                        literature_alignment=2,
                        comments="Baseline fallback is usable for comparison but less literature-grounded.",
                    ),
                )

    metrics = RunMetrics(
        condition=request.condition,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        retrieval_query=retrieval_query,
        retrieval_snapshot_id=retrieval_snapshot_id,
        latency_ms=round((perf_counter() - started_at) * 1000),
        prompt_tokens=llm.prompt_tokens or None,
        completion_tokens=llm.completion_tokens or None,
        total_tokens=llm.total_tokens or None,
        required_field_completeness=calculate_field_completeness(stage_data, evaluation, baseline),
        fallback_reason=fallback_reason,
    )
    result = ResearchResult(
        experiment_id=request.experiment_id,
        run_index=request.run_index,
        topic=request.topic,
        condition=request.condition,
        retrieval_query=retrieval_query,
        retrieval_snapshot_id=retrieval_snapshot_id,
        papers=papers,
        literature_themes=literature_themes,
        evidence_notes=evidence_notes,
        summary=stage_data.get("summary"),
        scholarly_positioning=stage_data.get("scholarly_positioning"),
        domain_router=stage_data.get("domain_router"),
        research_scope=stage_data.get("research_scope"),
        concept_definition=stage_data.get("concept_definition"),
        technical_feasibility=stage_data.get("technical_feasibility"),
        gap=stage_data.get("gap"),
        question=stage_data.get("question"),
        theoretical_mechanism=stage_data.get("theoretical_mechanism"),
        hypothesis=stage_data.get("hypothesis"),
        operationalization_causal_check=stage_data.get("operationalization_causal_check"),
        methodology=stage_data.get("methodology"),
        novelty_contribution=stage_data.get("novelty_contribution"),
        failure_analysis=stage_data.get("failure_analysis"),
        proposal_logic_graph=proposal_logic_graph,
        proposal=stage_data.get("proposal") or "",
        draft_proposal=draft_proposal,
        proposal_critique=proposal_critique,
        evaluation=evaluation,
        baseline=baseline,
        group_selection=group_selection,
        embedding_analysis=embedding_analysis,
        metrics=metrics,
    )
    run_path = save_run(request.topic, result)
    print(f"[pipeline] saved run: {run_path}", flush=True)
    return result


async def run_llm_stages(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    papers: list[Paper],
) -> tuple[dict[str, str], dict[str, Any]]:
    embedding_analysis: dict[str, Any] = {}
    use_retrieval_evidence = bool(papers_context.strip())
    summary = await run_text_stage(llm, "literature_summary.txt", topic, papers_context, {}, "summary")
    scholarly_positioning = await run_text_stage(
        llm,
        "scholarly_positioning.txt",
        topic,
        papers_context,
        {"summary": summary},
        "scholarly_positioning",
    )
    domain_router = await run_text_stage(
        llm,
        "domain_router.txt",
        topic,
        papers_context,
        {"summary": summary, "scholarly_positioning": scholarly_positioning},
        "domain_router",
    )
    research_scope = await run_text_stage(
        llm,
        "research_scope.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
        },
        "research_scope",
    )
    gap_data = await run_json_stage(
        llm,
        "research_gap.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
        },
        "gap",
    )
    gap = gap_data["gap"]
    concept_definition = await run_text_stage(
        llm,
        "concept_definition.txt",
        topic,
        grounded_context if "grounded_context" in locals() else papers_context,
        {
            "domain_router": domain_router,
            "research_scope": research_scope,
            "gap": gap,
        },
        "concept_definition",
    )
    technical_feasibility = await run_text_stage(
        llm,
        "technical_feasibility.txt",
        topic,
        grounded_context if "grounded_context" in locals() else papers_context,
        {
            "domain_router": domain_router,
            "concept_definition": concept_definition,
            "research_scope": research_scope,
            "gap": gap,
        },
        "technical_feasibility",
    )
    gap_data["supporting_paper_ids"] = validate_paper_ids(gap_data.get("supporting_paper_ids"), papers)
    gap_evidence = (
        await analyze_text_against_papers(llm, "research gap", gap, papers)
        if use_retrieval_evidence
        else None
    )
    if gap_evidence:
        embedding_analysis["gap_validation"] = gap_evidence
    evidence_context = format_embedding_evidence(gap_evidence)
    grounded_context = "\n\n".join(part for part in [papers_context, evidence_context] if part)
    question = await run_text_stage(
        llm,
        "research_question.txt",
        topic,
        grounded_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
        },
        "question",
    )
    theoretical_mechanism = await run_text_stage(
        llm,
        "theoretical_mechanism.txt",
        topic,
        grounded_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
        },
        "theoretical_mechanism",
    )
    hypothesis = await run_text_stage(
        llm,
        "hypothesis.txt",
        topic,
        grounded_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
        },
        "hypothesis",
    )
    operationalization_causal_check = await run_text_stage(
        llm,
        "operationalization_causal_check.txt",
        topic,
        grounded_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
        },
        "operationalization_causal_check",
    )
    methodology_data = await run_json_stage(
        llm,
        "methodology.txt",
        topic,
        grounded_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
        },
        "methodology",
    )
    methodology = methodology_data["methodology"]
    methodology_data["methodology_supporting_paper_ids"] = validate_paper_ids(
        methodology_data.get("methodology_supporting_paper_ids"),
        papers,
    )
    methodology_evidence = (
        await analyze_text_against_papers(llm, "methodology", methodology, papers)
        if use_retrieval_evidence
        else None
    )
    if methodology_evidence:
        embedding_analysis["methodology_citation_support"] = methodology_evidence
    novelty_contribution = await run_text_stage(
        llm,
        "novelty_contribution.txt",
        topic,
        grounded_context,
        {
            "research_scope": research_scope,
            "domain_router": domain_router,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "scholarly_positioning": scholarly_positioning,
            "gap": gap,
            "theoretical_mechanism": theoretical_mechanism,
            "methodology": methodology,
        },
        "novelty_contribution",
    )
    failure_analysis = await run_text_stage(
        llm,
        "failure_analysis.txt",
        topic,
        grounded_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "supporting_paper_ids": ", ".join(gap_data.get("supporting_paper_ids") or []),
            "evidence_reasoning": str(gap_data.get("evidence_reasoning") or ""),
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
            "methodology": methodology,
        },
        "failure_analysis",
    )
    proposal_evidence_context = "\n\n".join(
        part for part in [grounded_context, format_embedding_evidence(methodology_evidence)] if part
    )
    proposal = await run_text_stage(
        llm,
        "proposal.txt",
        topic,
        proposal_evidence_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "supporting_paper_ids": ", ".join(gap_data.get("supporting_paper_ids") or []),
            "evidence_reasoning": str(gap_data.get("evidence_reasoning") or ""),
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
            "methodology": methodology,
            "methodology_supporting_paper_ids": ", ".join(methodology_data.get("methodology_supporting_paper_ids") or []),
            "baseline_reasoning": str(methodology_data.get("baseline_reasoning") or ""),
            "failure_analysis": failure_analysis,
            "novelty_contribution": novelty_contribution,
        },
        "proposal",
    )
    return (
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "supporting_paper_ids": ", ".join(gap_data.get("supporting_paper_ids") or []),
            "evidence_reasoning": str(gap_data.get("evidence_reasoning") or ""),
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
            "methodology": methodology,
            "methodology_supporting_paper_ids": ", ".join(methodology_data.get("methodology_supporting_paper_ids") or []),
            "baseline_reasoning": str(methodology_data.get("baseline_reasoning") or ""),
            "novelty_contribution": novelty_contribution,
            "failure_analysis": failure_analysis,
            "proposal": proposal,
        },
        embedding_analysis,
    )


async def run_group_selection_stages(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    papers: list[Paper],
) -> tuple[dict[str, str], dict[str, Any]]:
    summary = await run_text_stage(llm, "literature_summary.txt", topic, papers_context, {}, "summary")
    scholarly_positioning = await run_text_stage(
        llm,
        "scholarly_positioning.txt",
        topic,
        papers_context,
        {"summary": summary},
        "scholarly_positioning",
    )
    domain_router = await run_text_stage(
        llm,
        "domain_router.txt",
        topic,
        papers_context,
        {"summary": summary, "scholarly_positioning": scholarly_positioning},
        "domain_router",
    )
    research_scope = await run_text_stage(
        llm,
        "research_scope.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
        },
        "research_scope",
    )
    gap_selection = await run_candidate_selection(
        llm,
        topic=topic,
        papers_context=papers_context,
        stage_name="gap",
        candidate_key="gap_candidates",
        criteria=["specificity", "literature_grounding", "novelty", "feasibility"],
        papers=papers,
        context={
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
        },
    )
    gap = gap_selection["selected_text"]
    gap_supporting_paper_ids = validate_paper_ids(selected_candidate_paper_ids(gap_selection), papers)
    gap_evidence_reasoning = selected_candidate_reason(gap_selection)
    concept_definition = await run_text_stage(
        llm,
        "concept_definition.txt",
        topic,
        papers_context,
        {
            "domain_router": domain_router,
            "research_scope": research_scope,
            "gap": gap,
        },
        "concept_definition",
    )
    technical_feasibility = await run_text_stage(
        llm,
        "technical_feasibility.txt",
        topic,
        papers_context,
        {
            "domain_router": domain_router,
            "concept_definition": concept_definition,
            "research_scope": research_scope,
            "gap": gap,
        },
        "technical_feasibility",
    )

    question = await run_text_stage(
        llm,
        "research_question.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
        },
        "question",
    )
    theoretical_mechanism = await run_text_stage(
        llm,
        "theoretical_mechanism.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
        },
        "theoretical_mechanism",
    )
    hypothesis_selection = await run_candidate_selection(
        llm,
        topic=topic,
        papers_context=papers_context,
        stage_name="hypothesis",
        candidate_key="hypothesis_candidates",
        criteria=["testability", "alignment_with_gap", "clarity", "methodological_implication"],
        papers=papers,
        context={
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
        },
    )
    hypothesis = hypothesis_selection["selected_text"]
    operationalization_causal_check = await run_text_stage(
        llm,
        "operationalization_causal_check.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
        },
        "operationalization_causal_check",
    )

    methodology_selection = await run_candidate_selection(
        llm,
        topic=topic,
        papers_context=papers_context,
        stage_name="methodology",
        candidate_key="methodology_candidates",
        criteria=["feasibility", "baseline_design", "evaluation_metrics", "scope_control", "alignment_with_hypothesis"],
        papers=papers,
        context={
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
        },
    )
    methodology = methodology_selection["selected_text"]
    methodology_supporting_paper_ids = validate_paper_ids(selected_candidate_paper_ids(methodology_selection), papers)
    baseline_reasoning = selected_candidate_reason(methodology_selection)
    novelty_contribution = await run_text_stage(
        llm,
        "novelty_contribution.txt",
        topic,
        papers_context,
        {
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "scholarly_positioning": scholarly_positioning,
            "gap": gap,
            "theoretical_mechanism": theoretical_mechanism,
            "methodology": methodology,
        },
        "novelty_contribution",
    )
    failure_analysis = await run_text_stage(
        llm,
        "failure_analysis.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "supporting_paper_ids": ", ".join(gap_supporting_paper_ids),
            "evidence_reasoning": gap_evidence_reasoning,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
            "methodology": methodology,
        },
        "failure_analysis",
    )

    proposal = await run_text_stage(
        llm,
        "proposal.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "supporting_paper_ids": ", ".join(gap_supporting_paper_ids),
            "evidence_reasoning": gap_evidence_reasoning,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
            "methodology": methodology,
            "methodology_supporting_paper_ids": ", ".join(methodology_supporting_paper_ids),
            "baseline_reasoning": baseline_reasoning,
            "failure_analysis": failure_analysis,
            "novelty_contribution": novelty_contribution,
        },
        "proposal",
    )
    return (
        {
            "summary": summary,
            "scholarly_positioning": scholarly_positioning,
            "domain_router": domain_router,
            "research_scope": research_scope,
            "concept_definition": concept_definition,
            "technical_feasibility": technical_feasibility,
            "gap": gap,
            "supporting_paper_ids": ", ".join(gap_supporting_paper_ids),
            "evidence_reasoning": gap_evidence_reasoning,
            "question": question,
            "theoretical_mechanism": theoretical_mechanism,
            "hypothesis": hypothesis,
            "operationalization_causal_check": operationalization_causal_check,
            "methodology": methodology,
            "methodology_supporting_paper_ids": ", ".join(methodology_supporting_paper_ids),
            "baseline_reasoning": baseline_reasoning,
            "novelty_contribution": novelty_contribution,
            "failure_analysis": failure_analysis,
            "proposal": proposal,
        },
        {
            "gap": gap_selection,
            "hypothesis": hypothesis_selection,
            "methodology": methodology_selection,
        },
    )


async def run_proposal_refinement(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    stage_data: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    print("[pipeline] LLM stage start: proposal critique", flush=True)
    critique_template = load_prompt("proposal_critique.txt")
    critique_data = await llm.complete_json(
        SYSTEM_PROMPT,
        critique_template.format(topic=topic, papers_context=papers_context, **stage_data),
    )
    critique = critique_data.get("critique")
    if not isinstance(critique, dict):
        raise ValueError("LLM response missing critique")
    print("[pipeline] LLM stage done: proposal critique", flush=True)

    print("[pipeline] LLM stage start: proposal revision", flush=True)
    revision_template = load_prompt("proposal_revision.txt")
    revision_data = await llm.complete_json(
        SYSTEM_PROMPT,
        revision_template.format(
            topic=topic,
            papers_context=papers_context,
            critique=critique,
            **stage_data,
        ),
    )
    revised_proposal = revision_data.get("revised_proposal")
    if not isinstance(revised_proposal, str) or not revised_proposal.strip():
        raise ValueError("LLM response missing revised_proposal")
    print("[pipeline] LLM stage done: proposal revision", flush=True)
    return {**stage_data, "proposal": revised_proposal.strip()}, critique


async def run_candidate_selection(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    stage_name: str,
    candidate_key: str,
    criteria: list[str],
    papers: list[Paper],
    context: dict[str, str],
) -> dict[str, Any]:
    print(f"[pipeline] LLM group-selection start: {stage_name}", flush=True)
    context_text = "\n".join(f"{key}: {value}" for key, value in context.items())
    criteria_text = ", ".join(criteria)
    user_prompt = f"""
Task: Generate and rank 3 candidate {stage_name} outputs for an academic research proposal.

Topic:
{topic}

Retrieved papers:
{papers_context}

Current context:
{context_text}

Generate exactly 3 candidates. Score each candidate from 1 to 5 on these criteria:
{criteria_text}

When retrieved papers are available, include supporting_paper_ids using the P1, P2, ... labels from the evidence notes.
Select the strongest candidate using group-relative comparison.

Return valid JSON exactly in this structure:
{{
  "candidates": [
    {{
      "id": "{stage_name}_1",
      "text": "...",
      "supporting_paper_ids": ["P1"],
      "scores": {{"{criteria[0]}": 1}},
      "total": 1,
      "reason": "Brief comparison-based reason."
    }}
  ],
  "selected_id": "{stage_name}_1",
  "selected_text": "The selected candidate text."
}}
"""
    data = await llm.complete_json(SYSTEM_PROMPT, user_prompt)
    candidates = data.get("candidates")
    selected_text = data.get("selected_text")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"LLM response missing {candidate_key}")
    anchor_text = candidate_anchor_text(stage_name, topic, context)
    enhanced_candidates = await add_candidate_embedding_alignment(llm, candidates, anchor_text, papers)
    if enhanced_candidates:
        candidates = enhanced_candidates
        best = max(enhanced_candidates, key=lambda candidate: candidate.get("combined_total", 0))
        selected_text = best.get("text")
        selected_id = best.get("id")
    else:
        selected_id = data.get("selected_id") or first_candidate_id(candidates)
    if not isinstance(selected_text, str) or not selected_text.strip():
        selected_text = extract_best_candidate_text(candidates)
        selected_id = selected_id or first_candidate_id(candidates)
    if not selected_text:
        raise ValueError(f"LLM response missing selected {stage_name}")
    print(f"[pipeline] LLM group-selection done: {stage_name}", flush=True)
    return {
        "candidates": candidates,
        "selected_id": selected_id,
        "selected_text": selected_text.strip(),
        "criteria": criteria,
        "selection_method": "llm_scores_plus_embedding_alignment" if enhanced_candidates else "llm_scores_only",
    }


def candidate_anchor_text(stage_name: str, topic: str, context: dict[str, str]) -> str:
    if stage_name == "gap":
        return f"{topic}\n{context.get('summary', '')}"
    if stage_name == "hypothesis":
        return f"{context.get('gap', '')}\n{context.get('question', '')}\n{context.get('theoretical_mechanism', '')}"
    if stage_name == "methodology":
        return f"{context.get('hypothesis', '')}\n{context.get('operationalization_causal_check', '')}\n{context.get('question', '')}"
    return "\n".join([topic, *context.values()])


def extract_best_candidate_text(candidates: list[Any]) -> str:
    best_candidate: dict[str, Any] | None = None
    best_total = -1
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        total = candidate.get("total")
        if not isinstance(total, int | float):
            scores = candidate.get("scores")
            total = sum(scores.values()) if isinstance(scores, dict) else 0
        if total > best_total:
            best_total = total
            best_candidate = candidate
    if not best_candidate:
        return ""
    text = best_candidate.get("text")
    return text if isinstance(text, str) else ""


def first_candidate_id(candidates: list[Any]) -> str | None:
    first = candidates[0] if candidates else None
    if isinstance(first, dict) and isinstance(first.get("id"), str):
        return first["id"]
    return None


def fallback_group_selection(stage_data: dict[str, str]) -> dict[str, Any]:
    return {
        "gap": {
            "candidates": [{"id": "gap_1", "text": stage_data["gap"], "scores": {}, "total": None, "reason": "Fallback selection."}],
            "selected_id": "gap_1",
            "selected_text": stage_data["gap"],
            "criteria": ["specificity", "literature_grounding", "novelty", "feasibility"],
        },
        "hypothesis": {
            "candidates": [{"id": "hypothesis_1", "text": stage_data["hypothesis"], "scores": {}, "total": None, "reason": "Fallback selection."}],
            "selected_id": "hypothesis_1",
            "selected_text": stage_data["hypothesis"],
            "criteria": ["testability", "alignment_with_gap", "clarity", "methodological_implication"],
        },
        "methodology": {
            "candidates": [{"id": "methodology_1", "text": stage_data["methodology"], "scores": {}, "total": None, "reason": "Fallback selection."}],
            "selected_id": "methodology_1",
            "selected_text": stage_data["methodology"],
            "criteria": ["feasibility", "baseline_design", "evaluation_metrics", "scope_control", "alignment_with_hypothesis"],
        },
    }


async def run_text_stage(
    llm: LLMClient,
    prompt_name: str,
    topic: str,
    papers_context: str,
    previous_outputs: dict[str, str],
    key: str,
) -> str:
    print(f"[pipeline] LLM stage start: {key}", flush=True)
    template = load_prompt(prompt_name)
    user_prompt = template.format(topic=topic, papers_context=papers_context, **previous_outputs)
    data = await llm.complete_json(SYSTEM_PROMPT, user_prompt)
    value = extract_stage_value(data, key)
    if not value:
        raise ValueError(f"LLM response missing {key}")
    print(f"[pipeline] LLM stage done: {key}", flush=True)
    return value


async def run_json_stage(
    llm: LLMClient,
    prompt_name: str,
    topic: str,
    papers_context: str,
    previous_outputs: dict[str, str],
    key: str,
) -> dict[str, Any]:
    print(f"[pipeline] LLM stage start: {key}", flush=True)
    template = load_prompt(prompt_name)
    user_prompt = template.format(topic=topic, papers_context=papers_context, **previous_outputs)
    data = await llm.complete_json(SYSTEM_PROMPT, user_prompt)
    value = extract_stage_value(data, key)
    if not value:
        raise ValueError(f"LLM response missing {key}")
    data[key] = value
    print(f"[pipeline] LLM stage done: {key}", flush=True)
    return data


def extract_stage_value(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        value = data.get(key.replace("_check", ""))
    if value is None and key == "operationalization_causal_check":
        value = (
            data.get("operationalization")
            or data.get("construct_operationalization")
            or data.get("operationalization_table")
            or data.get("causal_check")
            or data.get("constructs")
        )
    if value is None and len(data) == 1:
        value = next(iter(data.values()))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return ""


async def generate_evidence_notes(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    paper_count: int,
) -> list[EvidenceNote]:
    if not papers_context.strip() or paper_count <= 0:
        return []
    try:
        print("[pipeline] LLM stage start: evidence notes", flush=True)
        template = load_prompt("evidence_notes.txt")
        data = await llm.complete_json(
            SYSTEM_PROMPT,
            template.format(topic=topic, papers_context=papers_context),
        )
        notes_data = data.get("evidence_notes")
        if not isinstance(notes_data, list):
            raise ValueError("LLM response missing evidence_notes")
        allowed_ids = {f"P{index}" for index in range(1, paper_count + 1)}
        notes: list[EvidenceNote] = []
        for index, item in enumerate(notes_data[:paper_count], start=1):
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or f"P{index}").strip().upper()
            if paper_id not in allowed_ids:
                paper_id = f"P{index}" if f"P{index}" in allowed_ids else ""
            if not paper_id:
                continue
            notes.append(
                EvidenceNote(
                    paper_id=paper_id,
                    title=str(item.get("title") or "").strip(),
                    problem=str(item.get("problem") or "").strip(),
                    method=str(item.get("method") or "").strip(),
                    evidence=str(item.get("evidence") or "").strip(),
                    limitation=str(item.get("limitation") or "").strip(),
                    gap_relevance=str(item.get("gap_relevance") or "").strip(),
                )
            )
        print("[pipeline] LLM stage done: evidence notes", flush=True)
        return notes
    except Exception as exc:
        print(f"[pipeline] evidence notes unavailable: {type(exc).__name__}: {exc}", flush=True)
        return []


def format_evidence_notes(notes: list[EvidenceNote]) -> str:
    if not notes:
        return ""
    sections = ["Compressed evidence notes for grounding:"]
    for note in notes:
        sections.append(
            "\n".join(
                [
                    f"{note.paper_id}: {note.title or 'Untitled paper'}",
                    f"Problem: {note.problem}",
                    f"Method: {note.method}",
                    f"Evidence: {note.evidence}",
                    f"Limitation: {note.limitation}",
                    f"Gap relevance: {note.gap_relevance}",
                ]
            )
        )
    return "\n\n".join(sections)


def validate_paper_ids(value: Any, papers: list[Paper]) -> list[str]:
    allowed_ids = {f"P{index}" for index in range(1, len(papers) + 1)}
    raw_ids: list[str] = []
    if isinstance(value, str):
        raw_ids = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_ids = [str(item).strip() for item in value]
    valid_ids: list[str] = []
    for raw_id in raw_ids:
        normalized = raw_id.upper()
        if normalized in allowed_ids and normalized not in valid_ids:
            valid_ids.append(normalized)
    return valid_ids


def selected_candidate_paper_ids(selection: dict[str, Any]) -> list[str]:
    selected_id = selection.get("selected_id")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        return []
    selected_candidate = None
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("id") == selected_id:
            selected_candidate = candidate
            break
    if selected_candidate is None:
        selected_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else None
    if not isinstance(selected_candidate, dict):
        return []
    raw_value = (
        selected_candidate.get("supporting_paper_ids")
        or selected_candidate.get("paper_ids")
        or selected_candidate.get("methodology_supporting_paper_ids")
    )
    if isinstance(raw_value, str):
        return [part.strip().upper() for part in raw_value.replace(";", ",").split(",") if part.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip().upper() for item in raw_value if str(item).strip()]
    return []


def selected_candidate_reason(selection: dict[str, Any]) -> str:
    selected_id = selection.get("selected_id")
    candidates = selection.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("id") == selected_id:
                reason = candidate.get("reason")
                if isinstance(reason, str) and reason.strip():
                    return reason.strip()
    selected_text = selection.get("selected_text")
    return selected_text.strip() if isinstance(selected_text, str) else ""


async def run_logic_graph_check(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    stage_data: dict[str, str],
) -> dict[str, Any] | None:
    try:
        print("[pipeline] LLM stage start: proposal logic graph", flush=True)
        template = load_prompt("proposal_logic_graph.txt")
        data = await llm.complete_json(
            SYSTEM_PROMPT,
            template.format(topic=topic, papers_context=papers_context, **stage_data),
        )
        graph = data.get("proposal_logic_graph")
        if not isinstance(graph, dict):
            raise ValueError("LLM response missing proposal_logic_graph")
        print("[pipeline] LLM stage done: proposal logic graph", flush=True)
        return graph
    except Exception as exc:
        print(f"[pipeline] proposal logic graph unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None


async def run_evaluation(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    stage_data: dict[str, str],
) -> Evaluation:
    print("[pipeline] LLM stage start: evaluation", flush=True)
    data = await llm.complete_json(
        SYSTEM_PROMPT,
        build_evaluation_prompt(topic=topic, papers_context=papers_context, stage_data=stage_data),
    )
    print("[pipeline] LLM stage done: evaluation", flush=True)
    return evaluation_from_rubric_response(data)


async def run_baseline(llm: LLMClient, topic: str, papers_context: str) -> BaselineResult:
    print("[pipeline] LLM stage start: baseline", flush=True)
    template = load_prompt("baseline.txt")
    data = await llm.complete_json(SYSTEM_PROMPT, template.format(topic=topic, papers_context=papers_context))
    proposal = data.get("proposal")
    print("[pipeline] LLM stage done: baseline", flush=True)
    return BaselineResult(proposal=proposal, evaluation=None)


def calculate_field_completeness(
    stage_data: dict[str, str | None],
    evaluation: Evaluation | None,
    baseline: BaselineResult | None,
) -> float:
    if baseline is not None:
        required_values: list[object] = [baseline.proposal, baseline.evaluation]
    else:
        required_values = [
            stage_data.get("summary"),
            stage_data.get("scholarly_positioning"),
            stage_data.get("domain_router"),
            stage_data.get("research_scope"),
            stage_data.get("concept_definition"),
            stage_data.get("technical_feasibility"),
            stage_data.get("gap"),
            stage_data.get("question"),
            stage_data.get("theoretical_mechanism"),
            stage_data.get("hypothesis"),
            stage_data.get("operationalization_causal_check"),
            stage_data.get("methodology"),
            stage_data.get("novelty_contribution"),
            stage_data.get("failure_analysis"),
            stage_data.get("proposal"),
            evaluation,
        ]

    present = sum(1 for value in required_values if value)
    return round(present / len(required_values), 3)
