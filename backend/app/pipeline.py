from app.arxiv_client import search_arxiv
from app.fallback import fallback_evaluation, fallback_stage_outputs
from app.llm_client import LLMClient
from app.models import BaselineResult, Evaluation, ResearchRequest, ResearchResult
from app.prompts import format_papers_context, load_prompt

SYSTEM_PROMPT = (
    "You are an academic research assistant. Always return valid JSON only. "
    "Keep outputs concise, specific, feasible, and aligned with the retrieved literature."
)


async def run_research_pipeline(request: ResearchRequest) -> ResearchResult:
    print(f"[pipeline] start topic={request.topic!r}", flush=True)
    print("[pipeline] searching arXiv", flush=True)
    papers = await search_arxiv(request.topic, request.max_papers)
    print(f"[pipeline] arXiv returned {len(papers)} paper(s)", flush=True)
    paper_dicts = [paper.model_dump() for paper in papers]
    context = format_papers_context(paper_dicts)
    llm = LLMClient()

    try:
        stage_data = await run_llm_stages(llm, request.topic, context)
        evaluation = await run_evaluation(llm, request.topic, context, stage_data)
        baseline = await run_baseline(llm, request.topic, context)
        print("[pipeline] completed with LLM output", flush=True)
    except Exception as exc:
        print(f"[pipeline] falling back because: {type(exc).__name__}: {exc}", flush=True)
        stage_data = fallback_stage_outputs(request.topic)
        evaluation = fallback_evaluation()
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

    return ResearchResult(
        topic=request.topic,
        papers=papers,
        summary=stage_data["summary"],
        gap=stage_data["gap"],
        question=stage_data["question"],
        hypothesis=stage_data["hypothesis"],
        methodology=stage_data["methodology"],
        proposal=stage_data["proposal"],
        evaluation=evaluation,
        baseline=baseline,
    )


async def run_llm_stages(llm: LLMClient, topic: str, papers_context: str) -> dict[str, str]:
    summary = await run_text_stage(llm, "literature_summary.txt", topic, papers_context, {}, "summary")
    gap = await run_text_stage(llm, "research_gap.txt", topic, papers_context, {"summary": summary}, "gap")
    question = await run_text_stage(
        llm,
        "research_question.txt",
        topic,
        papers_context,
        {"summary": summary, "gap": gap},
        "question",
    )
    hypothesis = await run_text_stage(
        llm,
        "hypothesis.txt",
        topic,
        papers_context,
        {"summary": summary, "gap": gap, "question": question},
        "hypothesis",
    )
    methodology = await run_text_stage(
        llm,
        "methodology.txt",
        topic,
        papers_context,
        {"summary": summary, "gap": gap, "question": question, "hypothesis": hypothesis},
        "methodology",
    )
    proposal = await run_text_stage(
        llm,
        "proposal.txt",
        topic,
        papers_context,
        {
            "summary": summary,
            "gap": gap,
            "question": question,
            "hypothesis": hypothesis,
            "methodology": methodology,
        },
        "proposal",
    )
    return {
        "summary": summary,
        "gap": gap,
        "question": question,
        "hypothesis": hypothesis,
        "methodology": methodology,
        "proposal": proposal,
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
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM response missing {key}")
    print(f"[pipeline] LLM stage done: {key}", flush=True)
    return value.strip()


async def run_evaluation(
    llm: LLMClient,
    topic: str,
    papers_context: str,
    stage_data: dict[str, str],
) -> Evaluation:
    print("[pipeline] LLM stage start: evaluation", flush=True)
    template = load_prompt("evaluation.txt")
    data = await llm.complete_json(
        SYSTEM_PROMPT,
        template.format(topic=topic, papers_context=papers_context, **stage_data),
    )
    print("[pipeline] LLM stage done: evaluation", flush=True)
    return Evaluation(**data)


async def run_baseline(llm: LLMClient, topic: str, papers_context: str) -> BaselineResult:
    print("[pipeline] LLM stage start: baseline", flush=True)
    template = load_prompt("baseline.txt")
    data = await llm.complete_json(SYSTEM_PROMPT, template.format(topic=topic, papers_context=papers_context))
    proposal = data.get("proposal")
    evaluation_data = data.get("evaluation")
    evaluation = Evaluation(**evaluation_data) if isinstance(evaluation_data, dict) else None
    print("[pipeline] LLM stage done: baseline", flush=True)
    return BaselineResult(proposal=proposal, evaluation=evaluation)
