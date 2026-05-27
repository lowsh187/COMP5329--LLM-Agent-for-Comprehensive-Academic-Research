from app.models import Evaluation


def fallback_stage_outputs(topic: str) -> dict[str, str]:
    return {
        "summary": (
            f"Existing literature related to {topic} suggests active interest in applying LLMs "
            "to support academic or educational workflows, but reported outcomes depend on task design, "
            "evaluation criteria, and quality of retrieved evidence."
        ),
        "gap": (
            "A key gap is the limited comparison between direct single-prompt generation and a structured, "
            "literature-augmented multi-stage workflow."
        ),
        "scholarly_positioning": (
            "Prior work often demonstrates task-specific LLM assistance, but less often compares how staged "
            "research reasoning changes proposal quality under controlled evaluation."
        ),
        "domain_router": (
            "Fallback routing classifies the project as an LLM-agent evaluation and academic workflow prototype, "
            "so the appropriate method is a controlled comparison between generation conditions rather than a "
            "clinical intervention design."
        ),
        "research_scope": (
            "The fallback scope is a controlled evaluation of an academic research agent for one user-provided "
            "topic, comparing a raw single-prompt baseline with structured multi-stage generation using the same "
            "retrieved-paper set and the same PhD-level rubric."
        ),
        "concept_definition": (
            "The core concepts are proposal quality, literature grounding, staged reasoning, and baseline comparison; "
            "each should be measured through rubric scores, citation use, completeness, and condition-level outputs."
        ),
        "technical_feasibility": (
            "The fallback design only requires retrieved metadata, saved run logs, LLM generation, and rubric scoring; "
            "it does not require hidden-state access, human-subject trials, or GPU-heavy model training."
        ),
        "question": (
            f"How can a literature-augmented multi-stage LLM workflow improve the clarity, feasibility, "
            f"and literature alignment of research proposals about {topic}?"
        ),
        "theoretical_mechanism": (
            "Separating literature synthesis, gap formation, hypothesis design, and method planning should reduce "
            "cognitive overload and make intermediate assumptions easier to inspect."
        ),
        "hypothesis": (
            "A staged pipeline that separates literature summary, gap identification, question generation, "
            "methodology design, and evaluation will produce more coherent proposals than a single prompt."
        ),
        "operationalization_causal_check": (
            "Proposal quality can be operationalized through rubric scores for clarity, coherence, novelty, "
            "feasibility, literature alignment, academic depth, and communication, while causal claims should be "
            "limited to observed differences between controlled generation conditions."
        ),
        "methodology": (
            "Build a lightweight prototype, retrieve a small set of arXiv papers, generate outputs through "
            "independent prompt templates, and compare the final proposal with a single-prompt baseline using "
            "a manual rubric and GPT-assisted self-evaluation."
        ),
        "novelty_contribution": (
            "The fallback contribution is methodological: it decomposes proposal generation into inspectable "
            "research-reasoning stages and compares that workflow against a raw single-prompt baseline."
        ),
        "failure_analysis": (
            "The workflow may fail when retrieval is weak, when the LLM overstates novelty, or when the evaluator "
            "rewards polished writing over genuine research contribution."
        ),
        "proposal": (
            f"This project develops an LLM-based Academic Research Agent for {topic}. The system retrieves "
            "relevant literature, summarizes evidence, identifies research gaps, generates a research question "
            "and hypothesis, suggests a feasible methodology, drafts a proposal, and evaluates output quality. "
            "The expected contribution is a stable, modular workflow for literature-grounded academic ideation."
        ),
    }


def fallback_evaluation() -> Evaluation:
    return Evaluation(
        clarity=4,
        logic=4,
        novelty=3,
        feasibility=5,
        literature_alignment=3,
        comments=(
            "Fallback evaluation indicates a clear and feasible proposal structure. Literature alignment should "
            "be reassessed after successful arXiv retrieval and real LLM generation."
        ),
    )
