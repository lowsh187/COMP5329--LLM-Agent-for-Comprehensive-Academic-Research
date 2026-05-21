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
        "question": (
            f"How can a literature-augmented multi-stage LLM workflow improve the clarity, feasibility, "
            f"and literature alignment of research proposals about {topic}?"
        ),
        "hypothesis": (
            "A staged pipeline that separates literature summary, gap identification, question generation, "
            "methodology design, and evaluation will produce more coherent proposals than a single prompt."
        ),
        "methodology": (
            "Build a lightweight prototype, retrieve a small set of arXiv papers, generate outputs through "
            "independent prompt templates, and compare the final proposal with a single-prompt baseline using "
            "a manual rubric and GPT-assisted self-evaluation."
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
