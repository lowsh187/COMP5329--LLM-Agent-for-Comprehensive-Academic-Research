from app.models import Evaluation


def fallback_stage_outputs(topic: str) -> dict[str, object]:
    return {
        "summary": (
            f"Existing literature related to {topic} suggests active interest in applying LLMs "
            "to support academic or educational workflows, but reported outcomes depend on task design, "
            "evaluation criteria, and quality of retrieved evidence."
        ),
        "literature_tension_graph": (
            "Fallback tension graph: the literature often treats staged reasoning, retrieval grounding, and "
            "proposal quality as separately useful, but it remains unresolved whether staged reasoning improves "
            "research-depth judgments rather than only making outputs more complete or fluent."
        ),
        "gap": (
            "A key gap is the limited comparison between direct single-prompt generation and a structured, "
            "literature-augmented multi-stage workflow."
        ),
        "supporting_paper_ids": "",
        "evidence_reasoning": "Fallback output has no validated paper-level evidence links.",
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
        "research_problem_validity_check": (
            "The fallback problem is feasible and testable, but its PhD-level value depends on framing the work as "
            "a study of how staged reasoning changes research-depth judgments rather than as another proposal generator."
        ),
        "research_problem_blueprint": {
            "single_sentence_thesis": "A compact staged LLM workflow should be tested as a research-problem formation aid, not just a proposal-writing aid.",
            "core_gap": "It is unclear whether staged LLM reasoning improves research-depth judgments beyond surface completeness.",
            "literature_tension": "Prior work values retrieval and staged prompting separately, but their effect on PhD-level proposal quality is unresolved.",
            "phd_level_contribution": "A controlled evaluation of which reasoning stages improve genuine research-problem formation.",
            "falsifiable_hypothesis": "A compact staged workflow will improve rubric-rated research depth over a direct baseline.",
            "mechanism": "Structured intermediate constraints reduce generic gap framing and force alignment among gap, hypothesis, method, and contribution.",
            "strongest_baseline": "A direct single-prompt baseline using the same topic and evaluator.",
            "negative_result_meaning": "A null result would suggest staged decomposition improves readability more than research insight.",
            "score_limiting_risk": "The work may look like workflow engineering rather than a contribution to research-reasoning evaluation.",
            "minimum_evidence_required": ["Controlled condition comparison", "Rubric evidence for research-depth gains"],
            "proposal_must_include": ["A falsifiable comparison", "Explicit boundaries on claims"],
            "proposal_must_avoid": ["Overclaiming autonomous research ability", "Treating polished writing as research insight"],
        },
        "blueprint_compliance_check": {
            "complies": True,
            "missing_blueprint_element": "none",
            "score_limiting_risk_status": "Fallback proposal should avoid overclaiming autonomous research ability.",
            "one_sentence_repair": "Keep the proposal framed as a controlled evaluation of research-problem formation.",
        },
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
        "methodology_supporting_paper_ids": "",
        "baseline_reasoning": "Fallback output uses a direct single-prompt baseline as the minimum comparison.",
        "novelty_contribution": (
            "The fallback contribution is methodological: it decomposes proposal generation into inspectable "
            "research-reasoning stages and compares that workflow against a raw single-prompt baseline."
        ),
        "contribution_type_router": (
            "Primary contribution type: evaluation framework with a methodological component. The proposal should "
            "emphasize how staged reasoning is measured and avoid claiming a broad autonomous research system."
        ),
        "phd_contribution_design": (
            "The fallback PhD-level contribution design is to test whether explicit intermediate reasoning "
            "stages improve proposal quality beyond surface fluency, using controlled comparisons, rubric "
            "scores, and failure analysis to separate research-depth gains from formatting gains."
        ),
        "pre_proposal_critique": (
            "The strongest fallback reviewer objection is that the contribution could remain a workflow "
            "engineering demonstration rather than a research contribution; the proposal must therefore "
            "define what staged reasoning changes, how this is measured, and what result would falsify the claim."
        ),
        "failure_analysis": (
            "The workflow may fail when retrieval is weak, when the LLM overstates novelty, or when the evaluator "
            "rewards polished writing over genuine research contribution."
        ),
        "bad_proposal_pattern_detector": (
            "Fallback detector: likely risks include generic gap framing, method-first reasoning, weak baseline "
            "definition, and claims that overstate research contribution beyond the available evidence."
        ),
        "evidence_claim_alignment": (
            "Fallback alignment: core claims require explicit evidence links before they should be treated as "
            "strongly supported; unsupported novelty or feasibility claims should be narrowed."
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
