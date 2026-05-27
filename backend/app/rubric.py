from dataclasses import dataclass
from typing import Any

from app.models import Evaluation, RubricScore

FIELD_MAP = {
    "Research Clarity & Problem Definition": "clarity",
    "Logical Coherence & Multi-Stage Reasoning": "logic",
    "Novelty & Research Contribution": "novelty",
    "Feasibility & Experimental Design": "feasibility",
    "Literature Alignment & Scholarly Positioning": "literature_alignment",
    "PhD-Level Academic Depth": "phd_level_quality",
    "Presentation & Academic Communication": "presentation",
}


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    weight: int


RUBRIC_CRITERIA = [
    RubricCriterion("Research Clarity & Problem Definition", 15),
    RubricCriterion("Logical Coherence & Multi-Stage Reasoning", 15),
    RubricCriterion("Novelty & Research Contribution", 20),
    RubricCriterion("Feasibility & Experimental Design", 20),
    RubricCriterion("Literature Alignment & Scholarly Positioning", 15),
    RubricCriterion("PhD-Level Academic Depth", 10),
    RubricCriterion("Presentation & Academic Communication", 5),
]

STRICT_PHD_RUBRIC = """
Strict PhD-Level Research Proposal Evaluation Prompt

You are an extremely strict PhD-level research proposal reviewer.

Your task is to evaluate the following research proposal using the provided rubric criteria.

The evaluation standard must reflect:
- PhD candidacy proposal expectations
- Top-tier academic research standards
- Strong methodological rigor
- Genuine theoretical contribution
- Clear operationalization of concepts
- Scientifically defensible causal reasoning
- Publication-level scholarly positioning

DO NOT evaluate this as a normal coursework assignment or master-level proposal.

A proposal should ONLY receive high marks if it demonstrates:
- strong conceptual originality
- rigorous methodological grounding
- explicit theoretical mechanisms
- operationalizable constructs
- defensible causal claims
- realistic but impactful contribution to the field

Be highly critical of:
- vague novelty claims
- weak baselines
- unclear variable definitions
- missing theoretical framing
- purely engineering-oriented work without research contribution
- generic LLM applications lacking scientific insight
- unsupported causal assumptions
- shallow literature positioning
- missing failure analysis or limitations

High scores of 85 or above should be extremely rare and reserved for proposals approaching publishable PhD-level research quality.

Evaluation Instructions

For EACH criterion:
1. Give a numerical score strictly within the rubric range.
2. Explain WHY the proposal belongs in that score band.
3. Identify major weaknesses.
4. Explain what would be required to move into the next band.
5. Evaluate from a research perspective rather than an implementation perspective.

Avoid inflated praise.
Do not reward completeness alone.
Focus on depth, rigor, novelty, and scientific contribution.

Rubric

1. Research Clarity & Problem Definition (15)

Evaluate:
- clarity of research problem
- specificity of motivation
- precision of research question
- operational definition of key concepts
- clarity of research boundaries
- whether the proposal defines a genuine research problem rather than an application idea

Scoring guidance:
- 14-15: exceptionally precise, theoretically grounded, research-driven problem formulation
- 12-13: strong and clear research framing with minor conceptual weaknesses
- 9-11: understandable but partially broad or insufficiently operationalized
- 6-8: mostly descriptive or application-oriented framing
- 2-5: vague or weakly defined problem
- 0-1: no meaningful research problem

Pay special attention to whether concepts such as alignment, trust, safety, reasoning, fairness, interpretability, or therapeutic grounding are properly operationalized rather than used as buzzwords.

2. Logical Coherence & Multi-Stage Reasoning (15)

Evaluate:
- logical consistency across proposal stages
- coherence between problem -> gap -> hypothesis -> methodology -> evaluation
- whether assumptions are scientifically justified
- whether hypotheses follow logically from theory/literature
- whether causal claims are defensible
- whether the evaluation actually tests the stated hypothesis

Scoring guidance:
- 14-15: highly coherent end-to-end reasoning with strong scientific justification
- 12-13: mostly coherent with minor reasoning gaps
- 9-11: generally connected but some weak transitions or unsupported assumptions
- 6-8: several disconnected components or shallow justification
- 2-5: fragmented reasoning or inconsistent logic
- 0-1: incoherent structure

Critically assess whether the baseline truly isolates the claimed mechanism, whether the experiment actually validates the central research claim, and whether correlation is being mistaken for causation.

3. Novelty & Research Contribution (20)

Evaluate:
- conceptual novelty
- methodological novelty
- theoretical contribution
- whether the work advances understanding in the field
- whether the proposal merely combines existing techniques
- whether the contribution is scientifically meaningful

Important:
Combining "LLM + domain X" is NOT sufficient novelty.
Using RAG, prompting, fine-tuning, or standard evaluations alone is NOT strong research novelty.

High-scoring proposals should:
- introduce a new framework, theory, taxonomy, mechanism, or evaluation paradigm
- provide a new explanation of system behavior
- operationalize previously vague concepts
- meaningfully extend existing literature

Scoring guidance:
- 18-20: genuinely original and field-advancing contribution
- 15-17: clear novelty with meaningful research differentiation
- 12-14: moderate novelty but partially derivative
- 6-11: limited novelty; mostly recombination of existing ideas
- 3-5: highly derivative work
- 0-2: no identifiable contribution

Strongly penalize superficial "AI for X" framing, vague claims of innovation, and engineering improvements presented as scientific contribution.

4. Feasibility & Experimental Design (20)

Evaluate:
- realism of methodology
- appropriateness of baselines
- validity of evaluation metrics
- statistical defensibility
- quality of controls
- reproducibility
- risk awareness
- robustness of evaluation design
- handling of confounds
- ethics and safety procedures

Strong proposals should include:
- justified baselines
- clear ablations or comparison logic
- sample size reasoning
- operationalized variables
- reproducible methodology
- realistic deployment assumptions
- robust evaluation metrics

Scoring guidance:
- 18-20: highly rigorous, reproducible, and scientifically robust design
- 15-17: strong design with manageable limitations
- 12-14: feasible but methodologically incomplete
- 6-11: weak controls, weak baselines, or shallow evaluation
- 3-5: major feasibility or validity concerns
- 0-2: unrealistic or invalid design

Critically assess whether baselines are sufficiently strong, evaluations actually measure the claimed construct, metrics are appropriate proxies, confounding variables are controlled, and the proposal overclaims from self-report measures.

5. Literature Alignment & Scholarly Positioning (15)

Evaluate:
- depth of engagement with prior work
- positioning relative to literature
- identification of genuine gaps
- awareness of competing approaches
- understanding of limitations in prior work
- academic maturity of synthesis

High-level scholarly positioning should:
- compare methodologies rather than merely summarize papers
- explain why existing work is insufficient
- identify unresolved tensions or contradictions
- situate the work within broader research debates

Scoring guidance:
- 14-15: excellent scholarly positioning and critical synthesis
- 12-13: strong literature grounding with good comparative analysis
- 9-11: relevant literature used but somewhat surface-level
- 6-8: limited synthesis or weak gap justification
- 2-5: shallow or inaccurate literature usage
- 0-1: no meaningful literature grounding

Strongly penalize citation dumping, vague "few studies have explored" statements, and unsupported claims of research gaps.

6. PhD-Level Academic Depth (10)

Evaluate:
- theoretical sophistication
- abstraction level
- mechanism-level reasoning
- conceptual rigor
- operationalization depth
- awareness of limitations
- discussion of failure cases
- scientific maturity

This criterion separates strong coursework proposals from true PhD-level research proposals.

A strong PhD-level proposal should:
- explain WHY the mechanism should work
- define measurable constructs
- distinguish empirical correlation from causal interpretation
- discuss trade-offs and limitations
- articulate broader research implications
- move beyond implementation toward scientific understanding

Scoring guidance:
- 9-10: clear PhD-level conceptual and theoretical depth
- 7-8: approaching PhD-level with some shallow areas
- 5-6: strong master-level but limited theoretical depth
- 3-4: technically competent but conceptually shallow
- 1-2: mostly implementation-focused
- 0: no research depth

7. Presentation & Academic Communication (5)

Evaluate:
- clarity of writing
- academic tone
- structural organization
- precision of terminology
- conciseness
- readability
- professionalism

Scoring guidance:
- 5: publication-quality communication
- 4: strong academic communication
- 3: generally clear but inconsistent or repetitive
- 2: difficult to follow or imprecise
- 1: poorly written
- 0: unreadable

Final Output Format:
Provide criterion-by-criterion evaluation, numerical scores, total score, overall academic level classification, major strengths, major weaknesses, most important improvements required for true PhD-level quality, and final verdict on whether the proposal demonstrates genuine PhD-level research maturity.

Maintain a strict academic reviewing tone.
Do not inflate scores.
Assume the proposal is competing against serious PhD research proposals from strong research universities.
"""


def build_evaluation_prompt(
    *,
    topic: str,
    papers_context: str,
    stage_data: dict[str, str],
) -> str:
    return f"""
{STRICT_PHD_RUBRIC}

Research topic:
{topic}

Retrieved papers:
{papers_context}

Generated reasoning chain:

Summary:
{stage_data.get("summary", "")}

Scholarly positioning:
{stage_data.get("scholarly_positioning", "")}

Domain and methodology routing:
{stage_data.get("domain_router", "")}

Research scope:
{stage_data.get("research_scope", "")}

Concept definitions:
{stage_data.get("concept_definition", "")}

Technical feasibility:
{stage_data.get("technical_feasibility", "")}

Gap:
{stage_data.get("gap", "")}

Question:
{stage_data.get("question", "")}

Theoretical mechanism:
{stage_data.get("theoretical_mechanism", "")}

Hypothesis:
{stage_data.get("hypothesis", "")}

Operationalization and causal check:
{stage_data.get("operationalization_causal_check", "")}

Methodology:
{stage_data.get("methodology", "")}

Novelty and contribution:
{stage_data.get("novelty_contribution", "")}

Failure analysis:
{stage_data.get("failure_analysis", "")}

Proposal to evaluate:
{stage_data.get("proposal", "")}

Return valid JSON exactly in this structure:
{{
  "criterion_scores": [
    {{
      "criterion": "Research Clarity & Problem Definition",
      "score": 0,
      "level": "0-1: no meaningful research problem",
      "justification": "Why the proposal belongs in this score band.",
      "major_weaknesses": "Major weaknesses for this criterion.",
      "next_band_requirements": "What is required to move into the next score band."
    }}
  ],
  "total_score": 0,
  "academic_level": "Coursework-level",
  "major_strengths": ["..."],
  "major_weaknesses": ["..."],
  "phd_level_improvements": ["..."],
  "final_verdict": "Strict verdict on genuine PhD-level research maturity."
}}
"""


def evaluation_from_rubric_response(data: dict[str, Any]) -> Evaluation:
    by_name = {criterion.name: criterion for criterion in RUBRIC_CRITERIA}
    raw_scores = data.get("criterion_scores")
    if not isinstance(raw_scores, list):
        raise ValueError("LLM response missing criterion_scores")

    scores: list[RubricScore] = []
    seen: set[str] = set()
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        criterion_name = str(item.get("criterion") or "").strip()
        criterion = by_name.get(criterion_name)
        if not criterion:
            continue
        score = coerce_float(item.get("score"))
        score = min(max(score, 0), criterion.weight)
        justification = format_criterion_feedback(item)
        scores.append(
            RubricScore(
                criterion=criterion.name,
                weight=criterion.weight,
                score=round(score, 3),
                level=str(item.get("level") or infer_level(criterion.name, score)),
                justification=justification,
            )
        )
        seen.add(criterion.name)

    for criterion in RUBRIC_CRITERIA:
        if criterion.name not in seen:
            scores.append(
                RubricScore(
                    criterion=criterion.name,
                    weight=criterion.weight,
                    score=0,
                    level=infer_level(criterion.name, 0),
                    justification="No score returned for this criterion.",
                )
            )

    weighted_total = round(sum(score.score for score in scores), 3)
    field_values = {
        FIELD_MAP[score.criterion]: normalized_score(score.score, score.weight)
        for score in scores
        if score.criterion in FIELD_MAP
    }

    return Evaluation(
        clarity=field_values["clarity"],
        logic=field_values["logic"],
        novelty=field_values["novelty"],
        feasibility=field_values["feasibility"],
        literature_alignment=field_values["literature_alignment"],
        phd_level_quality=field_values["phd_level_quality"],
        presentation=field_values["presentation"],
        weighted_total=weighted_total,
        rubric_scores=scores,
        comments=overall_comments(data, weighted_total),
    )


def format_criterion_feedback(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("justification") or "").strip(),
        prefixed("Major weaknesses", item.get("major_weaknesses")),
        prefixed("Next band requirements", item.get("next_band_requirements")),
    ]
    return " ".join(part for part in parts if part)


def overall_comments(data: dict[str, Any], weighted_total: float) -> str:
    academic_level = str(data.get("academic_level") or classify_level(weighted_total))
    strengths = join_list(data.get("major_strengths"))
    weaknesses = join_list(data.get("major_weaknesses"))
    improvements = join_list(data.get("phd_level_improvements"))
    verdict = str(data.get("final_verdict") or "").strip()
    parts = [
        f"Total score: {weighted_total}/100.",
        f"Academic level: {academic_level}.",
        prefixed("Major strengths", strengths),
        prefixed("Major weaknesses", weaknesses),
        prefixed("Required improvements for true PhD-level quality", improvements),
        prefixed("Final verdict", verdict),
    ]
    return " ".join(part for part in parts if part)


def prefixed(label: str, value: Any) -> str:
    text = str(value or "").strip()
    return f"{label}: {text}" if text else ""


def join_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def normalized_score(score: float, weight: int) -> int:
    if weight <= 0:
        return 1
    return min(max(round((score / weight) * 4 + 1), 1), 5)


def coerce_float(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def classify_level(score: float) -> str:
    if score >= 85:
        return "Near publication-level"
    if score >= 75:
        return "Strong PhD-level"
    if score >= 65:
        return "Early PhD-level"
    if score >= 50:
        return "Strong Master-level"
    return "Coursework-level"


def infer_level(criterion_name: str, score: float) -> str:
    if criterion_name == "Novelty & Research Contribution":
        if score >= 18:
            return "18-20: genuinely original and field-advancing contribution"
        if score >= 15:
            return "15-17: clear novelty with meaningful research differentiation"
        if score >= 12:
            return "12-14: moderate novelty but partially derivative"
        if score >= 6:
            return "6-11: limited novelty; mostly recombination of existing ideas"
        if score >= 3:
            return "3-5: highly derivative work"
        return "0-2: no identifiable contribution"
    if criterion_name == "Feasibility & Experimental Design":
        if score >= 18:
            return "18-20: highly rigorous, reproducible, and scientifically robust design"
        if score >= 15:
            return "15-17: strong design with manageable limitations"
        if score >= 12:
            return "12-14: feasible but methodologically incomplete"
        if score >= 6:
            return "6-11: weak controls, weak baselines, or shallow evaluation"
        if score >= 3:
            return "3-5: major feasibility or validity concerns"
        return "0-2: unrealistic or invalid design"
    if criterion_name == "PhD-Level Academic Depth":
        if score >= 9:
            return "9-10: clear PhD-level conceptual and theoretical depth"
        if score >= 7:
            return "7-8: approaching PhD-level with some shallow areas"
        if score >= 5:
            return "5-6: strong master-level but limited theoretical depth"
        if score >= 3:
            return "3-4: technically competent but conceptually shallow"
        if score >= 1:
            return "1-2: mostly implementation-focused"
        return "0: no research depth"
    if criterion_name == "Presentation & Academic Communication":
        if score >= 5:
            return "5: publication-quality communication"
        if score >= 4:
            return "4: strong academic communication"
        if score >= 3:
            return "3: generally clear but inconsistent or repetitive"
        if score >= 2:
            return "2: difficult to follow or imprecise"
        if score >= 1:
            return "1: poorly written"
        return "0: unreadable"
    if score >= 14:
        return "14-15: exceptional or highly coherent PhD-level performance"
    if score >= 12:
        return "12-13: strong performance with minor weaknesses"
    if score >= 9:
        return "9-11: understandable but partially broad or underdeveloped"
    if score >= 6:
        return "6-8: shallow, descriptive, or weakly justified"
    if score >= 2:
        return "2-5: vague, weak, or fragmented"
    return "0-1: no meaningful evidence for this criterion"
