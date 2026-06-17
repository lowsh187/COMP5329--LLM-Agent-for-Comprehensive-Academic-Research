# Agent RL Learning Summary

Branch: `agent-rl-learning`

This project implements agent-level reinforcement-style learning for an academic research proposal agent. It does not tune LLM parameters. Instead, it learns how to choose external agent strategies, prompt profiles, retrieval depth, target-revision behavior, and rubric attention focus.

## Goal

The current optimization goal is not average reward. The agent is designed to search for higher proposal quality scores, especially scores approaching or exceeding the strict PhD-level target of `85/100`.

The agent learns from each training episode:

- which strategy was selected
- which prompt profile was used
- how many papers were retrieved
- whether target revision was enabled
- final rubric score
- token and latency cost
- fallback/error status
- weakness tags from the evaluator
- attention weights over rubric dimensions

## Main Training Endpoint

Training endpoint:

```text
POST /api/agent/train
```

Example command from `backend`:

```cmd
..\frontend\.venv\Scripts\python.exe -c "import json, urllib.request; topics=json.load(open('data/agent_training_topics.json', encoding='utf-8')); data=json.dumps({'topics':topics,'episodes':4,'max_papers':5,'epsilon':0.2,'warm_start':True}).encode('utf-8'); req=urllib.request.Request('http://127.0.0.1:8000/api/agent/train', data=data, headers={'Content-Type':'application/json'}, method='POST'); print(urllib.request.urlopen(req, timeout=7200).read().decode('utf-8'))"
```

Policy endpoint:

```text
GET /api/agent/policy
```

## Policy Mechanism

The current policy algorithm is:

```text
epsilon_greedy_quality_max_bandit
```

It is an epsilon-greedy bandit over agent strategies, but the ranking is quality-max oriented.

The policy prioritizes:

1. `best_quality`
2. `last_quality`
3. target-oriented strategy bonus
4. low-count exploration bonus
5. fallback rate
6. tiny token/latency tie-breaker

This means the agent is encouraged to find a higher observed quality score, not merely a stable average reward.

## Strategy State

Each strategy contains:

```json
{
  "strategy_id": "multi_stage_p5_compact_85",
  "condition": "multi_stage_with_retrieval",
  "max_papers": 5,
  "description": "...",
  "prompt_profile": "compact_85",
  "cost_profile": "quality",
  "target_revision": true
}
```

The main default strategies are:

- `multi_stage_p3_balanced`
- `multi_stage_p5_quality`
- `multi_stage_p5_compact_85`
- `multi_stage_no_retrieval_p0`
- `group_selection_p3_balanced`
- `group_selection_p5_quality`

The agent can dynamically generate additional strategies, such as:

- paper-count mutations: `p4`, `p6`, `p7`, `p8`
- cost-saving variants
- target-85 variants
- prompt-profile variants
- compact-85 variants
- group-selection variants

## Reward Mechanism

Reward is still recorded, but it is not the main policy objective. Quality score is more important.

Reward currently includes:

- `quality_reward`
- `high_score_bonus`
- `target_score_bonus`
- `quality_record_bonus`
- `near_target_bonus`
- `target_gap_penalty`
- `token_penalty`
- `latency_penalty`
- `completeness_bonus`
- `fallback_penalty`

Important behavior:

- If a run exceeds the previous trusted global best quality, `quality_record_bonus` is added.
- Scores close to or above 85 receive additional target-related bonuses.
- Token and latency penalties are intentionally small relative to quality, but still visible.
- Fallbacks are penalized.

## Warm Start Cleaning

Warm start uses recent run history, but old unreliable high scores are filtered.

Current behavior:

- default warm start max runs: `60`
- trusted warm-start score cap: `90`
- scores above the trusted cap are ignored during warm start
- existing polluted policy values are normalized on load

This prevents earlier evaluator/fallback artifacts such as `100/100` from dominating the quality-max policy.

## Agent-Level Attention

The agent includes an external attention mechanism over rubric dimensions.

This is not transformer attention and does not modify LLM internals. It is a learned prompt-control signal.

Attention dimensions:

```json
{
  "clarity": 0.12,
  "logic": 0.15,
  "novelty": 0.20,
  "feasibility": 0.15,
  "literature_alignment": 0.13,
  "phd_level_quality": 0.20,
  "presentation": 0.05
}
```

After each episode, attention is updated from:

- low rubric scores
- weakness tags
- whether the total score is below 85

The next episode receives prompt guidance such as:

```text
Agent attention focus: prioritize the highest-weight rubric dimensions in this run.
Current top attention weights are novelty=0.20, phd_level_quality=0.20, logic=0.15.
Spend proposal detail and revision effort on these dimensions first.
Keep lower-weight dimensions adequate but concise.
Do not add generic content unless it directly improves the attention-focused rubric dimensions.
```

This gives the agent a persistent learning signal about what to improve next.

## Prompt Profiles

Prompt profiles are stored in `app/pipeline.py` as `PROMPT_PROFILE_GUIDANCE`.

Current profiles:

### `strict_phd`

General PhD-level standards:

- specificity
- theoretical depth
- methodological rigor
- realistic contribution claims

### `target_85_phd`

Explicitly optimizes for a strict score of 85 or above.

Focus:

- concrete novelty
- why prior work is insufficient
- mechanism-level explanation
- baselines
- ablations
- controls
- measurable variables
- data sources
- evaluation metrics
- failure cases
- literature tensions
- realistic claim boundaries

### `compact_85`

High-score compact strategy.

Focus:

- one precise literature gap
- one concrete contribution
- one falsifiable hypothesis
- one mechanism-level explanation
- one strongest baseline or ablation set
- one metric family
- one negative-result interpretation

Purpose:

- reduce clutter
- avoid overlong proposal paragraphs
- avoid citation dumping
- avoid generic AI-for-X claims

### `novelty_focused`

Focuses on non-generic research gap and clear contribution.

### `methodology_focused`

Focuses on:

- rigorous experimental design
- valid baselines
- reproducibility
- ablations
- measurable variables
- evaluation metrics

### `literature_grounded`

Focuses on:

- critical synthesis
- scholarly positioning
- explicit links between claims and evidence

### `causal_rigor`

Focuses on:

- operationalized constructs
- causal mechanisms
- confound control
- claim boundaries

### `cost_saving`

Focuses on concise stages and less redundant text.

### `candidate_selection`

Used in group-selection mode to generate and compare alternatives.

## Pipeline Conditions

The agent can run these conditions:

### `multi_stage_with_retrieval`

Uses retrieved literature and full multi-stage reasoning.

### `multi_stage_without_retrieval`

Runs multi-stage reasoning without retrieved papers.

### `multi_stage_group_selection`

Generates candidate outputs for selected stages and chooses among them.

### `baseline_with_retrieval`

Currently treated as a baseline/control style condition. It is not part of the agent learning strategy set.

## Main Multi-Stage Pipeline

For `multi_stage_with_retrieval`, the current stage order is:

1. literature search
2. literature clustering
3. evidence notes
4. literature summary
5. scholarly positioning
6. literature tension graph
7. domain router
8. research scope
9. research gap
10. concept definition
11. technical feasibility
12. research question
13. research problem validity check
14. theoretical mechanism
15. hypothesis
16. operationalization and causal check
17. methodology
18. novelty and contribution
19. contribution type router
20. PhD contribution design
21. pre-proposal critique
22. failure analysis
23. proposal draft
24. bad proposal pattern detector
25. evidence-claim alignment
26. proposal critique
27. proposal revision
28. target-85 rubric diagnosis, if `target_revision=true`
29. target-85 proposal revision, if `target_revision=true`
30. proposal logic graph
31. final rubric evaluation

## Group-Selection Pipeline

For `multi_stage_group_selection`, the agent uses candidate selection for:

- research gap
- hypothesis
- methodology

Then it continues through:

- novelty and contribution
- PhD contribution design
- pre-proposal critique
- failure analysis
- proposal
- proposal critique/revision
- optional target-85 diagnosis/revision
- logic graph
- final evaluation

## Prompt Files

Prompt files are in `backend/prompts`.

Current prompt files:

- `baseline.txt`
- `bad_proposal_pattern_detector.txt`
- `concept_definition.txt`
- `domain_router.txt`
- `evaluation.txt`
- `evidence_notes.txt`
- `evidence_claim_alignment.txt`
- `failure_analysis.txt`
- `hypothesis.txt`
- `literature_summary.txt`
- `literature_tension_graph.txt`
- `methodology.txt`
- `novelty_contribution.txt`
- `operationalization_causal_check.txt`
- `phd_contribution_design.txt`
- `pre_proposal_critique.txt`
- `proposal.txt`
- `proposal_critique.txt`
- `proposal_logic_graph.txt`
- `proposal_revision.txt`
- `research_gap.txt`
- `research_problem_validity_check.txt`
- `research_question.txt`
- `research_scope.txt`
- `scholarly_positioning.txt`
- `target_85_revision.txt`
- `technical_feasibility.txt`
- `theoretical_mechanism.txt`

## Research Judgement Modules

Three modules were added to shift the system from proposal-form optimization toward research-problem formation:

- `literature_tension_graph.txt`: extracts cross-paper tensions, contradictions, incompatible assumptions, unresolved mechanisms, and unstable evaluation conditions.
- `research_problem_validity_check.txt`: checks whether the question is non-trivial, theoretically meaningful, empirically testable, and worth PhD-level committee attention.
- `contribution_type_router.txt`: routes the proposal to one primary contribution type, such as theoretical, methodological, empirical, benchmark/dataset, system, or evaluation framework.
- `bad_proposal_pattern_detector.txt`: detects generic AI-proposal failure patterns, such as generic gap, method-first framing, overclaiming novelty, weak baseline, and citation decoration.
- `evidence_claim_alignment.txt`: checks whether core proposal claims are actually supported by retrieved evidence and identifies claims that must be narrowed, hedged, or better grounded.

These modules make the proposal path closer to:

```text
literature tension
-> valid research problem
-> contribution type
-> mechanism
-> hypothesis
-> methodology
-> proposal
```

## Important New Prompts

### `phd_contribution_design.txt`

Creates a contribution skeleton before the proposal is drafted.

It asks for:

- concrete theoretical or methodological contribution
- mechanism-level claim
- what literature cannot explain, measure, compare, or validate
- falsifiable scientific claim
- strongest baseline/control/ablation
- negative-result interpretation

### `pre_proposal_critique.txt`

Attacks the contribution design before drafting.

It asks:

- why the contribution might still fail to reach 85+
- what the three most damaging reviewer objections are
- what exact repairs are required before drafting

### `target_85_revision.txt`

Runs a final target-85 revision after a temporary rubric diagnosis.

It uses:

- strict rubric diagnosis
- current proposal
- contribution design
- pre-proposal critique
- target-85 requirements

## Weakness Tags

Weakness tags are derived from rubric scores and weighted total.

Possible tags:

- `clarity`
- `logic`
- `novelty`
- `feasibility`
- `literature_alignment`
- `phd_level_quality`
- `presentation`

These tags drive:

- prompt-profile mutation
- attention update
- strategy mutation
- compact-85 exploration
- target-85 exploration

## Current Observed Behavior

Recent training has shown:

- the agent often reaches `76-78`
- the trusted best score is around `78`
- larger retrieval depth does not reliably improve quality
- `p6/p7/p8` can increase cost without breaking the quality ceiling
- attention has shifted toward `phd_level_quality` and `novelty`
- `compact_85` is being explored, but has not yet exceeded the trusted best score

## Practical Recommendation

For the next experiments:

- avoid increasing papers beyond 5 unless needed
- compare stable p5 strategies
- watch `quality_record_bonus`
- watch `attention_weights`
- treat any score above 78 as important progress
- if score reaches 79+, inspect that run's saved JSON carefully

Useful strategies to compare:

- `multi_stage_p5_quality`
- `multi_stage_no_retrieval_p0`
- `multi_stage_p5_compact_85`
- `multi_stage_p5_compact_85_target_85_phd`
- `group_selection_p5_quality`

## How To Explain This In A Report

A concise description:

```text
The system implements reinforcement-style agent learning without updating LLM parameters. The agent learns an external policy over research-generation strategies, prompt profiles, retrieval depth, target-revision behavior, and rubric attention weights. After each episode, a strict PhD-level evaluator provides quality scores and weakness signals. These signals update the strategy bandit, quality-max policy state, and attention distribution over rubric dimensions. The next episode uses the learned policy to select a strategy and injects attention-guided prompt guidance into the pipeline.
```
