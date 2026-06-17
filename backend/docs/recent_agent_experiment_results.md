# Recent Agent Experiment Results

This document summarizes the recent agent-training experiments around the research-judgement pipeline, attention-guided prompt steering, and fallback fixes.

## Overall Conclusion

The latest valid run confirms that the new research-judgement modules are now connected without runtime fallback. However, the stronger pipeline did not improve proposal quality yet. Scores remain around 67-74, while token usage increased sharply to about 96k-102k tokens per episode.

The main finding is:

> More research-judgement stages made the pipeline more complete, but also heavier and less stable in quality. The next optimization should reduce stage overload rather than add more modules.

## Recent Experiment Summary

| Training ID | Main Change / State | Valid Episodes | Score Range | Avg Score | Max Score | Total Time | Total Tokens | Fallbacks | Main Observation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260527T210506Z` | Early dynamic profile + target revision run | 3 / 4 | 69-73 | 71.33 | 73 | 473.37 min | 245,018 | 1 | One API connection error caused extremely long latency and polluted reward statistics. |
| `20260528T080747Z` | Attention mechanism + compact_85 strategy active | 4 / 4 | 67-78 | 71.50 | 78 | 18.01 min | 327,015 | 0 | Attention worked, but p6-p8 expansion increased cost; only one topic reached 78. |
| `20260528T103531Z` | Added research-judgement modules, first integration attempt | 0 / 4 | N/A | N/A | N/A | 7.60 min | 162,320 | 4 | All episodes fell back due to `KeyError: 'literature_tension_graph'`. |
| `20260528T113253Z` | Fixed first missing field, second integration attempt | 0 / 4 | N/A | N/A | N/A | 7.19 min | 162,631 | 4 | All episodes fell back due to `KeyError: 'research_problem_validity_check'`. |
| `20260528T131848Z` | Missing context fields fixed; full research-judgement path valid | 4 / 4 | 67-74 | 70.25 | 74 | 21.42 min | 397,477 | 0 | Runtime is fixed, but quality decreased and token cost became too high. |

## Latest Valid Run: `20260528T131848Z`

This is the most important run because all four episodes completed without fallback.

| Episode | Topic | Selected Strategy | Papers | Score | Latency | Tokens | Weakness Tags | Fallback |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | Interpretable and Controllable LLMs for High-Stakes Decision-Making | `multi_stage_p5_compact_85` | 5 | 74 | 327.22s | 96,092 | `phd_level_quality` | None |
| 2 | Multi-Agent RL for Dynamic Resource Allocation in Cloud Computing | `multi_stage_p6_adaptive_literature_depth` | 6 | 73 | 310.71s | 101,771 | `phd_level_quality` | None |
| 3 | Robust AI Systems Against Data Poisoning and Adversarial Attacks | `multi_stage_p5_adaptive_cost_saving` | 5 | 67 | 315.27s | 98,827 | `novelty`, `phd_level_quality` | None |
| 4 | GNNs for Large-Scale Knowledge Discovery in Scientific Literature | `multi_stage_p6_adaptive_quality` | 6 | 67 | 331.94s | 100,787 | `novelty`, `phd_level_quality` | None |

Latest run totals:

- Average score: 70.25
- Best score: 74
- Total runtime: about 21 min 25 sec
- Total tokens: 397,477
- Fallback count: 0
- Final best action in policy: `multi_stage_no_retrieval_p0`

## What Improved

The recent fixes solved the major integration errors:

- `literature_tension_graph` is now available to downstream prompts.
- `research_problem_validity_check` is now available to downstream prompts.
- The full pipeline can complete without fallback.
- Attention weights are returned and updated correctly.
- The policy now records quality-max statistics such as `best_quality`, `last_quality`, `best_reward`, and `quality_improvement_count`.

The pipeline now includes the intended research-judgement modules:

- literature tension graph
- research problem validity check
- contribution type router
- bad proposal pattern detector
- evidence-claim alignment

## Current Problems

### 1. The pipeline became too expensive

The latest valid run used around 96k-102k tokens per episode. This is much higher than earlier runs, and each episode now takes about 5.2-5.5 minutes.

This suggests that the added stages are increasing reasoning volume, but not yet producing higher-quality final proposals.

### 2. More papers did not improve quality

The p6 and p7/p8-style expansions have not consistently improved scores. In the latest valid run:

- p5 scored 74 in episode 1.
- p6 scored 73 in episode 2.
- p5/p6 later dropped to 67.

So the current bottleneck is not paper quantity. It is likely stage coordination and research-problem formation.

### 3. Attention is focusing on the right weaknesses, but not enough

The final attention weights emphasized:

- `phd_level_quality`: about 0.297
- `novelty`: about 0.289

This is reasonable because the evaluator keeps identifying weak PhD-level quality and novelty. However, prompt-level focus alone is not enough to break the 76-78 ceiling.

### 4. The policy still prefers `multi_stage_no_retrieval_p0`

The best action remains `multi_stage_no_retrieval_p0`, mainly because historical results show stable scores near 77-78 with lower cost.

This does not mean no-retrieval is scientifically better. It means the current retrieval-heavy research-judgement pipeline is not yet converting retrieved evidence into better final scores.

## Interpretation

The recent experiments show a clear pattern:

1. The agent is no longer failing because of missing code paths.
2. The research-judgement modules are running.
3. The added modules increased cost and complexity.
4. Quality did not improve after the full-stage integration.

The system is now structurally stronger but operationally too heavy. It is doing more reasoning steps, but the final proposal may be diluted by too many intermediate checks and revisions.

## Recommended Next Change

The next experiment should introduce a lightweight research-judgement mode.

Keep:

- literature tension graph
- research problem validity check
- contribution type router
- compact proposal generation
- target revision
- final rubric evaluation

Temporarily disable or make optional:

- pre-proposal critique
- bad proposal pattern detector
- evidence-claim alignment
- p6/p7/p8 paper expansion

Suggested strategy:

```text
multi_stage_p5_judgement_light
condition: multi_stage_with_retrieval
max_papers: 5
prompt_profile: compact_85
target_revision: true
pipeline_mode: judgement_light
```

Expected benefit:

- lower token usage
- fewer conflicting revision signals
- better comparison against `multi_stage_no_retrieval_p0`
- clearer evidence about whether research-judgement modules help quality

## Short Takeaway

The latest run is a successful engineering fix but not yet a quality breakthrough. The current priority should be simplification and controlled comparison, not adding more stages or reading more papers.
