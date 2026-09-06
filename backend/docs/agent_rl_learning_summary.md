# Agent 技术说明 · Agent Technical Notes

[中文](#chinese) | [English](#english)

<a id="chinese"></a>
## 中文

[Switch to English](#english)

本文对应 `agent-rl-learning` 分支的当前实现。安装、环境变量、团队分工与 API 请求示例统一见[主 README](../../README.md#chinese)。本文说明实现机制，不报告当前版本的实验成绩。

### 1. 架构与范围

系统以 Python 显式编排检索、研究生成、检查、修订和评价。学习发生在 Agent 的外部执行策略层，不更新 LLM 参数，也不使用模型微调。普通研究请求按照显式参数运行；训练接口才负责选择策略并更新学习状态。

| 模块 | 职责 |
| --- | --- |
| [pipeline.py](../app/pipeline.py) | 提示链、候选选择、蓝图、修订和评价 |
| [agent_rl.py](../app/agent_rl.py) | 策略定义、选择、奖励、弱项反馈及持久化 |
| [agent_training.py](../app/agent_training.py) | 训练 episode 的执行和策略更新 |
| [literature_clustering.py](../app/literature_clustering.py) | 向量重排、聚类和候选语义对齐 |
| [experiment.py](../app/experiment.py) | 重复条件对比及质量、稳定性和成本汇总 |
| [models.py](../app/models.py) | API 与策略的数据结构 |

### 2. 学习闭环

```text
加载策略并按需利用历史记录热启动
  -> 选择执行策略
  -> 运行研究流程
  -> 评价质量并收集成本与回退指标
  -> 提取弱项标签并更新关注权重
  -> 更新策略统计并探索变体
  -> 继续训练并保存策略
```

算法名称为 `epsilon_greedy_quality_max_bandit`。未探索策略优先；其余选择结合 epsilon 探索和质量优先排序。统计包括 `best_quality`、`last_quality`、平均质量、奖励、token、耗时和回退率。排序强调已观察的高质量结果，并考虑目标策略加成、探索及较小的成本影响；它不是仅按平均奖励选择的标准 bandit 实验。

奖励记录质量、目标接近程度、质量纪录提升、完整度，以及 token、延迟和回退惩罚。代码中的 `85` 是内部目标参数，不是已达到的质量承诺。奖励细节以 `compute_reward` 为准。

### 3. 策略配置

```json
{
  "strategy_id": "multi_stage_p5_judgement_light",
  "condition": "multi_stage_with_retrieval",
  "max_papers": 5,
  "prompt_profile": "compact_85",
  "pipeline_mode": "judgement_light",
  "cost_profile": "judgement_light",
  "target_revision": true
}
```

当前内置策略：

- `multi_stage_p3_balanced`
- `multi_stage_p5_quality`
- `multi_stage_p5_compact_85`
- `multi_stage_p5_judgement_light`
- `multi_stage_p5_judgement_light_target_85_phd`
- `multi_stage_no_retrieval_p0`
- `group_selection_p3_balanced`
- `group_selection_p5_quality`

策略可按运行反馈产生文献数量、提示风格或修订配置变体。轻量模式会限制部分文献扩展，并将轻量策略归一到规范 ID，以减少重复策略。已有策略加载时也会进行归一化，因此实际配置应通过 `GET /api/agent/policy` 查看，不能只依赖策略名称推断。

`baseline_with_retrieval` 是历史名称：当前实际为无检索的单提示词基线，不在内置训练策略列表中。`multi_stage_without_retrieval` 同样不把文献传入研究生成流程。

### 4. 评分关注权重

系统使用七个维度：`clarity`、`logic`、`novelty`、`feasibility`、`literature_alignment`、`phd_level_quality`、`presentation`。每次训练从评分和弱项标签更新权重，再把优先关注维度注入下一次提示。

这是一种外部提示控制机制，不是 Transformer 内部 attention。`phd_level_quality` 是内部 rubric 标签，不代表专家认证。

提示风格在 `PROMPT_PROFILE_GUIDANCE` 中定义，包括 `strict_phd`、`target_85_phd`、`compact_85`、`novelty_focused`、`methodology_focused`、`literature_grounded`、`causal_rigor`、`cost_saving` 和 `candidate_selection`。

### 5. 研究蓝图与轻量模式

研究生成包含文献主题与证据、学术定位与张力分析、问题有效性检查、机制与假设、方法与贡献设计，以及 proposal 生成。候选选择模式在缺口、假设和方法阶段比较多个方案。

[研究蓝图模板](../prompts/research_problem_blueprint.txt)将研究判断转成结构化约束，包括机制、最强基线、负结果意义、必要证据，以及必须包含或避免的内容。蓝图已接入普通多阶段与候选选择流程。

| 环节 | `full` | `judgement_light` |
| --- | --- | --- |
| 研究判断与结构化蓝图 | 保留 | 保留 |
| 写作前批评 | 执行 | 跳过 |
| 不良模式和证据主张检查 | 执行 | 跳过 |
| 通用草稿批评与修订 | 执行 | 跳过 |
| 蓝图遵循检查 | 不执行 | 执行 |
| 目标诊断与修订 | 按 `target_revision` | 按 `target_revision` |
| 逻辑图和最终评价 | 执行 | 执行 |

上述比较适用于多阶段及候选选择流程，不适用于单提示词基线。轻量模式的目标是减少冗余检查并保留核心研究判断；实现完成并不证明质量提升或成本降低。

### 6. 运行与持久化

- `POST /api/agent/train`：执行训练并保存策略
- `GET /api/agent/policy`：查看策略统计与关注权重
- `backend/data/agent_policy.json`：策略状态
- `backend/data/runs/`：逐次研究输出
- `backend/data/retrieval_snapshots/`：文献快照

`warm_start` 默认从最多 60 条历史运行补充统计，并过滤超过信任上限的历史评分。该过滤是启发式清理，不等于验证评分可信度。`warm_start: false` 仍会加载已有策略，不会重置训练。

可通过 `strategy_ids` 指定已有策略；训练按给定列表轮换，此时不使用自动 bandit 选择。主题也按列表轮换，设计对比时应确保各策略覆盖相同主题，避免把主题差异误当策略收益。

### 7. 验证边界

评价由模型及内部 rubric 产生，应结合原文证据和人工审阅解释。检索缓存、局部 embedding 回退、生成回退和历史策略状态都可能影响比较。`metrics.fallback_reason` 不能覆盖所有局部降级，需要同时查看日志和论文来源。

---

<a id="english"></a>
## English

[切换到中文](#chinese)

This document describes the current implementation on `agent-rl-learning`. See the [main README](../../README.md) for setup, environment variables, team attribution, and API examples. This document explains mechanisms rather than reporting current benchmark results.

### 1. Architecture and scope

Python explicitly orchestrates retrieval, research generation, checks, revision, and evaluation. Learning operates on external execution strategies without updating or fine-tuning LLM weights. Ordinary research requests follow explicit parameters; the training endpoint selects strategies and updates learning state.

| Module | Responsibility |
| --- | --- |
| [pipeline.py](../app/pipeline.py) | Prompt chains, candidate selection, blueprints, revision, and evaluation |
| [agent_rl.py](../app/agent_rl.py) | Strategy definitions, selection, rewards, weakness feedback, and persistence |
| [agent_training.py](../app/agent_training.py) | Learning episode execution and policy updates |
| [literature_clustering.py](../app/literature_clustering.py) | Vector reranking, clustering, and candidate alignment |
| [experiment.py](../app/experiment.py) | Repeated comparisons and quality, stability, and cost summaries |
| [models.py](../app/models.py) | API and policy data structures |

### 2. Learning loop

```text
Load policy / optional historical warm start
  -> Select a strategy
  -> Run the research pipeline
  -> Evaluate quality and collect cost/fallback metrics
  -> Derive weakness tags and update attention weights
  -> Update strategy statistics and explore variants
  -> Continue episodes and save policy
```

The algorithm is named `epsilon_greedy_quality_max_bandit`. Unexplored strategies are prioritized; subsequent selection combines epsilon exploration with quality-oriented ranking. Statistics track best, latest, and mean quality, rewards, tokens, latency, and fallback rates. Ranking emphasizes high observed quality with target-strategy bonuses, exploration, and small cost effects, rather than selecting solely by mean reward.

Rewards record quality, target proximity, improvements over previous records, completeness, and token, latency, and fallback penalties. The code's `85` is an internal target parameter, not an achieved quality guarantee. See `compute_reward` for the exact calculation.

### 3. Strategy configuration

```json
{
  "strategy_id": "multi_stage_p5_judgement_light",
  "condition": "multi_stage_with_retrieval",
  "max_papers": 5,
  "prompt_profile": "compact_85",
  "pipeline_mode": "judgement_light",
  "cost_profile": "judgement_light",
  "target_revision": true
}
```

Built-in strategies:

- `multi_stage_p3_balanced`
- `multi_stage_p5_quality`
- `multi_stage_p5_compact_85`
- `multi_stage_p5_judgement_light`
- `multi_stage_p5_judgement_light_target_85_phd`
- `multi_stage_no_retrieval_p0`
- `group_selection_p3_balanced`
- `group_selection_p5_quality`

Feedback can generate variants in retrieval depth, prompt profile, or revision settings. Lightweight mode restricts some retrieval expansion and canonicalizes lightweight strategy IDs to reduce duplicates. Existing policies are also normalized on load; inspect `GET /api/agent/policy` rather than inferring configuration solely from a strategy name.

`baseline_with_retrieval` is a legacy name: it currently runs a single-prompt baseline without retrieval and is absent from the built-in training strategy list. `multi_stage_without_retrieval` likewise supplies no literature to generation.

### 4. Rubric attention weights

The seven dimensions are `clarity`, `logic`, `novelty`, `feasibility`, `literature_alignment`, `phd_level_quality`, and `presentation`. Training updates weights from scores and weakness tags, then injects priority dimensions into subsequent prompts.

This is external prompt control, not internal Transformer attention. `phd_level_quality` is an internal rubric label, not expert certification.

Prompt profiles are defined in `PROMPT_PROFILE_GUIDANCE`: `strict_phd`, `target_85_phd`, `compact_85`, `novelty_focused`, `methodology_focused`, `literature_grounded`, `causal_rigor`, `cost_saving`, and `candidate_selection`.

### 5. Research blueprints and lightweight mode

Generation covers literature themes and evidence, scholarly positioning and tensions, problem validity, mechanisms and hypotheses, methodology and contribution design, and proposal drafting. Candidate-selection mode compares alternatives at the gap, hypothesis, and methodology stages.

The [research blueprint template](../prompts/research_problem_blueprint.txt) turns research judgement into structured constraints covering mechanisms, the strongest baseline, negative-result interpretation, required evidence, and elements to include or avoid. Blueprints are integrated into both ordinary multi-stage and candidate-selection paths.

| Stage | `full` | `judgement_light` |
| --- | --- | --- |
| Research judgement and blueprint | Included | Included |
| Pre-proposal critique | Executed | Skipped |
| Bad-pattern and evidence–claim checks | Executed | Skipped |
| General draft critique and revision | Executed | Skipped |
| Blueprint compliance check | Not executed | Executed |
| Target diagnosis and revision | Controlled by `target_revision` | Controlled by `target_revision` |
| Logic graph and final evaluation | Executed | Executed |

This comparison applies to multi-stage and candidate-selection paths, not the single-prompt baseline. Lightweight mode aims to reduce redundant checks while retaining research judgement; implementation alone does not establish quality improvements or cost reductions.

### 6. Execution and persistence

- `POST /api/agent/train`: Run training and save the policy.
- `GET /api/agent/policy`: Inspect strategy statistics and attention weights.
- `backend/data/agent_policy.json`: Policy state.
- `backend/data/runs/`: Individual research outputs.
- `backend/data/retrieval_snapshots/`: Literature snapshots.

Warm start incorporates up to 60 historical runs by default and filters historical scores above a trust cap. This is heuristic cleaning, not score validation. `warm_start: false` still loads the existing policy and does not reset training.

Providing `strategy_ids` cycles through existing strategies instead of automatic bandit selection. Topics also cycle through their list; comparisons should give strategies the same topic coverage to avoid confusing topic differences with strategy gains.

### 7. Validation boundaries

Evaluation comes from a model and an internal rubric and should be interpreted alongside source evidence and human review. Retrieval caches, local embedding fallback, generation fallback, and historical policy state can affect comparisons. Inspect logs and paper sources as well as `metrics.fallback_reason`, which does not capture every local degradation.
