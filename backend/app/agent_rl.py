import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models import (
    AgentActionStats,
    AgentPolicyState,
    AgentRewardBreakdown,
    AgentStrategy,
    IndependentExperimentCondition,
    ResearchResult,
)
from app.storage import DATA_DIR, RUNS_DIR, read_json, write_json

AGENT_ACTIONS: tuple[IndependentExperimentCondition, ...] = (
    "baseline_with_retrieval",
    "multi_stage_with_retrieval",
    "multi_stage_without_retrieval",
    "multi_stage_group_selection",
)

AGENT_STRATEGIES: tuple[AgentStrategy, ...] = (
    AgentStrategy(
        strategy_id="multi_stage_p3_balanced",
        condition="multi_stage_with_retrieval",
        max_papers=3,
        description="Multi-stage generation with a smaller retrieval set to reduce cost.",
        prompt_profile="strict_phd",
        cost_profile="balanced",
    ),
    AgentStrategy(
        strategy_id="multi_stage_p5_quality",
        condition="multi_stage_with_retrieval",
        max_papers=5,
        description="Full multi-stage generation with retrieval and refinement.",
        prompt_profile="strict_phd",
        cost_profile="quality",
    ),
    AgentStrategy(
        strategy_id="multi_stage_p5_compact_85",
        condition="multi_stage_with_retrieval",
        max_papers=5,
        description="Compact high-score strategy focused on one clear gap, contribution, hypothesis, baseline, and negative-result interpretation.",
        prompt_profile="compact_85",
        cost_profile="quality",
        target_revision=True,
    ),
    AgentStrategy(
        strategy_id="multi_stage_p5_judgement_light",
        condition="multi_stage_with_retrieval",
        max_papers=5,
        description="Lightweight research-judgement strategy using a structured problem blueprint without heavy post-proposal checks.",
        prompt_profile="compact_85",
        pipeline_mode="judgement_light",
        cost_profile="judgement_light",
        target_revision=True,
    ),
    AgentStrategy(
        strategy_id="multi_stage_p5_judgement_light_target_85_phd",
        condition="multi_stage_with_retrieval",
        max_papers=5,
        description="Lightweight research-judgement strategy with target-85 PhD prompt profile.",
        prompt_profile="target_85_phd",
        pipeline_mode="judgement_light",
        cost_profile="judgement_light",
        target_revision=True,
    ),
    AgentStrategy(
        strategy_id="multi_stage_no_retrieval_p0",
        condition="multi_stage_without_retrieval",
        max_papers=1,
        description="Multi-stage reasoning without retrieved literature context.",
        prompt_profile="strict_phd",
        cost_profile="medium_cost",
    ),
    AgentStrategy(
        strategy_id="group_selection_p3_balanced",
        condition="multi_stage_group_selection",
        max_papers=3,
        description="Candidate generation and selection with a smaller retrieval set.",
        prompt_profile="candidate_selection",
        cost_profile="balanced",
    ),
    AgentStrategy(
        strategy_id="group_selection_p5_quality",
        condition="multi_stage_group_selection",
        max_papers=5,
        description="Candidate generation and selection with fuller retrieval.",
        prompt_profile="candidate_selection",
        cost_profile="quality",
    ),
)

DEFAULT_STRATEGIES_BY_ID = {strategy.strategy_id: strategy for strategy in AGENT_STRATEGIES}
JUDGEMENT_LIGHT_COMPACT_ID = "multi_stage_p5_judgement_light"
JUDGEMENT_LIGHT_TARGET_ID = "multi_stage_p5_judgement_light_target_85_phd"

POLICY_PATH = DATA_DIR / "agent_policy.json"

DEFAULT_EPSILON = 0.2
QUALITY_TARGET_SCORE = 85.0
DEFAULT_WARM_START_MAX_RUNS = 60
WARM_START_MAX_TRUSTED_QUALITY = 90.0
QUALITY_REWARD_WEIGHT = 1.5
HIGH_SCORE_THRESHOLD = 75.0
HIGH_SCORE_BONUS_WEIGHT = 1.0
TARGET_SCORE_BONUS_WEIGHT = 2.0
QUALITY_RECORD_BONUS_WEIGHT = 5.0
QUALITY_RECORD_BONUS_CAP = 20.0
NEAR_TARGET_START = 78.0
NEAR_TARGET_BONUS_WEIGHT = 2.0
TARGET_GAP_PENALTY_WEIGHT = 0.4
TOKEN_PENALTY_WEIGHT = 0.0001
LATENCY_PENALTY_WEIGHT = 0.00005
COMPLETENESS_BONUS_WEIGHT = 5.0
FALLBACK_PENALTY = 3.0
ATTENTION_DIMENSIONS = (
    "clarity",
    "logic",
    "novelty",
    "feasibility",
    "literature_alignment",
    "phd_level_quality",
    "presentation",
)
DEFAULT_ATTENTION_WEIGHTS = {
    "clarity": 0.12,
    "logic": 0.15,
    "novelty": 0.2,
    "feasibility": 0.15,
    "literature_alignment": 0.13,
    "phd_level_quality": 0.2,
    "presentation": 0.05,
}
ATTENTION_DECAY = 0.85


class EpsilonGreedyAgent:
    def __init__(self, policy: AgentPolicyState | None = None, *, epsilon: float | None = None) -> None:
        self.policy = policy or new_policy(epsilon=epsilon if epsilon is not None else DEFAULT_EPSILON)
        if epsilon is not None:
            self.policy.epsilon = epsilon

    def select_strategy(self) -> AgentStrategy:
        unexplored = [strategy_id for strategy_id, stats in self.policy.actions.items() if stats.count == 0]
        if unexplored:
            return self.policy.strategies[prioritize_strategy_ids(unexplored, self.policy)[0]]
        if random.random() < self.policy.epsilon:
            candidate_ids = prioritize_strategy_ids(list(self.policy.actions), self.policy)
            return self.policy.strategies[random.choice(candidate_ids[: min(3, len(candidate_ids))])]
        return self.policy.strategies[best_action(self.policy)]

    def select_action(self) -> IndependentExperimentCondition:
        return self.select_strategy().condition

    def update(
        self,
        strategy_id: str,
        reward: AgentRewardBreakdown | float,
        *,
        total_tokens: float = 0.0,
        latency_ms: float = 0.0,
        fallback: bool = False,
    ) -> AgentPolicyState:
        if strategy_id not in self.policy.actions:
            raise ValueError(f"Unsupported agent strategy: {strategy_id}")
        reward_value = reward.reward if isinstance(reward, AgentRewardBreakdown) else reward
        quality_score = reward.quality_score if isinstance(reward, AgentRewardBreakdown) else 0.0
        stats = self.policy.actions[strategy_id]
        stats.count += 1
        stats.total_reward = round(stats.total_reward + reward_value, 6)
        stats.mean_reward = round(stats.total_reward / stats.count, 6)
        stats.last_reward = round(reward_value, 6)
        stats.best_reward = round(max(stats.best_reward or reward_value, reward_value), 6)
        stats.total_quality = round(stats.total_quality + quality_score, 6)
        stats.mean_quality = round(stats.total_quality / stats.count, 6)
        stats.last_quality = round(quality_score, 6)
        if quality_score > stats.best_quality:
            stats.best_quality = round(quality_score, 6)
            stats.quality_improvement_count += 1
        stats.total_tokens = round(stats.total_tokens + total_tokens, 6)
        stats.mean_tokens = round(stats.total_tokens / stats.count, 6)
        stats.total_latency_ms = round(stats.total_latency_ms + latency_ms, 6)
        stats.mean_latency_ms = round(stats.total_latency_ms / stats.count, 6)
        stats.fallback_count += 1 if fallback else 0
        stats.fallback_rate = round(stats.fallback_count / stats.count, 6)
        self.policy.episodes += 1
        self.policy.best_action = best_action(self.policy)
        self.policy.updated_at = utc_now()
        return self.policy

    def adapt_strategy(
        self,
        strategy: AgentStrategy,
        reward: AgentRewardBreakdown,
        *,
        total_tokens: float = 0.0,
        latency_ms: float = 0.0,
        fallback: bool = False,
        weakness_tags: list[str] | None = None,
    ) -> list[AgentStrategy]:
        generated: list[AgentStrategy] = []
        for candidate in propose_strategy_mutations(
            strategy,
            reward,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            fallback=fallback,
            weakness_tags=weakness_tags or [],
        ):
            if candidate.strategy_id in self.policy.strategies:
                continue
            canonical_id = canonical_judgement_light_strategy_id(candidate)
            if canonical_id != candidate.strategy_id:
                if canonical_id in self.policy.strategies:
                    continue
                candidate = DEFAULT_STRATEGIES_BY_ID[canonical_id]
            self.policy.strategies[candidate.strategy_id] = candidate
            self.policy.actions[candidate.strategy_id] = AgentActionStats()
            generated.append(candidate)
        if generated:
            self.policy.updated_at = utc_now()
        return generated

    def save(self, path: Path = POLICY_PATH) -> None:
        save_policy(self.policy, path)


def new_policy(*, epsilon: float = DEFAULT_EPSILON) -> AgentPolicyState:
    strategies = {strategy.strategy_id: strategy for strategy in AGENT_STRATEGIES}
    return AgentPolicyState(
        algorithm="epsilon_greedy_quality_max_bandit",
        epsilon=epsilon,
        strategies=strategies,
        actions={strategy_id: AgentActionStats() for strategy_id in strategies},
        attention_weights=dict(DEFAULT_ATTENTION_WEIGHTS),
        best_action=None,
        updated_at=utc_now(),
    )


def load_policy(path: Path = POLICY_PATH) -> AgentPolicyState:
    data = read_json(path)
    if not isinstance(data, dict):
        return new_policy()
    policy = AgentPolicyState.model_validate(data)
    policy.algorithm = "epsilon_greedy_quality_max_bandit"
    if not policy.strategies:
        policy.strategies = dict(DEFAULT_STRATEGIES_BY_ID)
    for strategy in AGENT_STRATEGIES:
        policy.strategies.setdefault(strategy.strategy_id, strategy)
    normalize_target_revision_flags(policy)
    migrated_actions: dict[str, AgentActionStats] = {}
    for action, stats in policy.actions.items():
        strategy_id = legacy_strategy_id(action)
        strategy_id = canonical_judgement_light_strategy_id(policy.strategies.get(strategy_id))
        if strategy_id in policy.strategies:
            if strategy_id in migrated_actions:
                merge_action_stats(migrated_actions[strategy_id], stats)
            else:
                migrated_actions[strategy_id] = stats
    policy.actions = migrated_actions
    normalize_judgement_light_strategies(policy)
    for strategy_id in policy.strategies:
        policy.actions.setdefault(strategy_id, AgentActionStats())
    normalize_quality_max_stats(policy)
    normalize_attention_weights(policy)
    policy.best_action = best_action(policy) if any(stats.count for stats in policy.actions.values()) else None
    return policy


def normalize_target_revision_flags(policy: AgentPolicyState) -> None:
    for strategy in policy.strategies.values():
        if strategy.prompt_profile in {"target_85_phd", "compact_85"}:
            strategy.target_revision = True
        if strategy.strategy_id in {JUDGEMENT_LIGHT_COMPACT_ID, JUDGEMENT_LIGHT_TARGET_ID}:
            strategy.pipeline_mode = "judgement_light"


def normalize_judgement_light_strategies(policy: AgentPolicyState) -> None:
    for strategy in AGENT_STRATEGIES:
        if strategy.strategy_id in {JUDGEMENT_LIGHT_COMPACT_ID, JUDGEMENT_LIGHT_TARGET_ID}:
            policy.strategies[strategy.strategy_id] = strategy
    remove_ids: list[str] = []
    for strategy_id, strategy in policy.strategies.items():
        canonical_id = canonical_judgement_light_strategy_id(strategy)
        if canonical_id != strategy_id:
            remove_ids.append(strategy_id)
    for strategy_id in remove_ids:
        policy.strategies.pop(strategy_id, None)


def canonical_judgement_light_strategy_id(strategy: AgentStrategy | None) -> str:
    if not strategy or strategy.pipeline_mode != "judgement_light":
        return strategy.strategy_id if strategy else ""
    if strategy.prompt_profile == "target_85_phd":
        return JUDGEMENT_LIGHT_TARGET_ID
    return JUDGEMENT_LIGHT_COMPACT_ID


def merge_action_stats(target: AgentActionStats, source: AgentActionStats) -> None:
    if source.count <= 0:
        return
    target.count += source.count
    target.total_reward = round(target.total_reward + source.total_reward, 6)
    target.mean_reward = round(target.total_reward / target.count, 6)
    target.last_reward = source.last_reward if source.last_reward is not None else target.last_reward
    rewards = [value for value in [target.best_reward, source.best_reward] if value is not None]
    target.best_reward = round(max(rewards), 6) if rewards else None
    target.total_quality = round(target.total_quality + source.total_quality, 6)
    target.mean_quality = round(target.total_quality / target.count, 6)
    target.last_quality = source.last_quality if source.last_quality is not None else target.last_quality
    target.best_quality = round(max(target.best_quality, source.best_quality), 6)
    target.quality_improvement_count += source.quality_improvement_count
    target.total_tokens = round(target.total_tokens + source.total_tokens, 6)
    target.mean_tokens = round(target.total_tokens / target.count, 6)
    target.total_latency_ms = round(target.total_latency_ms + source.total_latency_ms, 6)
    target.mean_latency_ms = round(target.total_latency_ms / target.count, 6)
    target.fallback_count += source.fallback_count
    target.fallback_rate = round(target.fallback_count / target.count, 6)


def normalize_quality_max_stats(policy: AgentPolicyState) -> None:
    for stats in policy.actions.values():
        if stats.best_quality <= 0 and stats.mean_quality > 0:
            stats.best_quality = stats.mean_quality
        if stats.last_quality is None and stats.mean_quality > 0:
            stats.last_quality = stats.mean_quality
        if stats.best_quality > WARM_START_MAX_TRUSTED_QUALITY:
            trusted_candidates = [
                value
                for value in [stats.last_quality, stats.mean_quality]
                if value is not None and 0 < value <= WARM_START_MAX_TRUSTED_QUALITY
            ]
            stats.best_quality = max(trusted_candidates) if trusted_candidates else 0.0
        if stats.best_reward is None and stats.last_reward is not None:
            stats.best_reward = stats.last_reward


def normalize_attention_weights(policy: AgentPolicyState) -> None:
    weights = dict(DEFAULT_ATTENTION_WEIGHTS)
    for dimension in ATTENTION_DIMENSIONS:
        value = numeric(policy.attention_weights.get(dimension), default=weights[dimension])
        weights[dimension] = max(0.01, value)
    total = sum(weights.values()) or 1.0
    policy.attention_weights = {
        dimension: round(weights[dimension] / total, 6)
        for dimension in ATTENTION_DIMENSIONS
    }


def save_policy(policy: AgentPolicyState, path: Path = POLICY_PATH) -> None:
    policy.updated_at = utc_now()
    write_json(path, policy.model_dump(mode="json"))


def best_action(policy: AgentPolicyState) -> str:
    explored_ids = [strategy_id for strategy_id, stats in policy.actions.items() if stats.count > 0]
    candidate_ids = explored_ids or list(policy.actions)
    ranked_ids = prioritize_best_action_ids(candidate_ids, policy)
    return ranked_ids[0]


def global_best_quality(policy: AgentPolicyState) -> float:
    return max((stats.best_quality for stats in policy.actions.values()), default=0.0)


def format_attention_guidance(attention_weights: dict[str, float]) -> str:
    if not attention_weights:
        attention_weights = DEFAULT_ATTENTION_WEIGHTS
    ranked = sorted(
        (
            (dimension, numeric(attention_weights.get(dimension), default=0.0))
            for dimension in ATTENTION_DIMENSIONS
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top = ranked[:3]
    focus_text = ", ".join(f"{label}={weight:.2f}" for label, weight in top)
    deprioritized = ", ".join(label for label, _ in ranked[-2:])
    return (
        "Agent attention focus: prioritize the highest-weight rubric dimensions in this run. "
        f"Current top attention weights are {focus_text}. "
        "Spend proposal detail and revision effort on these dimensions first; keep lower-weight dimensions "
        f"adequate but concise, especially {deprioritized}. "
        "Do not add generic content unless it directly improves the attention-focused rubric dimensions."
    )


def update_attention_weights(
    policy: AgentPolicyState,
    result: ResearchResult | dict[str, Any],
    weakness_tags: list[str],
) -> dict[str, float]:
    data = result.model_dump(mode="json") if isinstance(result, ResearchResult) else result
    evaluation = data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {}
    current = dict(policy.attention_weights or DEFAULT_ATTENTION_WEIGHTS)
    next_weights = {
        dimension: max(0.01, numeric(current.get(dimension), default=DEFAULT_ATTENTION_WEIGHTS[dimension]) * ATTENTION_DECAY)
        for dimension in ATTENTION_DIMENSIONS
    }
    for dimension in ATTENTION_DIMENSIONS:
        score = numeric(evaluation.get(dimension), default=0.0)
        if score:
            next_weights[dimension] += max(0.0, 5.0 - score) * 0.08
    for tag in weakness_tags:
        if tag in next_weights:
            next_weights[tag] += 0.12
    weighted_total = numeric(evaluation.get("weighted_total"))
    if weighted_total and weighted_total < QUALITY_TARGET_SCORE:
        next_weights["phd_level_quality"] += 0.08
        next_weights["novelty"] += 0.05
    total = sum(next_weights.values()) or 1.0
    policy.attention_weights = {
        dimension: round(next_weights[dimension] / total, 6)
        for dimension in ATTENTION_DIMENSIONS
    }
    policy.updated_at = utc_now()
    return policy.attention_weights


def compute_reward(
    result: ResearchResult | dict[str, Any],
    *,
    previous_best_quality: float = 0.0,
) -> AgentRewardBreakdown:
    data = result.model_dump(mode="json") if isinstance(result, ResearchResult) else result
    evaluation = data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    quality_score = quality_from_evaluation(evaluation)
    total_tokens = numeric(metrics.get("total_tokens"))
    latency_ms = numeric(metrics.get("latency_ms"))
    completeness = numeric(metrics.get("required_field_completeness"), default=1.0)
    fallback_reason = metrics.get("fallback_reason")

    quality_reward = QUALITY_REWARD_WEIGHT * quality_score
    high_score_bonus = max(0.0, quality_score - HIGH_SCORE_THRESHOLD) * HIGH_SCORE_BONUS_WEIGHT
    target_bonus = max(0.0, quality_score - QUALITY_TARGET_SCORE) * TARGET_SCORE_BONUS_WEIGHT
    record_bonus = min(
        max(0.0, quality_score - previous_best_quality) * QUALITY_RECORD_BONUS_WEIGHT,
        QUALITY_RECORD_BONUS_CAP,
    )
    near_target_bonus = (
        25.0
        if quality_score >= QUALITY_TARGET_SCORE
        else max(0.0, quality_score - NEAR_TARGET_START) * NEAR_TARGET_BONUS_WEIGHT
    )
    target_gap_penalty = max(0.0, QUALITY_TARGET_SCORE - quality_score) * TARGET_GAP_PENALTY_WEIGHT
    token_penalty = TOKEN_PENALTY_WEIGHT * total_tokens if total_tokens else 0.0
    latency_penalty = LATENCY_PENALTY_WEIGHT * latency_ms if latency_ms else 0.0
    completeness_bonus = COMPLETENESS_BONUS_WEIGHT * completeness
    fallback_penalty = FALLBACK_PENALTY if fallback_reason else 0.0
    reward = (
        quality_reward
        + high_score_bonus
        + target_bonus
        + record_bonus
        + near_target_bonus
        - target_gap_penalty
        - token_penalty
        - latency_penalty
        + completeness_bonus
        - fallback_penalty
    )

    return AgentRewardBreakdown(
        quality_score=round(quality_score, 6),
        quality_reward=round(quality_reward, 6),
        high_score_bonus=round(high_score_bonus, 6),
        target_score_bonus=round(target_bonus, 6),
        quality_record_bonus=round(record_bonus, 6),
        near_target_bonus=round(near_target_bonus, 6),
        target_gap_penalty=round(target_gap_penalty, 6),
        token_penalty=round(token_penalty, 6),
        latency_penalty=round(latency_penalty, 6),
        completeness_bonus=round(completeness_bonus, 6),
        fallback_penalty=round(fallback_penalty, 6),
        reward=round(reward, 6),
    )


def quality_from_evaluation(evaluation: dict[str, Any]) -> float:
    weighted_total = numeric(evaluation.get("weighted_total"))
    if weighted_total:
        return weighted_total
    score_fields = [
        "clarity",
        "logic",
        "novelty",
        "feasibility",
        "literature_alignment",
        "phd_level_quality",
        "presentation",
    ]
    scores = [numeric(evaluation.get(field)) for field in score_fields if numeric(evaluation.get(field))]
    if not scores:
        return 0.0
    return (sum(scores) / len(scores) - 1.0) * 25.0


def warm_start_policy_from_runs(
    *,
    runs_dir: Path = RUNS_DIR,
    epsilon: float = DEFAULT_EPSILON,
    max_runs: int | None = None,
) -> AgentPolicyState:
    policy = new_policy(epsilon=epsilon)
    agent = EpsilonGreedyAgent(policy)
    consumed = 0
    run_paths = sorted(runs_dir.glob("*.json"))
    effective_max_runs = max_runs if max_runs is not None else DEFAULT_WARM_START_MAX_RUNS
    run_paths = run_paths[-effective_max_runs:]
    for path in run_paths:
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        action = data.get("condition")
        if action not in AGENT_ACTIONS:
            continue
        if action == "baseline_with_retrieval":
            continue
        strategy_id = strategy_id_from_run(data)
        quality_score = quality_from_evaluation(data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {})
        if is_untrusted_warm_start_quality(quality_score):
            continue
        reward_breakdown = compute_reward(data)
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        agent.update(
            strategy_id,
            reward_breakdown,
            total_tokens=numeric(metrics.get("total_tokens")),
            latency_ms=numeric(metrics.get("latency_ms")),
            fallback=bool(metrics.get("fallback_reason")),
        )
        consumed += 1
    policy.warm_started_from_runs = consumed
    policy.best_action = best_action(policy) if consumed else None
    policy.updated_at = utc_now()
    return policy


def is_untrusted_warm_start_quality(quality_score: float) -> bool:
    return quality_score <= 0 or quality_score > WARM_START_MAX_TRUSTED_QUALITY


def warm_start_and_save_policy(
    *,
    runs_dir: Path = RUNS_DIR,
    policy_path: Path = POLICY_PATH,
    epsilon: float = DEFAULT_EPSILON,
    max_runs: int | None = None,
) -> AgentPolicyState:
    policy = warm_start_policy_from_runs(runs_dir=runs_dir, epsilon=epsilon, max_runs=max_runs)
    save_policy(policy, policy_path)
    return policy


def numeric(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def baseline_used_literature(run_data: dict[str, Any]) -> bool:
    papers = run_data.get("papers")
    retrieval_snapshot_id = run_data.get("retrieval_snapshot_id")
    return bool(papers) or bool(retrieval_snapshot_id)


def strategy_id_from_run(run_data: dict[str, Any]) -> str:
    condition = str(run_data.get("condition") or "")
    paper_count = len(run_data.get("papers") or [])
    if condition == "multi_stage_without_retrieval":
        return "multi_stage_no_retrieval_p0"
    if condition == "multi_stage_group_selection":
        return "group_selection_p3_balanced" if paper_count <= 3 else "group_selection_p5_quality"
    if condition == "multi_stage_with_retrieval":
        return "multi_stage_p3_balanced" if paper_count <= 3 else "multi_stage_p5_quality"
    return "multi_stage_p5_quality"


def legacy_strategy_id(action_or_strategy_id: str) -> str:
    if action_or_strategy_id in DEFAULT_STRATEGIES_BY_ID:
        return action_or_strategy_id
    if action_or_strategy_id == "multi_stage_without_retrieval":
        return "multi_stage_no_retrieval_p0"
    if action_or_strategy_id == "multi_stage_group_selection":
        return "group_selection_p5_quality"
    if action_or_strategy_id == "multi_stage_with_retrieval":
        return "multi_stage_p5_quality"
    return action_or_strategy_id


def prioritize_strategy_ids(strategy_ids: list[str], policy: AgentPolicyState) -> list[str]:
    return sorted(
        strategy_ids,
        key=lambda strategy_id: strategy_priority_tuple(strategy_id, policy),
        reverse=True,
    )


def prioritize_best_action_ids(strategy_ids: list[str], policy: AgentPolicyState) -> list[str]:
    return sorted(
        strategy_ids,
        key=lambda strategy_id: best_action_priority_tuple(strategy_id, policy),
        reverse=True,
    )


def strategy_priority_tuple(strategy_id: str, policy: AgentPolicyState) -> tuple[float, float, float, float, float, float, str]:
    strategy = policy.strategies[strategy_id]
    stats = policy.actions[strategy_id]
    unexplored_priority = 1.0 if stats.count == 0 else 0.0
    return (
        unexplored_priority,
        quality_max_policy_score(strategy, stats),
        target_strategy_bonus(strategy),
        low_count_exploration_bonus(stats),
        -stats.fallback_rate,
        -tiny_cost_tiebreaker(stats),
        strategy_id,
    )


def best_action_priority_tuple(strategy_id: str, policy: AgentPolicyState) -> tuple[float, float, float, float, float, str]:
    strategy = policy.strategies[strategy_id]
    stats = policy.actions[strategy_id]
    return (
        stats.best_quality,
        stats.last_quality or 0.0,
        target_strategy_bonus(strategy),
        -stats.fallback_rate,
        -tiny_cost_tiebreaker(stats),
        strategy_id,
    )


def quality_max_policy_score(strategy: AgentStrategy, stats: AgentActionStats) -> float:
    best_quality = stats.best_quality or stats.mean_quality
    last_quality = stats.last_quality or stats.mean_quality
    near_target = 25.0 if best_quality >= QUALITY_TARGET_SCORE else max(0.0, best_quality - NEAR_TARGET_START) * 2.0
    return (
        best_quality * 100.0
        + last_quality * 10.0
        + near_target
        + target_strategy_bonus(strategy)
        + low_count_exploration_bonus(stats)
        - stats.fallback_rate * 20.0
        - tiny_cost_tiebreaker(stats)
    )


def target_strategy_bonus(strategy: AgentStrategy) -> float:
    bonus = 0.0
    if strategy.prompt_profile == "target_85_phd":
        bonus += 10.0
    if strategy.prompt_profile == "compact_85":
        bonus += 18.0
    if strategy.pipeline_mode == "judgement_light":
        bonus += 12.0
    if strategy.target_revision:
        bonus += 15.0
    return bonus


def low_count_exploration_bonus(stats: AgentActionStats) -> float:
    if stats.count == 0:
        return 30.0
    if stats.count == 1:
        return 10.0
    if stats.count == 2:
        return 5.0
    return 0.0


def tiny_cost_tiebreaker(stats: AgentActionStats) -> float:
    token_cost = (stats.mean_tokens or 0.0) / 100000.0
    latency_cost = (stats.mean_latency_ms or 0.0) / 1000000.0
    return token_cost + latency_cost


def propose_strategy_mutations(
    strategy: AgentStrategy,
    reward: AgentRewardBreakdown,
    *,
    total_tokens: float = 0.0,
    latency_ms: float = 0.0,
    fallback: bool = False,
    weakness_tags: list[str] | None = None,
) -> list[AgentStrategy]:
    candidates: list[AgentStrategy] = []
    weakness_tags = weakness_tags or []
    expensive_or_unstable = fallback or total_tokens > 45000 or latency_ms > 180000
    weak_quality = reward.quality_score and reward.quality_score < 68
    below_target = reward.quality_score and reward.quality_score < QUALITY_TARGET_SCORE
    judgement_light = strategy.pipeline_mode == "judgement_light"
    compact_full = strategy.prompt_profile == "compact_85" and strategy.pipeline_mode != "judgement_light"

    if strategy.condition in {"multi_stage_with_retrieval", "multi_stage_group_selection"}:
        if expensive_or_unstable and strategy.max_papers > 2 and not judgement_light:
            candidates.append(
                mutate_paper_count(
                    strategy,
                    max(2, strategy.max_papers - 1),
                    "adaptive_cost_saving",
                    "Reduced retrieval size after high cost, latency, or fallback.",
                )
            )
        if weak_quality and not fallback and strategy.max_papers < 8 and not judgement_light and not compact_full:
            candidates.append(
                mutate_paper_count(
                    strategy,
                    strategy.max_papers + 1,
                    "adaptive_quality",
                    "Increased retrieval size after weak quality feedback.",
                )
            )
        if (
            below_target
            and not fallback
            and strategy.max_papers < 7
            and strategy.prompt_profile == "target_85_phd"
            and not judgement_light
        ):
            candidates.append(
                mutate_paper_count(
                    strategy,
                    strategy.max_papers + 1,
                    "target_85_quality",
                    "Increased retrieval size to improve evidence coverage toward the 85-point target.",
                )
            )
        if (
            below_target
            and not fallback
            and strategy.max_papers < 6
            and strategy.prompt_profile != "target_85_phd"
            and not judgement_light
            and not compact_full
        ):
            candidates.append(
                mutate_paper_count(
                    strategy,
                    strategy.max_papers + 1,
                    "adaptive_literature_depth",
                    "Increased retrieval size after missing the 85-point target.",
                )
            )

    if strategy.condition == "multi_stage_without_retrieval" and weak_quality:
        candidates.append(
            AgentStrategy(
                strategy_id="multi_stage_p3_adaptive_literature",
                condition="multi_stage_with_retrieval",
                max_papers=3,
                description="Adaptive switch from no retrieval to compact retrieval after weak quality.",
                prompt_profile=strategy.prompt_profile,
                cost_profile="adaptive_literature",
                target_revision=strategy.target_revision,
            )
        )

    if strategy.condition == "multi_stage_group_selection" and expensive_or_unstable:
        candidates.append(
            AgentStrategy(
                strategy_id=f"multi_stage_p{max(2, strategy.max_papers - 1)}_adaptive_from_group",
                condition="multi_stage_with_retrieval",
                max_papers=max(2, strategy.max_papers - 1),
                description="Adaptive fallback from group selection to cheaper multi-stage retrieval.",
                prompt_profile="strict_phd",
                cost_profile="adaptive_cost_saving",
                target_revision=False,
            )
        )

    if should_try_compact_85(strategy, reward, weakness_tags):
        candidates.append(mutate_prompt_profile(strategy, "compact_85", weakness_tags))

    for profile in prompt_profiles_for_weaknesses(weakness_tags):
        if profile != strategy.prompt_profile:
            candidates.append(mutate_prompt_profile(strategy, profile, weakness_tags))

    return candidates


def mutate_paper_count(
    strategy: AgentStrategy,
    max_papers: int,
    cost_profile: str,
    reason: str,
) -> AgentStrategy:
    prefix = "group_selection" if strategy.condition == "multi_stage_group_selection" else "multi_stage"
    strategy_id = f"{prefix}_p{max_papers}_{cost_profile}"
    return AgentStrategy(
        strategy_id=strategy_id,
        condition=strategy.condition,
        max_papers=max_papers,
        description=reason,
        prompt_profile=strategy.prompt_profile,
        cost_profile=cost_profile,
        target_revision=strategy.target_revision,
        pipeline_mode=strategy.pipeline_mode,
    )


def mutate_prompt_profile(
    strategy: AgentStrategy,
    prompt_profile: str,
    weakness_tags: list[str],
) -> AgentStrategy:
    if strategy.pipeline_mode == "judgement_light":
        canonical_id = JUDGEMENT_LIGHT_TARGET_ID if prompt_profile == "target_85_phd" else JUDGEMENT_LIGHT_COMPACT_ID
        return DEFAULT_STRATEGIES_BY_ID[canonical_id]
    strategy_id = f"{strategy.strategy_id}_{prompt_profile}"
    return AgentStrategy(
        strategy_id=strategy_id,
        condition=strategy.condition,
        max_papers=strategy.max_papers,
        description=f"Adaptive prompt profile for weaknesses: {', '.join(weakness_tags)}.",
        prompt_profile=prompt_profile,
        cost_profile=strategy.cost_profile,
        target_revision=prompt_profile in {"target_85_phd", "compact_85"} or strategy.target_revision,
        pipeline_mode=strategy.pipeline_mode,
    )


def should_try_compact_85(
    strategy: AgentStrategy,
    reward: AgentRewardBreakdown,
    weakness_tags: list[str],
) -> bool:
    if strategy.prompt_profile == "compact_85":
        return False
    if strategy.condition == "baseline_with_retrieval":
        return False
    quality_gap = 0 < reward.quality_score < QUALITY_TARGET_SCORE
    relevant_weakness = any(tag in weakness_tags for tag in ["novelty", "phd_level_quality", "clarity", "logic"])
    return quality_gap and relevant_weakness


def prompt_profiles_for_weaknesses(weakness_tags: list[str]) -> list[str]:
    profiles: list[str] = []
    mapping = {
        "methodology": "methodology_focused",
        "feasibility": "methodology_focused",
        "novelty": "novelty_focused",
        "literature_alignment": "literature_grounded",
        "logic": "causal_rigor",
        "clarity": "causal_rigor",
        "phd_level_quality": "target_85_phd",
        "presentation": "cost_saving",
    }
    for tag in weakness_tags:
        profile = mapping.get(tag)
        if profile and profile not in profiles:
            profiles.append(profile)
    return profiles[:2]


def derive_weakness_tags(result: ResearchResult | dict[str, Any]) -> list[str]:
    data = result.model_dump(mode="json") if isinstance(result, ResearchResult) else result
    evaluation = data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {}
    score_fields = [
        "clarity",
        "logic",
        "novelty",
        "feasibility",
        "literature_alignment",
        "phd_level_quality",
        "presentation",
    ]
    scored_fields = {
        field: numeric(evaluation.get(field), default=0.0)
        for field in score_fields
        if numeric(evaluation.get(field), default=0.0) > 0
    }
    if not scored_fields:
        return ["phd_level_quality"]
    max_score = max(scored_fields.values())
    weak = [
        field
        for field, score in scored_fields.items()
        if score <= 3.0 or score <= max_score - 1.0
    ]
    weighted_total = numeric(evaluation.get("weighted_total"))
    if weighted_total and weighted_total < QUALITY_TARGET_SCORE and "phd_level_quality" not in weak:
        weak.append("phd_level_quality")
    priority = [
        "novelty",
        "feasibility",
        "methodology",
        "literature_alignment",
        "logic",
        "phd_level_quality",
        "clarity",
        "presentation",
    ]
    weak = sorted(set(weak), key=lambda tag: priority.index(tag) if tag in priority else len(priority))
    return weak[:4]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
