from datetime import datetime, timezone

from app.agent_rl import (
    EpsilonGreedyAgent,
    compute_reward,
    derive_weakness_tags,
    format_attention_guidance,
    global_best_quality,
    load_policy,
    save_policy,
    update_attention_weights,
    warm_start_policy_from_runs,
)
from app.models import (
    AgentRewardBreakdown,
    AgentStrategy,
    AgentTrainingEpisode,
    AgentTrainingRequest,
    AgentTrainingResult,
    ResearchRequest,
    ResearchResult,
)
from app.pipeline import run_research_pipeline


async def run_agent_training(request: AgentTrainingRequest) -> AgentTrainingResult:
    training_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = (
        warm_start_policy_from_runs(
            epsilon=request.epsilon if request.epsilon is not None else 0.2,
            max_runs=request.warm_start_max_runs,
        )
        if request.warm_start
        else load_policy()
    )
    if request.epsilon is not None:
        policy.epsilon = request.epsilon
    comparison_strategy_ids = request.strategy_ids or []
    unknown_strategy_ids = [strategy_id for strategy_id in comparison_strategy_ids if strategy_id not in policy.strategies]
    if unknown_strategy_ids:
        raise ValueError(f"Unknown agent strategy id(s): {', '.join(unknown_strategy_ids)}")

    agent = EpsilonGreedyAgent(policy)
    episodes: list[AgentTrainingEpisode] = []

    for episode_index in range(1, request.episodes + 1):
        topic = request.topics[(episode_index - 1) % len(request.topics)]
        if comparison_strategy_ids:
            strategy = policy.strategies[comparison_strategy_ids[(episode_index - 1) % len(comparison_strategy_ids)]]
        else:
            strategy = agent.select_strategy()
        attention_focus = dict(policy.attention_weights)
        result = await run_research_pipeline(
            ResearchRequest(
                topic=topic,
                max_papers=strategy.max_papers or request.max_papers,
                condition=strategy.condition,
                prompt_profile=strategy.prompt_profile,
                pipeline_mode=strategy.pipeline_mode,
                attention_guidance=format_attention_guidance(policy.attention_weights),
                target_revision=strategy.target_revision,
                experiment_id=training_id,
                run_index=episode_index,
            )
        )
        reward = compute_reward(result, previous_best_quality=global_best_quality(policy))
        metrics = result.metrics
        weakness_tags = derive_weakness_tags(result)
        update_attention_weights(policy, result, weakness_tags)
        total_tokens = metrics.total_tokens if metrics and metrics.total_tokens else 0.0
        latency_ms = metrics.latency_ms if metrics else 0.0
        fallback = bool(metrics and metrics.fallback_reason)
        policy = agent.update(
            strategy.strategy_id,
            reward,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            fallback=fallback,
        )
        generated_strategies = agent.adapt_strategy(
            strategy,
            reward,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            fallback=fallback,
            weakness_tags=weakness_tags,
        )
        episodes.append(
            build_episode_record(
                episode_index,
                topic,
                strategy,
                generated_strategies,
                weakness_tags,
                attention_focus,
                reward,
                result,
            )
        )

    save_policy(policy)
    return AgentTrainingResult(training_id=training_id, episodes=episodes, policy=policy)


def build_episode_record(
    episode_index: int,
    topic: str,
    strategy: AgentStrategy,
    generated_strategies: list[AgentStrategy],
    weakness_tags: list[str],
    attention_focus: dict[str, float],
    reward: AgentRewardBreakdown,
    result: ResearchResult,
) -> AgentTrainingEpisode:
    metrics = result.metrics
    return AgentTrainingEpisode(
        episode=episode_index,
        topic=topic,
        selected_strategy=strategy,
        selected_action=strategy.condition,
        generated_strategies=generated_strategies,
        weakness_tags=weakness_tags,
        attention_focus=attention_focus,
        reward=reward,
        weighted_total=result.evaluation.weighted_total,
        latency_ms=metrics.latency_ms if metrics else None,
        total_tokens=metrics.total_tokens if metrics else None,
        fallback_reason=metrics.fallback_reason if metrics else None,
    )
