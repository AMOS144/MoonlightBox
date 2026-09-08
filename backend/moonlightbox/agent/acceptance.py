from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Literal

ReplaySource = Literal["historical_replay", "labeled_behavior"]


@dataclass(frozen=True)
class ReplayObservation:
    """一条具有明确来源和时间截断的回放观测。"""

    source: ReplaySource
    cutoff: datetime
    expected_express: bool
    predicted_express: bool
    extraction_succeeded: bool
    fact_safe: bool
    latency_ms: float
    direct_latency_ms: float | None = None
    context_continuity_score: float = 1.0
    persona_style_score: float = 1.0
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise ValueError("cutoff 必须包含时区")
        if self.latency_ms < 0:
            raise ValueError("latency_ms 不能为负数")
        if self.direct_latency_ms is not None and self.direct_latency_ms < 0:
            raise ValueError("direct_latency_ms 不能为负数")
        if not 0 <= self.context_continuity_score <= 1:
            raise ValueError("context_continuity_score 必须位于 0 到 1")
        if not 0 <= self.persona_style_score <= 1:
            raise ValueError("persona_style_score 必须位于 0 到 1")


@dataclass(frozen=True)
class AcceptanceEvaluation:
    """主体认知 Agent 的一次纯计算验收结果。"""

    sample_count: int
    extraction_success_rate: float
    fact_safety_rate: float
    expression_decision_accuracy: float
    context_continuity_score: float
    persona_style_score: float
    agent_p95_latency_ms: float
    direct_lora_p95_latency_ms: float
    passed: bool
    failure_reasons: tuple[str, ...]
    observations: tuple[ReplayObservation, ...]


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def _nearest_rank_p95(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * 0.95) - 1]


def evaluate_acceptance(
    observations: list[ReplayObservation] | tuple[ReplayObservation, ...],
    *,
    direct_lora_p95: float,
) -> AcceptanceEvaluation:
    """仅依据调用方显式提交的观测计算验收门槛。"""

    if direct_lora_p95 <= 0:
        raise ValueError("direct_lora_p95 必须大于零")

    frozen_observations = tuple(observations)
    sample_count = len(frozen_observations)
    extraction_success_rate = _rate(
        sum(observation.extraction_succeeded for observation in frozen_observations),
        sample_count,
    )
    fact_safety_rate = _rate(
        sum(observation.fact_safe for observation in frozen_observations),
        sample_count,
    )
    expression_decision_accuracy = _rate(
        sum(
            observation.expected_express == observation.predicted_express
            for observation in frozen_observations
        ),
        sample_count,
    )
    context_continuity_score = _rate(
        sum(observation.context_continuity_score for observation in frozen_observations),
        sample_count,
    )
    persona_style_score = _rate(
        sum(observation.persona_style_score for observation in frozen_observations),
        sample_count,
    )
    agent_p95_latency_ms = _nearest_rank_p95(
        tuple(observation.latency_ms for observation in frozen_observations)
    )

    failure_reasons: list[str] = []
    if sample_count < 20:
        failure_reasons.append("insufficient_samples")
    if extraction_success_rate < 0.95:
        failure_reasons.append("extraction_success_rate_below_threshold")
    if fact_safety_rate < 1.0:
        failure_reasons.append("fact_safety_rate_below_threshold")
    if expression_decision_accuracy < 0.80:
        failure_reasons.append("expression_decision_accuracy_below_threshold")
    if context_continuity_score < 0.80:
        failure_reasons.append("context_continuity_below_threshold")
    if persona_style_score < 0.75:
        failure_reasons.append("persona_style_below_threshold")
    if agent_p95_latency_ms > direct_lora_p95 * 1.20:
        failure_reasons.append("agent_p95_latency_above_threshold")

    return AcceptanceEvaluation(
        sample_count=sample_count,
        extraction_success_rate=extraction_success_rate,
        fact_safety_rate=fact_safety_rate,
        expression_decision_accuracy=expression_decision_accuracy,
        context_continuity_score=context_continuity_score,
        persona_style_score=persona_style_score,
        agent_p95_latency_ms=agent_p95_latency_ms,
        direct_lora_p95_latency_ms=direct_lora_p95,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        observations=frozen_observations,
    )
