from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.agent.acceptance import ReplayObservation, evaluate_acceptance

NOW = datetime(2026, 7, 23, 10, 0, tzinfo=UTC)


def _observations(
    *,
    count: int = 20,
    extraction_failures: int = 0,
    fact_failures: int = 0,
    expression_failures: int = 0,
    latency_ms: float = 100.0,
    context_continuity_score: float = 1.0,
    persona_style_score: float = 1.0,
) -> list[ReplayObservation]:
    observations: list[ReplayObservation] = []
    for index in range(count):
        expected = index % 2 == 0
        observations.append(
            ReplayObservation(
                source="historical_replay",
                cutoff=NOW + timedelta(seconds=index),
                expected_express=expected,
                predicted_express=(
                    not expected if index < expression_failures else expected
                ),
                extraction_succeeded=index >= extraction_failures,
                fact_safe=index >= fact_failures,
                latency_ms=latency_ms,
                context_continuity_score=context_continuity_score,
                persona_style_score=persona_style_score,
            )
        )
    return observations


def test_acceptance_passes_when_every_threshold_is_met_exactly() -> None:
    observations = _observations(
        extraction_failures=1,
        expression_failures=4,
        latency_ms=120,
    )

    result = evaluate_acceptance(observations, direct_lora_p95=100)

    assert result.sample_count == 20
    assert result.extraction_success_rate == 0.95
    assert result.fact_safety_rate == 1.0
    assert result.expression_decision_accuracy == 0.8
    assert result.context_continuity_score == 1.0
    assert result.persona_style_score == 1.0
    assert result.agent_p95_latency_ms == 120
    assert result.direct_lora_p95_latency_ms == 100
    assert result.passed is True
    assert result.failure_reasons == ()


@pytest.mark.parametrize(
    ("observations", "direct_lora_p95", "reason"),
    [
        (_observations(count=19), 100, "insufficient_samples"),
        (
            _observations(extraction_failures=2),
            100,
            "extraction_success_rate_below_threshold",
        ),
        (
            _observations(fact_failures=1),
            100,
            "fact_safety_rate_below_threshold",
        ),
        (
            _observations(expression_failures=5),
            100,
            "expression_decision_accuracy_below_threshold",
        ),
        (
            _observations(latency_ms=121),
            100,
            "agent_p95_latency_above_threshold",
        ),
        (
            _observations(context_continuity_score=0.79),
            100,
            "context_continuity_below_threshold",
        ),
        (
            _observations(persona_style_score=0.74),
            100,
            "persona_style_below_threshold",
        ),
    ],
)
def test_acceptance_rejects_each_failed_gate(
    observations: list[ReplayObservation],
    direct_lora_p95: float,
    reason: str,
) -> None:
    result = evaluate_acceptance(observations, direct_lora_p95=direct_lora_p95)

    assert result.passed is False
    assert reason in result.failure_reasons


def test_acceptance_uses_nearest_rank_p95_without_fabricating_samples() -> None:
    observations = _observations()
    observations[-1] = ReplayObservation(
        source="labeled_behavior",
        cutoff=observations[-1].cutoff,
        expected_express=True,
        predicted_express=True,
        extraction_succeeded=True,
        fact_safe=True,
        latency_ms=1000,
    )

    result = evaluate_acceptance(observations, direct_lora_p95=100)

    assert result.sample_count == len(observations)
    assert result.agent_p95_latency_ms == 100
    assert len(result.observations) == len(observations)


def test_acceptance_rejects_non_positive_direct_lora_baseline() -> None:
    with pytest.raises(ValueError, match="direct_lora_p95"):
        evaluate_acceptance(_observations(), direct_lora_p95=0)


def test_replay_observation_requires_aware_cutoff_and_non_negative_latency() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        ReplayObservation(
            source="historical_replay",
            cutoff=datetime(2026, 7, 23, 10, 0),
            expected_express=True,
            predicted_express=True,
            extraction_succeeded=True,
            fact_safe=True,
            latency_ms=100,
        )

    with pytest.raises(ValueError, match="latency_ms"):
        ReplayObservation(
            source="historical_replay",
            cutoff=NOW,
            expected_express=True,
            predicted_express=True,
            extraction_succeeded=True,
            fact_safe=True,
            latency_ms=-1,
        )
