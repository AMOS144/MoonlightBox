from datetime import UTC, datetime, timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.training.expression_policy import (
    ExpressionPolicy,
    _features,
    build_expression_policy,
    should_respond,
)


def _message(index: int, sender: str, content: str) -> ImportedMessage:
    return ImportedMessage(
        source_id=str(index),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        sender=sender,
        kind=MessageKind.TEXT,
        content=content,
        raw={},
    )


def test_expression_policy_learns_response_and_silence_without_raw_text() -> None:
    messages: list[ImportedMessage] = []
    index = 0
    for sample in range(20):
        messages.append(_message(index, "我", f"你觉得第{sample}个怎么样吗"))
        index += 1
        messages.append(_message(index, "她", "我觉得可以呀"))
        index += 1
    for _sample in range(6):
        messages.append(_message(index, "我", "嗯嗯"))
        index += 1
        # More than thirty minutes starts a new session and labels the prior
        # acknowledgement as an observed non-response.
        index += 31
        messages.append(_message(index, "我", "换个话题"))
        index += 1
        messages.append(_message(index, "她", "好"))
        index += 1

    policy = build_expression_policy(messages, target_sender="她")
    serialized = str(policy.to_metadata())

    assert policy.enabled is True
    assert policy.positive_count >= 20
    assert policy.negative_count >= 5
    assert "嗯嗯" not in serialized
    assert should_respond(policy, "你觉得呢？") is True


def test_expression_policy_fails_open_without_enough_negative_examples() -> None:
    policy = ExpressionPolicy.from_metadata(
        {"enabled": False, "positive_count": 100, "negative_count": 0}
    )

    assert should_respond(policy, "嗯") is True


def test_expression_policy_v2_does_not_treat_longer_message_as_more_silent() -> None:
    feature_weights = {
        feature: -0.2
        for feature in set(_features("普通长消息" * 20))
    }
    policy = ExpressionPolicy(
        version="historical-expression-nb-v2",
        enabled=True,
        positive_count=100,
        negative_count=20,
        bias=2.0,
        weights=feature_weights,
    )

    assert should_respond(policy, "普通长消息" * 20) is True


def test_expression_policy_never_silences_conversation_repair_requests() -> None:
    policy = ExpressionPolicy(
        version="historical-expression-nb-v2",
        enabled=True,
        positive_count=100,
        negative_count=100,
        bias=-20,
        weights={},
    )

    for content in ("什么意思呢", "并看不懂", "没明白", "你说清楚"):
        assert should_respond(policy, content) is True
