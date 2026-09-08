from datetime import UTC, datetime, timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.training.media_behavior_policy import (
    MediaBehaviorPolicy,
    build_media_behavior_policy,
    choose_media_modality,
)


def _message(
    index: int,
    sender: str,
    kind: MessageKind,
    content: str,
) -> ImportedMessage:
    return ImportedMessage(
        source_id=str(index),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
        sender=sender,
        kind=kind,
        content=content,
        raw={},
    )


def test_media_behavior_policy_learns_modality_without_retaining_raw_text() -> None:
    messages: list[ImportedMessage] = []
    index = 0
    for turn in range(30):
        messages.append(_message(index, "我", MessageKind.TEXT, f"看看这个 {turn}"))
        index += 1
        messages.append(
            _message(
                index,
                "她",
                MessageKind.IMAGE if turn < 6 else MessageKind.TEXT,
                "[图片]" if turn < 6 else "好",
            )
        )
        index += 1

    policy = build_media_behavior_policy(messages, target_sender="她")
    metadata = policy.to_metadata()

    assert policy.enabled is True
    assert policy.modality_counts == {"image": 6, "audio": 0, "video": 0}
    assert metadata["raw_content_retained"] is False
    assert "看看这个" not in str(metadata)


def test_media_behavior_never_uses_historical_media_proactively() -> None:
    policy = MediaBehaviorPolicy(
        version="test",
        total_target_messages=100,
        modality_counts={"image": 100},
        biases={"image": 20},
        weights={"image": {}},
    )

    assert (
        choose_media_modality(
            policy,
            "看看",
            seed="always",
            proactive=True,
        )
        is None
    )
    assert choose_media_modality(policy, "看看", seed="always") == "image"
