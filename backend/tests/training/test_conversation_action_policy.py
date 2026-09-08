from datetime import UTC, datetime, timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.training.conversation_action_policy import (
    ConversationActionPolicy,
    build_conversation_action_policy,
    choose_conversation_action,
    quote_content,
)


def _message(
    index: int,
    sender: str,
    content: str,
    *,
    kind: MessageKind = MessageKind.TEXT,
) -> ImportedMessage:
    return ImportedMessage(
        source_id=f"m{index}",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        sender=sender,
        kind=kind,
        content=content,
        raw={},
    )


def test_action_policy_learns_quote_call_and_reaction_without_raw_context() -> None:
    messages: list[ImportedMessage] = []
    for index in range(30):
        messages.append(_message(index * 2, "我", f"问题 {index}"))
        if index < 5:
            reply = f'回答 {index}\n> 我: 问题 {index}'
        elif index < 9:
            reply = '<voipmsg><room_type>0</room_type><msg>已取消</msg></voipmsg>'
        elif index < 12:
            reply = "我拍了拍你"
        else:
            reply = f"普通回答 {index}"
        messages.append(_message(index * 2 + 1, "她", reply))

    policy = build_conversation_action_policy(messages, target_sender="她")
    metadata = policy.to_metadata()

    assert policy.enabled is True
    assert policy.action_counts == {
        "quote": 5,
        "call": 4,
        "reaction": 3,
        "retract": 0,
    }
    assert metadata["raw_content_retained"] is False
    assert "问题 0" not in str(metadata)
    assert ConversationActionPolicy.from_metadata(metadata) == policy


def test_action_choice_is_deterministic_and_proactive_uses_no_unproven_actions() -> None:
    policy = ConversationActionPolicy(
        version="historical-conversation-actions-v1",
        total_target_messages=20,
        action_counts={"quote": 20, "call": 20, "reaction": 20},
        biases={"quote": 20, "call": 20, "reaction": 20},
        weights={"quote": {}, "call": {}, "reaction": {}},
    )

    first = choose_conversation_action(policy, "这一条", seed="same")
    second = choose_conversation_action(policy, "这一条", seed="same")

    assert first == second
    assert first in {"quote", "call", "reaction"}
    assert (
        choose_conversation_action(
            policy,
            "这一条",
            seed="same",
            proactive=True,
        )
        is None
    )


def test_quote_payload_keeps_reply_and_bounds_quoted_user_content() -> None:
    payload = quote_content("我知道了", "  很长   的消息" * 100)

    assert '"text": "我知道了"' in payload
    assert len(payload) < 230


def test_action_policy_attributes_system_retractions_through_quote_alias() -> None:
    messages: list[ImportedMessage] = []
    for index in range(30):
        messages.append(_message(index * 3, "我", f"问题 {index}"))
        messages.append(_message(index * 3 + 1, "她", f"回答 {index}"))
        if index < 5:
            messages.append(
                _message(
                    index * 3 + 2,
                    "我",
                    f"知道了\n> Feather：回答 {index}",
                )
            )
    for index in range(3):
        messages.append(
            _message(
                100 + index,
                "系统",
                '<?xml version="1.0"?><sysmsg type="revokemsg"><revokemsg>'
                '<content>"Feather" 撤回了一条消息</content></revokemsg></sysmsg>',
                kind=MessageKind.UNKNOWN,
            )
        )

    policy = build_conversation_action_policy(messages, target_sender="她")

    assert policy.action_counts["retract"] == 3
