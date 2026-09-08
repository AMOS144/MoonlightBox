from datetime import UTC, datetime, timedelta

from moonlightbox.world.bundles import WorldMessage, build_conversation_bundles


def message(
    message_id: str,
    minute: int,
    participant: str,
    content: str,
) -> WorldMessage:
    return WorldMessage(
        id=message_id,
        import_id="import-1",
        participant_id=f"participant-{participant}",
        participant_name=participant,
        participant_role="target" if participant == "小月" else "self",
        timestamp=datetime(2024, 6, 18, 10, minute, tzinfo=UTC),
        kind="text",
        content=content,
    )


def test_bundle_body_is_natural_dialogue_and_mapping_keeps_ids() -> None:
    bundles = build_conversation_bundles(
        [message("m1", 0, "我", "你今天上班吗"), message("m2", 1, "小月", "要去新公司")],
        project_id="project-1",
    )

    assert len(bundles) == 1
    assert "我：你今天上班吗" in bundles[0].content
    assert "小月：要去新公司" in bundles[0].content
    assert "m1" not in bundles[0].content
    assert [item.message.id for item in bundles[0].messages] == ["m1", "m2"]


def test_new_bundle_carries_last_three_turns_without_semantic_thresholds() -> None:
    source = [
        message("m1", 0, "我", "第一轮"),
        message("m2", 1, "我", "还是第一轮"),
        message("m3", 2, "小月", "第二轮"),
        message("m4", 3, "我", "第三轮"),
        message("m5", 4, "小月", "很长的第四轮"),
    ]
    bundles = build_conversation_bundles(
        source,
        project_id="project-1",
        max_characters=70,
        carry_in_turns=3,
    )

    assert len(bundles) >= 2
    carried = [item for item in bundles[1].messages if item.is_carry_in]
    assert carried
    assert all(item.message.id in {"m1", "m2", "m3", "m4"} for item in carried)
    assert bundles[1].primary_message_count > 0


def test_splits_a_continuous_chat_only_on_simple_gap_or_length() -> None:
    source = [
        message("m1", 0, "我", "早"),
        message("m2", 1, "小月", "早呀"),
        WorldMessage(
            **{
                **message("m3", 2, "我", "晚上聊").__dict__,
                "timestamp": datetime(2024, 6, 18, 20, 0, tzinfo=UTC),
            }
        ),
    ]
    bundles = build_conversation_bundles(
        source,
        project_id="project-1",
        session_gap=timedelta(hours=6),
    )

    assert len(bundles) == 2
    assert bundles[1].messages[-1].message.id == "m3"
