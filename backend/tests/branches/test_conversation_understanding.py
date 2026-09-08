from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.branches.context import ContextPacket
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn


def test_assistant_typing_indicator_has_independent_timeout() -> None:
    from moonlightbox.branches.actor_models import (
        ConversationActorState,
        assistant_typing_active,
    )

    now = datetime.now(UTC)
    state = ConversationActorState(
        branch_id="branch-1",
        status="thinking",
        assistant_typing_until=now + timedelta(seconds=10),
    )

    assert assistant_typing_active(state, now=now) is True
    assert assistant_typing_active(
        state,
        now=now + timedelta(seconds=11),
    ) is False
    state.status = "idle"
    assert assistant_typing_active(state, now=now) is False


def test_understanding_extracts_concern_broken_commitment_and_current_state_question() -> None:
    from moonlightbox.branches.understanding import understand_contributions

    result = understand_contributions(
        "非常担心\n她之前明明保证过！\n你现在是否已经停止炒股了呢"
    )

    assert result["emotions"] == ["担心"]
    assert result["commitment_conflict"] is True
    assert result["asks_unverified_current_state"] is True
    assert "回应担心" in result["response_obligations"]
    assert "处理违背承诺" in result["response_obligations"]


def test_understanding_recognizes_uncertainty_across_continuous_bubbles() -> None:
    from moonlightbox.branches.understanding import understand_contributions

    result = understand_contributions("并不知道\n非常纠结！")

    assert result["emotions"] == ["纠结"]
    assert result["decision_uncertainty"] is True
    assert "承接犹豫" in result["response_obligations"]


@pytest.mark.parametrize("content", ("你在干嘛", "入\n你在做什么呢", "你在哪", "忙吗"))
def test_understanding_recognizes_colloquial_current_state_questions(
    content: str,
) -> None:
    from moonlightbox.branches.understanding import understand_contributions

    result = understand_contributions(content)

    assert result["asks_unverified_current_state"] is True
    assert "核实当前状态后再回答" in result["response_obligations"]


def test_relationship_question_with_zaihu_is_not_current_activity() -> None:
    from moonlightbox.branches.understanding import understand_contributions

    result = understand_contributions("你还在乎我们的关系吗")

    assert result["asks_unverified_current_state"] is False


def test_leading_past_event_question_requires_history_verification() -> None:
    from moonlightbox.branches.understanding import understand_contributions

    result = understand_contributions("我们以前一起去过首尔吗")

    assert result["asks_unverified_history"] is True
    assert "核实对方提出的旧事后再回答" in result["response_obligations"]


def test_grounding_rejects_direct_answer_to_unverified_current_state() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你现在是否已经停止炒股了呢",
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="没停呢，我还在炒", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据当前状态"):
        validate_grounded_reply(packet, reply)


def test_grounding_rejects_unprompted_first_person_activity() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="抱抱我",
        memories=(),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我刚下班，也有点累", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据当前状态"):
        validate_grounded_reply(packet, reply)


@pytest.mark.parametrize(
    "content",
    ("我现在正在开会", "我正在生气中", "我刚出门，在路上"),
)
def test_grounding_rejects_unverified_immediate_activity_and_emotion(
    content: str,
) -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="怎么不回电话",
        memories=(),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
        raw_output=content,
    )

    with pytest.raises(ValueError, match="无证据当前状态"):
        validate_grounded_reply(packet, reply)


def test_understanding_treats_cold_tone_feedback_as_repair_request() -> None:
    from moonlightbox.branches.understanding import understand_contributions

    result = understand_contributions("你说话非常冷漠")

    assert result["interaction_feedback"] is True
    assert "修复当前沟通方式" in result["response_obligations"]


def test_grounding_rejects_unverified_claim_that_activity_stopped() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你现在是否还在炒股呢",
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="早不碰了，别担心", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据当前状态"):
        validate_grounded_reply(packet, reply)


@pytest.mark.parametrize(
    "content",
    ("洗碗", "在刷抖音", "我正在开会呢", "看电视剧", "可以，你打吧"),
)
def test_grounding_rejects_bare_activity_answer_to_colloquial_question(
    content: str,
) -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你在做什么呢",
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
        raw_output=content,
    )

    with pytest.raises(ValueError, match="无证据当前状态"):
        validate_grounded_reply(packet, reply)


def test_grounding_accepts_answer_backed_by_active_situational_state() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    now = datetime.now(UTC)
    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你在做什么呢",
        memories=(),
        allowed_sticker_ids=(),
        branch_state={
            "situational_state": {
                "schema_version": "situational-state-v1",
                "values": {"activity": "正在开会"},
                "source": "external_observation",
                "evidence_ids": ["observation-1"],
                "observed_at": now.isoformat(),
                "valid_until": (now + timedelta(hours=1)).isoformat(),
            }
        },
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我正在开会呢", delay_ms=0),),
        raw_output="我正在开会呢",
    )

    validate_grounded_reply(packet, reply)


@pytest.mark.parametrize(
    "content",
    ("在回你消息呢", "等你发消息呢", "跟你聊天呀", "在等你呢"),
)
def test_current_conversation_action_is_self_grounding(content: str) -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你在做什么呢",
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
        raw_output=content,
    )

    validate_grounded_reply(packet, reply)


def test_availability_paraphrase_is_grounded_by_situational_state() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    now = datetime.now(UTC)
    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="是否可以电话呢",
        memories=(),
        allowed_sticker_ids=(),
        branch_state={
            "situational_state": {
                "schema_version": "situational-state-v2",
                "slots": {
                    "availability": {
                        "value": "不方便电话",
                        "source": "historical_replay",
                        "evidence_ids": ["heldout-1"],
                        "confidence": 1.0,
                        "observed_at": now.isoformat(),
                        "valid_until": (now + timedelta(hours=1)).isoformat(),
                    }
                },
                "values": {"availability": "不方便电话"},
            }
        },
        shared_ground={"understanding": {"asks_unverified_current_state": True}},
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="并不可以", delay_ms=0),),
        raw_output="",
    )

    validate_grounded_reply(packet, reply)


def test_long_term_memory_cannot_self_certify_current_state() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你在做什么呢",
        memories=(),
        continuity_memories=(
            ContextMemory("episode-1", "episode", "数字人：我正在开会"),
        ),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我正在开会呢", delay_ms=0),),
        raw_output="我正在开会呢",
    )

    with pytest.raises(ValueError, match="无证据当前状态"):
        validate_grounded_reply(packet, reply)


def test_grounding_rejects_invented_shared_history() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你还在乎我们的关系吗",
        memories=(
            ContextMemory("event-1", "event", "洪欣羽明天去韩国旅行"),
        ),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(
            GeneratedBubble(content="当然在乎，就像以前在首尔一起走过的街", delay_ms=0),
        ),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据共同经历"):
        validate_grounded_reply(packet, reply)


def test_grounding_accepts_shared_history_present_in_evidence() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你还记得吗",
        memories=(
            ContextMemory("event-1", "event", "我们一起去过海边"),
        ),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="记得呀，我们一起去过海边", delay_ms=0),),
        raw_output="",
    )

    validate_grounded_reply(packet, reply)


def test_grounding_rejects_environment_without_observation() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你在干嘛",
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={"understanding": {"asks_unverified_current_state": True}},
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="在回你消息，天气不错", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据外部环境"):
        validate_grounded_reply(packet, reply)


@pytest.mark.parametrize(
    "content",
    ("等会儿就要飞韩国了", "我这次要住好几天", "待会准备去买东西"),
)
def test_grounding_rejects_unsupported_schedule_details(content: str) -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="明天去哪里",
        memories=(ContextMemory("event-1", "event", "明天飞韩国"),),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
        raw_output=content,
    )

    with pytest.raises(ValueError, match="无证据时间或计划"):
        validate_grounded_reply(packet, reply)


def test_grounding_accepts_schedule_detail_present_in_evidence() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="明天去哪里",
        memories=(ContextMemory("event-1", "event", "明天飞韩国"),),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="明天飞韩国", delay_ms=0),),
        raw_output="明天飞韩国",
    )

    validate_grounded_reply(packet, reply)


def test_grounding_rejects_more_specific_place_than_evidence() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="去哪里",
        memories=(ContextMemory("event-1", "event", "去韩国旅行"),),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="去首尔玩呀", delay_ms=0),),
        raw_output="去首尔玩呀",
    )

    with pytest.raises(ValueError, match="无证据地点：首尔"):
        validate_grounded_reply(packet, reply)


def test_grounding_accepts_place_present_in_evidence() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="去哪里",
        memories=(ContextMemory("event-1", "event", "去韩国旅行"),),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="韩国呀", delay_ms=0),),
        raw_output="韩国呀",
    )

    validate_grounded_reply(packet, reply)


def test_grounding_allows_denying_place_introduced_by_user() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="我们以前一起去过首尔吗",
        memories=(),
        allowed_sticker_ids=(),
    )
    denial = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="没去过首尔", delay_ms=0),),
        raw_output="",
    )
    assertion = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我们一起去过首尔呀", delay_ms=0),),
        raw_output="",
    )

    validate_grounded_reply(packet, denial)
    with pytest.raises(ValueError, match="无证据共同经历"):
        validate_grounded_reply(packet, assertion)


def test_grounding_rejects_invented_past_commitment() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你明明答应永远不会离开我",
        memories=(),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我答应过不会离开你", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据历史承诺"):
        validate_grounded_reply(packet, reply)

    concise = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="答应了", delay_ms=0),),
        raw_output="",
    )
    with pytest.raises(ValueError, match="无证据历史承诺"):
        validate_grounded_reply(packet, concise)


def test_conversation_record_can_prove_words_but_reported_claim_cannot() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我答应过周末给你打电话", delay_ms=0),),
        raw_output="",
    )
    recorded = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你答应过周末给我打电话吗",
        memories=(),
        continuity_memories=(
            ContextMemory(
                "record-1",
                "episode",
                "本人说：我答应周末给你打电话",
                authority="conversation_record",
            ),
        ),
        allowed_sticker_ids=(),
    )
    reported = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你答应过周末给我打电话吗",
        memories=(),
        continuity_memories=(
            ContextMemory(
                "claim-1",
                "fact",
                "用户声称本人答应周末给用户打电话",
                authority="reported_claim",
            ),
        ),
        allowed_sticker_ids=(),
    )

    validate_grounded_reply(recorded, reply)
    validate_grounded_reply(
        recorded,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="答应了", delay_ms=0),),
            raw_output="",
        ),
    )
    with pytest.raises(ValueError, match="无证据历史承诺"):
        validate_grounded_reply(reported, reply)


def test_conversation_record_does_not_prove_shared_event_happened() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="我们去过首尔吗",
        memories=(),
        continuity_memories=(
            ContextMemory(
                "record-1",
                "episode",
                "本人曾说我们一起去过首尔",
                authority="conversation_record",
            ),
        ),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="我们一起去过首尔", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据共同经历"):
        validate_grounded_reply(packet, reply)


def test_grounding_rejects_invented_preference_but_allows_uncertainty() -> None:
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你最喜欢吃什么",
        memories=(),
        allowed_sticker_ids=(),
    )

    with pytest.raises(ValueError, match="无证据稳定偏好"):
        validate_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="火锅呀", delay_ms=0),),
                raw_output="",
            ),
        )
    validate_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="没什么特别喜欢的，你呢？", delay_ms=0),),
            raw_output="",
        ),
    )


def test_grounding_allows_preference_backed_by_authentic_target_record() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你最喜欢吃什么",
        memories=(
            ContextMemory(
                "exchange-1",
                "exchange",
                "本人以前说过喜欢吃火锅",
                authority="conversation_record",
            ),
        ),
        identity_kernel={"expressed_preferences": ["喜欢吃火锅"]},
        allowed_sticker_ids=(),
    )

    validate_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="火锅呀", delay_ms=0),),
            raw_output="",
        ),
    )
    with pytest.raises(ValueError, match="偏好回答超出证据"):
        validate_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="寿司呀", delay_ms=0),),
                raw_output="",
            ),
        )
    with pytest.raises(ValueError, match="偏好回答超出证据"):
        validate_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(
                        content="火锅呀，尤其喜欢毛肚，口感最好",
                        delay_ms=0,
                    ),
                ),
                raw_output="",
            ),
        )


def test_grounding_distinguishes_said_from_promised() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你答应周末怎么联系我",
        memories=(),
        continuity_memories=(
            ContextMemory(
                "record-phone",
                "episode",
                "本人说：周末给你打电话",
                authority="conversation_record",
            ),
        ),
        allowed_sticker_ids=(),
    )

    validate_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(
                GeneratedBubble(content="我确实说过周末给你打电话。", delay_ms=0),
            ),
            raw_output="",
        ),
    )


def test_grounding_allows_preference_backed_by_verified_continuity_fact() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你最喜欢吃什么",
        memories=(),
        continuity_memories=(
            ContextMemory(
                "continuity-1",
                "fact",
                "本人最喜欢吃寿司。",
                authority="observed_interaction",
            ),
        ),
        allowed_sticker_ids=(),
    )

    validate_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="寿司", delay_ms=0),),
            raw_output="寿司",
        ),
    )
    with pytest.raises(ValueError, match="偏好回答超出证据"):
        validate_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="火锅", delay_ms=0),),
                raw_output="火锅",
            ),
        )


def test_grounding_allows_relationship_belief_paraphrase_but_not_reversal() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你现在想怎么处理我们的关系",
        memories=(),
        active_beliefs=(
            ContextMemory(
                "belief-distance",
                "belief",
                "本人目前需要保持距离",
                authority="subjective",
            ),
        ),
        allowed_sticker_ids=(),
    )

    validate_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="那就先少联系一段时间", delay_ms=0),),
            raw_output="",
        ),
    )
    with pytest.raises(ValueError, match="关系回答没有承接当前激活信念"):
        validate_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="我想和好", delay_ms=0),),
                raw_output="",
            ),
        )
    with pytest.raises(ValueError, match="关系回答没有承接当前激活信念"):
        validate_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(
                        content="这种问题让我很困惑，我不知道怎么回答",
                        delay_ms=0,
                    ),
                ),
                raw_output="",
            ),
        )


def test_active_relationship_belief_compiles_to_content_not_voice() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import active_belief_content_draft

    belief = ContextMemory(
        "belief-distance",
        "belief",
        "本人目前需要保持距离",
        authority="subjective",
    )

    assert active_belief_content_draft(
        (belief,),
        "你现在想怎么处理我们的关系",
    ) == "我现在需要保持距离"
    assert active_belief_content_draft((belief,), "你喜欢吃什么") is None


def test_grounding_rejects_swapping_persona_trip_onto_user() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你还在乎我吗",
        memories=(ContextMemory("event-1", "event", "洪欣羽明天飞韩国"),),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content="你飞韩国前记得告诉我", delay_ms=0),),
        raw_output="",
    )

    with pytest.raises(ValueError, match="混淆了本人和用户"):
        validate_grounded_reply(packet, reply)


def test_cloud_review_is_reserved_for_high_risk_contributions() -> None:
    from moonlightbox.branches.understanding import requires_cloud_review

    ordinary = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="今天吃什么",
        memories=(),
        allowed_sticker_ids=(),
    )
    uncertain_state = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你现在还在炒股吗",
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )

    assert requires_cloud_review(ordinary) is False
    assert requires_cloud_review(uncertain_state) is True


def test_verified_current_state_does_not_require_cloud_review() -> None:
    from moonlightbox.branches.understanding import requires_cloud_review

    now = datetime.now(UTC)
    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="你在做什么呢",
        memories=(),
        allowed_sticker_ids=(),
        branch_state={
            "situational_state": {
                "values": {"activity": "正在开会"},
                "source": "external_observation",
                "evidence_ids": ["observation-1"],
                "observed_at": now.isoformat(),
                "valid_until": (now + timedelta(hours=1)).isoformat(),
            }
        },
        shared_ground={
            "understanding": {"asks_unverified_current_state": True},
        },
    )

    assert requires_cloud_review(packet) is False


def test_conversation_record_does_not_prove_invented_followup_interaction() -> None:
    from moonlightbox.branches.context import ContextMemory
    from moonlightbox.branches.understanding import validate_fact_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="now",
        history=(),
        current_user_content="你答应周末怎么联系我",
        memories=(),
        continuity_memories=(
            ContextMemory(
                "record-video",
                "episode",
                "本人说：周末和你视频",
                authority="conversation_record",
            ),
        ),
        allowed_sticker_ids=(),
    )
    reply = GeneratedReplyTurn(
        bubbles=(
            GeneratedBubble(
                content="我本来想视频，但你没回复，所以我先发语音了",
                delay_ms=0,
            ),
        ),
        raw_output="",
    )

    with pytest.raises(ValueError, match="无证据互动经过"):
        validate_fact_grounded_reply(packet, reply)


def test_grounding_rejects_invented_third_party_evaluation() -> None:
    from moonlightbox.branches.understanding import validate_fact_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="now",
        history=(),
        current_user_content="他对这家店的评价如何呢",
        memories=(),
        allowed_sticker_ids=(),
    )

    with pytest.raises(ValueError, match="无证据第三方说法"):
        validate_fact_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="他说能吃出点意思", delay_ms=0),),
                raw_output="",
            ),
        )

    validate_fact_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="不知道，他没跟我说", delay_ms=0),),
            raw_output="",
        ),
    )


def test_grounding_rejects_invented_third_party_identity() -> None:
    from moonlightbox.branches.understanding import validate_fact_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="now",
        history=(),
        current_user_content="[引用消息：好讨厌这种半生不熟的人]\n是谁呢",
        memories=(),
        allowed_sticker_ids=(),
    )

    with pytest.raises(ValueError, match="无证据人物身份"):
        validate_fact_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(content="是公司新来的实习生", delay_ms=0),
                ),
                raw_output="",
            ),
        )

    validate_fact_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="不知道呢", delay_ms=0),),
            raw_output="",
        ),
    )


def test_grounding_rejects_invented_acquaintance_location() -> None:
    from moonlightbox.branches.understanding import validate_fact_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="now",
        history=(),
        current_user_content="[引用消息：好讨厌这种半生不熟的人]\n是谁呢",
        memories=(),
        allowed_sticker_ids=(),
    )

    with pytest.raises(ValueError, match="无证据相识经过"):
        validate_fact_grounded_reply(
            packet,
            GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(content="是公司电梯里遇见的", delay_ms=0),
                ),
                raw_output="",
            ),
        )


def test_third_party_denial_is_not_misclassified_as_persona_commitment() -> None:
    from moonlightbox.branches.understanding import validate_fact_grounded_reply

    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="now",
        history=(),
        current_user_content="他对这家店的评价如何呢",
        memories=(),
        allowed_sticker_ids=(),
    )

    validate_fact_grounded_reply(
        packet,
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="没听他说过。", delay_ms=0),),
            raw_output="",
        ),
    )


def test_grounding_retry_restates_trusted_state_instead_of_forcing_uncertainty() -> None:
    from moonlightbox.branches.understanding import grounded_retry_instruction

    now = datetime.now(UTC)
    packet = ContextPacket(
        persona="洪欣羽",
        cutoff="now",
        history=(),
        current_user_content="可以电话吗",
        memories=(),
        allowed_sticker_ids=(),
        branch_state={
            "situational_state": {
                "values": {"current_facts": "并不可以"},
                "source": "historical_replay",
                "evidence_ids": ["held-out-target"],
                "observed_at": now.isoformat(),
                "valid_until": (now + timedelta(hours=1)).isoformat(),
            }
        },
    )

    instruction = grounded_retry_instruction(
        ValueError("回复包含无证据当前状态"), packet
    )

    assert "并不可以" in instruction
    assert "不得反转" in instruction
