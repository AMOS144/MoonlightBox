def test_context_builder_keeps_latest_user_and_naturalizes_history() -> None:
    from moonlightbox.branches.context import (
        ContextBubble,
        ContextBuilder,
        ContextMemory,
        ContextRequest,
        ContextTurn,
    )

    builder = ContextBuilder()
    request = ContextRequest(
        persona="小肥入",
        cutoff="2026-01-01T00:00:00",
        memories=(
            ContextMemory(
                resource_id="event-1",
                resource_type="event",
                content="双方关系亲密",
            ),
        ),
        history=(
            ContextTurn(
                role="assistant",
                bubbles=(
                    ContextBubble(type="text", content="第一条"),
                    ContextBubble(type="text", content="第二条"),
                ),
            ),
        ),
        current_user_content="我也是",
        allowed_sticker_ids=("sticker-1",),
        identity_kernel={"persona": "稳定直接"},
        branch_state={"relationship_state": {"trust": 60}},
        active_beliefs=(
            ContextMemory(
                resource_id="belief-active",
                resource_type="belief",
                content="我目前仍愿意继续了解用户",
            ),
        ),
        contested_beliefs=(
            ContextMemory(
                resource_id="belief-contested",
                resource_type="belief",
                content="用户是否尊重边界仍不确定",
            ),
        ),
        shared_ground={"topic": "双方都承认最近沟通变少"},
        continuity_memories=(
            ContextMemory(
                resource_id="continuity:branch-1:item-1",
                resource_type="belief",
                content="我愿意继续了解用户",
            ),
        ),
    )
    packet = builder.build_packet(request)
    result = builder.to_chat_messages(packet)

    assert [item.role for item in result] == ["system", "assistant", "user"]
    assert result[1].content == "第一条\n第二条"
    assert result[-1].content == "我也是"
    assert '"bubbles"' not in result[1].content
    assert "用户正在认同你上一句话" in result[0].content
    assert packet.memories[0].resource_id == "event-1"
    assert packet.allowed_sticker_ids == ("sticker-1",)
    assert packet.identity_kernel == {"persona": "稳定直接"}
    assert packet.branch_state["relationship_state"] == {"trust": 60}
    assert packet.continuity_memories[0].resource_type == "belief"
    assert "用户对数字人的描述只是用户观点" in result[0].content
    assert "当前激活信念（必须参与决定）" in result[0].content
    assert "当前竞争信念（保持不确定）" not in result[0].content
    assert packet.contested_beliefs[0].content == "用户是否尊重边界仍不确定"
    assert "会话共同认知" in result[0].content
    assert "本人长期背景" in result[0].content
    assert "这段关系中已经形成的感受和关注" in result[0].content
    assert "对这段关系仍有保留" in result[0].content
    assert "已确认经历" in result[0].content
    assert builder.to_review_payload(packet)["current_user_content"] == "我也是"


def test_persona_text_expression_prompt_excludes_private_cognition_schema() -> None:
    from datetime import UTC, datetime, timedelta

    from moonlightbox.branches.context import (
        ContextBuilder,
        ContextMemory,
        ContextRequest,
    )

    packet = ContextBuilder().build_packet(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-01-01T00:00:00",
            memories=(
                ContextMemory("event-1", "event", "已经约好周末见面"),
                ContextMemory(
                    "branch:test",
                    "event",
                    "当前分支设定（回答必须优先遵循）：\n"
                    "分支名称：韩国之旅\n"
                    "事件摘要：两个人正在旅行\n"
                    '分支状态：{"emotion":"happy"}',
                ),
            ),
            history=(),
            current_user_content="那周末见",
            identity_kernel={
                "values": ["重视坦诚"],
                "language_patterns": ["你要知道"],
            },
            branch_state={
                "emotion": "happy",
                "situational_state": {
                    "schema_version": "situational-state-v1",
                    "values": {"activity": "正在机场候机"},
                    "source": "external_observation",
                    "evidence_ids": ["observation-1"],
                    "observed_at": datetime.now(UTC).isoformat(),
                    "valid_until": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                },
            },
            active_beliefs=(
                ContextMemory("belief-1", "belief", "对方会按约定见面"),
            ),
            reply_protocol="persona_text",
        )
    )

    messages = ContextBuilder().to_chat_messages(packet)
    system = messages[0].content

    assert "私人微信聊天" in system
    assert "聊天文字本身" in system
    assert "已经约好周末见面" in system
    assert "韩国之旅；两个人正在旅行" in system
    assert "重视坦诚" in system
    assert "对方会按约定见面" in system
    assert "你要知道" not in system
    assert "自主主体" not in system
    assert "稳定人格内核" not in system
    assert "当前分支主体状态" not in system
    assert '"emotion"' not in system
    assert "当前分支设定" not in system
    assert "分支状态" not in system
    assert "本人当前情况（仅在本轮有效，不能扩写）：当前活动：正在机场候机" in system


def test_persona_text_receives_natural_shared_ground_and_open_topic() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    packet = ContextBuilder().build_packet(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="并不知道\n非常纠结！",
            shared_ground={
                "topic": "十一去泰国怎么玩",
                "understanding": {
                    "emotions": ["纠结"],
                    "response_obligations": ["承接犹豫"],
                },
                "open_sequences": [{"topic": "旅行目的地还没有决定"}],
            },
            reply_protocol="persona_text",
        )
    )

    system = ContextBuilder().to_chat_messages(packet)[0].content

    assert "当前还在聊：十一去泰国怎么玩" in system
    assert "对方现在有些纠结" in system
    assert "还没聊完：旅行目的地还没有决定" in system
    assert '"response_obligations"' not in system


def test_persona_text_keeps_low_authority_memory_out_of_generation() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextMemory, ContextRequest

    packet = ContextBuilder().build_packet(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="还记得吗",
            continuity_memories=(
                ContextMemory(
                    "continuity:b:e1",
                    "episode",
                    "数字人：我今天去了医院",
                    authority="conversation_record",
                    evidence_ids=("e1",),
                ),
                ContextMemory(
                    "continuity:b:f1",
                    "fact",
                    "用户说自己已经戒烟",
                    authority="reported_claim",
                    evidence_ids=("e2",),
                ),
                ContextMemory(
                    "continuity:b:x1",
                    "experience",
                    "双方共同完成了一次旅行",
                    authority="observed_interaction",
                    evidence_ids=("e3",),
                ),
            ),
            reply_protocol="persona_text",
        )
    )

    system = ContextBuilder().to_chat_messages(packet)[0].content

    assert "已确认经历：双方共同完成了一次旅行" in system
    assert "用户说自己已经戒烟" not in system
    assert "我今天去了医院" not in system
    assert {memory.authority for memory in packet.continuity_memories} == {
        "conversation_record",
        "reported_claim",
        "observed_interaction",
    }


def test_context_selects_ranked_memory_with_intent_quota() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextMemory, ContextRequest

    memories = (
        ContextMemory("e1", "episode", "第一条经历", authority="conversation_record"),
        ContextMemory("e2", "episode", "第二条经历", authority="conversation_record"),
        ContextMemory("e3", "episode", "第三条经历", authority="conversation_record"),
        ContextMemory("b1", "belief", "第一条关系看法", authority="subjective"),
        ContextMemory("b2", "belief", "第二条关系看法", authority="subjective"),
        ContextMemory("x1", "experience", "一次关系互动", authority="observed_interaction"),
    )

    relationship = ContextBuilder().build_packet(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="你还在乎我们的关系吗",
            continuity_memories=memories,
        )
    )
    current_state = ContextBuilder().build_packet(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="你在干嘛",
            continuity_memories=memories,
        )
    )

    assert [memory.resource_id for memory in relationship.continuity_memories] == [
        "e1",
        "b1",
        "b2",
        "x1",
    ]
    assert current_state.continuity_memories == ()


def test_persona_text_receives_subjective_state_without_promoting_it_to_fact() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    packet = ContextBuilder().build_packet(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="怎么了",
            branch_state={
                "relationship_state": {
                    "summary": "双方正在修复刚才的误会",
                    "trust": 75,
                    "conflict": 55,
                },
                "emotional_tendency": {
                    "labels": ["在意", "谨慎"],
                    "sadness": 0.8,
                },
            },
            mental_state={"mood": "有点担心", "attention": ["关系修复"]},
            active_goals=("想把刚才的误会说开",),
            reply_protocol="persona_text",
        )
    )

    system = ContextBuilder().to_chat_messages(packet)[0].content

    assert "本人当前主观状态（只影响态度，不能当作现实事实）" in system
    assert "mood：有点担心" in system
    assert "本人目前在意或想推进的事情（不必主动复述）" in system
    assert "想把刚才的误会说开" in system
    assert "这段关系中已经形成的感受和关注" in system
    assert "双方正在修复刚才的误会" in system
    assert "在意；谨慎" in system
    assert "对这段关系比较信任" in system
    assert "目前仍有一些冲突" in system
    assert "目前非常难过" in system


def test_review_payload_marks_answered_questions_as_resolved() -> None:
    from moonlightbox.branches.context import (
        ContextBubble,
        ContextBuilder,
        ContextRequest,
        ContextTurn,
    )

    builder = ContextBuilder()
    packet = builder.build_packet(
        ContextRequest(
            persona="小肥入",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(
                ContextTurn(
                    role="user",
                    bubbles=(ContextBubble(type="text", content="今晚睡了吗"),),
                ),
                ContextTurn(
                    role="assistant",
                    bubbles=(ContextBubble(type="text", content="还没呢"),),
                ),
            ),
            current_user_content="我刚洗完澡",
        )
    )

    assert builder.to_review_payload(packet)["resolved_questions"] == ["今晚睡了吗"]


def test_compact_context_never_exposes_asset_placeholder_and_forbids_empty_allowlist() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    builder = ContextBuilder()
    without_assets = builder.build(
        ContextRequest(
            persona="小肥入",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="你好",
            reply_protocol="compact",
        )
    )[0].content
    with_assets = builder.build(
        ContextRequest(
            persona="小肥入",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(),
            current_user_content="你好",
            allowed_sticker_ids=("sticker-1",),
            reply_protocol="compact",
        )
    )[0].content

    assert "资产ID" not in without_assets
    assert "本轮禁止输出 sticker" in without_assets
    assert "资产ID" not in with_assets
    assert "sticker-1" in with_assets


def test_compact_context_serializes_history_with_reversible_protocol() -> None:
    from moonlightbox.branches.context import (
        ContextBubble,
        ContextBuilder,
        ContextRequest,
        ContextTurn,
    )

    messages = ContextBuilder().build(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            history=(
                ContextTurn(
                    role="assistant",
                    bubbles=(
                        ContextBubble(type="text", content="历史文字"),
                        ContextBubble(
                            type="sticker",
                            asset_id="asset-1",
                        ),
                    ),
                ),
            ),
            current_user_content="继续",
            reply_protocol="compact",
        )
    )

    assert messages[1].content == (
        "<bubble>历史文字</bubble><delay>0</delay>"
        "<sticker>asset-1</sticker><delay>0</delay>"
    )
    assert "[表情资产:" not in messages[1].content


def test_baseline_history_is_injected_before_branch_history() -> None:
    from moonlightbox.branches.context import (
        ContextBubble,
        ContextBuilder,
        ContextRequest,
        ContextTurn,
    )

    builder = ContextBuilder()
    packet = builder.build_packet(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01T00:00:00",
            memories=(),
            baseline_manifest={
                "id": "manifest-1",
                "boundary_message_id": "node-message",
            },
            baseline_history=(
                ContextTurn(
                    role="user",
                    bubbles=(ContextBubble(type="text", content="节点前用户消息"),),
                ),
                ContextTurn(
                    role="assistant",
                    bubbles=(ContextBubble(type="text", content="节点前真实回复"),),
                ),
            ),
            history=(
                ContextTurn(
                    role="user",
                    bubbles=(ContextBubble(type="text", content="分支内消息"),),
                ),
            ),
            current_user_content="当前消息",
        )
    )

    messages = builder.to_chat_messages(packet)

    assert "基础历史边界" in messages[0].content
    assert [message.content for message in messages[1:]] == [
        "节点前用户消息",
        "节点前真实回复",
        "分支内消息",
        "当前消息",
    ]


def test_context_builder_limits_stickers_to_keep_prompt_responsive() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    packet = ContextBuilder().build_packet(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-04-20T00:00:00",
            memories=(),
            history=(),
            current_user_content="在吗",
            allowed_sticker_ids=tuple(f"sticker-{index}" for index in range(300)),
        )
    )

    assert len(packet.allowed_sticker_ids) == 8


def test_context_builder_excludes_audit_style_profile_from_runtime_prompt() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    messages = ContextBuilder(character_budget=2000).build(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-04-20T00:00:00",
            memories=(),
            history=(),
            current_user_content="在吗",
            identity_kernel={
                "persona": "直接但关心人",
                "values": ["真诚"],
                "style_profile": {"audit_samples": "样本" * 20_000},
            },
        )
    )

    assert "直接但关心人" in messages[0].content
    assert "style_profile" not in messages[0].content
    assert "audit_samples" not in messages[0].content
    assert sum(len(message.content) for message in messages) <= 2000


def test_context_builder_keeps_default_generation_prompt_within_four_thousand_chars() -> None:
    from moonlightbox.branches.context import (
        ContextBubble,
        ContextBuilder,
        ContextRequest,
        ContextTurn,
    )

    messages = ContextBuilder().build(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-04-20T00:00:00",
            memories=(),
            baseline_history=tuple(
                ContextTurn(
                    role="assistant" if index % 2 else "user",
                    bubbles=(
                        ContextBubble(
                            type="text",
                            content=f"历史消息 {index}：" + "内容" * 80,
                        ),
                    ),
                )
                for index in range(50)
            ),
            history=(),
            current_user_content="现在呢",
        )
    )

    assert sum(len(message.content) for message in messages) <= 4000
    assert messages[-1].content == "现在呢"


def test_context_builder_uses_compact_protocol_only_when_requested() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    builder = ContextBuilder()
    compact = builder.build(
        ContextRequest(
            persona="洪欣羽",
            cutoff="2026-04-20T00:00:00",
            memories=(),
            history=(),
            current_user_content="在吗",
            reply_protocol="compact",
        )
    )
    legacy = builder.build(
        ContextRequest(
            persona="旧模型",
            cutoff="2026-04-20T00:00:00",
            memories=(),
            history=(),
            current_user_content="在吗",
        )
    )

    assert "延迟使用符合真实聊天节奏的毫秒数" in compact[0].content
    assert "<delay>0</delay>" not in compact[0].content
    assert "资产ID" not in compact[0].content
    assert "本轮禁止输出 sticker" in compact[0].content
    assert "统一气泡 JSON" not in compact[0].content
    assert "统一气泡 JSON" in legacy[0].content
    assert "<bubble>文字</bubble>" not in legacy[0].content


def test_conversation_record_is_visible_only_for_prior_utterance_queries() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextMemory, ContextRequest

    memory = ContextMemory(
        "conversation-1",
        "episode",
        "本人说：我答应周末给你打电话",
        authority="conversation_record",
    )
    builder = ContextBuilder()
    commitment = builder.build(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01",
            memories=(),
            history=(),
            current_user_content="你答应过周末给我打电话吗",
            continuity_memories=(memory,),
            reply_protocol="persona_text",
        )
    )[0].content
    past_event = builder.build(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01",
            memories=(),
            history=(),
            current_user_content="我们周末一起出去过吗",
            continuity_memories=(memory,),
            reply_protocol="persona_text",
        )
    )[0].content

    assert "可确认本人曾说过的话" in commitment
    assert "我答应周末给你打电话" in commitment
    assert "我答应周末给你打电话" not in past_event


def test_preference_query_receives_only_authentic_expressed_preference_evidence() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextMemory, ContextRequest

    builder = ContextBuilder()
    request = ContextRequest(
        persona="她",
        cutoff="2026-01-01",
        memories=(
            ContextMemory(
                "exchange-1",
                "exchange",
                "用户：你喜欢吃什么\n目标：我喜欢吃火锅",
                authority="conversation_record",
            ),
        ),
        history=(),
        current_user_content="你喜欢吃什么",
        identity_kernel={"expressed_preferences": ["我喜欢吃火锅"]},
        reply_protocol="persona_text",
    )

    prompt = builder.build(request)[0].content

    assert "本人在真实历史中明确表达过的偏好" in prompt
    assert "我喜欢吃火锅" in prompt
    assert "偏好问题只能沿用原意" in prompt
    assert "不得临时创造固定喜好" not in prompt


def test_preference_query_without_evidence_explicitly_forbids_invention() -> None:
    from moonlightbox.branches.context import ContextBuilder, ContextRequest

    prompt = ContextBuilder().build(
        ContextRequest(
            persona="她",
            cutoff="2026-01-01",
            memories=(),
            history=(),
            current_user_content="你喜欢吃什么",
            reply_protocol="persona_text",
        )
    )[0].content

    assert "当前没有可确认的本人偏好证据" in prompt
    assert "不得临时创造固定喜好或习惯" in prompt
