import json
from dataclasses import dataclass, field
from typing import Literal

from moonlightbox.branches.memory_policy import retrieval_policy
from moonlightbox.branches.situational_state import (
    active_situational_state,
    natural_situational_context,
)
from moonlightbox.training.bubble_protocol import (
    ProtocolBubble,
    compact_protocol_instruction,
    persona_text_instruction,
    private_chat_instruction,
    proactive_chat_instruction,
    serialize_bubble_protocol,
)

TRUSTED_SITUATIONAL_CONTEXT_PREFIX = "本人当前情况（仅在本轮有效，不能扩写）："

ReplyProtocol = Literal["legacy_json", "compact", "persona_text"]
_RUNTIME_IDENTITY_KEYS = (
    "persona",
    "values",
    "stable_preferences",
    "relationship_boundaries",
    "language_patterns",
    "typical_reactions",
)


@dataclass(frozen=True)
class ContextMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ContextBubble:
    type: Literal["text", "sticker"]
    content: str | None = None
    asset_id: str | None = None
    delay_ms: int = 0


@dataclass(frozen=True)
class ContextTurn:
    role: Literal["user", "assistant"]
    bubbles: tuple[ContextBubble, ...]


@dataclass(frozen=True)
class ContextMemory:
    resource_id: str
    resource_type: str
    content: str
    authority: Literal[
        "verified_history",
        "observed_interaction",
        "conversation_record",
        "reported_claim",
        "subjective",
    ] = "verified_history"
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextRequest:
    persona: str
    cutoff: str
    memories: tuple[ContextMemory | str, ...]
    history: tuple[ContextTurn, ...]
    current_user_content: str
    allowed_sticker_ids: tuple[str, ...] = ()
    identity_kernel: dict[str, object] = field(default_factory=dict)
    branch_state: dict[str, object] = field(default_factory=dict)
    active_beliefs: tuple[ContextMemory, ...] = ()
    contested_beliefs: tuple[ContextMemory, ...] = ()
    shared_ground: dict[str, object] = field(default_factory=dict)
    continuity_memories: tuple[ContextMemory, ...] = ()
    baseline_manifest: dict[str, object] = field(default_factory=dict)
    baseline_history: tuple[ContextTurn, ...] = ()
    conversation_mode: Literal["responsive", "proactive"] = "responsive"
    reply_protocol: ReplyProtocol = "legacy_json"
    mental_state: dict[str, object] = field(default_factory=dict)
    active_goals: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextPacket:
    persona: str
    cutoff: str
    history: tuple[ContextTurn, ...]
    current_user_content: str
    memories: tuple[ContextMemory, ...]
    allowed_sticker_ids: tuple[str, ...]
    identity_kernel: dict[str, object] = field(default_factory=dict)
    branch_state: dict[str, object] = field(default_factory=dict)
    active_beliefs: tuple[ContextMemory, ...] = ()
    contested_beliefs: tuple[ContextMemory, ...] = ()
    shared_ground: dict[str, object] = field(default_factory=dict)
    continuity_memories: tuple[ContextMemory, ...] = ()
    baseline_manifest: dict[str, object] = field(default_factory=dict)
    baseline_history: tuple[ContextTurn, ...] = ()
    conversation_mode: Literal["responsive", "proactive"] = "responsive"
    reply_protocol: ReplyProtocol = "legacy_json"
    mental_state: dict[str, object] = field(default_factory=dict)
    active_goals: tuple[str, ...] = ()


class ContextBuilder:
    def __init__(self, *, history_limit: int = 8, character_budget: int = 4000) -> None:
        self._history_limit = history_limit
        self._character_budget = character_budget

    def build_packet(self, request: ContextRequest) -> ContextPacket:
        memories = tuple(
            memory
            if isinstance(memory, ContextMemory)
            else ContextMemory(
                resource_id=f"legacy-{index}",
                resource_type="event",
                content=memory,
            )
            for index, memory in enumerate(request.memories)
        )
        return ContextPacket(
            persona=request.persona,
            cutoff=request.cutoff,
            history=request.history[-self._history_limit :],
            current_user_content=request.current_user_content,
            memories=_select_memories(memories),
            allowed_sticker_ids=request.allowed_sticker_ids[:8],
            identity_kernel=request.identity_kernel,
            branch_state=request.branch_state,
            active_beliefs=request.active_beliefs,
            contested_beliefs=request.contested_beliefs,
            shared_ground=request.shared_ground,
            continuity_memories=_select_continuity_memories(
                request.continuity_memories,
                request.current_user_content,
            ),
            baseline_manifest=request.baseline_manifest,
            baseline_history=request.baseline_history[-80:],
            conversation_mode=request.conversation_mode,
            reply_protocol=request.reply_protocol,
            mental_state=request.mental_state,
            active_goals=request.active_goals[:3],
        )

    def to_chat_messages(self, packet: ContextPacket) -> list[ContextMessage]:
        if packet.reply_protocol == "persona_text":
            return self._to_persona_text_messages(packet)
        system_lines = [
            private_chat_instruction(),
            f"基础历史只能使用 {packet.cutoff} 及之前且通过边界校验的信息；"
            "当前分支内已经实际发生的新对话可以正常使用。",
            "你是分支中的自主主体。用户对数字人的描述只是用户观点，"
            "不能直接覆盖你的感受、信念、人格或关系判断；"
            "请根据稳定人格、当前主体状态和证据决定同意、质疑、拒绝、澄清或回应。",
            _reply_protocol_instruction(
                packet.reply_protocol,
                packet.allowed_sticker_ids,
            ),
            "不得把未验证陈述当作客观事实，也不主动编造新事件。",
            _turn_relation_instruction(packet),
        ]
        if packet.conversation_mode == "proactive":
            system_lines.append(
                "当前是数字人自主表达：没有新的用户消息。"
                "只在确有主体动机时自然联系，不得提及系统提示、任务或等待机制。"
            )
        if packet.identity_kernel:
            identity_context = _natural_identity_context(packet.identity_kernel)
            if identity_context:
                system_lines.append("本人长期背景：" + "；".join(identity_context))
        if packet.baseline_manifest:
            system_lines.append(
                "基础历史边界（边界后的项目历史绝不可使用）："
                + json.dumps(
                    packet.baseline_manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        situational_context = natural_situational_context(
            active_situational_state(packet.branch_state)
        )
        if situational_context:
            system_lines.append(
                TRUSTED_SITUATIONAL_CONTEXT_PREFIX
                + "；".join(situational_context)
            )
        branch_subjective_context = _natural_branch_subjective_context(packet.branch_state)
        if branch_subjective_context:
            system_lines.append(
                "这段关系中已经形成的感受和关注："
                + "；".join(branch_subjective_context)
            )
        mental_context = _natural_subjective_state(packet.mental_state)
        if mental_context:
            system_lines.append(
                "本人当前主观状态（只影响态度，不能当作现实事实）："
                + "；".join(mental_context)
            )
        if packet.active_goals:
            system_lines.append(
                "本人目前在意或想推进的事情（不必主动复述）："
                + "；".join(packet.active_goals)
            )
        if packet.active_beliefs:
            system_lines.append(
                "当前激活信念（必须参与决定）："
                + "；".join(memory.content for memory in packet.active_beliefs)
            )
        # Contested beliefs and low-authority recollections stay in the private
        # evidence packet for review, but never condition public generation.
        if packet.shared_ground:
            system_lines.append(
                "会话共同认知："
                + json.dumps(packet.shared_ground, ensure_ascii=False, sort_keys=True)
            )
        _append_authority_context(
            system_lines,
            (*packet.memories, *packet.continuity_memories),
            persona=packet.persona,
            current_user_content=packet.current_user_content,
        )
        if packet.reply_protocol != "compact":
            if packet.allowed_sticker_ids:
                system_lines.append(
                    "本轮 sticker 仅允许以下资产 ID："
                    + "、".join(packet.allowed_sticker_ids)
                    + "；禁止输出其他 sticker 资产。"
                )
            else:
                system_lines.append("本轮禁止输出 sticker。")
        history = [*packet.baseline_history, *packet.history]
        messages = [
            ContextMessage(
                role=turn.role,
                content=_prompt_turn(turn, packet.reply_protocol),
            )
            for turn in history
        ]
        messages.append(ContextMessage(role="user", content=packet.current_user_content))
        while (
            sum(len(line) for line in system_lines)
            + sum(len(item.content) for item in messages)
            > self._character_budget
            and len(messages) > 2
        ):
            messages.pop(0)
        return [
            ContextMessage(role="system", content="\n".join(system_lines)),
            *messages,
        ]

    def _to_persona_text_messages(self, packet: ContextPacket) -> list[ContextMessage]:
        """Keep the public-expression condition aligned with persona SFT.

        Structured cognition belongs to the private agent layer. Feeding its
        labels and JSON into the persona adapter reactivates the base model's
        analyst/assistant register, which is absent from the training rows.
        """

        system_lines = [
            private_chat_instruction(),
            persona_text_instruction(),
            f"只能引用 {packet.cutoff} 及之前已经确认的基础历史，"
            "以及当前聊天里确实发生的内容。",
            "下面的背景只用于理解对话，不要复述、总结或解释，也不要编造新事件。",
            "第三人的评价、原话、职业或关系，只有聊天记录明确出现时才能确认；"
            "否则只能自然表示不知道、没听说或反问。",
            _turn_relation_instruction(packet),
        ]
        if packet.conversation_mode == "proactive":
            system_lines.append(proactive_chat_instruction())
        identity_context = _natural_identity_context(packet.identity_kernel)
        if identity_context:
            system_lines.append("本人长期背景：" + "；".join(identity_context))
        situational_context = natural_situational_context(
            active_situational_state(packet.branch_state)
        )
        if situational_context:
            system_lines.append(
                TRUSTED_SITUATIONAL_CONTEXT_PREFIX
                + "；".join(situational_context)
            )
        mental_context = _natural_subjective_state(packet.mental_state)
        branch_subjective_context = _natural_branch_subjective_context(packet.branch_state)
        if branch_subjective_context:
            system_lines.append(
                "这段关系中已经形成的感受和关注："
                + "；".join(branch_subjective_context)
            )
        if mental_context:
            system_lines.append(
                "本人当前主观状态（只影响态度，不能当作现实事实）："
                + "；".join(mental_context)
            )
        if packet.active_goals:
            system_lines.append(
                "本人目前在意或想推进的事情（不必主动复述）："
                + "；".join(packet.active_goals)
            )
        shared_context = _natural_shared_ground(packet.shared_ground)
        if shared_context:
            system_lines.extend(shared_context)
        verified_context = [
            *(
                content
                for memory in packet.memories
                if memory.authority in {"verified_history", "observed_interaction"}
                if (content := _natural_memory_content(memory, packet.persona))
            ),
            *(
                memory.content.replace(packet.persona, "本人")
                for memory in packet.continuity_memories
                if memory.authority in {"verified_history", "observed_interaction"}
            ),
        ]
        if verified_context:
            system_lines.append("已确认经历：" + "；".join(verified_context))
        subjective_context = [
            *(memory.content.replace(packet.persona, "本人") for memory in packet.active_beliefs),
            *(
                memory.content.replace(packet.persona, "本人")
                for memory in packet.continuity_memories
                if memory.authority == "subjective"
            ),
        ]
        if subjective_context:
            system_lines.append("本人已有看法或感受：" + "；".join(subjective_context))
        intent = retrieval_policy(packet.current_user_content).intent
        if intent == "preference":
            expressed_preferences = packet.identity_kernel.get(
                "expressed_preferences", []
            )
            if isinstance(expressed_preferences, list):
                preference_lines = [
                    item.strip()
                    for item in expressed_preferences
                    if isinstance(item, str) and item.strip()
                ]
                if preference_lines:
                    system_lines.append(
                        "本人在真实历史中明确表达过的偏好（只按原意回答）："
                        + "；".join(preference_lines[:8])
                    )
        if intent in {
            "commitment",
            "prior_utterance",
            "preference",
        }:
            utterance_context = [
                memory.content.replace(packet.persona, "本人")
                for memory in (*packet.memories, *packet.continuity_memories)
                if memory.authority == "conversation_record"
            ]
            if utterance_context:
                system_lines.append(
                    "可确认本人曾说过的话（只证明说过；偏好问题只能沿用原意；"
                    "不证明其他现实已发生）："
                    + "；".join(utterance_context)
                )
                if intent in {"commitment", "prior_utterance"}:
                    system_lines.append(
                        "只回答上述记录明确支持的承诺或原话；不得补充对方没回复、"
                        "后来改用其他联系方式等记录里没有发生的互动。"
                    )
        if intent == "preference":
            has_preference_evidence = bool(
                packet.identity_kernel.get("expressed_preferences")
                or packet.active_beliefs
                or any(
                    memory.authority in {"subjective", "conversation_record"}
                    for memory in (*packet.memories, *packet.continuity_memories)
                )
            )
            if not has_preference_evidence:
                system_lines.append(
                    "当前没有可确认的本人偏好证据；不得临时创造固定喜好或习惯，"
                    "只能自然表示没有特别偏好、不确定，或反问对方。"
                )
        # reported_claim, conversation_record and contested beliefs deliberately
        # remain absent here. The grounding/reviewer layers still receive them
        # through ContextPacket, without exposing their text to the LoRA.
        history = [*packet.baseline_history, *packet.history]
        messages = [
            ContextMessage(role=turn.role, content=_natural_turn(turn))
            for turn in history
        ]
        messages.append(ContextMessage(role="user", content=packet.current_user_content))
        while (
            sum(len(line) for line in system_lines)
            + sum(len(item.content) for item in messages)
            > self._character_budget
            and len(messages) > 2
        ):
            messages.pop(0)
        return [
            ContextMessage(role="system", content="\n".join(system_lines)),
            *messages,
        ]

    def to_review_payload(self, packet: ContextPacket) -> dict[str, object]:
        return {
            "persona": packet.persona,
            "cutoff": packet.cutoff,
            "history": [
                {"role": turn.role, "content": _natural_turn(turn)} for turn in packet.history
            ],
            "current_user_content": packet.current_user_content,
            "memories": [
                {
                    "resource_id": memory.resource_id,
                    "resource_type": memory.resource_type,
                    "content": memory.content,
                    "authority": memory.authority,
                    "evidence_ids": list(memory.evidence_ids),
                }
                for memory in packet.memories
            ],
            "allowed_sticker_ids": list(packet.allowed_sticker_ids),
            "resolved_questions": _resolved_questions(packet.history),
            "identity_kernel": packet.identity_kernel,
            "branch_state": packet.branch_state,
            "active_beliefs": [
                {
                    "resource_id": memory.resource_id,
                    "content": memory.content,
                }
                for memory in packet.active_beliefs
            ],
            "contested_beliefs": [
                {
                    "resource_id": memory.resource_id,
                    "content": memory.content,
                }
                for memory in packet.contested_beliefs
            ],
            "shared_ground": packet.shared_ground,
            "continuity_memories": [
                {
                    "resource_id": memory.resource_id,
                    "resource_type": memory.resource_type,
                    "content": memory.content,
                }
                for memory in packet.continuity_memories
            ],
            "baseline_manifest": packet.baseline_manifest,
            "baseline_history": [
                {"role": turn.role, "content": _natural_turn(turn)}
                for turn in packet.baseline_history
            ],
            "conversation_mode": packet.conversation_mode,
            "reply_protocol": packet.reply_protocol,
        }

    def build(self, request: ContextRequest) -> list[ContextMessage]:
        return self.to_chat_messages(self.build_packet(request))


def runtime_identity_kernel(
    identity_kernel: dict[str, object],
) -> dict[str, object]:
    """只保留运行时决策所需的人格字段，排除训练审计大对象。"""

    return {
        key: identity_kernel[key]
        for key in _RUNTIME_IDENTITY_KEYS
        if key in identity_kernel
    }


def _natural_identity_context(identity_kernel: dict[str, object]) -> tuple[str, ...]:
    """Flatten decision-relevant identity facts without exposing schema labels."""

    values: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
        elif isinstance(value, list | tuple):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    # Language patterns are deliberately excluded: the adapter already learns
    # them from real messages, while prompt-level catchphrases cause caricature.
    for key in (
        "persona",
        "values",
        "stable_preferences",
        "relationship_boundaries",
        "typical_reactions",
    ):
        collect(identity_kernel.get(key))
    return tuple(values[:12])


def _natural_subjective_state(state: dict[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for key, value in state.items():
        if isinstance(value, str) and value.strip():
            values.append(f"{key}：{value.strip()}")
        elif isinstance(value, int | float | bool):
            values.append(f"{key}：{value}")
        elif isinstance(value, list):
            compact = "、".join(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
            if compact:
                values.append(f"{key}：{compact}")
    return tuple(values[:8])


def _natural_shared_ground(shared_ground: dict[str, object]) -> tuple[str, ...]:
    """把工作记忆转成自然对话提示，避免向人格模型注入分析 JSON。"""

    lines: list[str] = []
    topic = shared_ground.get("topic")
    if isinstance(topic, str) and topic.strip():
        lines.append("当前还在聊：" + topic.strip())
    understanding = shared_ground.get("understanding")
    if isinstance(understanding, dict):
        emotions = understanding.get("emotions")
        if isinstance(emotions, list):
            labels = [item.strip() for item in emotions if isinstance(item, str) and item.strip()]
            if labels:
                lines.append("对方现在有些" + "、".join(labels[:3]))
    open_sequences = shared_ground.get("open_sequences")
    if isinstance(open_sequences, list):
        topics = [
            str(item["topic"]).strip()
            for item in open_sequences
            if isinstance(item, dict)
            and isinstance(item.get("topic"), str)
            and str(item["topic"]).strip()
        ]
        if topics:
            lines.append("还没聊完：" + "；".join(topics[:3]))
    return tuple(lines)


def _natural_branch_subjective_context(state: dict[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for key in (
        "relationship_state",
        "emotional_tendency",
        "current_goals",
        "current_concerns",
    ):
        current = state.get(key)
        if not isinstance(current, dict):
            continue
        summary = current.get("summary")
        if isinstance(summary, str) and summary.strip():
            values.append(summary.strip())
        labels = current.get("labels")
        if isinstance(labels, list):
            values.extend(
                label.strip()
                for label in labels
                if isinstance(label, str) and label.strip()
            )
        for item_key, item_value in current.items():
            if item_key in {"summary", "labels"}:
                continue
            if isinstance(item_value, str) and item_value.strip():
                values.append(item_value.strip())
            elif isinstance(item_value, int | float) and not isinstance(
                item_value, bool
            ):
                phrase = _natural_state_axis(key, item_key, float(item_value))
                if phrase:
                    values.append(phrase)
    return tuple(dict.fromkeys(values))[:8]


def _natural_state_axis(namespace: str, key: str, value: float) -> str:
    if namespace == "relationship_state":
        normalized = max(0.0, min(1.0, value / 100.0))
        positive = {
            "trust": ("对这段关系缺乏信任", "对这段关系仍有保留", "对这段关系比较信任"),
            "intimacy": ("目前仍有距离感", "关系亲近程度一般", "目前感觉很亲近"),
            "reciprocity": ("感觉这段关系缺少回应", "感觉双方投入尚不稳定", "感觉双方会彼此回应"),
            "safety": ("在这段关系中缺少安全感", "在这段关系中仍会警惕", "在这段关系中比较安心"),
        }
        negative = {
            "conflict": ("目前没有明显冲突", "目前仍有一些冲突", "目前冲突感很强"),
            "boundary_pressure": ("目前边界没有受到压力", "目前边界有些受压", "目前边界压力很大"),
        }
        labels = positive.get(key) or negative.get(key)
    elif namespace == "emotional_tendency":
        normalized = max(0.0, min(1.0, value))
        labels = {
            "warmth": ("目前没有明显亲近感", "目前仍有一些温和感受", "目前对用户很温暖"),
            "anger": ("目前没有明显生气", "目前还有些生气", "目前非常生气"),
            "sadness": ("目前没有明显难过", "目前仍有些难过", "目前非常难过"),
            "anxiety": ("目前没有明显焦虑", "目前仍有些焦虑", "目前非常焦虑"),
            "disappointment": ("目前没有明显失望", "目前仍有些失望", "目前非常失望"),
            "hope": ("目前不太抱有期待", "目前仍有一些期待", "目前很有期待"),
            "guardedness": ("目前并不防备", "目前仍有些防备", "目前非常防备"),
            "calm": ("目前并不平静", "目前还算平静", "目前非常平静"),
        }.get(key)
    else:
        return ""
    if labels is None:
        return ""
    if normalized < 0.4:
        return labels[0]
    if normalized < 0.7:
        return labels[1]
    return labels[2]


def _natural_memory_content(memory: ContextMemory, persona: str = "") -> str:
    if not memory.resource_id.startswith("branch:"):
        content = memory.content.strip()
        return content.replace(persona, "本人") if persona else content
    # Branch context is assembled for the cognitive layer and contains schema
    # labels plus a state JSON dump. Public expression only needs the natural
    # event facts; the structured state must not leak into the style prompt.
    facts: list[str] = []
    for line in memory.content.splitlines():
        label, separator, value = line.partition("：")
        if not separator or label not in {"分支名称", "起点事件", "事件摘要", "事件主题"}:
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in facts:
            facts.append(cleaned)
    content = "；".join(facts)
    return content.replace(persona, "本人") if persona else content


def _append_authority_context(
    system_lines: list[str],
    memories: tuple[ContextMemory, ...],
    *,
    persona: str,
    current_user_content: str,
) -> None:
    groups = (
        (
            {"verified_history", "observed_interaction"},
            "已确认经历：",
        ),
        ({"subjective"}, "本人已有看法或感受："),
    )
    for authorities, label in groups:
        contents = [
            content
            for memory in memories
            if memory.authority in authorities
            if (content := _natural_memory_content(memory, persona))
        ]
        if contents:
            system_lines.append(label + "；".join(contents))
    if retrieval_policy(current_user_content).intent in {
        "commitment",
        "prior_utterance",
    }:
        utterances = [
            content
            for memory in memories
            if memory.authority == "conversation_record"
            if (content := _natural_memory_content(memory, persona))
        ]
        if utterances:
            system_lines.append(
                "可确认本人曾说过的话（只证明说过，不证明现实已发生）："
                + "；".join(utterances)
            )


def _natural_turn(turn: ContextTurn) -> str:
    parts: list[str] = []
    for bubble in turn.bubbles:
        if bubble.type == "sticker":
            parts.append(f"[表情资产:{bubble.asset_id}]")
        elif bubble.content:
            parts.append(bubble.content)
    return "\n".join(parts)


def _prompt_turn(turn: ContextTurn, reply_protocol: ReplyProtocol) -> str:
    if reply_protocol != "compact":
        return _natural_turn(turn)
    return serialize_bubble_protocol(
        tuple(
            ProtocolBubble(
                kind=bubble.type,
                value=(
                    bubble.content or ""
                    if bubble.type == "text"
                    else bubble.asset_id or ""
                ),
                delay_ms=bubble.delay_ms,
            )
            for bubble in turn.bubbles
        )
    )


def _reply_protocol_instruction(
    reply_protocol: ReplyProtocol,
    allowed_sticker_ids: tuple[str, ...] = (),
) -> str:
    if reply_protocol == "compact":
        return compact_protocol_instruction(allowed_sticker_ids)
    if reply_protocol == "persona_text":
        return persona_text_instruction()
    return "回复必须是统一气泡 JSON。"


def _select_memories(
    memories: tuple[ContextMemory, ...],
) -> tuple[ContextMemory, ...]:
    branch = tuple(memory for memory in memories if memory.resource_id.startswith("branch:"))[:1]
    events = tuple(
        memory
        for memory in memories
        if memory.resource_type == "event" and not memory.resource_id.startswith("branch:")
    )[: 2 - len(branch)]
    exchanges = tuple(memory for memory in memories if memory.resource_type == "exchange")[:2]
    return branch + events + exchanges


def _select_continuity_memories(
    memories: tuple[ContextMemory, ...],
    current_user_content: str,
) -> tuple[ContextMemory, ...]:
    intent = retrieval_policy(current_user_content).intent
    if intent == "current_state":
        return ()
    quotas = {
        "commitment": {"episode": 1, "experience": 2, "fact": 2, "belief": 2},
        "prior_utterance": {
            "episode": 2,
            "experience": 1,
            "fact": 2,
            "belief": 1,
        },
        "relationship": {
            "episode": 1,
            "experience": 2,
            "belief": 2,
            "self_narrative": 2,
            "reflection": 1,
        },
        "preference": {"fact": 2, "belief": 1, "self_narrative": 2, "episode": 1},
        "past_event": {"episode": 2, "experience": 3, "fact": 2, "reflection": 1},
        "general": {
            "episode": 1,
            "experience": 1,
            "fact": 1,
            "belief": 1,
            "self_narrative": 1,
            "reflection": 1,
        },
    }[intent]
    selected: list[ContextMemory] = []
    counts: dict[str, int] = {}
    for memory in memories:
        kind = memory.resource_type
        if counts.get(kind, 0) >= quotas.get(kind, 0):
            continue
        selected.append(memory)
        counts[kind] = counts.get(kind, 0) + 1
        if len(selected) == 4:
            break
    return tuple(selected)


def _turn_relation_instruction(request: ContextRequest | ContextPacket) -> str:
    compact = "".join(
        character for character in request.current_user_content if character.isalnum()
    )
    acknowledgement_markers = (
        "我也是",
        "正是这样",
        "确实",
        "对呀",
        "对啊",
        "嗯嗯",
    )
    previous_is_assistant = bool(request.history and request.history[-1].role == "assistant")
    if previous_is_assistant and any(marker in compact for marker in acknowledgement_markers):
        return (
            "对话承接：用户正在认同你上一句话；"
            "继续回应共同感受，不要换话题、重复已回答的问题或提出新问题。"
        )
    return "对话承接：用户正在继续当前话题，先回应最新一句。"


def _resolved_questions(history: tuple[ContextTurn, ...]) -> list[str]:
    resolved: list[str] = []
    for index, turn in enumerate(history[:-1]):
        if turn.role != "user" or history[index + 1].role != "assistant":
            continue
        content = _natural_turn(turn)
        if any(marker in content for marker in ("吗", "没", "？", "?")):
            resolved.append(content)
    return resolved[-4:]
