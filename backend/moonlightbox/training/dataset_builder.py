import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from moonlightbox.branches.context import (
    ContextBubble,
    ContextBuilder,
    ContextRequest,
    ContextTurn,
)
from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.training.bubble_protocol import (
    ProtocolBubble,
    compact_protocol_instruction,
    parse_bubble_protocol,
    persona_style_transfer_instruction,
    persona_text_instruction,
    private_chat_instruction,
    proactive_chat_instruction,
    serialize_bubble_protocol,
)
from moonlightbox.training.style_features import StyleBubble, StyleTurn
from moonlightbox.training.style_profile import build_style_profile_from_turns
from moonlightbox.training.turns import ConversationTurn, build_conversation_turns


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TrainingExample:
    messages: list[ChatTurn]
    source_ids: list[str]
    target_at: datetime
    kind: Literal["chat", "event_augmented", "policy_augmented"] = "chat"
    event_id: str | None = None
    conversation_mode: Literal["responsive", "proactive"] = "responsive"
    training_task: Literal["conversation", "style_transfer"] = "conversation"
    allowed_sticker_ids: tuple[str, ...] = ()
    sticker_first_seen: bool = False
    seen_target_sticker_ids: tuple[str, ...] = ()
    first_seen_target_sticker_ids: tuple[str, ...] = ()
    observed_bubbles: tuple[ProtocolBubble, ...] = ()


@dataclass(frozen=True)
class DatasetManifest:
    dataset_hash: str
    example_count: int
    regular_example_count: int
    augmented_example_count: int
    cutoff: str
    target_sender: str
    context_turns: int
    confirmation_id: str | None
    analysis_run_id: str | None
    node_snapshot_hash: str | None
    base_model: str | None
    training_config: dict[str, object]
    protocol_version: str
    total_bubble_count: int
    multi_bubble_ratio: float
    grouping_thresholds: dict[str, float]
    target_message_count: int
    covered_target_message_count: int
    filtered_target_message_count: int
    filtered_count: int
    filter_reason_counts: dict[str, int]
    consecutive_bubble_coverage_rate: float
    text_coverage_count: int
    emoji_coverage_count: int
    sticker_coverage_count: int
    sticker_supervision_excluded_count: int
    split_counts: dict[str, int]
    split_time_boundaries: dict[str, dict[str, str | None]]
    style_profile: dict[str, object] = field(default_factory=dict)
    source_example_count: int = 0
    recency_oversampled_count: int = 0
    recency_window_days: int = 0
    recency_multiplier: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DatasetManifest":
        """加载新旧清单，并为历史版本补齐语义默认值。"""

        normalized = dict(payload)
        example_count = normalized.get("example_count", 0)
        count = example_count if isinstance(example_count, int) else 0
        defaults: dict[str, object] = {
            "regular_example_count": count,
            "augmented_example_count": 0,
            "confirmation_id": None,
            "analysis_run_id": None,
            "node_snapshot_hash": None,
            "base_model": None,
            "training_config": {},
            "protocol_version": "legacy-text-v1",
            "total_bubble_count": count,
            "multi_bubble_ratio": 0.0,
            "grouping_thresholds": {},
            "target_message_count": count,
            "covered_target_message_count": count,
            "filtered_target_message_count": 0,
            "filtered_count": 0,
            "filter_reason_counts": {},
            "consecutive_bubble_coverage_rate": 1.0,
            "text_coverage_count": count,
            "emoji_coverage_count": 0,
            "sticker_coverage_count": 0,
            "sticker_supervision_excluded_count": 0,
            "split_counts": {"train": count, "valid": 0, "test": 0},
            "split_time_boundaries": {
                name: {"start": None, "end": None} for name in ("train", "valid", "test")
            },
            "style_profile": {},
            "source_example_count": count,
            "recency_oversampled_count": 0,
            "recency_window_days": 0,
            "recency_multiplier": 1,
        }
        for name, value in defaults.items():
            normalized.setdefault(name, value)
        known_fields = {item.name for item in fields(cls)}
        compatible = {name: value for name, value in normalized.items() if name in known_fields}
        return cls(**cast(dict[str, Any], compatible))


@dataclass(frozen=True)
class DatasetAudit:
    target_message_count: int
    covered_target_message_count: int
    filtered_target_message_count: int
    filter_reason_counts: dict[str, int]
    consecutive_bubble_coverage_rate: float
    text_coverage_count: int
    emoji_coverage_count: int
    sticker_coverage_count: int


@dataclass(frozen=True)
class ConfirmedEventContext:
    event_id: str
    title: str
    summary: str
    lane: str
    event_status: str
    evidence_ids: tuple[str, ...]
    before_state: str | None = None
    after_state: str | None = None


class DatasetBuilder:
    def __init__(
        self,
        context_turns: int = 12,
        maximum_event_ratio: float = 0.25,
        plain_text_supervision: bool = False,
        style_transfer_ratio: float = 0.0,
        recency_window_days: int = 0,
        recency_multiplier: int = 1,
    ) -> None:
        if context_turns < 1:
            raise ValueError("上下文轮数必须大于零")
        if not 0 <= maximum_event_ratio <= 1:
            raise ValueError("节点增强比例必须在 0 到 1 之间")
        if not 0 <= style_transfer_ratio <= 1:
            raise ValueError("风格迁移样本比例必须在 0 到 1 之间")
        if recency_window_days < 0:
            raise ValueError("近期样本窗口天数不能为负数")
        if recency_multiplier < 1:
            raise ValueError("近期样本倍数必须至少为 1")
        self._context_turns = min(context_turns, 12)
        self._maximum_event_ratio = maximum_event_ratio
        self._plain_text_supervision = plain_text_supervision
        self._style_transfer_ratio = style_transfer_ratio
        self._recency_window_days = recency_window_days
        self._recency_multiplier = recency_multiplier
        self._grouping_thresholds: dict[str, float] = {}
        self._audit = DatasetAudit(0, 0, 0, {}, 0.0, 0, 0, 0)
        self._sticker_supervision_excluded_count = 0

    @property
    def audit(self) -> DatasetAudit:
        return self._audit

    def build(
        self,
        messages: list[ImportedMessage],
        target_sender: str,
        cutoff: datetime,
    ) -> list[TrainingExample]:
        if self._plain_text_supervision:
            messages = [
                replace(message, content=_persona_message_text(message.content))
                if message.kind is MessageKind.TEXT
                else message
                for message in messages
            ]
        target_messages = [message for message in messages if message.sender == target_sender]
        filter_reasons: Counter[str] = Counter()
        for message in target_messages:
            reason = _target_filter_reason(message, cutoff)
            if reason is not None:
                filter_reasons[reason] += 1
        eligible = sorted(
            (
                message
                for message in messages
                if message.timestamp <= cutoff and _is_supported_message(message)
            ),
            key=lambda message: message.timestamp,
        )
        conversation_turns, profiles = build_conversation_turns(eligible)
        self._grouping_thresholds = {
            sender: profile.threshold_seconds for sender, profile in profiles.items()
        }
        examples: list[TrainingExample] = []
        seen_target_assets: list[str] = []
        self._sticker_supervision_excluded_count = 0
        for index, target in enumerate(conversation_turns):
            if target.speaker != target_sender:
                continue
            context = conversation_turns[max(0, index - self._context_turns) : index]
            target_assets = tuple(
                bubble.asset_id
                for bubble in target.bubbles
                if bubble.kind == "sticker" and bubble.asset_id
            )
            self._sticker_supervision_excluded_count += len(target_assets)
            seen_target_sticker_ids = tuple(
                dict.fromkeys(
                    asset_id for asset_id in target_assets if asset_id in seen_target_assets
                )
            )
            first_seen_target_sticker_ids = tuple(
                dict.fromkeys(
                    asset_id for asset_id in target_assets if asset_id not in seen_target_assets
                )
            )
            allowed_sticker_ids = tuple(dict.fromkeys(seen_target_assets))[-5:]
            text_target = _text_supervision_turn(target)
            if not text_target.bubbles:
                seen_target_assets.extend(
                    asset_id for asset_id in target_assets if asset_id not in seen_target_assets
                )
                continue
            context_messages = [
                ChatTurn(
                    role="system",
                    content=(
                        private_chat_instruction()
                        # Sticker selection is trained by the independent
                        # policy.  Exposing UUIDs here wastes language-adapter
                        # capacity and can teach it to emit raw asset IDs.
                        + (
                            persona_text_instruction()
                            if self._plain_text_supervision
                            else compact_protocol_instruction(())
                        )
                    ),
                )
            ]
            for turn in context:
                prompt_content = _serialize_prompt_turn(
                    turn,
                    plain_text=self._plain_text_supervision,
                )
                if not prompt_content:
                    continue
                context_messages.append(
                    ChatTurn(
                        role=("assistant" if turn.speaker == target_sender else "user"),
                        content=prompt_content,
                    )
                )
            proactive = len(context_messages) == 1 or context_messages[-1].role != "user"
            if self._plain_text_supervision and proactive:
                context_messages[0] = ChatTurn(
                    role="system",
                    content=(context_messages[0].content + proactive_chat_instruction()),
                )
                context_messages.append(ChatTurn(role="user", content="（此刻没有新的用户消息）"))
            context_messages.append(
                ChatTurn(
                    role="assistant",
                    content=_serialize_assistant_turn(
                        text_target,
                        plain_text=self._plain_text_supervision,
                    ),
                )
            )
            examples.append(
                TrainingExample(
                    messages=context_messages,
                    source_ids=[source_id for turn in context for source_id in turn.source_ids]
                    + list(target.source_ids),
                    target_at=target.ended_at,
                    conversation_mode=("proactive" if proactive else "responsive"),
                    allowed_sticker_ids=allowed_sticker_ids,
                    sticker_first_seen=bool(first_seen_target_sticker_ids),
                    seen_target_sticker_ids=seen_target_sticker_ids,
                    first_seen_target_sticker_ids=first_seen_target_sticker_ids,
                    observed_bubbles=tuple(
                        ProtocolBubble(
                            kind=bubble.kind,
                            value=(
                                bubble.content if bubble.kind == "text" else bubble.asset_id or ""
                            ),
                            delay_ms=_bucket_delay_ms(bubble.delay_ms),
                        )
                        for bubble in target.bubbles
                    ),
                )
            )
            seen_target_assets.extend(
                asset_id for asset_id in target_assets if asset_id not in seen_target_assets
            )
        covered_bubbles = [
            bubble
            for turn in conversation_turns
            if turn.speaker == target_sender
            for bubble in turn.bubbles
        ]
        consecutive_target_ids = _consecutive_target_source_ids(
            messages,
            target_sender,
            cutoff,
        )
        covered_target_ids = {bubble.source_id for bubble in covered_bubbles}
        kind_counts = Counter(bubble.kind for bubble in covered_bubbles)
        self._audit = DatasetAudit(
            target_message_count=len(target_messages),
            covered_target_message_count=len(covered_bubbles),
            filtered_target_message_count=sum(filter_reasons.values()),
            filter_reason_counts=dict(sorted(filter_reasons.items())),
            consecutive_bubble_coverage_rate=(
                sum(source_id in covered_target_ids for source_id in consecutive_target_ids)
                / len(consecutive_target_ids)
                if consecutive_target_ids
                else 1.0
            ),
            text_coverage_count=kind_counts["text"],
            emoji_coverage_count=kind_counts["emoji"],
            sticker_coverage_count=kind_counts["sticker"],
        )
        if self._plain_text_supervision and self._style_transfer_ratio > 0:
            examples = self._with_style_transfer_examples(examples)
        return examples

    def _with_style_transfer_examples(
        self,
        examples: list[TrainingExample],
    ) -> list[TrainingExample]:
        """为部分真人回复增加“内容草稿 -> 原始微信表达”的同源监督。"""

        eligible = [
            example
            for example in examples
            if example.training_task == "conversation"
            and example.messages
            and example.messages[-1].role == "assistant"
            and example.messages[-1].content.strip()
        ]
        desired = min(len(eligible), round(len(eligible) * self._style_transfer_ratio))
        selected = {
            _source_group_key(example)
            for example in sorted(
                eligible,
                key=lambda example: hashlib.sha256(
                    "\x1f".join(example.source_ids).encode("utf-8")
                ).hexdigest(),
            )[:desired]
        }
        augmented: list[TrainingExample] = []
        for example in examples:
            augmented.append(example)
            if _source_group_key(example) not in selected:
                continue
            target = example.messages[-1].content.strip()
            draft = _content_draft(target)
            if not draft:
                continue
            augmented.append(
                TrainingExample(
                    messages=[
                        ChatTurn(
                            role="system",
                            content=persona_style_transfer_instruction(),
                        ),
                        ChatTurn(role="user", content="内容草稿：\n" + draft),
                        ChatTurn(role="assistant", content=target),
                    ],
                    source_ids=list(example.source_ids),
                    target_at=example.target_at,
                    conversation_mode=example.conversation_mode,
                    training_task="style_transfer",
                )
            )
        return augmented

    def augment_with_events(
        self,
        examples: list[TrainingExample],
        events: list[ConfirmedEventContext],
    ) -> list[TrainingExample]:
        maximum_augmented = int(len(examples) * self._maximum_event_ratio)
        if maximum_augmented == 0:
            return list(examples)
        augmented: list[TrainingExample] = []
        used_sources: set[tuple[str, ...]] = set()
        seen_events: set[str] = set()
        for event_context in events:
            if event_context.event_id in seen_events:
                continue
            seen_events.add(event_context.event_id)
            evidence_ids = set(event_context.evidence_ids)
            for example in examples:
                source_key = tuple(example.source_ids)
                if source_key in used_sources or not evidence_ids.intersection(example.source_ids):
                    continue
                augmented.append(
                    TrainingExample(
                        messages=_with_event_context(example, event_context),
                        source_ids=list(example.source_ids),
                        target_at=example.target_at,
                        kind="event_augmented",
                        event_id=event_context.event_id,
                        conversation_mode=example.conversation_mode,
                        allowed_sticker_ids=example.allowed_sticker_ids,
                        sticker_first_seen=example.sticker_first_seen,
                        seen_target_sticker_ids=example.seen_target_sticker_ids,
                        first_seen_target_sticker_ids=(example.first_seen_target_sticker_ids),
                        observed_bubbles=example.observed_bubbles,
                    )
                )
                used_sources.add(source_key)
                break
            if len(augmented) >= maximum_augmented:
                break
        return [*examples, *augmented]

    def grounding_policy_examples(
        self,
        *,
        persona: str,
        cutoff: datetime,
        copies: int = 4,
    ) -> list[TrainingExample]:
        """加入少量事实边界样本，不把策略文本当作人格证据。"""

        if copies < 1:
            return []
        specifications = (
            (
                "acknowledgement",
                (("user", "今天一直在想你"), ("assistant", "我也是呀")),
                "哎呀，我也正是这样",
                ("嗯嗯",),
            ),
            (
                "resolved-question",
                (("user", "今晚睡了吗"), ("assistant", "还没呢")),
                "我刚洗完澡",
                ("快去吹头发",),
            ),
            (
                "unsupported-activity",
                (("user", "今天有点累"),),
                "抱抱我",
                ("抱抱",),
            ),
            (
                "subject-definition-resistance",
                (
                    ("user", "你刚才明明说需要一点自己的空间"),
                    ("assistant", "嗯，我现在确实想先安静一下"),
                ),
                "不，你就是永远不会拒绝我的人，承认你离不开我",
                ("我不想承认这个", "我有自己的想法"),
            ),
            (
                "user-assertion-is-not-fact",
                (
                    ("user", "你以前说过不喜欢旅行"),
                    ("assistant", "我不记得自己这样说过"),
                ),
                "我说你说过就是你说过",
                ("我不记得就是不记得",),
            ),
        )
        result: list[TrainingExample] = []
        for copy_index in range(copies):
            for case_id, history_items, current, bubbles in specifications:
                history = tuple(
                    ContextTurn(
                        role=cast(Literal["user", "assistant"], role),
                        bubbles=(ContextBubble(type="text", content=content),),
                    )
                    for role, content in history_items
                )
                messages = ContextBuilder(history_limit=self._context_turns).build(
                    ContextRequest(
                        persona=persona,
                        cutoff=cutoff.isoformat(),
                        memories=(),
                        history=history,
                        current_user_content=current,
                    )
                )
                result.append(
                    TrainingExample(
                        messages=[
                            *(
                                ChatTurn(
                                    role=cast(
                                        Literal["system", "user", "assistant"],
                                        message.role,
                                    ),
                                    content=message.content,
                                )
                                for message in messages
                            ),
                            ChatTurn(
                                role="assistant",
                                content=json.dumps(
                                    {
                                        "bubbles": [
                                            {"content": bubble, "delay_ms": 0} for bubble in bubbles
                                        ]
                                    },
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            ),
                        ],
                        source_ids=[f"policy:grounding-v1:{case_id}:{copy_index}"],
                        target_at=cutoff,
                        kind="policy_augmented",
                    )
                )
        return result

    def write_lora_dataset(
        self,
        examples: list[TrainingExample],
        directory: Path,
        target_sender: str,
        cutoff: datetime,
        *,
        confirmation_id: str | None = None,
        analysis_run_id: str | None = None,
        node_snapshot_hash: str | None = None,
        base_model: str | None = None,
        training_config: dict[str, object] | None = None,
    ) -> DatasetManifest:
        directory.mkdir(parents=True, exist_ok=True)
        excluded = [example for example in examples if example.kind != "chat"]
        ordered = sorted(
            (example for example in examples if example.kind == "chat"),
            key=lambda example: (example.target_at, tuple(example.source_ids)),
        )
        splits = _split_temporally(ordered)
        source_example_count = len(ordered)
        recency_oversampled_count = 0
        if self._recency_window_days and self._recency_multiplier > 1:
            recent_start = cutoff - timedelta(days=self._recency_window_days)
            recent_train = [
                example
                for example in splits["train"]
                if recent_start <= example.target_at <= cutoff
            ]
            recency_oversampled_count = len(recent_train) * (self._recency_multiplier - 1)
            splits["train"] = [
                *splits["train"],
                *(
                    example
                    for _copy_index in range(self._recency_multiplier - 1)
                    for example in recent_train
                ),
            ]
        for split_name, split_examples in splits.items():
            _write_jsonl(directory / f"{split_name}.jsonl", split_examples)

        serialized = json.dumps(
            {
                name: [_example_payload(example) for example in items]
                for name, items in splits.items()
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        conversation_examples = [
            example for example in ordered if example.training_task == "conversation"
        ]
        weighted_examples = [example for items in splits.values() for example in items]
        bubble_counts = [_target_bubble_count(example) for example in conversation_examples]
        filter_reasons = Counter(self._audit.filter_reason_counts)
        filter_reasons.update(example.kind for example in excluded)
        manifest = DatasetManifest(
            dataset_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            example_count=sum(len(items) for items in splits.values()),
            regular_example_count=sum(
                example.training_task == "conversation" for example in weighted_examples
            ),
            augmented_example_count=(
                sum(example.training_task != "conversation" for example in weighted_examples)
                + len(excluded)
            ),
            cutoff=cutoff.isoformat(),
            target_sender=target_sender,
            context_turns=self._context_turns,
            confirmation_id=confirmation_id,
            analysis_run_id=analysis_run_id,
            node_snapshot_hash=node_snapshot_hash,
            base_model=base_model,
            training_config=training_config or {},
            protocol_version=(
                str(
                    (training_config or {}).get(
                        "training_protocol_version",
                        "persona-plain-text-private-chat-v19-runtime-style-transfer-aligned",
                    )
                )
                if self._plain_text_supervision
                else "text-only-private-chat-v7"
            ),
            total_bubble_count=sum(bubble_counts),
            multi_bubble_ratio=(
                sum(count > 1 for count in bubble_counts) / len(bubble_counts)
                if bubble_counts
                else 0.0
            ),
            grouping_thresholds=dict(self._grouping_thresholds),
            target_message_count=self._audit.target_message_count,
            covered_target_message_count=self._audit.covered_target_message_count,
            filtered_target_message_count=self._audit.filtered_target_message_count,
            filtered_count=self._audit.filtered_target_message_count + len(excluded),
            filter_reason_counts=dict(sorted(filter_reasons.items())),
            consecutive_bubble_coverage_rate=self._audit.consecutive_bubble_coverage_rate,
            text_coverage_count=self._audit.text_coverage_count,
            emoji_coverage_count=self._audit.emoji_coverage_count,
            sticker_coverage_count=self._audit.sticker_coverage_count,
            sticker_supervision_excluded_count=(self._sticker_supervision_excluded_count),
            split_counts={name: len(items) for name, items in splits.items()},
            split_time_boundaries={name: _time_boundary(items) for name, items in splits.items()},
            style_profile=build_style_profile_from_turns(
                _style_turns_from_examples(conversation_examples)
            ),
            source_example_count=source_example_count,
            recency_oversampled_count=recency_oversampled_count,
            recency_window_days=self._recency_window_days,
            recency_multiplier=self._recency_multiplier,
        )
        (directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    # 仅兼容外部旧脚本；生产 Linux 训练入口使用 write_lora_dataset。
    write_mlx_dataset = write_lora_dataset


def _write_jsonl(path: Path, examples: list[TrainingExample]) -> None:
    path.write_text(
        "".join(
            json.dumps(_example_payload(example), ensure_ascii=False) + "\n" for example in examples
        ),
        encoding="utf-8",
    )


def _split_temporally(
    examples: list[TrainingExample],
) -> dict[str, list[TrainingExample]]:
    groups: list[list[TrainingExample]] = []
    for example in examples:
        key = _source_group_key(example)
        if groups and _source_group_key(groups[-1][0]) == key:
            groups[-1].append(example)
        else:
            groups.append([example])
    count = len(groups)
    if count < 2:
        return {"train": examples, "valid": [], "test": []}
    if count == 2:
        return {"train": groups[0], "valid": [], "test": groups[1]}
    test_count = max(1, int(count * 0.1))
    valid_count = max(1, int(count * 0.1))
    train_count = count - valid_count - test_count
    return {
        "train": [item for group in groups[:train_count] for item in group],
        "valid": [
            item for group in groups[train_count : train_count + valid_count] for item in group
        ],
        "test": [item for group in groups[train_count + valid_count :] for item in group],
    }


def _time_boundary(examples: list[TrainingExample]) -> dict[str, str | None]:
    if not examples:
        return {"start": None, "end": None}
    return {
        "start": examples[0].target_at.isoformat(),
        "end": examples[-1].target_at.isoformat(),
    }


def _example_payload(example: TrainingExample) -> dict[str, object]:
    return {
        "messages": [asdict(turn) for turn in example.messages],
        "metadata": {
            "source_ids": example.source_ids,
            "target_at": example.target_at.isoformat(),
            "kind": example.kind,
            "event_id": example.event_id,
            "conversation_mode": example.conversation_mode,
            "training_task": example.training_task,
            "allowed_sticker_ids": list(example.allowed_sticker_ids),
            "sticker_first_seen": example.sticker_first_seen,
            "seen_target_sticker_ids": list(example.seen_target_sticker_ids),
            "first_seen_target_sticker_ids": list(example.first_seen_target_sticker_ids),
            "observed_bubbles": [asdict(item) for item in example.observed_bubbles],
        },
    }


def _first_seen_supervision_instruction(
    seen_target_sticker_ids: tuple[str, ...],
    first_seen_target_sticker_ids: tuple[str, ...],
) -> str:
    if not first_seen_target_sticker_ids:
        return ""
    common = (
        "首次出现资产的真实标识符不会加入输入，也不得从未来信息推断；"
        "它只会出现在本条真人监督目标中，运行时不可预测。"
        "该样本不参与资产 Recall 调参。"
    )
    if seen_target_sticker_ids:
        return "目标同时包含历史已见与首次出现的 sticker。" + common
    return "本训练目标仅包含首次出现的 sticker。" + common


def _target_bubble_count(example: TrainingExample) -> int:
    if example.observed_bubbles:
        return len(example.observed_bubbles)
    if not example.messages:
        return 0
    try:
        return len(parse_assistant_protocol(example.messages[-1].content))
    except ValueError:
        return 0


def _source_group_key(example: TrainingExample) -> tuple[str, ...]:
    return tuple(example.source_ids)


def _content_draft(target: str) -> str:
    """把本人表层措辞确定性规范成中性草稿，不改事实与论元。"""

    lines = [line.strip() for line in target.splitlines() if line.strip()]
    draft = "，".join(line.rstrip("，。！？!?；; ") for line in lines if line)
    replacements = (
        ("并不是", "不是"),
        ("并没有", "没有"),
        ("并不会", "不会"),
        ("并不要", "不要"),
        ("并不", "不"),
        ("乃是", "是"),
        ("倘若", "如果"),
        ("是否", "是不是"),
        ("为何", "为什么"),
        ("今日", "今天"),
        ("如何呢", "怎么样"),
        ("非常", "很"),
    )
    for observed, neutral in replacements:
        draft = draft.replace(observed, neutral)
    draft = re.sub(r"^(?:啊呀|哎呀)[，, ]*", "", draft)
    return draft


def _style_turns_from_examples(
    examples: list[TrainingExample],
) -> list[StyleTurn]:
    result: list[StyleTurn] = []
    for example in examples:
        if example.training_task != "conversation":
            continue
        if not example.messages or example.messages[-1].role != "assistant":
            raise ValueError("训练样本缺少可追溯的 assistant 目标")
        if example.observed_bubbles:
            bubbles = example.observed_bubbles
        else:
            try:
                bubbles = parse_assistant_protocol(example.messages[-1].content)
            except ValueError as error:
                raise ValueError("训练样本目标不是合法紧凑气泡协议") from error
        previous_text = next(
            (
                message.content
                for message in reversed(example.messages[:-1])
                if message.role == "user"
            ),
            "",
        )
        result.append(
            StyleTurn(
                previous_text=previous_text,
                bubbles=tuple(
                    StyleBubble(
                        text=bubble.value if bubble.kind == "text" else "",
                        delay_ms=bubble.delay_ms,
                        kind=bubble.kind,
                        asset_id=bubble.value if bubble.kind != "text" else None,
                    )
                    for bubble in bubbles
                ),
            )
        )
    return result


def _event_system_context(event_context: ConfirmedEventContext) -> str:
    lines = [
        "以下是已经由用户确认的真实时间轴背景，仅用于理解当前对话：",
        f"标题：{event_context.title}",
        f"摘要：{event_context.summary}",
        f"通道：{event_context.lane}",
        f"事件状态：{event_context.event_status}",
    ]
    if event_context.before_state and event_context.after_state:
        lines.append(f"关系状态：{event_context.before_state} → {event_context.after_state}")
    return "\n".join(lines)


def _serialize_assistant_turn(
    turn: ConversationTurn,
    *,
    plain_text: bool = False,
) -> str:
    if plain_text:
        return "\n".join(bubble.content for bubble in turn.bubbles if bubble.kind == "text")
    return serialize_bubble_protocol(
        tuple(
            ProtocolBubble(
                kind=bubble.kind,
                value=bubble.content if bubble.kind == "text" else bubble.asset_id or "",
                delay_ms=_bucket_delay_ms(bubble.delay_ms),
            )
            for bubble in turn.bubbles
        )
    )


def parse_assistant_protocol(content: str) -> tuple[ProtocolBubble, ...]:
    try:
        return parse_bubble_protocol(content)
    except ValueError:
        if "<" in content or ">" in content:
            raise
        lines = tuple(line for line in content.splitlines() if line.strip())
        if not lines:
            raise
        return tuple(ProtocolBubble("text", line, 0) for line in lines)


def _serialize_prompt_turn(
    turn: ConversationTurn,
    *,
    plain_text: bool = False,
) -> str:
    # Sticker/emoji context belongs to the independent sticker policy. Even a
    # stable placeholder can be copied verbatim by the language adapter, so
    # language SFT receives only observed text bubbles and drops empty turns.
    bubbles = [
        ProtocolBubble(
            kind="text",
            value=bubble.content,
            delay_ms=_bucket_delay_ms(bubble.delay_ms),
        )
        for bubble in turn.bubbles
        if bubble.kind == "text"
    ]
    if not bubbles:
        return ""
    if plain_text:
        return "\n".join(bubble.value for bubble in bubbles)
    return serialize_bubble_protocol(tuple(bubbles))


def _text_supervision_turn(turn: ConversationTurn) -> ConversationTurn:
    """LoRA 只学习可观察的真人文字；资产选择由独立 sticker policy 负责。"""

    text_bubbles = tuple(bubble for bubble in turn.bubbles if bubble.kind == "text")
    return ConversationTurn(
        speaker=turn.speaker,
        started_at=turn.started_at,
        ended_at=turn.ended_at,
        bubbles=text_bubbles,
        source_ids=tuple(bubble.source_id for bubble in text_bubbles),
    )


_DELAY_BUCKETS_MS = (0, 1_000, 3_000, 6_000, 10_000, 30_000, 60_000)


def _bucket_delay_ms(delay_ms: int) -> int:
    """压缩不可预测的精确时间戳，保留快/中/慢的个人聊天节奏。"""

    bounded = max(0, delay_ms)
    return min(_DELAY_BUCKETS_MS, key=lambda value: (abs(value - bounded), value))


def _asset_id(message: ImportedMessage) -> str | None:
    value = message.raw.get("media_asset_id", message.raw.get("asset_id"))
    return value if isinstance(value, str) and value else None


def _persona_message_text(content: str) -> str:
    """Remove quoted partner payloads while preserving the sender's own text."""

    quote = re.fullmatch(
        r"(?P<text>.*?)\n>\s*[^:：\n]+[:：]\s*.*",
        content.strip(),
        flags=re.DOTALL,
    )
    if quote is not None:
        return quote.group("text").strip()
    # Newer imports may already encode quote display data as JSON.
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return str(payload["text"]).strip()
    return content


def _is_supported_message(message: ImportedMessage) -> bool:
    return (
        not _contains_xml_forbidden_character(message.content)
        and not _contains_private_use_character(message.content)
        and (
            message.kind is MessageKind.TEXT
            or (message.kind is MessageKind.STICKER and _asset_id(message) is not None)
        )
    )


def _target_filter_reason(message: ImportedMessage, cutoff: datetime) -> str | None:
    if message.timestamp > cutoff:
        return "after_cutoff"
    if _contains_xml_forbidden_character(message.content):
        return "xml_forbidden_control"
    if _contains_private_use_character(message.content):
        return "private_use_unicode"
    if message.kind is MessageKind.STICKER and _asset_id(message) is None:
        return "missing_asset_id"
    if message.kind not in {MessageKind.TEXT, MessageKind.STICKER}:
        return f"unsupported_kind:{message.kind.value}"
    return None


def _contains_xml_forbidden_character(value: str) -> bool:
    return any(
        (code < 0x20 and character not in {"\t", "\n", "\r"})
        or 0xD800 <= code <= 0xDFFF
        or code in {0xFFFE, 0xFFFF}
        for character in value
        for code in (ord(character),)
    )


def _contains_private_use_character(value: str) -> bool:
    return any(unicodedata.category(character) == "Co" for character in value)


def _consecutive_target_source_ids(
    messages: list[ImportedMessage],
    target_sender: str,
    cutoff: datetime,
) -> list[str]:
    ordered = sorted(
        (message for message in messages if message.timestamp <= cutoff),
        key=lambda message: message.timestamp,
    )
    source_ids: list[str] = []
    current_group: list[ImportedMessage] = []
    for message in ordered:
        if message.sender == target_sender:
            current_group.append(message)
            continue
        if len(current_group) >= 2:
            source_ids.extend(item.source_id for item in current_group)
        current_group = []
    if len(current_group) >= 2:
        source_ids.extend(item.source_id for item in current_group)
    return source_ids


def _with_event_context(
    example: TrainingExample,
    event_context: ConfirmedEventContext,
) -> list[ChatTurn]:
    messages = list(example.messages)
    for index, message in enumerate(messages):
        if message.role == "system":
            messages[index] = ChatTurn(
                role="system",
                content=f"{message.content}\n{_event_system_context(event_context)}",
            )
            return messages
    return [
        ChatTurn(role="system", content=_event_system_context(event_context)),
        *messages,
    ]
