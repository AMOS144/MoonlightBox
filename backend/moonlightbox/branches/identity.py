import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import IdentityKernel
from moonlightbox.training.bubble_protocol import parse_bubble_protocol
from moonlightbox.training.dataset_builder import (
    ConfirmedEventContext,
    TrainingExample,
)
from moonlightbox.training.style_features import StyleBubble, StyleTurn
from moonlightbox.training.style_profile import build_style_profile_from_turns


class IdentityKernelLockedError(RuntimeError):
    pass


class IdentityKernelProposal(BaseModel):
    persona: str = Field(min_length=1)
    values: list[str]
    stable_preferences: list[str]
    expressed_preferences: list[str] = Field(default_factory=list)
    relationship_boundaries: list[str]
    language_patterns: list[str]
    typical_reactions: list[str]
    style_profile: dict[str, object] = Field(default_factory=dict)
    behavioral_rhythm: dict[str, object] = Field(default_factory=dict)
    field_evidence: dict[str, list[str]] = Field(default_factory=dict, exclude=True)
    field_confidence: dict[str, float] = Field(default_factory=dict, exclude=True)


class EvidenceBackedIdentityKernelBuilder:
    """根据训练样本生成稳定、可追溯的人格内核初稿。"""

    def __init__(self, *, default_timezone_offset_minutes: int = 480) -> None:
        if not -720 <= default_timezone_offset_minutes <= 840:
            raise ValueError("默认时区偏移必须在 UTC-12 到 UTC+14 之间")
        self._default_timezone_offset_minutes = default_timezone_offset_minutes

    def build(
        self,
        *,
        persona: str,
        examples: list[TrainingExample],
        event_contexts: list[ConfirmedEventContext],
    ) -> IdentityKernelProposal:
        evidence_examples = [
            example
            for example in examples
            if not any(
                source_id.startswith("policy:")
                for source_id in example.source_ids
            )
        ]
        style_turns = [
            _assistant_style_turn(example)
            for example in evidence_examples
            if example.messages and example.messages[-1].role == "assistant"
        ]
        assistant_targets = [
            "\n".join(
                bubble.text
                for bubble in turn.bubbles
                if bubble.kind == "text" and bubble.text
            )
            for turn in style_turns
        ]
        average_length = (
            sum(len(item) for item in assistant_targets) / len(assistant_targets)
            if assistant_targets
            else 0
        )
        source_ids = list(
            dict.fromkeys(
                source_id for example in examples for source_id in example.source_ids
                if not source_id.startswith("policy:")
            )
        )
        refusal_count = sum(
            any(marker in text for marker in ("不要", "不想", "不行", "别"))
            for text in assistant_targets
        )
        question_count = sum(
            text.rstrip().endswith(("吗", "呢", "？", "?"))
            for text in assistant_targets
        )
        style_profile = build_style_profile_from_turns(style_turns)
        behavioral_rhythm = _build_behavioral_rhythm(
            evidence_examples,
            default_timezone_offset_minutes=self._default_timezone_offset_minutes,
        )
        expressed_preferences, preference_evidence = _extract_expressed_preferences(
            evidence_examples
        )
        raw_forbidden_markers = style_profile["forbidden_unobserved_markers"]
        forbidden_markers = (
            raw_forbidden_markers
            if isinstance(raw_forbidden_markers, list)
            else []
        )
        language_patterns = [
            f"真实回复平均约 {average_length:.0f} 个字符",
            (
                "通常使用短句和紧凑表达"
                if average_length < 40
                else "通常使用较完整的连续表达"
            ),
            (
                "会通过自然追问继续交流"
                if question_count
                else "较少使用连续追问"
            ),
            (
                "未观察到的语气标记不得擅自使用："
                + "、".join(str(item) for item in forbidden_markers)
            ),
            "低频语气标记不得被模板化重复使用",
        ]
        raw_forbidden_ai_register = style_profile.get(
            "forbidden_unobserved_ai_register",
            [],
        )
        if isinstance(raw_forbidden_ai_register, list) and raw_forbidden_ai_register:
            language_patterns.append(
                "不要使用本人语料中未出现的客服式套话："
                + "、".join(str(item) for item in raw_forbidden_ai_register)
            )
        values = ["依据真实共同经历形成关系判断"]
        if refusal_count:
            values.append("在关系中保留自己的判断和边界")
        if event_contexts:
            values.append("重视双方共同经历")
        confidence = min(0.9, 0.55 + len(assistant_targets) / 100)
        field_names = (
            "values",
            "stable_preferences",
            "expressed_preferences",
            "relationship_boundaries",
            "language_patterns",
            "typical_reactions",
            "style_profile",
            "behavioral_rhythm",
        )
        return IdentityKernelProposal(
            persona=persona,
            values=values,
            stable_preferences=["不确定时先结合上下文理解，而非编造事实"],
            expressed_preferences=expressed_preferences,
            relationship_boundaries=[
                "用户对数字人的定义只是用户观点，不能直接覆盖数字人的感受",
                "人格变化需要数字人的自主表达或持续互动证据",
            ],
            language_patterns=language_patterns,
            typical_reactions=[
                (
                    "面对不愿接受的要求会直接拒绝"
                    if refusal_count
                    else "面对冲突时保留自己的判断"
                ),
                "证据不足时自然澄清",
            ],
            style_profile=style_profile,
            behavioral_rhythm=behavioral_rhythm,
            field_evidence={
                name: (
                    preference_evidence
                    if name == "expressed_preferences"
                    else source_ids
                )
                for name in field_names
            },
            field_confidence={
                name: (
                    min(0.95, 0.6 + len(expressed_preferences) * 0.03)
                    if name == "expressed_preferences"
                    else confidence
                )
                for name in field_names
            },
        )


class IdentityKernelService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_and_lock(
        self,
        *,
        project_id: str,
        model_version_id: str,
        proposal: IdentityKernelProposal,
        evidence_message_ids: list[str],
        acceptance_report_id: str | None = None,
        commit: bool = True,
    ) -> IdentityKernel:
        existing = self._session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
        )
        if existing is not None:
            return existing
        evidence = list(dict.fromkeys(evidence_message_ids))
        if not evidence:
            raise ValueError("人格内核必须引用训练消息证据")
        content = proposal.model_dump()
        field_names = [
            "values",
            "stable_preferences",
            "expressed_preferences",
            "relationship_boundaries",
            "language_patterns",
            "typical_reactions",
            "style_profile",
            "behavioral_rhythm",
        ]
        field_evidence = {
            name: list(dict.fromkeys(proposal.field_evidence.get(name, evidence)))
            for name in field_names
        }
        field_confidence = {
            name: max(0.0, min(1.0, proposal.field_confidence.get(name, 0.6)))
            for name in field_names
        }
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "project_id": project_id,
                    "model_version_id": model_version_id,
                    "content": content,
                    "evidence_message_ids": evidence,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        kernel = IdentityKernel(
            project_id=project_id,
            model_version_id=model_version_id,
            schema_version=(
                "subject-persona-v2"
                if acceptance_report_id is not None
                else "subject-centric-v1"
            ),
            content=content,
            evidence_message_ids=evidence,
            field_evidence=field_evidence,
            field_confidence=field_confidence,
            acceptance_report_id=acceptance_report_id,
            content_hash=content_hash,
            locked_at=datetime.now(UTC),
        )
        self._session.add(kernel)
        if commit:
            self._session.commit()
            self._session.refresh(kernel)
        else:
            self._session.flush()
        return kernel

    def get_for_model(self, model_version_id: str) -> IdentityKernel:
        kernel = self._session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
        )
        if kernel is None:
            raise LookupError("模型人格内核不存在")
        return kernel

    def replace(
        self,
        kernel_id: str,
        _proposal: IdentityKernelProposal,
    ) -> None:
        kernel = self._session.get(IdentityKernel, kernel_id)
        if kernel is None:
            raise LookupError("人格内核不存在")
        raise IdentityKernelLockedError("人格内核锁定后不可修改")


def _build_behavioral_rhythm(
    examples: list[TrainingExample],
    *,
    default_timezone_offset_minutes: int,
) -> dict[str, object]:
    eligible = [
        example
        for example in examples
        if example.kind == "chat"
        and example.training_task == "conversation"
        and not any(source_id.startswith("policy:") for source_id in example.source_ids)
    ]
    if not eligible:
        return {
            "schema_version": "behavioral-rhythm-v1",
            "sample_count": 0,
            "confidence": 0.0,
            "active_hours": [],
        }
    hour_counts = Counter(example.target_at.hour for example in eligible)
    weekday_counts = Counter(example.target_at.weekday() for example in eligible)
    offsets = [
        int(offset.total_seconds() // 60)
        for example in eligible
        if (offset := example.target_at.utcoffset()) is not None
    ]
    sample_count = len(eligible)
    ranked_hours = sorted(hour_counts, key=lambda hour: (-hour_counts[hour], hour))
    active_hours: list[int] = []
    covered = 0
    if sample_count >= 10:
        for hour in ranked_hours:
            active_hours.append(hour)
            covered += hour_counts[hour]
            if covered / sample_count >= 0.7 or len(active_hours) == 8:
                break
    proactive_count = sum(
        example.conversation_mode == "proactive" for example in eligible
    )
    inter_bubble_delays = sorted(
        bubble.delay_ms
        for example in eligible
        for bubble in _assistant_style_turn(example).bubbles[1:]
        if bubble.delay_ms > 0
    )
    delay_count = len(inter_bubble_delays)
    delay_p50 = (
        inter_bubble_delays[max(0, (delay_count - 1) // 2)]
        if inter_bubble_delays
        else 0
    )
    delay_p90 = (
        inter_bubble_delays[
            min(delay_count - 1, max(0, round(delay_count * 0.9) - 1))
        ]
        if inter_bubble_delays
        else 0
    )
    return {
        "schema_version": "behavioral-rhythm-v1",
        "sample_count": sample_count,
        "confidence": round(min(1.0, sample_count / 100), 4),
        "timezone_offset_minutes": (
            Counter(offsets).most_common(1)[0][0]
            if offsets
            else default_timezone_offset_minutes
        ),
        "timezone_derivation": (
            "authentic_timestamp_offset"
            if offsets
            else "configured_default_for_naive_source_timestamps"
        ),
        "hour_counts": {str(hour): hour_counts.get(hour, 0) for hour in range(24)},
        "weekday_counts": {
            str(weekday): weekday_counts.get(weekday, 0) for weekday in range(7)
        },
        "active_hours": sorted(active_hours),
        "proactive_turn_count": proactive_count,
        "proactive_turn_rate": round(
            (proactive_count + 1) / (sample_count + 2),
            6,
        ),
        "inter_bubble_delay_ms": {
            "count": delay_count,
            "mean": (
                round(sum(inter_bubble_delays) / delay_count, 2)
                if delay_count
                else 0.0
            ),
            "p50": delay_p50,
            "p90": delay_p90,
        },
        "derivation": "authentic_target_chat_examples_only",
    }


_PREFERENCE_PATTERN = re.compile(
    r"(?:我)?(?:最|一直|挺|很|比较|不太)?"
    r"(?:喜欢|爱吃|爱喝|爱看|讨厌|不喜欢|习惯)"
    r"[^，。！？!?\n]{1,16}"
)


def _extract_expressed_preferences(
    examples: list[TrainingExample],
) -> tuple[list[str], list[str]]:
    preferences: list[str] = []
    evidence: list[str] = []
    for example in examples:
        if (
            example.kind != "chat"
            or example.training_task != "conversation"
            or not example.messages
            or example.messages[-1].role != "assistant"
        ):
            continue
        text = _assistant_text(example.messages[-1].content)
        for match in _PREFERENCE_PATTERN.finditer(text):
            preference = match.group(0).strip(" ，。！？!?、；;：:")
            if not preference or preference.endswith(("吗", "呢")):
                continue
            if preference not in preferences:
                preferences.append(preference)
            evidence.extend(example.source_ids)
            if len(preferences) == 12:
                return preferences, list(dict.fromkeys(evidence))
    return preferences, list(dict.fromkeys(evidence))


def _assistant_text(content: str) -> str:
    if any(tag in content for tag in ("<bubble", "<sticker", "<delay")):
        try:
            protocol_bubbles = parse_bubble_protocol(content)
        except ValueError:
            return ""
        return "\n".join(
            bubble.value
            for bubble in protocol_bubbles
            if bubble.kind == "text" and bubble.value
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    bubbles = parsed.get("bubbles") if isinstance(parsed, dict) else None
    if not isinstance(bubbles, list):
        return content
    return "\n".join(
        str(item.get("content", ""))
        for item in bubbles
        if isinstance(item, dict) and item.get("content")
    )


def _assistant_style_turn(example: TrainingExample) -> StyleTurn:
    content = example.messages[-1].content
    previous_text = next(
        (
            message.content
            for message in reversed(example.messages[:-1])
            if message.role == "user"
        ),
        "",
    )
    if example.observed_bubbles:
        return StyleTurn(
            previous_text=previous_text,
            bubbles=tuple(
                StyleBubble(
                    text=bubble.value if bubble.kind == "text" else "",
                    delay_ms=bubble.delay_ms,
                    kind=bubble.kind,
                    asset_id=bubble.value if bubble.kind != "text" else None,
                )
                for bubble in example.observed_bubbles
            ),
        )
    if any(tag in content for tag in ("<bubble", "<sticker", "<delay")):
        try:
            protocol_bubbles = parse_bubble_protocol(content)
        except ValueError:
            protocol_bubbles = ()
        return StyleTurn(
            previous_text=previous_text,
            bubbles=tuple(
                StyleBubble(
                    text=bubble.value if bubble.kind == "text" else "",
                    delay_ms=bubble.delay_ms,
                    kind=bubble.kind,
                    asset_id=bubble.value if bubble.kind != "text" else None,
                )
                for bubble in protocol_bubbles
            ),
        )
    text = _assistant_text(content)
    return StyleTurn(
        previous_text=previous_text,
        bubbles=(StyleBubble(text=text),) if text else (),
    )
