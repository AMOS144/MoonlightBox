"""Cloud decision cognition with a local persona-expression boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moonlightbox.agent.types import (
    CognitionDraft,
    CognitionRequest,
    ExpressionDecision,
    WakeupSuggestion,
)
from moonlightbox.events.cloud_client import (
    NodeAnalysisCloudClient,
    NodeAnalysisCloudError,
    NodeAnalysisCloudErrorCode,
)


class CloudCognitionFailedError(RuntimeError):
    """The authoritative cloud cognition did not produce a safe decision."""


class _EmotionalDynamicsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)
    intensity: float = Field(ge=0, le=1)
    relationship_threat: float = Field(ge=0, le=1)
    attachment_activation: float = Field(ge=0, le=1)
    dominant_emotions: list[str] = Field(max_length=6)
    impulse: Literal[
        "none",
        "seek_clarity",
        "seek_reassurance",
        "protest",
        "confront",
        "withdraw",
        "repair",
        "accept",
    ]
    persistence: Literal["momentary", "short", "sustained"]


class _CloudCognitionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    private_content: str = Field(min_length=1, max_length=240)
    emotion_summary: str = Field(max_length=120)
    attention_summary: str = Field(max_length=160)
    desired_actions: list[str] = Field(max_length=6)
    express: bool
    content_draft: str | None = Field(default=None, max_length=500)
    decision_reason: str = Field(max_length=160)
    next_wakeup_delay_minutes: int | None = Field(default=None, ge=1, le=10080)
    next_wakeup_reason: str | None = Field(default=None, max_length=160)
    emotional_dynamics: _EmotionalDynamicsPayload
    relationship_appraisal: Literal[
        "stable",
        "uncertain",
        "threatened",
        "ruptured",
        "recovering",
    ]
    event_significance: float = Field(ge=0, le=1)
    follow_up_needed: bool
    follow_up_max_attempts: int = Field(ge=0, le=5)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_empty_optional_fields(cls, value: object) -> object:
        """Accept DeepSeek's harmless empty string for schema-null fields."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        wakeup_reason = normalized.get("next_wakeup_reason")
        if (
            normalized.get("next_wakeup_delay_minutes") is None
            and isinstance(wakeup_reason, str)
            and not wakeup_reason.strip()
        ):
            normalized["next_wakeup_reason"] = None
        content_draft = normalized.get("content_draft")
        if (
            normalized.get("express") is False
            and isinstance(content_draft, str)
            and not content_draft.strip()
        ):
            normalized["content_draft"] = None
        return normalized

    @model_validator(mode="after")
    def validate_expression_partition(self) -> _CloudCognitionPayload:
        if self.express and not (self.content_draft or "").strip():
            raise ValueError("决定表达时必须提供内容草稿")
        if not self.express and self.content_draft is not None:
            raise ValueError("决定沉默时内容草稿必须为空")
        if (self.next_wakeup_delay_minutes is None) != (
            self.next_wakeup_reason is None
        ):
            raise ValueError("唤醒时间和原因必须同时存在或同时为空")
        if self.follow_up_needed:
            if self.next_wakeup_delay_minutes is None or self.follow_up_max_attempts < 1:
                raise ValueError("需要继续联系时必须给出唤醒计划和次数上限")
        elif self.follow_up_max_attempts != 0:
            raise ValueError("不需要继续联系时次数必须为零")
        return self


class DeepSeekCognitionGenerator:
    """Use DeepSeek for meaning/decision; never for final persona wording."""

    def __init__(
        self,
        client: NodeAnalysisCloudClient,
        *,
        adaptive_deliberation: bool = True,
    ) -> None:
        self._client = client
        self._adaptive_deliberation = adaptive_deliberation

    def close(self) -> None:
        self._client.close()

    def generate(self, request: CognitionRequest) -> CognitionDraft:
        system_content = (
            "你是数字人的私有决策认知层，不是客服，也不是最终说话风格模型。"
            "你负责结合完整双边聊天、稳定人格、关系状态、心理状态、目标和有证据记忆，"
            "判断对方在表达什么、数字人真实会怎么想、是否回应，以及回应的语义草稿。"
            "情绪不是礼貌标签：必须根据稳定人格、关系亲密程度、事件的突然性、"
            "目标受阻和关系威胁，给出连续的 valence、arousal、intensity、"
            "relationship_threat、attachment_activation、冲动和持续性。"
            "高意义关系事件不能自动压成温和澄清；同时也不能机械规定某类事件必然愤怒，"
            "反应必须符合这个具体人物，可能是震惊、追问、抗议、愤怒、挽留、退缩或接受。"
            "区分当轮表达和跨轮余波：未解决且仍强烈的事件可设置 follow_up_needed，"
            "安排下一次重新思考，并给出最多 1 到 5 次的上限。"
            "如果 current_mental_state 已有 remaining_attempts，后续不得无新事件地增加次数。"
            "conversation_transcript 按时间排序；speaker=self 是数字人自己已经说过的话，"
            "speaker=user 才是聊天对方，绝不能反转说话人。"
            "遇到‘什么意思’‘没看懂’等省略追问，先承接数字人自己的上一条表达。"
            "content_draft 只写准备表达的意思，简短、第一人称、私人聊天口吻；"
            "不要写分析、规则、客服措辞、标点定义或多个备选答案。"
            "不得把用户的说法直接当成客观事实，不得虚构当前地点、活动或第三方信息。"
            "identity_kernel 是稳定边界，approved_memories 仍须服从 verification_status。"
            "authentic_dialogue_examples 是分支边界之前这个真人实际发过的双边对话，"
            "是判断其立场、关注点、简短程度、称呼、追问方式和多消息结构的最高价值证据；"
            "先据此判断这个具体的人在当前情境会说什么，但不要照抄例子中的具体事实、人物或地点。"
            "这些例子不只是语气样本，也是行为样本：若本人惯于直接提议、追问、调侃或连续表达，"
            "content_draft 也要保留这种行动方式；不得退回任何人都能说的客服式安抚或机械一问一答。"
            "私有判断只能放在 private_content，不能塞进公开 content_draft。"
            "content_draft 本身就要符合此人的反应和关系立场；"
            "最终口癖和更细的多气泡节奏再由本地人格模型处理。"
        )
        transcript = _conversation_transcript(request)
        user_content = json.dumps(
            {
                "trigger_event": dict(request.trigger_event),
                "conversation_transcript": transcript,
                "current_mental_state": dict(request.current_mental_state),
                "goals": [dict(goal) for goal in request.goals],
                "decision_context": dict(request.decision_context),
                "expression_required": request.expression_required,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._complete_with_schema_retry(
                request,
                system_content=system_content,
                user_content=user_content,
                thinking_mode="disabled",
            )
            if self._adaptive_deliberation and _requires_deliberation(
                request,
                payload,
            ):
                payload = self._complete_with_schema_retry(
                    request,
                    system_content=system_content,
                    user_content=user_content,
                    thinking_mode="default",
                )
        except NodeAnalysisCloudError as error:
            raise CloudCognitionFailedError("云端决策认知失败") from error
        if (
            request.expression_required is not None
            and payload.express is not request.expression_required
        ):
            raise CloudCognitionFailedError("云端认知违反真人表达策略")

        follow_up_needed, remaining_attempts = _effective_follow_up(request, payload)
        wakeup = _wakeup_suggestion(
            request,
            payload,
            follow_up_needed=follow_up_needed,
        )
        emotional = payload.emotional_dynamics
        mental_state_changes = {
            "emotional_state": {
                "valence": emotional.valence,
                "arousal": emotional.arousal,
                "intensity": emotional.intensity,
                "relationship_threat": emotional.relationship_threat,
                "attachment_activation": emotional.attachment_activation,
                "dominant_emotions": list(emotional.dominant_emotions),
                "impulse": emotional.impulse,
                "persistence": emotional.persistence,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            "relationship_appraisal": {
                "status": payload.relationship_appraisal,
                "event_significance": payload.event_significance,
            },
            "conversation_drive": {
                "follow_up_needed": follow_up_needed,
                "remaining_attempts": remaining_attempts,
                "reason": (payload.next_wakeup_reason or "").strip(),
            },
        }
        goal_changes = (
            [
                {
                    "goal_type": "relationship_follow_up",
                    "content": (payload.next_wakeup_reason or "继续处理未解决的关系事件").strip(),
                    "priority": max(
                        emotional.intensity,
                        emotional.relationship_threat,
                        emotional.attachment_activation,
                    ),
                    "status": "active",
                }
            ]
            if follow_up_needed
            else _completed_follow_up_goal(request)
        )
        intention_changes = (
            [
                {
                    "intention_type": "follow_up",
                    "content": (payload.next_wakeup_reason or "重新判断是否继续表达").strip(),
                    "status": "active",
                    "expression_plan": {
                        "max_attempts": remaining_attempts,
                        "impulse": emotional.impulse,
                    },
                }
            ]
            if follow_up_needed
            else _fulfilled_follow_up_intention(request)
        )
        authoritative_extraction = {
            "subjective_feelings": {
                "summary": payload.emotion_summary.strip(),
                "intensity": emotional.intensity,
            },
            "attention_target": {"summary": payload.attention_summary.strip()},
            "desired_actions": [
                {"content": action.strip()}
                for action in payload.desired_actions
                if action.strip()
            ],
            "mental_state_changes": mental_state_changes,
            "goal_changes": goal_changes,
            "intention_changes": intention_changes,
            "next_wakeup": (
                {
                    "wake_at": wakeup.wake_at.isoformat(),
                    "reason": wakeup.reason,
                    "idempotency_key": wakeup.idempotency_key,
                }
                if wakeup is not None
                else None
            ),
        }
        return CognitionDraft(
            private_content=payload.private_content.strip(),
            subjective_feelings=(
                {"summary": payload.emotion_summary.strip()}
                if payload.emotion_summary.strip()
                else {}
            ),
            attention_target=(
                {"summary": payload.attention_summary.strip()}
                if payload.attention_summary.strip()
                else {}
            ),
            desired_actions=tuple(
                {"content": action.strip()}
                for action in payload.desired_actions
                if action.strip()
            ),
            expression_decision=ExpressionDecision(
                express=payload.express,
                content=(payload.content_draft or "").strip() or None,
                reason=payload.decision_reason.strip() or None,
            ),
            suggested_next_wakeup=wakeup,
            structured_changes={
                "extraction_status": "authoritative_ready",
                "authoritative_extraction": authoritative_extraction,
            },
            confidence=payload.confidence,
        )

    def _complete_with_schema_retry(
        self,
        request: CognitionRequest,
        *,
        system_content: str,
        user_content: str,
        thinking_mode: Literal["default", "disabled"],
    ) -> _CloudCognitionPayload:
        """Retry one malformed structured response without hiding service errors."""

        options: dict[str, Any] = {
            "system_content": system_content,
            "user_content": user_content,
            "response_model": _CloudCognitionPayload,
            "operation_id": f"cognition:{request.branch_id}",
            "run_id": str(request.trigger_event.get("id", "")) or None,
            "max_prompt_chars": 120_000,
            "thinking_mode_override": thinking_mode,
        }
        try:
            return self._client.create_structured_completion(**options)
        except NodeAnalysisCloudError as error:
            if error.code is not NodeAnalysisCloudErrorCode.INVALID_RESPONSE:
                raise
        return self._client.create_structured_completion(**options)


def _requires_deliberation(
    request: CognitionRequest,
    payload: _CloudCognitionPayload,
) -> bool:
    # A human reacts to a newly delivered message before doing a long private
    # rumination.  Keep the visible turn fast; elapsed-time wakeups can perform
    # the deeper second pass and drive later follow-up behavior.
    if request.trigger_event.get("event_type") == "user_message":
        return False
    emotional = payload.emotional_dynamics
    return (
        payload.event_significance >= 0.75
        or emotional.relationship_threat >= 0.65
        or emotional.intensity >= 0.75
    )


def _conversation_transcript(request: CognitionRequest) -> list[dict[str, object]]:
    transcript: list[dict[str, object]] = []
    for item in request.relevant_context:
        event_type = item.get("event_type")
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            continue
        content = evidence.get("content")
        if not isinstance(content, str):
            continue
        speaker: Literal["self", "user"]
        if event_type == "agent_expression":
            speaker = "self"
        elif event_type == "user_message":
            speaker = "user"
        else:
            continue
        transcript.append(
            {
                "speaker": speaker,
                "content": content,
                "message_type": evidence.get("message_type", "text"),
            }
        )
    return transcript


def _wakeup_suggestion(
    request: CognitionRequest,
    payload: _CloudCognitionPayload,
    *,
    follow_up_needed: bool,
) -> WakeupSuggestion | None:
    if (
        not follow_up_needed
        or payload.next_wakeup_delay_minutes is None
        or payload.next_wakeup_reason is None
    ):
        return None
    trigger_id = str(request.trigger_event.get("id", "unknown"))
    digest = hashlib.sha256(
        f"{request.branch_id}:{trigger_id}:{payload.next_wakeup_delay_minutes}".encode()
    ).hexdigest()[:20]
    return WakeupSuggestion(
        wake_at=datetime.now(UTC)
        + timedelta(minutes=payload.next_wakeup_delay_minutes),
        reason=payload.next_wakeup_reason.strip(),
        idempotency_key=f"cloud-cognition:{digest}",
    )


def _effective_follow_up(
    request: CognitionRequest,
    payload: _CloudCognitionPayload,
) -> tuple[bool, int]:
    """Enforce a decreasing retry budget for wakeup-triggered follow-ups."""

    if not payload.follow_up_needed:
        return False, 0
    remaining = payload.follow_up_max_attempts
    if request.trigger_event.get("event_type") == "elapsed_time":
        drive = request.current_mental_state.get("conversation_drive")
        previous = drive.get("remaining_attempts") if isinstance(drive, dict) else None
        if isinstance(previous, int) and not isinstance(previous, bool):
            remaining = min(remaining, max(0, previous - 1))
    return remaining > 0, remaining


def _had_pending_follow_up(request: CognitionRequest) -> bool:
    drive = request.current_mental_state.get("conversation_drive")
    return isinstance(drive, dict) and drive.get("follow_up_needed") is True


def _completed_follow_up_goal(
    request: CognitionRequest,
) -> list[dict[str, object]]:
    if not _had_pending_follow_up(request):
        return []
    return [
        {
            "goal_type": "relationship_follow_up",
            "content": "未解决的关系事件已经重新评估",
            "priority": 0.0,
            "status": "completed",
        }
    ]


def _fulfilled_follow_up_intention(
    request: CognitionRequest,
) -> list[dict[str, object]]:
    if not _had_pending_follow_up(request):
        return []
    return [
        {
            "intention_type": "follow_up",
            "content": "本轮后不再主动追问",
            "status": "fulfilled",
            "expression_plan": {},
        }
    ]
