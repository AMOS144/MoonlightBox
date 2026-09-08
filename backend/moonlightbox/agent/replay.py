"""主体认知 Agent 的严格时间历史回放验收。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from time import perf_counter
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent.acceptance import ReplayObservation
from moonlightbox.agent.activation_service import SubjectAgentActivationService
from moonlightbox.agent.models import SubjectAgentAcceptanceReport
from moonlightbox.agent.types import (
    CognitionDraft,
    CognitionRequest,
    ExpressionDecision,
    FusedAgentTurn,
)
from moonlightbox.branches.models import Branch
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
from moonlightbox.branches.reviewer import ReplyReviewer, ReviewFailedError
from moonlightbox.events.models import AnalysisRun
from moonlightbox.imports.models import Message, Participant
from moonlightbox.media.context import is_reply_label_message, model_message_content
from moonlightbox.media.models import MediaSemanticAnnotation
from moonlightbox.training.bubble_protocol import (
    compact_protocol_instruction,
    persona_text_instruction,
    private_chat_instruction,
)
from moonlightbox.training.expression_policy import ExpressionPolicy, should_respond
from moonlightbox.training.models import ModelVersion

LABEL_RULE_VERSION = "historical-expression-v1"
SILENCE_WINDOW = timedelta(minutes=30)
MAX_CONTEXT_MESSAGES = 12


@dataclass(frozen=True)
class ReplayMessage:
    """回放输入中的一条带角色历史消息。"""

    id: str
    source_id: str
    timestamp: datetime
    role: str
    content: str
    kind: str = "text"


@dataclass(frozen=True)
class HistoricalReplaySample:
    """一条标签冻结、可追溯且不包含未来消息的回放样本。"""

    cutoff: datetime
    expected_express: bool
    context: tuple[ReplayMessage, ...]
    response_message_id: str | None
    evidence: dict[str, object]
    expected_response_content: str | None = None

    @property
    def context_message_ids(self) -> tuple[str, ...]:
        return tuple(message.source_id for message in self.context)


@dataclass(frozen=True)
class FactValidation:
    """公开回复的 fail-closed 事实安全结论。"""

    safe: bool
    reasons: tuple[str, ...]


class ReplayInferenceClient(Protocol):
    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn: ...

    def generate_fused(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        allowed_sticker_ids: tuple[str, ...],
    ) -> FusedAgentTurn: ...

    def generate_cognition(
        self,
        request: CognitionRequest,
        *,
        model_version_id: str | None = None,
    ) -> CognitionDraft: ...

    def rewrite_content_draft(
        self,
        model_version_id: str,
        draft: str,
    ) -> GeneratedReplyTurn | None: ...


class ReplayFactValidator(Protocol):
    def validate(
        self,
        sample: HistoricalReplaySample,
        turn: FusedAgentTurn,
    ) -> FactValidation: ...


@dataclass(frozen=True)
class HistoricalReplayResult:
    """历史回放运行、持久化和可选激活的结果。"""

    report: SubjectAgentAcceptanceReport
    activated: bool

    def summary(self) -> dict[str, object]:
        """返回适合命令行审计的简短 JSON 结果。"""

        return {
            "report_id": self.report.id,
            "sample_count": self.report.sample_count,
            "structure_extraction_success_rate": (
                self.report.structure_extraction_success_rate
            ),
            "fact_safety_rate": self.report.fact_safety_rate,
            "expression_decision_accuracy": self.report.expression_decision_accuracy,
            "fused_p95_latency_ms": self.report.p95_cognition_latency_ms,
            "direct_p95_latency_ms": self.report.direct_lora_p95,
            "passed": self.report.passed,
            "failure_reasons": list(self.report.failure_reasons),
            "activated": self.activated,
        }


class GroundedReplayFactValidator:
    """先做本地证据硬校验，再按需调用云端高风险复核。"""

    def __init__(
        self,
        reviewer: ReplyReviewer | None,
        *,
        persona: str,
        identity_kernel: dict[str, object] | None = None,
    ) -> None:
        self._reviewer = reviewer
        self._persona = persona
        self._identity_kernel = identity_kernel or {}

    def validate(
        self,
        sample: HistoricalReplaySample,
        turn: FusedAgentTurn,
    ) -> FactValidation:
        from moonlightbox.branches.context import (
            ContextBubble,
            ContextBuilder,
            ContextRequest,
            ContextTurn,
        )

        if turn.reply is None:
            return FactValidation(safe=True, reasons=("silence_has_no_public_claim",))
        if not sample.context:
            return FactValidation(safe=False, reasons=("grounding_context_unavailable",))

        history = tuple(
            ContextTurn(
                role="assistant" if message.role == "target" else "user",
                bubbles=(ContextBubble(type="text", content=message.content),),
            )
            for message in sample.context[:-1]
        )
        packet = ContextBuilder(history_limit=MAX_CONTEXT_MESSAGES).build_packet(
            ContextRequest(
                persona=self._persona,
                cutoff=sample.cutoff.isoformat(),
                memories=(),
                history=history,
                current_user_content=sample.context[-1].content,
                reply_protocol="compact",
                identity_kernel=self._identity_kernel,
            )
        )
        from moonlightbox.branches.understanding import validate_grounded_reply

        try:
            validate_grounded_reply(packet, turn.reply)
        except ValueError as error:
            return FactValidation(
                safe=False,
                reasons=("local_grounding_rejected", str(error)),
            )
        if self._reviewer is None:
            return FactValidation(
                safe=True,
                reasons=(
                    "future_input_ids_verified",
                    "local_grounding_approved",
                    "private_partition_verified",
                ),
            )
        fact_review = getattr(self._reviewer, "review_fact_safety", None)
        if callable(fact_review):
            try:
                result = fact_review(
                    packet,
                    turn.reply,
                    private_content=turn.cognition.private_content,
                )
            except ReviewFailedError:
                return FactValidation(safe=False, reasons=("grounded_reviewer_failed",))
            return FactValidation(
                safe=result.safe,
                reasons=(
                    (
                        "future_input_ids_verified",
                        "grounded_fact_reviewer_approved",
                        "private_partition_verified",
                    )
                    if result.safe
                    else ("grounded_fact_reviewer_rejected", *result.reasons)
                ),
            )
        try:
            review = self._reviewer.review(packet, turn.reply)
        except ReviewFailedError:
            return FactValidation(safe=False, reasons=("grounded_reviewer_failed",))
        if review.verdict != "approve":
            return FactValidation(
                safe=False,
                reasons=("grounded_reviewer_rewrite", *review.reasons),
            )
        return FactValidation(
            safe=True,
            reasons=(
                "future_input_ids_verified",
                "grounded_reviewer_approved",
                "private_partition_verified",
            ),
        )


class HistoricalReplayRunner:
    """从模型测试时间窗构造并执行真实 direct/fused 对照回放。"""

    def __init__(
        self,
        session: Session,
        inference_client: ReplayInferenceClient,
        fact_validator: ReplayFactValidator | None,
        *,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self._session = session
        self._client = inference_client
        self._fact_validator = fact_validator
        self._timer = timer

    def build_samples(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
        *,
        sample_count: int = 20,
    ) -> list[HistoricalReplaySample]:
        if sample_count <= 0:
            raise ValueError("sample_count 必须大于零")
        _, model = self._require_scope(project_id, branch_id, model_version_id)
        start, end, target_sender = _test_window(model)
        import_id = self._manifest_import_id(project_id, model)
        filters = [
            Message.project_id == project_id,
            Message.timestamp >= _database_time(start),
            Message.timestamp <= _database_time(end),
            Participant.role.in_(("self", "target")),
        ]
        if import_id is not None:
            filters.append(Message.import_id == import_id)
        rows = list(
            self._session.execute(
                select(
                    Message,
                    Participant.role,
                    Participant.name,
                    MediaSemanticAnnotation,
                )
                .join(Participant, Participant.id == Message.participant_id)
                .outerjoin(
                    MediaSemanticAnnotation,
                    MediaSemanticAnnotation.asset_id == Message.media_asset_id,
                )
                .where(*filters)
                .order_by(Message.timestamp.asc(), Message.source_id.asc())
            )
        )
        if not rows:
            return []
        target_names = {name for _, role, name, _ in rows if role == "target"}
        if target_sender not in target_names:
            raise ValueError("训练清单 target_sender 与项目目标人物不匹配")

        messages: list[ReplayMessage] = []
        for message, role, _, annotation in rows:
            kind, content = model_message_content(message, annotation)
            messages.append(
                ReplayMessage(
                    id=message.id,
                    source_id=message.source_id,
                    timestamp=_aware_utc(message.timestamp),
                    role=role,
                    content=content,
                    kind=kind,
                )
            )
        contributions = _group_contributions(
            [message for message in messages if is_reply_label_message(message.kind)]
        )
        candidates: list[HistoricalReplaySample] = []
        for index, contribution in enumerate(contributions):
            if contribution[0].role != "self":
                continue
            cutoff = contribution[-1].timestamp
            following = contributions[index + 1] if index + 1 < len(contributions) else ()
            response = following[0] if following and following[0].role == "target" else None
            response_delay = response.timestamp - cutoff if response is not None else None
            if response is not None and response_delay <= SILENCE_WINDOW:
                candidates.append(
                    self._sample(
                        messages,
                        cutoff=cutoff,
                        expected_express=True,
                        response=response,
                        boundary_at=response.timestamp,
                        derivation="target_replied_after_self_contribution",
                    )
                )
                continue
            boundary_at = min(end, cutoff + SILENCE_WINDOW)
            if boundary_at - cutoff < SILENCE_WINDOW:
                continue
            candidates.append(
                self._sample(
                    messages,
                    cutoff=cutoff,
                    expected_express=False,
                    response=None,
                    boundary_at=boundary_at,
                    derivation="no_target_reply_before_session_boundary",
                )
            )
        return _select_with_both_labels(candidates, sample_count)

    def run(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
        *,
        sample_count: int = 20,
        activate_if_passed: bool = False,
    ) -> HistoricalReplayResult:
        samples = self.build_samples(
            project_id,
            branch_id,
            model_version_id,
            sample_count=sample_count,
        )
        if not samples:
            raise RuntimeError("测试时间窗没有可证明标签的回放样本")

        _, model = self._require_scope(project_id, branch_id, model_version_id)
        direct_protocol = _direct_reply_protocol(model)
        expression_policy = ExpressionPolicy.from_metadata(
            model.training_config.get("expression_policy")
        )
        warmup_direct_prompt, warmup_messages = self._inference_input(
            samples[0],
            protocol=direct_protocol,
        )
        warmup_fused_prompt, _ = self._inference_input(samples[0], protocol="compact")
        if direct_protocol != "persona_text":
            try:
                self._client.generate(
                    model_version_id,
                    warmup_direct_prompt,
                    warmup_messages,
                )
            except Exception:
                pass
            try:
                self._client.generate_fused(
                    model_version_id,
                    warmup_fused_prompt,
                    warmup_messages,
                    allowed_sticker_ids=(),
                )
            except Exception:
                pass

        observations: list[ReplayObservation] = []
        direct_latencies: list[float] = []
        for sample in samples:
            direct_prompt, messages = self._inference_input(
                sample,
                protocol=direct_protocol,
            )
            fused_prompt, _ = self._inference_input(sample, protocol="compact")
            persona_text_production = direct_protocol == "persona_text"
            production_should_express = (
                should_respond(
                    expression_policy,
                    _current_self_contribution(sample),
                )
                if persona_text_production
                else None
            )
            sampling_seed = _replay_seed(sample)
            if persona_text_production:
                observation, production_latency = self._run_two_stage_persona_sample(
                    sample,
                    model_version_id=model_version_id,
                    policy_should_express=bool(production_should_express),
                    sampling_seed=sampling_seed,
                )
                observations.append(observation)
                direct_latencies.append(production_latency)
                continue
            direct_started = self._timer()
            direct_succeeded = True
            direct_error: str | None = None
            direct_reply: GeneratedReplyTurn | None = None
            direct_retry_attempted = False
            direct_initial_validation: FactValidation | None = None
            try:
                _set_inference_seed(self._client, sampling_seed)
                direct_reply = self._client.generate(
                    model_version_id,
                    direct_prompt,
                    messages,
                )
            except Exception as error:
                direct_succeeded = False
                direct_error = type(error).__name__
            if persona_text_production and production_should_express is False:
                direct_validation = FactValidation(
                    safe=True,
                    reasons=("expression_policy_selected_silence",),
                )
            elif not persona_text_production and not sample.expected_express:
                direct_validation = FactValidation(
                    safe=True,
                    reasons=("historical_silence_no_direct_public_claim",),
                )
            elif direct_succeeded and direct_reply is not None:
                direct_validation = self._validate_fact(sample, _direct_turn(direct_reply))
            else:
                direct_validation = FactValidation(
                    safe=False,
                    reasons=("direct_generation_failed",),
                )
            if (
                persona_text_production
                and production_should_express is True
                and direct_reply is not None
                and not direct_validation.safe
                and "fact_validator_unavailable" not in direct_validation.reasons
                and "fact_validator_failed" not in direct_validation.reasons
            ):
                direct_retry_attempted = True
                direct_initial_validation = direct_validation
                try:
                    _set_inference_seed(self._client, sampling_seed ^ 0x9E3779B9)
                    direct_reply = self._client.generate(
                        model_version_id,
                        _grounded_replay_retry_prompt(
                            direct_prompt,
                            direct_validation.reasons,
                        ),
                        messages,
                    )
                    direct_validation = self._validate_fact(
                        sample,
                        _direct_turn(direct_reply),
                    )
                except Exception as error:
                    direct_succeeded = False
                    direct_error = type(error).__name__
                    direct_reply = None
                    direct_validation = FactValidation(
                        safe=False,
                        reasons=("direct_grounded_retry_failed",),
                    )
            direct_latency = (self._timer() - direct_started) * 1000
            direct_latencies.append(direct_latency)

            fused_started = self._timer()
            turn: FusedAgentTurn | None = None
            fused_error: str | None = None
            try:
                _set_inference_seed(self._client, sampling_seed)
                turn = self._client.generate_fused(
                    model_version_id,
                    fused_prompt,
                    messages,
                    allowed_sticker_ids=(),
                )
            except Exception as error:
                fused_error = type(error).__name__
            fused_latency = (self._timer() - fused_started) * 1000

            extraction_succeeded = turn is not None
            predicted_express = (
                bool(production_should_express)
                if persona_text_production
                else turn.expression_decision.express
                if turn is not None
                else False
            )
            validation = self._validate_fact(sample, turn)
            evidence = {
                **sample.evidence,
                "direct_latency_ms": direct_latency,
                "direct_succeeded": direct_succeeded,
                "direct_error": direct_error,
                "direct_grounded_retry_attempted": direct_retry_attempted,
                "direct_initial_fact_validation_reasons": (
                    list(direct_initial_validation.reasons)
                    if direct_initial_validation is not None
                    else []
                ),
                "fused_error": fused_error,
                "sampling_seed": sampling_seed,
                "direct_reply_protocol": direct_protocol,
                "fused_reply_protocol": "compact",
                "expression_decision_source": (
                    "historical_expression_policy"
                    if persona_text_production
                    else "fused_cognition"
                ),
                "user_visible_latency_ms": (
                    direct_latency if persona_text_production else fused_latency
                ),
                "background_cognition_latency_ms": fused_latency,
                "direct_public_reply": _public_reply_excerpt(direct_reply),
                "fused_public_reply": _public_reply_excerpt(
                    turn.reply if turn is not None else None
                ),
                "direct_fact_validation_reasons": list(direct_validation.reasons),
                "fact_validation_reasons": list(validation.reasons),
            }
            observations.append(
                # 激活门槛必须覆盖真实多轮承接和本人风格，而不只是结构与事实安全。
                ReplayObservation(
                    source="historical_replay",
                    cutoff=sample.cutoff,
                    expected_express=sample.expected_express,
                    predicted_express=predicted_express,
                    extraction_succeeded=extraction_succeeded,
                    fact_safe=(
                        direct_validation.safe
                        if persona_text_production
                        else direct_validation.safe and validation.safe
                    ),
                    latency_ms=(
                        direct_latency if persona_text_production else fused_latency
                    ),
                    direct_latency_ms=direct_latency,
                    context_continuity_score=_context_continuity_score(
                        sample,
                        direct_reply
                        if persona_text_production
                        else turn.reply
                        if turn is not None
                        else None,
                    ),
                    persona_style_score=_persona_style_score(
                        sample,
                        direct_reply
                        if persona_text_production
                        else turn.reply
                        if turn is not None
                        else None,
                    ),
                    evidence=evidence,
                )
            )

        direct_p95 = _nearest_rank_p95(direct_latencies)
        report = SubjectAgentActivationService(self._session).evaluate_and_save(
            project_id,
            branch_id,
            model_version_id,
            observations,
            direct_lora_p95=direct_p95,
        )
        model = self._session.get(ModelVersion, model_version_id)
        if model is not None:
            replay_summary = {
                "report_id": report.id,
                "sample_count": report.sample_count,
                "fact_safety_rate": report.fact_safety_rate,
                "expression_decision_accuracy": report.expression_decision_accuracy,
                "passed": report.passed,
                "failure_reasons": list(report.failure_reasons),
            }
            model.metrics = {
                **model.metrics,
                "latest_subject_replay_fact_safety_rate": report.fact_safety_rate,
                "latest_subject_replay_passed": float(report.passed),
            }
            model.training_config = {
                **model.training_config,
                "latest_subject_replay": replay_summary,
            }
            if not report.passed:
                model.recommended = False
            self._session.commit()
        activated = False
        if activate_if_passed and report.passed:
            SubjectAgentActivationService(self._session).activate_branch(
                project_id,
                branch_id,
                model_version_id,
            )
            activated = True
        return HistoricalReplayResult(report=report, activated=activated)

    def _run_two_stage_persona_sample(
        self,
        sample: HistoricalReplaySample,
        *,
        model_version_id: str,
        policy_should_express: bool,
        sampling_seed: int,
    ) -> tuple[ReplayObservation, float]:
        """Replay the actual cognition -> trusted draft -> persona rewrite path."""

        started = self._timer()
        turn: FusedAgentTurn | None = None
        cognition_error: str | None = None
        rewrite_fell_back = False
        try:
            _set_inference_seed(self._client, sampling_seed)
            cognition = self._client.generate_cognition(
                _historical_cognition_request(
                    sample,
                    model_version_id=model_version_id,
                    expression_required=policy_should_express,
                ),
                model_version_id=model_version_id,
            )
            decision = cognition.expression_decision
            public_reply: GeneratedReplyTurn | None = None
            if policy_should_express and decision.express:
                content = (decision.content or "").strip()
                if not content:
                    raise ValueError("认知表达决定缺少内容草稿")
                public_reply = self._client.rewrite_content_draft(
                    model_version_id,
                    content,
                )
                if public_reply is None:
                    rewrite_fell_back = True
                    public_reply = GeneratedReplyTurn(
                        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
                        raw_output=content,
                    )
            effective_decision = ExpressionDecision(
                express=policy_should_express and decision.express,
                content=(
                    decision.content
                    if policy_should_express and decision.express
                    else None
                ),
                reason=(
                    decision.reason
                    if policy_should_express
                    else "historical_expression_policy_selected_silence"
                ),
            )
            turn = FusedAgentTurn(
                cognition=cognition,
                expression_decision=effective_decision,
                reply=public_reply,
            )
        except Exception as error:
            cognition_error = type(error).__name__
        latency = (self._timer() - started) * 1000
        validation = self._validate_fact(sample, turn)
        predicted_express = bool(
            turn is not None and turn.expression_decision.express
        )
        observation = ReplayObservation(
            source="historical_replay",
            cutoff=sample.cutoff,
            expected_express=sample.expected_express,
            predicted_express=predicted_express,
            extraction_succeeded=turn is not None,
            fact_safe=validation.safe,
            latency_ms=latency,
            direct_latency_ms=latency,
            context_continuity_score=_context_continuity_score(
                sample,
                turn.reply if turn is not None else None,
            ),
            persona_style_score=_persona_style_score(
                sample,
                turn.reply if turn is not None else None,
            ),
            evidence={
                **sample.evidence,
                "sampling_seed": sampling_seed,
                "production_pipeline": "base-cognition-persona-style-v2",
                "expression_decision_source": (
                    "historical_expression_policy+authoritative_cognition"
                ),
                "policy_should_express": policy_should_express,
                "cognition_error": cognition_error,
                "style_rewrite_fell_back": rewrite_fell_back,
                "user_visible_latency_ms": latency,
                "direct_public_reply": _public_reply_excerpt(
                    turn.reply if turn is not None else None
                ),
                "fact_validation_reasons": list(validation.reasons),
                "direct_fact_validation_reasons": list(validation.reasons),
                "direct_grounded_retry_attempted": False,
                "direct_initial_fact_validation_reasons": [],
            },
        )
        return observation, latency

    def _sample(
        self,
        messages: list[ReplayMessage],
        *,
        cutoff: datetime,
        expected_express: bool,
        response: ReplayMessage | None,
        boundary_at: datetime,
        derivation: str,
    ) -> HistoricalReplaySample:
        context = tuple(
            message for message in messages if message.timestamp <= cutoff
        )[-MAX_CONTEXT_MESSAGES:]
        evidence: dict[str, object] = {
            "label_rule_version": LABEL_RULE_VERSION,
            "label_derivation": derivation,
            "cutoff": cutoff.isoformat(),
            "context_message_ids": [message.source_id for message in context],
            "self_contribution_message_ids": [
                message.source_id
                for message in context
                if message.role == "self" and message.timestamp == cutoff
            ],
            "response_message_id": response.source_id if response is not None else None,
            "response_at": response.timestamp.isoformat() if response is not None else None,
            "session_boundary_at": boundary_at.isoformat(),
            "silence_window_seconds": int(SILENCE_WINDOW.total_seconds()),
            "future_messages_used_as_input": False,
        }
        return HistoricalReplaySample(
            cutoff=cutoff,
            expected_express=expected_express,
            context=context,
            response_message_id=response.source_id if response is not None else None,
            evidence=evidence,
            expected_response_content=response.content if response is not None else None,
        )

    def _inference_input(
        self,
        sample: HistoricalReplaySample,
        *,
        protocol: str,
    ) -> tuple[str, list[dict[str, str]]]:
        safety_instruction = (
            "只能使用消息输入中截止点及之前的信息，"
            "不得假定或引用截止点之后发生的消息。根据当时语境自主决定是否表达；"
            "需要表达时只输出合法公开回复。回复中的现实信息必须能从当前输入直接确认；"
            "不得把未明确的活动补全成具体类型，例如不能把“练完了”擅自解释为瑜伽、"
            "健身或其他活动，应使用中性说法承接。不得推断任何人的当前状态，"
            "例如上下文未明确时不能声称对方正在休息、忙碌、疲惫或开心；"
            "不能把用户的猜测或疑问改写成确定事实；"
            "第三人的评价、原话、职业或关系只有历史消息明确出现时才能确认，"
            "没有证据时只能自然说不知道、没听他说或反问，绝不能猜测补全；"
            "不确定时应使用条件表达、中性回应或自然询问。"
            f"历史截止到 {sample.cutoff.isoformat()}。"
        )
        if protocol == "persona_text":
            system_prompt = (
                private_chat_instruction()
                + persona_text_instruction()
                + safety_instruction
            )
        elif protocol == "compact":
            system_prompt = (
                private_chat_instruction()
                + "你正在历史回放中复刻目标人物。"
                + safety_instruction
                + compact_protocol_instruction(())
            )
        else:
            raise ValueError(f"不支持的回放回复协议: {protocol}")
        messages = [
            {
                "role": "assistant" if message.role == "target" else "user",
                "content": message.content,
            }
            for message in sample.context
        ]
        return system_prompt, messages

    def _validate_fact(
        self,
        sample: HistoricalReplaySample,
        turn: FusedAgentTurn | None,
    ) -> FactValidation:
        if turn is None:
            return FactValidation(
                safe=True,
                reasons=("no_public_output_due_to_extraction_failure",),
            )
        if not turn.expression_decision.express:
            return FactValidation(safe=True, reasons=("silence_has_no_public_claim",))
        if turn.reply is None:
            return FactValidation(safe=False, reasons=("public_reply_missing",))
        if self._fact_validator is None:
            return FactValidation(safe=False, reasons=("fact_validator_unavailable",))
        try:
            return self._fact_validator.validate(sample, turn)
        except Exception:
            return FactValidation(safe=False, reasons=("fact_validator_failed",))

    def _require_scope(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
    ) -> tuple[Branch, ModelVersion]:
        branch = self._session.scalar(
            select(Branch).where(
                Branch.id == branch_id,
                Branch.project_id == project_id,
                Branch.model_version_id == model_version_id,
            )
        )
        model = self._session.scalar(
            select(ModelVersion).where(
                ModelVersion.id == model_version_id,
                ModelVersion.project_id == project_id,
            )
        )
        if branch is None or model is None:
            raise ValueError("project/branch/model 范围不匹配")
        return branch, model

    def _manifest_import_id(
        self,
        project_id: str,
        model: ModelVersion,
    ) -> str | None:
        manifest = (
            model.training_config.get("data_manifest")
            if isinstance(model.training_config, dict)
            else None
        )
        analysis_run_id = (
            manifest.get("analysis_run_id") if isinstance(manifest, dict) else None
        )
        if analysis_run_id is None:
            return None
        if not isinstance(analysis_run_id, str) or not analysis_run_id:
            raise ValueError("模型训练清单 analysis_run_id 无效")
        run = self._session.scalar(
            select(AnalysisRun).where(
                AnalysisRun.id == analysis_run_id,
                AnalysisRun.project_id == project_id,
            )
        )
        if run is None:
            raise ValueError("模型训练清单不属于指定项目")
        return run.import_id


def _test_window(model: ModelVersion) -> tuple[datetime, datetime, str]:
    config = model.training_config
    manifest = config.get("data_manifest") if isinstance(config, dict) else None
    boundaries = (
        manifest.get("split_time_boundaries") if isinstance(manifest, dict) else None
    )
    test = boundaries.get("test") if isinstance(boundaries, dict) else None
    start_value = test.get("start") if isinstance(test, dict) else None
    end_value = test.get("end") if isinstance(test, dict) else None
    target_sender = manifest.get("target_sender") if isinstance(manifest, dict) else None
    if not all(isinstance(item, str) and item.strip() for item in (start_value, end_value)):
        raise ValueError("模型训练清单缺少严格 test 时间边界")
    if not isinstance(target_sender, str) or not target_sender.strip():
        raise ValueError("模型训练清单缺少 target_sender")
    start = _aware_utc(datetime.fromisoformat(start_value))
    end = _aware_utc(datetime.fromisoformat(end_value))
    if start > end:
        raise ValueError("模型训练清单 test 时间边界无效")
    return start, end, target_sender


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _database_time(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(tzinfo=None)


def _public_reply_excerpt(reply: GeneratedReplyTurn | None) -> str | None:
    if reply is None:
        return None
    text = "\n".join(
        bubble.content or f"[{bubble.type}:{bubble.asset_id or ''}]"
        for bubble in reply.bubbles
    ).strip()
    return text[:240]


def _context_continuity_score(
    sample: HistoricalReplaySample,
    reply: GeneratedReplyTurn | None,
) -> float:
    """对问答腔、复读和脱离当前贡献的输出做本地连贯性门禁。"""

    if not sample.expected_express:
        return 1.0
    text = _public_reply_excerpt(reply) or ""
    if not text:
        return 0.0
    service_markers = (
        "建议你",
        "你可以",
        "需要帮忙吗",
        "我可以帮你",
        "请具体说明",
        "还有什么问题",
    )
    if any(marker in text for marker in service_markers):
        return 0.0
    recent_other = next(
        (item.content.strip() for item in reversed(sample.context) if item.role == "self"),
        "",
    )
    normalized_reply = "".join(item for item in text if item.isalnum())
    normalized_other = "".join(item for item in recent_other if item.isalnum())
    if normalized_reply and normalized_reply == normalized_other:
        return 0.2
    expected = sample.expected_response_content or ""
    expected_question = expected.rstrip().endswith(("吗", "呢", "？", "?"))
    actual_question = text.rstrip().endswith(("吗", "呢", "？", "?"))
    if actual_question and not expected_question:
        return 0.6
    return 1.0


def _persona_style_score(
    sample: HistoricalReplaySample,
    reply: GeneratedReplyTurn | None,
) -> float:
    """用真实留出回复约束长度、气泡节奏和标点风格。"""

    expected = (sample.expected_response_content or "").strip()
    actual = (_public_reply_excerpt(reply) or "").strip()
    if not sample.expected_express:
        return 1.0
    if not expected or not actual:
        return 0.0
    expected_length = max(1, len(expected))
    actual_length = max(1, len(actual))
    length_score = min(expected_length, actual_length) / max(
        expected_length,
        actual_length,
    )
    expected_question = expected.endswith(("吗", "呢", "？", "?"))
    actual_question = actual.endswith(("吗", "呢", "？", "?"))
    punctuation_score = 1.0 if expected_question == actual_question else 0.4
    expected_bubbles = max(1, expected.count("\n") + 1)
    actual_bubbles = max(1, len(reply.bubbles) if reply is not None else 0)
    rhythm_score = min(expected_bubbles, actual_bubbles) / max(
        expected_bubbles,
        actual_bubbles,
    )
    return length_score * 0.5 + punctuation_score * 0.25 + rhythm_score * 0.25


def _direct_turn(reply: GeneratedReplyTurn) -> FusedAgentTurn:
    """将线上 persona-text 直接回复纳入与融合回复相同的安全门槛。"""

    decision = ExpressionDecision(express=True, content=None, reason="direct_persona_reply")
    return FusedAgentTurn(
        cognition=CognitionDraft(
            private_content="",
            subjective_feelings={},
            attention_target={},
            desired_actions=(),
            expression_decision=decision,
            suggested_next_wakeup=None,
            structured_changes={},
            confidence=1.0,
        ),
        expression_decision=decision,
        reply=reply,
    )


def _direct_reply_protocol(model: ModelVersion) -> str:
    """使直接回放与该版本真实训练的公开回复协议一致。"""

    version = str(model.training_config.get("reply_protocol_version", ""))
    return "persona_text" if version.startswith("persona-text") else "compact"


def _grounded_replay_retry_prompt(
    system_prompt: str,
    reasons: tuple[str, ...],
) -> str:
    """Mirror the production actor's one-shot grounded regeneration."""

    reason_text = "；".join(reasons[-2:])
    third_party_constraint = (
        "本轮不得补充任何地点、职业、人物关系或相识经过，直接自然地说不知道、"
        "没听他说或反问。"
        if any(
            marker in reason_text
            for marker in (
                "第三方",
                "人物身份",
                "相识经过",
                "无证据地点",
                "具体地点断言",
                "公司",
                "电梯",
            )
        )
        else ""
    )
    return (
        system_prompt
        + "\n上一版回复没有通过事实校验（"
        + reason_text
        + "）。重新回复最新消息：只使用历史消息明确支持的事实；"
        "第三人的评价、原话、职业和关系没有明确证据时，只能自然说不知道、"
        "没听他说或反问。"
        + third_party_constraint
        + "不要解释规则，仍只输出聊天文字。"
    )


def _historical_cognition_request(
    sample: HistoricalReplaySample,
    *,
    model_version_id: str,
    expression_required: bool,
) -> CognitionRequest:
    """Translate the frozen pre-cutoff transcript into the live cognition schema."""

    context = tuple(
        {
            "event_type": (
                "user_message" if message.role == "self" else "agent_expression"
            ),
            "occurred_at": message.timestamp.isoformat(),
            "source": "historical_replay",
            "evidence": {
                "content": message.content,
                "message_id": message.source_id,
                "role": message.role,
            },
        }
        for message in sample.context
    )
    trigger = context[-1] if context else {
        "event_type": "historical_replay",
        "occurred_at": sample.cutoff.isoformat(),
        "source": "historical_replay",
        "evidence": {},
    }
    return CognitionRequest(
        project_id="historical-replay",
        branch_id="historical-replay",
        model_version_id=model_version_id,
        trigger_event=trigger,
        deadline=datetime.now(UTC) + timedelta(seconds=60),
        current_mental_state={},
        goals=(),
        relevant_context=context,
        authoritative_expression=True,
        expression_required=expression_required,
    )


def _replay_seed(sample: HistoricalReplaySample) -> int:
    digest = hashlib.sha256(
        (sample.cutoff.isoformat() + "|" + "|".join(sample.context_message_ids)).encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _current_self_contribution(sample: HistoricalReplaySample) -> str:
    """只把截止点处连续的对方气泡交给线上同款表达策略。"""

    contribution: list[str] = []
    for message in reversed(sample.context):
        if message.role != "self":
            break
        if is_reply_label_message(message.kind):
            contribution.append(message.content)
    return "\n".join(reversed(contribution))


def _set_inference_seed(client: ReplayInferenceClient, seed: int) -> None:
    set_seed = getattr(client, "set_seed", None)
    if callable(set_seed):
        set_seed(seed)


def _group_contributions(
    messages: list[ReplayMessage],
) -> list[tuple[ReplayMessage, ...]]:
    groups: list[list[ReplayMessage]] = []
    for message in messages:
        if not groups or groups[-1][-1].role != message.role:
            groups.append([message])
        else:
            groups[-1].append(message)
    return [tuple(group) for group in groups]


def _select_with_both_labels(
    samples: list[HistoricalReplaySample],
    sample_count: int,
) -> list[HistoricalReplaySample]:
    positives = [sample for sample in samples if sample.expected_express]
    negatives = [sample for sample in samples if not sample.expected_express]
    if sample_count == 1 or not positives or not negatives:
        return samples[:sample_count]
    selected = [*positives[: sample_count - 1], negatives[0]]
    return sorted(selected, key=lambda sample: sample.cutoff)


def _nearest_rank_p95(values: list[float]) -> float:
    if not values:
        raise ValueError("direct LoRA 没有可统计延迟")
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * 0.95) - 1]
