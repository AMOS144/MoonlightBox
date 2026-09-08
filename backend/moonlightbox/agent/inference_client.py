"""人格推理服务的标准库 HTTP 客户端。"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Literal, overload
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from moonlightbox.agent.inference import InferencePriority
from moonlightbox.agent.types import (
    CognitionDraft,
    CognitionRequest,
    ExpressionDecision,
    FusedAgentTurn,
    WakeupSuggestion,
)
from moonlightbox.branches.generation import (
    GenerationFailedError,
    GeneratorUnavailableError,
)
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
from moonlightbox.training.bubble_protocol import (
    evidence_grounded_style_fallback_lines,
    persona_style_transfer_instruction,
)


class _BubbleResponse(BaseModel):
    content: str | None = None
    delay_ms: int
    type: Literal["text", "sticker", "emoji"] = "text"
    asset_id: str | None = None


class _ReplyResponse(BaseModel):
    bubbles: list[_BubbleResponse] = Field(min_length=1)
    raw_output: str = ""
    degraded: bool = False
    normalization: str | None = None
    policy_appended: bool = False
    attempted_invalid_sticker_ids: list[str] = Field(default_factory=list)


class _ExpressionResponse(BaseModel):
    express: bool
    content: str | None = None
    reason: str | None = None


class _WakeupResponse(BaseModel):
    wake_at: datetime
    reason: str
    idempotency_key: str


class _CognitionResponse(BaseModel):
    private_content: str
    subjective_feelings: dict[str, object]
    attention_target: dict[str, object]
    desired_actions: list[dict[str, object]]
    expression_decision: _ExpressionResponse
    suggested_next_wakeup: _WakeupResponse | None
    structured_changes: dict[str, object] = Field(default_factory=dict)
    confidence: float = 1.0


class _FusedPrivateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    private_content: str
    subjective_feelings: dict[str, object]
    attention_target: dict[str, object]
    desired_actions: list[dict[str, object]]
    suggested_next_wakeup: _WakeupResponse | None
    structured_changes: dict[str, object] = Field(default_factory=dict)
    confidence: float = 1.0


class _FusedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    private_cognition: _FusedPrivateResponse
    expression_decision: _ExpressionResponse
    reply: _ReplyResponse | None

    @model_validator(mode="after")
    def validate_decision(self) -> _FusedResponse:
        if self.expression_decision.express != (self.reply is not None):
            raise ValueError("融合表达决定与公开回复不一致")
        return self


class _Envelope(BaseModel):
    request_id: str
    request_type: str
    output: dict[str, object]


class _RuntimeDirectorResponse(BaseModel):
    """Runtime v1 Director 的传输协议，不复用旧 CognitionDraft。"""

    content: str = Field(min_length=1)


class _RuntimeTokenCountResponse(BaseModel):
    """Runtime ContextAssembler 使用的真实 tokenizer 计数协议。"""

    count: int = Field(ge=0)


class PersonaInferenceClient:
    """同时适配 BranchGenerator 与 CognitionGenerator 的本地客户端。"""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 60.0,
        default_model_version_id: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._default_model_version_id = default_model_version_id
        self._sampling_seed: int | None = None

    def set_seed(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("采样 seed 不能为负数")
        self._sampling_seed = seed

    @overload
    def generate(
        self,
        request_or_model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn: ...

    @overload
    def generate(
        self,
        request_or_model: CognitionRequest,
        *,
        model_version_id: str | None = None,
    ) -> CognitionDraft: ...

    def generate(
        self,
        request_or_model: CognitionRequest | str,
        system_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        *,
        model_version_id: str | None = None,
    ) -> GeneratedReplyTurn | CognitionDraft:
        if isinstance(request_or_model, CognitionRequest):
            resolved_model = (
                model_version_id
                or request_or_model.model_version_id
                or self._default_model_version_id
            )
            if not resolved_model:
                raise GeneratorUnavailableError("认知推理缺少模型版本")
            return self._generate_cognition(request_or_model, resolved_model)
        if system_prompt is None or messages is None:
            raise GenerationFailedError("回复推理参数无效")
        return self._generate_reply(request_or_model, system_prompt, messages)

    def _generate_reply(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        envelope = self._post(
            "/v1/inference/reply",
            {
                "request_id": str(uuid4()),
                "request_type": "reply",
                "priority": int(InferencePriority.REALTIME),
                "deadline": (datetime.now(UTC) + timedelta(seconds=self._timeout)).isoformat(),
                "model_version_id": model_version_id,
                "payload": {
                    "system_prompt": system_prompt,
                    "messages": messages,
                    **self._seed_payload(),
                },
            },
        )
        try:
            payload = _ReplyResponse.model_validate(envelope.output)
        except ValidationError as error:
            raise GenerationFailedError("人格推理回复协议无效") from error
        return GeneratedReplyTurn(
            bubbles=tuple(
                GeneratedBubble(
                    content=bubble.content,
                    delay_ms=bubble.delay_ms,
                    type=bubble.type,
                    asset_id=bubble.asset_id,
                )
                for bubble in payload.bubbles
            ),
            raw_output=payload.raw_output,
            degraded=payload.degraded,
            normalization=payload.normalization,
            policy_appended=payload.policy_appended,
            attempted_invalid_sticker_ids=tuple(payload.attempted_invalid_sticker_ids),
        )

    def rewrite_content_draft(
        self,
        model_version_id: str,
        draft: str,
    ) -> GeneratedReplyTurn | None:
        """Use the style-transfer task trained into the persona adapter."""

        return self.rewrite_content_draft_with_examples(
            model_version_id,
            draft,
            (),
        )

    def rewrite_content_draft_with_examples(
        self,
        model_version_id: str,
        draft: str,
        style_examples: tuple[str, ...],
    ) -> GeneratedReplyTurn | None:
        """Rewrite a trusted draft with cutoff-safe authentic style evidence."""

        from moonlightbox.branches.generation_semantics import (
            rewrite_preserves_hard_semantics,
        )

        rewritten = self._generate_reply(
            model_version_id,
            persona_style_transfer_instruction(style_examples),
            [{"role": "user", "content": "内容草稿：\n" + draft}],
        )
        text = "\n".join(
            bubble.content or "" for bubble in rewritten.bubbles if bubble.type == "text"
        )
        fallback_lines = evidence_grounded_style_fallback_lines(
            draft,
            text,
            style_examples,
        )
        if fallback_lines != tuple(
            bubble.content
            for bubble in rewritten.bubbles
            if bubble.type == "text" and bubble.content is not None
        ):
            rewritten = GeneratedReplyTurn(
                bubbles=tuple(
                    GeneratedBubble(type="text", content=line, delay_ms=0)
                    for line in fallback_lines
                ),
                raw_output="\n".join(fallback_lines),
            )
            text = "\n".join(fallback_lines)
        return rewritten if rewrite_preserves_hard_semantics(draft, text) else None

    def generate_cognition(
        self,
        request: CognitionRequest,
        *,
        model_version_id: str | None = None,
    ) -> CognitionDraft:
        """Explicit cognition entry point used by production-equivalent replay."""

        resolved_model = (
            model_version_id or request.model_version_id or self._default_model_version_id
        )
        if not resolved_model:
            raise GeneratorUnavailableError("认知推理缺少模型版本")
        return self._generate_cognition(request, resolved_model)

    def _generate_cognition(
        self,
        request: CognitionRequest,
        model_version_id: str,
    ) -> CognitionDraft:
        envelope = self._post(
            "/v1/inference/cognition",
            {
                "request_id": str(uuid4()),
                "request_type": "cognition",
                # Director 是发送前的结构化决策，不能按离线反思排队；真正的文本
                # PersonaActor 仍以 REALTIME 优先级执行。
                "priority": int(InferencePriority.PRE_SEND),
                "deadline": request.deadline.isoformat(),
                "model_version_id": model_version_id,
                "payload": {
                    "shadow_mode": not request.authoritative_expression,
                    "expression_required": request.expression_required,
                    **self._seed_payload(),
                    "project_id": request.project_id,
                    "branch_id": request.branch_id,
                    "trigger_event": dict(request.trigger_event),
                    "deadline": request.deadline.isoformat(),
                    "current_mental_state": dict(request.current_mental_state),
                    "goals": [dict(item) for item in request.goals],
                    "decision_context": dict(request.decision_context),
                    "relevant_context": [dict(item) for item in request.relevant_context],
                },
            },
        )
        try:
            payload = _CognitionResponse.model_validate(envelope.output)
        except ValidationError as error:
            raise GenerationFailedError("人格推理认知协议无效") from error
        wakeup = payload.suggested_next_wakeup
        return CognitionDraft(
            private_content=payload.private_content,
            subjective_feelings=payload.subjective_feelings,
            attention_target=payload.attention_target,
            desired_actions=tuple(payload.desired_actions),
            expression_decision=ExpressionDecision(
                express=payload.expression_decision.express,
                content=payload.expression_decision.content,
                reason=payload.expression_decision.reason,
            ),
            suggested_next_wakeup=(
                WakeupSuggestion(
                    wake_at=wakeup.wake_at,
                    reason=wakeup.reason,
                    idempotency_key=wakeup.idempotency_key,
                )
                if wakeup is not None
                else None
            ),
            structured_changes=payload.structured_changes,
            confidence=payload.confidence,
        )

    def generate_fused(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        allowed_sticker_ids: tuple[str, ...],
    ) -> FusedAgentTurn:
        """通过实时优先级端点生成融合认知与可选回复。"""

        envelope = self._post(
            "/v1/inference/fused-reply",
            {
                "request_id": str(uuid4()),
                "request_type": "fused_reply",
                "priority": int(InferencePriority.REALTIME),
                "deadline": (datetime.now(UTC) + timedelta(seconds=self._timeout)).isoformat(),
                "model_version_id": model_version_id,
                "payload": {
                    "system_prompt": system_prompt,
                    "messages": messages,
                    "allowed_sticker_ids": list(allowed_sticker_ids),
                    **self._seed_payload(),
                },
            },
        )
        try:
            payload = _FusedResponse.model_validate(envelope.output)
        except ValidationError as error:
            raise GenerationFailedError("人格融合推理协议无效") from error
        decision = ExpressionDecision(
            express=payload.expression_decision.express,
            content=payload.expression_decision.content,
            reason=payload.expression_decision.reason,
        )
        private = payload.private_cognition
        wakeup = private.suggested_next_wakeup
        cognition = CognitionDraft(
            private_content=private.private_content,
            subjective_feelings=private.subjective_feelings,
            attention_target=private.attention_target,
            desired_actions=tuple(private.desired_actions),
            expression_decision=decision,
            suggested_next_wakeup=(
                WakeupSuggestion(
                    wake_at=wakeup.wake_at,
                    reason=wakeup.reason,
                    idempotency_key=wakeup.idempotency_key,
                )
                if wakeup is not None
                else None
            ),
            structured_changes=private.structured_changes,
            confidence=private.confidence,
        )
        reply_payload = payload.reply
        reply = (
            GeneratedReplyTurn(
                bubbles=tuple(
                    GeneratedBubble(
                        content=bubble.content,
                        delay_ms=bubble.delay_ms,
                        type=bubble.type,
                        asset_id=bubble.asset_id,
                    )
                    for bubble in reply_payload.bubbles
                ),
                raw_output=reply_payload.raw_output,
                degraded=reply_payload.degraded,
                normalization=reply_payload.normalization,
                policy_appended=reply_payload.policy_appended,
                attempted_invalid_sticker_ids=tuple(reply_payload.attempted_invalid_sticker_ids),
            )
            if reply_payload is not None
            else None
        )
        return FusedAgentTurn(
            cognition=cognition,
            expression_decision=decision,
            reply=reply,
        )

    def generate_runtime_director(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """请求 Linux 人格服务执行 Runtime Director。

        Worker/API 不加载 PyTorch 权重；基座与现有 PEFT LoRA 只驻留在独立服务。
        """

        envelope = self._post(
            "/v1/inference/runtime-director",
            {
                "request_id": str(uuid4()),
                "request_type": "runtime_director",
                "priority": int(InferencePriority.OFFLINE),
                "deadline": (datetime.now(UTC) + timedelta(seconds=self._timeout)).isoformat(),
                "model_version_id": model_version_id,
                "payload": {
                    "system_prompt": system_prompt,
                    "messages": messages,
                    **self._seed_payload(),
                },
            },
        )
        try:
            return _RuntimeDirectorResponse.model_validate(envelope.output).content
        except ValidationError as error:
            raise GenerationFailedError("Runtime Director 推理协议无效") from error

    def generate_runtime_actor(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """请求现有 LoRA 生成 Runtime PersonaActor 的 JSON 或内部工具调用。"""

        envelope = self._post(
            "/v1/inference/runtime-actor",
            {
                "request_id": str(uuid4()),
                "request_type": "runtime_actor",
                "priority": int(InferencePriority.REALTIME),
                "deadline": (datetime.now(UTC) + timedelta(seconds=self._timeout)).isoformat(),
                "model_version_id": model_version_id,
                "payload": {
                    "system_prompt": system_prompt,
                    "messages": messages,
                    **self._seed_payload(),
                },
            },
        )
        try:
            return _RuntimeDirectorResponse.model_validate(envelope.output).content
        except ValidationError as error:
            raise GenerationFailedError("Runtime PersonaActor 推理协议无效") from error

    def count_runtime_director_tokens(self, model_version_id: str, text: str) -> int:
        """委托常驻 Linux 人格服务，避免 Worker 自己加载 tokenizer/模型。"""

        envelope = self._post(
            "/v1/inference/runtime-token-count",
            {
                "request_id": str(uuid4()),
                "request_type": "runtime_token_count",
                "priority": int(InferencePriority.OFFLINE),
                "deadline": (datetime.now(UTC) + timedelta(seconds=self._timeout)).isoformat(),
                "model_version_id": model_version_id,
                "payload": {"text": text},
            },
        )
        try:
            return _RuntimeTokenCountResponse.model_validate(envelope.output).count
        except ValidationError as error:
            raise GenerationFailedError("Runtime token 计数协议无效") from error

    def _seed_payload(self) -> dict[str, int]:
        return {"sampling_seed": self._sampling_seed} if self._sampling_seed is not None else {}

    def _post(self, path: str, payload: dict[str, object]) -> _Envelope:
        request = Request(
            self._base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            raise GeneratorUnavailableError("人格推理服务不可用") from error
        try:
            return _Envelope.model_validate_json(body)
        except ValidationError as error:
            raise GenerationFailedError("人格推理服务协议无效") from error
