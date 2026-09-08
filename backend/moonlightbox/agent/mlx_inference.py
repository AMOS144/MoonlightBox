"""人格推理请求协议与历史 MLX runtime 兼容实现。"""

from __future__ import annotations

import gc
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from sqlalchemy.orm import Session

from moonlightbox.agent.inference import (
    InferencePreemptedError,
    PersonaInferenceRequest,
    PersonaInferenceResult,
)
from moonlightbox.branches.generation import (
    GenerationFailedError,
    GeneratorUnavailableError,
)
from moonlightbox.branches.mlx_generation import (
    DatabaseReplyGenerator,
    MlxRuntime,
    MlxSampleUtils,
    MlxTokenizer,
)
from moonlightbox.branches.replies import ReplyStructureError, parse_reply_turn
from moonlightbox.db import Database
from moonlightbox.training.models import ModelVersion


class _RuntimeLike(Protocol):
    def generate_raw(
        self,
        *,
        base_model: str,
        adapter_path: str | None,
        messages: list[dict[str, str]],
        max_tokens: int,
        should_cancel: Callable[[], bool] | None = None,
        deadline: datetime | None = None,
    ) -> str: ...

    def count_text_tokens(self, *, base_model: str, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class PersonaSamplingConfig:
    """保留真人表达的低温采样，避免贪心解码退回通用助手腔。"""

    temperature: float = 0.65
    top_p: float = 0.9
    min_p: float = 0.02
    repetition_penalty: float = 1.02
    repetition_context_size: int = 128


class SharedMlxModelRuntime:
    """只持有一组当前基座模型、LoRA 与 tokenizer。"""

    def __init__(
        self,
        *,
        mlx_runtime: MlxRuntime | None = None,
        sample_utils: MlxSampleUtils | None = None,
        import_module: Callable[[str], object] = importlib.import_module,
        sampling: PersonaSamplingConfig | None = None,
    ) -> None:
        self.loaded_key: tuple[str, str | None] | None = None
        self.model: object | None = None
        self.tokenizer: MlxTokenizer | None = None
        self._mlx_runtime = mlx_runtime
        self._sample_utils = sample_utils
        self._import_module = import_module
        self._sampling = sampling or PersonaSamplingConfig()

    def generate_raw(
        self,
        *,
        base_model: str,
        adapter_path: str | None,
        messages: list[dict[str, str]],
        max_tokens: int,
        should_cancel: Callable[[], bool] | None = None,
        deadline: datetime | None = None,
    ) -> str:
        self._raise_if_stopped(should_cancel, deadline)
        runtime, sample_utils = self._dependencies()
        loaded_key = (base_model, adapter_path)
        if self.loaded_key != loaded_key:
            self.release()
            load_options = {"adapter_path": adapter_path} if adapter_path else {}
            self.model, self.tokenizer = runtime.load(base_model, **load_options)
            self.loaded_key = loaded_key
        if self.model is None or self.tokenizer is None:
            raise GeneratorUnavailableError("本地模型加载失败")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        sampler = sample_utils.make_sampler(
            temp=self._sampling.temperature,
            top_p=self._sampling.top_p,
            min_p=self._sampling.min_p,
        )
        logits_processors = [
            sample_utils.make_repetition_penalty(
                penalty=self._sampling.repetition_penalty,
                context_size=self._sampling.repetition_context_size,
            )
        ]
        stream_generate = getattr(runtime, "stream_generate", None)
        if callable(stream_generate):
            chunks: list[str] = []
            for chunk in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                self._raise_if_stopped(should_cancel, deadline)
                text = chunk if isinstance(chunk, str) else getattr(chunk, "text", None)
                if not isinstance(text, str):
                    raise GeneratorUnavailableError("流式模型返回了无效片段")
                chunks.append(text)
            self._raise_if_stopped(should_cancel, deadline)
            return "".join(chunks)

        result = runtime.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        self._raise_if_stopped(should_cancel, deadline)
        return result

    def set_seed(self, seed: int) -> None:
        core = self._import_module("mlx.core")
        random = getattr(core, "random", None)
        seed_fn = getattr(random, "seed", None)
        if not callable(seed_fn):
            raise GeneratorUnavailableError("MLX runtime 不支持确定性采样")
        seed_fn(seed)

    @staticmethod
    def _raise_if_stopped(
        should_cancel: Callable[[], bool] | None,
        deadline: datetime | None,
    ) -> None:
        if should_cancel is not None and should_cancel():
            raise InferencePreemptedError("推理已被更高优先级请求抢占")
        if deadline is not None and datetime.now(UTC) >= deadline:
            raise InferencePreemptedError("推理已超过截止时间")

    def release(self) -> None:
        """切换版本或关闭服务时释放旧权重。"""

        had_model = self.model is not None or self.tokenizer is not None
        self.model = None
        self.tokenizer = None
        self.loaded_key = None
        if not had_model:
            return
        gc.collect()
        try:
            mlx = importlib.import_module("mlx.core")
        except ModuleNotFoundError:
            return
        clear_cache = getattr(mlx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

    def _dependencies(self) -> tuple[MlxRuntime, MlxSampleUtils]:
        if self._mlx_runtime is not None and self._sample_utils is not None:
            return self._mlx_runtime, self._sample_utils
        try:
            self._mlx_runtime = cast(
                MlxRuntime,
                self._import_module("mlx_lm"),
            )
            self._sample_utils = cast(
                MlxSampleUtils,
                self._import_module("mlx_lm.sample_utils"),
            )
        except ImportError as error:
            raise GeneratorUnavailableError("未安装官方 mlx-lm 依赖") from error
        return self._mlx_runtime, self._sample_utils


class _ExpressionPayload(BaseModel):
    express: bool
    content: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_content_partition(self) -> _ExpressionPayload:
        if self.express and not (self.content or "").strip():
            raise ValueError("决定表达时必须提供公开内容草稿")
        if not self.express and self.content is not None:
            raise ValueError("决定沉默时公开内容必须为空")
        return self


class _WakeupPayload(BaseModel):
    wake_at: datetime
    reason: str
    idempotency_key: str


def _normalize_private_fields(value: object) -> object:
    """兼容模型的简短字符串容器，不增加原输出之外的语义。"""

    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for key in ("subjective_feelings", "attention_target"):
        current = normalized.get(key)
        if isinstance(current, str):
            normalized[key] = {"summary": current}
    actions = normalized.get("desired_actions")
    if isinstance(actions, str):
        normalized["desired_actions"] = [{"content": actions}]
    elif isinstance(actions, list):
        normalized["desired_actions"] = [
            {"content": item} if isinstance(item, str) else item for item in actions
        ]
    if normalized.get("suggested_next_wakeup") is not None and not isinstance(
        normalized.get("suggested_next_wakeup"),
        dict,
    ):
        normalized["suggested_next_wakeup"] = None
    if not isinstance(normalized.get("structured_changes", {}), dict):
        normalized["structured_changes"] = {}
    return normalized


class _CognitionPayload(BaseModel):
    private_content: str = Field(min_length=1)
    subjective_feelings: dict[str, object]
    attention_target: dict[str, object]
    desired_actions: list[dict[str, object]]
    expression_decision: _ExpressionPayload
    suggested_next_wakeup: _WakeupPayload | None
    structured_changes: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, value: object) -> object:
        return _normalize_private_fields(value)


class _FusedPrivatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    private_content: str = Field(min_length=1, max_length=500)
    subjective_feelings: dict[str, object]
    attention_target: dict[str, object]
    desired_actions: list[dict[str, object]]
    suggested_next_wakeup: _WakeupPayload | None
    structured_changes: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, value: object) -> object:
        return _normalize_private_fields(value)


class _FusedExpressionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    express: bool
    content: str | None = None
    reason: str | None = None


class _FusedPayload(BaseModel):
    """严格分隔私密认知与唯一可解析公开区。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    private_cognition: _FusedPrivatePayload
    expression_decision: _FusedExpressionPayload
    public_compact_bubbles: str | None

    @model_validator(mode="after")
    def validate_public_partition(self) -> _FusedPayload:
        public = self.public_compact_bubbles
        if self.expression_decision.express and not (public or "").strip():
            raise ValueError("表达决定为真时公开区不能为空")
        if not self.expression_decision.express and public is not None:
            raise ValueError("沉默决定的公开区必须为 null")
        return self


class _CompactFusedPayload(BaseModel):
    """低延迟融合协议，仍保持私密区与公开区严格分离。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    private: str = Field(
        min_length=1,
        max_length=200,
        validation_alias=AliasChoices("p", "private"),
    )
    express: bool = Field(validation_alias=AliasChoices("e", "express"))
    public: str | None = Field(validation_alias=AliasChoices("o", "public"))

    @model_validator(mode="after")
    def validate_public_partition(self) -> _CompactFusedPayload:
        if self.express and not (self.public or "").strip():
            raise ValueError("表达决定为真时公开区不能为空")
        if not self.express and self.public is not None:
            raise ValueError("沉默决定的公开区必须为 null")
        return self


def _parse_fused_payload(raw_output: str) -> _FusedPayload:
    decoded = json.loads(raw_output)
    try:
        compact = _CompactFusedPayload.model_validate(decoded)
    except ValidationError:
        return _FusedPayload.model_validate(decoded)
    return _FusedPayload.model_validate(
        {
            "private_cognition": {
                "private_content": compact.private,
                "subjective_feelings": {},
                "attention_target": {},
                "desired_actions": [],
                "suggested_next_wakeup": None,
                "structured_changes": {},
                "confidence": 1,
            },
            "expression_decision": {
                "express": compact.express,
                "content": None,
                "reason": "融合紧凑协议决定",
            },
            "public_compact_bubbles": compact.public,
        }
    )


def _parse_strict_compact_fused(
    raw_output: str,
    allowed_sticker_ids: tuple[str, ...],
) -> _FusedPayload:
    """兼容 LoRA 已学会的严格气泡输出，避免仅为外层 JSON 重跑推理。"""

    stripped = raw_output.strip()
    if not stripped.startswith(("<bubble>", "<sticker>")):
        raise ReplyStructureError("融合输出不是严格紧凑气泡")
    parse_reply_turn(
        stripped,
        allowed_sticker_ids=allowed_sticker_ids,
        normalize_compact=False,
    )
    return _FusedPayload.model_validate(
        {
            "private_cognition": {
                "private_content": "决定回应",
                "subjective_feelings": {},
                "attention_target": {},
                "desired_actions": [],
                "suggested_next_wakeup": None,
                "structured_changes": {},
                "confidence": 0.8,
            },
            "expression_decision": {
                "express": True,
                "content": None,
                "reason": "LoRA 直接生成公开气泡",
            },
            "public_compact_bubbles": stripped,
        }
    )


def _conversation_transcript(payload: Mapping[str, object]) -> list[dict[str, str]]:
    """Expose chat events to cognition with unambiguous speaker ownership."""

    raw_context = payload.get("relevant_context")
    if not isinstance(raw_context, list):
        return []
    transcript: list[dict[str, str]] = []
    for item in raw_context:
        if not isinstance(item, dict):
            continue
        event_type = item.get("event_type")
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            continue
        content = evidence.get("content")
        if not isinstance(content, str):
            continue
        if event_type == "agent_expression":
            speaker = "数字人自己（你）"
        elif event_type == "user_message":
            speaker = "聊天对方（用户）"
        else:
            continue
        transcript.append({"speaker": speaker, "content": content})
    return transcript


class DatabasePersonaInferenceBackend:
    """从数据库解析版本，并复用注入的模型 runtime 生成请求结果。"""

    def __init__(
        self,
        database: Database,
        runtime: _RuntimeLike | None = None,
    ) -> None:
        self._database = database
        if runtime is None:
            # 默认后端必须保持 Linux；SharedMlxModelRuntime 只留给历史测试或
            # 显式兼容调用，避免遗漏注入时意外回退到已移除的依赖。
            from moonlightbox.agent.linux_inference import SharedLinuxModelRuntime

            runtime = SharedLinuxModelRuntime()
        self._runtime = runtime
        self._reply_generator = DatabaseReplyGenerator(database, self._runtime)

    def infer(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PersonaInferenceResult:
        self._apply_sampling_seed(request)
        if request.request_type == "reply":
            output = self._generate_reply(request, should_cancel)
        elif request.request_type == "cognition":
            output = self._generate_cognition(request, should_cancel)
        elif request.request_type == "runtime_director":
            output = self._generate_runtime_director(request, should_cancel)
        elif request.request_type == "runtime_actor":
            output = self._generate_runtime_actor(request, should_cancel)
        elif request.request_type == "runtime_token_count":
            output = self._count_runtime_tokens(request)
        else:
            output = self._generate_fused_reply(request, should_cancel)
        return PersonaInferenceResult(
            request_id=request.request_id,
            request_type=request.request_type,
            output=output,
        )

    def _apply_sampling_seed(self, request: PersonaInferenceRequest) -> None:
        seed = request.payload.get("sampling_seed")
        if seed is None:
            return
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise GenerationFailedError("采样 seed 无效")
        set_seed = getattr(self._runtime, "set_seed", None)
        if not callable(set_seed):
            raise GeneratorUnavailableError("人格推理 runtime 不支持确定性采样")
        set_seed(seed)

    def _generate_reply(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None,
    ) -> Mapping[str, object]:
        system_prompt = request.payload.get("system_prompt")
        raw_messages = request.payload.get("messages")
        if not isinstance(system_prompt, str) or not isinstance(raw_messages, list):
            raise GenerationFailedError("回复推理请求结构无效")
        messages = self._validate_messages(raw_messages)
        reply = self._reply_generator.generate(
            request.model_version_id,
            system_prompt,
            messages,
            should_cancel=should_cancel,
            deadline=request.deadline,
        )
        return {
            "bubbles": [
                {
                    "content": bubble.content,
                    "delay_ms": bubble.delay_ms,
                    "type": bubble.type,
                    "asset_id": bubble.asset_id,
                }
                for bubble in reply.bubbles
            ],
            "raw_output": reply.raw_output,
            "degraded": reply.degraded,
            "normalization": reply.normalization,
            "policy_appended": reply.policy_appended,
            "attempted_invalid_sticker_ids": list(reply.attempted_invalid_sticker_ids),
        }

    def _generate_cognition(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None,
    ) -> Mapping[str, object]:
        base_model, _ = self._model_paths(request.model_version_id)
        cognition_payload = dict(request.payload)
        cognition_payload["conversation_transcript"] = _conversation_transcript(cognition_payload)
        user_payload = json.dumps(
            cognition_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_prompt = (
            "你正在生成主体的私密影子认知。只输出一个严格 JSON 对象，"
            "字段必须为 private_content、subjective_feelings、attention_target、"
            "desired_actions、expression_decision、suggested_next_wakeup、"
            "structured_changes、confidence。private_content 必须是少于 120 字的字符串；"
            "expression_decision.express=true 时 content 必须是准备表达的内容草稿，"
            "express=false 时 content 必须为 null；"
            "其余内容保持简短，不得复述输入事件、ID 或上下文列表。"
            "禁止输出聊天气泡或额外文字。"
            "表达内容只能使用 relevant_context、current_mental_state 和 goals 中明确给出的事实；"
            "不得用‘听说’补写第三方近况，不得猜测任何人此刻的位置、活动或安排。"
            "没有证据时改成简短反问或只回应情绪。"
            "relevant_context 按时间排序；agent_expression 是你自己已经公开发送的话。"
            "conversation_transcript 是权威双边记录，严禁把‘数字人自己（你）’说过的话归给用户。"
            "遇到‘什么意思’‘没看懂’等省略追问，必须先联系紧邻的 agent_expression 解读指代。"
            "若用户追问你刚发的单个标点，必须承认那是你发的；不得反问用户为何发送该标点。"
            "此时 content 必须用第一人称简短解释自己当时的情绪，"
            "禁止讲解标点符号的一般含义、定义或举例。"
            "例如记录为‘数字人自己（你）：！’、‘聊天对方（用户）：什么意思呢’时，"
            "content 应类似‘我刚才就是有点惊讶’，绝不能写‘感叹号表示……’。"
            "这是亲密私人聊天，不是客服：content 禁止使用‘您’、‘请您’、‘我可以帮你’、"
            "‘需要帮忙吗’、‘查找相关信息’、‘具体说明’、‘建议你’等服务式措辞。"
        )
        expression_required = request.payload.get("expression_required")
        if expression_required is not None and not isinstance(
            expression_required,
            bool,
        ):
            raise GenerationFailedError("认知表达约束无效")
        if expression_required is True:
            system_prompt += (
                "\n真人行为策略已判定本轮必须回复：expression_decision.express 必须为 true，"
                "content 必须是简短、相关、可直接表达的内容草稿。"
            )
        elif expression_required is False:
            system_prompt += (
                "\n真人行为策略已判定本轮保持沉默：expression_decision.express 必须为 false，"
                "content 必须为 null。"
            )
        shadow_mode = request.payload.get("shadow_mode") is True
        attempts = 1 if shadow_mode else 2
        for attempt in range(attempts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ]
            if attempt:
                messages[0]["content"] += "\n上一个结构无效，请严格按 JSON 协议重试。"
            raw_output = self._runtime.generate_raw(
                base_model=base_model,
                # The persona LoRA is trained to express a trusted content draft,
                # not to emit private JSON or decide facts. Keep reasoning on the
                # clean instruction-tuned base model and apply the adapter only
                # to the eventual public style-transfer step.
                adapter_path=None,
                messages=messages,
                max_tokens=256,
                should_cancel=should_cancel,
                deadline=request.deadline,
            )
            try:
                parsed = _CognitionPayload.model_validate(json.loads(raw_output))
            except (json.JSONDecodeError, ValidationError, TypeError):
                if shadow_mode and raw_output.strip():
                    return {
                        "private_content": raw_output.strip(),
                        "subjective_feelings": {},
                        "attention_target": {},
                        "desired_actions": [],
                        "expression_decision": {
                            "express": False,
                            "content": None,
                            "reason": "影子认知尚未结构化",
                        },
                        "suggested_next_wakeup": None,
                        "structured_changes": {
                            "extraction_status": "pending",
                        },
                        "confidence": 0.2,
                    }
                continue
            if (
                expression_required is not None
                and parsed.expression_decision.express is not expression_required
            ):
                continue
            return parsed.model_dump(mode="json")
        raise GenerationFailedError("模型未能生成有效认知结构")

    def _generate_runtime_director(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None,
    ) -> Mapping[str, object]:
        """以未挂 LoRA 的基础模型生成 Runtime v1 的 LifeDecision。

        LoRA 只承担 PersonaActor 的语言风格迁移；Director 的职责是结构化决策，
        因此明确使用同一模型版本对应的基础模型，避免让表达风格污染状态决策。
        """

        system_prompt = request.payload.get("system_prompt")
        raw_messages = request.payload.get("messages")
        if not isinstance(system_prompt, str) or not isinstance(raw_messages, list):
            raise GenerationFailedError("Runtime Director 请求结构无效")
        messages = self._validate_runtime_messages(raw_messages)
        base_model, _ = self._model_paths(request.model_version_id)
        raw_output = self._runtime.generate_raw(
            base_model=base_model,
            adapter_path=None,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            max_tokens=512,
            should_cancel=should_cancel,
            deadline=request.deadline,
        )
        # 解析和业务校验属于 Runtime v1；推理侧只传输原始 JSON，避免复用旧认知结构。
        return {"content": raw_output}

    def _generate_runtime_actor(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None,
    ) -> Mapping[str, object]:
        """使用当前模型版本的 LoRA 生成 PersonaActor 的严格结构化表达。"""

        system_prompt = request.payload.get("system_prompt")
        raw_messages = request.payload.get("messages")
        if not isinstance(system_prompt, str) or not isinstance(raw_messages, list):
            raise GenerationFailedError("Runtime PersonaActor 请求结构无效")
        messages = self._validate_runtime_messages(raw_messages)
        base_model, adapter_path = self._model_paths(request.model_version_id)
        raw_output = self._runtime.generate_raw(
            base_model=base_model,
            adapter_path=adapter_path,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            max_tokens=512,
            should_cancel=should_cancel,
            deadline=request.deadline,
        )
        return {"content": raw_output}

    def _count_runtime_tokens(self, request: PersonaInferenceRequest) -> Mapping[str, object]:
        """只读取 Director 基座 tokenizer；不会加载 LoRA 或生成文本。"""

        text = request.payload.get("text")
        if not isinstance(text, str):
            raise GenerationFailedError("Runtime token 计数请求结构无效")
        base_model, _ = self._model_paths(request.model_version_id)
        counter = getattr(self._runtime, "count_text_tokens", None)
        if not callable(counter):
            raise GeneratorUnavailableError("当前人格 runtime 不支持真实 tokenizer 计数")
        return {"count": int(counter(base_model=base_model, text=text))}

    def _generate_fused_reply(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None,
    ) -> Mapping[str, object]:
        system_prompt = request.payload.get("system_prompt")
        raw_messages = request.payload.get("messages")
        raw_allowed = request.payload.get("allowed_sticker_ids", [])
        if (
            not isinstance(system_prompt, str)
            or not isinstance(raw_messages, list)
            or not isinstance(raw_allowed, list)
            or any(not isinstance(item, str) for item in raw_allowed)
        ):
            raise GenerationFailedError("融合推理请求结构无效")
        messages = self._validate_messages(raw_messages)
        base_model, adapter_path = self._model_paths(request.model_version_id)
        partition_prompt = (
            f"{system_prompt}\n\n"
            "只输出一行严格JSON，且仅有p、e、o三个字段。"
            "p为少于15字且绝不公开的私密判断；e为布尔值；"
            "o是唯一公开区，e=false时必须为null，"
            "e=true时必须是至少一个合法紧凑气泡。"
            '示例：{"p":"想回应","e":true,'
            '"o":"<bubble>好呀</bubble>"}。'
            "严禁把p复制到o，严禁输出额外字段或解释。"
        )
        parsed: _FusedPayload | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            retry_prompt = (
                partition_prompt
                if attempt == 0
                else (
                    partition_prompt + f"\n第 {attempt} 个输出结构无效。不要换行，"
                    "请精确复制示例结构并替换字段值。"
                )
            )
            raw_output = self._runtime.generate_raw(
                base_model=base_model,
                adapter_path=adapter_path,
                messages=[
                    {"role": "system", "content": retry_prompt},
                    *messages,
                ],
                max_tokens=96,
                should_cancel=should_cancel,
                deadline=request.deadline,
            )
            try:
                parsed = _parse_fused_payload(raw_output)
            except (json.JSONDecodeError, ValidationError, TypeError) as error:
                last_error = error
                try:
                    parsed = _parse_strict_compact_fused(
                        raw_output,
                        tuple(raw_allowed),
                    )
                except ReplyStructureError as compact_error:
                    last_error = compact_error
                    continue
            break
        if parsed is None:
            raise GenerationFailedError("模型未能生成有效融合结构") from last_error

        reply: Mapping[str, object] | None = None
        if parsed.expression_decision.express:
            assert parsed.public_compact_bubbles is not None
            turn = parse_reply_turn(
                parsed.public_compact_bubbles,
                allowed_sticker_ids=tuple(raw_allowed),
                normalize_compact=True,
            )
            reply = {
                "bubbles": [
                    {
                        "content": bubble.content,
                        "delay_ms": bubble.delay_ms,
                        "type": bubble.type,
                        "asset_id": bubble.asset_id,
                    }
                    for bubble in turn.bubbles
                ],
                "raw_output": turn.raw_output,
                "degraded": turn.degraded,
                "normalization": turn.normalization,
                "policy_appended": turn.policy_appended,
                "attempted_invalid_sticker_ids": list(turn.attempted_invalid_sticker_ids),
            }
        return {
            "private_cognition": parsed.private_cognition.model_dump(mode="json"),
            "expression_decision": parsed.expression_decision.model_dump(mode="json"),
            "reply": reply,
        }

    def _model_paths(self, model_version_id: str) -> tuple[str, str]:
        with Session(self._database.engine) as session:
            version = session.get(ModelVersion, model_version_id)
            if version is None:
                raise GeneratorUnavailableError("模型版本不存在")
            return version.base_model, version.adapter_path

    @staticmethod
    def _validate_messages(raw_messages: list[object]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise GenerationFailedError("回复消息结构无效")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise GenerationFailedError("回复消息结构无效")
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _validate_runtime_messages(raw_messages: list[object]) -> list[dict[str, str]]:
        """验证 Runtime 的 LangChain 消息投影，支持工具结果回填。"""

        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise GenerationFailedError("Runtime Director 消息结构无效")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant", "tool"} or not isinstance(content, str):
                raise GenerationFailedError("Runtime Director 消息结构无效")
            # MLX chat template 不认识 tool 角色时，将只读工具结果安全投影为用户上下文。
            messages.append({"role": "user" if role == "tool" else role, "content": content})
        return messages

    def close(self) -> None:
        """释放共享 runtime 当前持有的模型。"""

        release = getattr(self._runtime, "release", None)
        if callable(release):
            release()
