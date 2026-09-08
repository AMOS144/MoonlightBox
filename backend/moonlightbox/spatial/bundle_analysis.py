import importlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.spatial.episodes import EpisodeMessage, extract_structured_location

MentionType = Literal[
    "named_poi",
    "administrative",
    "personal_anchor",
    "relative_place",
    "deictic_place",
    "route_or_station",
    "unknown",
]
SpatialRelation = Literal[
    "at",
    "to",
    "from",
    "near",
    "through",
    "discussed",
    "unknown",
]
MovementPhase = Literal[
    "stationary",
    "departing",
    "in_transit",
    "arriving",
    "planned",
    "unknown",
]
AssertionMode = Literal[
    "observed",
    "reported",
    "planned",
    "hypothetical",
    "negated",
    "cancelled",
    "unknown",
]


class BundlePlaceMention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    message_id: str = Field(min_length=1)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    raw_text: str = Field(min_length=1)
    mention_type: MentionType
    subject: str = Field(min_length=1)
    relation: SpatialRelation
    movement_phase: MovementPhase
    assertion_mode: AssertionMode
    time_start: datetime | None = None
    time_end: datetime | None = None
    time_precision: Literal["message", "minute", "hour", "day", "range", "unknown"]
    evidence_message_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_span(self) -> "BundlePlaceMention":
        if self.span_start is None and self.span_end is not None:
            raise ValueError("span_start 与 span_end 必须同时存在")
        if self.span_start is not None and self.span_end is None:
            raise ValueError("span_start 与 span_end 必须同时存在")
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end <= self.span_start
        ):
            raise ValueError("span_end 必须大于 span_start")
        return self


class BundlePlaceMentionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mentions: list[BundlePlaceMention]


@dataclass(frozen=True, slots=True)
class BundleAnalysisResult:
    mentions: tuple[BundlePlaceMention, ...]
    method: str
    version: str
    rejected_count: int = 0


class BundleAnalyzer(Protocol):
    def analyze(
        self,
        *,
        messages: list[EpisodeMessage],
        target_person_id: str,
        run_id: str,
        bundle_id: str,
    ) -> BundleAnalysisResult: ...


SPATIAL_BUNDLE_SYSTEM_PROMPT = """
你只负责从一小段私人聊天原文中提取有证据的物理空间提及，不负责把地点链接到地图 POI。
每条输入都带有真实 message_id、说话者 ID、说话者角色和时间。subject 必须填写地点所描述的
人物 ID；确实无法判断才写 unknown。raw_text 必须是原消息里的连续原文，不能创造地点名。
结合上下文区分已发生、计划、假设、否定和取消；“网上、群里、视频里”等非物理空间不提取。
“家、公司、楼下、那里、老地方”可以提取，但不得擅自补成具体地址。时间无法从原文确定时，
使用消息时间并将 time_precision 设为 message。evidence_message_ids 只能引用输入中的消息。
没有可靠地点时返回 {"mentions": []}。只返回符合 schema 的 JSON。
""".strip()


class StructuredLocationBundleAnalyzer:
    """不运行语言模型时，只保留聊天平台自带的结构化位置消息。"""

    version = "spatial-structured-location-v1"

    def analyze(
        self,
        *,
        messages: list[EpisodeMessage],
        target_person_id: str,
        run_id: str,
        bundle_id: str,
    ) -> BundleAnalysisResult:
        del target_person_id, run_id, bundle_id
        mentions: list[BundlePlaceMention] = []
        for message in messages:
            location = extract_structured_location(message.raw)
            if location is None:
                continue
            raw_text = location.label or location.address or "共享位置"
            mentions.append(
                BundlePlaceMention(
                    message_id=message.id,
                    raw_text=raw_text,
                    mention_type="named_poi" if location.label else "unknown",
                    subject=message.participant_id,
                    relation="at",
                    movement_phase="stationary",
                    assertion_mode="observed",
                    time_start=message.timestamp,
                    time_precision="message",
                    evidence_message_ids=[message.id],
                    confidence=1.0,
                )
            )
        return BundleAnalysisResult(
            mentions=tuple(mentions),
            method="structured_location",
            version=self.version,
        )


class CloudBundleAnalyzer:
    version = "spatial-cloud-bundle-v1"

    def __init__(self, client: NodeAnalysisCloudClient) -> None:
        self._client = client

    def analyze(
        self,
        *,
        messages: list[EpisodeMessage],
        target_person_id: str,
        run_id: str,
        bundle_id: str,
    ) -> BundleAnalysisResult:
        payload = _analysis_payload(messages, target_person_id)
        response = self._client.create_structured_completion(
            system_content=SPATIAL_BUNDLE_SYSTEM_PROMPT,
            user_content=json.dumps(payload, ensure_ascii=False),
            response_model=BundlePlaceMentionBatch,
            operation_id="spatial_bundle_analysis",
            window_id=bundle_id,
            run_id=run_id,
        )
        if not isinstance(response, BundlePlaceMentionBatch):
            raise TypeError("空间分析器返回类型无效")
        mentions, rejected = _retain_supported(response.mentions, messages)
        return BundleAnalysisResult(
            mentions=mentions,
            method="cloud_structured_output",
            version=self.version,
            rejected_count=rejected,
        )


class LocalMlxBundleAnalyzer:
    version = "spatial-mlx-bundle-v1"

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._loaded: tuple[Any, Any, Any, Any] | None = None

    def analyze(
        self,
        *,
        messages: list[EpisodeMessage],
        target_person_id: str,
        run_id: str,
        bundle_id: str,
    ) -> BundleAnalysisResult:
        del run_id, bundle_id
        runtime, sample_utils, model, tokenizer = self._load()
        chat = [
            {"role": "system", "content": SPATIAL_BUNDLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    _analysis_payload(messages, target_person_id),
                    ensure_ascii=False,
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = runtime.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=2048,
            sampler=sample_utils.make_sampler(temp=0.1, top_p=0.9, min_p=0.05),
            verbose=False,
        )
        raw = _extract_json_object(output)
        parsed = BundlePlaceMentionBatch.model_validate(raw)
        mentions, rejected = _retain_supported(parsed.mentions, messages)
        return BundleAnalysisResult(
            mentions=mentions,
            method="local_mlx",
            version=self.version,
            rejected_count=rejected,
        )

    def _load(self) -> tuple[Any, Any, Any, Any]:
        if self._loaded is not None:
            return self._loaded
        runtime = importlib.import_module("mlx_lm")
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        model, tokenizer = runtime.load(self._model_path)
        self._loaded = runtime, sample_utils, model, tokenizer
        return self._loaded


class LocalLinuxBundleAnalyzer:
    """在 Linux Transformers 基座上执行本地空间结构化提取。"""

    version = "spatial-linux-bundle-v1"

    def __init__(self, model_path: str, *, device: str = "auto", load_in_4bit: bool = True) -> None:
        from moonlightbox.agent.linux_inference import SharedLinuxModelRuntime

        self._model_path = model_path
        self._runtime = SharedLinuxModelRuntime(device=device, load_in_4bit=load_in_4bit)

    def analyze(
        self,
        *,
        messages: list[EpisodeMessage],
        target_person_id: str,
        run_id: str,
        bundle_id: str,
    ) -> BundleAnalysisResult:
        del run_id, bundle_id
        output = self._runtime.generate_raw(
            base_model=self._model_path,
            adapter_path=None,
            messages=[
                {"role": "system", "content": SPATIAL_BUNDLE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        _analysis_payload(messages, target_person_id),
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=2048,
        )
        parsed = BundlePlaceMentionBatch.model_validate(_extract_json_object(output))
        mentions, rejected = _retain_supported(parsed.mentions, messages)
        return BundleAnalysisResult(
            mentions=mentions,
            method="local_linux",
            version=self.version,
            rejected_count=rejected,
        )


def _analysis_payload(
    messages: list[EpisodeMessage],
    target_person_id: str,
) -> dict[str, object]:
    return {
        "target_person_id": target_person_id,
        "messages": [
            {
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat(),
                "participant_id": message.participant_id,
                "participant_name": message.participant_name,
                "participant_role": message.participant_role,
                "kind": message.kind,
                "content": message.content,
            }
            for message in messages
        ],
    }


def _retain_supported(
    mentions: list[BundlePlaceMention],
    messages: list[EpisodeMessage],
) -> tuple[tuple[BundlePlaceMention, ...], int]:
    by_id = {message.id: message for message in messages}
    retained: list[BundlePlaceMention] = []
    for mention in mentions:
        source = by_id.get(mention.message_id)
        if source is None or not set(mention.evidence_message_ids).issubset(by_id):
            continue
        structured_location = extract_structured_location(source.raw)
        if mention.raw_text not in source.content and structured_location is None:
            continue
        if mention.span_start is not None and mention.span_end is not None:
            if source.content[mention.span_start : mention.span_end] != mention.raw_text:
                continue
        retained.append(mention)
    return tuple(retained), len(mentions) - len(retained)


def _extract_json_object(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise ValueError("本地模型输出必须是字符串")
    cleaned = value.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("本地模型输出中不存在 JSON 对象")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("本地模型输出根节点必须是对象")
    return cast(dict[str, object], parsed)


def parse_bundle_analysis(value: object) -> BundlePlaceMentionBatch:
    """测试和导入工具使用的严格边界。"""

    try:
        return BundlePlaceMentionBatch.model_validate(value)
    except ValidationError as error:
        raise ValueError("空间分析输出不符合协议") from error
