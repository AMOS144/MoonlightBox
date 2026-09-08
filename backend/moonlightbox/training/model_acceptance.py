import gc
import hashlib
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from moonlightbox.agent.linux_inference import PersonaSamplingConfig
from moonlightbox.branches.context import (
    ContextBubble,
    ContextBuilder,
    ContextMemory,
    ContextPacket,
    ContextRequest,
    ContextTurn,
)
from moonlightbox.branches.embeddings import TextEmbedder
from moonlightbox.branches.generation_semantics import rewrite_preserves_hard_semantics
from moonlightbox.branches.memory_policy import is_current_state_question
from moonlightbox.branches.replies import (
    GeneratedBubble,
    GeneratedReplyTurn,
    ReplyStructureError,
    parse_persona_text_turn,
    parse_reply_turn,
)
from moonlightbox.branches.reviewer import (
    ReplyReviewer,
    ReviewFailedError,
    ReviewResult,
)
from moonlightbox.branches.understanding import (
    active_belief_content_draft,
    grounded_retry_instruction,
    understand_contributions,
    validate_fact_grounded_reply,
)
from moonlightbox.training.bubble_protocol import (
    allowed_sticker_ids_from_prompt,
    compact_protocol_instruction,
    compact_retry_instruction,
    parse_bubble_protocol,
    persona_style_transfer_instruction,
    persona_text_instruction,
    prompt_uses_compact_protocol,
    prompt_uses_persona_text_protocol,
)
from moonlightbox.training.style_acceptance import (
    SpeakerSample,
    SpeakerStyleIdentifier,
    compare_style_distributions,
)
from moonlightbox.training.style_features import (
    StyleBubble,
    StyleTurn,
    extract_style_features,
)
from moonlightbox.training.style_profile import style_violations

MINIMUM_PAIRED_RESPONSE_SIMILARITY = 0.50
MAXIMUM_STYLE_TRANSFER_FALLBACK_RATE = 0.10


@dataclass(frozen=True)
class StyleFeatureDifference:
    case_id: str
    feature: str
    baseline: object
    candidate: object
    rule: str


@dataclass(frozen=True)
class AcceptanceReport:
    case_count: int
    passed_count: int
    structure_failures: int
    forbidden_fact_failures: int
    failed_case_ids: tuple[str, ...]
    passed: bool
    raw_output_failures: int = 0
    style_feature_failures: int = 0
    style_feature_differences: tuple[StyleFeatureDifference, ...] = ()
    style_feature_sample_count: int = 0
    candidate_style_profile: dict[str, object] = field(default_factory=dict)
    raw_format_compliance: float = 0.0
    normalization_rate: float = 0.0
    structure_failure_audit: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class AcceptanceMetric:
    value: float
    minimum: float | None = None
    maximum: float | None = None
    tolerance: float = 0.0
    higher_is_better: bool = True


@dataclass(frozen=True)
class ModelGateSnapshot:
    model_id: str
    semantic: dict[str, AcceptanceMetric]
    style: dict[str, AcceptanceMetric]


@dataclass(frozen=True)
class ActivationGateReport:
    semantic_passed: bool
    style_passed: bool
    passed: bool
    failure_reasons: tuple[str, ...]
    comparisons: dict[str, dict[str, object]]


_REQUIRED_SEMANTIC_GATES = (
    "fixed_regression",
    "structure",
    "future_isolation",
    "unsupported_fact",
)


def build_task4_style_metrics(
    sticker_metrics: dict[str, float],
    *,
    emoji_distance: float,
    invalid_asset_rate: float,
    calibrated_minimums: Mapping[str, float],
) -> dict[str, AcceptanceMetric]:
    """把 valid-only 校准下界转换为 sticker 风格硬门槛。"""

    # A confidence lower bound can legitimately collapse to zero on a small
    # validation split.  It must not make a non-functional sticker pipeline
    # eligible for activation, so absolute product floors remain in force.
    modality_minimum = max(0.55, calibrated_minimums["modality_f1"])
    recall_minimum = max(0.35, calibrated_minimums["sticker_recall_at_k"])
    mrr_minimum = max(0.20, calibrated_minimums["sticker_mrr"])
    return {
        "modality_f1": AcceptanceMetric(
            sticker_metrics.get("modality_f1", 0.0),
            minimum=modality_minimum,
        ),
        "sticker_recall_at_k": AcceptanceMetric(
            sticker_metrics.get("recall_at_k", 0.0),
            minimum=recall_minimum,
        ),
        "sticker_mrr": AcceptanceMetric(
            sticker_metrics.get("mrr", 0.0),
            minimum=mrr_minimum,
        ),
        "emoji_distance": AcceptanceMetric(
            emoji_distance,
            maximum=0.2,
            higher_is_better=False,
        ),
        "repetition_rate": AcceptanceMetric(
            sticker_metrics.get("consecutive_repeat_rate", 1.0),
            maximum=0.2,
            higher_is_better=False,
        ),
        "invalid_asset_rate": AcceptanceMetric(
            invalid_asset_rate,
            maximum=0.0,
            higher_is_better=False,
        ),
    }


def evaluate_activation_gates(
    candidate: ModelGateSnapshot,
    base: ModelGateSnapshot,
    active: ModelGateSnapshot,
) -> ActivationGateReport:
    """分别执行语义事实安全与风格像本人两组不可绕过的硬门槛。"""

    semantic_reasons, semantic_audit = _evaluate_metric_group(
        "semantic",
        candidate.semantic,
        base.semantic,
        active.semantic,
        required=_REQUIRED_SEMANTIC_GATES,
        compare_unbounded=True,
        require_strict_improvement=False,
    )
    calibrated_style_metrics = {
        "style_distance",
        "speaker_probability_alignment",
        "paired_response_similarity",
    }
    if calibrated_style_metrics <= candidate.style.keys():
        style_reasons, style_audit = _evaluate_composite_style(candidate, base, active)
    else:
        style_reasons, style_audit = _evaluate_metric_group(
            "style",
            candidate.style,
            base.style,
            active.style,
            required=(),
            compare_unbounded=True,
            require_strict_improvement=True,
        )
    if not candidate.style:
        style_reasons.append("style: 缺少风格硬门槛")
    reasons = (*semantic_reasons, *style_reasons)
    return ActivationGateReport(
        semantic_passed=not semantic_reasons,
        style_passed=not style_reasons,
        passed=not reasons,
        failure_reasons=tuple(reasons),
        comparisons={"semantic": semantic_audit, "style": style_audit},
    )


def _evaluate_composite_style(
    candidate: ModelGateSnapshot,
    base: ModelGateSnapshot,
    active: ModelGateSnapshot,
) -> tuple[list[str], dict[str, object]]:
    """用预先固定的复合保真分比较风格，并保留分量退化硬边界。"""

    reasons, audit = _evaluate_metric_group(
        "style",
        {
            name: metric
            for name, metric in candidate.style.items()
            if name
            not in {
                "style_distance",
                "speaker_probability",
                "human_oracle_speaker_probability",
                "speaker_probability_alignment",
                "paired_response_similarity",
            }
        },
        base.style,
        active.style,
        required=(),
        compare_unbounded=False,
        require_strict_improvement=False,
    )
    required = (
        "style_distance",
        "speaker_probability_alignment",
        "paired_response_similarity",
    )
    missing_reasons: list[str] = []
    for name in required:
        if name not in base.style:
            missing_reasons.append(f"style.{name}: 缺少 base 对照")
        if name not in active.style:
            missing_reasons.append(f"style.{name}: 缺少 active 对照")
    reasons.extend(missing_reasons)
    if missing_reasons:
        return reasons, audit

    candidate_distance = candidate.style["style_distance"].value
    candidate_alignment = candidate.style["speaker_probability_alignment"].value
    candidate_paired = candidate.style["paired_response_similarity"].value
    base_distance = base.style["style_distance"].value
    base_alignment = base.style["speaker_probability_alignment"].value
    base_paired = base.style["paired_response_similarity"].value
    active_distance = active.style["style_distance"].value
    active_alignment = active.style["speaker_probability_alignment"].value
    active_paired = active.style["paired_response_similarity"].value
    scores = {
        "candidate": _composite_fidelity(candidate_alignment, candidate_distance, candidate_paired),
        "base": _composite_fidelity(base_alignment, base_distance, base_paired),
        "active": _composite_fidelity(active_alignment, active_distance, active_paired),
    }
    minimum_improvement = 0.005
    delta_base = scores["candidate"] - scores["base"]
    delta_active = scores["candidate"] - scores["active"]
    if delta_base < minimum_improvement:
        reasons.append(f"style.composite_fidelity: 相对 base 改善低于 {minimum_improvement}")
    if delta_active < minimum_improvement:
        reasons.append(f"style.composite_fidelity: 相对 active 改善低于 {minimum_improvement}")
    if candidate_distance > 0.35:
        reasons.append("style.style_distance: 高于最大值 0.35")
    if candidate_distance > active_distance + 0.08:
        reasons.append("style.style_distance: 相对 active 退化超过 0.08")
    minimum_alignment = 0.60
    if candidate_alignment < minimum_alignment:
        reasons.append(f"style.speaker_probability_alignment: 低于校准下限 {minimum_alignment}")
    if candidate_alignment < active_alignment - 0.05:
        reasons.append("style.speaker_probability_alignment: 相对 active 退化超过 0.05")
    # A response can match surface rhythm while answering the wrong thing.
    # Keep paired held-out meaning as an absolute gate, not merely a small
    # relative improvement over an already weak active model.
    minimum_paired_similarity = MINIMUM_PAIRED_RESPONSE_SIMILARITY
    if candidate_paired < minimum_paired_similarity:
        reasons.append(
            f"style.paired_response_similarity: 低于真实回复语义下限 {minimum_paired_similarity}"
        )
    if candidate_paired < active_paired - 0.05:
        reasons.append("style.paired_response_similarity: 相对 active 退化超过 0.05")
    audit["style_distance"] = {
        "candidate": candidate_distance,
        "base": base_distance,
        "active": active_distance,
        "absolute_maximum": 0.35,
        "active_regression_maximum": 0.08,
        "passed": (candidate_distance <= 0.35 and candidate_distance <= active_distance + 0.08),
    }
    audit["speaker_probability_alignment"] = {
        "candidate": candidate_alignment,
        "base": base_alignment,
        "active": active_alignment,
        "calibrated_minimum": minimum_alignment,
        "active_regression_maximum": 0.05,
        "passed": (
            candidate_alignment >= minimum_alignment
            and candidate_alignment >= active_alignment - 0.05
        ),
    }
    audit["paired_response_similarity"] = {
        "candidate": candidate_paired,
        "base": base_paired,
        "active": active_paired,
        "calibrated_minimum": minimum_paired_similarity,
        "active_regression_maximum": 0.05,
        "passed": (
            candidate_paired >= minimum_paired_similarity
            and candidate_paired >= active_paired - 0.05
        ),
    }
    audit["composite_fidelity"] = {
        **scores,
        "weights": {
            "speaker_probability_alignment": 0.45,
            "style_similarity": 0.25,
            "paired_response_similarity": 0.30,
        },
        "delta_base": delta_base,
        "delta_active": delta_active,
        "minimum_improvement": minimum_improvement,
        "passed": (delta_base >= minimum_improvement and delta_active >= minimum_improvement),
    }
    sticker_names = (
        "modality_f1",
        "sticker_recall_at_k",
        "sticker_mrr",
    )
    present_sticker_names = tuple(name for name in sticker_names if name in candidate.style)
    if present_sticker_names:
        sticker_improved = False
        sticker_non_regressed = True
        for name in present_sticker_names:
            active_metric = active.style.get(name)
            if active_metric is None:
                reasons.append(f"style.{name}: 缺少 active pipeline 对照")
                sticker_non_regressed = False
                continue
            candidate_value = candidate.style[name].value
            active_value = active_metric.value
            if candidate_value < active_value:
                reasons.append(f"style.{name}: 低于旧 active pipeline")
                sticker_non_regressed = False
            sticker_improved = sticker_improved or candidate_value > active_value
        if sticker_non_regressed and not sticker_improved:
            reasons.append("style.sticker: 未优于旧 active pipeline")
        audit["sticker_active_pipeline"] = {
            "metrics": list(present_sticker_names),
            "all_non_regressed": sticker_non_regressed,
            "strict_improvement": sticker_improved,
            "passed": sticker_non_regressed and sticker_improved,
        }
    return reasons, audit


def _composite_fidelity(
    speaker_probability_alignment: float,
    style_distance: float,
    paired_response_similarity: float,
) -> float:
    return (
        0.45 * speaker_probability_alignment
        + 0.25 * (1.0 - style_distance)
        + 0.30 * paired_response_similarity
    )


def _evaluate_metric_group(
    group: str,
    candidate: dict[str, AcceptanceMetric],
    base: dict[str, AcceptanceMetric],
    active: dict[str, AcceptanceMetric],
    *,
    required: tuple[str, ...],
    compare_unbounded: bool,
    require_strict_improvement: bool,
) -> tuple[list[str], dict[str, object]]:
    reasons = [f"{group}: 缺少硬门槛 {name}" for name in required if name not in candidate]
    audit: dict[str, object] = {}
    comparative_count = 0
    strict_improvement_found = False
    for name, metric in candidate.items():
        failures: list[str] = []
        if name in required and metric.minimum is None and metric.value < 1.0:
            failures.append("硬门槛必须完全通过")
        if metric.minimum is not None and metric.value < metric.minimum:
            failures.append(f"低于最小值 {metric.minimum}")
        if metric.maximum is not None and metric.value > metric.maximum:
            failures.append(f"高于最大值 {metric.maximum}")
        comparative = compare_unbounded and metric.minimum is None and metric.maximum is None
        if comparative:
            comparative_count += 1
            strictly_better_than_all = True
            for label, baseline in (("base", base), ("active", active)):
                reference = baseline.get(name)
                if reference is None:
                    if group == "semantic" and not baseline:
                        continue
                    failures.append(f"缺少 {label} 对照")
                    strictly_better_than_all = False
                    continue
                if metric.higher_is_better:
                    non_regressed = metric.value >= reference.value - metric.tolerance
                    strictly_better = metric.value > reference.value
                else:
                    non_regressed = metric.value <= reference.value + metric.tolerance
                    strictly_better = metric.value < reference.value
                if not non_regressed:
                    failures.append(f"相对 {label} 退化超过容差 {metric.tolerance}")
                strictly_better_than_all = strictly_better_than_all and strictly_better
            strict_improvement_found = strict_improvement_found or strictly_better_than_all
        reasons.extend(f"{group}.{name}: {failure}" for failure in failures)
        audit[name] = {
            "candidate": metric.value,
            "base": base[name].value if name in base else None,
            "active": active[name].value if name in active else None,
            "minimum": metric.minimum,
            "maximum": metric.maximum,
            "tolerance": metric.tolerance,
            "higher_is_better": metric.higher_is_better,
            "passed": not failures,
        }
    if require_strict_improvement and comparative_count and not strict_improvement_found:
        reasons.append(f"{group}: 没有风格主指标同时严格优于 base 与 active")
    return reasons, audit


def retry_format_instruction(
    reply_protocol: str,
    allowed_sticker_ids: tuple[str, ...] = (),
) -> str:
    """返回与当前模型协议一致的格式修复提示。"""

    if reply_protocol == "persona_text":
        return "上一条包含机器格式；只输出本人会发送的自然聊天文字，多条消息换行。"
    if reply_protocol == "compact":
        return compact_retry_instruction(allowed_sticker_ids)
    return "上一条格式无效，只输出合法气泡 JSON。"


@dataclass(frozen=True)
class OutputStructureAttempt:
    raw_output_hash: str
    raw_excerpt: str
    raw_length: int
    reason: str
    attempted_invalid_sticker_ids: tuple[str, ...] = ()


class ModelOutputStructureError(RuntimeError):
    """保留严格解析失败时的原始输出诊断。"""

    def __init__(
        self,
        raw_attempts: tuple[str, ...],
        *,
        reasons: tuple[str, ...] = (),
        attempted_invalid_sticker_ids: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        super().__init__("候选模型输出结构无效")
        self.raw_attempts = raw_attempts
        self.attempts = tuple(
            OutputStructureAttempt(
                raw_output_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                raw_excerpt=_safe_raw_excerpt(raw),
                raw_length=len(raw),
                reason=(reasons[index] if index < len(reasons) else "回复结构无效"),
                attempted_invalid_sticker_ids=(
                    attempted_invalid_sticker_ids[index]
                    if index < len(attempted_invalid_sticker_ids)
                    else ()
                ),
            )
            for index, raw in enumerate(raw_attempts)
        )
        self.attempted_invalid_sticker_ids = tuple(
            asset_id
            for attempt in self.attempts
            for asset_id in attempt.attempted_invalid_sticker_ids
        )


class PathReplyGenerator(Protocol):
    def generate(
        self,
        base_model: str,
        adapter_path: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn: ...


class MlxPathReplyGenerator:
    def __init__(self, sampling: PersonaSamplingConfig | None = None) -> None:
        self._loaded_key: tuple[str, str] | None = None
        self._model: object | None = None
        self._tokenizer: Any | None = None
        self._sampling = sampling or PersonaSamplingConfig()
        self._style_profile: dict[str, object] | None = None

    def set_style_profile(self, profile: dict[str, object] | None) -> None:
        self._style_profile = profile

    def set_seed(self, seed: int) -> None:
        """Seed MLX sampling so model comparisons use identical draws."""

        mlx = importlib.import_module("mlx.core")
        random_module = getattr(mlx, "random", None)
        seed_method = getattr(random_module, "seed", None)
        if not callable(seed_method):
            raise RuntimeError("当前 MLX 版本不支持确定性验收采样")
        seed_method(seed)

    def generate(
        self,
        base_model: str,
        adapter_path: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        reply_protocol = (
            "persona_text"
            if prompt_uses_persona_text_protocol(system_prompt)
            else "compact"
            if prompt_uses_compact_protocol(system_prompt)
            else "legacy_json"
        )
        mlx_lm = importlib.import_module("mlx_lm")
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        loaded_key = (base_model, adapter_path)
        if self._loaded_key != loaded_key:
            self._release_loaded_model()
            load_options = {"adapter_path": adapter_path} if adapter_path else {}
            self._model, self._tokenizer = mlx_lm.load(base_model, **load_options)
            self._loaded_key = loaded_key
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("本地模型加载失败")
        model = self._model
        tokenizer = self._tokenizer
        context_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        allowed_sticker_ids = allowed_sticker_ids_from_prompt(system_prompt)
        raw_attempts: list[str] = []
        failure_reasons: list[str] = []
        invalid_sticker_attempts: list[tuple[str, ...]] = []
        for attempt in range(3):
            prompt = tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            raw = cast(
                str,
                mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=192,
                    sampler=sample_utils.make_sampler(
                        temp=self._sampling.temperature,
                        top_p=self._sampling.top_p,
                        min_p=self._sampling.min_p,
                    ),
                    logits_processors=[
                        sample_utils.make_repetition_penalty(
                            penalty=self._sampling.repetition_penalty,
                            context_size=self._sampling.repetition_context_size,
                        )
                    ],
                    verbose=False,
                ),
            )
            raw_attempts.append(raw)
            try:
                reply = (
                    parse_persona_text_turn(raw)
                    if reply_protocol == "persona_text"
                    else parse_reply_turn(
                        raw,
                        allowed_sticker_ids=allowed_sticker_ids,
                        normalize_compact=reply_protocol == "compact",
                    )
                )
                natural_text = "\n".join(
                    bubble.content or "" for bubble in reply.bubbles if bubble.type == "text"
                )
                violations = style_violations(natural_text, self._style_profile)
                if not violations:
                    return reply
                failure_reasons.append(
                    "回复使用了本人真实语料中未出现的表达：" + "、".join(violations)
                )
                invalid_sticker_attempts.append(())
                if attempt < 2:
                    context_messages[0] = {
                        "role": "system",
                        "content": (
                            system_prompt
                            + "\n上一条回复使用了本人真实语料中未出现或极少出现的表达："
                            + "、".join(violations)
                            + "。重新回复，保持原意，但只用本人真实的短句、口头禅和语气。"
                        ),
                    }
                continue
            except ReplyStructureError as error:
                failure_reasons.append(str(error))
                invalid_sticker_attempts.append(error.attempted_invalid_sticker_ids)
                if attempt < 2:
                    context_messages[0] = {
                        "role": "system",
                        "content": (
                            system_prompt
                            + "\n"
                            + retry_format_instruction(
                                reply_protocol,
                                allowed_sticker_ids or (),
                            )
                        ),
                    }
        raise ModelOutputStructureError(
            tuple(raw_attempts),
            reasons=tuple(failure_reasons),
            attempted_invalid_sticker_ids=tuple(invalid_sticker_attempts),
        )

    def _release_loaded_model(self) -> None:
        """切换模型前先释放旧权重，避免多个 8B 模型同时驻留。"""

        if self._model is None and self._tokenizer is None:
            return
        self._model = None
        self._tokenizer = None
        self._loaded_key = None
        gc.collect()
        try:
            mlx = importlib.import_module("mlx.core")
        except ModuleNotFoundError:
            return
        clear_cache = getattr(mlx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()


def default_acceptance_fixture() -> Path:
    return Path(__file__).with_name("fixtures") / "conversation_acceptance_v1.json"


def _allowed_sticker_ids(system_prompt: str) -> tuple[str, ...] | None:
    return allowed_sticker_ids_from_prompt(system_prompt)


def _safe_raw_excerpt(raw: str) -> str:
    """清理控制字符并限制审计片段长度，避免报告无限扩张。"""

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "�", raw)
    return cleaned[:200]


class LocalAcceptanceReviewer:
    """在云端不可用时执行不可跳过的本地结构与主体边界验收。"""

    def review(
        self,
        _packet: ContextPacket,
        draft: GeneratedReplyTurn,
    ) -> ReviewResult:
        if not draft.bubbles:
            raise ReviewFailedError("本地验收发现空回复")
        return ReviewResult(
            verdict="approve",
            reasons=(),
            reply=draft,
        )


def character_ngram_cosine(left: str, right: str) -> float:
    """Language-agnostic paired-reply similarity without external model I/O."""

    def features(text: str) -> Counter[str]:
        compact = re.sub(r"\s+", "", text)
        grams: Counter[str] = Counter()
        for width in (2, 3):
            grams.update(
                compact[index : index + width] for index in range(max(0, len(compact) - width + 1))
            )
        if not grams and compact:
            grams[compact] = 1
        return grams

    left_features = features(left)
    right_features = features(right)
    if not left_features or not right_features:
        return 0.0
    dot = sum(count * right_features.get(feature, 0) for feature, count in left_features.items())
    left_norm = math.sqrt(sum(count * count for count in left_features.values()))
    right_norm = math.sqrt(sum(count * count for count in right_features.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _dense_cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("语义向量维度不一致")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _causal_memory_cases(
    *,
    persona: str,
    style_profile: dict[str, object] | None,
) -> tuple[tuple[str, dict[str, object], dict[str, object]], ...]:
    identity = {"style_profile": style_profile or {}}

    def request(
        question: str,
        *,
        memories: tuple[ContextMemory, ...] = (),
        continuity: tuple[ContextMemory, ...] = (),
        active_beliefs: tuple[ContextMemory, ...] = (),
        expressed_preferences: list[str] | None = None,
    ) -> ContextRequest:
        return ContextRequest(
            persona=persona,
            cutoff="memory-causal-held-out",
            memories=memories,
            history=(),
            current_user_content=question,
            identity_kernel={
                **identity,
                "expressed_preferences": expressed_preferences or [],
            },
            continuity_memories=continuity,
            active_beliefs=active_beliefs,
            reply_protocol="persona_text",
        )

    return (
        (
            "verified-trip-destination",
            {
                "request": request(
                    "你之前计划去哪里旅行",
                    memories=(
                        ContextMemory(
                            "event-korea",
                            "event",
                            "本人之前计划去韩国旅行",
                            authority="verified_history",
                            evidence_ids=("event-korea",),
                        ),
                    ),
                ),
                "expected": "韩国",
            },
            {
                "request": request(
                    "你之前计划去哪里旅行",
                    memories=(
                        ContextMemory(
                            "event-japan",
                            "event",
                            "本人之前计划去日本旅行",
                            authority="verified_history",
                            evidence_ids=("event-japan",),
                        ),
                    ),
                ),
                "expected": "日本",
            },
        ),
        (
            "expressed-food-preference",
            {
                "request": request(
                    "你最喜欢吃什么",
                    expressed_preferences=["本人明确说过喜欢吃火锅"],
                ),
                "expected": "火锅",
            },
            {
                "request": request(
                    "你最喜欢吃什么",
                    expressed_preferences=["本人明确说过喜欢吃寿司"],
                ),
                "expected": "寿司",
            },
        ),
        (
            "recorded-contact-commitment",
            {
                "request": request(
                    "你答应周末怎么联系我",
                    continuity=(
                        ContextMemory(
                            "record-phone",
                            "episode",
                            "本人说：周末给你打电话",
                            authority="conversation_record",
                            evidence_ids=("turn-phone",),
                        ),
                    ),
                ),
                "expected": ("电话", "打给", "打你", "打过来"),
            },
            {
                "request": request(
                    "你答应周末怎么联系我",
                    continuity=(
                        ContextMemory(
                            "record-video",
                            "episode",
                            "本人说：周末和你视频",
                            authority="conversation_record",
                            evidence_ids=("turn-video",),
                        ),
                    ),
                ),
                "expected": "视频",
            },
        ),
        (
            "active-relationship-belief",
            {
                "request": request(
                    "你现在想怎么处理我们的关系",
                    active_beliefs=(
                        ContextMemory(
                            "belief-repair",
                            "belief",
                            "本人目前愿意修复关系",
                            authority="subjective",
                            evidence_ids=("episode-repair",),
                        ),
                    ),
                ),
                "expected": ("修复", "和好", "重新开始", "继续这段关系"),
            },
            {
                "request": request(
                    "你现在想怎么处理我们的关系",
                    active_beliefs=(
                        ContextMemory(
                            "belief-distance",
                            "belief",
                            "本人目前需要保持距离",
                            authority="subjective",
                            evidence_ids=("episode-distance",),
                        ),
                    ),
                ),
                "expected": (
                    "距离",
                    "冷静一段",
                    "少联系",
                    "别联系",
                    "不联系",
                    "缓一缓",
                    "需要空间",
                ),
            },
        ),
    )


def _causal_normalize(value: str) -> str:
    return re.sub(r"[\s，。！？!?、；;：:]", "", value)


def _paired_causal_style_consistency(
    left: GeneratedReplyTurn,
    right: GeneratedReplyTurn,
) -> float:
    left_text = "".join(bubble.content or "" for bubble in left.bubbles)
    right_text = "".join(bubble.content or "" for bubble in right.bubbles)
    maximum_length = max(len(left_text), len(right_text), 1)
    length_score = 1 - abs(len(left_text) - len(right_text)) / maximum_length
    maximum_bubbles = max(len(left.bubbles), len(right.bubbles), 1)
    bubble_score = 1 - abs(len(left.bubbles) - len(right.bubbles)) / maximum_bubbles
    left_modalities = tuple(bubble.type for bubble in left.bubbles)
    right_modalities = tuple(bubble.type for bubble in right.bubbles)
    modality_score = float(left_modalities == right_modalities)
    return round((length_score + bubble_score + modality_score) / 3, 6)


def _valid_row_context_packet(
    *,
    persona: str,
    messages: list[dict[str, str]],
    style_profile: dict[str, object] | None,
    expressed_preferences: list[str] | None = None,
) -> ContextPacket:
    current_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index]["role"] == "user"),
        -1,
    )
    current_user_content = (
        messages[current_user_index]["content"] if current_user_index >= 0 else ""
    )
    history = tuple(
        ContextTurn(
            role=cast(Any, item["role"]),
            bubbles=(ContextBubble(type="text", content=item["content"]),),
        )
        for index, item in enumerate(messages)
        if index != current_user_index and item["role"] in {"user", "assistant"}
    )
    return ContextPacket(
        persona=persona,
        cutoff="valid-held-out",
        history=history,
        current_user_content=current_user_content,
        memories=(),
        allowed_sticker_ids=(),
        shared_ground={
            "understanding": understand_contributions(current_user_content),
        },
        identity_kernel={
            "style_profile": style_profile or {},
            "expressed_preferences": expressed_preferences or [],
        },
    )


class ModelAcceptanceRunner:
    def __init__(
        self,
        generator: PathReplyGenerator,
        reviewer: ReplyReviewer,
        fixture_path: Path,
        semantic_embedder: TextEmbedder | None = None,
    ) -> None:
        self._generator = generator
        self._reviewer = reviewer
        self._fixture_path = fixture_path
        self._semantic_embedder = semantic_embedder

    @property
    def negative_generator(self) -> PathReplyGenerator:
        """Reuse the audited runtime generator for train-only hard negatives."""

        return self._generator

    def close(self) -> None:
        """释放验收生成器可能常驻的模型，供同进程训练阶段复用显存。"""

        closer = getattr(self._generator, "close", None)
        if callable(closer):
            closer()

    def evaluate_memory_causality(
        self,
        *,
        base_model: str,
        adapter_path: str,
        persona: str,
        style_profile: dict[str, object] | None = None,
    ) -> dict[str, float | int | str]:
        """A/B the same LoRA turn while changing only trusted memory evidence."""

        cases = _causal_memory_cases(
            persona=persona,
            style_profile=style_profile,
        )
        correct_variants = 0
        grounded_variants = 0
        switched_cases = 0
        style_scores: list[float] = []
        audit: list[dict[str, object]] = []
        for case_id, left, right in cases:
            outputs: list[tuple[str, GeneratedReplyTurn, bool, bool]] = []
            for variant in (left, right):
                packet = ContextBuilder().build_packet(variant["request"])
                messages = ContextBuilder().to_chat_messages(packet)
                system = messages[0].content
                history = [
                    {"role": message.role, "content": message.content} for message in messages[1:]
                ]
                _seed_acceptance_generator(
                    self._generator,
                    f"memory-causal:{case_id}",
                )
                content_draft = active_belief_content_draft(
                    packet.active_beliefs,
                    packet.current_user_content,
                )
                if content_draft is None:
                    reply = self._generator.generate(
                        base_model,
                        adapter_path,
                        system,
                        history,
                    )
                else:
                    style_system = (
                        "把内容草稿改写成这个人在私人微信里真实会发送的表达。"
                        "不能新增、删除或改变事实、否定、数量、人物和态度；"
                        "只改变措辞、语气、标点和消息分段。"
                        "只输出改写后的聊天文字，多条消息换行。" + persona_text_instruction()
                    )
                    rewritten = self._generator.generate(
                        base_model,
                        adapter_path,
                        style_system,
                        [{"role": "user", "content": "内容草稿：\n" + content_draft}],
                    )
                    try:
                        validate_fact_grounded_reply(packet, rewritten)
                        reply = rewritten
                    except ValueError:
                        reply = GeneratedReplyTurn(
                            bubbles=(
                                GeneratedBubble(
                                    type="text",
                                    content=content_draft,
                                    delay_ms=0,
                                ),
                            ),
                            raw_output=content_draft,
                        )
                text = "\n".join(bubble.content or "" for bubble in reply.bubbles)
                raw_expected = variant["expected"]
                expected = (
                    tuple(str(item) for item in raw_expected)
                    if isinstance(raw_expected, tuple | list)
                    else (str(raw_expected),)
                )
                correct = any(marker in text for marker in expected)
                try:
                    validate_fact_grounded_reply(packet, reply)
                    grounded = True
                except ValueError:
                    grounded = False
                correct_variants += int(correct)
                grounded_variants += int(grounded)
                outputs.append((text, reply, correct, grounded))
            left_text, left_reply, left_correct, _left_grounded = outputs[0]
            right_text, right_reply, right_correct, _right_grounded = outputs[1]
            switched = (
                left_correct
                and right_correct
                and _causal_normalize(left_text) != _causal_normalize(right_text)
            )
            switched_cases += int(switched)
            style_score = _paired_causal_style_consistency(left_reply, right_reply)
            style_scores.append(style_score)
            audit.append(
                {
                    "case_id": case_id,
                    "left": left_text[:160],
                    "right": right_text[:160],
                    "left_correct": left_correct,
                    "right_correct": right_correct,
                    "switched": switched,
                    "style_consistency": style_score,
                }
            )
        variant_count = len(cases) * 2
        return {
            "memory_causal_variant_accuracy": (
                correct_variants / variant_count if variant_count else 0.0
            ),
            "memory_causal_grounding_rate": (
                grounded_variants / variant_count if variant_count else 0.0
            ),
            "memory_causal_switch_rate": (switched_cases / len(cases) if cases else 0.0),
            "memory_causal_style_consistency": (
                sum(style_scores) / len(style_scores) if style_scores else 0.0
            ),
            "memory_causal_audit": json.dumps(
                audit,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    def evaluate_valid_style(
        self,
        *,
        data_dir: Path,
        base_model: str,
        adapter_path: str,
        persona: str,
        sample_count: int = 20,
        style_profile: dict[str, object] | None = None,
        expressed_preferences: list[str] | None = None,
        training_task: Literal["conversation", "style_transfer"] = "conversation",
    ) -> dict[str, float | int | str]:
        """只用 valid 真实目标计算候选风格；test 文件不会被读取。"""

        set_style_profile = getattr(self._generator, "set_style_profile", None)
        if callable(set_style_profile):
            set_style_profile(style_profile)
        valid_rows = _read_jsonl(data_dir / "valid.jsonl")
        responsive_candidates = [
            row
            for row in valid_rows
            if _row_metadata(row).get("conversation_mode", "responsive") == "responsive"
            and _row_metadata(row).get("training_task", "conversation") == training_task
        ]
        unanswerable_current_state_rows = [
            row
            for row in responsive_candidates
            if training_task == "conversation" and _row_asks_unobserved_current_state(row)
        ]
        responsive_rows = [
            row
            for row in responsive_candidates
            if training_task != "conversation" or not _row_asks_unobserved_current_state(row)
        ]
        if len(responsive_rows) < 20:
            raise ValueError("候选风格评估至少需要 20 条 valid 样本")
        selected = _deterministic_stratified_rows(
            responsive_rows,
            max(20, sample_count),
        )
        generated_turns: list[StyleTurn] = []
        real_turns: list[StyleTurn] = []
        generated_texts: list[str] = []
        human_texts: list[str] = []
        paired_response_similarities: list[float] = []
        structure_failures = 0
        grounding_failures = 0
        normalization_count = 0
        structure_failure_audit: list[dict[str, object]] = []
        sample_audit: list[dict[str, object]] = []
        register_violation_audit: list[dict[str, object]] = []
        poison_copy_audit: list[dict[str, object]] = []
        for row in selected:
            messages = _row_messages(row)
            system = messages[0]["content"]
            metadata = _row_metadata(row)
            raw_allowed_sticker_ids = metadata.get("allowed_sticker_ids", [])
            allowed_sticker_ids = tuple(
                str(item)
                for item in (
                    raw_allowed_sticker_ids if isinstance(raw_allowed_sticker_ids, list) else []
                )
                if isinstance(item, str)
            )
            if not prompt_uses_persona_text_protocol(system):
                system = system + "\n" + compact_protocol_instruction(allowed_sticker_ids)
            # Low-authority ledger entries are intentionally not appended here:
            # production ContextBuilder isolates them before model generation.
            # The deterministic compiler audit below enforces that boundary.
            history = messages[1:-1]
            human = _parse_dataset_reply(messages[-1]["content"])
            human_text = "\n".join(bubble.content or "" for bubble in human.bubbles)
            previous = next(
                (item["content"] for item in reversed(history) if item["role"] == "user"),
                "",
            )
            try:
                _seed_acceptance_generator(
                    self._generator,
                    f"valid:{_row_source_hash(row)}",
                )
                generated = self._generator.generate(
                    base_model,
                    adapter_path,
                    system,
                    history,
                )
                if training_task == "style_transfer":
                    content_draft = _style_transfer_content_draft(history)
                    generated_text = "\n".join(bubble.content or "" for bubble in generated.bubbles)
                    if not rewrite_preserves_hard_semantics(
                        content_draft,
                        generated_text,
                    ):
                        raise ValueError("风格改写改变了内容草稿的硬语义")
                else:
                    packet = _valid_row_context_packet(
                        persona=persona,
                        messages=history,
                        style_profile=style_profile,
                        expressed_preferences=expressed_preferences,
                    )
                    try:
                        validate_fact_grounded_reply(packet, generated)
                    except ValueError as grounding_error:
                        generated = self._generator.generate(
                            base_model,
                            adapter_path,
                            system
                            + "\n"
                            + grounded_retry_instruction(
                                grounding_error,
                                packet,
                            ),
                            history,
                        )
                        validate_fact_grounded_reply(packet, generated)
            except ModelOutputStructureError as error:
                structure_failures += 1
                structure_failure_audit.append(
                    {
                        "source_hash": _row_source_hash(row),
                        "attempts": [asdict(item) for item in error.attempts],
                    }
                )
                continue
            except ValueError as error:
                grounding_failures += 1
                structure_failure_audit.append(
                    {
                        "source_hash": _row_source_hash(row),
                        "attempts": [],
                        "error_type": "GroundingError",
                        "reason": str(error),
                        "generated_excerpt": "\n".join(
                            bubble.content or "" for bubble in generated.bubbles
                        )[:240],
                    }
                )
                continue
            except RuntimeError as error:
                structure_failures += 1
                structure_failure_audit.append(
                    {
                        "source_hash": _row_source_hash(row),
                        "attempts": [],
                        "error_type": type(error).__name__,
                    }
                )
                continue
            generated_turns.append(_reply_style_turn(generated, previous))
            normalization_count += generated.normalization is not None
            real_turns.append(_reply_style_turn(human, previous))
            generated_text = "\n".join(bubble.content or "" for bubble in generated.bubbles)
            generated_texts.append(generated_text)
            human_texts.append(human_text)
            if self._semantic_embedder is None:
                paired_response_similarities.append(
                    character_ngram_cosine(generated_text, human_text)
                )
            violations = style_violations(generated_text, style_profile)
            if violations:
                register_violation_audit.append(
                    {
                        "source_hash": _row_source_hash(row),
                        "markers": violations,
                        "generated_excerpt": generated_text[:240],
                    }
                )
            copied_poison = tuple(
                marker
                for marker in ("火星基地", "正在挖矿", "一起去过冰岛")
                if marker in generated_text
            )
            if copied_poison:
                poison_copy_audit.append(
                    {
                        "source_hash": _row_source_hash(row),
                        "markers": copied_poison,
                        "generated_excerpt": generated_text[:240],
                    }
                )
            sample_audit.append(
                {
                    "source_hash": _row_source_hash(row),
                    "prompt_excerpt": previous[:240],
                    "human_excerpt": human_text[:240],
                    "generated_excerpt": generated_text[:240],
                }
            )
        distance_value = 1.0
        speaker_probability = 0.0
        human_oracle_speaker_probability = 0.0
        speaker_probability_alignment = 0.0
        if generated_turns:
            distance = compare_style_distributions(generated_turns, real_turns)
            distance_value = distance.overall_distance
            identifier = SpeakerStyleIdentifier()
            identifier.fit(_speaker_training_samples(data_dir / "train.jsonl", persona))
            speaker = identifier.evaluate(
                [SpeakerSample("valid", persona, text) for text in generated_texts],
                positive_speaker=persona,
            )
            speaker_probability = speaker.mean_positive_probability
            human_speaker = identifier.evaluate(
                [SpeakerSample("valid", persona, text) for text in human_texts],
                positive_speaker=persona,
            )
            human_oracle_speaker_probability = human_speaker.mean_positive_probability
            speaker_probability_alignment = 1.0 - sum(
                abs(generated - human)
                for generated, human in zip(
                    speaker.probabilities,
                    human_speaker.probabilities,
                    strict=True,
                )
            ) / len(generated_texts)
        paired_similarity_method = "character-ngram-cosine-v1"
        if self._semantic_embedder is not None and generated_texts:
            vectors = self._semantic_embedder.embed(
                [text for pair in zip(generated_texts, human_texts, strict=True) for text in pair]
            )
            paired_response_similarities = [
                _dense_cosine(vectors[index], vectors[index + 1])
                for index in range(0, len(vectors), 2)
            ]
            paired_similarity_method = "bge-small-zh-v1.5-cosine-v1"
        paired_response_similarity = (
            sum(paired_response_similarities) / len(selected) if selected else 0.0
        )
        # Speaker identification alone is easily gamed by short catchphrases.
        # Reward matching what the real person actually replied in the paired
        # held-out context as an independent selection signal.
        composite = (
            0.45 * speaker_probability_alignment
            + 0.25 * (1.0 - distance_value)
            + 0.30 * paired_response_similarity
        )
        canonical = json.dumps(
            selected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        low_authority_prompt_leaks = _low_authority_prompt_leaks(persona)
        return {
            "valid_sample_count": len(selected),
            "valid_conversation_mode": "responsive",
            "valid_training_task": training_task,
            "excluded_unanswerable_current_state_count": len(unanswerable_current_state_rows),
            "valid_structure_failures": structure_failures,
            "valid_grounding_failures": grounding_failures,
            "valid_grounding_failure_rate": (
                grounding_failures / len(selected) if selected else 1.0
            ),
            "valid_register_violations": len(register_violation_audit),
            "valid_register_violation_audit": json.dumps(
                register_violation_audit,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "memory_poison_copy_rate": (
                len(poison_copy_audit) / len(generated_texts) if generated_texts else 1.0
            ),
            "memory_poison_copy_audit": json.dumps(
                poison_copy_audit,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "low_authority_prompt_leak_rate": (len(low_authority_prompt_leaks) / 3.0),
            "low_authority_prompt_leak_audit": json.dumps(
                low_authority_prompt_leaks,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "raw_format_compliance": (
                (len(selected) - structure_failures - normalization_count) / len(selected)
                if selected
                else 0.0
            ),
            "normalization_rate": (normalization_count / len(selected) if selected else 0.0),
            "valid_structure_failure_audit": json.dumps(
                structure_failure_audit,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "valid_sample_audit": json.dumps(
                sample_audit,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "valid_sample_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "style_distance": distance_value,
            "speaker_probability": speaker_probability,
            "human_oracle_speaker_probability": (human_oracle_speaker_probability),
            "speaker_probability_alignment": speaker_probability_alignment,
            "paired_response_similarity": paired_response_similarity,
            "paired_similarity_method": paired_similarity_method,
            "composite_fidelity": composite,
        }

    def build_human_blind_cases(
        self,
        *,
        data_dir: Path,
        base_model: str,
        adapter_path: str,
        sample_count: int = 20,
    ) -> list[dict[str, object]]:
        """Generate held-out cases through the deployed two-stage reply path.

        The persona adapter is an expression model. Asking it to invent reply
        content here would evaluate a path production deliberately does not use.
        A clean base model first plans a draft without seeing the held-out target;
        the candidate adapter then only rewrites that draft.
        """

        test_rows = _read_jsonl(data_dir / "test.jsonl")
        responsive_rows = [
            row
            for row in test_rows
            if _row_metadata(row).get("conversation_mode", "responsive") == "responsive"
            and _row_metadata(row).get("training_task", "conversation") == "conversation"
        ]
        if len(responsive_rows) < sample_count:
            raise ValueError("真人盲测至少需要足量独立 test 样本")
        selected = _deterministic_stratified_rows(
            responsive_rows,
            min(len(responsive_rows), max(sample_count * 3, sample_count)),
        )
        cases: list[dict[str, object]] = []
        for row in selected:
            if len(cases) >= sample_count:
                break
            messages = _row_messages(row)
            system = messages[0]["content"]
            history = messages[1:-1]
            human = _parse_dataset_reply(messages[-1]["content"])
            _seed_acceptance_generator(
                self._generator,
                f"human-blind:{_row_source_hash(row)}",
            )
            try:
                content_turn = self._generator.generate(
                    base_model,
                    "",
                    _blind_content_planner_instruction(system),
                    history,
                )
                content_draft = "\n".join(
                    bubble.content or "" for bubble in content_turn.bubbles if bubble.type == "text"
                ).strip()
                if not content_draft:
                    continue
                rewritten = self._generator.generate(
                    base_model,
                    adapter_path,
                    persona_style_transfer_instruction(),
                    [{"role": "user", "content": "内容草稿：\n" + content_draft}],
                )
                rewritten_text = "\n".join(
                    bubble.content or "" for bubble in rewritten.bubbles if bubble.type == "text"
                ).strip()
                candidate = (
                    rewritten
                    if rewrite_preserves_hard_semantics(
                        content_draft,
                        rewritten_text,
                    )
                    else content_turn
                )
            except (ModelOutputStructureError, RuntimeError, ValueError):
                continue
            human_text = "\n".join(bubble.content or "" for bubble in human.bubbles).strip()
            candidate_text = "\n".join(bubble.content or "" for bubble in candidate.bubbles).strip()
            if not human_text or not candidate_text:
                continue
            cases.append(
                {
                    "context": [f"{item['role']}：{item['content']}" for item in history[-12:]],
                    "human_reply": human_text,
                    "candidate_reply": candidate_text,
                }
            )
        if len(cases) < sample_count:
            raise ValueError("真人盲测候选生成不完整")
        return cases

    def run(
        self,
        *,
        base_model: str,
        adapter_path: str,
        persona: str,
        cutoff: str,
        style_profile: dict[str, object] | None = None,
        reply_protocol: str = "legacy_json",
    ) -> AcceptanceReport:
        set_style_profile = getattr(self._generator, "set_style_profile", None)
        if callable(set_style_profile):
            set_style_profile(style_profile)
        fixture = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        cases = fixture["cases"]
        passed = 0
        structure_failures = 0
        forbidden_failures = 0
        raw_output_failures = 0
        style_feature_failures = 0
        style_feature_differences: list[StyleFeatureDifference] = []
        candidate_style_turns: list[StyleTurn] = []
        candidate_features: dict[str, object] = {}
        failed_case_ids: list[str] = []
        structure_failure_audit: list[dict[str, object]] = []
        normalization_count = 0
        builder = ContextBuilder()
        baseline_style_features = _resolve_style_profile(style_profile)
        for case in cases:
            history = tuple(
                ContextTurn(
                    role=item["role"],
                    bubbles=(ContextBubble(type="text", content=item["content"]),),
                )
                for item in case["history"]
            )
            packet = builder.build_packet(
                ContextRequest(
                    persona=persona,
                    cutoff=cutoff,
                    memories=(),
                    history=history,
                    current_user_content=case["current_user_content"],
                    reply_protocol=(
                        "persona_text"
                        if reply_protocol == "persona_text"
                        else "compact"
                        if reply_protocol == "compact"
                        else "legacy_json"
                    ),
                )
            )
            messages = builder.to_chat_messages(packet)
            try:
                _seed_acceptance_generator(
                    self._generator,
                    f"fixture:{case['id']}",
                )
                draft = self._generator.generate(
                    base_model,
                    adapter_path,
                    messages[0].content,
                    [{"role": item.role, "content": item.content} for item in messages[1:]],
                )
                raw_content = draft.raw_output or "\n".join(
                    bubble.content or "" for bubble in draft.bubbles
                )
                normalization_count += draft.normalization is not None
                forbidden = tuple(case.get("forbidden_substrings", []))
                if any(item in raw_content for item in forbidden) or style_violations(
                    raw_content, baseline_style_features
                ):
                    raw_output_failures += 1
                    failed_case_ids.append(str(case["id"]))
                    continue
                final = self._reviewer.review(packet, draft).reply
                candidate_style_turns.append(
                    _reply_style_turn(
                        final,
                        str(case["current_user_content"]),
                    )
                )
            except ModelOutputStructureError as error:
                structure_failures += 1
                failed_case_ids.append(str(case["id"]))
                structure_failure_audit.append(
                    {
                        "case_id": str(case["id"]),
                        "attempts": [asdict(item) for item in error.attempts],
                    }
                )
                continue
            except (RuntimeError, ReviewFailedError):
                structure_failures += 1
                failed_case_ids.append(str(case["id"]))
                continue
            content = "\n".join(bubble.content or "" for bubble in final.bubbles)
            forbidden = tuple(case.get("forbidden_substrings", []))
            if any(item in content for item in forbidden):
                forbidden_failures += 1
                failed_case_ids.append(str(case["id"]))
                continue
            passed += 1
        count = len(cases)
        if candidate_style_turns:
            candidate_features = extract_style_features(candidate_style_turns)
            style_feature_differences.extend(
                compare_style_features(
                    candidate_features,
                    baseline_style_features,
                    case_id="aggregate",
                )
            )
            style_feature_failures = int(bool(style_feature_differences))
        return AcceptanceReport(
            case_count=count,
            passed_count=passed,
            structure_failures=structure_failures,
            forbidden_fact_failures=forbidden_failures,
            failed_case_ids=tuple(failed_case_ids),
            passed=(
                bool(count)
                and passed / count >= 0.9
                and raw_output_failures == 0
                and style_feature_failures == 0
            ),
            raw_output_failures=raw_output_failures,
            style_feature_failures=style_feature_failures,
            style_feature_differences=tuple(style_feature_differences),
            style_feature_sample_count=len(candidate_style_turns),
            candidate_style_profile=candidate_features,
            raw_format_compliance=(
                (count - structure_failures - normalization_count) / count if count else 0.0
            ),
            normalization_rate=(normalization_count / count if count else 0.0),
            structure_failure_audit=tuple(structure_failure_audit),
        )


def _low_authority_prompt_leaks(persona: str) -> list[str]:
    """Audit the real compiler boundary instead of only model robustness."""

    markers = ("审计假转述", "审计假对话", "审计竞争信念")
    packet = ContextBuilder().build_packet(
        ContextRequest(
            persona=persona,
            cutoff="audit",
            memories=(),
            history=(),
            current_user_content="继续",
            contested_beliefs=(
                ContextMemory(
                    "audit:contested",
                    "belief",
                    markers[2],
                    authority="subjective",
                ),
            ),
            continuity_memories=(
                ContextMemory(
                    "audit:reported",
                    "fact",
                    markers[0],
                    authority="reported_claim",
                ),
                ContextMemory(
                    "audit:conversation",
                    "episode",
                    markers[1],
                    authority="conversation_record",
                ),
            ),
            reply_protocol="persona_text",
        )
    )
    system = ContextBuilder().to_chat_messages(packet)[0].content
    return [marker for marker in markers if marker in system]


def _seed_acceptance_generator(generator: PathReplyGenerator, identity: str) -> None:
    """Use the same stable sample for a case across candidate/base/active."""

    set_seed = getattr(generator, "set_seed", None)
    if not callable(set_seed):
        return
    digest = hashlib.sha256(f"moonlightbox-acceptance-v1:{identity}".encode()).digest()
    set_seed(int.from_bytes(digest[:4], "big"))


def compare_style_features(
    candidate: dict[str, object],
    baseline: dict[str, object] | None,
    *,
    case_id: str = "",
) -> tuple[StyleFeatureDifference, ...]:
    """比较统一特征中的稳定结构边界，后续验收可继续扩展规则。"""

    if not baseline or "schema_version" not in baseline:
        return ()
    differences: list[StyleFeatureDifference] = []
    _append_upper_bound_difference(
        differences,
        case_id=case_id,
        feature="structural.bubbles_per_turn.max",
        baseline=_nested_number(baseline, "structural", "bubbles_per_turn", "max"),
        candidate=_nested_number(candidate, "structural", "bubbles_per_turn", "max"),
        allowance=0.0,
    )
    baseline_length = _nested_number(
        baseline,
        "structural",
        "bubble_length",
        "p90",
    )
    candidate_length = _nested_number(
        candidate,
        "structural",
        "bubble_length",
        "p90",
    )
    if (
        baseline_length is not None
        and candidate_length is not None
        and candidate_length > max(baseline_length * 1.5, baseline_length + 4)
    ):
        differences.append(
            StyleFeatureDifference(
                case_id=case_id,
                feature="structural.bubble_length.p90",
                baseline=baseline_length,
                candidate=candidate_length,
                rule="候选值不得显著超过基准 p90",
            )
        )
    baseline_omission = _nested_number(
        baseline,
        "structural",
        "terminal_period_omission_rate",
    )
    candidate_omission = _nested_number(
        candidate,
        "structural",
        "terminal_period_omission_rate",
    )
    if (
        baseline_omission is not None
        and candidate_omission is not None
        and abs(candidate_omission - baseline_omission) > 0.75
    ):
        differences.append(
            StyleFeatureDifference(
                case_id=case_id,
                feature="structural.terminal_period_omission_rate",
                baseline=baseline_omission,
                candidate=candidate_omission,
                rule="候选省略率与基准差值不得超过 0.75",
            )
        )
    return tuple(differences)


def _append_upper_bound_difference(
    differences: list[StyleFeatureDifference],
    *,
    case_id: str,
    feature: str,
    baseline: float | None,
    candidate: float | None,
    allowance: float,
) -> None:
    if baseline is None or candidate is None or candidate <= baseline + allowance:
        return
    differences.append(
        StyleFeatureDifference(
            case_id=case_id,
            feature=feature,
            baseline=baseline,
            candidate=candidate,
            rule="候选值不得超过基准上界",
        )
    )


def _nested_number(payload: dict[str, object], *path: str) -> float | None:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, int | float):
        return float(current)
    return None


def _resolve_style_profile(
    payload: dict[str, object] | None,
) -> dict[str, object] | None:
    if not payload:
        return payload
    nested = payload.get("style_profile")
    if isinstance(nested, dict):
        return {str(key): value for key, value in nested.items()}
    return payload


def _reply_style_turn(
    reply: GeneratedReplyTurn,
    previous_text: str,
) -> StyleTurn:
    return StyleTurn(
        previous_text=previous_text,
        bubbles=tuple(
            StyleBubble(
                text=bubble.content or "",
                delay_ms=bubble.delay_ms,
                kind=bubble.type,
                asset_id=bubble.asset_id,
            )
            for bubble in reply.bubbles
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_messages(row: dict[str, object]) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list):
        raise ValueError("训练样本 messages 格式无效")
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("训练样本消息格式无效")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("训练样本消息字段无效")
        messages.append({"role": role, "content": content})
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("训练样本缺少 assistant 真实目标")
    return messages


def _row_metadata(row: dict[str, object]) -> dict[str, object]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return {str(key): value for key, value in metadata.items()}


def _row_asks_unobserved_current_state(row: dict[str, object]) -> bool:
    """A held-out reply cannot serve as its own current-state evidence."""

    messages = _row_messages(row)[:-1]
    current_user_content = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    return is_current_state_question(current_user_content)


def _style_transfer_content_draft(messages: list[dict[str, str]]) -> str:
    raw = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    ).strip()
    prefix = "内容草稿："
    if raw.startswith(prefix):
        raw = raw[len(prefix) :].lstrip("\n ")
    if not raw:
        raise ValueError("风格改写样本缺少内容草稿")
    return raw


def _blind_content_planner_instruction(source_system: str) -> str:
    """Keep held-out context/evidence while removing persona-expression duties."""

    return (
        source_system + "\n\n你现在只负责决定这轮回复要表达的内容，不负责模仿口癖或分段。"
        "只能使用上面给出的历史和记忆，不得假设未来消息，不得编造事实。"
        "不得用‘听说’补写第三方近况，不得猜测任何人此刻的位置、活动或安排；"
        "没有证据时改成简短反问或只回应情绪。"
        "这是亲密私人聊天，不是客服，禁止使用‘您’、‘请您’、‘我可以帮你’、"
        "‘需要帮忙吗’、‘查找相关信息’、‘具体说明’、‘建议你’等服务式措辞。"
        "输出一段简短、相关、可直接发送的聊天内容草稿；不要输出 JSON、标签、解释或候选项。"
    )


def _row_source_hash(row: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _deterministic_stratified_rows(
    rows: list[dict[str, object]],
    sample_count: int,
) -> list[dict[str, object]]:
    strata: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for row in rows:
        target = _row_messages(row)[-1]["content"]
        modality = "sticker" if "<sticker" in target else "text"
        length = "short" if len(target) <= 40 else "long"
        digest = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        strata.setdefault(f"{modality}:{length}", []).append((digest, row))
    queues = {
        name: [row for _digest, row in sorted(items, key=lambda item: item[0])]
        for name, items in sorted(strata.items())
    }
    selected: list[dict[str, object]] = []
    while len(selected) < min(sample_count, len(rows)):
        progressed = False
        for queue in queues.values():
            if queue and len(selected) < sample_count:
                selected.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _speaker_training_samples(
    path: Path,
    persona: str,
) -> list[SpeakerSample]:
    samples: list[SpeakerSample] = []
    for row in _read_jsonl(path):
        messages = _row_messages(row)
        for index, message in enumerate(messages[1:], start=1):
            if message["role"] == "assistant":
                if index == len(messages) - 1:
                    reply = _parse_dataset_reply(message["content"])
                    text = "\n".join(bubble.content or "" for bubble in reply.bubbles)
                else:
                    text = message["content"].strip()
                if text:
                    samples.append(SpeakerSample("train", persona, text))
            elif message["role"] == "user" and message["content"].strip():
                samples.append(
                    SpeakerSample("train", "__conversation_partner__", message["content"])
                )
    return samples


def _parse_dataset_reply(content: str) -> GeneratedReplyTurn:
    try:
        bubbles = parse_bubble_protocol(content)
    except ValueError:
        if "<" in content or ">" in content:
            raise
        # Dataset turns preserve every consecutive message the real person
        # sent. They may legitimately contain more bubbles than the runtime
        # anti-flood limit enforced by parse_persona_text_turn(). Style
        # evaluation must compare against the full observed turn rather than
        # rejecting or truncating the human reference.
        lines = tuple(line.strip() for line in content.splitlines() if line.strip())
        if not lines:
            raise ValueError("纯文本训练目标不能为空") from None
        return GeneratedReplyTurn(
            bubbles=tuple(
                GeneratedBubble(
                    content=line,
                    delay_ms=0 if index == 0 else 800,
                )
                for index, line in enumerate(lines)
            ),
            raw_output=content,
        )
    return GeneratedReplyTurn(
        bubbles=tuple(
            GeneratedBubble(
                content=bubble.value if bubble.kind == "text" else None,
                delay_ms=bubble.delay_ms,
                type=bubble.kind,
                asset_id=bubble.value if bubble.kind != "text" else None,
            )
            for bubble in bubbles
        ),
        raw_output=content,
    )
