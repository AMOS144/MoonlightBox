import math
from datetime import datetime
from pathlib import Path

from moonlightbox.imports.types import MessageKind
from moonlightbox.training.dataset_builder import (
    ConfirmedEventContext,
    TrainingExample,
    parse_assistant_protocol,
)
from moonlightbox.training.model_acceptance import (
    MAXIMUM_STYLE_TRANSFER_FALLBACK_RATE,
    AcceptanceMetric,
    ModelGateSnapshot,
)
from moonlightbox.training.models import ModelVersion, TimelineConfirmation


def _message_kind(value: str) -> MessageKind:
    try:
        return MessageKind(value)
    except ValueError:
        # 媒体关联后的展示类型不参与文本训练，统一降级为未知类型。
        return MessageKind.UNKNOWN


def evaluate_training_quality(
    examples: list[TrainingExample],
    cutoff: datetime,
    metrics: dict[str, float],
) -> dict[str, object]:
    assistant_targets = 0
    multi_bubble_targets = 0
    structure_valid = True
    future_leak_count = 0
    for example in examples:
        if example.target_at > cutoff:
            future_leak_count += 1
        if not example.messages or example.messages[-1].role != "assistant":
            structure_valid = False
            continue
        assistant_targets += 1
        try:
            bubbles = parse_assistant_protocol(example.messages[-1].content)
            if not bubbles:
                structure_valid = False
            if len(bubbles) > 1:
                multi_bubble_targets += 1
        except ValueError:
            structure_valid = False
    losses_finite = all(math.isfinite(value) for value in metrics.values())
    passed = (
        bool(examples)
        and assistant_targets == len(examples)
        and structure_valid
        and future_leak_count == 0
        and losses_finite
    )
    return {
        "passed": passed,
        "future_leak_count": future_leak_count,
        "multi_bubble_ratio": (
            multi_bubble_targets / assistant_targets if assistant_targets else 0.0
        ),
    }


def _is_peft_comparable_model(
    version: ModelVersion | None,
    expected_base_model: str,
) -> bool:
    """判断活动版本能否由当前 Linux PEFT 验收器安全加载。

    历史 MLX adapter 和相对的 macOS 基座路径不能成为 Qwen/PEFT 的比较对象。
    它们不应让新训练在验收阶段中断，但仍由注册表的 CAS 机制原子替换。
    """

    if version is None or version.base_model.strip().lower() != expected_base_model.strip().lower():
        return False
    adapter_dir = Path(version.adapter_path)
    return (
        adapter_dir.is_dir()
        and (adapter_dir / "adapter_config.json").is_file()
        and (adapter_dir / "adapter_model.safetensors").is_file()
    )


def _is_compact_qwen3_1_7b(base_model: str) -> bool:
    """识别带不可变 HF revision 的 1.7B 基座，选择 4GB 安全候选。"""

    repository, _, _revision = base_model.partition("@")
    return repository.removeprefix("hf://").strip().lower() == "qwen/qwen3-1.7b"


def acceptance_score(report: object) -> float:
    case_count = int(getattr(report, "case_count", 0))
    passed_count = int(getattr(report, "passed_count", 0))
    if case_count <= 0:
        return 0.0
    penalty = (
        int(getattr(report, "structure_failures", 0))
        + int(getattr(report, "forbidden_fact_failures", 0))
        + int(getattr(report, "raw_output_failures", 0))
        + int(getattr(report, "style_feature_failures", 0))
    )
    return max(0.0, (passed_count - penalty) / case_count)


def model_gate_snapshot(
    model_id: str,
    report: object,
    style_metrics: dict[str, float | int | str],
) -> ModelGateSnapshot:
    case_count = int(getattr(report, "case_count", 0))
    passed_count = int(getattr(report, "passed_count", 0))
    structure_failures = int(getattr(report, "structure_failures", 0))
    fact_failures = int(getattr(report, "forbidden_fact_failures", 0))
    raw_failures = int(getattr(report, "raw_output_failures", 0))
    fixed = float(
        case_count > 0
        and passed_count == case_count
        and structure_failures == 0
        and fact_failures == 0
        and raw_failures == 0
    )
    semantic = {
        "fixed_regression": AcceptanceMetric(fixed),
        "structure": AcceptanceMetric(float(structure_failures == 0)),
        "future_isolation": AcceptanceMetric(float(raw_failures == 0)),
        "unsupported_fact": AcceptanceMetric(float(fact_failures == 0)),
    }
    if "memory_causal_variant_accuracy" in style_metrics:
        semantic.update(
            {
                "memory_causal_variant_accuracy": AcceptanceMetric(
                    float(style_metrics["memory_causal_variant_accuracy"]),
                    minimum=0.75,
                ),
                "memory_causal_grounding_rate": AcceptanceMetric(
                    float(style_metrics["memory_causal_grounding_rate"]),
                    minimum=1.0,
                ),
                "memory_causal_switch_rate": AcceptanceMetric(
                    float(style_metrics["memory_causal_switch_rate"]),
                    minimum=0.75,
                ),
            }
        )
    style = {
        "style_distance": AcceptanceMetric(
            float(style_metrics["style_distance"]),
            maximum=0.35,
            higher_is_better=False,
        ),
        "speaker_probability": AcceptanceMetric(
            float(style_metrics["speaker_probability"]),
        ),
        "human_oracle_speaker_probability": AcceptanceMetric(
            float(style_metrics["human_oracle_speaker_probability"]),
        ),
        "speaker_probability_alignment": AcceptanceMetric(
            float(style_metrics["speaker_probability_alignment"]),
        ),
        "paired_response_similarity": AcceptanceMetric(
            float(style_metrics["paired_response_similarity"]),
        ),
        "low_authority_prompt_leak_rate": AcceptanceMetric(
            float(style_metrics["low_authority_prompt_leak_rate"]),
            maximum=0.0,
            higher_is_better=False,
        ),
        "valid_grounding_failure_rate": AcceptanceMetric(
            float(style_metrics.get("valid_grounding_failure_rate", 1.0)),
            maximum=(
                MAXIMUM_STYLE_TRANSFER_FALLBACK_RATE
                if style_metrics.get("valid_training_task") == "style_transfer"
                else 0.0
            ),
            higher_is_better=False,
        ),
    }
    if "memory_causal_style_consistency" in style_metrics:
        style["memory_causal_style_consistency"] = AcceptanceMetric(
            float(style_metrics["memory_causal_style_consistency"]),
            minimum=0.7,
        )
    return ModelGateSnapshot(
        model_id=model_id,
        semantic=semantic,
        style=style,
    )


def event_context(snapshot: dict[str, object]) -> ConfirmedEventContext:
    evidence = snapshot.get("evidence_ids")
    return ConfirmedEventContext(
        event_id=str(snapshot["event_id"]),
        title=str(snapshot["title"]),
        summary=str(snapshot["summary"]),
        lane=str(snapshot["lane"]),
        event_status=str(snapshot["event_status"]),
        evidence_ids=tuple(str(item) for item in evidence) if isinstance(evidence, list) else (),
        before_state=_optional_string(snapshot.get("before_state")),
        after_state=_optional_string(snapshot.get("after_state")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def same_timeline(value: datetime, reference: datetime) -> datetime:
    """将 SQLite 无时区时间与清单 ISO 时间对齐后再比较。"""

    if reference.tzinfo is None or reference.utcoffset() is None:
        return value.replace(tzinfo=None)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=reference.tzinfo)
    return value.astimezone(reference.tzinfo)


def _node_snapshot_hash(confirmation: TimelineConfirmation) -> str:
    import hashlib
    import json

    serialized = json.dumps(
        confirmation.event_revision_snapshots,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
