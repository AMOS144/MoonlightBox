import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from moonlightbox.training.model_identity import _stream_sha256


@dataclass(frozen=True)
class PreferenceTrainingPlan:
    initial_adapter_file: Path
    output_dir: Path
    iterations: int
    resumed_from_iteration: int = 0
    cached_adapter_dir: Path | None = None


PREFERENCE_TRAINING_PROTOCOL_VERSION = "one-effective-pass-v2"


def preference_training_iterations(
    train_pair_count: int,
    *,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
) -> int:
    """Return one effective pass over preference pairs.

    A training iteration consumes ``batch_size * gradient_accumulation_steps``
    pairs.  Treating the raw pair count as the iteration count silently runs
    several epochs and can overwrite the conversational SFT adapter.
    """

    if min(train_pair_count, batch_size, gradient_accumulation_steps) < 1:
        raise ValueError("偏好训练样本数和有效 batch 必须为正数")
    return math.ceil(train_pair_count / (batch_size * gradient_accumulation_steps))


def plan_preference_training(
    training_root: Path,
    *,
    source_adapter_file: Path,
    total_iterations: int,
) -> PreferenceTrainingPlan:
    """Resume interrupted preference tuning from the newest durable weight snapshot."""

    if total_iterations < 1:
        raise ValueError("偏好训练总迭代数必须为正数")
    selected_path = training_root / "memory-authority-selected.json"
    if selected_path.is_file():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        raw_adapter_dir = selected.get("adapter_dir") if isinstance(selected, dict) else None
        raw_sha256 = selected.get("sha256") if isinstance(selected, dict) else None
        raw_audit_path = selected.get("audit_path") if isinstance(selected, dict) else None
        raw_cumulative_iteration = (
            selected.get("cumulative_iteration") if isinstance(selected, dict) else None
        )
        cumulative_iteration = (
            raw_cumulative_iteration
            if isinstance(raw_cumulative_iteration, int)
            and not isinstance(raw_cumulative_iteration, bool)
            else 0
        )
        adapter_dir = Path(raw_adapter_dir).resolve() if isinstance(raw_adapter_dir, str) else None
        audit_path = Path(raw_audit_path).resolve() if isinstance(raw_audit_path, str) else None
        root = training_root.resolve()
        audit: dict[str, object] = {}
        if audit_path is not None and audit_path.is_file() and audit_path.is_relative_to(root):
            try:
                loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded_audit = {}
            if isinstance(loaded_audit, dict):
                audit = loaded_audit
        selection_valid = (
            isinstance(selected, dict)
            and selected.get("schema_version") == "moonlightbox.preference-checkpoint-selection.v2"
            and adapter_dir is not None
            and adapter_dir.is_relative_to(root)
            and isinstance(raw_sha256, str)
            and (adapter_dir / "adapter_model.safetensors").is_file()
            and _stream_sha256(adapter_dir / "adapter_model.safetensors") == raw_sha256
            and 0 < cumulative_iteration <= total_iterations
            and audit.get("schema_version") == "moonlightbox.preference-checkpoint-acceptance.v2"
            and audit.get("passed") is True
        )
        if selection_valid and adapter_dir is not None:
            return PreferenceTrainingPlan(
                initial_adapter_file=adapter_dir / "adapter_model.safetensors",
                output_dir=adapter_dir,
                iterations=0,
                resumed_from_iteration=cumulative_iteration,
                cached_adapter_dir=adapter_dir,
            )
    completed_dirs: list[tuple[int, float, Path]] = []
    partial_checkpoints: list[tuple[int, Path]] = []
    for output_dir in training_root.glob(
        f"memory-authority-{PREFERENCE_TRAINING_PROTOCOL_VERSION}*"
    ):
        if not output_dir.is_dir():
            continue
        offset = _preference_resume_offset(output_dir.name)
        if offset is None:
            continue
        adapter_file = output_dir / "adapter_model.safetensors"
        metrics_file = output_dir / "preference_metrics.json"
        if adapter_file.is_file() and metrics_file.is_file():
            completed_dirs.append((offset, metrics_file.stat().st_mtime, output_dir))
            continue
        numbered = [
            (int(match.group("iteration")), path)
            for path in output_dir.glob("*_adapter_model.safetensors")
            if (
                match := re.fullmatch(
                    r"(?P<iteration>[0-9]{7})_adapter_model\.safetensors",
                    path.name,
                )
            )
        ]
        if numbered:
            local_iteration, checkpoint = max(numbered, key=lambda item: item[0])
            partial_checkpoints.append((offset + local_iteration, checkpoint))
    if completed_dirs:
        _offset, _mtime, cached = max(
            completed_dirs,
            key=lambda item: (item[1], item[0]),
        )
        return PreferenceTrainingPlan(
            initial_adapter_file=cached / "adapter_model.safetensors",
            output_dir=cached,
            iterations=0,
            resumed_from_iteration=total_iterations,
            cached_adapter_dir=cached,
        )
    if partial_checkpoints:
        completed, checkpoint = max(partial_checkpoints, key=lambda item: item[0])
        if completed >= total_iterations:
            raise ValueError("偏好训练已保存最终权重但缺少完整验证指标")
        return PreferenceTrainingPlan(
            initial_adapter_file=checkpoint,
            output_dir=(
                training_root
                / f"memory-authority-{PREFERENCE_TRAINING_PROTOCOL_VERSION}-resume-{completed}"
            ),
            iterations=total_iterations - completed,
            resumed_from_iteration=completed,
        )
    return PreferenceTrainingPlan(
        initial_adapter_file=source_adapter_file,
        output_dir=training_root / f"memory-authority-{PREFERENCE_TRAINING_PROTOCOL_VERSION}",
        iterations=total_iterations,
    )


def _preference_resume_offset(directory_name: str) -> int | None:
    prefix = f"memory-authority-{PREFERENCE_TRAINING_PROTOCOL_VERSION}"
    if directory_name == prefix:
        return 0
    match = re.fullmatch(rf"{re.escape(prefix)}-resume-(?P<offset>[0-9]+)", directory_name)
    return int(match.group("offset")) if match is not None else None
