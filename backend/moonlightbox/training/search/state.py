import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from moonlightbox.training.model_identity import _canonical_digest, _stream_sha256
from moonlightbox.training.peft_adapter import (
    BestCheckpointSelection,
    PeftLmAdapter,
    PeftLoraConfig,
    PeftTrainingError,
    TrainingResult,
)
from moonlightbox.training.search.lock import (
    FencingTokenError,
    SearchStateStore,
    TrainingFingerprintMismatch,
    _atomic_write_json,
)


@dataclass(frozen=True)
class LoraSearchResult:
    best_candidate_id: str
    adapter_dir: Path
    best_checkpoint: Path
    completed_epochs: int
    validation_curve: tuple[dict[str, float | int], ...]
    state_path: Path
    search_results: tuple[dict[str, object], ...]
    early_stopping_applied: bool
    best_validation_loss: float


StyleEvaluator = Callable[[Path], dict[str, float | int | str]]


CheckpointEvaluator = Callable[[PeftLoraConfig, Path], float]


SearchCheckpoint = Callable[[dict[str, Any]], None]


def _recover_completed_full_training_result(
    *,
    adapter: PeftLmAdapter,
    full_config: PeftLoraConfig,
    full_state: dict[str, object],
    full_dir: Path,
    expected_config: str,
    expected_config_fingerprint: str,
    state_path: Path,
    state: dict[str, Any],
    checkpoint: SearchCheckpoint | None,
) -> TrainingResult | None:
    """恢复“最后一个权重已保存、汇总状态尚未提交”的完整训练。

    这种边界发生在训练循环完成后，Worker 在 checkpoint 风格验收之前被外部
    中断。它与中途停止不同：adapter 权重和所有定期 checkpoint 都已落盘，
    因而重新训练既浪费时间也会改变随机训练轨迹。我们只在配置、末 checkpoint
    和最终 adapter 都通过完整性检查时复用；其他半途状态仍从干净基座开始。
    """

    recoverable_statuses = {"running", "failed", "recovering_completed_run"}
    if full_state.get("status") not in recoverable_statuses:
        return None
    latest = full_state.get("latest_progress")
    if not isinstance(latest, dict):
        return None
    completed_iteration = latest.get("iteration")
    if (
        latest.get("stage") != "saved"
        or not isinstance(completed_iteration, int)
        or completed_iteration < full_config.resolved_iterations
        or full_state.get("config_fingerprint") != expected_config_fingerprint
    ):
        return None
    config_path = full_dir / "training.json"
    final_weights = full_dir / "adapter_model.safetensors"
    terminal_checkpoint = full_dir / (
        f"{full_config.resolved_iterations:07d}_adapter_model.safetensors"
    )
    if (
        not config_path.is_file()
        or config_path.read_text(encoding="utf-8") != expected_config
        or not final_weights.is_file()
        or not terminal_checkpoint.is_file()
        # ``latest_progress.stage=saved`` 只有在复制 checkpoint 后才写入；再
        # 比对最终 adapter 与末 checkpoint 的摘要，排除进程在覆盖权重文件
        # 期间中断造成的半写入文件。
        or _stream_sha256(final_weights) != _stream_sha256(terminal_checkpoint)
    ):
        return None

    checkpoint_pattern = re.compile(r"^(?P<iteration>[0-9]{7})_adapter_model\.safetensors$")
    checkpoints = sorted(
        (
            path
            for path in full_dir.iterdir()
            if path.is_file() and checkpoint_pattern.fullmatch(path.name)
        ),
        key=lambda path: int(checkpoint_pattern.fullmatch(path.name).group("iteration")),  # type: ignore[union-attr]
    )
    if terminal_checkpoint not in checkpoints:
        return None

    # 新版本会在每次验证后持久化曲线；旧作业没有这份审计数据时，只重算
    # valid split。绝不改用独立 test split，后者只用于最终一次复验。
    raw_curve = full_state.get("recovery_validation_curve", full_state.get("validation_curve"))
    known_losses: dict[int, float] = {}
    if isinstance(raw_curve, list):
        for point in raw_curve:
            if not isinstance(point, dict):
                continue
            iteration = point.get("iteration")
            loss = point.get("validation_loss")
            if (
                isinstance(iteration, int)
                and isinstance(loss, int | float)
                and math.isfinite(float(loss))
            ):
                known_losses[iteration] = float(loss)

    evaluator = getattr(adapter, "evaluate_split_loss", None)
    if not callable(evaluator) and len(known_losses) < len(checkpoints):
        raise PeftTrainingError("无法恢复完整训练：缺少 valid checkpoint 评估器")
    full_state.update(
        {
            "status": "recovering_completed_run",
            "resume_strategy": "reuse_saved_completed_run",
            "optimizer_state_restored": False,
            "normalized_config": expected_config,
            "config_fingerprint": expected_config_fingerprint,
        }
    )
    state["stage"] = "full_training"
    _publish_search_state(state_path, state, checkpoint)
    for weights in checkpoints:
        match = checkpoint_pattern.fullmatch(weights.name)
        assert match is not None
        iteration = int(match.group("iteration"))
        if iteration not in known_losses:
            staging = full_dir / f".recovery-valid-{iteration}"
            try:
                _restore_best_adapter(full_dir, staging, weights)
                known_losses[iteration] = float(evaluator(full_config, staging, split="valid"))
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            full_state["recovery_validation_curve"] = [
                {"iteration": index, "validation_loss": known_losses[index]}
                for index in sorted(known_losses)
            ]
            _publish_search_state(state_path, state, checkpoint)

    validation_curve = tuple(
        {"iteration": iteration, "validation_loss": known_losses[iteration]}
        for iteration in sorted(known_losses)
    )
    best_iteration = min(
        known_losses,
        key=lambda iteration: (known_losses[iteration], iteration),
    )
    best_checkpoint = full_dir / f"{best_iteration:07d}_adapter_model.safetensors"
    return TrainingResult(
        adapter_dir=full_dir,
        iterations=full_config.resolved_iterations,
        config_path=config_path,
        peft_version=adapter.capabilities.version,
        validation_curve=validation_curve,
        best_checkpoint_selection=BestCheckpointSelection(
            validation_loss=known_losses[best_iteration],
            validation_iteration=best_iteration,
            checkpoint_iteration=best_iteration,
            checkpoint=best_checkpoint,
        ),
        command=("python", "-m", "moonlightbox.training.peft_adapter"),
    )


def _load_or_create_search_state(
    path: Path,
    *,
    fingerprint: str,
    fingerprint_payload: dict[str, object],
) -> dict[str, Any]:
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TrainingFingerprintMismatch("候选训练状态格式无效")
        state = cast(dict[str, Any], loaded)
        if state.get("fingerprint") != fingerprint:
            raise TrainingFingerprintMismatch("训练数据或配置指纹已变化，禁止复用旧 checkpoint")
        return state
    return {
        "schema_version": "moonlightbox-lora-search-state-v1",
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "stage": "pending",
        "candidates": {},
        "full_training": {},
    }


def _migrate_legacy_full_checkpoint_preservation_fingerprint(
    path: Path,
    *,
    fingerprint: str,
    fingerprint_payload: dict[str, object],
) -> None:
    """仅迁移早期漏记 ``preserve_checkpoints`` 的等价搜索状态。

    旧 Linux 训练会实际保留完整训练 checkpoint，却在指纹配置中记录为
    ``false``。这会让一个已完成的末批次被误判为不可恢复。迁移只接受把
    此单一字段从 false 修正为 true 后得到的精确旧指纹；任意数据、模型或
    其他训练参数变化仍由常规指纹校验拒绝。
    """

    if not path.is_file():
        return
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(loaded, dict) or loaded.get("fingerprint") == fingerprint:
        return
    legacy_payload = json.loads(json.dumps(fingerprint_payload, ensure_ascii=False))
    normalized = legacy_payload.get("normalized_config")
    if not isinstance(normalized, dict):
        return
    changed = False
    for candidate_config in normalized.values():
        if not isinstance(candidate_config, dict):
            return
        full_config = candidate_config.get("best_candidate_full_run")
        if not isinstance(full_config, str):
            return
        try:
            parsed = json.loads(full_config)
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict) or parsed.get("preserve_checkpoints") is not True:
            return
        parsed["preserve_checkpoints"] = False
        candidate_config["best_candidate_full_run"] = json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        changed = True
    if not changed:
        return
    legacy_payload["config_fingerprint"] = _canonical_digest(normalized)
    if loaded.get("fingerprint") != _canonical_digest(legacy_payload):
        return

    loaded["fingerprint"] = fingerprint
    loaded["fingerprint_payload"] = fingerprint_payload
    full = loaded.get("full_training")
    current_configs = fingerprint_payload.get("normalized_config")
    best_candidate_id = loaded.get("best_candidate_id")
    if (
        isinstance(full, dict)
        and isinstance(current_configs, dict)
        and isinstance(best_candidate_id, str)
        and isinstance(current_configs.get(best_candidate_id), dict)
    ):
        corrected = current_configs[best_candidate_id].get("best_candidate_full_run")
        if isinstance(corrected, str):
            full["normalized_config"] = corrected
            full["config_fingerprint"] = hashlib.sha256(corrected.encode("utf-8")).hexdigest()
    _atomic_write_json(path, loaded)


def _publish_search_state(
    path: Path,
    state: dict[str, Any],
    checkpoint: SearchCheckpoint | None,
) -> None:
    generation = state.get("run_generation")
    if not isinstance(generation, str):
        raise FencingTokenError("搜索状态缺少 fencing token")
    SearchStateStore(path, run_generation=generation).write(state)
    if checkpoint is not None:
        checkpoint(json.loads(json.dumps(state, ensure_ascii=False)))


def _candidate_progress(
    path: Path,
    state: dict[str, Any],
    candidate_id: str,
    progress: dict[str, object],
    checkpoint: SearchCheckpoint | None,
) -> None:
    candidate_states = state["candidates"]
    candidate_state = candidate_states[candidate_id]
    candidate_state["latest_progress"] = progress
    _publish_search_state(path, state, checkpoint)


def _full_progress(
    path: Path,
    state: dict[str, Any],
    max_epochs: int,
    max_iterations: int,
    progress: dict[str, object],
    checkpoint: SearchCheckpoint | None,
) -> None:
    full_state = state["full_training"]
    full_state["latest_progress"] = {
        **progress,
        "max_epochs": max_epochs,
        "total_iterations": max_iterations,
    }
    # 训练循环的 validation point 同时是 checkpoint 的审计索引。逐点写入，
    # 这样即使 Worker 恰好在最后一次权重保存后退出，也无需重新跑完整 epoch。
    iteration = progress.get("iteration")
    validation_loss = progress.get("validation_loss")
    if isinstance(iteration, int) and isinstance(validation_loss, int | float):
        existing = full_state.get("recovery_validation_curve")
        curve = (
            [item for item in existing if isinstance(item, dict)]
            if isinstance(existing, list)
            else []
        )
        by_iteration = {
            int(item["iteration"]): dict(item)
            for item in curve
            if isinstance(item.get("iteration"), int)
        }
        by_iteration[iteration] = {
            "iteration": iteration,
            "validation_loss": float(validation_loss),
        }
        full_state["recovery_validation_curve"] = [
            by_iteration[key] for key in sorted(by_iteration)
        ]
    _publish_search_state(path, state, checkpoint)


def _search_job_progress(
    state: dict[str, Any],
    *,
    maximum_iterations: int,
) -> float:
    candidates = state.get("candidates")
    succeeded_count = (
        sum(
            isinstance(value, dict) and value.get("status") == "succeeded"
            for value in candidates.values()
        )
        if isinstance(candidates, dict)
        else 0
    )
    if state.get("stage") not in {"full_training", "search_completed"}:
        return 0.15 + min(0.25, succeeded_count / 3 * 0.25)
    full = state.get("full_training")
    latest = full.get("latest_progress") if isinstance(full, dict) else None
    raw_iteration = latest.get("iteration", 0) if isinstance(latest, dict) else 0
    iteration = raw_iteration if isinstance(raw_iteration, int | float) else 0
    total = max(1, maximum_iterations)
    return 0.4 + 0.35 * min(1.0, max(0.0, float(iteration)) / total)


def _metrics(candidate_state: dict[str, object]) -> dict[str, object]:
    raw = candidate_state.get("metrics")
    return raw if isinstance(raw, dict) else {}


def _metric_float(
    metrics: dict[str, object],
    name: str,
    default: float,
) -> float:
    value = metrics.get(name)
    return float(value) if isinstance(value, int | float) else default


def _optional_existing_path(value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    return path if path.is_file() else None


def _checkpoint_state_valid(
    state: dict[str, object],
    checkpoint: Path,
    expected_config_fingerprint: str,
    *,
    digest_key: str = "checkpoint_sha256",
) -> bool:
    expected_digest = state.get(digest_key)
    return (
        state.get("config_fingerprint") == expected_config_fingerprint
        and isinstance(expected_digest, str)
        and checkpoint.is_file()
        and _stream_sha256(checkpoint) == expected_digest
    )


def _quarantine_directory(path: Path) -> Path | None:
    """隔离校验失败的产物，自动流程中绝不立即销毁。"""

    if not path.exists():
        return None
    stale = path.with_name(f"{path.name}.stale-{uuid4().hex}")
    path.replace(stale)
    # 被篡改、配置漂移或旧版本格式不匹配的权重不能继续参与自动发布，但仍是
    # 训练事故的唯一证据。由受控的显式清理命令处理空间回收。
    return stale


def _archive_interrupted_training_directory(path: Path) -> Path:
    """保留不可续训的现场，而非在自动恢复前销毁它。"""

    archived = path.with_name(
        f"{path.name}.interrupted-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex}"
    )
    path.replace(archived)
    return archived


def _read_training_yaml(path: Path | None) -> str:
    return path.read_text(encoding="utf-8") if path is not None and path.is_file() else ""


def _training_command(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        "python",
        "-m",
        "moonlightbox.training.peft_adapter",
        "--config",
        str(path),
    ]


def _restore_best_adapter(
    source: Path,
    target: Path,
    best_checkpoint: Path,
    *,
    archive_existing: bool = False,
) -> Path:
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for item in source.iterdir():
        if item.name == "adapter_model.safetensors" or item.name.endswith(
            "_adapter_model.safetensors"
        ):
            continue
        destination = temporary / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copyfile(item, destination)
    temporary_weights = temporary / "adapter_model.safetensors.tmp"
    shutil.copyfile(best_checkpoint, temporary_weights)
    temporary_weights.replace(temporary / "adapter_model.safetensors")
    if target.exists():
        if archive_existing:
            _archive_interrupted_training_directory(target)
        else:
            shutil.rmtree(target)
    temporary.replace(target)
    return target / "adapter_model.safetensors"


def _replace_adapter_weights(source: Path, target: Path) -> None:
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _search_result_from_state(
    state_path: Path,
    state: dict[str, Any],
) -> LoraSearchResult:
    full = state["full_training"]
    checkpoint = Path(str(full["best_checkpoint"]))
    raw_curve = full.get("validation_curve", [])
    curve = tuple(dict(point) for point in raw_curve if isinstance(point, dict))
    raw_results = state.get("search_results", [])
    results = tuple(dict(item) for item in raw_results if isinstance(item, dict))
    return LoraSearchResult(
        best_candidate_id=str(state["best_candidate_id"]),
        adapter_dir=Path(str(full["adapter_dir"])),
        best_checkpoint=checkpoint,
        completed_epochs=int(full["completed_epochs"]),
        validation_curve=curve,
        state_path=state_path,
        search_results=results,
        early_stopping_applied=bool(full.get("early_stopping_applied", False)),
        best_validation_loss=float(full["best_validation_loss"]),
    )
