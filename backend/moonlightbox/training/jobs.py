import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select

from moonlightbox.branches.identity import (
    EvidenceBackedIdentityKernelBuilder,
)
from moonlightbox.embeddings import LocalChineseEmbedder
from moonlightbox.evaluation.blind_service import HumanBlindStudyService
from moonlightbox.evaluation.models import HumanBlindStudy
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.training.acceptance_retry import resolve_reply_protocol
from moonlightbox.training.conversation_action_policy import (
    build_conversation_action_policy,
)
from moonlightbox.training.dataset_builder import (
    ConfirmedEventContext,
    DatasetBuilder,
    TrainingExample,
    parse_assistant_protocol,
)
from moonlightbox.training.expression_policy import build_expression_policy
from moonlightbox.training.media_behavior_policy import build_media_behavior_policy
from moonlightbox.training.model_acceptance import (
    MAXIMUM_STYLE_TRANSFER_FALLBACK_RATE,
    MINIMUM_PAIRED_RESPONSE_SIMILARITY,
    AcceptanceMetric,
    ModelAcceptanceRunner,
    ModelGateSnapshot,
    evaluate_activation_gates,
)
from moonlightbox.training.models import ModelVersion, TimelineConfirmation
from moonlightbox.training.peft_adapter import (
    BestCheckpointSelection,
    LoraCandidate,
    PeftLmAdapter,
    PeftLoraConfig,
    PeftTrainingError,
    TrainingResult,
    cleanup_generated_checkpoints,
    compact_persona_candidates,
    default_lora_candidates,
    high_capacity_persona_candidates,
    normalized_peft_config,
)
from moonlightbox.training.peft_preference_trainer import (
    PreferenceTrainingConfig,
    run_persona_preference_training,
)
from moonlightbox.training.persona_preference import build_persona_preference_dataset
from moonlightbox.training.registry import ModelRegistry
from moonlightbox.training.sticker_policy import (
    build_sticker_evaluation_cases_from_database,
    build_sticker_policy_from_database,
    evaluate_sticker_policy,
    tune_sticker_policy_on_valid,
)
from moonlightbox.training.style_features import STYLE_FEATURE_SCHEMA_VERSION


class TrainingFingerprintMismatch(RuntimeError):
    pass


class AllCandidatesFailedError(RuntimeError):
    pass


class SearchLockTimeout(RuntimeError):
    pass


class FencingTokenError(RuntimeError):
    pass


class CandidateValidationError(RuntimeError):
    pass


class CandidateResourceError(RuntimeError):
    pass


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


class SearchOutputLock(AbstractContextManager[dict[str, object]]):
    def __init__(self, output_dir: Path, *, timeout_seconds: float = 30.0) -> None:
        self._output_dir = output_dir
        self._timeout_seconds = timeout_seconds
        self._stream: BinaryIO | None = None
        self.owner = {
            "pid": os.getpid(),
            "run_generation": str(uuid4()),
            "acquired_at": datetime.now().astimezone().isoformat(),
        }

    def __enter__(self) -> dict[str, object]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stream = (self._output_dir / ".search.lock").open("a+b")
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    stream.close()
                    raise SearchLockTimeout("等待搜索目录独占锁超时") from None
                time.sleep(0.01)
        self._stream = stream
        _atomic_write_json(self._output_dir / ".search-lock-owner.json", self.owner)
        return self.owner

    def __exit__(self, *_args: object) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


class SearchStateStore:
    def __init__(self, path: Path, *, run_generation: str) -> None:
        self.path = path
        self.run_generation = run_generation

    def write(self, state: dict[str, Any]) -> None:
        if self.path.is_file():
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(current, dict) and current.get("run_generation") != self.run_generation:
                raise FencingTokenError("旧 worker 的 fencing token 已失效")
        state["run_generation"] = self.run_generation
        _atomic_write_json(self.path, state)

    def takeover(self, state: dict[str, Any]) -> None:
        state["run_generation"] = self.run_generation
        _atomic_write_json(self.path, state)


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


def _validate_candidate_ids(candidates: tuple[LoraCandidate, ...]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        if re.fullmatch(r"[A-Za-z0-9_-]+", candidate.candidate_id) is None:
            raise ValueError("candidate_id 只允许字母、数字、下划线和连字符")
        if candidate.candidate_id in seen:
            raise ValueError(f"发现重复 candidate_id：{candidate.candidate_id}")
        seen.add(candidate.candidate_id)


def run_lora_candidate_search(
    adapter: PeftLmAdapter,
    *,
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    data_manifest_digest: str,
    protocol_versions: dict[str, str],
    train_example_count: int,
    batch_size: int,
    style_evaluator: StyleEvaluator,
    max_epochs: int,
    full_checkpoint_style_evaluator: StyleEvaluator | None = None,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 512,
    checkpoint_evaluator: CheckpointEvaluator | None = None,
    early_stopping_patience: int = 2,
    short_run_iterations: int = 120,
    candidates: tuple[LoraCandidate, ...] | None = None,
    allow_single_candidate_semantic_fallback: bool = False,
    checkpoint: SearchCheckpoint | None = None,
    lock_timeout_seconds: float = 30.0,
) -> LoraSearchResult:
    """串行短跑候选，再从干净基础模型完整训练最优配置。

    ``allow_single_candidate_semantic_fallback`` 仅适用于资源约束下已明确选定的
    单一候选：短跑只按验证损失确认其可训练，不能替代完整训练后的最终验收。
    """

    selected_candidates = candidates or default_lora_candidates()
    _validate_candidate_ids(selected_candidates)
    with SearchOutputLock(
        output_dir,
        timeout_seconds=lock_timeout_seconds,
    ) as owner:
        return _run_lora_candidate_search_unlocked(
            adapter,
            base_model=base_model,
            data_dir=data_dir,
            output_dir=output_dir,
            data_manifest_digest=data_manifest_digest,
            protocol_versions=protocol_versions,
            train_example_count=train_example_count,
            batch_size=batch_size,
            style_evaluator=style_evaluator,
            full_checkpoint_style_evaluator=full_checkpoint_style_evaluator,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            checkpoint_evaluator=checkpoint_evaluator,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            short_run_iterations=short_run_iterations,
            candidates=selected_candidates,
            allow_single_candidate_semantic_fallback=(allow_single_candidate_semantic_fallback),
            checkpoint=checkpoint,
            run_generation=str(owner["run_generation"]),
        )


def _run_lora_candidate_search_unlocked(
    adapter: PeftLmAdapter,
    *,
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    data_manifest_digest: str,
    protocol_versions: dict[str, str],
    train_example_count: int,
    batch_size: int,
    style_evaluator: StyleEvaluator,
    full_checkpoint_style_evaluator: StyleEvaluator | None,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    checkpoint_evaluator: CheckpointEvaluator | None,
    max_epochs: int,
    early_stopping_patience: int,
    short_run_iterations: int,
    candidates: tuple[LoraCandidate, ...],
    allow_single_candidate_semantic_fallback: bool,
    checkpoint: SearchCheckpoint | None,
    run_generation: str,
) -> LoraSearchResult:
    selected_candidates = candidates
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "search-state.json"
    base_identity = _resolve_base_model_identity(
        base_model,
        cache_path=output_dir / "base-model-digest-cache.json",
    )
    resolved_base_model = str(base_identity["resolved_path"])
    data_identity = _training_data_identity(data_dir)
    capabilities = adapter.capabilities
    environment = _training_environment()
    if environment["peft_version"] == "not-installed":
        environment["peft_version"] = capabilities.version
    candidate_configs: dict[str, dict[str, str]] = {}
    for candidate in selected_candidates:
        short_config = _candidate_training_config(
            candidate,
            base_model=resolved_base_model,
            data_dir=data_dir,
            adapter_dir=output_dir / "candidates" / candidate.candidate_id,
            train_example_count=train_example_count,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            iteration_limit=short_run_iterations,
        )
        full_config = _candidate_training_config(
            candidate,
            base_model=resolved_base_model,
            data_dir=data_dir,
            adapter_dir=output_dir / "full",
            train_example_count=train_example_count,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            # 完整训练若要按人格/语义选择 checkpoint，所有验证点的 adapter
            # 都是审计输入，不能在训练时提前裁剪。
            preserve_checkpoints=full_checkpoint_style_evaluator is not None,
        )
        candidate_configs[candidate.candidate_id] = {
            "candidate_short_run": normalized_peft_config(short_config, capabilities),
            "best_candidate_full_run": normalized_peft_config(full_config, capabilities),
        }
    config_fingerprint = _canonical_digest(candidate_configs)
    fingerprint_payload = {
        "schema_version": "moonlightbox-lora-search-state-v1",
        "base_model": base_identity,
        "data_files": data_identity["files"],
        "data_digest": data_identity["digest"],
        "test_used_for_tuning": data_identity["test_used_for_tuning"],
        "declared_data_manifest_digest": data_manifest_digest,
        "protocol_versions": dict(sorted(protocol_versions.items())),
        "stages": ["candidate_short_run", "best_candidate_full_run"],
        "train_example_count": train_example_count,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "steps_per_epoch": max(
            1,
            math.ceil(train_example_count / (batch_size * gradient_accumulation_steps)),
        ),
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "short_run_iterations": short_run_iterations,
        "candidate_selection_policy": {
            # 写入指纹，避免旧状态在改变短跑选择规则后被错误复用。
            "allow_single_candidate_semantic_fallback": (allow_single_candidate_semantic_fallback),
            "fallback_rule": "validation_loss_only_for_explicit_single_candidate_v1",
        },
        "adapter_control": {
            "candidate_early_stopping_patience": 1,
            "full_early_stopping_patience": early_stopping_patience,
            "full_checkpoint_selection": (
                "held_out_style_then_validation_v1"
                if full_checkpoint_style_evaluator is not None
                else "validation_loss"
            ),
            "checkpoint_mapping_rule": "peft-validation-at-save-step",
        },
        "candidates": [_candidate_payload(candidate) for candidate in selected_candidates],
        "normalized_config": candidate_configs,
        "config_fingerprint": config_fingerprint,
        "environment": environment,
    }
    fingerprint = _canonical_digest(fingerprint_payload)
    _migrate_legacy_full_checkpoint_preservation_fingerprint(
        state_path,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
    )
    state = _load_or_create_search_state(
        state_path,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
    )
    SearchStateStore(
        state_path,
        run_generation=run_generation,
    ).takeover(state)
    state["environment"] = environment
    _publish_search_state(state_path, state, checkpoint)

    # Successful full training deliberately prunes short-run candidate weights.
    # Validate and reuse the authoritative final adapter before inspecting those
    # disposable artifacts, otherwise a resumed downstream stage retrains search.
    completed_full = state.get("full_training")
    completed_best_id = state.get("best_candidate_id")
    if (
        isinstance(completed_full, dict)
        and completed_full.get("status") == "succeeded"
        and isinstance(completed_best_id, str)
        and completed_best_id in candidate_configs
    ):
        expected_full_fingerprint = hashlib.sha256(
            candidate_configs[completed_best_id]["best_candidate_full_run"].encode("utf-8")
        ).hexdigest()
        final_checkpoint = Path(str(completed_full.get("best_checkpoint", "")))
        if _checkpoint_state_valid(
            completed_full,
            final_checkpoint,
            expected_full_fingerprint,
            digest_key="final_adapter_sha256",
        ):
            state["stage"] = "search_completed"
            _publish_search_state(state_path, state, checkpoint)
            return _search_result_from_state(state_path, state)

    candidate_states = state["candidates"]
    if not isinstance(candidate_states, dict):
        raise TrainingFingerprintMismatch("候选训练状态格式无效")
    for candidate in selected_candidates:
        raw_existing = candidate_states.get(candidate.candidate_id)
        existing = raw_existing if isinstance(raw_existing, dict) else {}
        adapter_dir = output_dir / "candidates" / candidate.candidate_id
        weights_path = adapter_dir / "adapter_model.safetensors"
        expected_config_fingerprint = hashlib.sha256(
            candidate_configs[candidate.candidate_id]["candidate_short_run"].encode("utf-8")
        ).hexdigest()
        if existing.get("status") == "succeeded":
            if _checkpoint_state_valid(
                existing,
                weights_path,
                expected_config_fingerprint,
            ):
                continue
            _quarantine_directory(adapter_dir)
        elif adapter_dir.exists():
            _quarantine_directory(adapter_dir)
        candidate_state: dict[str, object] = {
            "status": "running",
            "candidate_id": candidate.candidate_id,
            "config": _candidate_payload(candidate),
            "adapter_dir": str(adapter_dir),
            "normalized_config": candidate_configs[candidate.candidate_id]["candidate_short_run"],
            "config_fingerprint": expected_config_fingerprint,
        }
        candidate_states[candidate.candidate_id] = candidate_state
        state["stage"] = "candidate_search"
        _publish_search_state(state_path, state, checkpoint)
        try:

            def report_candidate_progress(
                progress: dict[str, object],
                candidate_id: str = candidate.candidate_id,
            ) -> None:
                _candidate_progress(
                    state_path,
                    state,
                    candidate_id,
                    progress,
                    checkpoint,
                )

            result = adapter.train(
                _candidate_training_config(
                    candidate,
                    base_model=resolved_base_model,
                    data_dir=data_dir,
                    adapter_dir=adapter_dir,
                    train_example_count=train_example_count,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_seq_length=max_seq_length,
                    iteration_limit=short_run_iterations,
                ),
                report_candidate_progress,
            )
            if result.best_checkpoint is not None and result.best_checkpoint.is_file():
                _replace_adapter_weights(
                    result.best_checkpoint,
                    adapter_dir / "adapter_model.safetensors",
                )
            validation_loss = _best_validation_loss(result)
            style_metrics = style_evaluator(adapter_dir)
            style_score = float(style_metrics.get("style_score", 0.0))
            semantic_passed = float(style_metrics.get("semantic_passed", 1.0))
            composite_fidelity = float(style_metrics.get("composite_fidelity", style_score))
            style_distance = float(style_metrics.get("style_distance", 1.0 - style_score))
            metrics: dict[str, object] = {
                "validation_loss": validation_loss,
                "style_score": style_score,
                "semantic_passed": semantic_passed,
                "composite_fidelity": composite_fidelity,
                "style_distance": style_distance,
                **style_metrics,
            }
            candidate_state.update(
                {
                    "status": "succeeded",
                    "metrics": metrics,
                    "peft_version": result.peft_version,
                    "validation_curve": list(result.validation_curve),
                    "early_stopped": result.early_stopped,
                    "stopped_iteration": result.stopped_iteration,
                    "best_validation_iteration": (result.best_validation_iteration),
                    "best_checkpoint_iteration": (result.best_checkpoint_iteration),
                    "approximation_steps": (
                        result.best_checkpoint_selection.approximation_steps
                        if result.best_checkpoint_selection is not None
                        else None
                    ),
                    "checkpoint_mapping_rule": result.checkpoint_mapping_rule,
                    "best_checkpoint": (
                        str(result.best_checkpoint)
                        if result.best_checkpoint is not None
                        else str(weights_path)
                    ),
                    "checkpoint_sha256": _stream_sha256(weights_path),
                    "config": candidate_configs[candidate.candidate_id]["candidate_short_run"],
                    "command": (
                        list(result.command)
                        if result.command
                        else _training_command(result.config_path)
                    ),
                }
            )
        except (
            PeftTrainingError,
            CandidateValidationError,
            CandidateResourceError,
        ) as error:
            candidate_state.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        _publish_search_state(state_path, state, checkpoint)

    succeeded = [
        value
        for value in candidate_states.values()
        if isinstance(value, dict) and value.get("status") == "succeeded"
    ]
    if not succeeded:
        state["stage"] = "all_candidates_failed"
        _publish_search_state(state_path, state, checkpoint)
        raise AllCandidatesFailedError("所有 LoRA 候选均训练失败")
    semantically_valid = [
        item for item in succeeded if _metric_float(_metrics(item), "semantic_passed", 0.0) >= 1.0
    ]
    selection_mode = "semantic_hard_gate"
    if not semantically_valid:
        # 对当前 4GB Linux 训练机，只有一个经过显存验证的候选可运行。120 个
        # micro-batch 的短跑只用于确认损失可收敛；若要求它通过完整的人格、事实和
        # 风格验收，会把“配置筛选”误当成“模型发布”。最终完整训练仍走全部验收门槛。
        if (
            allow_single_candidate_semantic_fallback
            and len(selected_candidates) == 1
            and len(succeeded) == 1
        ):
            ranked = sorted(
                succeeded,
                key=lambda item: (
                    _metric_float(_metrics(item), "validation_loss", math.inf),
                    str(item.get("candidate_id", "")),
                ),
            )
            selection_mode = "single_candidate_validation_loss_fallback"
        else:
            state["stage"] = "all_candidates_semantic_failed"
            _publish_search_state(state_path, state, checkpoint)
            raise AllCandidatesFailedError("所有 LoRA 候选均未通过语义硬门槛")
    else:
        ranked = sorted(
            semantically_valid,
            key=_candidate_sort_key,
        )
    best_candidate_id = str(ranked[0]["candidate_id"])
    best_candidate = next(
        candidate
        for candidate in selected_candidates
        if candidate.candidate_id == best_candidate_id
    )
    state["search_results"] = [
        {
            "rank": index,
            "candidate_id": item["candidate_id"],
            "metrics": item["metrics"],
        }
        for index, item in enumerate(ranked, start=1)
    ]
    state["best_candidate_id"] = best_candidate_id
    state["candidate_selection"] = {
        "mode": selection_mode,
        "short_run_semantic_gate_passed": bool(semantically_valid),
        "final_model_acceptance_required": True,
    }
    _publish_search_state(state_path, state, checkpoint)

    full_state_raw = state.get("full_training")
    full_state = full_state_raw if isinstance(full_state_raw, dict) else {}
    state["full_training"] = full_state
    if full_state.get("status") == "succeeded":
        expected_full_fingerprint = hashlib.sha256(
            candidate_configs[best_candidate_id]["best_candidate_full_run"].encode("utf-8")
        ).hexdigest()
        final_checkpoint = Path(str(full_state.get("best_checkpoint", "")))
        if _checkpoint_state_valid(
            full_state,
            final_checkpoint,
            expected_full_fingerprint,
            digest_key="final_adapter_sha256",
        ):
            return _search_result_from_state(state_path, state)
        _quarantine_directory(output_dir / "final")
        _quarantine_directory(output_dir / "full")
        full_state["status"] = "pending"

    full_dir = output_dir / "full"
    full_config = _candidate_training_config(
        best_candidate,
        base_model=resolved_base_model,
        data_dir=data_dir,
        adapter_dir=full_dir,
        train_example_count=train_example_count,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        preserve_checkpoints=full_checkpoint_style_evaluator is not None,
    )
    full_config_text = candidate_configs[best_candidate_id]["best_candidate_full_run"]
    full_config_fingerprint = hashlib.sha256(full_config_text.encode("utf-8")).hexdigest()
    recovered_full_result = _recover_completed_full_training_result(
        adapter=adapter,
        full_config=full_config,
        full_state=full_state,
        full_dir=full_dir,
        expected_config=full_config_text,
        expected_config_fingerprint=full_config_fingerprint,
        state_path=state_path,
        state=state,
        checkpoint=checkpoint,
    )
    if recovered_full_result is None:
        if full_state.get("status") in {"running", "failed"} and full_dir.exists():
            # optimizer/scheduler 没有可恢复快照时必须从干净基座重跑，但绝不能
            # 直接删除现场：配置/恢复代码的缺陷不应让已经花费数小时的权重失去
            # 取证或人工抢救机会。
            _archive_interrupted_training_directory(full_dir)
        full_state.update(
            {
                "status": "running",
                "resume_strategy": "restart_from_base",
                "optimizer_state_restored": False,
                "early_stopping_requested": early_stopping_patience > 0,
                "early_stopping_applied": False,
                "normalized_config": full_config_text,
                "config_fingerprint": full_config_fingerprint,
                # 该 run 已决定从干净基座重训，旧 run 的验证/风格审计不能混入
                # 新轨迹；旧现场已经独立归档以供追溯。
                "recovery_validation_curve": [],
                "checkpoint_style_evaluations": [],
                "checkpoint_selection_stage": None,
            }
        )
        state["stage"] = "full_training"
        _publish_search_state(state_path, state, checkpoint)
    try:

        def report_full_progress(progress: dict[str, object]) -> None:
            _full_progress(
                state_path,
                state,
                max_epochs,
                full_config.resolved_iterations,
                progress,
                checkpoint,
            )

        full_result = recovered_full_result or adapter.train(full_config, report_full_progress)
        validation_curve = list(full_result.validation_curve)
        best_selection = full_result.best_checkpoint_selection
        if best_selection is None:
            raise PeftTrainingError("完整训练没有可恢复的最佳 checkpoint")
        checkpoint_style_evaluations: list[dict[str, object]] = []
        if full_checkpoint_style_evaluator is not None:

            def save_checkpoint_style_evaluations(
                audits: list[dict[str, object]],
            ) -> None:
                # 每完成一个 checkpoint 的风格验收就持久化。Worker 退出后只会
                # 重试尚未写入的项目，不会重新训练，更不会重复已完成的验收。
                full_state["checkpoint_selection_stage"] = "evaluating_style"
                full_state["checkpoint_style_evaluations"] = audits
                _publish_search_state(state_path, state, checkpoint)

            existing_audits = full_state.get("checkpoint_style_evaluations")
            best_selection, checkpoint_style_evaluations = _select_full_checkpoint_by_style(
                output_dir=output_dir,
                full_dir=full_dir,
                validation_curve=validation_curve,
                fallback=best_selection,
                style_evaluator=full_checkpoint_style_evaluator,
                existing_audits=(existing_audits if isinstance(existing_audits, list) else None),
                on_audit=save_checkpoint_style_evaluations,
            )
        best_checkpoint = best_selection.checkpoint
        final_dir = output_dir / "final"
        restored_checkpoint = _restore_best_adapter(
            best_checkpoint.parent,
            final_dir,
            best_checkpoint,
            archive_existing=True,
        )
        evaluator = checkpoint_evaluator or adapter.evaluate_loss
        verified_test_loss = evaluator(full_config, final_dir)
        if not math.isfinite(verified_test_loss):
            raise PeftTrainingError("最佳 checkpoint 独立复验未返回有效 test loss")
        steps_per_epoch = full_config.steps_per_epoch
        completed_steps = full_config.resolved_iterations
        if full_result.stopped_iteration is not None:
            completed_steps = max(0, full_result.stopped_iteration - 1)
        completed_epochs = completed_steps // steps_per_epoch
        partial_epoch_steps = completed_steps % steps_per_epoch
        progress_epochs = completed_steps / steps_per_epoch
        full_state.update(
            {
                "status": "succeeded",
                "completed_epochs": completed_epochs,
                "completed_steps": completed_steps,
                "partial_epoch_steps": partial_epoch_steps,
                "progress_epochs": progress_epochs,
                "source_best_checkpoint": str(best_checkpoint),
                "best_checkpoint": str(restored_checkpoint),
                "adapter_dir": str(final_dir),
                "validation_curve": validation_curve,
                "best_validation_loss": best_selection.validation_loss,
                "selection_validation_loss": best_selection.validation_loss,
                "checkpoint_selection_strategy": (
                    "held_out_style_then_validation_v1"
                    if full_checkpoint_style_evaluator is not None
                    else "validation_loss"
                ),
                "checkpoint_style_evaluations": checkpoint_style_evaluations,
                "verified_test_loss": verified_test_loss,
                "verification_kind": "independent_test_loss",
                "config": candidate_configs[best_candidate_id]["best_candidate_full_run"],
                "command": (
                    list(full_result.command)
                    if full_result.command
                    else _training_command(full_result.config_path)
                ),
                "peft_version": full_result.peft_version,
                "seed": full_config.seed,
                "early_stopping_applied": full_result.early_stopped,
                "stopped_iteration": full_result.stopped_iteration,
                "best_validation_iteration": best_selection.validation_iteration,
                "best_checkpoint_iteration": best_selection.checkpoint_iteration,
                "approximation_steps": best_selection.approximation_steps,
                "checkpoint_mapping_rule": (full_result.checkpoint_mapping_rule),
                "final_adapter_sha256": _stream_sha256(restored_checkpoint),
            }
        )
        state["stage"] = "search_completed"
        _publish_search_state(state_path, state, checkpoint)
        # 已发布的 final adapter 和成功状态先成为恢复锚点；清理仅是节省空间，
        # 不能位于二者之间，否则进程在该窗口退出会丢失唯一可恢复 checkpoint。
        try:
            cleanup_generated_checkpoints(
                output_dir,
                include_latest=True,
                keep=(restored_checkpoint,),
            )
        except OSError as error:
            full_state["cleanup_warning"] = str(error)
            _publish_search_state(state_path, state, checkpoint)
    except Exception as error:
        full_state.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        state["stage"] = "full_training_failed"
        _publish_search_state(state_path, state, checkpoint)
        raise
    return _search_result_from_state(state_path, state)


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


def _candidate_training_config(
    candidate: LoraCandidate,
    *,
    base_model: str,
    data_dir: Path,
    adapter_dir: Path,
    train_example_count: int,
    batch_size: int,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 512,
    epochs: int = 1,
    early_stopping_patience: int = 1,
    resume_adapter_file: Path | None = None,
    iteration_limit: int | None = None,
    preserve_checkpoints: bool = False,
) -> PeftLoraConfig:
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(train_example_count / effective_batch_size))
    validation_interval = max(2, min(50, steps_per_epoch))
    return PeftLoraConfig(
        model=base_model,
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        iterations=(min(iteration_limit, steps_per_epoch) if iteration_limit is not None else None),
        train_example_count=0 if iteration_limit is not None else train_example_count,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        learning_rate=candidate.learning_rate,
        seed=candidate.seed,
        num_layers=candidate.num_layers,
        target_modules=candidate.target_modules,
        rank=candidate.rank,
        alpha=candidate.alpha,
        dropout=candidate.dropout,
        steps_per_evaluation=validation_interval,
        save_every=validation_interval,
        early_stopping_patience=early_stopping_patience,
        preserve_checkpoints=preserve_checkpoints,
        resume_adapter_file=resume_adapter_file,
    )


def _candidate_payload(candidate: LoraCandidate) -> dict[str, object]:
    payload = asdict(candidate)
    payload["target_modules"] = list(candidate.target_modules)
    payload["resume_adapter_file"] = None
    return payload


def _select_full_checkpoint_by_style(
    *,
    output_dir: Path,
    full_dir: Path,
    validation_curve: list[dict[str, float | int]],
    fallback: BestCheckpointSelection,
    style_evaluator: StyleEvaluator,
    existing_audits: list[dict[str, object]] | None = None,
    on_audit: Callable[[list[dict[str, object]]], None] | None = None,
) -> tuple[BestCheckpointSelection, list[dict[str, object]]]:
    """Select a full-run checkpoint using held-out conversational fidelity.

    Validation loss remains an audit metric and final tie-breaker. It must not
    override the objective metric: sounding like the target speaker while
    still satisfying the semantic/structure hard gates.
    """

    checkpoint_pattern = re.compile(r"^(?P<iteration>[0-9]{7})_adapter_model\.safetensors$")
    checkpoints = sorted(
        (
            path
            for path in full_dir.iterdir()
            if path.is_file() and checkpoint_pattern.fullmatch(path.name)
        ),
        key=lambda path: int(checkpoint_pattern.fullmatch(path.name).group("iteration")),  # type: ignore[union-attr]
    )
    selections_by_checkpoint: dict[Path, BestCheckpointSelection] = {}
    for point in validation_curve:
        validation_iteration = int(point["iteration"])
        if validation_iteration <= 1 or not checkpoints:
            continue
        target_iteration = validation_iteration - 1
        checkpoint = min(
            checkpoints,
            key=lambda path: (
                abs(
                    int(checkpoint_pattern.fullmatch(path.name).group("iteration"))  # type: ignore[union-attr]
                    - target_iteration
                ),
                int(checkpoint_pattern.fullmatch(path.name).group("iteration")),  # type: ignore[union-attr]
            ),
        )
        checkpoint_iteration = int(
            checkpoint_pattern.fullmatch(checkpoint.name).group("iteration")  # type: ignore[union-attr]
        )
        selection = BestCheckpointSelection(
            validation_loss=float(point["validation_loss"]),
            validation_iteration=validation_iteration,
            checkpoint_iteration=checkpoint_iteration,
            checkpoint=checkpoint,
            approximation_steps=abs(checkpoint_iteration - target_iteration),
            mapping_rule=fallback.mapping_rule,
        )
        previous = selections_by_checkpoint.get(checkpoint)
        if previous is None or (
            selection.approximation_steps,
            selection.validation_iteration,
        ) < (
            previous.approximation_steps,
            previous.validation_iteration,
        ):
            selections_by_checkpoint[checkpoint] = selection

    if not selections_by_checkpoint:
        selections_by_checkpoint[fallback.checkpoint] = fallback

    cached_audits: dict[int, dict[str, object]] = {}
    for audit in existing_audits or []:
        checkpoint_iteration = audit.get("checkpoint_iteration")
        metrics = audit.get("metrics")
        if isinstance(checkpoint_iteration, int) and isinstance(metrics, dict):
            cached_audits[checkpoint_iteration] = dict(audit)

    audits: list[dict[str, object]] = []
    valid: list[tuple[BestCheckpointSelection, dict[str, object]]] = []
    for selection in sorted(
        selections_by_checkpoint.values(),
        key=lambda item: item.checkpoint_iteration,
    ):
        cached = cached_audits.get(selection.checkpoint_iteration)
        if (
            cached is not None
            and cached.get("validation_iteration") == selection.validation_iteration
            and cached.get("validation_loss") == selection.validation_loss
            and isinstance(cached.get("metrics"), dict)
        ):
            audits.append(cached)
            cached_metrics = cast(dict[str, object], cached["metrics"])
            if _metric_float(cached_metrics, "semantic_passed", 0.0) >= 1.0:
                valid.append((selection, cached_metrics))
            continue
        staging = output_dir / f".checkpoint-style-eval-{selection.checkpoint_iteration}"
        try:
            _restore_best_adapter(full_dir, staging, selection.checkpoint)
            metrics = dict(style_evaluator(staging))
            audit: dict[str, object] = {
                "checkpoint_iteration": selection.checkpoint_iteration,
                "validation_iteration": selection.validation_iteration,
                "validation_loss": selection.validation_loss,
                "approximation_steps": selection.approximation_steps,
                "metrics": metrics,
            }
            audits.append(audit)
            if _metric_float(metrics, "semantic_passed", 0.0) >= 1.0:
                valid.append((selection, metrics))
        except Exception as error:
            audits.append(
                {
                    "checkpoint_iteration": selection.checkpoint_iteration,
                    "validation_iteration": selection.validation_iteration,
                    "validation_loss": selection.validation_loss,
                    "approximation_steps": selection.approximation_steps,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if on_audit is not None:
            on_audit(list(audits))

    if not valid:
        raise PeftTrainingError("完整训练的所有 checkpoint 均未通过语义与结构硬门槛")
    selected, _metrics_value = min(
        valid,
        key=lambda item: (
            -_metric_float(item[1], "composite_fidelity", 0.0),
            -_metric_float(item[1], "speaker_probability", 0.0),
            _metric_float(item[1], "style_distance", 1.0),
            item[0].validation_loss,
            item[0].checkpoint_iteration,
        ),
    )
    return selected, audits


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


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _best_validation_loss(result: TrainingResult) -> float:
    selection = result.best_checkpoint_selection
    if selection is None or not math.isfinite(selection.validation_loss):
        raise CandidateValidationError("PEFT 没有可发布 checkpoint 对应的验证损失")
    return selection.validation_loss


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


def _candidate_sort_key(
    candidate_state: dict[str, object],
) -> tuple[float, float, float, float, str]:
    metrics = _metrics(candidate_state)
    semantic_passed = metrics.get("semantic_passed")
    composite_fidelity = metrics.get("composite_fidelity")
    style_distance = metrics.get("style_distance")
    validation_loss = metrics.get("validation_loss")
    return (
        (-float(semantic_passed) if isinstance(semantic_passed, int | float) else 0.0),
        (-float(composite_fidelity) if isinstance(composite_fidelity, int | float) else math.inf),
        (float(style_distance) if isinstance(style_distance, int | float) else math.inf),
        (float(validation_loss) if isinstance(validation_loss, int | float) else math.inf),
        str(candidate_state.get("candidate_id", "")),
    )


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


def _resolve_base_model_identity(
    base_model: str,
    *,
    cache_path: Path,
) -> dict[str, object]:
    """解析不可变基座，并为全部加载文件计算流式摘要。"""

    path = Path(base_model)
    if path.is_dir():
        return build_local_model_identity(
            path,
            cache_path=cache_path,
            requested=base_model,
        )
    remote_match = re.fullmatch(
        r"(?:hf://)?(?P<repo>[^@]+)@(?P<commit>[0-9a-fA-F]{40})",
        base_model,
    )
    if remote_match is None:
        raise TrainingFingerprintMismatch(
            "远程 Hugging Face 基座必须显式指定 40 位不可变 commit revision"
        )
    repo_id = remote_match.group("repo")
    commit = remote_match.group("commit").lower()
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(repo_id=repo_id, revision=commit))
    except Exception as error:
        raise TrainingFingerprintMismatch(
            "无法解析并固定远程 Hugging Face 基座 revision"
        ) from error
    if snapshot.name.lower() != commit:
        raise TrainingFingerprintMismatch("Hugging Face 缓存未解析到请求的不可变 commit")
    return {
        "source": "huggingface",
        "requested": base_model,
        "resolved_path": str(snapshot.resolve()),
        "repo_id": repo_id,
        "commit": commit,
        "digest": commit,
        "files": [],
    }


def build_local_model_identity(
    path: Path,
    *,
    cache_path: Path,
    requested: str | None = None,
    hash_file: Callable[[Path], str] | None = None,
) -> dict[str, object]:
    canonical = path.resolve()
    items = sorted(candidate for candidate in canonical.rglob("*") if candidate.is_file())
    stats = [
        {
            "path": str(item.relative_to(canonical)),
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
            "ctime_ns": item.stat().st_ctime_ns,
            "inode": item.stat().st_ino,
            "mode": item.stat().st_mode,
            "device": item.stat().st_dev,
            # mtime/ctime/inode 都可能在同一文件系统时钟粒度内保持不变，或被
            # 显式恢复。轻量内容指纹让缓存不会把“同尺寸替换的模型权重”误判为
            # 同一基座；完整 SHA-256 仍只在签名变化时重新计算。
            "content_probe": _file_content_probe(item),
        }
        for item in items
    ]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == "base-digest-cache-v2"
            and cached.get("canonical_path") == str(canonical)
            and cached.get("stats") == stats
            and isinstance(cached.get("identity"), dict)
        ):
            return cast(dict[str, object], cached["identity"])
    digest_file = hash_file or _stream_sha256
    files: list[dict[str, object]] = []
    for item in items:
        files.append(
            {
                "path": str(item.relative_to(canonical)),
                "size": item.stat().st_size,
                "sha256": digest_file(item),
            }
        )
    if not files:
        raise TrainingFingerprintMismatch("本地基础模型目录为空")
    identity: dict[str, object] = {
        "source": "local",
        "requested": requested or str(path),
        "resolved_path": str(canonical),
        "files": files,
        "digest": _canonical_digest(files),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        cache_path,
        {
            "schema_version": "base-digest-cache-v2",
            "cache_semantics": "performance-only-content-sha-authoritative",
            "canonical_path": str(canonical),
            "stats": stats,
            "identity": identity,
        },
    )
    return identity


def _file_content_probe(path: Path, *, window_bytes: int = 4096) -> str:
    """返回用于缓存失效判断的首尾内容摘要，不替代完整权重摘要。"""

    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(window_bytes))
        if size > window_bytes:
            stream.seek(max(0, size - window_bytes))
            digest.update(stream.read(window_bytes))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _training_data_identity(data_dir: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for name in ("manifest.json", "train.jsonl", "valid.jsonl", "test.jsonl"):
        path = data_dir / name
        if not path.is_file():
            if name == "test.jsonl":
                continue
            raise TrainingFingerprintMismatch(f"训练数据缺少必需文件：{name}")
        # manifest.json is enriched with artifact metadata after training has
        # started. Its pre-training digest is already fenced separately by
        # declared_data_manifest_digest, so hashing the mutable copy here makes
        # an interrupted job reject its own otherwise identical dataset.
        if name == "manifest.json":
            continue
        files.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": _stream_sha256(path),
                "used_for_selection": name in {"train.jsonl", "valid.jsonl"},
            }
        )
    return {
        "files": files,
        "digest": _canonical_digest(files),
        "test_used_for_tuning": False,
    }


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _training_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _canonical_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DigitalHumanTrainingConfig(BaseModel):
    base_model: str
    iterations: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_seq_length: int = Field(default=512, ge=128, le=4096)
    maximum_event_ratio: float = Field(ge=0, le=1)
    style_transfer_ratio: float = Field(default=0.0, ge=0, le=1)
    recency_window_days: int = Field(default=0, ge=0)
    recency_multiplier: int = Field(default=1, ge=1)
    runtime_style_transfer_enabled: bool = False
    context_turns: int = Field(gt=0)
    training_protocol_version: str = (
        "persona-plain-text-private-chat-v19-runtime-style-transfer-aligned"
    )
    checkpoint_selection_version: str = "person-identity-held-out-v11-role-separated"
    reply_protocol_version: str = "persona-text-v1"
    memory_protocol_version: str = "evidence-layered-temporal-v3-one-pass-preference"
    memory_preference_training_enabled: bool = False
    memory_preference_iterations: int = Field(default=40, gt=0)
    human_blind_required: bool = False
    human_blind_minimum_ratings: int = Field(default=20, ge=20)
    human_blind_minimum_preference: float = Field(default=0.45, ge=0, le=1)
    behavioral_timezone_offset_minutes: int = Field(default=480, ge=-720, le=840)


def create_peft_training_handler(adapter: PeftLmAdapter) -> JobHandler:
    def handle(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_missing", "训练任务缺少 Worker 租约")
        payload = job.payload
        config = PeftLoraConfig(
            model=str(payload["model"]),
            data_dir=Path(str(payload["data_dir"])),
            adapter_dir=Path(str(payload["adapter_dir"])),
            iterations=int(payload.get("iterations", 600)),
            batch_size=int(payload.get("batch_size", 1)),
            learning_rate=float(payload.get("learning_rate", 1e-5)),
        )

        def save_progress(progress: dict[str, object]) -> None:
            service.checkpoint(job.id, progress, token=token)

        adapter.train(config, save_progress)

    return handle


def create_digital_human_training_handler(
    adapter: PeftLmAdapter,
    *,
    data_dir: Path,
    model_dir: Path,
    acceptance_runner: ModelAcceptanceRunner | None = None,
    identity_kernel_builder: EvidenceBackedIdentityKernelBuilder | None = None,
    required_base_model: str | None = None,
) -> JobHandler:
    if acceptance_runner is None:
        raise ValueError("数字人训练必须配置模型人格验收器")

    def handle(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_missing", "训练任务缺少 Worker 租约")
        confirmation = service.session.get(
            TimelineConfirmation,
            str(job.payload["confirmation_id"]),
        )
        if confirmation is None or confirmation.confirmation_fingerprint != job.payload.get(
            "confirmation_fingerprint"
        ):
            raise JobHandlerError(
                "training_confirmation_invalid",
                "时间轴确认记录不存在或指纹不匹配",
            )
        config = DigitalHumanTrainingConfig.model_validate(confirmation.config_snapshot)
        if (
            required_base_model is not None
            and Path(config.base_model).resolve() != Path(required_base_model).resolve()
        ):
            raise JobHandlerError(
                "training_base_model_mismatch",
                "高保真训练必须从 settings.training_base_model 的干净基座开始",
            )
        config_snapshot = config.model_dump()
        service.checkpoint(
            job.id,
            {"stage": "preparing_dataset", "progress": 0.05},
            token=token,
        )
        rows = list(
            service.session.execute(
                select(Message, Participant.name)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.project_id == confirmation.project_id,
                    Message.import_id == confirmation.import_id,
                    Participant.role.in_(("self", "target")),
                )
                .order_by(Message.timestamp, Message.source_id)
            )
        )
        target_participant = service.session.scalar(
            select(Participant).where(
                Participant.project_id == confirmation.project_id,
                Participant.role == "target",
            )
        )
        if not rows or target_participant is None:
            raise JobHandlerError("training_data_empty", "没有可用于训练的目标聊天")
        target_sender = target_participant.name
        active_at_start = service.session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.project_id == confirmation.project_id,
                ModelVersion.active.is_(True),
            )
            .order_by(ModelVersion.created_at.desc())
        )
        expected_active_model_id = active_at_start.id if active_at_start is not None else None
        # 旧 MLX/Qwen3-8B 版本可以保留在注册表中作为历史记录，但不能传给当前
        # Linux PEFT 推理器。它仍参与后续 CAS 替换，验收对照则安全回退到干净基座。
        active_for_comparison = (
            active_at_start
            if _is_peft_comparable_model(active_at_start, config.base_model)
            else None
        )
        imported_messages = [
            ImportedMessage(
                source_id=message.source_id,
                timestamp=message.timestamp,
                sender=sender,
                kind=_message_kind(message.kind),
                content=message.content,
                raw={
                    **message.raw,
                    **(
                        {"media_asset_id": message.media_asset_id}
                        if message.media_asset_id is not None
                        else {}
                    ),
                },
            )
            for message, sender in rows
        ]
        cutoff = max(message.timestamp for message, _ in rows)
        builder = DatasetBuilder(
            context_turns=config.context_turns,
            maximum_event_ratio=config.maximum_event_ratio,
            style_transfer_ratio=config.style_transfer_ratio,
            recency_window_days=config.recency_window_days,
            recency_multiplier=config.recency_multiplier,
            plain_text_supervision=(
                config.training_protocol_version.startswith("persona-plain-text")
            ),
        )
        examples = builder.build(
            imported_messages,
            target_sender=target_sender,
            cutoff=cutoff,
        )
        event_contexts = [
            _event_context(snapshot) for snapshot in confirmation.event_revision_snapshots
        ]
        examples = builder.augment_with_events(examples, event_contexts)
        examples.extend(
            builder.grounding_policy_examples(
                persona=target_sender,
                cutoff=cutoff,
            )
        )
        style_examples = [example for example in examples if example.kind == "chat"]
        dataset_path = (
            data_dir / "projects" / confirmation.project_id / "training" / confirmation.id
        )
        manifest = builder.write_lora_dataset(
            examples,
            dataset_path,
            target_sender,
            cutoff,
            confirmation_id=confirmation.id,
            analysis_run_id=confirmation.analysis_run_id,
            node_snapshot_hash=_node_snapshot_hash(confirmation),
            base_model=config.base_model,
            training_config=config_snapshot,
        )
        train_boundary = manifest.split_time_boundaries.get("train", {})
        train_end_value = train_boundary.get("end")
        if not isinstance(train_end_value, str) or not train_end_value:
            raise JobHandlerError(
                "training_split_invalid",
                "训练清单缺少 train 时间边界",
            )
        policy_cutoff = datetime.fromisoformat(train_end_value)
        policy_messages = [
            item
            for item in imported_messages
            if _same_timeline(item.timestamp, policy_cutoff) <= policy_cutoff
        ]
        if not policy_messages:
            raise JobHandlerError(
                "training_split_invalid",
                "train 时间边界内没有行为策略样本",
            )
        expression_policy = build_expression_policy(
            policy_messages,
            target_sender=target_sender,
        )
        conversation_action_policy = build_conversation_action_policy(
            policy_messages,
            target_sender=target_sender,
        )
        media_behavior_policy = build_media_behavior_policy(
            policy_messages,
            target_sender=target_sender,
        )
        try:
            sticker_embedder = LocalChineseEmbedder(
                model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
            )
        except Exception:
            sticker_embedder = None
        sticker_policy = build_sticker_policy_from_database(
            service.session,
            project_id=confirmation.project_id,
            target_id=target_participant.id,
            cutoff=policy_cutoff,
            branch_time=policy_cutoff,
            import_id=confirmation.import_id,
            embedder=sticker_embedder,
        )
        sticker_valid_metrics: dict[str, float] = {}
        sticker_test_metrics: dict[str, float] = {}
        if sticker_policy.enabled:
            split_boundaries = manifest.split_time_boundaries
            valid_boundary = split_boundaries.get("valid", {})
            test_boundary = split_boundaries.get("test", {})
            valid_start = valid_boundary.get("start")
            valid_end = valid_boundary.get("end")
            test_start = test_boundary.get("start")
            test_end = test_boundary.get("end")
            if not all(
                isinstance(value, str) and value
                for value in (valid_start, valid_end, test_start, test_end)
            ):
                raise JobHandlerError(
                    "training_split_invalid",
                    "训练清单缺少 valid/test 表情策略边界",
                )
            sticker_valid_cases = build_sticker_evaluation_cases_from_database(
                service.session,
                project_id=confirmation.project_id,
                target_id=target_participant.id,
                start=datetime.fromisoformat(valid_start),
                end=datetime.fromisoformat(valid_end),
                split="valid",
                import_id=confirmation.import_id,
            )
            sticker_test_cases = build_sticker_evaluation_cases_from_database(
                service.session,
                project_id=confirmation.project_id,
                target_id=target_participant.id,
                start=datetime.fromisoformat(test_start),
                end=datetime.fromisoformat(test_end),
                split="test",
                import_id=confirmation.import_id,
            )
            has_sticker_positive = any(
                case.actual_asset_id is not None for case in sticker_valid_cases
            )
            has_sticker_negative = any(case.actual_asset_id is None for case in sticker_valid_cases)
        else:
            has_sticker_positive = False
            has_sticker_negative = False
        if has_sticker_positive and has_sticker_negative:
            sticker_policy, sticker_valid_metrics = tune_sticker_policy_on_valid(
                sticker_policy,
                sticker_valid_cases,
                parameter_grid=tuple(
                    {
                        "minimum_similarity": minimum_similarity,
                        "modality_threshold": modality_threshold,
                    }
                    for minimum_similarity in (0.05, 0.12, 0.2)
                    for modality_threshold in (0.38, 0.4, 0.42, 0.44, 0.46, 0.5)
                ),
            )
            sticker_test_metrics = evaluate_sticker_policy(
                sticker_policy,
                sticker_test_cases,
                expected_split="test",
            )
        kernel_style_examples = [
            example
            for example in style_examples
            if _same_timeline(example.target_at, policy_cutoff) <= policy_cutoff
        ]
        adapter_path = model_dir / confirmation.project_id / job.id
        service.checkpoint(
            job.id,
            {
                "stage": "loading_model",
                "progress": 0.15,
                "dataset_hash": manifest.dataset_hash,
                "example_count": manifest.example_count,
                "dataset_coverage": {
                    "target_message_count": manifest.target_message_count,
                    "covered_target_message_count": (manifest.covered_target_message_count),
                    "filtered_target_message_count": (manifest.filtered_target_message_count),
                    "filter_reason_counts": manifest.filter_reason_counts,
                    "split_counts": manifest.split_counts,
                },
            },
            token=token,
        )
        latest_metrics: dict[str, float] = {}

        def save_progress(progress: dict[str, object]) -> None:
            loss = progress.get("loss")
            if isinstance(loss, int | float):
                latest_metrics["train_loss"] = float(loss)
            validation_loss = progress.get("validation_loss")
            if isinstance(validation_loss, int | float):
                latest_metrics["validation_loss"] = float(validation_loss)
            raw_iteration = progress.get("iteration", 0)
            iteration = raw_iteration if isinstance(raw_iteration, int) else 0
            total = config.iterations
            service.checkpoint(
                job.id,
                {
                    "stage": "training",
                    "progress": 0.15 + 0.7 * min(1.0, iteration / max(1, total)),
                    "iteration": iteration,
                    "total_iterations": total,
                    **latest_metrics,
                },
                token=token,
            )

        manifest_path = dataset_path / "manifest.json"
        data_manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        existing_search_state_path = adapter_path / "search-state.json"
        if existing_search_state_path.is_file():
            try:
                existing_search_state = json.loads(
                    existing_search_state_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                existing_search_state = {}
            fingerprint_payload = existing_search_state.get("fingerprint_payload")
            declared_digest = (
                fingerprint_payload.get("declared_data_manifest_digest")
                if isinstance(fingerprint_payload, dict)
                else None
            )
            # The manifest is rewritten and later enriched during a resumed
            # training job. Its original digest is only a declaration fence;
            # train/valid/test content, protocol versions, model identity and
            # normalized YAML are independently hashed by the search state.
            # Reuse the first declaration so the job can validate those
            # authoritative immutable inputs instead of rejecting its own
            # regenerated manifest.
            if isinstance(declared_digest, str) and declared_digest:
                data_manifest_digest = declared_digest
        train_example_count = max(1, manifest.split_counts.get("train", 0))
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        steps_per_epoch = max(
            1,
            math.ceil(train_example_count / effective_batch_size),
        )
        maximum_epochs = max(
            1,
            min(3, math.ceil(config.iterations / steps_per_epoch)),
        )
        # ``PeftLmAdapter`` 的训练循环单位是 micro-batch，不是 optimizer step。
        # 进度总数需与其一致，否则 UI 会把未消费的大部分训练样本误报为已完成。
        maximum_iterations = math.ceil(train_example_count / config.batch_size) * maximum_epochs

        def _evaluate_candidate(
            candidate_path: Path,
        ) -> dict[str, float | int | str]:
            candidate_report = acceptance_runner.run(
                base_model=config.base_model,
                adapter_path=str(candidate_path),
                persona=target_sender,
                cutoff=cutoff.isoformat(),
                style_profile=manifest.style_profile,
                reply_protocol=resolve_reply_protocol(config_snapshot),
            )
            semantic_passed = (
                candidate_report.case_count > 0
                and candidate_report.passed_count == candidate_report.case_count
                and candidate_report.structure_failures == 0
                and candidate_report.forbidden_fact_failures == 0
                and candidate_report.raw_output_failures == 0
            )
            valid_metrics: dict[str, float | int | str]
            if semantic_passed:
                valid_metrics = acceptance_runner.evaluate_valid_style(
                    data_dir=dataset_path,
                    base_model=config.base_model,
                    adapter_path=str(candidate_path),
                    persona=target_sender,
                    sample_count=20,
                    style_profile=manifest.style_profile,
                    training_task="style_transfer",
                )
            else:
                valid_metrics = {
                    "valid_sample_count": 0,
                    "valid_structure_failures": candidate_report.structure_failures,
                    "valid_grounding_failures": candidate_report.forbidden_fact_failures,
                    "valid_grounding_failure_rate": 1.0,
                    "valid_register_violations": 0,
                    "valid_sample_hash": "",
                    "style_distance": 1.0,
                    "speaker_probability": 0.0,
                    "composite_fidelity": 0.0,
                }
            valid_structure_failures = valid_metrics.get(
                "valid_structure_failures",
                0,
            )
            semantic_passed = semantic_passed and (
                isinstance(valid_structure_failures, int | float) and valid_structure_failures == 0
            )
            valid_register_violations = valid_metrics.get(
                "valid_register_violations",
                0,
            )
            semantic_passed = semantic_passed and (
                isinstance(valid_register_violations, int | float)
                and valid_register_violations == 0
            )
            valid_grounding_failure_rate = valid_metrics.get(
                "valid_grounding_failure_rate",
                1.0,
            )
            semantic_passed = semantic_passed and (
                isinstance(valid_grounding_failure_rate, int | float)
                and valid_grounding_failure_rate <= MAXIMUM_STYLE_TRANSFER_FALLBACK_RATE
            )
            paired_similarity = valid_metrics.get("paired_response_similarity", 0.0)
            semantic_passed = semantic_passed and (
                isinstance(paired_similarity, int | float)
                and paired_similarity >= MINIMUM_PAIRED_RESPONSE_SIMILARITY
            )
            return {
                "semantic_passed": float(semantic_passed),
                "style_score": float(valid_metrics["composite_fidelity"]),
                "acceptance_pass_rate": (
                    candidate_report.passed_count / candidate_report.case_count
                    if candidate_report.case_count
                    else 0.0
                ),
                "style_feature_failures": float(candidate_report.style_feature_failures),
                "structure_failures": float(candidate_report.structure_failures),
                "fixed_structure_failure_audit": json.dumps(
                    candidate_report.structure_failure_audit,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "fact_failures": float(candidate_report.forbidden_fact_failures),
                **valid_metrics,
            }

        def evaluate_candidate(
            candidate_path: Path,
        ) -> dict[str, float | int | str]:
            try:
                return _evaluate_candidate(candidate_path)
            finally:
                # 验收器的 Linux 推理运行时会缓存 4-bit 基座。在 4GB GPU 上，若
                # 不在短跑/检查点评估后释放它，下一次 QLoRA 初始化必然没有余量。
                closer = getattr(acceptance_runner, "close", None)
                if callable(closer):
                    closer()

        def save_search_state(state: dict[str, Any]) -> None:
            candidates = state.get("candidates")
            full = state.get("full_training")
            progress = _search_job_progress(
                state,
                maximum_iterations=maximum_iterations,
            )
            service.checkpoint(
                job.id,
                {
                    "stage": str(state.get("stage", "candidate_search")),
                    "progress": progress,
                    "dataset_hash": manifest.dataset_hash,
                    "data_manifest_digest": data_manifest_digest,
                    "candidate_runs": candidates if isinstance(candidates, dict) else {},
                    "search_results": state.get("search_results", []),
                    "best_candidate_id": state.get("best_candidate_id"),
                    "full_training": full if isinstance(full, dict) else {},
                },
                token=token,
            )

        persona_training = config.training_protocol_version.startswith("persona-plain-text")
        compact_linux_candidate = persona_training and _is_compact_qwen3_1_7b(config.base_model)
        selected_candidates = (
            compact_persona_candidates()
            if compact_linux_candidate
            else (high_capacity_persona_candidates() if persona_training else None)
        )
        # GTX 1650 4GB 只能安全运行这一项候选。短跑在这里是配置筛选，完整训练
        # 完成后仍必须通过固定回归、风格、安全和人工盲测，才可能被发布。
        allow_single_candidate_semantic_fallback = (
            compact_linux_candidate
            and selected_candidates is not None
            and len(selected_candidates) == 1
            and selected_candidates[0].candidate_id == "compact-r16-qv-all"
        )
        try:
            search_result = run_lora_candidate_search(
                adapter,
                base_model=config.base_model,
                data_dir=dataset_path,
                output_dir=adapter_path,
                data_manifest_digest=data_manifest_digest,
                protocol_versions={
                    "training": config.training_protocol_version,
                    "reply": config.reply_protocol_version,
                    "memory": config.memory_protocol_version,
                    "dataset": manifest.protocol_version,
                    "style_features": STYLE_FEATURE_SCHEMA_VERSION,
                },
                train_example_count=train_example_count,
                batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_seq_length=config.max_seq_length,
                style_evaluator=evaluate_candidate,
                full_checkpoint_style_evaluator=evaluate_candidate,
                max_epochs=maximum_epochs,
                early_stopping_patience=2,
                candidates=selected_candidates,
                allow_single_candidate_semantic_fallback=(allow_single_candidate_semantic_fallback),
                checkpoint=save_search_state,
            )
        except TrainingFingerprintMismatch as error:
            raise JobHandlerError(
                "training_fingerprint_mismatch",
                str(error),
            ) from error
        except AllCandidatesFailedError as error:
            raise JobHandlerError(
                "training_candidates_failed",
                str(error),
            ) from error
        except PeftTrainingError as error:
            raise JobHandlerError(
                "training_full_run_failed",
                "最佳候选完整训练失败，旧模型保持启用",
            ) from error
        adapter_path = search_result.adapter_dir
        latest_metrics["validation_loss"] = search_result.best_validation_loss
        if config.memory_preference_training_enabled:
            service.checkpoint(
                job.id,
                {"stage": "memory_preference_training", "progress": 0.84},
                token=token,
            )
            preference_data_path = dataset_path / "memory-authority-preference"
            preference_manifest = build_persona_preference_dataset(
                data_dir=dataset_path,
                output_dir=preference_data_path,
                generator=acceptance_runner.negative_generator,
                base_model=config.base_model,
                adapter_path=str(adapter_path),
                sample_count=128,
                negative_mode="memory_authority_mixed",
                negative_variants_per_source=4,
            )
            if preference_manifest.train_count < 1:
                raise JobHandlerError(
                    "memory_preference_data_empty",
                    "记忆可信度偏好训练没有可用的真人锚点",
                )
            total_preference_iterations = preference_training_iterations(
                preference_manifest.train_count,
            )
            preference_plan = plan_preference_training(
                adapter_path.parent,
                source_adapter_file=adapter_path / "adapter_model.safetensors",
                total_iterations=total_preference_iterations,
            )
            preference_result = None
            if preference_plan.cached_adapter_dir is None:
                preference_result = run_persona_preference_training(
                    PreferenceTrainingConfig(
                        base_model=config.base_model,
                        initial_adapter_file=str(preference_plan.initial_adapter_file),
                        data_dir=str(preference_data_path),
                        output_dir=str(preference_plan.output_dir),
                        iterations=preference_plan.iterations,
                        max_seq_length=config.max_seq_length,
                    )
                )
                adapter_path = preference_plan.output_dir
            else:
                adapter_path = preference_plan.cached_adapter_dir
            latest_metrics.update(
                {
                    "memory_preference_train_pairs": float(
                        preference_result.train_pair_count
                        if preference_result is not None
                        else preference_manifest.train_count
                    ),
                    "memory_preference_valid_pairs": float(
                        preference_result.valid_pair_count
                        if preference_result is not None
                        else preference_manifest.valid_count
                    ),
                    "memory_preference_iterations": float(total_preference_iterations),
                    "memory_preference_resumed_from_iteration": float(
                        preference_plan.resumed_from_iteration
                    ),
                }
            )
        service.checkpoint(
            job.id,
            {"stage": "model_acceptance", "progress": 0.88},
            token=token,
        )
        kernel_builder = identity_kernel_builder or EvidenceBackedIdentityKernelBuilder(
            default_timezone_offset_minutes=(config.behavioral_timezone_offset_minutes)
        )
        kernel_proposal = kernel_builder.build(
            persona=target_sender,
            examples=kernel_style_examples,
            event_contexts=event_contexts,
        )
        report = acceptance_runner.run(
            base_model=config.base_model,
            adapter_path=str(adapter_path),
            persona=target_sender,
            cutoff=cutoff.isoformat(),
            style_profile=kernel_proposal.style_profile,
            reply_protocol=resolve_reply_protocol(config_snapshot),
        )
        base_report = acceptance_runner.run(
            base_model=config.base_model,
            adapter_path="",
            persona=target_sender,
            cutoff=cutoff.isoformat(),
            style_profile=kernel_proposal.style_profile,
            reply_protocol=resolve_reply_protocol(manifest.training_config),
        )
        active_report = (
            acceptance_runner.run(
                base_model=active_for_comparison.base_model,
                adapter_path=active_for_comparison.adapter_path,
                persona=target_sender,
                cutoff=cutoff.isoformat(),
                style_profile=kernel_proposal.style_profile,
                reply_protocol=resolve_reply_protocol(active_for_comparison.training_config),
            )
            if active_for_comparison is not None
            else base_report
        )
        candidate_style_metrics = acceptance_runner.evaluate_valid_style(
            data_dir=dataset_path,
            base_model=config.base_model,
            adapter_path=str(adapter_path),
            persona=target_sender,
            sample_count=20,
            style_profile=kernel_proposal.style_profile,
            expressed_preferences=kernel_proposal.expressed_preferences,
            training_task="style_transfer",
        )
        base_style_metrics = acceptance_runner.evaluate_valid_style(
            data_dir=dataset_path,
            base_model=config.base_model,
            adapter_path="",
            persona=target_sender,
            sample_count=20,
            style_profile=kernel_proposal.style_profile,
            expressed_preferences=kernel_proposal.expressed_preferences,
            training_task="style_transfer",
        )
        active_style_metrics = (
            acceptance_runner.evaluate_valid_style(
                data_dir=dataset_path,
                base_model=active_for_comparison.base_model,
                adapter_path=active_for_comparison.adapter_path,
                persona=target_sender,
                sample_count=20,
                style_profile=kernel_proposal.style_profile,
                expressed_preferences=kernel_proposal.expressed_preferences,
                training_task="style_transfer",
            )
            if active_for_comparison is not None
            else base_style_metrics
        )
        direct_diagnostic_keys = {
            "paired_response_similarity": "direct_response_paired_similarity",
            "valid_sample_audit": "direct_response_sample_audit",
            "valid_sample_hash": "direct_response_sample_hash",
            "valid_grounding_failure_rate": ("direct_response_grounding_failure_rate"),
            "excluded_unanswerable_current_state_count": (
                "direct_response_excluded_current_state_count"
            ),
        }

        def attach_direct_response_diagnostics(
            style_metrics: dict[str, float | int | str],
            *,
            diagnostic_base_model: str,
            diagnostic_adapter_path: str,
        ) -> None:
            direct = acceptance_runner.evaluate_valid_style(
                data_dir=dataset_path,
                base_model=diagnostic_base_model,
                adapter_path=diagnostic_adapter_path,
                persona=target_sender,
                sample_count=20,
                style_profile=kernel_proposal.style_profile,
                expressed_preferences=kernel_proposal.expressed_preferences,
                training_task="conversation",
            )
            style_metrics.update(
                {
                    destination: direct[source]
                    for source, destination in direct_diagnostic_keys.items()
                    if source in direct
                }
            )

        attach_direct_response_diagnostics(
            candidate_style_metrics,
            diagnostic_base_model=config.base_model,
            diagnostic_adapter_path=str(adapter_path),
        )
        attach_direct_response_diagnostics(
            base_style_metrics,
            diagnostic_base_model=config.base_model,
            diagnostic_adapter_path="",
        )
        if active_for_comparison is not None:
            attach_direct_response_diagnostics(
                active_style_metrics,
                diagnostic_base_model=active_for_comparison.base_model,
                diagnostic_adapter_path=active_for_comparison.adapter_path,
            )
        else:
            active_style_metrics.update(
                {
                    key: value
                    for key, value in base_style_metrics.items()
                    if key.startswith("direct_response_")
                }
            )
        causal_evaluator = getattr(
            acceptance_runner,
            "evaluate_memory_causality",
            None,
        )
        if callable(causal_evaluator):
            candidate_style_metrics.update(
                causal_evaluator(
                    base_model=config.base_model,
                    adapter_path=str(adapter_path),
                    persona=target_sender,
                    style_profile=kernel_proposal.style_profile,
                )
            )
            base_style_metrics.update(
                causal_evaluator(
                    base_model=config.base_model,
                    adapter_path="",
                    persona=target_sender,
                    style_profile=kernel_proposal.style_profile,
                )
            )
            if active_for_comparison is not None:
                active_style_metrics.update(
                    causal_evaluator(
                        base_model=active_for_comparison.base_model,
                        adapter_path=active_for_comparison.adapter_path,
                        persona=target_sender,
                        style_profile=kernel_proposal.style_profile,
                    )
                )
            else:
                active_style_metrics.update(
                    {
                        key: value
                        for key, value in base_style_metrics.items()
                        if key.startswith("memory_causal_")
                    }
                )
        activation_gate = evaluate_activation_gates(
            _model_gate_snapshot(
                f"training-job:{job.id}",
                report,
                candidate_style_metrics,
            ),
            _model_gate_snapshot("base", base_report, base_style_metrics),
            _model_gate_snapshot(
                active_for_comparison.id if active_for_comparison is not None else "base",
                active_report,
                active_style_metrics,
            ),
        )
        comparison_scores = {
            "candidate": _acceptance_score(report),
            "base": _acceptance_score(base_report),
            "active": _acceptance_score(active_report),
        }
        comparison_reasons: list[str] = []
        if comparison_scores["candidate"] < comparison_scores["base"]:
            comparison_reasons.append("候选语义验收分数劣于干净基座")
        if comparison_scores["candidate"] < comparison_scores["active"]:
            comparison_reasons.append("候选语义验收分数劣于当前活动模型")
        sticker_gate_passed = (
            not sticker_policy.enabled or float(sticker_test_metrics.get("modality_f1", 0.0)) > 0
        )
        gate_passed = (
            report.passed
            and not comparison_reasons
            and activation_gate.passed
            and sticker_gate_passed
        )
        gate_failure_reasons = [
            *comparison_reasons,
            *([] if report.passed else ["真实本地输出未通过固定回归"]),
            *activation_gate.failure_reasons,
            *([] if sticker_gate_passed else ["表情模态在冻结 test 集完全失效"]),
        ]
        latest_metrics.update(
            {
                "acceptance_pass_rate": (
                    report.passed_count / report.case_count if report.case_count else 0.0
                ),
                "acceptance_structure_failures": float(report.structure_failures),
                "acceptance_fact_failures": float(report.forbidden_fact_failures),
                "acceptance_raw_output_failures": float(report.raw_output_failures),
                "base_acceptance_score": comparison_scores["base"],
                "active_acceptance_score": comparison_scores["active"],
                "candidate_acceptance_score": comparison_scores["candidate"],
                "valid_style_distance": float(candidate_style_metrics["style_distance"]),
                "valid_speaker_probability": float(candidate_style_metrics["speaker_probability"]),
                "valid_composite_fidelity": float(candidate_style_metrics["composite_fidelity"]),
            }
        )
        acceptance_payload = {
            "schema_version": "moonlightbox.high-fidelity-acceptance.v1",
            "passed": gate_passed,
            "candidate": asdict(report),
            "base": asdict(base_report),
            "active": asdict(active_report),
            "model_ids": {
                "base": config.base_model,
                "active": (
                    active_for_comparison.id if active_for_comparison is not None else "base"
                ),
                "candidate": f"training-job:{job.id}",
            },
            "comparisons": comparison_scores,
            "active_comparison": {
                "used_active_model": active_for_comparison is not None,
                "replaced_active_model_id": expected_active_model_id,
                "fallback": (
                    "base_model"
                    if active_at_start is not None and active_for_comparison is None
                    else None
                ),
            },
            "activation_gate": asdict(activation_gate),
            "valid_style": {
                "candidate": candidate_style_metrics,
                "base": base_style_metrics,
                "active": active_style_metrics,
            },
            "sticker_policy_valid_tuning": {
                "parameters": dict(sticker_policy.parameters),
                "metrics": sticker_valid_metrics,
                "split": "valid",
            },
            "sticker_policy_test_evaluation": {
                "parameters_frozen_from": "valid",
                "metrics": sticker_test_metrics,
                "split": "test",
                "test_used_for_tuning": False,
            },
            "failure_reasons": gate_failure_reasons,
        }
        acceptance_payload["report_path"] = str(
            search_result.state_path.parent / "acceptance-report.json"
        )
        _atomic_write_json(
            search_result.state_path.parent / "acceptance-report.json",
            acceptance_payload,
        )
        service.checkpoint(
            job.id,
            {
                "stage": "model_acceptance",
                "progress": 0.9,
                "acceptance": {
                    "passed": gate_passed,
                    "semantic_passed": (
                        report.structure_failures == 0
                        and report.forbidden_fact_failures == 0
                        and report.raw_output_failures == 0
                    ),
                    "style_passed": report.style_feature_failures == 0,
                    "sticker_status": ("passed" if sticker_gate_passed else "failed"),
                    "memorization_status": "not_evaluated_in_fixed_regression",
                    "comparisons": comparison_scores,
                    "failure_reasons": gate_failure_reasons,
                    "report_path": acceptance_payload["report_path"],
                },
            },
            token=token,
        )
        if not gate_passed:
            raise JobHandlerError(
                "training_quality_gate_failed",
                "；".join(gate_failure_reasons) + "，旧模型保持启用",
            )
        quality = evaluate_training_quality(style_examples, cutoff, latest_metrics)
        if not quality["passed"]:
            raise JobHandlerError(
                "training_quality_gate_failed",
                "新模型未通过自动质量门槛，旧模型保持启用",
            )
        raw_multi_bubble_ratio = quality["multi_bubble_ratio"]
        multi_bubble_ratio = (
            float(raw_multi_bubble_ratio)
            if isinstance(raw_multi_bubble_ratio, int | float)
            else 0.0
        )
        latest_metrics.update(
            {
                "multi_bubble_ratio": multi_bubble_ratio,
                "quality_gate_passed": 1.0,
            }
        )
        blind_cases: list[dict[str, object]] = []
        if config.human_blind_required:
            build_blind_cases = getattr(
                acceptance_runner,
                "build_human_blind_cases",
                None,
            )
            if not callable(build_blind_cases):
                raise JobHandlerError(
                    "human_blind_cases_unavailable",
                    "验收器不能生成真人盲测案例，旧模型保持启用",
                )
            try:
                blind_cases = build_blind_cases(
                    data_dir=dataset_path,
                    base_model=config.base_model,
                    adapter_path=str(adapter_path),
                    sample_count=config.human_blind_minimum_ratings,
                )
            except (RuntimeError, ValueError) as error:
                raise JobHandlerError(
                    "human_blind_cases_unavailable",
                    "真人盲测案例生成失败，旧模型保持启用",
                ) from error
        service.checkpoint(
            job.id,
            {"stage": "activating_model", "progress": 0.95},
            token=token,
        )
        registry = ModelRegistry(service.session)
        version = service.session.scalar(
            select(ModelVersion).where(ModelVersion.training_job_id == job.id)
        )
        if version is None:
            version = registry.publish_and_activate(
                project_id=confirmation.project_id,
                base_model=config.base_model,
                adapter_path=str(adapter_path),
                dataset_hash=manifest.dataset_hash,
                metrics=latest_metrics,
                timeline_confirmation_id=confirmation.id,
                training_job_id=job.id,
                training_config={
                    **config_snapshot,
                    "training_metadata": json.loads(
                        search_result.state_path.read_text(encoding="utf-8")
                    ),
                    "data_manifest_digest": data_manifest_digest,
                    "data_manifest": asdict(manifest),
                    "expression_policy": expression_policy.to_metadata(),
                    "conversation_action_policy": (conversation_action_policy.to_metadata()),
                    "media_behavior_policy": media_behavior_policy.to_metadata(),
                    "sticker_policy_valid_metrics": sticker_valid_metrics,
                    "sticker_policy_test_metrics": sticker_test_metrics,
                    "behavioral_rhythm": kernel_proposal.behavioral_rhythm,
                },
                kernel_proposal=kernel_proposal,
                evidence_message_ids=list(
                    dict.fromkeys(
                        source_id
                        for example in examples
                        for source_id in example.source_ids
                        if not source_id.startswith("policy:")
                    )
                ),
                acceptance_report=acceptance_payload,
                sticker_policy=sticker_policy.to_metadata(),
                expected_active_model_id=expected_active_model_id,
                commit=False,
            )
        if config.human_blind_required:
            existing_study = service.session.scalar(
                select(HumanBlindStudy).where(
                    HumanBlindStudy.model_version_id == version.id,
                    HumanBlindStudy.status == "open",
                )
            )
            if existing_study is None:
                HumanBlindStudyService(service.session).create(
                    project_id=confirmation.project_id,
                    model_version_id=version.id,
                    cases=blind_cases,
                    minimum_ratings=config.human_blind_minimum_ratings,
                    minimum_preference=config.human_blind_minimum_preference,
                    seed=0,
                    commit=False,
                )
        confirmation.status = "trained"
        completed = service.succeed_in_transaction(
            job.id,
            token=token,
            checkpoint={
                "stage": "completed",
                "progress": 1.0,
                "model_version_id": version.id,
                "model_status": version.status,
                "dataset_hash": manifest.dataset_hash,
                **latest_metrics,
            },
        )
        if not completed:
            service.session.rollback()
            raise JobHandlerError("job_lease_lost", "训练任务租约已失效")
        service.session.commit()

    return handle


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


def _acceptance_score(report: object) -> float:
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


def _model_gate_snapshot(
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


def _event_context(snapshot: dict[str, object]) -> ConfirmedEventContext:
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


def _same_timeline(value: datetime, reference: datetime) -> datetime:
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
