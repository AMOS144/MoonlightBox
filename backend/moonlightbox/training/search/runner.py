import hashlib
import math
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

from moonlightbox.training.model_identity import (
    _canonical_digest,
    _resolve_base_model_identity,
    _stream_sha256,
    _training_data_identity,
    _training_environment,
)
from moonlightbox.training.peft_adapter import (
    BestCheckpointSelection,
    LoraCandidate,
    PeftLmAdapter,
    PeftLoraConfig,
    PeftTrainingError,
    TrainingResult,
    cleanup_generated_checkpoints,
    default_lora_candidates,
    normalized_peft_config,
)
from moonlightbox.training.search.lock import (
    AllCandidatesFailedError,
    CandidateResourceError,
    CandidateValidationError,
    SearchOutputLock,
    SearchStateStore,
    TrainingFingerprintMismatch,
)
from moonlightbox.training.search.state import (
    CheckpointEvaluator,
    LoraSearchResult,
    SearchCheckpoint,
    StyleEvaluator,
    _archive_interrupted_training_directory,
    _candidate_progress,
    _checkpoint_state_valid,
    _full_progress,
    _load_or_create_search_state,
    _metric_float,
    _metrics,
    _migrate_legacy_full_checkpoint_preservation_fingerprint,
    _publish_search_state,
    _quarantine_directory,
    _recover_completed_full_training_result,
    _replace_adapter_weights,
    _restore_best_adapter,
    _search_result_from_state,
    _training_command,
)


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


def _best_validation_loss(result: TrainingResult) -> float:
    selection = result.best_checkpoint_selection
    if selection is None or not math.isfinite(selection.validation_loss):
        raise CandidateValidationError("PEFT 没有可发布 checkpoint 对应的验证损失")
    return selection.validation_loss


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
