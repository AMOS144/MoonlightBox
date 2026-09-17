import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from moonlightbox.branches.identity import (
    EvidenceBackedIdentityKernelBuilder,
)
from moonlightbox.embeddings import LocalChineseEmbedder
from moonlightbox.evaluation.blind_service import HumanBlindStudyService
from moonlightbox.evaluation.models import HumanBlindStudy
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.types import ImportedMessage
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.training.acceptance import (
    _is_compact_qwen3_1_7b,
    _is_peft_comparable_model,
    _message_kind,
    _node_snapshot_hash,
    acceptance_score,
    evaluate_training_quality,
    event_context,
    model_gate_snapshot,
    same_timeline,
)
from moonlightbox.training.acceptance_retry import resolve_reply_protocol
from moonlightbox.training.conversation_action_policy import (
    build_conversation_action_policy,
)
from moonlightbox.training.dataset_builder import (
    DatasetBuilder,
)
from moonlightbox.training.expression_policy import build_expression_policy
from moonlightbox.training.media_behavior_policy import build_media_behavior_policy
from moonlightbox.training.model_acceptance import (
    MAXIMUM_STYLE_TRANSFER_FALLBACK_RATE,
    MINIMUM_PAIRED_RESPONSE_SIMILARITY,
    ModelAcceptanceRunner,
    evaluate_activation_gates,
)
from moonlightbox.training.models import ModelVersion, TimelineConfirmation
from moonlightbox.training.peft_adapter import (
    PeftLmAdapter,
    PeftLoraConfig,
    PeftTrainingError,
    compact_persona_candidates,
    high_capacity_persona_candidates,
)
from moonlightbox.training.peft_preference_trainer import (
    PreferenceTrainingConfig,
    run_persona_preference_training,
)
from moonlightbox.training.persona_preference import build_persona_preference_dataset
from moonlightbox.training.preference_plan import (
    plan_preference_training,
    preference_training_iterations,
)
from moonlightbox.training.registry import ModelRegistry
from moonlightbox.training.search.lock import (
    AllCandidatesFailedError,
    TrainingFingerprintMismatch,
    _atomic_write_json,
)
from moonlightbox.training.search.runner import run_lora_candidate_search
from moonlightbox.training.search.state import _search_job_progress
from moonlightbox.training.sticker_policy import (
    build_sticker_evaluation_cases_from_database,
    build_sticker_policy_from_database,
    evaluate_sticker_policy,
    tune_sticker_policy_on_valid,
)
from moonlightbox.training.style_features import STYLE_FEATURE_SCHEMA_VERSION


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
            event_context(snapshot) for snapshot in confirmation.event_revision_snapshots
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
            if same_timeline(item.timestamp, policy_cutoff) <= policy_cutoff
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
            if same_timeline(example.target_at, policy_cutoff) <= policy_cutoff
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
            model_gate_snapshot(
                f"training-job:{job.id}",
                report,
                candidate_style_metrics,
            ),
            model_gate_snapshot("base", base_report, base_style_metrics),
            model_gate_snapshot(
                active_for_comparison.id if active_for_comparison is not None else "base",
                active_report,
                active_style_metrics,
            ),
        )
        comparison_scores = {
            "candidate": acceptance_score(report),
            "base": acceptance_score(base_report),
            "active": acceptance_score(active_report),
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
