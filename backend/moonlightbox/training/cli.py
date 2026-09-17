import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.embeddings import LocalChineseEmbedder
from moonlightbox.evaluation.blind_service import HumanBlindStudyService
from moonlightbox.evaluation.models import HumanBlindStudy
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.jobs.models import Job
from moonlightbox.training.acceptance import (
    acceptance_score,
    event_context,
    model_gate_snapshot,
    same_timeline,
)
from moonlightbox.training.acceptance_retry import (
    resolve_reply_protocol,
    retry_acceptance_only,
)
from moonlightbox.training.confirmation import ConfirmationService
from moonlightbox.training.conversation_action_policy import (
    build_conversation_action_policy,
)
from moonlightbox.training.dataset_builder import ConfirmedEventContext, DatasetBuilder
from moonlightbox.training.expression_policy import build_expression_policy
from moonlightbox.training.jobs import DigitalHumanTrainingConfig
from moonlightbox.training.media_behavior_policy import build_media_behavior_policy
from moonlightbox.training.model_acceptance import (
    LocalAcceptanceReviewer,
    ModelAcceptanceRunner,
    default_acceptance_fixture,
    evaluate_activation_gates,
)
from moonlightbox.training.models import ModelVersion, TimelineConfirmation
from moonlightbox.training.peft_adapter import prune_adapter_checkpoints
from moonlightbox.training.peft_generation import PeftPathReplyGenerator
from moonlightbox.training.sticker_policy import (
    build_sticker_evaluation_cases_from_database,
    build_sticker_policy_from_database,
    evaluate_sticker_policy,
    tune_sticker_policy_on_valid,
)


def backfill_behavior_policies(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    model_version_id: str,
) -> dict[str, object]:
    """Derive missing chat behavior metadata from the model's own training import."""

    model = session.scalar(
        select(ModelVersion).where(
            ModelVersion.id == model_version_id,
            ModelVersion.project_id == project_id,
        )
    )
    if model is None:
        raise RuntimeError("项目中不存在指定模型")
    manifest = model.training_config.get("data_manifest")
    confirmation_id = manifest.get("confirmation_id") if isinstance(manifest, dict) else None
    if not isinstance(confirmation_id, str) or not confirmation_id:
        raise RuntimeError("模型训练清单缺少 confirmation_id")
    confirmation = session.get(TimelineConfirmation, confirmation_id)
    if confirmation is None or confirmation.project_id != project_id:
        raise RuntimeError("模型训练确认记录不存在")
    target = session.scalar(
        select(Participant).where(
            Participant.project_id == project_id,
            Participant.role == "target",
        )
    )
    if target is None:
        raise RuntimeError("项目缺少目标参与者")
    rows = list(
        session.execute(
            select(Message, Participant.name)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.project_id == project_id,
                Message.import_id == confirmation.import_id,
                Participant.role.in_(("self", "target")),
            )
            .order_by(Message.timestamp, Message.source_id)
        )
    )
    if not rows:
        raise RuntimeError("训练确认记录没有对应聊天消息")
    imported = [
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
    config = model.training_config
    builder = DatasetBuilder(
        context_turns=int(config.get("context_turns", 12)),
        maximum_event_ratio=float(config.get("maximum_event_ratio", 0.25)),
        style_transfer_ratio=float(config.get("style_transfer_ratio", 0.5)),
        recency_window_days=int(config.get("recency_window_days", 10)),
        recency_multiplier=int(config.get("recency_multiplier", 2)),
        plain_text_supervision=str(config.get("training_protocol_version", "")).startswith(
            "persona-plain-text"
        ),
    )
    cutoff = max(item.timestamp for item in imported)
    examples = builder.build(imported, target_sender=target.name, cutoff=cutoff)
    boundaries = manifest.get("split_time_boundaries") if isinstance(manifest, dict) else None
    train = boundaries.get("train") if isinstance(boundaries, dict) else None
    train_end_value = train.get("end") if isinstance(train, dict) else None
    if not isinstance(train_end_value, str) or not train_end_value:
        raise RuntimeError("模型训练清单缺少 train 时间边界")
    policy_cutoff = datetime.fromisoformat(train_end_value)
    policy_messages = [
        item for item in imported if same_timeline(item.timestamp, policy_cutoff) <= policy_cutoff
    ]
    if not policy_messages:
        raise RuntimeError("train 时间边界内没有行为策略样本")
    kernel_examples = [
        item
        for item in examples
        if item.kind == "chat" and same_timeline(item.target_at, policy_cutoff) <= policy_cutoff
    ]
    rhythm = (
        EvidenceBackedIdentityKernelBuilder(
            default_timezone_offset_minutes=int(
                config.get(
                    "behavioral_timezone_offset_minutes",
                    settings.behavioral_timezone_offset_minutes,
                )
            )
        )
        .build(
            persona=target.name,
            examples=kernel_examples,
            event_contexts=[],
        )
        .behavioral_rhythm
    )
    expression = build_expression_policy(policy_messages, target_sender=target.name)
    actions = build_conversation_action_policy(policy_messages, target_sender=target.name)
    media = build_media_behavior_policy(policy_messages, target_sender=target.name)
    valid = boundaries.get("valid") if isinstance(boundaries, dict) else None
    test = boundaries.get("test") if isinstance(boundaries, dict) else None
    valid_start = valid.get("start") if isinstance(valid, dict) else None
    valid_end = valid.get("end") if isinstance(valid, dict) else None
    test_start = test.get("start") if isinstance(test, dict) else None
    test_end = test.get("end") if isinstance(test, dict) else None
    if not all(
        isinstance(value, str) and value for value in (valid_start, valid_end, test_start, test_end)
    ):
        raise RuntimeError("模型训练清单缺少 valid/test 时间边界")
    try:
        sticker_embedder = LocalChineseEmbedder(
            settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
        )
    except Exception:
        sticker_embedder = None
    sticker_policy = build_sticker_policy_from_database(
        session,
        project_id=project_id,
        target_id=target.id,
        cutoff=policy_cutoff,
        branch_time=policy_cutoff,
        import_id=confirmation.import_id,
        embedder=sticker_embedder,
    )
    sticker_valid_cases = build_sticker_evaluation_cases_from_database(
        session,
        project_id=project_id,
        target_id=target.id,
        start=datetime.fromisoformat(valid_start),
        end=datetime.fromisoformat(valid_end),
        split="valid",
        import_id=confirmation.import_id,
    )
    sticker_test_cases = build_sticker_evaluation_cases_from_database(
        session,
        project_id=project_id,
        target_id=target.id,
        start=datetime.fromisoformat(test_start),
        end=datetime.fromisoformat(test_end),
        split="test",
        import_id=confirmation.import_id,
    )
    has_valid_positive = any(case.actual_asset_id is not None for case in sticker_valid_cases)
    has_valid_negative = any(case.actual_asset_id is None for case in sticker_valid_cases)
    if sticker_policy.enabled and has_valid_positive and has_valid_negative:
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
    else:
        sticker_valid_metrics = {}
        sticker_test_metrics = {}
    model.training_config = {
        **model.training_config,
        "expression_policy": expression.to_metadata(),
        "conversation_action_policy": actions.to_metadata(),
        "media_behavior_policy": media.to_metadata(),
        "sticker_policy": sticker_policy.to_metadata(),
        "sticker_policy_valid_metrics": sticker_valid_metrics,
        "sticker_policy_test_metrics": sticker_test_metrics,
        "behavioral_rhythm": rhythm,
        "behavior_policy_backfill": {
            "version": "historical-behavior-backfill-v2-train-only",
            "confirmation_id": confirmation.id,
            "import_id": confirmation.import_id,
            "training_cutoff": policy_cutoff.isoformat(),
            "source_message_count": len(policy_messages),
            "derived_at": datetime.now(UTC).isoformat(),
        },
    }
    session.commit()
    return {
        "model_version_id": model.id,
        "expression_policy_enabled": expression.enabled,
        "expression_positive_count": expression.positive_count,
        "expression_negative_count": expression.negative_count,
        "behavioral_rhythm_sample_count": rhythm.get("sample_count", 0),
        "proactive_turn_rate": rhythm.get("proactive_turn_rate", 0),
        "inter_bubble_delay_ms": rhythm.get("inter_bubble_delay_ms", {}),
        "conversation_action_counts": actions.action_counts,
        "media_modality_counts": media.modality_counts,
        "sticker_policy_parameters": sticker_policy.parameters,
        "sticker_valid_metrics": sticker_valid_metrics,
        "sticker_test_metrics": sticker_test_metrics,
    }


def backfill_human_blind_study(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    model_version_id: str,
    sample_count: int,
) -> dict[str, object]:
    """Create a real blind study for a previously published model."""

    model = session.scalar(
        select(ModelVersion).where(
            ModelVersion.id == model_version_id,
            ModelVersion.project_id == project_id,
        )
    )
    if model is None:
        raise RuntimeError("项目中不存在指定模型")
    model.training_config = {
        **model.training_config,
        "human_blind_required": True,
        "human_blind_minimum_ratings": max(
            settings.human_blind_minimum_ratings,
            sample_count,
        ),
        "human_blind_minimum_preference": (settings.human_blind_minimum_preference),
    }
    existing = session.scalar(
        select(HumanBlindStudy)
        .where(
            HumanBlindStudy.model_version_id == model.id,
            HumanBlindStudy.status == "open",
        )
        .order_by(HumanBlindStudy.created_at.desc())
    )
    if existing is not None:
        existing.minimum_ratings = max(
            existing.minimum_ratings,
            settings.human_blind_minimum_ratings,
            sample_count,
        )
        existing.minimum_preference = max(
            existing.minimum_preference,
            settings.human_blind_minimum_preference,
        )
        session.commit()
        return {
            "study_id": existing.id,
            "status": existing.status,
            "minimum_ratings": existing.minimum_ratings,
            "minimum_preference": existing.minimum_preference,
            "created": False,
        }
    manifest = model.training_config.get("data_manifest")
    confirmation_id = manifest.get("confirmation_id") if isinstance(manifest, dict) else None
    if not isinstance(confirmation_id, str) or not confirmation_id:
        raise RuntimeError("模型训练清单缺少 confirmation_id")
    data_dir = settings.data_dir / "projects" / project_id / "training" / confirmation_id
    generator = PeftPathReplyGenerator(
        device=settings.persona_device,
        load_in_4bit=settings.persona_load_in_4bit,
    )
    runner = ModelAcceptanceRunner(
        generator,
        LocalAcceptanceReviewer(),
        default_acceptance_fixture(),
    )
    minimum_ratings = max(settings.human_blind_minimum_ratings, sample_count)
    cases = runner.build_human_blind_cases(
        data_dir=data_dir,
        base_model=model.base_model,
        adapter_path=model.adapter_path,
        sample_count=minimum_ratings,
    )
    study = HumanBlindStudyService(session).create(
        project_id=project_id,
        model_version_id=model.id,
        cases=cases,
        minimum_ratings=minimum_ratings,
        minimum_preference=settings.human_blind_minimum_preference,
    )
    return {
        "study_id": study.id,
        "status": study.status,
        "case_count": len(cases),
        "created": True,
    }


def audit_acceptance_checkpoint(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    job_id: str,
    adapter_path: Path,
    acceptance_runner: ModelAcceptanceRunner,
    select_if_passed: bool = False,
) -> dict[str, object]:
    """Evaluate one preference checkpoint with the current training activation gate."""

    job = session.get(Job, job_id)
    if job is None or job.payload.get("project_id") != project_id:
        raise RuntimeError("训练任务不属于指定项目")
    confirmation = session.get(
        TimelineConfirmation,
        str(job.payload.get("confirmation_id", "")),
    )
    if confirmation is None:
        raise RuntimeError("训练任务缺少确认记录")
    config = DigitalHumanTrainingConfig.model_validate(confirmation.config_snapshot)
    root = (settings.model_dir / project_id / job_id).resolve()
    candidate_dir = adapter_path.resolve()
    if not candidate_dir.is_relative_to(root):
        raise RuntimeError("候选 checkpoint 越过训练任务目录")
    candidate_file = candidate_dir / "adapters.safetensors"
    if not candidate_file.is_file():
        raise RuntimeError("候选 checkpoint 权重不存在")
    rows = list(
        session.execute(
            select(Message, Participant.name)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.project_id == project_id,
                Message.import_id == confirmation.import_id,
                Participant.role.in_(("self", "target")),
            )
            .order_by(Message.timestamp, Message.source_id)
        )
    )
    target = session.scalar(
        select(Participant).where(
            Participant.project_id == project_id,
            Participant.role == "target",
        )
    )
    if not rows or target is None:
        raise RuntimeError("训练任务缺少目标人物聊天")
    imported = [
        ImportedMessage(
            source_id=message.source_id,
            timestamp=message.timestamp,
            sender=sender,
            kind=_message_kind(message.kind),
            content=message.content,
            raw=message.raw,
        )
        for message, sender in rows
    ]
    cutoff = max(message.timestamp for message, _sender in rows)
    builder = DatasetBuilder(
        context_turns=config.context_turns,
        maximum_event_ratio=config.maximum_event_ratio,
        style_transfer_ratio=config.style_transfer_ratio,
        recency_window_days=config.recency_window_days,
        recency_multiplier=config.recency_multiplier,
        plain_text_supervision=config.training_protocol_version.startswith("persona-plain-text"),
    )
    examples = builder.build(imported, target_sender=target.name, cutoff=cutoff)
    event_contexts = [
        event_context(snapshot) for snapshot in confirmation.event_revision_snapshots
    ]
    examples = builder.augment_with_events(examples, event_contexts)
    examples.extend(builder.grounding_policy_examples(persona=target.name, cutoff=cutoff))
    style_examples = [example for example in examples if example.kind == "chat"]
    kernel = EvidenceBackedIdentityKernelBuilder(
        default_timezone_offset_minutes=config.behavioral_timezone_offset_minutes
    ).build(
        persona=target.name,
        examples=style_examples,
        event_contexts=event_contexts,
    )
    dataset_path = settings.data_dir / "projects" / project_id / "training" / confirmation.id
    reference = _latest_training_acceptance_reference(root)
    model_ids = reference.get("model_ids")
    if not isinstance(model_ids, dict):
        raise RuntimeError("训练验收对照缺少模型身份")
    active_model_id = model_ids.get("active")
    if not isinstance(active_model_id, str) or not active_model_id:
        raise RuntimeError("训练验收对照缺少原线上模型身份")
    active_model = None if active_model_id == "base" else session.get(ModelVersion, active_model_id)
    if active_model_id != "base" and (
        active_model is None or active_model.project_id != project_id
    ):
        raise RuntimeError("训练验收对照中的原线上模型不存在")

    def evaluate_model(
        *,
        base_model: str,
        model_adapter_path: str,
        reply_protocol: str,
    ) -> tuple[object, dict[str, float | int | str]]:
        fresh_report = acceptance_runner.run(
            base_model=base_model,
            adapter_path=model_adapter_path,
            persona=target.name,
            cutoff=cutoff.isoformat(),
            style_profile=kernel.style_profile,
            reply_protocol=reply_protocol,
        )
        fresh_style = acceptance_runner.evaluate_valid_style(
            data_dir=dataset_path,
            base_model=base_model,
            adapter_path=model_adapter_path,
            persona=target.name,
            sample_count=20,
            style_profile=kernel.style_profile,
            expressed_preferences=kernel.expressed_preferences,
            training_task="style_transfer",
        )
        direct_style = acceptance_runner.evaluate_valid_style(
            data_dir=dataset_path,
            base_model=base_model,
            adapter_path=model_adapter_path,
            persona=target.name,
            sample_count=20,
            style_profile=kernel.style_profile,
            expressed_preferences=kernel.expressed_preferences,
            training_task="conversation",
        )
        for source, destination in {
            "paired_response_similarity": "direct_response_paired_similarity",
            "valid_sample_audit": "direct_response_sample_audit",
            "valid_sample_hash": "direct_response_sample_hash",
            "valid_grounding_failure_rate": "direct_response_grounding_failure_rate",
            "excluded_unanswerable_current_state_count": (
                "direct_response_excluded_current_state_count"
            ),
        }.items():
            if source in direct_style:
                fresh_style[destination] = direct_style[source]
        causal_evaluator = getattr(
            acceptance_runner,
            "evaluate_memory_causality",
            None,
        )
        if callable(causal_evaluator):
            fresh_style.update(
                causal_evaluator(
                    base_model=base_model,
                    adapter_path=model_adapter_path,
                    persona=target.name,
                    style_profile=kernel.style_profile,
                )
            )
        return fresh_report, fresh_style

    report, style_metrics = evaluate_model(
        base_model=config.base_model,
        model_adapter_path=str(candidate_dir),
        reply_protocol=resolve_reply_protocol(confirmation.config_snapshot),
    )
    base_report, base_style = evaluate_model(
        base_model=config.base_model,
        model_adapter_path="",
        reply_protocol=resolve_reply_protocol(confirmation.config_snapshot),
    )
    if active_model is None:
        active_report, active_style = base_report, base_style
    else:
        active_report, active_style = evaluate_model(
            base_model=active_model.base_model,
            model_adapter_path=active_model.adapter_path,
            reply_protocol=resolve_reply_protocol(active_model.training_config),
        )
    gate = evaluate_activation_gates(
        model_gate_snapshot(f"checkpoint:{candidate_dir.name}", report, style_metrics),
        model_gate_snapshot("base", base_report, base_style),
        model_gate_snapshot(active_model_id, active_report, active_style),
    )
    candidate_score = acceptance_score(report)
    base_score = acceptance_score(base_report)
    active_score = acceptance_score(active_report)
    passed = (
        report.passed
        and candidate_score >= base_score
        and candidate_score >= active_score
        and gate.passed
    )
    payload: dict[str, object] = {
        "schema_version": "moonlightbox.preference-checkpoint-acceptance.v2",
        "passed": passed,
        "adapter_path": str(candidate_dir),
        "adapter_sha256": _file_sha256(candidate_file),
        "candidate": asdict(report),
        "candidate_style": style_metrics,
        "base": asdict(base_report),
        "base_style": base_style,
        "active": asdict(active_report),
        "active_style": active_style,
        "activation_gate": asdict(gate),
        "comparison_scores": {
            "candidate": candidate_score,
            "base": base_score,
            "active": active_score,
        },
        "reference_report": reference["_path"],
    }
    audit_path = root / f"checkpoint-acceptance-{candidate_dir.name}.json"
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["audit_path"] = str(audit_path)
    if select_if_passed:
        if not passed:
            raise RuntimeError("候选 checkpoint 未通过当前完整激活门槛")
        cumulative_iteration = int(candidate_dir.name) if candidate_dir.name.isdigit() else 0
        selection = {
            "schema_version": "moonlightbox.preference-checkpoint-selection.v2",
            "adapter_dir": str(candidate_dir),
            "sha256": payload["adapter_sha256"],
            "cumulative_iteration": cumulative_iteration,
            "audit_path": str(audit_path),
            "selected_at": datetime.now(UTC).isoformat(),
        }
        selection_path = root / "memory-authority-selected.json"
        temporary = selection_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(selection_path)
        payload["selection_path"] = str(selection_path)
    return payload


def _latest_training_acceptance_reference(root: Path) -> dict[str, object]:
    paths = sorted(
        root.glob("acceptance-report*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == "moonlightbox.high-fidelity-acceptance.v1"
            and isinstance(payload.get("valid_style"), dict)
        ):
            return {**payload, "_path": str(path)}
    raise RuntimeError("训练任务缺少同版本基座与线上验收对照")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_project(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    target_name: str,
    output_dir: Path,
) -> dict[str, object]:
    """生成不入队、不激活的训练数据审计与时间切分。"""

    run = session.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.project_id == project_id,
            AnalysisRun.analysis_version == "hybrid-v3",
            AnalysisRun.status == "succeeded",
        )
        .order_by(AnalysisRun.completed_at.desc(), AnalysisRun.created_at.desc())
    )
    if run is None:
        raise RuntimeError("项目没有已完成的 V3 分析")
    target = session.scalar(
        select(Participant).where(
            Participant.project_id == project_id,
            Participant.role == "target",
            Participant.name == target_name,
        )
    )
    if target is None:
        raise RuntimeError("目标人物不存在或角色不是 target")
    rows = list(
        session.execute(
            select(Message, Participant.name)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.project_id == project_id,
                Message.import_id == run.import_id,
                Participant.role.in_(("self", "target")),
            )
            .order_by(Message.timestamp, Message.source_id)
        )
    )
    if not rows:
        raise RuntimeError("目标导入记录没有可训练消息")
    cutoff = max(message.timestamp for message, _ in rows)
    messages = [
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
    builder = DatasetBuilder(
        context_turns=settings.training_context_turns,
        maximum_event_ratio=settings.training_maximum_event_ratio,
        style_transfer_ratio=settings.training_style_transfer_ratio,
        recency_window_days=settings.training_recency_window_days,
        recency_multiplier=settings.training_recency_multiplier,
        plain_text_supervision=settings.training_protocol_version.startswith("persona-plain-text"),
    )
    examples = builder.build(messages, target_sender=target_name, cutoff=cutoff)
    events = _active_event_contexts(session, run)
    examples = builder.augment_with_events(examples, events)
    examples.extend(builder.grounding_policy_examples(persona=target_name, cutoff=cutoff))
    manifest = builder.write_lora_dataset(
        examples,
        output_dir,
        target_name,
        cutoff,
        analysis_run_id=run.id,
        base_model=settings.training_base_model,
        training_config=settings.training_config_snapshot(),
    )
    return {
        "mode": "dry-run",
        "project_id": project_id,
        "target": target_name,
        "analysis_run_id": run.id,
        "output_dir": str(output_dir),
        "manifest": asdict(manifest),
        "database_mutated": False,
    }


def enqueue_training(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    target_name: str,
) -> dict[str, object]:
    """以最新 V3 节点和高保真协议创建新的确认记录与训练任务。"""

    target = session.scalar(
        select(Participant).where(
            Participant.project_id == project_id,
            Participant.role == "target",
            Participant.name == target_name,
        )
    )
    if target is None:
        raise RuntimeError("目标人物不存在或角色不是 target")
    run = session.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.project_id == project_id,
            AnalysisRun.analysis_version == "hybrid-v3",
            AnalysisRun.status == "succeeded",
        )
        .order_by(AnalysisRun.completed_at.desc(), AnalysisRun.created_at.desc())
    )
    if run is None:
        raise RuntimeError("项目没有已完成的 V3 分析")
    references = [
        (event.id, revision.revision_number)
        for event, revision in _active_event_revisions(session, run)
    ]
    result = ConfirmationService(session).confirm(
        project_id=project_id,
        analysis_run_id=run.id,
        event_revisions=references,
        config=settings.training_config_snapshot(),
    )
    return {
        "mode": "train",
        "project_id": project_id,
        "target": target_name,
        "confirmation_id": result.confirmation.id,
        "training_job_id": result.job.id,
        "status": result.job.status,
        "protocols": {
            "training": settings.training_protocol_version,
            "reply": settings.reply_protocol_version,
            "memory": settings.memory_protocol_version,
        },
    }


def cleanup_failed_training(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    job_id: str,
) -> dict[str, object]:
    """清理失败训练的冗余 checkpoint，同时保留候选与状态审计。"""

    job = session.get(Job, job_id)
    if job is None or job.status != "failed" or job.payload.get("project_id") != project_id:
        raise RuntimeError("只能清理本项目已失败的训练任务")
    root = (settings.model_dir / project_id / job_id).resolve()
    expected_parent = (settings.model_dir / project_id).resolve()
    if root.parent != expected_parent:
        raise RuntimeError("训练目录越过项目模型边界")
    state_path = root / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    fingerprint_payload = state.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise RuntimeError("搜索状态缺少训练指纹载荷")
    data_files = fingerprint_payload.get("data_files")
    if not isinstance(data_files, list):
        raise RuntimeError("搜索状态缺少数据文件指纹")
    data_dir = (
        settings.data_dir
        / "projects"
        / project_id
        / "training"
        / str(job.payload["confirmation_id"])
    )
    for item in data_files:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        path = data_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name != "manifest.json" and digest != item.get("sha256"):
            raise RuntimeError("训练切分内容已变化，拒绝刷新指纹")
        item["size"] = path.stat().st_size
        item["sha256"] = digest
    fingerprint_payload["data_digest"] = _canonical_digest(data_files)
    manifest_path = data_dir / "manifest.json"
    fingerprint_payload["declared_data_manifest_digest"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    state["fingerprint"] = _canonical_digest(fingerprint_payload)
    candidates = state.get("candidates", {})
    if isinstance(candidates, dict):
        for value in candidates.values():
            if not isinstance(value, dict) or value.get("status") != "succeeded":
                continue
            best = Path(str(value.get("best_checkpoint", "")))
            adapter_dir = Path(str(value.get("adapter_dir", "")))
            if best.is_file() and adapter_dir.is_dir():
                prune_adapter_checkpoints(adapter_dir, keep=best)
    full_dir = root / "full"
    if full_dir.exists():
        shutil.rmtree(full_dir)
    full = state.get("full_training")
    if isinstance(full, dict):
        full.clear()
        full["status"] = "pending"
    state["stage"] = "full_training_pending_retry"
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(state_path)
    return {
        "mode": "cleanup-failed",
        "project_id": project_id,
        "training_job_id": job_id,
        "state_path": str(state_path),
        "full_training_removed": True,
        "fingerprint_refreshed": True,
    }


def _active_event_revisions(
    session: Session,
    run: AnalysisRun,
) -> list[tuple[EventNode, AnalysisRevision]]:
    events = list(
        session.scalars(
            select(EventNode)
            .join(AnalysisRevision, AnalysisRevision.event_id == EventNode.id)
            .where(
                EventNode.project_id == run.project_id,
                EventNode.status == "active",
                AnalysisRevision.run_id == run.id,
            )
            .distinct()
            .order_by(EventNode.id)
        )
    )
    result: list[tuple[EventNode, AnalysisRevision]] = []
    for event in events:
        revision = session.scalar(
            select(AnalysisRevision)
            .where(
                AnalysisRevision.event_id == event.id,
                AnalysisRevision.run_id == run.id,
            )
            .order_by(AnalysisRevision.revision_number.desc())
        )
        if revision is not None:
            result.append((event, revision))
    if not result:
        raise RuntimeError("最新 V3 分析没有活动节点")
    return result


def _active_event_contexts(
    session: Session,
    run: AnalysisRun,
) -> list[ConfirmedEventContext]:
    return [
        ConfirmedEventContext(
            event_id=event.id,
            title=event.title,
            summary=event.summary,
            lane=event.lane,
            event_status=event.event_status,
            evidence_ids=tuple(event.evidence_ids),
            before_state=event.before_state,
            after_state=event.after_state,
        )
        for event, _ in _active_event_revisions(session, run)
    ]


def _message_kind(value: str) -> MessageKind:
    try:
        return MessageKind(value)
    except ValueError:
        return MessageKind.UNKNOWN


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="高保真 LoRA 数据审计与训练入口")
    parser.add_argument(
        "action",
        choices=(
            "audit",
            "train",
            "cleanup-failed",
            "acceptance-only",
            "checkpoint-acceptance",
            "behavior-policy-backfill",
            "human-blind-backfill",
        ),
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--target")
    parser.add_argument("--job-id")
    parser.add_argument("--branch-id")
    parser.add_argument("--model-version-id")
    parser.add_argument("--activate-if-passed", action="store_true")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--select-if-passed", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    settings = Settings()
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        if args.action == "audit":
            if not args.target:
                parser.error("audit 必须指定 --target")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = args.output_dir or (
                settings.data_dir / "projects" / args.project_id / "training-audits" / timestamp
            )
            payload = audit_project(
                session,
                settings,
                project_id=args.project_id,
                target_name=args.target,
                output_dir=output,
            )
        elif args.action == "train":
            if not args.target:
                parser.error("train 必须指定 --target")
            payload = enqueue_training(
                session,
                settings,
                project_id=args.project_id,
                target_name=args.target,
            )
        elif args.action == "cleanup-failed":
            if not args.job_id:
                parser.error("cleanup-failed 必须指定 --job-id")
            payload = cleanup_failed_training(
                session,
                settings,
                project_id=args.project_id,
                job_id=args.job_id,
            )
        elif args.action == "acceptance-only":
            if not args.job_id:
                parser.error("acceptance-only 必须指定 --job-id")
            generator = PeftPathReplyGenerator(
                device=settings.persona_device,
                load_in_4bit=settings.persona_load_in_4bit,
            )
            payload = retry_acceptance_only(
                session,
                job_id=args.job_id,
                data_root=settings.data_dir,
                model_root=settings.model_dir,
                acceptance_runner=ModelAcceptanceRunner(
                    generator,
                    LocalAcceptanceReviewer(),
                    default_acceptance_fixture(),
                    semantic_embedder=LocalChineseEmbedder(
                        settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
                    ),
                ),
                generator=generator,
                sample_count=args.sample_count,
                candidate_adapter_path=args.adapter_path,
            )
        elif args.action == "checkpoint-acceptance":
            if not args.job_id or args.adapter_path is None:
                parser.error("checkpoint-acceptance 必须指定 --job-id 和 --adapter-path")
            generator = PeftPathReplyGenerator(
                device=settings.persona_device,
                load_in_4bit=settings.persona_load_in_4bit,
            )
            payload = audit_acceptance_checkpoint(
                session,
                settings,
                project_id=args.project_id,
                job_id=args.job_id,
                adapter_path=args.adapter_path,
                acceptance_runner=ModelAcceptanceRunner(
                    generator,
                    LocalAcceptanceReviewer(),
                    default_acceptance_fixture(),
                    semantic_embedder=LocalChineseEmbedder(
                        settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
                    ),
                ),
                select_if_passed=args.select_if_passed,
            )
        elif args.action == "behavior-policy-backfill":
            if not args.model_version_id:
                parser.error("behavior-policy-backfill 必须指定 --model-version-id")
            payload = backfill_behavior_policies(
                session,
                settings,
                project_id=args.project_id,
                model_version_id=args.model_version_id,
            )
        elif args.action == "human-blind-backfill":
            if not args.model_version_id:
                parser.error("human-blind-backfill 必须指定 --model-version-id")
            payload = backfill_human_blind_study(
                session,
                settings,
                project_id=args.project_id,
                model_version_id=args.model_version_id,
                sample_count=args.sample_count,
            )
        else:
            parser.error("未知训练操作")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
