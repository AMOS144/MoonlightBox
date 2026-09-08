from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.events.runs import AnalysisRunService
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.jobs.service import JobService
from moonlightbox.projects.models import Project
from moonlightbox.training.confirmation import ConfirmationService
from moonlightbox.training.model_acceptance import AcceptanceReport
from moonlightbox.training.peft_adapter import (
    BestCheckpointSelection,
    PeftCapabilities,
    PeftLoraConfig,
    PeftTrainingError,
    TrainingResult,
)
from sqlalchemy.orm import Session


def test_preference_training_uses_one_effective_pass_not_one_step_per_pair() -> None:
    from moonlightbox.training.jobs import preference_training_iterations

    assert preference_training_iterations(682) == 171
    assert preference_training_iterations(
        682,
        batch_size=2,
        gradient_accumulation_steps=4,
    ) == 86


def test_style_transfer_gate_allows_only_bounded_safe_fallbacks() -> None:
    from moonlightbox.training.jobs import _model_gate_snapshot

    report = AcceptanceReport(
        case_count=1,
        passed_count=1,
        structure_failures=0,
        forbidden_fact_failures=0,
        failed_case_ids=(),
        passed=True,
    )
    metrics: dict[str, float | int | str] = {
        "valid_training_task": "style_transfer",
        "style_distance": 0.1,
        "speaker_probability": 0.8,
        "human_oracle_speaker_probability": 0.8,
        "speaker_probability_alignment": 0.9,
        "paired_response_similarity": 0.88,
        "low_authority_prompt_leak_rate": 0.0,
        "valid_grounding_failure_rate": 0.1,
    }

    style_transfer = _model_gate_snapshot("candidate", report, metrics)
    conversation = _model_gate_snapshot(
        "candidate",
        report,
        {**metrics, "valid_training_task": "conversation"},
    )

    assert style_transfer.style["valid_grounding_failure_rate"].maximum == 0.1
    assert conversation.style["valid_grounding_failure_rate"].maximum == 0.0


def test_preference_training_plan_resumes_latest_durable_checkpoint(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.jobs import plan_preference_training

    source = tmp_path / "final" / "adapter_model.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"source")
    interrupted = tmp_path / "memory-authority-one-effective-pass-v2"
    interrupted.mkdir()
    (interrupted / "0000440_adapter_model.safetensors").write_bytes(b"440")
    latest = interrupted / "0000460_adapter_model.safetensors"
    latest.write_bytes(b"460")

    plan = plan_preference_training(
        tmp_path,
        source_adapter_file=source,
        total_iterations=618,
    )

    assert plan.initial_adapter_file == latest
    assert plan.iterations == 158
    assert plan.resumed_from_iteration == 460
    assert plan.output_dir.name == "memory-authority-one-effective-pass-v2-resume-460"


def test_preference_training_plan_reuses_completed_result(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import plan_preference_training

    source = tmp_path / "final" / "adapter_model.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"source")
    completed = tmp_path / "memory-authority-one-effective-pass-v2-resume-460"
    completed.mkdir()
    (completed / "adapter_model.safetensors").write_bytes(b"done")
    (completed / "preference_metrics.json").write_text("{}", encoding="utf-8")

    plan = plan_preference_training(
        tmp_path,
        source_adapter_file=source,
        total_iterations=618,
    )

    assert plan.cached_adapter_dir == completed
    assert plan.iterations == 0


def test_preference_training_plan_honors_audited_selection(tmp_path: Path) -> None:
    import hashlib
    import json

    from moonlightbox.training.jobs import plan_preference_training

    source = tmp_path / "final" / "adapter_model.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"source")
    selected = tmp_path / "acceptance-checkpoints" / "0000540"
    selected.mkdir(parents=True)
    selected_file = selected / "adapter_model.safetensors"
    selected_file.write_bytes(b"selected")
    audit = tmp_path / "checkpoint-acceptance-0000540.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "moonlightbox.preference-checkpoint-acceptance.v2",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "memory-authority-selected.json").write_text(
        json.dumps(
            {
                "schema_version": "moonlightbox.preference-checkpoint-selection.v2",
                "adapter_dir": str(selected),
                "audit_path": str(audit),
                "sha256": hashlib.sha256(b"selected").hexdigest(),
                "cumulative_iteration": 100,
            }
        ),
        encoding="utf-8",
    )

    plan = plan_preference_training(
        tmp_path,
        source_adapter_file=source,
        total_iterations=618,
    )

    assert plan.cached_adapter_dir == selected.resolve()
    assert plan.resumed_from_iteration == 100


def test_preference_training_plan_ignores_old_overtrained_selection(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    from moonlightbox.training.jobs import plan_preference_training

    source = tmp_path / "final" / "adapter_model.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"source")
    selected = tmp_path / "acceptance-checkpoints" / "0000460"
    selected.mkdir(parents=True)
    selected_file = selected / "adapter_model.safetensors"
    selected_file.write_bytes(b"old")
    (tmp_path / "memory-authority-selected.json").write_text(
        json.dumps(
            {
                "schema_version": "moonlightbox.preference-checkpoint-selection.v1",
                "adapter_dir": str(selected),
                "sha256": hashlib.sha256(b"old").hexdigest(),
                "cumulative_iteration": 460,
            }
        ),
        encoding="utf-8",
    )

    plan = plan_preference_training(
        tmp_path,
        source_adapter_file=source,
        total_iterations=171,
    )

    assert plan.initial_adapter_file == source
    assert plan.iterations == 171
    assert plan.cached_adapter_dir is None


def test_quality_gate_rejects_future_or_non_structured_targets() -> None:
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample
    from moonlightbox.training.jobs import evaluate_training_quality

    cutoff = datetime(2026, 1, 1)
    examples = [
        TrainingExample(
            messages=[ChatTurn(role="assistant", content="普通文本")],
            source_ids=["m1"],
            target_at=cutoff + timedelta(seconds=1),
        )
    ]

    result = evaluate_training_quality(examples, cutoff, {})

    assert result["passed"] is False
    assert result["future_leak_count"] == 1


class FakePeftAdapter:
    def __init__(self) -> None:
        self.config: PeftLoraConfig | None = None
        self.capabilities = PeftCapabilities("0.20.0", "2.5.1", "5.14.1")

    def evaluate_loss(self, config: PeftLoraConfig, checkpoint: Path) -> float:
        assert checkpoint.name == "final"
        assert config.data_dir.is_dir()
        return 0.9

    def train(self, config: PeftLoraConfig, on_progress: object) -> TrainingResult:
        self.config = config
        on_progress({"stage": "training", "iteration": 10, "loss": 1.2})
        on_progress(
            {
                "stage": "training",
                "iteration": 2,
                "validation_loss": 1.1,
            }
        )
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        weights = config.adapter_dir / "adapter_model.safetensors"
        weights.write_bytes(b"adapter")
        checkpoint = config.adapter_dir / "0000001_adapter_model.safetensors"
        checkpoint.write_bytes(b"adapter")
        config_path = config.adapter_dir / "training.json"
        config_path.write_text(f"seed: {config.seed}\n", encoding="utf-8")
        return TrainingResult(
            config.adapter_dir,
            config.resolved_iterations,
            config_path,
            "0.20.0",
            (
                {
                    "iteration": 2,
                    "validation_loss": 1.1,
                },
            ),
            BestCheckpointSelection(
                validation_loss=1.1,
                validation_iteration=2,
                checkpoint_iteration=1,
                checkpoint=checkpoint,
            ),
        )


class FullFailingFakePeftAdapter(FakePeftAdapter):
    def train(self, config: PeftLoraConfig, on_progress: object) -> TrainingResult:
        if config.adapter_dir.name == "full":
            raise PeftTrainingError("完整训练失败")
        return super().train(config, on_progress)


class PassingAcceptanceRunner:
    def __init__(self) -> None:
        self.called = False
        self.adapter_paths: list[str] = []

    def run(self, **kwargs: object) -> AcceptanceReport:
        self.called = True
        self.adapter_paths.append(str(kwargs["adapter_path"]))
        adapter_path = str(kwargs["adapter_path"])
        is_candidate = (
            "/candidates/" in adapter_path
            or adapter_path.endswith("/final")
            or "/.checkpoint-style-eval-" in adapter_path
        )
        return AcceptanceReport(
            case_count=10,
            passed_count=10 if is_candidate else 8,
            structure_failures=0,
            forbidden_fact_failures=0,
            failed_case_ids=(),
            passed=is_candidate,
        )

    def evaluate_valid_style(self, **kwargs: object) -> dict[str, float | int | str]:
        adapter_path = str(kwargs["adapter_path"])
        is_candidate = (
            "/candidates/" in adapter_path
            or adapter_path.endswith("/final")
            or "/.checkpoint-style-eval-" in adapter_path
        )
        return {
            "valid_sample_count": 20,
            "valid_sample_hash": "candidate" if is_candidate else "baseline",
            "style_distance": 0.1 if is_candidate else 0.3,
            "speaker_probability": 0.9 if is_candidate else 0.7,
            "human_oracle_speaker_probability": 0.35,
            "speaker_probability_alignment": 0.8 if is_candidate else 0.7,
            "paired_response_similarity": 0.55 if is_candidate else 0.35,
            "memory_poison_copy_rate": 0.0,
            "low_authority_prompt_leak_rate": 0.0,
            "valid_grounding_failures": 0,
            "valid_grounding_failure_rate": 0.0,
            "composite_fidelity": 0.9 if is_candidate else 0.7,
        }


def test_training_handler_requires_model_acceptance_runner(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import create_digital_human_training_handler

    with pytest.raises(ValueError, match="验收"):
        create_digital_human_training_handler(
            FakePeftAdapter(),
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
        )


def test_training_job_builds_dataset_and_activates_model(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_models import IdentityKernel
    from moonlightbox.training.jobs import create_digital_human_training_handler
    from moonlightbox.training.models import ModelVersion

    database = Database(f"sqlite:///{tmp_path / 'training.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="训练测试"))
        session.flush()
        session.add(
            ImportSource(
                id="import-1",
                project_id="project-1",
                preview_id="preview-1",
                source_path="/tmp/chat.json",
                message_count=2,
                confirmed_at=datetime.now(UTC),
            )
        )
        self_participant = Participant(
            id="participant-self",
            project_id="project-1",
            name="我",
            role="self",
        )
        target = Participant(
            id="participant-target",
            project_id="project-1",
            name="她",
            role="target",
        )
        session.add_all([self_participant, target])
        session.flush()
        session.add_all(
            [
                Message(
                    project_id="project-1",
                    import_id="import-1",
                    participant_id=self_participant.id,
                    source_id="m1",
                    timestamp=datetime.now(UTC),
                    kind="text",
                    content="五一去杭州吗",
                    raw={},
                ),
                Message(
                    project_id="project-1",
                    import_id="import-1",
                    participant_id=target.id,
                    source_id="m2",
                    timestamp=datetime.now(UTC) + timedelta(minutes=1),
                    kind="text",
                    content="好呀",
                    raw={},
                ),
            ]
        )
        session.commit()
        run = AnalysisRunService(session).get_or_create(
            project_id="project-1",
            import_id="import-1",
            analysis_version="hybrid-v3",
            prompt_version="v3",
            model="deepseek",
            config={"threshold": 0.7},
            window_ids=[],
        )
        run.status = "succeeded"
        run.completed_at = datetime.now(UTC)
        event = EventNode(
            id="event-1",
            project_id="project-1",
            type="travel",
            lane="shared_experience",
            event_status="confirmed",
            title="杭州旅行",
            summary="双方确认去杭州",
            start_message_id="m1",
            end_message_id="m2",
            source_lanes=["shared_experience"],
            before_state=None,
            after_state=None,
            emotion_labels=[],
            topic="旅行",
            conflict_level=0,
            importance=0.9,
            reason="共同经历",
            evidence_ids=["m1", "m2"],
            status="active",
        )
        session.add(event)
        session.flush()
        session.add(
            AnalysisRevision(
                event_id=event.id,
                revision_number=1,
                snapshot={},
                action_reason="V3 自动发布",
                analysis_version="hybrid-v3",
                prompt_version="v3",
                model="deepseek",
                run_id=run.id,
            )
        )
        session.commit()
        result = ConfirmationService(session).confirm(
            project_id="project-1",
            analysis_run_id=run.id,
            event_revisions=[("event-1", 1)],
            config={"iterations": 10, "maximum_event_ratio": 1.0},
        )
        job = JobService(session).start(
            result.job.id,
            worker_token="worker-1",
        )
        old_version = ModelVersion(
            project_id="project-1",
            base_model="models/old-base",
            adapter_path="/models/old-adapter",
            dataset_hash="old-dataset",
            metrics={"style_score": 0.8},
            active=True,
        )
        session.add(old_version)
        session.commit()
        adapter = FakePeftAdapter()
        acceptance = PassingAcceptanceRunner()
        handler = create_digital_human_training_handler(
            adapter,
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            acceptance_runner=acceptance,  # type: ignore[arg-type]
        )

        failing_handler = create_digital_human_training_handler(
            FullFailingFakePeftAdapter(),
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            acceptance_runner=acceptance,  # type: ignore[arg-type]
        )
        from moonlightbox.jobs.registry import JobHandlerError

        with pytest.raises(JobHandlerError, match="旧模型保持启用"):
            failing_handler(JobService(session), job)
        session.refresh(old_version)
        assert old_version.active is True
        assert session.query(ModelVersion).count() == 1

        handler(JobService(session), job)

        completed = JobService(session).get(job.id)
        versions = list(session.query(ModelVersion).all())
        kernels = list(session.query(IdentityKernel).all())
        assert completed.status == "succeeded"
        assert completed.checkpoint is not None
        assert completed.checkpoint["stage"] == "completed"
        assert len(versions) == 2
        trained_version = next(version for version in versions if version.id != old_version.id)
        assert trained_version.active is True
        assert old_version.active is False
        assert "training_metadata" in trained_version.training_config
        sticker_policy = trained_version.training_config["sticker_policy"]
        assert isinstance(sticker_policy, dict)
        assert sticker_policy["version"] == "context-only-sticker-v2"
        assert sticker_policy["enabled"] is False
        assert sticker_policy["boundary"]["effective_cutoff"]
        assert sticker_policy["asset_summary"] == {
            "linked_sticker_count": 0,
            "sticker_event_count": 0,
        }
        assert sticker_policy["parameters"]["top_k"] == 5
        assert len(kernels) == 1
        assert kernels[0].model_version_id == trained_version.id
        assert kernels[0].evidence_message_ids == ["m1", "m2"]
        assert kernels[0].acceptance_report_id is not None
        assert kernels[0].schema_version == "subject-persona-v2"
        assert acceptance.called is True
        assert "" in acceptance.adapter_paths
        # 旧 MLX adapter 不可由 Linux PEFT 运行时加载，只与当前基座作比较。
        assert "/models/old-adapter" not in acceptance.adapter_paths
        assert adapter.config is not None
        assert (adapter.config.data_dir / "train.jsonl").is_file()
        assert "五一去杭州吗" not in str(completed.checkpoint)
