from datetime import UTC, datetime
from pathlib import Path

import pytest
from moonlightbox.branches.continuity_models import IdentityKernel
from moonlightbox.branches.models import Branch
from moonlightbox.branches.replies import GeneratedReplyTurn
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class UnusedGenerator:
    def generate(
        self,
        _model_version_id: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        raise AssertionError("升级分支不应调用生成器")


def test_stage_candidate_branch_is_preview_idempotent_and_keeps_live_branch(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.service import stage_candidate_branch

    database = Database(f"sqlite:///{tmp_path / 'candidate-stage.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="候选影子分支"))
        session.flush()
        source_import = ImportSource(
            project_id="project-1",
            preview_id="preview-candidate",
            source_path="/tmp/chat.txt",
            message_count=1,
            confirmed_at=datetime.now(UTC),
        )
        target = Participant(project_id="project-1", name="她", role="target")
        session.add_all([source_import, target])
        session.flush()
        session.add(
            Message(
                project_id="project-1",
                import_id=source_import.id,
                participant_id=target.id,
                source_id="m1",
                timestamp=datetime.now(UTC),
                kind="text",
                content="节点消息",
                raw={},
            )
        )
        event = EventNode(
            id="event-1",
            project_id="project-1",
            type="shared_experience",
            title="节点",
            summary="摘要",
            start_message_id="m1",
            end_message_id="m1",
            before_state=None,
            after_state=None,
            emotion_labels=[],
            topic="节点",
            conflict_level=0,
            importance=0.8,
            reason="共同经历",
            evidence_ids=["m1"],
        )
        live_model = ModelVersion(
            id="model-live",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/live",
            dataset_hash="live",
            metrics={},
            active=True,
        )
        candidate_model = ModelVersion(
            id="model-candidate",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/candidate",
            dataset_hash="candidate",
            metrics={},
            status="awaiting_human_review",
            training_config={
                "human_blind_required": True,
                "upgraded_from_model_version_id": "model-live",
            },
        )
        source_branch = Branch(
            id="branch-live",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-live",
            title="线上分支",
            origin_time=datetime(2026, 1, 1),
            state_snapshot={"relationship_state": {"label": "朋友"}},
            baseline_status="ready",
            subject_agent_mode="active",
        )
        session.add_all([event, live_model, candidate_model])
        session.flush()
        session.add(source_branch)
        session.add(
            IdentityKernel(
                project_id="project-1",
                model_version_id="model-candidate",
                schema_version="subject-persona-v2",
                content={"persona": "她"},
                evidence_message_ids=["m1"],
                content_hash="candidate-kernel",
                acceptance_report_id="accepted-report",
                locked_at=datetime.now(UTC),
            )
        )
        session.commit()

        staged = stage_candidate_branch(
            session,
            project_id="project-1",
            source_branch_id="branch-live",
            model_version_id="model-candidate",
        )
        repeated = stage_candidate_branch(
            session,
            project_id="project-1",
            source_branch_id="branch-live",
            model_version_id="model-candidate",
        )

        session.refresh(source_branch)
        session.refresh(live_model)
        assert staged.id == repeated.id
        assert staged.model_version_id == "model-candidate"
        assert staged.subject_agent_mode == "preview"
        assert staged.baseline_status == "preparing"
        assert staged.baseline_job_id is not None
        assert staged.state_snapshot["_candidate_staged_from_branch_id"] == "branch-live"
        assert source_branch.lifecycle_status == "active"
        assert source_branch.subject_agent_mode == "active"
        assert live_model.active is True
    database.close()


def test_branch_upgrade_is_clean_read_only_and_idempotent(tmp_path: Path) -> None:
    from moonlightbox.branches.service import (
        ActiveModelUnavailableError,
        BranchBaselineNotReadyError,
        BranchReadOnlyError,
        BranchService,
    )

    database = Database(f"sqlite:///{tmp_path / 'upgrade.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="升级测试"))
        session.flush()
        source = ImportSource(
            project_id="project-1",
            preview_id="preview-upgrade",
            source_path="/tmp/chat.txt",
            message_count=1,
            confirmed_at=datetime.now(UTC),
        )
        target = Participant(project_id="project-1", name="她", role="target")
        session.add_all([source, target])
        session.flush()
        session.add(
            Message(
                project_id="project-1",
                import_id=source.id,
                participant_id=target.id,
                source_id="m1",
                timestamp=datetime.now(UTC),
                kind="text",
                content="节点消息",
                raw={},
            )
        )
        event = EventNode(
            id="event-1",
            project_id="project-1",
            type="shared_experience",
            title="节点",
            summary="摘要",
            start_message_id="m1",
            end_message_id="m1",
            before_state=None,
            after_state=None,
            emotion_labels=[],
            topic="节点",
            conflict_level=0,
            importance=0.8,
            reason="共同经历",
            evidence_ids=["m1"],
        )
        old_model = ModelVersion(
            id="model-old",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/old",
            dataset_hash="old",
            metrics={},
            active=False,
        )
        active_model = ModelVersion(
            id="model-active",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/active",
            dataset_hash="active",
            metrics={},
            active=True,
        )
        old_branch = Branch(
            id="branch-old",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-old",
            title="旧分支",
            origin_time=datetime(2026, 1, 1),
            state_snapshot={"relationship": "朋友"},
            generation_policy_version="legacy",
        )
        session.add_all([event, old_model, active_model])
        session.flush()
        session.add(old_branch)
        session.commit()
        service = BranchService(session, UnusedGenerator())

        with pytest.raises(ActiveModelUnavailableError):
            service.upgrade("project-1", "branch-old")
        session.add(
            IdentityKernel(
                project_id="project-1",
                model_version_id="model-active",
                    schema_version="subject-persona-v2",
                content={"persona": "她"},
                evidence_message_ids=["m1"],
                content_hash="active-kernel",
                acceptance_report_id="accepted-report",
                locked_at=datetime.now(UTC),
            )
        )
        session.commit()
        replacement = service.upgrade("project-1", "branch-old")
        repeated = service.upgrade("project-1", "branch-old")

        assert replacement.id == repeated.id
        assert replacement.model_version_id == "model-active"
        assert replacement.baseline_status == "preparing"
        assert replacement.state_snapshot == old_branch.state_snapshot
        with pytest.raises(BranchBaselineNotReadyError):
            service.messages("project-1", replacement.id)
        with pytest.raises(BranchReadOnlyError):
            service.add_user_message("project-1", "branch-old", "你好")
