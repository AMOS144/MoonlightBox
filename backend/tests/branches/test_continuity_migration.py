from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


def test_existing_branch_migration_is_idempotent(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_migration import (
        ContinuityMigrationService,
    )

    database = Database(f"sqlite:///{tmp_path / 'migration.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="迁移"))
        session.flush()
        session.add(
            ImportSource(
                id="import-1",
                project_id="project-1",
                preview_id="preview-1",
                source_path="/tmp/chat.txt",
                message_count=2,
                confirmed_at=now,
            )
        )
        target = Participant(
            id="target-1",
            project_id="project-1",
            name="她",
            role="target",
        )
        user = Participant(
            id="user-1",
            project_id="project-1",
            name="我",
            role="self",
        )
        session.add_all([target, user])
        session.flush()
        session.add_all(
            [
                Message(
                    project_id="project-1",
                    import_id="import-1",
                    participant_id=user.id,
                    source_id="m1",
                    timestamp=now,
                    kind="text",
                    content="出去玩吗",
                    raw={},
                ),
                Message(
                    project_id="project-1",
                    import_id="import-1",
                    participant_id=target.id,
                    source_id="m2",
                    timestamp=now + timedelta(seconds=10),
                    kind="text",
                    content="好呀",
                    raw={},
                ),
            ]
        )
        session.add(
            EventNode(
                id="event-1",
                project_id="project-1",
                type="travel",
                title="起点",
                summary="一次约定",
                start_message_id="m1",
                end_message_id="m2",
                emotion_labels=[],
                topic="旅行",
                conflict_level=0,
                importance=0.8,
                reason="起点",
                evidence_ids=["m1", "m2"],
            )
        )
        model = ModelVersion(
            id="model-1",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/tmp/adapter",
            dataset_hash="hash",
            metrics={},
            active=True,
        )
        session.add(model)
        session.flush()
        session.add(
            IdentityKernel(
                id="kernel-1",
                project_id="project-1",
                model_version_id="model-1",
                schema_version="subject-persona-v2",
                content={"persona": "她"},
                evidence_message_ids=["m1", "m2"],
                field_evidence={"values": ["m1", "m2"]},
                field_confidence={"values": 0.8},
                acceptance_report_id="accepted-report",
                content_hash="accepted-kernel",
                locked_at=now,
            )
        )
        branch = Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-1",
            title="旧分支",
            origin_time=now,
            state_snapshot={"relationship_state": {"trust": 55}},
        )
        session.add(branch)
        session.flush()
        session.add_all(
            [
                BranchMessage(
                    branch_id=branch.id,
                    sequence=0,
                    role="user",
                    content="还去吗",
                    turn_id="user-turn-1",
                    bubble_index=0,
                ),
                BranchMessage(
                    branch_id=branch.id,
                    sequence=1,
                    role="assistant",
                    content="去呀",
                    turn_id="assistant-turn-1",
                    bubble_index=0,
                    generation_metadata={"model_version_id": "model-1"},
                ),
                BranchMessage(
                    branch_id=branch.id,
                    sequence=2,
                    role="assistant",
                    content="正好想散散心",
                    turn_id="assistant-turn-1",
                    bubble_index=1,
                    generation_metadata={"model_version_id": "model-1"},
                ),
                BranchMessage(
                    branch_id=branch.id,
                    sequence=3,
                    role="user",
                    content="这条还没回复",
                    turn_id="user-turn-2",
                    bubble_index=0,
                ),
            ]
        )
        session.commit()
        service = ContinuityMigrationService(session)

        first = service.migrate_project("project-1")
        second = service.migrate_project("project-1")

        assert first.kernel_id == second.kernel_id
        assert first.branch_reports[0].episode_count == 1
        assert second.branch_reports[0].episode_count == 1
        assert session.query(IdentityKernel).count() == 1
        assert session.query(BranchMemoryEpisode).count() == 1
        assert session.query(BranchStateVersion).count() == 1
        assert branch.lifecycle_status == "archived"
        assert branch.state_snapshot["relationship_state"] == {}
        assert (
            session.query(Job).filter(Job.kind == "branch_continual_memory").count()
            == 1
        )
    database.close()
