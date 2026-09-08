from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.branches.baseline_boundary import BaselineBoundaryResolver
from moonlightbox.branches.baseline_service import BranchBaselineService
from moonlightbox.branches.history import BranchHistoryService
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


def test_history_cursor_includes_complete_node_and_starts_branch_after_it(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'history.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        project = Project(id="project-1", name="历史分页")
        session.add(project)
        session.flush()
        source = ImportSource(
            id="import-1",
            project_id=project.id,
            preview_id="preview-1",
            source_path="/tmp/chat.txt",
            message_count=26,
            confirmed_at=datetime.now(UTC),
        )
        self_participant = Participant(
            id="self-1", project_id=project.id, name="我", role="self"
        )
        target = Participant(
            id="target-1", project_id=project.id, name="她", role="target"
        )
        model = ModelVersion(
            id="model-1",
            project_id=project.id,
            base_model="qwen",
            adapter_path="/tmp/model",
            dataset_hash="hash",
            metrics={},
        )
        session.add_all([source, self_participant, target, model])
        session.flush()
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        for index in range(26):
            session.add(
                Message(
                    id=f"message-{index:02d}",
                    project_id=project.id,
                    import_id=source.id,
                    participant_id=(
                        self_participant.id if index % 2 == 0 else target.id
                    ),
                    source_id=f"{index:02d}",
                    timestamp=started_at + timedelta(seconds=index),
                    kind="text",
                    content=f"消息 {index}",
                    raw={},
                )
            )
        event = EventNode(
            id="event-1",
            project_id=project.id,
            type="shared_experience",
            title="节点",
            summary="节点摘要",
            start_message_id="25",
            end_message_id="25",
            started_at=started_at + timedelta(seconds=25),
            ended_at=started_at + timedelta(seconds=25),
            before_state=None,
            after_state=None,
            emotion_labels=[],
            topic="节点",
            conflict_level=0,
            importance=0.9,
            reason="测试边界",
            evidence_ids=["25"],
        )
        session.add(event)
        session.flush()
        branch = Branch(
            id="branch-1",
            project_id=project.id,
            origin_event_id=event.id,
            model_version_id=model.id,
            title="节点分支",
            origin_time=started_at + timedelta(seconds=25),
            state_snapshot={},
            baseline_status="preparing",
        )
        session.add(branch)
        session.commit()

        boundary = BaselineBoundaryResolver(session).resolve(project.id, event.id)
        manifest, _state, _version = BranchBaselineService(session).build(
            branch, boundary
        )
        branch.baseline_status = "ready"
        session.commit()

        first = BranchHistoryService(session).page(
            project.id, branch.id, before=None, limit=20
        )
        second = BranchHistoryService(session).page(
            project.id,
            branch.id,
            before=first.next_cursor,
            limit=20,
        )

        assert manifest.message_count == 26
        assert [item.source_id for item in first.items] == [
            f"{index:02d}" for index in range(6, 26)
        ]
        assert [item.source_id for item in second.items] == [
            f"{index:02d}" for index in range(6)
        ]
        assert second.has_more is False
        assert first.items[-1].source_id == "25"
    database.close()
