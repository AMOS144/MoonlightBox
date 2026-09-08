from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.events.runs import AnalysisRunService
from moonlightbox.imports.models import ImportSource
from moonlightbox.projects.models import Project
from sqlalchemy.orm import Session


def test_confirmation_fingerprint_is_stable_for_event_order() -> None:
    from moonlightbox.training.confirmation import confirmation_fingerprint

    config = {
        "base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "iterations": 600,
    }
    first = confirmation_fingerprint(
        project_id="project-1",
        import_id="import-1",
        analysis_run_id="run-1",
        active_revisions=[("event-b", 2), ("event-a", 1)],
        rejected_event_ids=["event-z", "event-y"],
        config=config,
    )
    second = confirmation_fingerprint(
        project_id="project-1",
        import_id="import-1",
        analysis_run_id="run-1",
        active_revisions=[("event-a", 1), ("event-b", 2)],
        rejected_event_ids=["event-y", "event-z"],
        config=deepcopy(config),
    )

    assert first == second
    assert len(first) == 64


def test_confirmation_fingerprint_changes_with_revision_or_config() -> None:
    from moonlightbox.training.confirmation import confirmation_fingerprint

    common = {
        "project_id": "project-1",
        "import_id": "import-1",
        "analysis_run_id": "run-1",
        "rejected_event_ids": [],
    }
    first = confirmation_fingerprint(
        **common,
        active_revisions=[("event-a", 1)],
        config={"iterations": 600},
    )
    changed_revision = confirmation_fingerprint(
        **common,
        active_revisions=[("event-a", 2)],
        config={"iterations": 600},
    )
    changed_config = confirmation_fingerprint(
        **common,
        active_revisions=[("event-a", 1)],
        config={"iterations": 10},
    )

    assert first != changed_revision
    assert first != changed_config


def test_confirm_v3_timeline_is_idempotent(tmp_path: Path) -> None:
    from moonlightbox.training.confirmation import ConfirmationService

    database = Database(f"sqlite:///{tmp_path / 'confirmation.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="确认测试"))
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
        session.commit()
        run = AnalysisRunService(session).get_or_create(
            project_id="project-1",
            import_id="import-1",
            analysis_version="hybrid-v3",
            prompt_version="relationship-v3+shared-experience-v3",
            model="deepseek",
            config={"threshold": 0.72},
            window_ids=[],
        )
        run.status = "succeeded"
        run.completed_at = datetime.now(UTC)
        event = EventNode(
            id="event-1",
            project_id="project-1",
            type="travel",
            lane="shared_experience",
            event_status="occurred",
            title="杭州旅行",
            summary="共同去杭州",
            start_message_id="m1",
            end_message_id="m2",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            source_lanes=["shared_experience"],
            before_state=None,
            after_state=None,
            emotion_labels=[],
            topic="旅行",
            conflict_level=0,
            importance=0.9,
            reason="双方共同经历",
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
                prompt_version=run.prompt_version,
                model=run.model,
                run_id=run.id,
            )
        )
        session.commit()
        service = ConfirmationService(session)

        first = service.confirm(
            project_id="project-1",
            analysis_run_id=run.id,
            event_revisions=[("event-1", 1)],
        )
        second = service.confirm(
            project_id="project-1",
            analysis_run_id=run.id,
            event_revisions=[("event-1", 1)],
        )

        assert first.confirmation.id == second.confirmation.id
        assert first.job.id == second.job.id
        assert first.job.kind == "digital_human_training_v1"
        assert "content" not in first.job.payload


def test_confirm_and_train_api_is_registered(client: TestClient) -> None:
    response = client.post(
        "/api/projects/missing/events/confirm-and-train",
        json={"analysis_run_id": "missing", "event_revisions": []},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "只能确认已完成的 V3 时间轴"
