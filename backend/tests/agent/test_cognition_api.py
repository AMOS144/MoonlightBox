from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.agent.models import (
    AgentGoal,
    AgentIntention,
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
    PrivateCognitionNote,
)
from moonlightbox.api import create_app
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session

NOW = datetime(2026, 7, 23, 9, 0, tzinfo=UTC)


def _add_branch(
    session: Session,
    *,
    project_id: str,
    branch_id: str,
    model_id: str,
    baseline_status: str = "ready",
) -> None:
    session.add(Project(id=project_id, name=project_id))
    session.flush()
    session.add(
        ModelVersion(
            id=model_id,
            project_id=project_id,
            base_model="test",
            adapter_path="test",
            dataset_hash=model_id,
            metrics={},
        )
    )
    session.add(
        EventNode(
            id=f"origin-{branch_id}",
            project_id=project_id,
            type="origin",
            start_message_id="1",
            end_message_id="1",
            emotion_labels=[],
            topic="",
            conflict_level=0,
            importance=0,
            reason="",
            evidence_ids=[],
        )
    )
    session.flush()
    session.add(
        Branch(
            id=branch_id,
            project_id=project_id,
            origin_event_id=f"origin-{branch_id}",
            model_version_id=model_id,
            title=branch_id,
            origin_time=NOW,
            state_snapshot={},
            baseline_status=baseline_status,
        )
    )
    session.flush()


def _seed_cognition(session: Session) -> None:
    _add_branch(
        session,
        project_id="project-1",
        branch_id="branch-1",
        model_id="model-1",
    )
    session.add(
        EventNode(
            id="origin-empty",
            project_id="project-1",
            type="origin",
            start_message_id="1",
            end_message_id="1",
            emotion_labels=[],
            topic="",
            conflict_level=0,
            importance=0,
            reason="",
            evidence_ids=[],
        )
    )
    session.flush()
    session.add(
        Branch(
            id="branch-empty",
            project_id="project-1",
            origin_event_id="origin-empty",
            model_version_id="model-1",
            title="branch-empty",
            origin_time=NOW,
            state_snapshot={},
            baseline_status="pending",
        )
    )
    _add_branch(
        session,
        project_id="project-2",
        branch_id="branch-2",
        model_id="model-2",
    )
    session.flush()

    first_event = PerceptionEvent(
        id="event-1",
        project_id="project-1",
        branch_id="branch-1",
        event_type="elapsed_time",
        occurred_at=NOW,
        source="agent_wakeup",
        idempotency_key="event-1",
        visible_through=NOW,
        evidence={"ordinal": 1},
    )
    second_event = PerceptionEvent(
        id="event-2",
        project_id="project-1",
        branch_id="branch-1",
        event_type="user_message",
        occurred_at=NOW + timedelta(seconds=1),
        source="conversation",
        idempotency_key="event-2",
        visible_through=NOW + timedelta(seconds=1),
        evidence={"ordinal": 2},
    )
    other_event = PerceptionEvent(
        id="event-other",
        project_id="project-2",
        branch_id="branch-2",
        event_type="elapsed_time",
        occurred_at=NOW,
        source="agent_wakeup",
        idempotency_key="event-other",
        visible_through=NOW,
        evidence={"secret": "other-branch"},
    )
    session.add_all([first_event, second_event, other_event])
    session.flush()
    state = MentalStateVersion(
        id="state-1",
        project_id="project-1",
        branch_id="branch-1",
        version=1,
        state={"mood": "calm"},
        evidence={"event_id": "event-1"},
        is_current=True,
        model_version_id="model-1",
    )
    session.add(state)
    session.add_all(
        [
            AgentGoal(
                id="goal-active",
                project_id="project-1",
                branch_id="branch-1",
                goal_type="relationship",
                content="继续理解对方",
                priority=0.8,
                status="active",
            ),
            AgentGoal(
                id="goal-completed",
                project_id="project-1",
                branch_id="branch-1",
                goal_type="temporary",
                content="已完成",
                status="completed",
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            AgentIntention(
                id="intention-active",
                project_id="project-1",
                branch_id="branch-1",
                intention_type="wait",
                content="暂时等待",
                trigger_event_id="event-1",
                status="active",
            ),
            AgentIntention(
                id="intention-cancelled",
                project_id="project-1",
                branch_id="branch-1",
                intention_type="reply",
                content="不再回复",
                trigger_event_id="event-1",
                status="cancelled",
            ),
        ]
    )
    session.add_all(
        [
            AgentWakeup(
                id="wakeup-next",
                project_id="project-1",
                branch_id="branch-1",
                wake_at=NOW + timedelta(minutes=5),
                reason="下一次醒来",
                idempotency_key="wakeup-next",
                status="scheduled",
            ),
            AgentWakeup(
                id="wakeup-later",
                project_id="project-1",
                branch_id="branch-1",
                wake_at=NOW + timedelta(minutes=10),
                reason="更晚醒来",
                idempotency_key="wakeup-later",
                status="scheduled",
            ),
        ]
    )
    session.flush()
    first_cycle = CognitiveCycle(
        id="cycle-1",
        project_id="project-1",
        branch_id="branch-1",
        trigger_event_id="event-1",
        input_cutoff_at=NOW,
        starting_state_version_id="state-1",
        status="succeeded",
        model_version_id="model-1",
        model_protocol_version="subject-cognition-v1",
        created_at=NOW,
    )
    second_cycle = CognitiveCycle(
        id="cycle-2",
        project_id="project-1",
        branch_id="branch-1",
        trigger_event_id="event-2",
        input_cutoff_at=NOW + timedelta(seconds=1),
        starting_state_version_id="state-1",
        status="succeeded",
        model_version_id="model-1",
        model_protocol_version="subject-cognition-v1",
        created_at=NOW + timedelta(seconds=1),
    )
    session.add_all([first_cycle, second_cycle])
    session.flush()
    note = PrivateCognitionNote(
        id="note-private",
        project_id="project-1",
        branch_id="branch-1",
        trigger_event_id="event-2",
        content="这是只允许开发者审计查看的私密认知",
        model_version_id="model-1",
        model_protocol_version="subject-cognition-v1",
        created_at=NOW + timedelta(seconds=1),
    )
    session.add(note)
    session.flush()
    second_cycle.private_note_id = note.id
    session.add(
        BranchMessage(
            id="message-1",
            branch_id="branch-1",
            sequence=1,
            role="assistant",
            content="普通聊天内容",
            turn_id="turn-1",
            bubble_index=0,
        )
    )
    session.commit()


def test_cognition_state_is_scoped_and_baseline_not_ready_returns_empty(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'cognition-api.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings)) as client:
        database = Database(settings.database_url)
        with Session(database.engine) as session:
            _seed_cognition(session)

        state = client.get(
            "/api/projects/project-1/branches/branch-1/cognition/state"
        )
        assert state.status_code == 200
        payload = state.json()
        assert payload["mental_state"]["state"] == {"mood": "calm"}
        assert [goal["id"] for goal in payload["active_goals"]] == ["goal-active"]
        assert [item["id"] for item in payload["active_intentions"]] == [
            "intention-active"
        ]
        assert payload["next_wakeup"]["id"] == "wakeup-next"

        empty = client.get(
            "/api/projects/project-1/branches/branch-empty/cognition/state"
        )
        assert empty.status_code == 200
        assert empty.json() == {
            "mental_state": None,
            "active_goals": [],
            "active_intentions": [],
            "next_wakeup": None,
        }
        mismatch = client.get(
            "/api/projects/project-1/branches/branch-2/cognition/state"
        )
        assert mismatch.status_code == 404
        database.close()


def test_cognition_cycles_and_events_apply_limit_and_keep_notes_private(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'cognition-lists.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings)) as client:
        database = Database(settings.database_url)
        with Session(database.engine) as session:
            _seed_cognition(session)

        cycles = client.get(
            "/api/projects/project-1/branches/branch-1/cognition/cycles",
            params={"limit": 1},
        )
        assert cycles.status_code == 200
        assert len(cycles.json()) == 1
        assert cycles.json()[0]["id"] == "cycle-2"
        assert (
            cycles.json()[0]["private_note"]["content"]
            == "这是只允许开发者审计查看的私密认知"
        )

        events = client.get(
            "/api/projects/project-1/branches/branch-1/cognition/events",
            params={"limit": 1},
        )
        assert events.status_code == 200
        assert len(events.json()) == 1
        assert events.json()[0]["id"] == "event-2"
        assert events.json()[0]["evidence"] == {"ordinal": 2}

        messages = client.get(
            "/api/projects/project-1/branches/branch-1/messages"
        )
        assert messages.status_code == 200
        assert [item["content"] for item in messages.json()] == ["普通聊天内容"]
        assert "私密认知" not in messages.text

        invalid_limit = client.get(
            "/api/projects/project-1/branches/branch-1/cognition/events",
            params={"limit": 0},
        )
        assert invalid_limit.status_code == 422
        database.close()
