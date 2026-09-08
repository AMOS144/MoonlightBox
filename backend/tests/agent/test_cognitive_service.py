from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from moonlightbox.agent.models import (
    AgentGoal,
    AgentIntention,
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
    PrivateCognitionNote,
)
from moonlightbox.branches.continuity_models import BranchMemoryItem
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import UniqueConstraint, func, select
from sqlalchemy.orm import Session

NOW = datetime(2026, 7, 23, 6, 0, tzinfo=UTC)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    import moonlightbox.api as api

    assert api.app is not None
    value = Database(f"sqlite:///{tmp_path / 'cognition.db'}")
    Base.metadata.create_all(value.engine)
    return value


def _seed_branch(
    session: Session,
    *,
    project_id: str = "project-1",
    branch_id: str = "branch-1",
    model_version_id: str = "model-1",
) -> None:
    if session.get(Project, project_id) is None:
        session.add(Project(id=project_id, name=project_id))
        session.flush()
        session.add(
            ModelVersion(
                id=model_version_id,
                project_id=project_id,
                base_model="test",
                adapter_path="test",
                dataset_hash=model_version_id,
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
            model_version_id=model_version_id,
            title=branch_id,
            origin_time=NOW,
            state_snapshot={},
        )
    )
    session.commit()


def _append_trigger(service: object, *, branch_id: str = "branch-1") -> PerceptionEvent:
    return service.append_perception_event(
        project_id="project-1",
        branch_id=branch_id,
        event_type="user_message",
        occurred_at=NOW,
        source="conversation",
        idempotency_key="message-1",
        evidence={"content": "你好"},
    )


def test_cognition_types_are_immutable() -> None:
    from moonlightbox.agent.types import CognitionRequest

    request = CognitionRequest(
        project_id="project-1",
        branch_id="branch-1",
        trigger_event={"event_type": "user_message"},
        deadline=NOW + timedelta(seconds=5),
        current_mental_state={"mood": "calm"},
        goals=(),
        relevant_context=(),
    )

    with pytest.raises(FrozenInstanceError):
        request.__setattr__("branch_id", "other")


def test_append_perception_event_is_branch_local_and_idempotent(database: Database) -> None:
    from moonlightbox.agent.service import BranchScopeError, ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        service = ShadowCognitionService(session)

        first = _append_trigger(service)
        repeated = _append_trigger(service)

        assert repeated.id == first.id
        assert session.scalar(select(func.count()).select_from(PerceptionEvent)) == 1
        with pytest.raises(BranchScopeError):
            service.append_perception_event(
                project_id="wrong-project",
                branch_id="branch-1",
                event_type="user_message",
                occurred_at=NOW,
                source="conversation",
                idempotency_key="wrong-scope",
            )


def test_situational_observation_atomically_updates_expiring_branch_state(
    database: Database,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService
    from moonlightbox.branches.situational_state import active_situational_state

    with Session(database.engine) as session:
        _seed_branch(session)
        event = ShadowCognitionService(session).append_situational_observation(
            project_id="project-1",
            branch_id="branch-1",
            values={"activity": "正在开会", "location": "公司"},
            occurred_at=NOW,
            valid_until=NOW + timedelta(hours=1),
            source="external_observation",
            idempotency_key="world-1",
            confidence=0.95,
        )

        branch = session.get(Branch, "branch-1")
        assert branch is not None
        active = active_situational_state(branch.state_snapshot, now=NOW)
        assert active is not None
        assert active["values"] == {"activity": "正在开会", "location": "公司"}
        assert active["slots"]["activity"]["evidence_ids"] == [event.id]
        assert event.event_type == "situational_observation"


def test_initial_mental_state_is_created_once_per_branch(database: Database) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        service = ShadowCognitionService(session)

        first = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={"mood": "neutral"},
        )
        repeated = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={"mood": "ignored"},
        )

        assert repeated.id == first.id
        assert first.version == 1
        assert first.is_current is True
        assert first.state == {"mood": "neutral"}


def test_cycle_trigger_has_database_uniqueness_and_service_idempotency(
    database: Database,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    unique_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in CognitiveCycle.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("branch_id", "trigger_event_id") in unique_columns

    with Session(database.engine) as session:
        _seed_branch(session)
        service = ShadowCognitionService(session)
        event = _append_trigger(service)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={},
        )

        first = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=event.id,
            input_cutoff_at=NOW,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )
        repeated = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=event.id,
            input_cutoff_at=NOW,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )

        assert repeated.id == first.id
        assert session.scalar(select(func.count()).select_from(CognitiveCycle)) == 1


def test_stale_cycle_is_invalidated_by_state_change_or_new_user_message(
    database: Database,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        _seed_branch(
            session,
            project_id="project-2",
            branch_id="branch-2",
            model_version_id="model-2",
        )
        service = ShadowCognitionService(session)
        event = _append_trigger(service)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={},
        )
        cycle = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=event.id,
            input_cutoff_at=NOW,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )
        service.append_perception_event(
            project_id="project-2",
            branch_id="branch-2",
            event_type="user_message",
            occurred_at=NOW + timedelta(seconds=1),
            source="conversation",
            idempotency_key="other-branch",
        )
        assert service.invalidate_if_stale(cycle.id) is False

        service.append_perception_event(
            project_id="project-1",
            branch_id="branch-1",
            event_type="user_message",
            occurred_at=NOW + timedelta(seconds=1),
            source="conversation",
            idempotency_key="new-message",
        )
        assert service.invalidate_if_stale(cycle.id) is True
        assert cycle.status == "invalidated"


def test_shadow_result_persists_note_without_evolving_state_goals_or_intentions(
    database: Database,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService
    from moonlightbox.agent.types import CognitionDraft, ExpressionDecision

    with Session(database.engine) as session:
        _seed_branch(session)
        service = ShadowCognitionService(session)
        event = _append_trigger(service)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={"mood": "neutral"},
        )
        cycle = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=event.id,
            input_cutoff_at=NOW,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )
        draft = CognitionDraft(
            private_content="我想先理解对方。",
            subjective_feelings={"concern": 0.4},
            attention_target={"topic": "对方的感受"},
            desired_actions=({"kind": "listen"},),
            expression_decision=ExpressionDecision(express=False, reason="影子模式"),
            suggested_next_wakeup=None,
            structured_changes={"mental_state": {"mood": "concerned"}},
        )

        note = service.persist_shadow_result(cycle.id, draft)

        assert note.content == "我想先理解对方。"
        assert cycle.status == "succeeded"
        assert cycle.private_note_id == note.id
        assert cycle.final_decision == {
            "express": False,
            "content": None,
            "reason": "影子模式",
        }
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 1
        assert session.scalar(select(func.count()).select_from(MentalStateVersion)) == 1
        assert session.scalar(select(func.count()).select_from(AgentGoal)) == 0
        assert session.scalar(select(func.count()).select_from(AgentIntention)) == 0


def test_schedule_wakeup_merges_by_branch_and_idempotency_key(database: Database) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        service = ShadowCognitionService(session)

        first = service.schedule_wakeup(
            project_id="project-1",
            branch_id="branch-1",
            wake_at=NOW + timedelta(hours=1),
            reason="稍后重新关注",
            idempotency_key="cycle-1",
        )
        repeated = service.schedule_wakeup(
            project_id="project-1",
            branch_id="branch-1",
            wake_at=NOW + timedelta(hours=2),
            reason="合并后的原因",
            idempotency_key="cycle-1",
        )

        assert repeated.id == first.id
        assert repeated.wake_at.replace(tzinfo=UTC) == NOW + timedelta(hours=2)
        assert repeated.reason == "合并后的原因"
        assert session.scalar(select(func.count()).select_from(AgentWakeup)) == 1


def test_cognition_request_uses_recent_bounded_events_and_background_deadline(
    database: Database,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        service = ShadowCognitionService(session)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={},
        )
        trigger = None
        for index in range(20):
            trigger = service.append_perception_event(
                project_id="project-1",
                branch_id="branch-1",
                event_type="elapsed_time",
                occurred_at=NOW + timedelta(seconds=index),
                source="test",
                idempotency_key=f"event-{index}",
            )
        assert trigger is not None
        cycle = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=trigger.id,
            input_cutoff_at=trigger.occurred_at,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )

        before = datetime.now(UTC)
        request = service.build_request(cycle.id)

        assert len(request.relevant_context) == 12
        assert request.relevant_context[-1]["id"] == trigger.id
        assert request.deadline >= before + timedelta(seconds=55)


def test_cognition_request_includes_the_agents_own_public_reply(database: Database) -> None:
    """A follow-up must be interpreted against the complete two-sided transcript."""

    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        assistant = BranchMessage(
            branch_id="branch-1",
            sequence=0,
            role="assistant",
            content="！",
            type="text",
            turn_id="assistant-turn-1",
            bubble_index=0,
            created_at=NOW,
        )
        user = BranchMessage(
            branch_id="branch-1",
            sequence=1,
            role="user",
            content="什么意思呢",
            type="text",
            turn_id="user-turn-1",
            bubble_index=0,
            created_at=NOW + timedelta(seconds=1),
        )
        session.add_all([assistant, user])
        session.flush()

        service = ShadowCognitionService(session)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={},
        )
        trigger = service.append_perception_event(
            project_id="project-1",
            branch_id="branch-1",
            event_type="user_message",
            occurred_at=NOW + timedelta(seconds=1),
            source="branch_conversation",
            idempotency_key=f"user-message:{user.id}",
            evidence={"branch_message_id": user.id, "content": user.content},
        )
        cycle = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=trigger.id,
            input_cutoff_at=trigger.occurred_at,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )

        request = service.build_request(cycle.id)

        conversation = [
            item
            for item in request.relevant_context
            if item["event_type"] in {"user_message", "agent_expression"}
        ]
        assert [item["event_type"] for item in conversation] == [
            "agent_expression",
            "user_message",
        ]
        assert [item["evidence"]["content"] for item in conversation] == [
            "！",
            "什么意思呢",
        ]
        assert conversation[-1]["id"] == trigger.id


def test_candidate_preview_cognition_is_authoritative(database: Database) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        branch = session.get(Branch, "branch-1")
        assert branch is not None
        branch.subject_agent_mode = "preview"
        session.commit()
        service = ShadowCognitionService(session)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={},
        )
        trigger = service.append_perception_event(
            project_id="project-1",
            branch_id="branch-1",
            event_type="user_message",
            occurred_at=NOW,
            source="conversation",
            idempotency_key="preview-message",
            evidence={"content": "在吗"},
        )
        cycle = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=trigger.id,
            input_cutoff_at=trigger.occurred_at,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )

        request = service.build_request(cycle.id)

        assert request.authoritative_expression is True


def test_cognition_request_selects_relevant_branch_memory_without_fallback(
    database: Database,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        _seed_branch(session)
        relevant = BranchMemoryItem(
            branch_id="branch-1",
            kind="experience",
            content="用户对旅行目的地感到纠结和不确定",
            subject="用户",
            predicate="感到",
            object="旅行选择纠结",
            confidence=1,
            importance=5,
            valid_from=NOW,
            source_episode_ids=[],
            lineage_hash="relevant",
            review_status="approved",
        )
        unrelated = BranchMemoryItem(
            branch_id="branch-1",
            kind="experience",
            content="用户昨天讨论了牛仔裤尺码",
            subject="用户",
            predicate="讨论",
            object="牛仔裤",
            confidence=1,
            importance=10,
            valid_from=NOW,
            source_episode_ids=[],
            lineage_hash="unrelated",
            review_status="approved",
        )
        session.add_all([relevant, unrelated])
        session.flush()
        service = ShadowCognitionService(session)
        state = service.ensure_initial_mental_state(
            project_id="project-1",
            branch_id="branch-1",
            state={},
        )
        trigger = service.append_perception_event(
            project_id="project-1",
            branch_id="branch-1",
            event_type="user_contribution_cluster",
            occurred_at=NOW,
            source="conversation_actor",
            idempotency_key="cluster-1",
            evidence={"content": "并不知道\n非常纠结！"},
        )
        cycle = service.create_pending_cycle(
            project_id="project-1",
            branch_id="branch-1",
            trigger_event_id=trigger.id,
            input_cutoff_at=NOW,
            starting_state_version_id=state.id,
            model_version_id="model-1",
        )

        request = service.build_request(cycle.id)

        assert cycle.memory_ids == [relevant.id]
        assert [item["id"] for item in request.decision_context["approved_memories"]] == [
            relevant.id
        ]
