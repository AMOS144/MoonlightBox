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
from moonlightbox.agent.types import (
    CognitionDraft,
    CognitionRequest,
    ExpressionDecision,
    WakeupSuggestion,
)
from moonlightbox.branches.continuity_models import BranchMemoryItem
from moonlightbox.branches.models import Branch
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import func, select
from sqlalchemy.orm import Session

NOW = datetime(2026, 7, 23, 7, 0, tzinfo=UTC)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    import moonlightbox.api as api

    assert api.app is not None
    value = Database(f"sqlite:///{tmp_path / 'cognitive-jobs.db'}")
    Base.metadata.create_all(value.engine)
    return value


def _seed_cycle(session: Session) -> CognitiveCycle:
    from moonlightbox.agent.service import ShadowCognitionService

    session.add(Project(id="project-1", name="project-1"))
    session.flush()
    session.add(
        ModelVersion(
            id="model-1",
            project_id="project-1",
            base_model="test",
            adapter_path="test",
            dataset_hash="model-1",
            metrics={},
        )
    )
    session.add(
        EventNode(
            id="origin-1",
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
            id="branch-1",
            project_id="project-1",
            origin_event_id="origin-1",
            model_version_id="model-1",
            title="branch-1",
            origin_time=NOW,
            state_snapshot={},
        )
    )
    session.commit()
    service = ShadowCognitionService(session)
    event = service.append_perception_event(
        project_id="project-1",
        branch_id="branch-1",
        event_type="user_message",
        occurred_at=NOW,
        source="conversation",
        idempotency_key="trigger",
        evidence={"content": "你好"},
    )
    state = service.ensure_initial_mental_state(
        project_id="project-1",
        branch_id="branch-1",
        state={"mood": "neutral"},
    )
    return service.create_pending_cycle(
        project_id="project-1",
        branch_id="branch-1",
        trigger_event_id=event.id,
        input_cutoff_at=NOW,
        starting_state_version_id=state.id,
        model_version_id="model-1",
    )


def _draft(*, wakeup: bool = False) -> CognitionDraft:
    suggestion = (
        WakeupSuggestion(
            wake_at=NOW + timedelta(hours=1),
            reason="稍后回看",
            idempotency_key="cycle-wakeup",
        )
        if wakeup
        else None
    )
    return CognitionDraft(
        private_content="先观察，不表达。",
        subjective_feelings={"calm": 0.8},
        attention_target={"topic": "用户意图"},
        desired_actions=({"kind": "observe"},),
        expression_decision=ExpressionDecision(express=False, reason="影子模式"),
        suggested_next_wakeup=suggestion,
    )


def test_handler_marks_running_then_persists_shadow_result_and_wakeup(
    database: Database,
) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler,
    )

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        job = JobService(session).enqueue(
            COGNITIVE_CYCLE_JOB_KIND,
            {"cycle_id": cycle.id},
        )
        observed: list[tuple[str, CognitionRequest]] = []

        class Generator:
            def generate(self, request: CognitionRequest) -> CognitionDraft:
                observed.append((cycle.status, request))
                return _draft(wakeup=True)

        create_cognitive_cycle_handler(Generator())(JobService(session), job)

        assert observed[0][0] == "running"
        assert observed[0][1].trigger_event["event_type"] == "user_message"
        assert observed[0][1].model_version_id == "model-1"
        assert cycle.status == "succeeded"
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 1
        assert session.scalar(select(func.count()).select_from(AgentWakeup)) == 1
        assert session.scalar(select(func.count()).select_from(MentalStateVersion)) == 1


def test_completed_message_cognition_requeues_deferred_conversation(
    database: Database,
) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler,
    )
    from moonlightbox.branches.actor import BRANCH_CONVERSATION_JOB_KIND

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        job = JobService(session).enqueue(
            COGNITIVE_CYCLE_JOB_KIND,
            {"cycle_id": cycle.id, "message_id": "message-1"},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                return _draft()

        create_cognitive_cycle_handler(Generator())(JobService(session), job)

        queued = session.scalar(
            select(Job).where(Job.kind == BRANCH_CONVERSATION_JOB_KIND)
        )
        assert queued is not None
        assert queued.payload["message_id"] == "message-1"
        assert queued.payload["cognitive_cycle_id"] == cycle.id


def test_elapsed_cognition_can_enqueue_proactive_expression(
    database: Database,
) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler,
    )
    from moonlightbox.branches.actor import BRANCH_CONVERSATION_JOB_KIND

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        trigger = session.get(PerceptionEvent, cycle.trigger_event_id)
        assert trigger is not None
        trigger.event_type = "elapsed_time"
        job = JobService(session).enqueue(
            COGNITIVE_CYCLE_JOB_KIND,
            {"cycle_id": cycle.id},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                return CognitionDraft(
                    private_content="这件事还压在心里，我想再确认一次。",
                    subjective_feelings={"summary": "仍然不安"},
                    attention_target={"summary": "关系是否结束"},
                    desired_actions=({"content": "再次确认"},),
                    expression_decision=ExpressionDecision(
                        express=True,
                        content="你真就这么决定了吗",
                        reason="关系事件仍未解决",
                    ),
                    suggested_next_wakeup=None,
                )

        create_cognitive_cycle_handler(Generator())(JobService(session), job)

        queued = session.scalar(
            select(Job).where(Job.kind == BRANCH_CONVERSATION_JOB_KIND)
        )
        assert queued is not None
        assert queued.payload == {
            "project_id": cycle.project_id,
            "branch_id": cycle.branch_id,
            "proactive": True,
            "cognitive_cycle_id": cycle.id,
        }


def test_new_user_message_during_generation_invalidates_result(database: Database) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler,
    )
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        job = JobService(session).enqueue(
            COGNITIVE_CYCLE_JOB_KIND,
            {"cycle_id": cycle.id},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                with Session(database.engine) as concurrent_session:
                    ShadowCognitionService(concurrent_session).append_perception_event(
                        project_id="project-1",
                        branch_id="branch-1",
                        event_type="user_message",
                        occurred_at=NOW + timedelta(seconds=1),
                        source="conversation",
                        idempotency_key="message-during-generation",
                    )
                return _draft(wakeup=True)

        create_cognitive_cycle_handler(Generator())(JobService(session), job)

        session.expire_all()
        assert session.get(CognitiveCycle, cycle.id).status == "invalidated"
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 0
        assert session.scalar(select(func.count()).select_from(AgentWakeup)) == 0


def test_generator_failure_marks_cycle_failed_and_raises_safe_error(
    database: Database,
) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler,
    )

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        job = JobService(session).enqueue(
            COGNITIVE_CYCLE_JOB_KIND,
            {"cycle_id": cycle.id},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                raise RuntimeError("不可泄漏的模型错误")

        with pytest.raises(JobHandlerError) as captured:
            create_cognitive_cycle_handler(Generator())(JobService(session), job)

        assert captured.value.code == "subject_cognition_failed"
        assert captured.value.safe_message == "主体认知任务处理失败"
        assert "不可泄漏" not in str(captured.value)
        assert cycle.status == "failed"
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 0


def test_failed_cognition_is_retried_with_bounded_durable_backoff(
    database: Database,
) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        resume_retryable_cognitive_jobs,
    )

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        cycle.status = "failed"
        cycle.completed_at = NOW
        job = Job(
            kind=COGNITIVE_CYCLE_JOB_KIND,
            payload={"cycle_id": cycle.id},
            status="failed",
            error_code="subject_cognition_failed",
            error_message="主体认知任务处理失败",
            updated_at=NOW,
        )
        session.add(job)
        session.commit()

        too_early = resume_retryable_cognitive_jobs(
            session, now=NOW + timedelta(seconds=59)
        )
        resumed = resume_retryable_cognitive_jobs(
            session, now=NOW + timedelta(minutes=1)
        )

        assert too_early == 0
        assert resumed == 1
        assert job.status == "queued"
        assert job.payload["automatic_retry_count"] == 1
        assert cycle.status == "pending"


def test_cognition_retry_stops_after_three_attempts(database: Database) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        resume_retryable_cognitive_jobs,
    )

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        cycle.status = "failed"
        cycle.completed_at = NOW
        job = Job(
            kind=COGNITIVE_CYCLE_JOB_KIND,
            payload={"cycle_id": cycle.id, "automatic_retry_count": 3},
            status="failed",
            error_code="subject_cognition_failed",
            error_message="主体认知任务处理失败",
            updated_at=NOW,
        )
        session.add(job)
        session.commit()

        resumed = resume_retryable_cognitive_jobs(
            session, now=NOW + timedelta(days=1)
        )

        assert resumed == 0
        assert job.status == "failed"
        assert cycle.status == "failed"


def test_handler_rejects_missing_cycle_id_without_touching_conversation_data(
    database: Database,
) -> None:
    from moonlightbox.agent.jobs import (
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler,
    )
    from moonlightbox.branches.models import BranchMessage

    with Session(database.engine) as session:
        _seed_cycle(session)
        before = session.scalar(select(func.count()).select_from(BranchMessage))
        job = JobService(session).enqueue(COGNITIVE_CYCLE_JOB_KIND, {})

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                pytest.fail("无效任务不应调用生成器")

        with pytest.raises(JobHandlerError) as captured:
            create_cognitive_cycle_handler(Generator())(JobService(session), job)

        assert captured.value.code == "invalid_subject_cognition_job"
        assert session.scalar(select(func.count()).select_from(BranchMessage)) == before
        assert session.scalar(select(func.count()).select_from(PerceptionEvent)) == 1


def _persist_unstructured_shadow_note(
    session: Session,
) -> tuple[CognitiveCycle, PrivateCognitionNote]:
    from moonlightbox.agent.jobs import create_cognitive_cycle_handler

    cycle = _seed_cycle(session)
    job = JobService(session).enqueue("subject_cognitive_cycle", {"cycle_id": cycle.id})

    class Generator:
        def generate(self, _request: CognitionRequest) -> CognitionDraft:
            return CognitionDraft(
                private_content="我有点担心，想先观察用户是否愿意继续谈。",
                subjective_feelings={},
                attention_target={},
                desired_actions=(),
                expression_decision=ExpressionDecision(express=False, reason="影子模式"),
                suggested_next_wakeup=None,
                structured_changes={"extraction_status": "pending"},
            )

    create_cognitive_cycle_handler(Generator())(JobService(session), job)
    note = session.get(PrivateCognitionNote, cycle.private_note_id)
    assert note is not None
    return cycle, note


def test_unstructured_shadow_note_enqueues_unique_extraction_job(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import COGNITIVE_EXTRACTION_JOB_KIND
    from moonlightbox.agent.jobs import create_cognitive_cycle_handler

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        JobService(session).enqueue_unique(
            COGNITIVE_EXTRACTION_JOB_KIND,
            {
                "project_id": cycle.project_id,
                "branch_id": cycle.branch_id,
                "cycle_id": cycle.id,
                "note_id": "预先占位",
            },
            dedupe_key=f"{COGNITIVE_EXTRACTION_JOB_KIND}:{cycle.id}",
        )
        cycle_job = JobService(session).enqueue(
            "subject_cognitive_cycle",
            {"cycle_id": cycle.id},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                return _draft()

        create_cognitive_cycle_handler(Generator())(JobService(session), cycle_job)

        jobs = list(
            session.scalars(
                select(Job).where(Job.kind == COGNITIVE_EXTRACTION_JOB_KIND)
            )
        )
        assert len(jobs) == 1


def test_stale_cognition_does_not_enqueue_extraction_job(database: Database) -> None:
    from moonlightbox.agent.extraction import COGNITIVE_EXTRACTION_JOB_KIND
    from moonlightbox.agent.jobs import create_cognitive_cycle_handler
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        cycle_job = JobService(session).enqueue(
            "subject_cognitive_cycle",
            {"cycle_id": cycle.id},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                with Session(database.engine) as concurrent_session:
                    ShadowCognitionService(concurrent_session).append_perception_event(
                        project_id=cycle.project_id,
                        branch_id=cycle.branch_id,
                        event_type="user_message",
                        occurred_at=NOW + timedelta(seconds=1),
                        source="conversation",
                        idempotency_key="stale-extraction",
                    )
                return _draft()

        create_cognitive_cycle_handler(Generator())(JobService(session), cycle_job)

        assert (
            session.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.kind == COGNITIVE_EXTRACTION_JOB_KIND)
            )
            == 0
        )


def test_already_stale_cycle_skips_expensive_generation(database: Database) -> None:
    from moonlightbox.agent.jobs import create_cognitive_cycle_handler
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        ShadowCognitionService(session).append_perception_event(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            event_type="user_message",
            occurred_at=NOW + timedelta(seconds=1),
            source="conversation",
            idempotency_key="newer-before-worker-starts",
            evidence={"content": "补充一句"},
        )
        cycle_job = JobService(session).enqueue(
            "subject_cognitive_cycle",
            {"cycle_id": cycle.id},
        )

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                pytest.fail("已过期的认知周期不应再调用云模型")

        create_cognitive_cycle_handler(Generator())(
            JobService(session),
            cycle_job,
        )

        assert session.get(CognitiveCycle, cycle.id).status == "invalidated"


def test_extraction_handler_strictly_updates_note_and_audit_only(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import (
        COGNITIVE_EXTRACTION_JOB_KIND,
        CognitiveStructureExtractor,
        create_cognitive_extraction_handler,
    )

    with Session(database.engine) as session:
        cycle, note = _persist_unstructured_shadow_note(session)
        job = JobService(session).find_latest_by_payload(
            COGNITIVE_EXTRACTION_JOB_KIND,
            "cycle_id",
            cycle.id,
        )
        assert job is not None
        before_counts = (
            session.scalar(select(func.count()).select_from(MentalStateVersion)),
            session.scalar(select(func.count()).select_from(AgentGoal)),
            session.scalar(select(func.count()).select_from(AgentIntention)),
        )

        class JsonGenerator:
            def generate_json(self, **kwargs: object) -> dict[str, object]:
                assert kwargs["model_version_id"] == "model-1"
                payload = kwargs["payload"]
                assert isinstance(payload, dict)
                assert payload["private_cognition"] == note.content
                return {
                    "subjective_feelings": {"担心": 0.7},
                    "attention_target": {"对象": "用户继续交流的意愿"},
                    "desired_actions": [{"动作": "观察"}],
                    "mental_state_changes": {"警觉": "提高"},
                    "goal_changes": [{"目标": "理解用户意图"}],
                    "intention_changes": [{"意图": "暂不表达"}],
                    "next_wakeup": None,
                }

        handler = create_cognitive_extraction_handler(
            CognitiveStructureExtractor(JsonGenerator())
        )
        handler(JobService(session), job)

        session.refresh(note)
        session.refresh(cycle)
        assert note.subjective_feelings == {"担心": 0.7}
        assert note.attention_target == {"对象": "用户继续交流的意愿"}
        assert note.desired_actions == [{"动作": "观察"}]
        assert cycle.structured_changes == {
            "extraction_status": "completed",
            "mental_state_changes": {"警觉": "提高"},
            "goal_changes": [{"目标": "理解用户意图"}],
            "intention_changes": [{"意图": "暂不表达"}],
            "next_wakeup": None,
        }
        assert cycle.status == "succeeded"
        assert before_counts == (
            session.scalar(select(func.count()).select_from(MentalStateVersion)),
            session.scalar(select(func.count()).select_from(AgentGoal)),
            session.scalar(select(func.count()).select_from(AgentIntention)),
        )


def test_extraction_rejects_unsupported_field_and_preserves_shadow_state(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import (
        CognitiveStructureExtractor,
        create_cognitive_extraction_handler,
    )

    with Session(database.engine) as session:
        cycle, note = _persist_unstructured_shadow_note(session)
        job = JobService(session).find_latest_by_payload(
            "subject_cognition_extract",
            "cycle_id",
            cycle.id,
        )
        assert job is not None
        original_note = (
            dict(note.subjective_feelings),
            dict(note.attention_target),
            list(note.desired_actions),
        )
        before_counts = (
            session.scalar(select(func.count()).select_from(MentalStateVersion)),
            session.scalar(select(func.count()).select_from(AgentGoal)),
            session.scalar(select(func.count()).select_from(AgentIntention)),
        )

        class JsonGenerator:
            def generate_json(self, **_kwargs: object) -> dict[str, object]:
                return {
                    "subjective_feelings": {},
                    "attention_target": {},
                    "desired_actions": [],
                    "mental_state_changes": {},
                    "goal_changes": [],
                    "intention_changes": [],
                    "next_wakeup": None,
                    "invented_fact": "用户一定会回来",
                }

        with pytest.raises(JobHandlerError) as captured:
            create_cognitive_extraction_handler(
                CognitiveStructureExtractor(JsonGenerator())
            )(JobService(session), job)

        session.refresh(cycle)
        session.refresh(note)
        assert captured.value.code == "subject_cognition_extraction_failed"
        assert cycle.status == "succeeded"
        assert cycle.structured_changes == {
            "extraction_status": "failed",
            "error_code": "invalid_extraction_output",
        }
        assert (
            note.subjective_feelings,
            note.attention_target,
            note.desired_actions,
        ) == original_note
        assert before_counts == (
            session.scalar(select(func.count()).select_from(MentalStateVersion)),
            session.scalar(select(func.count()).select_from(AgentGoal)),
            session.scalar(select(func.count()).select_from(AgentIntention)),
        )


def test_extraction_normalizes_concise_string_fields_without_inventing_content(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import CognitiveStructureExtractor

    with Session(database.engine) as session:
        _cycle, note = _persist_unstructured_shadow_note(session)

        class JsonGenerator:
            def generate_json(self, **_kwargs: object) -> dict[str, object]:
                return {
                    "subjective_feelings": "有些担心",
                    "attention_target": "用户的语气",
                    "desired_actions": "先等待",
                    "mental_state_changes": ["担心略微增加", "信任保持不变"],
                    "goal_changes": ["继续理解用户"],
                    "intention_changes": "暂时不发送消息",
                    "next_wakeup": None,
                }

        result = CognitiveStructureExtractor(JsonGenerator()).extract(note)

        assert result.subjective_feelings == {"summary": "有些担心"}
        assert result.attention_target == {"summary": "用户的语气"}
        assert result.desired_actions == [{"content": "先等待"}]
        assert result.mental_state_changes == {
            "items": ["担心略微增加", "信任保持不变"]
        }
        assert result.goal_changes == [{"content": "继续理解用户"}]
        assert result.intention_changes == [{"content": "暂时不发送消息"}]


def test_extraction_rejects_cross_branch_scope_without_calling_generator(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import (
        CognitiveStructureExtractor,
        create_cognitive_extraction_handler,
    )

    with Session(database.engine) as session:
        cycle, _note = _persist_unstructured_shadow_note(session)
        job = JobService(session).find_latest_by_payload(
            "subject_cognition_extract",
            "cycle_id",
            cycle.id,
        )
        assert job is not None
        job.payload = {**job.payload, "branch_id": "其他分支"}
        session.commit()

        class JsonGenerator:
            def generate_json(self, **_kwargs: object) -> dict[str, object]:
                pytest.fail("跨分支任务不应调用生成器")

        with pytest.raises(JobHandlerError) as captured:
            create_cognitive_extraction_handler(
                CognitiveStructureExtractor(JsonGenerator())
            )(JobService(session), job)

        session.refresh(cycle)
        assert captured.value.code == "invalid_subject_cognition_extraction_job"
        assert cycle.status == "succeeded"
        assert cycle.structured_changes == {"extraction_status": "pending"}


def test_active_extraction_applies_state_goals_intentions_and_wakeup(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import (
        create_cognitive_extraction_handler,
    )

    with Session(database.engine) as session:
        cycle, _note = _persist_unstructured_shadow_note(session)
        branch = session.get(Branch, cycle.branch_id)
        starting = session.get(MentalStateVersion, cycle.starting_state_version_id)
        assert branch is not None
        assert starting is not None
        branch.subject_agent_mode = "active"
        starting.state = {"mood": "neutral", "focus": "existing"}
        session.commit()
        job = JobService(session).find_latest_by_payload(
            "subject_cognition_extract",
            "cycle_id",
            cycle.id,
        )
        assert job is not None
        before_fact_count = session.scalar(
            select(func.count()).select_from(BranchMemoryItem)
        )

        class Extractor:
            def extract(self, _note: PrivateCognitionNote) -> object:
                from moonlightbox.agent.extraction import CognitiveExtractionResult

                return CognitiveExtractionResult(
                    subjective_feelings={"担心": 0.6},
                    attention_target={"对象": "用户"},
                    desired_actions=[{"type": "wait"}],
                    mental_state_changes={"mood": "concerned"},
                    goal_changes=[
                        {
                            "goal_type": "relationship",
                            "content": "保持诚实交流",
                            "priority": 0.8,
                            "status": "active",
                        }
                    ],
                    intention_changes=[
                        {
                            "intention_type": "respond",
                            "content": "下次先确认对方意愿",
                            "status": "active",
                        }
                    ],
                    next_wakeup={
                        "wake_at": (NOW + timedelta(hours=2)).isoformat(),
                        "reason": "重新确认是否需要表达",
                        "idempotency_key": "active-next-review",
                    },
                )

        create_cognitive_extraction_handler(Extractor())(JobService(session), job)

        session.expire_all()
        states = list(
            session.scalars(
                select(MentalStateVersion)
                .where(MentalStateVersion.branch_id == cycle.branch_id)
                .order_by(MentalStateVersion.version)
            )
        )
        goal = session.scalar(
            select(AgentGoal).where(AgentGoal.branch_id == cycle.branch_id)
        )
        intention = session.scalar(
            select(AgentIntention).where(AgentIntention.branch_id == cycle.branch_id)
        )
        wakeup = session.scalar(
            select(AgentWakeup).where(
                AgentWakeup.branch_id == cycle.branch_id,
                AgentWakeup.idempotency_key == "active-next-review",
            )
        )
        applied_cycle = session.get(CognitiveCycle, cycle.id)
        assert len(states) == 2
        assert states[0].is_current is False
        assert states[1].is_current is True
        assert states[1].state == {"mood": "concerned", "focus": "existing"}
        assert states[1].source_cycle_id == cycle.id
        assert goal is not None
        assert goal.source["cycle_id"] == cycle.id
        assert goal.source["event_id"] == cycle.trigger_event_id
        assert goal.model_version_id == cycle.model_version_id
        assert intention is not None
        assert intention.trigger_event_id == cycle.trigger_event_id
        assert intention.evidence["cycle_id"] == cycle.id
        assert intention.model_version_id == cycle.model_version_id
        assert wakeup is not None
        assert wakeup.event_id == cycle.trigger_event_id
        assert wakeup.evidence["cycle_id"] == cycle.id
        assert wakeup.model_version_id == cycle.model_version_id
        assert applied_cycle is not None
        assert applied_cycle.structured_changes["state_apply_status"] == "applied"
        assert session.scalar(
            select(func.count()).select_from(BranchMemoryItem)
        ) == before_fact_count


def test_active_cloud_extraction_is_authoritative_without_local_reinterpretation(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import (
        COGNITIVE_EXTRACTION_JOB_KIND,
        create_cognitive_extraction_handler,
    )
    from moonlightbox.agent.jobs import create_cognitive_cycle_handler

    with Session(database.engine) as session:
        cycle = _seed_cycle(session)
        branch = session.get(Branch, cycle.branch_id)
        assert branch is not None
        branch.subject_agent_mode = "active"
        session.commit()
        cycle_job = JobService(session).enqueue(
            "subject_cognitive_cycle", {"cycle_id": cycle.id}
        )

        authoritative = {
            "subjective_feelings": {"summary": "震惊和委屈", "intensity": 0.95},
            "attention_target": {"summary": "关系是否结束"},
            "desired_actions": [{"content": "继续确认"}],
            "mental_state_changes": {
                "emotional_state": {
                    "valence": -0.9,
                    "arousal": 0.95,
                    "intensity": 0.95,
                },
                "conversation_drive": {
                    "follow_up_needed": True,
                    "remaining_attempts": 2,
                },
            },
            "goal_changes": [
                {
                    "goal_type": "relationship_follow_up",
                    "content": "确认关系是否真的结束",
                    "priority": 0.95,
                    "status": "active",
                }
            ],
            "intention_changes": [
                {
                    "intention_type": "follow_up",
                    "content": "稍后再确认一次",
                    "status": "active",
                    "expression_plan": {"max_attempts": 2},
                }
            ],
            "next_wakeup": None,
        }

        class Generator:
            def generate(self, _request: CognitionRequest) -> CognitionDraft:
                return CognitionDraft(
                    private_content="我一下慌了，这不是能平静带过的话。",
                    subjective_feelings={"summary": "震惊和委屈"},
                    attention_target={"summary": "关系是否结束"},
                    desired_actions=({"content": "继续确认"},),
                    expression_decision=ExpressionDecision(
                        express=True,
                        content="你是认真的吗",
                        reason="关系威胁很高",
                    ),
                    suggested_next_wakeup=None,
                    structured_changes={
                        "extraction_status": "authoritative_ready",
                        "authoritative_extraction": authoritative,
                    },
                )

        create_cognitive_cycle_handler(Generator())(
            JobService(session), cycle_job
        )
        extraction_job = JobService(session).find_latest_by_payload(
            COGNITIVE_EXTRACTION_JOB_KIND,
            "cycle_id",
            cycle.id,
        )
        assert extraction_job is not None

        class LocalExtractorMustNotRun:
            def extract(self, _note: PrivateCognitionNote) -> object:
                pytest.fail("云端认知结构不能再被本地模型二次解释")

        create_cognitive_extraction_handler(LocalExtractorMustNotRun())(
            JobService(session), extraction_job
        )

        session.expire_all()
        current = session.scalar(
            select(MentalStateVersion).where(
                MentalStateVersion.branch_id == cycle.branch_id,
                MentalStateVersion.is_current.is_(True),
            )
        )
        goal = session.scalar(select(AgentGoal))
        intention = session.scalar(select(AgentIntention))
        assert current is not None
        assert current.version == 2
        assert current.state["emotional_state"]["intensity"] == 0.95
        assert current.state["conversation_drive"]["remaining_attempts"] == 2
        assert goal is not None and goal.status == "active"
        assert intention is not None and intention.status == "active"


def test_active_extraction_marks_stale_without_overwriting_new_state(
    database: Database,
) -> None:
    from moonlightbox.agent.extraction import (
        CognitiveExtractionResult,
        create_cognitive_extraction_handler,
    )

    with Session(database.engine) as session:
        cycle, _note = _persist_unstructured_shadow_note(session)
        branch = session.get(Branch, cycle.branch_id)
        starting = session.get(MentalStateVersion, cycle.starting_state_version_id)
        assert branch is not None
        assert starting is not None
        branch.subject_agent_mode = "active"
        starting.is_current = False
        newer = MentalStateVersion(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            version=2,
            state={"mood": "newer"},
            previous_version_id=starting.id,
            evidence={"source": "concurrent"},
            is_current=True,
            model_version_id=cycle.model_version_id,
            model_protocol_version=cycle.model_protocol_version,
        )
        session.add(newer)
        session.commit()
        job = JobService(session).find_latest_by_payload(
            "subject_cognition_extract",
            "cycle_id",
            cycle.id,
        )
        assert job is not None

        class Extractor:
            def extract(self, _note: PrivateCognitionNote) -> CognitiveExtractionResult:
                return CognitiveExtractionResult(
                    subjective_feelings={},
                    attention_target={},
                    desired_actions=[],
                    mental_state_changes={"mood": "stale-result"},
                    goal_changes=[{"goal_type": "temporary", "content": "不应创建"}],
                    intention_changes=[],
                    next_wakeup=None,
                )

        create_cognitive_extraction_handler(Extractor())(JobService(session), job)

        session.expire_all()
        stale_cycle = session.get(CognitiveCycle, cycle.id)
        current = session.scalar(
            select(MentalStateVersion).where(
                MentalStateVersion.branch_id == cycle.branch_id,
                MentalStateVersion.is_current.is_(True),
            )
        )
        assert stale_cycle is not None
        assert stale_cycle.structured_changes["state_apply_status"] == "stale"
        assert current is not None
        assert current.id == newer.id
        assert current.state == {"mood": "newer"}
        assert session.scalar(select(func.count()).select_from(AgentGoal)) == 0
