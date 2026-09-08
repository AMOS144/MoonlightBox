from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from moonlightbox.agent.extraction import COGNITIVE_EXTRACTION_JOB_KIND
from moonlightbox.agent.jobs import COGNITIVE_CYCLE_JOB_KIND
from moonlightbox.agent.models import (
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
    PrivateCognitionNote,
)
from moonlightbox.agent.types import (
    CognitionDraft,
    ExpressionDecision,
    FusedAgentTurn,
)
from moonlightbox.branches.actor import (
    BRANCH_CONVERSATION_JOB_KIND,
    BranchConversationActor,
    deliver_due_pending_bubbles,
)
from moonlightbox.branches.actor_models import (
    ConversationActorState,
    ConversationExpressionPlan,
    ConversationPendingBubble,
)
from moonlightbox.branches.context import ContextPacket
from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
)
from moonlightbox.branches.episodes import CONTINUAL_MEMORY_JOB_KIND
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn, ReplyStructureError
from moonlightbox.branches.service import BranchService
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.media.selection import MediaSelection
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import func, select
from sqlalchemy.orm import Session


@pytest.fixture
def database(tmp_path: Path) -> Database:
    import moonlightbox.api as api

    assert api.app is not None
    database = Database(f"sqlite:///{tmp_path / 'active-subject-agent.db'}")
    Base.metadata.create_all(database.engine)
    return database


def _seed_branch(
    session: Session,
    *,
    mode: str,
    training_config: dict[str, object] | None = None,
) -> Branch:
    session.add(Project(id="project-1", name="主动主体"))
    session.flush()
    session.add(
        ModelVersion(
            id="model-1",
            project_id="project-1",
            base_model="test",
            adapter_path="test",
            dataset_hash="hash",
            metrics={},
            training_config=training_config or {},
        )
    )
    session.add(
        EventNode(
            id="event-1",
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
    branch = Branch(
        id="branch-1",
        project_id="project-1",
        origin_event_id="event-1",
        model_version_id="model-1",
        title="主动主体",
        origin_time=datetime(2026, 7, 23, tzinfo=UTC),
        state_snapshot={},
        baseline_status="ready",
        lifecycle_status="active",
        subject_agent_mode=mode,
    )
    session.add(branch)
    session.commit()
    return branch


def _packet(content: str) -> ContextPacket:
    return ContextPacket(
        persona="她",
        cutoff="2026-07-23",
        history=(),
        current_user_content=content,
        memories=(),
        allowed_sticker_ids=(),
        reply_protocol="compact",
    )


def _persona_text_packet(content: str) -> ContextPacket:
    return ContextPacket(
        persona="她",
        cutoff="2026-07-23",
        history=(),
        current_user_content=content,
        memories=(),
        allowed_sticker_ids=(),
        reply_protocol="persona_text",
    )


def _fused_turn(*, express: bool) -> FusedAgentTurn:
    decision = ExpressionDecision(
        express=express,
        reason="愿意回应" if express else "现在保持沉默",
    )
    return FusedAgentTurn(
        cognition=CognitionDraft(
            private_content="我想先确认自己的感受。",
            subjective_feelings={"犹豫": 0.4},
            attention_target={"对象": "用户消息"},
            desired_actions=({"type": "observe"},),
            expression_decision=decision,
            suggested_next_wakeup=None,
            structured_changes={"extraction_status": "pending"},
        ),
        expression_decision=decision,
        reply=(
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="我在听", delay_ms=0),),
                raw_output="<bubble>我在听</bubble>",
            )
            if express
            else None
        ),
    )


class FusedGenerator:
    def __init__(self, turn: FusedAgentTurn) -> None:
        self.turn = turn
        self.fused_calls = 0

    def generate(
        self,
        _model_version_id: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        pytest.fail("active 分支不得调用直接回复生成")

    def generate_fused(
        self,
        _model_version_id: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
        *,
        allowed_sticker_ids: tuple[str, ...],
    ) -> FusedAgentTurn:
        assert allowed_sticker_ids == ()
        self.fused_calls += 1
        return self.turn


class GroundingRetryFusedGenerator(FusedGenerator):
    def __init__(self) -> None:
        turn = _fused_turn(express=True)
        super().__init__(
            FusedAgentTurn(
                cognition=turn.cognition,
                expression_decision=turn.expression_decision,
                reply=GeneratedReplyTurn(
                    bubbles=(GeneratedBubble(content="我现在正在开会", delay_ms=0),),
                    raw_output="我现在正在开会",
                ),
            )
        )
        self.retry_prompts: list[str] = []

    def generate(
        self,
        _model_version_id: str,
        system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        self.retry_prompts.append(system_prompt)
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="干嘛", delay_ms=0),),
            raw_output="干嘛",
        )


class PersonaTextGenerator:
    def __init__(self, response: str = "干嘛") -> None:
        self.generate_calls = 0
        self.response = response
        self.system_prompts: list[str] = []
        self.message_batches: list[list[dict[str, str]]] = []

    def generate(
        self,
        _model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        self.generate_calls += 1
        self.system_prompts.append(system_prompt)
        self.message_batches.append(messages)
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content=self.response, delay_ms=0),),
            raw_output=self.response,
        )

    def generate_fused(self, *_args: object, **_kwargs: object) -> FusedAgentTurn:
        pytest.fail("persona-text 模型不得生成私有认知 JSON")


class MultiBubblePersonaTextGenerator(PersonaTextGenerator):
    def generate(
        self,
        _model_version_id: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        self.generate_calls += 1
        return GeneratedReplyTurn(
            bubbles=(
                GeneratedBubble(content="第一句", delay_ms=0),
                GeneratedBubble(content="第二句", delay_ms=1200),
            ),
            raw_output="第一句\n第二句",
        )


class GroundingRetryPersonaTextGenerator:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def generate(
        self,
        _model_version_id: str,
        system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        self.system_prompts.append(system_prompt)
        content = "我现在正在开会" if len(self.system_prompts) == 1 else "干嘛"
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content=content, delay_ms=0),),
            raw_output=content,
        )


class StateDraftPersonaTextGenerator:
    def __init__(self) -> None:
        self.drafts: list[str] = []

    def generate(self, *_args: object, **_kwargs: object) -> GeneratedReplyTurn:
        pytest.fail("可信状态问题不得进入自由事实生成")

    def rewrite_content_draft(
        self,
        _model_version_id: str,
        draft: str,
    ) -> GeneratedReplyTurn:
        self.drafts.append(draft)
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content=draft, delay_ms=0),),
            raw_output=draft,
        )


class UnsafeStylePersonaTextGenerator(StateDraftPersonaTextGenerator):
    def rewrite_content_draft(
        self,
        _model_version_id: str,
        draft: str,
    ) -> GeneratedReplyTurn:
        self.drafts.append(draft)
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="我现在正在开会", delay_ms=0),),
            raw_output="我现在正在开会",
        )


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_actor_only_applies_semantically_selected_media_behavior(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
) -> None:
    policy = {
        "version": "historical-media-behavior-v1",
        "total_target_messages": 100,
        "modality_counts": {modality: 100},
        "biases": {modality: 20},
        "weights": {modality: {}},
    }
    with Session(database.engine) as session:
        branch = _seed_branch(
            session,
            mode="active",
            training_config={"media_behavior_policy": policy},
        )
        actor = BranchConversationActor(session, PersonaTextGenerator())
        monkeypatch.setattr(
            "moonlightbox.branches.actor.select_reusable_media",
            lambda *_args, **_kwargs: MediaSelection(
                asset_id="approved-asset",
                modality=modality,
                semantic_text="我刚到家",
                similarity=0.95,
            ),
        )

        reply = actor._apply_learned_media_behavior(
            branch,
            GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="我刚到家", delay_ms=0),),
                raw_output="我刚到家",
            ),
            user_content="到家了吗",
            proactive=False,
            seed="media-test",
        )

        assert reply.policy_appended is True
        if modality == "audio":
            assert [(item.type, item.asset_id) for item in reply.bubbles] == [
                ("audio", "approved-asset")
            ]
        else:
            assert [(item.type, item.asset_id) for item in reply.bubbles] == [
                ("text", None),
                ("image", "approved-asset"),
            ]


def test_shadow_persona_text_model_can_use_direct_expression_generation(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "在吗",
        )
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assert generator.generate_calls == 1
        assistant = session.scalar(select(BranchMessage).where(BranchMessage.role == "assistant"))
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert assistant.generation_metadata["grounding_retry_count"] == 0


def test_actor_failure_moves_proactive_review_into_bounded_backoff(
    database: Database,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        state = ConversationActorState(
            branch_id="branch-1",
            status="thinking",
            next_review_at=datetime.now(UTC) - timedelta(seconds=1),
            inhibition={"score": 0.05},
        )
        session.add(state)
        session.commit()
        before = datetime.now(UTC)

        BranchConversationActor(session, PersonaTextGenerator()).reset_after_failure(
            "branch-1"
        )

        session.refresh(state)
        assert state.status == "idle"
        assert state.next_review_at is not None
        assert state.next_review_at.replace(tzinfo=UTC) >= before + timedelta(minutes=5)
        assert state.inhibition["failure_count"] == 1


def test_active_persona_text_generates_from_full_context_without_waiting_for_cognition(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        user = BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "你怎么想",
        )
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assert generator.generate_calls == 1
        assert generator.message_batches[-1][-1] == {
            "role": "user",
            "content": "你怎么想",
        }
        assert session.get(BranchMessage, user.id).observed_at is not None
        assert session.scalar(
            select(func.count()).select_from(BranchMessage).where(
                BranchMessage.role == "assistant"
            )
        ) == 1


def test_persona_text_builds_one_background_cognition_cycle_for_contribution_cluster(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        service = BranchService(session, PersonaTextGenerator())
        first = service.add_user_message("project-1", "branch-1", "并不知道")
        second = service.add_user_message("project-1", "branch-1", "非常纠结！")

        assert session.scalar(select(func.count()).select_from(CognitiveCycle)) == 0

        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )
        actor.process("project-1", "branch-1", lease_owner="worker-1")

        cycles = list(session.scalars(select(CognitiveCycle)))
        assert len(cycles) == 1
        event = session.get(PerceptionEvent, cycles[0].trigger_event_id)
        assert event is not None
        assert event.event_type == "user_contribution_cluster"
        assert event.evidence["content"] == "并不知道\n非常纠结！"
        assert event.evidence["branch_message_ids"] == [first.id, second.id]


def test_candidate_preview_does_not_use_base_cognition_as_public_content(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="preview",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        user = BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "你怎么想",
        )
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(
            select(BranchMessage).where(BranchMessage.role == "assistant")
        )
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert generator.generate_calls == 1
        assert generator.message_batches[-1][-1]["content"] == "你怎么想"
        assert session.get(BranchMessage, user.id).observed_at is not None


def test_elapsed_cognitive_cycle_can_drive_proactive_persona_expression(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.agent.service import ShadowCognitionService

    with Session(database.engine) as session:
        branch = _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        cognition = ShadowCognitionService(session)
        event = cognition.append_perception_event(
            project_id=branch.project_id,
            branch_id=branch.id,
            event_type="elapsed_time",
            occurred_at=datetime.now(UTC),
            source="scheduled_wakeup",
            idempotency_key="relationship-follow-up-1",
            evidence={"reason": "关系事件仍未解决"},
        )
        state = cognition.ensure_initial_mental_state(
            project_id=branch.project_id,
            branch_id=branch.id,
            state={
                "conversation_drive": {
                    "follow_up_needed": True,
                    "remaining_attempts": 2,
                }
            },
        )
        cycle = cognition.create_pending_cycle(
            project_id=branch.project_id,
            branch_id=branch.id,
            trigger_event_id=event.id,
            input_cutoff_at=datetime.now(UTC),
            starting_state_version_id=state.id,
            model_version_id=branch.model_version_id,
        )
        cycle.status = "succeeded"
        cycle.final_decision = {
            "express": True,
            "content": "你真就这么决定了吗",
            "reason": "高强度关系余波仍未解决",
        }
        session.commit()

        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process(
            branch.project_id,
            branch.id,
            proactive=True,
            cognitive_cycle_id=cycle.id,
            lease_owner="worker-1",
        )

        assistant = session.scalar(
            select(BranchMessage).where(BranchMessage.role == "assistant")
        )
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert assistant.is_proactive is True


def test_active_persona_text_can_learn_to_leave_low_signal_message_unanswered(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={
                "reply_protocol_version": "persona-text-v1",
                "expression_policy": {
                    "version": "historical-expression-nb-v1",
                    "enabled": True,
                    "positive_count": 100,
                    "negative_count": 20,
                    "bias": -10,
                    "weights": {},
                    "silence_threshold": 0.2,
                },
            },
        )
        generator = PersonaTextGenerator()
        user = BranchService(session, generator).add_user_message(
            "project-1",
            "branch-1",
            "请告诉我你的想法",
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assert generator.generate_calls == 0
        assert session.get(BranchMessage, user.id).observed_at is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(BranchMessage)
                .where(BranchMessage.role == "assistant")
            )
            == 0
        )


def test_shadow_compact_model_uses_backfilled_silence_policy(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={
                "reply_protocol_version": "high-fidelity-compact-bubble-v2",
                "expression_policy": {
                    "version": "historical-expression-nb-v1",
                    "enabled": True,
                    "positive_count": 100,
                    "negative_count": 20,
                    "bias": -10,
                    "weights": {},
                    "silence_threshold": 0.2,
                },
            },
        )
        generator = PersonaTextGenerator()
        user = BranchService(session, generator).add_user_message(
            "project-1",
            "branch-1",
            "嗯",
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assert generator.generate_calls == 0
        assert session.get(BranchMessage, user.id).observed_at is not None


def test_historical_quote_policy_creates_real_quote_message(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={
                "conversation_action_policy": {
                    "version": "historical-conversation-actions-v1",
                    "enabled": True,
                    "total_target_messages": 20,
                    "action_counts": {"quote": 20},
                    "biases": {"quote": 20},
                    "weights": {"quote": {}},
                }
            },
        )
        generator = PersonaTextGenerator()
        BranchService(session, generator).add_user_message(
            "project-1", "branch-1", "你说的是这一条吗"
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(
            select(BranchMessage).where(BranchMessage.role == "assistant")
        )
        assert assistant is not None and assistant.type == "quote"
        assert '"quoted_content": "你说的是这一条吗"' in assistant.content


def test_historical_retract_policy_removes_sent_bubble_server_side(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={
                "conversation_action_policy": {
                    "version": "historical-conversation-actions-v1",
                    "enabled": True,
                    "total_target_messages": 20,
                    "action_counts": {"retract": 20},
                    "biases": {"retract": 20},
                    "weights": {"retract": {}},
                }
            },
        )
        generator = PersonaTextGenerator()
        BranchService(session, generator).add_user_message(
            "project-1", "branch-1", "刚才那句"
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")
        assert deliver_due_pending_bubbles(
            session,
            branch_id="branch-1",
            now=datetime.now(UTC) + timedelta(seconds=20),
        ) == 1

        assistants = list(
            session.scalars(
                select(BranchMessage)
                .where(BranchMessage.role == "assistant")
                .order_by(BranchMessage.sequence)
            )
        )
        assert assistants[0].generation_metadata["retracted"] is True
        assert assistants[1].type == "system"
        assert assistants[1].content == "撤回了一条消息"


def test_trusted_current_state_is_context_for_persona_expression(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "现在可以电话吗",
        )
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        packet = _persona_text_packet("现在可以电话吗")
        packet = ContextPacket(
            **{
                **packet.__dict__,
                "branch_state": {
                    "situational_state": {
                        "source": "external_observation",
                        "evidence_ids": ["observation-1"],
                        "observed_at": datetime.now(UTC).isoformat(),
                        "valid_until": datetime(2099, 1, 1, tzinfo=UTC).isoformat(),
                        "values": {"availability": "并不方便电话"},
                    }
                },
            }
        )
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, _content, **_kwargs: packet,
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(select(BranchMessage).where(BranchMessage.role == "assistant"))
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert "当前是否方便联系：并不方便电话" in generator.system_prompts[-1]


def test_active_relationship_belief_is_context_for_persona_expression(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.branches.context import ContextMemory

    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "你现在想怎么处理我们的关系",
        )
        generator = PersonaTextGenerator("我现在需要保持距离")
        actor = BranchConversationActor(session, generator)
        packet = _persona_text_packet("你现在想怎么处理我们的关系")
        packet = ContextPacket(
            **{
                **packet.__dict__,
                "active_beliefs": (
                    ContextMemory(
                        "belief-distance",
                        "belief",
                        "本人目前需要保持距离",
                        authority="subjective",
                    ),
                ),
            }
        )
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, _content, **_kwargs: packet,
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(
            select(BranchMessage).where(BranchMessage.role == "assistant")
        )
        assert assistant is not None
        assert assistant.content == "我现在需要保持距离"
        assert "本人目前需要保持距离" in generator.system_prompts[-1]


def test_persona_text_generation_does_not_require_cognitive_content_draft(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "你怎么想",
        )
        assert session.scalar(select(CognitiveCycle)) is None
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(
            select(BranchMessage).where(BranchMessage.role == "assistant")
        )
        assert assistant is not None
        assert assistant.content == "干嘛"


def test_persona_grounding_failure_retries_with_full_context(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "怎么不回电话",
        )
        generator = GroundingRetryPersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(
            select(BranchMessage).where(BranchMessage.role == "assistant")
        )
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert assistant.generation_metadata["grounding_retry_count"] == 1


def test_persona_text_silence_is_decided_by_expression_policy_not_base_cognition(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "嗯",
        )
        assert session.scalar(select(CognitiveCycle)) is None
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assert generator.generate_calls == 1
        assert session.scalar(
            select(func.count()).select_from(BranchMessage).where(
                BranchMessage.role == "assistant"
            )
        ) == 1


def test_persona_text_retries_locally_after_unsupported_current_state(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "怎么不回电话",
        )
        generator = GroundingRetryPersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assert len(generator.system_prompts) == 2
        assert "证据没有明确写出的本人当前状态一律不要说" in generator.system_prompts[1]
        assistant = session.scalar(select(BranchMessage).where(BranchMessage.role == "assistant"))
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert assistant.generation_metadata["grounding_retry_count"] == 1
        assert "当前状态" in assistant.generation_metadata["grounding_retry_reason"]
        assert "我现在正在开会" not in str(assistant.generation_metadata)
        episode = session.scalar(select(BranchMemoryEpisode))
        assert episode is not None
        assert [bubble["content"] for bubble in episode.assistant_bubbles] == ["干嘛"]
        assert "我现在正在开会" not in episode.user_content
        assert "我现在正在开会" not in str(episode.assistant_bubbles)
        assert session.scalar(select(func.count()).select_from(BranchMemoryItem)) == 0
        assert (
            session.scalar(
                select(func.count()).select_from(Job).where(Job.kind == CONTINUAL_MEMORY_JOB_KIND)
            )
            == 1
        )


def test_actor_episode_keeps_every_message_that_caused_the_reply(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        service = BranchService(session, PersonaTextGenerator())
        first = service.add_user_message("project-1", "branch-1", "我周末去上海")
        second = service.add_user_message("project-1", "branch-1", "你要一起吗")
        generator = PersonaTextGenerator()
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        episode = session.scalar(select(BranchMemoryEpisode))
        assert episode is not None
        assert episode.user_content == "我周末去上海\n你要一起吗"
        assert [item["message_id"] for item in episode.user_messages] == [
            first.id,
            second.id,
        ]
        assert [item["turn_id"] for item in episode.user_messages] == [
            first.turn_id,
            second.turn_id,
        ]


def test_server_delivers_multi_bubble_turn_only_when_each_bubble_is_due(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        generator = MultiBubblePersonaTextGenerator()
        BranchService(session, generator).add_user_message("project-1", "branch-1", "分两句说")
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistants = list(
            session.scalars(
                select(BranchMessage)
                .where(BranchMessage.role == "assistant")
                .order_by(BranchMessage.sequence)
            )
        )
        assert [item.content for item in assistants] == ["第一句"]
        pending = session.scalar(
            select(ConversationPendingBubble).where(ConversationPendingBubble.status == "pending")
        )
        plan = session.scalar(select(ConversationExpressionPlan))
        assert pending is not None and pending.send_not_before is not None
        assert plan is not None and plan.status == "expressing"
        assert session.scalar(select(BranchMemoryEpisode)) is None

        delivered = deliver_due_pending_bubbles(
            session,
            branch_id="branch-1",
            now=pending.send_not_before + timedelta(milliseconds=1),
        )

        assert delivered == 1
        assistants = list(
            session.scalars(
                select(BranchMessage)
                .where(BranchMessage.role == "assistant")
                .order_by(BranchMessage.sequence)
            )
        )
        assert [item.content for item in assistants] == ["第一句", "第二句"]
        assert len({item.turn_id for item in assistants}) == 1
        assert session.get(ConversationExpressionPlan, plan.id).status == "completed"
        episode = session.scalar(select(BranchMemoryEpisode))
        assert episode is not None
        assert [item["content"] for item in episode.assistant_bubbles] == [
            "第一句",
            "第二句",
        ]
        expression_events = list(
            session.scalars(
                select(PerceptionEvent)
                .where(PerceptionEvent.event_type == "agent_expression")
                .order_by(PerceptionEvent.occurred_at, PerceptionEvent.id)
            )
        )
        assert [item.evidence["content"] for item in expression_events] == [
            "第一句",
            "第二句",
        ]
        assert [item.evidence["branch_message_id"] for item in expression_events] == [
            item.id for item in assistants
        ]


def test_new_user_message_cancels_not_yet_sent_assistant_bubbles(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="shadow",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        generator = MultiBubblePersonaTextGenerator()
        service = BranchService(session, generator)
        service.add_user_message("project-1", "branch-1", "先说")
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _persona_text_packet(content),
        )
        actor.process("project-1", "branch-1", lease_owner="worker-1")
        pending = session.scalar(
            select(ConversationPendingBubble).where(ConversationPendingBubble.status == "pending")
        )
        plan = session.scalar(select(ConversationExpressionPlan))
        assert pending is not None and pending.send_not_before is not None
        assert plan is not None

        service.add_user_message("project-1", "branch-1", "等等，我插一句")
        deliver_due_pending_bubbles(
            session,
            branch_id="branch-1",
            now=pending.send_not_before + timedelta(seconds=10),
        )

        assistants = list(
            session.scalars(
                select(BranchMessage)
                .where(BranchMessage.role == "assistant")
                .order_by(BranchMessage.sequence)
            )
        )
        assert [item.content for item in assistants] == ["第一句"]
        assert session.get(ConversationPendingBubble, pending.id).status == "cancelled"
        assert session.get(ConversationExpressionPlan, plan.id).status == "interrupted"
        episode = session.scalar(select(BranchMemoryEpisode))
        assert episode is not None
        assert [item["content"] for item in episode.assistant_bubbles] == ["第一句"]


def test_active_message_creates_cycle_without_offline_cognition_job(
    database: Database,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        BranchService(session, FusedGenerator(_fused_turn(express=False))).add_user_message(
            "project-1",
            "branch-1",
            "在吗",
        )

        assert session.scalar(select(func.count()).select_from(CognitiveCycle)) == 1
        kinds = set(session.scalars(select(Job.kind)))
        assert BRANCH_CONVERSATION_JOB_KIND in kinds
        assert COGNITIVE_CYCLE_JOB_KIND not in kinds


def test_active_persona_text_message_defers_cognition_until_cluster_is_observed(
    database: Database,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(
            session,
            mode="active",
            training_config={"reply_protocol_version": "persona-text-v1"},
        )
        BranchService(session, PersonaTextGenerator()).add_user_message(
            "project-1",
            "branch-1",
            "在吗",
        )

        kinds = set(session.scalars(select(Job.kind)))
        assert kinds == {BRANCH_CONVERSATION_JOB_KIND}
        assert session.scalar(select(func.count()).select_from(CognitiveCycle)) == 0


def test_active_message_bases_cycle_on_current_mental_state(
    database: Database,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        first = MentalStateVersion(
            project_id="project-1",
            branch_id="branch-1",
            version=1,
            state={"mood": "old"},
            evidence={},
            is_current=False,
            model_version_id="model-1",
            model_protocol_version="subject-cognition-v1",
        )
        session.add(first)
        session.flush()
        current = MentalStateVersion(
            project_id="project-1",
            branch_id="branch-1",
            version=2,
            state={"mood": "current"},
            previous_version_id=first.id,
            evidence={},
            is_current=True,
            model_version_id="model-1",
            model_protocol_version="subject-cognition-v1",
        )
        session.add(current)
        session.commit()

        BranchService(session, FusedGenerator(_fused_turn(express=False))).add_user_message(
            "project-1",
            "branch-1",
            "新消息",
        )

        cycle = session.scalar(select(CognitiveCycle))
        assert cycle is not None
        assert cycle.starting_state_version_id == current.id


def test_shadow_message_keeps_offline_cognition_job(database: Database) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="shadow")
        BranchService(session, FusedGenerator(_fused_turn(express=False))).add_user_message(
            "project-1",
            "branch-1",
            "在吗",
        )

        kinds = set(session.scalars(select(Job.kind)))
        assert {BRANCH_CONVERSATION_JOB_KIND, COGNITIVE_CYCLE_JOB_KIND} <= kinds


def test_active_silence_observes_user_and_persists_private_cognition(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        generator = FusedGenerator(_fused_turn(express=False))
        user = BranchService(session, generator).add_user_message(
            "project-1",
            "branch-1",
            "你想说话吗",
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        session.expire_all()
        cycle = session.scalar(select(CognitiveCycle))
        assert cycle is not None
        assert generator.fused_calls == 1
        assert cycle.status == "succeeded"
        assert cycle.final_decision["express"] is False
        assert session.get(BranchMessage, user.id).observed_at is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(BranchMessage)
                .where(BranchMessage.role == "assistant")
            )
            == 0
        )
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 1
        assert (
            session.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.kind == COGNITIVE_EXTRACTION_JOB_KIND)
            )
            == 1
        )


def test_active_expression_persists_public_bubble_and_private_note(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        generator = FusedGenerator(_fused_turn(express=True))
        BranchService(session, generator).add_user_message(
            "project-1",
            "branch-1",
            "你在听吗",
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(select(BranchMessage).where(BranchMessage.role == "assistant"))
        cycle = session.scalar(select(CognitiveCycle))
        assert assistant is not None
        assert assistant.content == "我在听"
        assert cycle is not None
        assert cycle.final_decision["express"] is True
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 1


def test_active_fused_grounding_failure_retries_only_public_expression(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        generator = GroundingRetryFusedGenerator()
        BranchService(session, generator).add_user_message("project-1", "branch-1", "怎么不回电话")
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        actor.process("project-1", "branch-1", lease_owner="worker-1")

        assistant = session.scalar(select(BranchMessage).where(BranchMessage.role == "assistant"))
        cycle = session.scalar(select(CognitiveCycle))
        episode = session.scalar(select(BranchMemoryEpisode))
        assert assistant is not None
        assert assistant.content == "干嘛"
        assert assistant.generation_metadata["grounding_retry_count"] == 1
        assert "当前状态" in assistant.generation_metadata["grounding_retry_reason"]
        assert "我现在正在开会" not in str(assistant.generation_metadata)
        assert cycle is not None and cycle.status == "succeeded"
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 1
        assert episode is not None
        assert [item["content"] for item in episode.assistant_bubbles] == ["干嘛"]


def test_active_expression_rejects_sticker_outside_current_allowlist(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        turn = _fused_turn(express=True)
        invalid_turn = FusedAgentTurn(
            cognition=turn.cognition,
            expression_decision=turn.expression_decision,
            reply=GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(
                        content=None,
                        delay_ms=0,
                        type="sticker",
                        asset_id="越界表情",
                    ),
                ),
                raw_output="<sticker>越界表情</sticker>",
            ),
        )
        generator = FusedGenerator(invalid_turn)
        BranchService(session, generator).add_user_message(
            "project-1",
            "branch-1",
            "发个表情",
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        with pytest.raises(ReplyStructureError):
            actor.process("project-1", "branch-1", lease_owner="worker-1")
        session.rollback()

        assert (
            session.scalar(
                select(func.count())
                .select_from(BranchMessage)
                .where(BranchMessage.role == "assistant")
            )
            == 0
        )
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 0


def test_active_extraction_failure_does_not_rollback_sent_reply(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.agent.extraction import create_cognitive_extraction_handler
    from moonlightbox.jobs.service import JobService

    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        generator = FusedGenerator(_fused_turn(express=True))
        BranchService(session, generator).add_user_message(
            "project-1",
            "branch-1",
            "先回复再演化",
        )
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )
        actor.process("project-1", "branch-1", lease_owner="worker-1")
        extraction_job = session.scalar(
            select(Job).where(Job.kind == COGNITIVE_EXTRACTION_JOB_KIND)
        )
        assert extraction_job is not None

        class FailingExtractor:
            def extract(self, _note: PrivateCognitionNote) -> object:
                raise RuntimeError("后台状态演化失败")

        with pytest.raises(JobHandlerError) as captured:
            create_cognitive_extraction_handler(FailingExtractor())(
                JobService(session),
                extraction_job,
            )

        assistant = session.scalar(select(BranchMessage).where(BranchMessage.role == "assistant"))
        assert captured.value.code == "subject_cognition_extraction_failed"
        assert assistant is not None
        assert assistant.content == "我在听"


def test_new_user_message_invalidates_fused_result_before_commit(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(database.engine) as session:
        _seed_branch(session, mode="active")
        service = BranchService(session, FusedGenerator(_fused_turn(express=True)))
        first = service.add_user_message("project-1", "branch-1", "第一条")

        class ConcurrentGenerator(FusedGenerator):
            def generate_fused(self, *args: object, **kwargs: object) -> FusedAgentTurn:
                with Session(database.engine) as concurrent:
                    BranchService(concurrent, self).add_user_message(
                        "project-1",
                        "branch-1",
                        "生成期间的新消息",
                    )
                return super().generate_fused(*args, **kwargs)

        generator = ConcurrentGenerator(_fused_turn(express=True))
        actor = BranchConversationActor(session, generator)
        monkeypatch.setattr(
            actor,
            "_context_packet",
            lambda _branch, content, **_kwargs: _packet(content),
        )

        with pytest.raises(JobHandlerError, match="用户仍在连续输入"):
            actor.process("project-1", "branch-1", lease_owner="worker-1")

        session.expire_all()
        first_cycle = session.scalar(
            select(CognitiveCycle)
            .join(
                BranchMessage,
                CognitiveCycle.evidence["branch_message_id"].as_string() == BranchMessage.id,
            )
            .where(BranchMessage.id == first.id)
        )
        assert first_cycle is not None
        assert first_cycle.status == "invalidated"
        assert session.scalar(select(func.count()).select_from(PrivateCognitionNote)) == 0
        assert (
            session.scalar(
                select(func.count())
                .select_from(BranchMessage)
                .where(BranchMessage.role == "assistant")
            )
            == 0
        )
