from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.continuity_types import (
    MemoryCandidate,
    MemoryProposal,
    MemoryReviewResult,
    StateDeltaProposal,
)
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import select
from sqlalchemy.orm import Session


def _setup(session: Session) -> tuple[Branch, IdentityKernel, BranchMemoryEpisode]:
    now = datetime.now(UTC)
    session.add(Project(id="project-1", name="状态引擎"))
    session.flush()
    session.add(
        EventNode(
            id="event-1",
            project_id="project-1",
            type="relationship",
            title="起点",
            summary="重新交流",
            start_message_id="m1",
            end_message_id="m2",
            emotion_labels=[],
            topic="关系",
            conflict_level=0,
            importance=0.8,
            reason="起点",
            evidence_ids=["m1"],
        )
    )
    session.add(
        ModelVersion(
            id="model-1",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/tmp/adapter",
            dataset_hash="hash",
            metrics={},
        )
    )
    session.flush()
    branch = Branch(
        id="branch-1",
        project_id="project-1",
        origin_event_id="event-1",
        model_version_id="model-1",
        title="分支",
        origin_time=now,
        state_snapshot={
            "relationship_state": {"trust": 50.0},
            "emotional_tendency": {"sadness": 0.2},
        },
    )
    kernel = IdentityKernel(
        id="kernel-1",
        project_id="project-1",
        model_version_id="model-1",
        schema_version="v1",
        content={"persona": "她"},
        evidence_message_ids=["m1"],
        content_hash="kernel-hash",
        locked_at=now,
    )
    episode = BranchMemoryEpisode(
        id="episode-1",
        branch_id="branch-1",
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
        user_content="你是不是不在乎我了",
        assistant_bubbles=[{"type": "text", "content": "我只是需要一点空间"}],
        model_version_id="model-1",
        episode_hash="episode-hash",
        importance=8,
        processing_status="pending",
        started_at=now,
        ended_at=now,
    )
    session.add_all([branch, kernel])
    session.flush()
    session.add(episode)
    session.commit()
    return branch, kernel, episode


def _review() -> MemoryReviewResult:
    return MemoryReviewResult(
        verdict="approve",
        approved_candidate_indexes=(0,),
        approved_state_delta=StateDeltaProposal(
            relationship_delta={"trust": -20},
            emotional_delta={"sadness": 0.5},
            supporting_candidate_indexes=(0,),
        ),
    )


def _next_episode(
    branch_id: str,
    index: int,
    *,
    ended_at: datetime | None = None,
) -> BranchMemoryEpisode:
    now = ended_at or datetime.now(UTC)
    return BranchMemoryEpisode(
        id=f"episode-{index}",
        branch_id=branch_id,
        user_turn_id=f"user-turn-{index}",
        assistant_turn_id=f"assistant-turn-{index}",
        user_content="继续聊聊这段关系",
        assistant_bubbles=[{"type": "text", "content": "好"}],
        model_version_id="model-1",
        episode_hash=f"episode-hash-{index}",
        importance=8,
        processing_status="pending",
        started_at=now,
        ended_at=now,
    )


def _proposal(
    *,
    source_role: str = "digital_human",
    object_value: str = "需要空间",
    kind: str = "belief",
    confidence: float = 0.7,
    stance: str = "support",
    user_turn_id: str = "user-turn",
    assistant_turn_id: str = "assistant-turn",
) -> MemoryProposal:
    return MemoryProposal(
        candidates=(
            MemoryCandidate(
                kind=kind,  # type: ignore[arg-type]
                content="我认为自己现在需要一点空间",
                subject="digital_human",
                predicate="关系需要",
                object=object_value,
                confidence=confidence,
                importance=8,
                evidence_message_ids=(
                    (user_turn_id, assistant_turn_id)
                    if source_role == "interaction"
                    else (assistant_turn_id,)
                ),
                source_role=source_role,  # type: ignore[arg-type]
                stance=stance,  # type: ignore[arg-type]
            ),
        ),
        state_delta=StateDeltaProposal(
            relationship_delta={"trust": -20},
            emotional_delta={"sadness": 0.5},
            supporting_candidate_indexes=(0,),
        ),
    )


def test_state_change_is_bounded_and_idempotent(tmp_path: Path) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'state.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        engine = BranchStateEngine(session)

        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction"),
            review=_review(),
            identity_kernel=kernel,
        )
        second = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction"),
            review=_review(),
            identity_kernel=kernel,
        )

        assert first.id == second.id
        assert first.relationship_state["trust"] == 45.0
        assert first.emotional_tendency["sadness"] == 0.3
        assert first.version == 1
        assert session.query(BranchMemoryItem).count() == 1
        assert session.query(BranchStateVersion).count() == 1
        assert branch.state_snapshot["state_version_id"] == first.id
        assert first.field_evidence == {
            "relationship_state.trust": [episode.id],
            "emotional_tendency.sadness": [episode.id],
        }
        assert first.field_confidence == {
            "relationship_state.trust": 0.7,
            "emotional_tendency.sadness": 0.7,
        }
        assert branch.state_snapshot["state_evidence"] == first.field_evidence
    database.close()


def test_free_form_state_keys_and_user_model_overwrite_are_discarded() -> None:
    delta = StateDeltaProposal(
        relationship_delta={"trust": 1, "soulmate_certainty": 100},
        emotional_delta={"sadness": 0.1, "destiny": 1},
        user_model_updates={"identity": "用户其实是另一个人"},
        supporting_candidate_indexes=(0,),
    )

    assert delta.relationship_delta == {"trust": 1.0}
    assert delta.emotional_delta == {"sadness": 0.1}
    assert delta.user_model_updates == {}


def test_user_definition_cannot_create_subject_belief(tmp_path: Path) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'user-definition.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)

        state = BranchStateEngine(session).apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="user", object_value="不爱用户"),
            review=_review(),
            identity_kernel=kernel,
        )

        assert state.active_belief_ids == []
        assert session.query(BranchMemoryItem).count() == 0
        assert state.relationship_state["trust"] == 50.0
    database.close()


def test_generated_belief_is_remembered_but_cannot_certify_durable_state(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'self-certification.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)

        state = BranchStateEngine(session).apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="digital_human", confidence=0.99),
            review=_review(),
            identity_kernel=kernel,
        )
        item = session.query(BranchMemoryItem).one()

        assert item.verification_status == "inferred"
        assert state.active_belief_ids == []
        assert state.relationship_state["trust"] == 50.0
        assert state.emotional_tendency["sadness"] == 0.2
    database.close()


def test_user_definition_cannot_create_subject_fact(tmp_path: Path) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'user-fact.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)

        state = BranchStateEngine(session).apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(
                source_role="user",
                object_value="不爱用户",
                kind="fact",
            ),
            review=_review(),
            identity_kernel=kernel,
        )

        assert state.active_belief_ids == []
        assert session.query(BranchMemoryItem).count() == 0
        assert state.relationship_state["trust"] == 50.0
    database.close()


def test_stronger_competing_belief_wins_without_erasing_prior_evidence(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'belief-competition.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        engine = BranchStateEngine(session)
        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(
                source_role="interaction",
                object_value="需要空间",
                confidence=0.7,
            ),
            review=_review(),
            identity_kernel=kernel,
        )
        first_belief = session.get(BranchMemoryItem, first.active_belief_ids[0])
        assert first_belief is not None

        now = datetime.now(UTC)
        episode_2 = BranchMemoryEpisode(
            id="episode-2",
            branch_id=branch.id,
            user_turn_id="user-turn-2",
            assistant_turn_id="assistant-turn-2",
            user_content="你想清楚了吗",
            assistant_bubbles=[{"type": "text", "content": "我还是想继续好好相处"}],
            model_version_id="model-1",
            episode_hash="episode-hash-2",
            importance=8,
            processing_status="pending",
            started_at=now,
            ended_at=now,
        )
        session.add(episode_2)
        session.flush()
        second = engine.apply(
            branch=branch,
            episode=episode_2,
            proposal=_proposal(
                source_role="interaction",
                object_value="继续关系",
                confidence=0.95,
                user_turn_id="user-turn-2",
                assistant_turn_id="assistant-turn-2",
            ),
            review=_review(),
            identity_kernel=kernel,
        )
        second_belief = session.get(BranchMemoryItem, second.active_belief_ids[0])

        assert len(second.active_belief_ids) == 1
        assert second_belief is not None
        assert second_belief.object == "继续关系"
        assert first_belief.valid_to is None
        assert first_belief.id not in second.active_belief_ids
    database.close()


def test_observed_interaction_fact_is_marked_verified(tmp_path: Path) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'verified-fact.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)

        state = BranchStateEngine(session).apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(
                source_role="interaction",
                object_value="完成了一次对话",
                kind="fact",
            ),
            review=_review(),
            identity_kernel=kernel,
        )
        item = session.query(BranchMemoryItem).one()

        assert item.verification_status == "verified_interaction"
        assert item.state_version_id == state.id
        assert item.root_episode_hashes == [episode.episode_hash]
    database.close()


def test_independent_interactions_aggregate_before_belief_activation(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'belief-aggregation.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        engine = BranchStateEngine(session)
        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction", confidence=0.5),
            review=_review(),
            identity_kernel=kernel,
        )
        assert first.active_belief_ids == []

        episode_2 = _next_episode(branch.id, 2)
        session.add(episode_2)
        session.flush()
        second = engine.apply(
            branch=branch,
            episode=episode_2,
            proposal=_proposal(
                source_role="interaction",
                confidence=0.5,
                user_turn_id=episode_2.user_turn_id,
                assistant_turn_id=episode_2.assistant_turn_id,
            ),
            review=_review(),
            identity_kernel=kernel,
        )

        assert len(second.active_belief_ids) == 1
        assert len(session.query(BranchMemoryItem).filter_by(kind="belief").all()) == 2
    database.close()


def test_observed_counterevidence_deactivates_belief_without_erasing_history(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'belief-counterevidence.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        engine = BranchStateEngine(session)
        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction", confidence=0.9),
            review=_review(),
            identity_kernel=kernel,
        )
        assert len(first.active_belief_ids) == 1

        episode_2 = _next_episode(branch.id, 2)
        session.add(episode_2)
        session.flush()
        second = engine.apply(
            branch=branch,
            episode=episode_2,
            proposal=_proposal(
                source_role="interaction",
                confidence=0.9,
                stance="oppose",
                user_turn_id=episode_2.user_turn_id,
                assistant_turn_id=episode_2.assistant_turn_id,
            ),
            review=_review(),
            identity_kernel=kernel,
        )

        assert second.active_belief_ids == []
        assert session.query(BranchMemoryItem).filter_by(kind="belief").count() == 2
        assert all(item.valid_to is None for item in session.query(BranchMemoryItem).all())
    database.close()


def test_interaction_role_without_user_evidence_cannot_activate_belief(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'forged-interaction.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        proposal = _proposal(source_role="interaction", confidence=0.99).model_copy(
            update={
                "candidates": (
                    _proposal(source_role="interaction", confidence=0.99)
                    .candidates[0]
                    .model_copy(update={"evidence_message_ids": ("assistant-turn",)}),
                )
            }
        )

        state = BranchStateEngine(session).apply(
            branch=branch,
            episode=episode,
            proposal=proposal,
            review=_review(),
            identity_kernel=kernel,
        )

        assert state.active_belief_ids == []
        assert state.relationship_state["trust"] == 50.0
    database.close()


def test_stale_belief_decays_when_a_later_state_version_is_built(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine, StateEngineConfig

    database = Database(f"sqlite:///{tmp_path / 'belief-decay.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        episode.started_at -= timedelta(days=3)
        episode.ended_at -= timedelta(days=3)
        engine = BranchStateEngine(
            session,
            StateEngineConfig(
                belief_decay_grace_days=0,
                belief_half_life_days=1,
            ),
        )
        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction", confidence=0.9),
            review=_review(),
            identity_kernel=kernel,
        )
        assert len(first.active_belief_ids) == 1

        episode_2 = _next_episode(branch.id, 2)
        session.add(episode_2)
        session.flush()
        second = engine.apply(
            branch=branch,
            episode=episode_2,
            proposal=MemoryProposal(),
            review=MemoryReviewResult(verdict="approve"),
            identity_kernel=kernel,
        )

        assert second.active_belief_ids == []
        assert second.contested_belief_ids == []
    database.close()


def test_rollback_restores_prior_belief_resolution_and_invalidates_later_evidence(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'belief-rollback.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        engine = BranchStateEngine(session)
        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction", confidence=0.9),
            review=_review(),
            identity_kernel=kernel,
        )
        first_belief_id = first.active_belief_ids[0]

        episode_2 = _next_episode(branch.id, 2)
        session.add(episode_2)
        session.flush()
        second = engine.apply(
            branch=branch,
            episode=episode_2,
            proposal=_proposal(
                source_role="interaction",
                confidence=0.9,
                stance="oppose",
                user_turn_id=episode_2.user_turn_id,
                assistant_turn_id=episode_2.assistant_turn_id,
            ),
            review=_review(),
            identity_kernel=kernel,
        )
        later_item = session.scalar(
            select(BranchMemoryItem).where(BranchMemoryItem.state_version_id == second.id)
        )
        assert second.active_belief_ids == []
        assert later_item is not None

        restored = engine.rollback(branch, first.id)

        assert restored.active_belief_ids == [first_belief_id]
        assert restored.field_evidence == first.field_evidence
        assert later_item.valid_to is not None
        assert later_item.invalidated_at is not None
    database.close()


def test_generated_experience_is_stored_as_self_claim_not_world_fact(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'self-claimed-experience.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)

        BranchStateEngine(session).apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(
                source_role="digital_human",
                object_value="今天去了医院",
                kind="experience",
            ),
            review=_review(),
            identity_kernel=kernel,
        )
        item = session.query(BranchMemoryItem).one()

        assert item.verification_status == "self_claimed"
    database.close()


def test_rollback_creates_new_current_version(tmp_path: Path) -> None:
    from moonlightbox.branches.state_engine import BranchStateEngine

    database = Database(f"sqlite:///{tmp_path / 'rollback.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, kernel, episode = _setup(session)
        engine = BranchStateEngine(session)
        first = engine.apply(
            branch=branch,
            episode=episode,
            proposal=_proposal(source_role="interaction"),
            review=_review(),
            identity_kernel=kernel,
        )
        episode_2 = BranchMemoryEpisode(
            id="episode-2",
            branch_id=branch.id,
            user_turn_id="user-turn-2",
            assistant_turn_id="assistant-turn-2",
            user_content="好",
            assistant_bubbles=[{"type": "text", "content": "谢谢"}],
            model_version_id="model-1",
            episode_hash="episode-hash-2",
            importance=3,
            processing_status="pending",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )
        session.add(episode_2)
        session.commit()
        second = engine.apply(
            branch=branch,
            episode=episode_2,
            proposal=_proposal(
                source_role="interaction",
                object_value="继续关系",
                confidence=0.95,
            ),
            review=_review(),
            identity_kernel=kernel,
        )
        second_item = session.scalar(
            session.query(BranchMemoryItem)
            .filter(BranchMemoryItem.state_version_id == second.id)
            .statement
        )
        assert second_item is not None

        rolled_back = engine.rollback(branch, first.id)

        assert second.is_current is False
        assert rolled_back.version == 3
        assert rolled_back.relationship_state == first.relationship_state
        assert rolled_back.is_current is True
        assert branch.state_snapshot["state_version_id"] == rolled_back.id
        assert rolled_back.rollback_of_version_id == first.id
        assert second_item.valid_to is not None
        assert second_item.invalidated_at is not None
    database.close()
