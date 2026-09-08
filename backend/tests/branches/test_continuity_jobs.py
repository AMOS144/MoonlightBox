from datetime import UTC, datetime
from pathlib import Path

from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchReflectionRun,
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
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class FakeProposer:
    def propose(
        self,
        *,
        episode: BranchMemoryEpisode,
        identity_kernel: dict[str, object],
        current_state: dict[str, object],
    ) -> MemoryProposal:
        del identity_kernel, current_state
        return MemoryProposal(
            candidates=(
                MemoryCandidate(
                    kind="self_narrative",
                    content="我愿意继续聊",
                    subject="digital_human",
                    predicate="交流意愿",
                    object="愿意",
                    confidence=0.7,
                    importance=episode.importance,
                    evidence_message_ids=(episode.assistant_turn_id,),
                    source_role="digital_human",
                ),
            ),
            state_delta=StateDeltaProposal(
                relationship_delta={"trust": 1},
                supporting_candidate_indexes=(0,),
            ),
        )


class FakeReviewer:
    def review(
        self,
        episode: BranchMemoryEpisode,
        proposal: MemoryProposal,
        **_context: object,
    ) -> MemoryReviewResult:
        del episode
        return MemoryReviewResult(
            verdict="approve",
            approved_candidate_indexes=(0,),
            approved_state_delta=proposal.state_delta,
        )


class ReflectionProposer:
    def propose(
        self,
        *,
        episode: BranchMemoryEpisode,
        identity_kernel: dict[str, object],
        current_state: dict[str, object],
    ) -> MemoryProposal:
        del identity_kernel
        assert current_state["reflection_inputs"]
        return MemoryProposal(
            candidates=(
                MemoryCandidate(
                    kind="reflection",
                    content="我逐渐觉得可靠的互动值得珍惜",
                    subject="digital_human",
                    predicate="关系反思",
                    object="珍惜可靠互动",
                    confidence=0.99,
                    importance=8,
                    evidence_message_ids=(episode.assistant_turn_id,),
                    source_role="digital_human",
                ),
            )
        )


def _setup(session: Session) -> tuple[Branch, BranchMemoryEpisode, Job]:
    now = datetime.now(UTC)
    session.add(Project(id="project-1", name="记忆任务"))
    session.flush()
    session.add(
        EventNode(
            id="event-1",
            project_id="project-1",
            type="relationship",
            title="起点",
            summary="开始",
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
        state_snapshot={},
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
    session.add_all([branch, kernel])
    session.flush()
    episode = BranchMemoryEpisode(
        id="episode-1",
        branch_id="branch-1",
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
        user_content="继续聊吗",
        assistant_bubbles=[{"type": "text", "content": "好"}],
        model_version_id="model-1",
        episode_hash="episode-hash",
        importance=10,
        processing_status="pending",
        started_at=now,
        ended_at=now,
    )
    job = Job(
        id="job-1",
        kind="branch_continual_memory",
        payload={
            "project_id": "project-1",
            "branch_id": "branch-1",
            "episode_id": "episode-1",
        },
        status="queued",
        dedupe_key="branch-memory:episode-1",
    )
    session.add_all([episode, job])
    session.commit()
    running = JobService(session).start(job.id, worker_token="worker-1")
    return branch, episode, running


def test_continual_memory_handler_is_idempotent(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_jobs import (
        create_continual_memory_handler,
    )

    database = Database(f"sqlite:///{tmp_path / 'handler.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _branch, episode, job = _setup(session)
        handler = create_continual_memory_handler(FakeProposer(), FakeReviewer())

        handler(JobService(session), job)
        handler(JobService(session), job)

        assert episode.processing_status == "processed"
        assert session.query(BranchMemoryItem).count() == 1
        assert session.query(Job).filter(Job.kind == "branch_reflection").count() == 1
    database.close()


def test_reflection_trigger_counts_each_episode_once(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_jobs import create_reflection_if_due

    database = Database(f"sqlite:///{tmp_path / 'reflection.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, episode, _job = _setup(session)
        episode.importance = 9
        session.add_all(
            [
                BranchMemoryItem(
                    branch_id=branch.id,
                    kind="experience",
                    content="第一条",
                    subject="interaction",
                    predicate="交流",
                    object="继续",
                    confidence=0.8,
                    importance=10,
                    valid_from=episode.ended_at,
                    source_episode_ids=[episode.id],
                    source_item_ids=[],
                    lineage_hash="lineage-1",
                    review_status="approved",
                ),
                BranchMemoryItem(
                    branch_id=branch.id,
                    kind="self_narrative",
                    content="第二条",
                    subject="digital_human",
                    predicate="感受",
                    object="愿意",
                    confidence=0.8,
                    importance=10,
                    valid_from=episode.ended_at,
                    source_episode_ids=[episode.id],
                    source_item_ids=[],
                    lineage_hash="lineage-2",
                    review_status="approved",
                ),
            ]
        )
        session.commit()

        run = create_reflection_if_due(
            session,
            branch_id=branch.id,
            trigger_episode_id=episode.id,
            threshold=20,
        )

        assert run is None
    database.close()


def test_default_reflection_threshold_matches_normal_conversation_importance(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_jobs import create_reflection_if_due

    database = Database(f"sqlite:///{tmp_path / 'normal-reflection.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, first, _job = _setup(session)
        first.importance = 2.6
        second = BranchMemoryEpisode(
            id="episode-2",
            branch_id=branch.id,
            user_turn_id="user-turn-2",
            assistant_turn_id="assistant-turn-2",
            user_content="又见面了",
            assistant_bubbles=[{"type": "text", "content": "嗯"}],
            model_version_id="model-1",
            episode_hash="episode-hash-2",
            importance=2.5,
            processing_status="processed",
            started_at=first.started_at,
            ended_at=first.ended_at,
        )
        session.add(second)
        session.add_all(
            [
                BranchMemoryItem(
                    branch_id=branch.id,
                    kind="experience",
                    content="第一次继续交流",
                    subject="interaction",
                    predicate="交流",
                    object="继续",
                    confidence=0.8,
                    importance=2.6,
                    valid_from=first.ended_at,
                    source_episode_ids=[first.id],
                    source_item_ids=[],
                    lineage_hash="normal-lineage-1",
                    review_status="approved",
                ),
                BranchMemoryItem(
                    branch_id=branch.id,
                    kind="experience",
                    content="再次回来交流",
                    subject="interaction",
                    predicate="重逢",
                    object="再次交流",
                    confidence=0.8,
                    importance=2.5,
                    valid_from=second.ended_at,
                    source_episode_ids=[second.id],
                    source_item_ids=[],
                    lineage_hash="normal-lineage-2",
                    review_status="approved",
                ),
            ]
        )
        session.commit()

        run = create_reflection_if_due(
            session,
            branch_id=branch.id,
            trigger_episode_id=second.id,
        )

        assert run is not None
        assert run.input_importance_sum == 5.1
    database.close()


def test_transient_memory_failure_is_retried_with_bounded_backoff(
    tmp_path: Path,
) -> None:
    from datetime import timedelta

    from moonlightbox.branches.continuity_jobs import (
        resume_retryable_continuity_jobs,
    )

    database = Database(f"sqlite:///{tmp_path / 'memory-retry.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _branch, _episode, running = _setup(session)
        token = running.worker_token
        assert token is not None
        failed = JobService(session).fail(
            running.id,
            "memory_cloud_review_failed",
            "云端记忆复核失败",
            token=token,
        )
        assert failed is not None
        retry_at = failed.updated_at + timedelta(minutes=1, seconds=1)

        assert resume_retryable_continuity_jobs(session, now=retry_at) == 1

        resumed = session.get(Job, running.id)
        assert resumed is not None
        assert resumed.status == "queued"
        assert resumed.payload["automatic_retry_count"] == 1
        resumed.status = "failed"
        resumed.error_code = "memory_cloud_review_failed"
        resumed.payload = {**resumed.payload, "automatic_retry_count": 3}
        session.commit()
        assert (
            resume_retryable_continuity_jobs(
                session,
                now=retry_at + timedelta(days=1),
            )
            == 0
        )
    database.close()


def test_reflection_preserves_root_provenance_and_cannot_gain_authority(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_jobs import create_branch_reflection_handler

    database = Database(f"sqlite:///{tmp_path / 'reflection-lineage.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch, episode, _memory_job = _setup(session)
        source = BranchMemoryItem(
            id="source-item",
            branch_id=branch.id,
            kind="experience",
            content="用户按约定回来继续交流",
            subject="interaction",
            predicate="守约",
            object="继续交流",
            confidence=0.72,
            importance=8,
            valid_from=episode.ended_at,
            source_episode_ids=[episode.id],
            source_item_ids=[],
            lineage_hash="source-lineage",
            review_status="approved",
            verification_status="verified_interaction",
            root_episode_hashes=[episode.episode_hash],
        )
        run = BranchReflectionRun(
            id="reflection-run",
            branch_id=branch.id,
            trigger_episode_id=episode.id,
            input_item_ids=[source.id],
            input_importance_sum=10,
            output_item_ids=[],
            status="pending",
            attempt_count=0,
        )
        job = Job(
            id="reflection-job",
            kind="branch_reflection",
            payload={"branch_id": branch.id, "reflection_run_id": run.id},
            status="queued",
            dedupe_key="reflection-lineage-test",
        )
        session.add_all([source, run, job])
        session.commit()
        running = JobService(session).start(job.id, worker_token="reflection-worker")

        create_branch_reflection_handler(ReflectionProposer(), FakeReviewer())(
            JobService(session), running
        )

        reflection = session.query(BranchMemoryItem).filter_by(kind="reflection").one()
        assert reflection.source_item_ids == [source.id]
        assert reflection.root_episode_hashes == [episode.episode_hash]
        assert reflection.verification_status == "inferred"
        assert reflection.confidence == source.confidence
        assert reflection.state_version_id is None
        assert branch.state_snapshot == {}
    database.close()
