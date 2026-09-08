import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchReflectionRun,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.continuity_types import (
    MemoryProposal,
    MemoryReviewResult,
)
from moonlightbox.branches.models import Branch
from moonlightbox.branches.state_engine import BranchStateEngine
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService

BRANCH_REFLECTION_JOB_KIND = "branch_reflection"
_RETRY_DELAYS = (timedelta(minutes=1), timedelta(minutes=5), timedelta(minutes=30))
_RETRYABLE_FAILURES = {
    "memory_local_proposal_failed",
    "memory_cloud_review_failed",
    "reflection_local_proposal_failed",
    "reflection_cloud_review_failed",
}


class MemoryProposer(Protocol):
    def propose(
        self,
        *,
        episode: BranchMemoryEpisode,
        identity_kernel: dict[str, object],
        current_state: dict[str, object],
    ) -> MemoryProposal: ...


class MemoryReviewer(Protocol):
    def review(
        self,
        episode: BranchMemoryEpisode,
        proposal: MemoryProposal,
        *,
        identity_kernel: dict[str, object] | None = None,
        current_state: dict[str, object] | None = None,
        competing_beliefs: tuple[str, ...] = (),
    ) -> MemoryReviewResult: ...


def create_continual_memory_handler(
    proposer: MemoryProposer,
    reviewer: MemoryReviewer,
    continuity_repository: BranchContinuityRepository | None = None,
) -> JobHandler:
    def handle(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_missing", "记忆任务缺少 Worker 租约")
        episode = service.session.get(
            BranchMemoryEpisode,
            str(job.payload["episode_id"]),
        )
        branch = service.session.get(Branch, str(job.payload["branch_id"]))
        if (
            episode is None
            or branch is None
            or episode.branch_id != branch.id
            or branch.project_id != job.payload.get("project_id")
        ):
            raise JobHandlerError("memory_episode_missing", "分支记忆 episode 不存在")
        current = service.session.scalar(
            select(BranchStateVersion)
            .where(
                BranchStateVersion.branch_id == branch.id,
                BranchStateVersion.is_current.is_(True),
            )
            .order_by(BranchStateVersion.version.desc())
        )
        if episode.processing_status == "processed" and current is not None:
            create_reflection_if_due(
                service.session,
                branch_id=branch.id,
                trigger_episode_id=episode.id,
            )
            service.session.commit()
            if continuity_repository is not None:
                continuity_repository.rebuild_branch(service.session, branch.id)
            return
        kernel = service.session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == branch.model_version_id)
        )
        if kernel is None:
            raise JobHandlerError("identity_kernel_missing", "当前模型缺少锁定人格内核")
        service.checkpoint(
            job.id,
            {"stage": "local_proposal", "progress": 0.2},
            token=token,
        )
        try:
            proposal = proposer.propose(
                episode=episode,
                identity_kernel=kernel.content,
                current_state=branch.state_snapshot,
            )
        except Exception as error:
            raise JobHandlerError("memory_local_proposal_failed", "本地记忆提取失败") from error
        service.checkpoint(
            job.id,
            {"stage": "cloud_review", "progress": 0.5},
            token=token,
        )
        if not proposal.candidates:
            review = MemoryReviewResult(verdict="approve")
        else:
            try:
                competing_beliefs = tuple(
                    service.session.scalars(
                        select(BranchMemoryItem.content).where(
                            BranchMemoryItem.branch_id == branch.id,
                            BranchMemoryItem.kind == "belief",
                            BranchMemoryItem.review_status == "approved",
                            BranchMemoryItem.valid_to.is_(None),
                        )
                    )
                )
                review = reviewer.review(
                    episode,
                    proposal,
                    identity_kernel=kernel.content,
                    current_state=branch.state_snapshot,
                    competing_beliefs=competing_beliefs,
                )
            except Exception as error:
                raise JobHandlerError(
                    "memory_cloud_review_failed",
                    "云端记忆复核失败",
                ) from error
        service.checkpoint(
            job.id,
            {"stage": "applying_state", "progress": 0.75},
            token=token,
        )
        BranchStateEngine(service.session).apply(
            branch=branch,
            episode=episode,
            proposal=proposal,
            review=review,
            identity_kernel=kernel,
        )
        create_reflection_if_due(
            service.session,
            branch_id=branch.id,
            trigger_episode_id=episode.id,
        )
        service.session.commit()
        if continuity_repository is not None:
            service.checkpoint(
                job.id,
                {"stage": "indexing", "progress": 0.9},
                token=token,
            )
            continuity_repository.rebuild_branch(service.session, branch.id)

    return handle


def create_reflection_if_due(
    session: Session,
    *,
    branch_id: str,
    trigger_episode_id: str,
    threshold: float = 5,
) -> BranchReflectionRun | None:
    existing = session.scalar(
        select(BranchReflectionRun).where(
            BranchReflectionRun.branch_id == branch_id,
            BranchReflectionRun.trigger_episode_id == trigger_episode_id,
        )
    )
    if existing is not None:
        return existing
    consumed_item_ids = {
        item_id
        for run in session.scalars(
            select(BranchReflectionRun).where(
                BranchReflectionRun.branch_id == branch_id,
                BranchReflectionRun.status.in_(("pending", "completed")),
            )
        )
        for item_id in run.input_item_ids
    }
    items = [
        item
        for item in session.scalars(
            select(BranchMemoryItem)
            .where(
                BranchMemoryItem.branch_id == branch_id,
                BranchMemoryItem.review_status == "approved",
                BranchMemoryItem.valid_to.is_(None),
                BranchMemoryItem.kind != "reflection",
            )
            .order_by(BranchMemoryItem.created_at.asc())
        )
        if item.id not in consumed_item_ids
    ]
    episode_ids = list(
        dict.fromkeys(episode_id for item in items for episode_id in item.source_episode_ids)
    )
    episodes = list(
        session.scalars(
            select(BranchMemoryEpisode).where(
                BranchMemoryEpisode.branch_id == branch_id,
                BranchMemoryEpisode.id.in_(episode_ids),
            )
        )
    )
    importance_sum = sum(episode.importance for episode in episodes)
    critical_single = len(episodes) == 1 and episodes[0].importance >= 10 if episodes else False
    if not items or (not critical_single and (importance_sum < threshold or len(episodes) < 2)):
        return None
    run = BranchReflectionRun(
        branch_id=branch_id,
        trigger_episode_id=trigger_episode_id,
        input_item_ids=[item.id for item in items],
        input_importance_sum=importance_sum,
        output_item_ids=[],
        status="pending",
        attempt_count=0,
    )
    session.add(run)
    session.flush()
    input_hash = hashlib.sha256(json.dumps(sorted(run.input_item_ids)).encode("utf-8")).hexdigest()
    session.add(
        Job(
            kind=BRANCH_REFLECTION_JOB_KIND,
            payload={
                "branch_id": branch_id,
                "reflection_run_id": run.id,
            },
            dedupe_key=f"branch-reflection:{branch_id}:{input_hash}",
        )
    )
    session.flush()
    return run


def create_branch_reflection_handler(
    proposer: MemoryProposer,
    reviewer: MemoryReviewer,
    continuity_repository: BranchContinuityRepository | None = None,
) -> JobHandler:
    def handle(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_missing", "反思任务缺少 Worker 租约")
        run = service.session.get(
            BranchReflectionRun,
            str(job.payload["reflection_run_id"]),
        )
        if run is None or run.branch_id != job.payload.get("branch_id"):
            raise JobHandlerError("reflection_run_missing", "反思任务不存在")
        if run.status == "completed":
            return
        branch = service.session.get(Branch, run.branch_id)
        trigger = service.session.get(BranchMemoryEpisode, run.trigger_episode_id)
        if branch is None or trigger is None:
            raise JobHandlerError("reflection_source_missing", "反思来源不存在")
        kernel = service.session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == branch.model_version_id)
        )
        items = list(
            service.session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.id.in_(run.input_item_ids),
                    BranchMemoryItem.branch_id == branch.id,
                )
            )
        )
        if kernel is None or len(items) != len(run.input_item_ids):
            raise JobHandlerError("reflection_source_missing", "反思证据不完整")
        run.attempt_count += 1
        current_state = {
            **branch.state_snapshot,
            "reflection_instruction": (
                "根据以下已复核记忆形成更高层主体反思；候选 kind 必须是 reflection"
            ),
            "reflection_inputs": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "content": item.content,
                    "confidence": item.confidence,
                }
                for item in items
            ],
        }
        try:
            proposal = proposer.propose(
                episode=trigger,
                identity_kernel=kernel.content,
                current_state=current_state,
            )
        except Exception as error:
            raise JobHandlerError(
                "reflection_local_proposal_failed",
                "本地阶段反思提取失败",
            ) from error
        run.local_proposal = proposal.model_dump(mode="json")
        try:
            review = reviewer.review(
                trigger,
                proposal,
                identity_kernel=kernel.content,
                current_state=current_state,
                competing_beliefs=tuple(
                    item.content for item in items if item.kind == "belief"
                ),
            )
        except Exception as error:
            raise JobHandlerError(
                "reflection_cloud_review_failed",
                "阶段反思复核失败",
            ) from error
        run.review_result = review.model_dump(mode="json")
        root_episode_ids = list(
            dict.fromkeys(episode_id for item in items for episode_id in item.source_episode_ids)
        )
        root_episode_hashes = list(
            dict.fromkeys(
                root_hash
                for item in items
                for root_hash in item.root_episode_hashes
            )
        )
        # Older rows may predate root hashes. Resolve those roots once, without
        # treating derived items as new independent evidence.
        if not root_episode_hashes:
            root_episode_hashes = list(
                dict.fromkeys(
                    episode.episode_hash
                    for episode in service.session.scalars(
                        select(BranchMemoryEpisode).where(
                            BranchMemoryEpisode.id.in_(root_episode_ids),
                            BranchMemoryEpisode.branch_id == branch.id,
                        )
                    )
                )
            )
        outputs: list[str] = []
        for index in review.approved_candidate_indexes:
            candidate = proposal.candidates[index]
            if candidate.kind != "reflection":
                continue
            lineage_hash = hashlib.sha256(
                json.dumps(
                    {
                        "branch_id": branch.id,
                        "root_episode_hashes": sorted(root_episode_hashes),
                        "subject": candidate.subject,
                        "predicate": candidate.predicate,
                        "object": candidate.object,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            existing = service.session.scalar(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.lineage_hash == lineage_hash,
                )
            )
            if existing is None:
                existing = BranchMemoryItem(
                    branch_id=branch.id,
                    kind="reflection",
                    content=candidate.content,
                    subject=candidate.subject,
                    predicate=candidate.predicate,
                    object=candidate.object,
                    # Reflection may compress evidence, but cannot manufacture
                    # greater certainty than its strongest source.
                    confidence=min(
                        candidate.confidence,
                        max((item.confidence for item in items), default=0.0),
                    ),
                    importance=candidate.importance,
                    valid_from=datetime.now(UTC),
                    source_episode_ids=root_episode_ids,
                    source_item_ids=run.input_item_ids,
                    lineage_hash=lineage_hash,
                    review_status="approved",
                    verification_status="inferred",
                    root_episode_hashes=root_episode_hashes,
                )
                service.session.add(existing)
                service.session.flush()
            outputs.append(existing.id)
        run.output_item_ids = outputs
        run.status = "completed"
        run.completed_at = datetime.now(UTC)
        service.session.commit()
        if continuity_repository is not None:
            service.checkpoint(
                job.id,
                {"stage": "indexing", "progress": 0.9},
                token=token,
            )
            continuity_repository.rebuild_branch(service.session, branch.id)

    return handle


def resume_retryable_continuity_jobs(
    session: Session,
    *,
    now: datetime | None = None,
) -> int:
    """以有限持久退避恢复瞬时记忆/反思失败，避免人格成长永久冻结。"""

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    resumed = 0
    jobs = list(
        session.scalars(
            select(Job).where(
                Job.kind.in_(("branch_continual_memory", BRANCH_REFLECTION_JOB_KIND)),
                Job.status == "failed",
                Job.error_code.in_(_RETRYABLE_FAILURES),
            )
        )
    )
    for job in jobs:
        attempt = _retry_attempt(job.payload)
        if attempt >= len(_RETRY_DELAYS):
            continue
        updated_at = job.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        if updated_at + _RETRY_DELAYS[attempt] > current:
            continue
        if job.kind == "branch_continual_memory":
            episode_id = job.payload.get("episode_id")
            episode = (
                session.get(BranchMemoryEpisode, episode_id)
                if isinstance(episode_id, str)
                else None
            )
            if episode is None or episode.processing_status != "pending":
                continue
        else:
            run_id = job.payload.get("reflection_run_id")
            run = (
                session.get(BranchReflectionRun, run_id)
                if isinstance(run_id, str)
                else None
            )
            if run is None or run.status != "pending":
                continue
        job.payload = {**job.payload, "automatic_retry_count": attempt + 1}
        session.commit()
        JobService(session).resume(job.id)
        resumed += 1
    return resumed


def _retry_attempt(payload: dict[str, object]) -> int:
    value = payload.get("automatic_retry_count", 0)
    return int(value) if isinstance(value, int | float) and value >= 0 else 0
