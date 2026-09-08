from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.agent.models import CognitiveCycle
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_migration import (
    ContinuityMigrationService,
    ProjectMigrationReport,
)
from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchReflectionRun,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.continuity_schemas import (
    BranchMemoryOverviewRead,
    BranchStateVersionRead,
    IdentityKernelRead,
    LongitudinalGrowthHealthRead,
    MemoryItemRead,
)
from moonlightbox.branches.episodes import CONTINUAL_MEMORY_JOB_KIND
from moonlightbox.branches.models import Branch
from moonlightbox.branches.state_engine import BranchStateEngine
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService


class ContinuityNotReadyError(RuntimeError):
    pass


class ContinuityService:
    def __init__(
        self,
        session: Session,
        continuity_repository: BranchContinuityRepository | None = None,
    ) -> None:
        self._session = session
        self._repository = continuity_repository

    def overview(
        self,
        project_id: str,
        branch_id: str,
    ) -> BranchMemoryOverviewRead:
        branch = self._branch(project_id, branch_id)
        kernel = self._session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == branch.model_version_id)
        )
        state = self._current_state(branch.id)
        if kernel is None or state is None:
            raise ContinuityNotReadyError("该分支尚未迁移持续人格记忆")
        active_ids = set(state.active_belief_ids)
        beliefs = list(
            self._session.scalars(
                select(BranchMemoryItem)
                .where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.kind == "belief",
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                )
                .order_by(BranchMemoryItem.confidence.desc())
            )
        )
        reflections = list(
            self._session.scalars(
                select(BranchMemoryItem)
                .where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.kind == "reflection",
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                )
                .order_by(BranchMemoryItem.created_at.desc())
                .limit(10)
            )
        )
        jobs = self._jobs(branch.id)
        pending = sum(job.status in {"queued", "running", "interrupted"} for job in jobs)
        failed = sum(job.status == "failed" for job in jobs)
        return BranchMemoryOverviewRead(
            identity_kernel=IdentityKernelRead.model_validate(kernel),
            current_state=BranchStateVersionRead.model_validate(state),
            active_beliefs=[
                MemoryItemRead.model_validate(item) for item in beliefs if item.id in active_ids
            ],
            competing_beliefs=[
                MemoryItemRead.model_validate(item) for item in beliefs if item.id not in active_ids
            ],
            recent_reflections=[MemoryItemRead.model_validate(item) for item in reflections],
            pending_jobs=pending,
            failed_jobs=failed,
            evolution_frozen=failed > 0,
            growth_health=self._growth_health(branch.id, failed_jobs=failed),
        )

    def _growth_health(
        self,
        branch_id: str,
        *,
        failed_jobs: int,
    ) -> LongitudinalGrowthHealthRead:
        episodes = list(
            self._session.scalars(
                select(BranchMemoryEpisode).where(
                    BranchMemoryEpisode.branch_id == branch_id,
                    BranchMemoryEpisode.processing_status == "processed",
                )
            )
        )
        episode_ids = {episode.id for episode in episodes}
        approved = list(
            self._session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch_id,
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                )
            )
        )
        state_count = len(
            list(
                self._session.scalars(
                    select(BranchStateVersion.id).where(
                        BranchStateVersion.branch_id == branch_id
                    )
                )
            )
        )
        evidence_backed = sum(
            bool(item.source_episode_ids)
            and set(item.source_episode_ids).issubset(episode_ids)
            for item in approved
        )
        evidence_rate = evidence_backed / len(approved) if approved else 1.0
        lineages = [item.lineage_hash for item in approved]
        duplicate_lineages = len(lineages) - len(set(lineages))
        span_days = 0.0
        if len(episodes) >= 2:
            first = min(episode.started_at for episode in episodes)
            last = max(episode.ended_at for episode in episodes)
            span_days = max(0.0, (last - first).total_seconds() / 86400)
        reflection_count = sum(item.kind == "reflection" for item in approved)
        successful_cognitive_cycles = self._session.scalar(
            select(func.count(CognitiveCycle.id)).where(
                CognitiveCycle.branch_id == branch_id,
                CognitiveCycle.status == "succeeded",
            )
        ) or 0
        failed_cognitive_cycles = self._session.scalar(
            select(func.count(CognitiveCycle.id)).where(
                CognitiveCycle.branch_id == branch_id,
                CognitiveCycle.status == "failed",
            )
        ) or 0
        unmet: list[str] = []
        if len(episodes) < 20:
            unmet.append("需要至少 20 个已处理互动 episode")
        if span_days < 14:
            unmet.append("需要覆盖至少 14 天真实互动")
        if state_count < 5:
            unmet.append("需要至少 5 个可追溯状态版本")
        if len(approved) < 5:
            unmet.append("需要至少 5 条已批准长期记忆")
        if reflection_count < 1:
            unmet.append("需要至少 1 条跨 episode 阶段反思")
        if successful_cognitive_cycles < 5:
            unmet.append("需要至少 5 次成功的主体认知周期")
        if failed_cognitive_cycles:
            unmet.append("失败的主体认知周期必须先恢复")
        if evidence_rate < 1:
            unmet.append("所有有效记忆必须完整引用本分支 episode")
        if duplicate_lineages:
            unmet.append("记忆 lineage 不能重复自我强化")
        if failed_jobs:
            unmet.append("失败的演化任务必须先修复")
        status = (
            "frozen"
            if (
                failed_jobs
                or failed_cognitive_cycles
                or evidence_rate < 1
                or duplicate_lineages
            )
            else "validated"
            if not unmet
            else "observing"
            if episodes
            else "insufficient"
        )
        return LongitudinalGrowthHealthRead(
            status=status,
            processed_episode_count=len(episodes),
            observation_span_days=round(span_days, 2),
            state_version_count=state_count,
            approved_memory_count=len(approved),
            approved_reflection_count=reflection_count,
            successful_cognitive_cycle_count=successful_cognitive_cycles,
            failed_cognitive_cycle_count=failed_cognitive_cycles,
            evidence_coverage_rate=round(evidence_rate, 4),
            duplicate_lineage_count=duplicate_lineages,
            unmet_requirements=unmet,
        )

    def episodes(
        self,
        project_id: str,
        branch_id: str,
    ) -> list[BranchMemoryEpisode]:
        branch = self._branch(project_id, branch_id)
        return list(
            self._session.scalars(
                select(BranchMemoryEpisode)
                .where(BranchMemoryEpisode.branch_id == branch.id)
                .order_by(BranchMemoryEpisode.started_at.desc())
            )
        )

    def beliefs(
        self,
        project_id: str,
        branch_id: str,
    ) -> list[BranchMemoryItem]:
        branch = self._branch(project_id, branch_id)
        return list(
            self._session.scalars(
                select(BranchMemoryItem)
                .where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.kind.in_(("belief", "self_narrative", "fact", "experience")),
                )
                .order_by(BranchMemoryItem.created_at.desc())
            )
        )

    def reflections(
        self,
        project_id: str,
        branch_id: str,
    ) -> list[BranchReflectionRun]:
        branch = self._branch(project_id, branch_id)
        return list(
            self._session.scalars(
                select(BranchReflectionRun)
                .where(BranchReflectionRun.branch_id == branch.id)
                .order_by(BranchReflectionRun.created_at.desc())
            )
        )

    def versions(
        self,
        project_id: str,
        branch_id: str,
    ) -> list[BranchStateVersion]:
        branch = self._branch(project_id, branch_id)
        return list(
            self._session.scalars(
                select(BranchStateVersion)
                .where(BranchStateVersion.branch_id == branch.id)
                .order_by(BranchStateVersion.version.desc())
            )
        )

    def rollback(
        self,
        project_id: str,
        branch_id: str,
        version_id: str,
    ) -> BranchStateVersion:
        branch = self._branch(project_id, branch_id)
        version = BranchStateEngine(self._session).rollback(branch, version_id)
        self._session.commit()
        if self._repository is not None:
            self._repository.rebuild_branch(self._session, branch.id)
        return version

    def retry_job(
        self,
        project_id: str,
        branch_id: str,
        job_id: str,
    ) -> Job:
        branch = self._branch(project_id, branch_id)
        job = self._session.get(Job, job_id)
        if (
            job is None
            or job.payload.get("branch_id") != branch.id
            or job.kind not in {CONTINUAL_MEMORY_JOB_KIND, "branch_reflection"}
        ):
            raise LookupError("记忆任务不存在")
        return JobService(self._session).resume(job.id)

    def jobs(
        self,
        project_id: str,
        branch_id: str,
    ) -> list[Job]:
        branch = self._branch(project_id, branch_id)
        return sorted(
            self._jobs(branch.id),
            key=lambda job: job.created_at,
            reverse=True,
        )

    def migrate(
        self,
        project_id: str,
        branch_id: str,
    ) -> ProjectMigrationReport:
        branch = self._branch(project_id, branch_id)
        migration = ContinuityMigrationService(
            self._session,
            self._repository,
        )
        kernel = migration.migrate_active_model(project_id)
        if branch.model_version_id != kernel.model_version_id:
            raise ContinuityNotReadyError("分支未绑定当前活动模型，请先升级分支")
        report = migration.backfill_branch(branch.id)
        self._session.commit()
        return ProjectMigrationReport(
            project_id=project_id,
            kernel_id=kernel.id,
            branch_reports=[report],
        )

    def _branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self._session.get(Branch, branch_id)
        if branch is None or branch.project_id != project_id:
            raise LookupError("分支不存在")
        return branch

    def _current_state(self, branch_id: str) -> BranchStateVersion | None:
        return self._session.scalar(
            select(BranchStateVersion)
            .where(
                BranchStateVersion.branch_id == branch_id,
                BranchStateVersion.is_current.is_(True),
            )
            .order_by(BranchStateVersion.version.desc())
        )

    def _jobs(self, branch_id: str) -> list[Job]:
        return [
            job
            for job in self._session.scalars(
                select(Job).where(Job.kind.in_((CONTINUAL_MEMORY_JOB_KIND, "branch_reflection")))
            )
            if job.payload.get("branch_id") == branch_id
        ]
