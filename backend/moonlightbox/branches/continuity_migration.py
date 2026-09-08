from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.episodes import (
    CONTINUAL_MEMORY_JOB_KIND,
    EpisodeService,
)
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.jobs.models import Job
from moonlightbox.training.models import ModelVersion


class BranchMigrationReport(BaseModel):
    branch_id: str
    episode_count: int
    state_version_count: int
    queued_job_count: int
    index_document_count: int


class ProjectMigrationReport(BaseModel):
    project_id: str
    kernel_id: str
    branch_reports: list[BranchMigrationReport]


class ContinuityMigrationService:
    def __init__(
        self,
        session: Session,
        continuity_repository: BranchContinuityRepository | None = None,
    ) -> None:
        self._session = session
        self._continuity_repository = continuity_repository

    def migrate_project(self, project_id: str) -> ProjectMigrationReport:
        kernel = self.migrate_active_model(project_id)
        branches = list(
            self._session.scalars(
                select(Branch)
                .where(Branch.project_id == project_id)
                .order_by(Branch.created_at.asc())
            )
        )
        reports = [self.backfill_branch(branch.id) for branch in branches]
        self._session.commit()
        return ProjectMigrationReport(
            project_id=project_id,
            kernel_id=kernel.id,
            branch_reports=reports,
        )

    def migrate_active_model(self, project_id: str) -> IdentityKernel:
        model = self._session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.project_id == project_id,
                ModelVersion.active.is_(True),
            )
            .order_by(ModelVersion.created_at.desc())
        )
        if model is None:
            raise LookupError("项目没有活动模型")
        existing = self._session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == model.id)
        )
        if (
            existing is not None
            and existing.schema_version == "subject-persona-v2"
            and existing.acceptance_report_id is not None
        ):
            return existing
        raise ValueError(
            "活动模型缺少主体人格 V2 验收内核，必须重新训练后才能迁移"
        )

    def backfill_branch(self, branch_id: str) -> BranchMigrationReport:
        branch = self._session.get(Branch, branch_id)
        if branch is None:
            raise LookupError("分支不存在")
        if branch.generation_policy_version != "subject-v2-actor-v1":
            branch.lifecycle_status = "archived"
        self._ensure_initial_state(branch)
        messages = list(
            self._session.scalars(
                select(BranchMessage)
                .where(BranchMessage.branch_id == branch.id)
                .order_by(BranchMessage.sequence.asc())
            )
        )
        episode_service = EpisodeService(self._session)
        index = 0
        while index < len(messages):
            user = messages[index]
            if user.role != "user" or user.generation_status != "completed":
                index += 1
                continue
            assistants: list[BranchMessage] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].role != "user":
                candidate = messages[cursor]
                if candidate.role == "assistant" and candidate.generation_status == "completed":
                    assistants.append(candidate)
                cursor += 1
            if assistants:
                episode_service.record_turn(branch, user, assistants)
            index = cursor
        self._session.flush()
        index_document_count = (
            self._continuity_repository.rebuild_branch(
                self._session,
                branch.id,
            )
            if self._continuity_repository is not None
            else 0
        )
        jobs = list(self._session.scalars(select(Job).where(Job.kind == CONTINUAL_MEMORY_JOB_KIND)))
        return BranchMigrationReport(
            branch_id=branch.id,
            episode_count=int(
                self._session.scalar(
                    select(func.count(BranchMemoryEpisode.id)).where(
                        BranchMemoryEpisode.branch_id == branch.id
                    )
                )
                or 0
            ),
            state_version_count=int(
                self._session.scalar(
                    select(func.count(BranchStateVersion.id)).where(
                        BranchStateVersion.branch_id == branch.id
                    )
                )
                or 0
            ),
            queued_job_count=sum(job.payload.get("branch_id") == branch.id for job in jobs),
            index_document_count=index_document_count,
        )

    def rebuild_branch_index(self, branch_id: str) -> int:
        if self._continuity_repository is None:
            raise RuntimeError("分支长期记忆索引未配置")
        return self._continuity_repository.rebuild_branch(
            self._session,
            branch_id,
        )

    def _ensure_initial_state(self, branch: Branch) -> BranchStateVersion:
        existing = self._session.scalar(
            select(BranchStateVersion)
            .where(BranchStateVersion.branch_id == branch.id)
            .order_by(BranchStateVersion.version.asc())
        )
        if existing is not None:
            return existing
        original = (
            dict(branch.state_snapshot)
            if branch.generation_policy_version == "subject-v2-actor-v1"
            else {}
        )
        state = BranchStateVersion(
            branch_id=branch.id,
            version=1,
            previous_version_id=None,
            persona_state=_object_dict(original.get("persona_state")),
            relationship_state=_object_dict(original.get("relationship_state")),
            user_model=_object_dict(original.get("user_model")),
            emotional_tendency=_object_dict(original.get("emotional_tendency")),
            active_belief_ids=_string_list(original.get("active_belief_ids")),
            reason="迁移现有分支状态",
            source_episode_ids=[],
            is_current=True,
        )
        self._session.add(state)
        self._session.flush()
        branch.state_snapshot = {
            "protocol_version": "subject-persona-v2",
            "state_version_id": state.id,
            "version": state.version,
            "persona_state": state.persona_state,
            "relationship_state": state.relationship_state,
            "user_model": state.user_model,
            "emotional_tendency": state.emotional_tendency,
            "active_belief_ids": state.active_belief_ids,
            "contested_belief_ids": state.contested_belief_ids,
            "current_goals": state.current_goals,
            "current_concerns": state.current_concerns,
        }
        self._session.flush()
        return state


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
