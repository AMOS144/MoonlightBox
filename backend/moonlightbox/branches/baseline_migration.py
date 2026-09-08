from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.baseline_boundary import (
    BaselineBoundaryError,
    BaselineBoundaryResolver,
)
from moonlightbox.branches.baseline_jobs import BRANCH_BASELINE_JOB_KIND
from moonlightbox.branches.models import Branch
from moonlightbox.jobs.service import JobService


@dataclass(frozen=True)
class BranchBaselineMigrationResult:
    branch_id: str
    status: str
    job_id: str | None = None
    error: str | None = None


@dataclass
class BaselineMigrationReport:
    project_id: str
    branches: list[BranchBaselineMigrationResult] = field(default_factory=list)


class BaselineMigrationService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def migrate_project(self, project_id: str) -> BaselineMigrationReport:
        report = BaselineMigrationReport(project_id=project_id)
        for branch in self._session.scalars(
            select(Branch)
            .where(Branch.project_id == project_id)
            .order_by(Branch.created_at, Branch.id)
        ):
            report.branches.append(self.migrate_branch(branch.id))
        return report

    def migrate_branch(self, branch_id: str) -> BranchBaselineMigrationResult:
        branch = self._session.get(Branch, branch_id)
        if branch is None:
            raise LookupError("时间分支不存在")
        if branch.baseline_manifest_id is not None and branch.baseline_status == "ready":
            return BranchBaselineMigrationResult(branch.id, "ready", branch.baseline_job_id)
        try:
            boundary = BaselineBoundaryResolver(self._session).resolve(
                branch.project_id, branch.origin_event_id
            )
        except BaselineBoundaryError as error:
            branch.baseline_status = "failed"
            branch.baseline_error_code = "baseline_boundary_ambiguous"
            branch.baseline_error_message = str(error)
            self._session.commit()
            return BranchBaselineMigrationResult(
                branch.id, "failed", error=str(error)
            )
        branch.origin_import_id = boundary.import_id
        branch.origin_boundary_message_id = boundary.message_id
        branch.baseline_status = "preparing"
        branch.baseline_error_code = None
        branch.baseline_error_message = None
        job = JobService(self._session).enqueue_unique(
            BRANCH_BASELINE_JOB_KIND,
            {"project_id": branch.project_id, "branch_id": branch.id},
            dedupe_key=f"{BRANCH_BASELINE_JOB_KIND}:{branch.id}",
        )
        branch.baseline_job_id = job.id
        self._session.commit()
        return BranchBaselineMigrationResult(branch.id, "queued", job.id)
