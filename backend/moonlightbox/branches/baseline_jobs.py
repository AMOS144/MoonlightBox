from datetime import UTC, datetime

from moonlightbox.branches.baseline_boundary import (
    BaselineBoundaryError,
    BaselineBoundaryResolver,
)
from moonlightbox.branches.baseline_service import (
    BaselineIntegrityError,
    BranchBaselineService,
)
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.models import Branch
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService

BRANCH_BASELINE_JOB_KIND = "branch_baseline_build"


def create_branch_baseline_handler(
    memory_repository: ProjectMemoryRepository | None = None,
) -> JobHandler:
    def handle(job_service: JobService, job: Job) -> None:
        session = job_service.session
        branch_id = job.payload.get("branch_id")
        project_id = job.payload.get("project_id")
        if not isinstance(branch_id, str) or not isinstance(project_id, str):
            raise JobHandlerError("invalid_payload", "基础历史任务参数无效")
        branch = session.get(Branch, branch_id)
        if branch is None or branch.project_id != project_id:
            raise JobHandlerError("branch_not_found", "时间分支不存在")
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_lost", "基础历史任务租约已失效")
        try:
            _checkpoint(job_service, job, token, "resolving_boundary", 0.1)
            boundary = BaselineBoundaryResolver(session).resolve(project_id, branch.origin_event_id)
            _checkpoint(job_service, job, token, "freezing_messages", 0.25)
            _checkpoint(job_service, job, token, "freezing_events", 0.4)
            _checkpoint(job_service, job, token, "building_state", 0.6)
            manifest, _state, _version = BranchBaselineService(session).build(branch, boundary)
            _checkpoint(job_service, job, token, "building_index", 0.8)
            if memory_repository is not None:
                memory_repository.rebuild_project(session, project_id)
            manifest.index_fingerprint = manifest.message_digest
            _checkpoint(job_service, job, token, "validating", 0.95)
            BranchBaselineService(session).validate(branch, manifest, boundary)
            branch.baseline_status = "ready"
            branch.baseline_error_code = None
            branch.baseline_error_message = None
            branch.baseline_ready_at = datetime.now(UTC)
            _checkpoint(job_service, job, token, "completed", 1.0)
        except (BaselineBoundaryError, BaselineIntegrityError) as error:
            session.rollback()
            branch = session.get(Branch, branch_id)
            if branch is not None:
                branch.baseline_status = "failed"
                branch.baseline_error_code = "baseline_build_failed"
                branch.baseline_error_message = str(error)
                session.commit()
            raise JobHandlerError("baseline_build_failed", str(error)) from error
        except Exception as error:
            session.rollback()
            branch = session.get(Branch, branch_id)
            if branch is not None:
                branch.baseline_status = "failed"
                branch.baseline_error_code = "baseline_build_failed"
                branch.baseline_error_message = "基础历史构建失败"
                session.commit()
            raise JobHandlerError(
                "baseline_build_failed", "基础历史构建失败"
            ) from error

    return handle


def _checkpoint(
    service: JobService,
    job: Job,
    token: str,
    stage: str,
    progress: float,
) -> None:
    service.checkpoint(
        job.id,
        {"stage": stage, "progress": progress},
        token=token,
    )
