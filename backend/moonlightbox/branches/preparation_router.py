from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from moonlightbox.branches.baseline_migration import BaselineMigrationService
from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.history import (
    BranchHistoryCursorError,
    BranchHistoryService,
)
from moonlightbox.branches.models import Branch
from moonlightbox.branches.schemas import (
    BranchHistoryPageRead,
    BranchPreparationRead,
)
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import InvalidJobTransitionError, JobService


def create_branch_preparation_router(database: Database) -> APIRouter:
    router = APIRouter(
        prefix="/api/projects/{project_id}/branches",
        tags=["branch-preparation"],
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.get("/{branch_id}/preparation", response_model=BranchPreparationRead)
    def preparation(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        branch = _branch(session, project_id, branch_id)
        job = session.get(Job, branch.baseline_job_id)
        manifest = (
            session.get(BranchBaselineManifest, branch.baseline_manifest_id)
            if branch.baseline_manifest_id is not None
            else None
        )
        checkpoint = job.checkpoint if job is not None and job.checkpoint else {}
        return {
            "branch_id": branch.id,
            "status": branch.baseline_status,
            "progress": job.progress
            if job is not None
            else (1.0 if branch.baseline_status == "ready" else 0.0),
            "stage": str(checkpoint.get("stage", branch.baseline_status)),
            "message_count": manifest.message_count if manifest is not None else 0,
            "event_count": (manifest.event_snapshot_count if manifest is not None else 0),
            "error_code": branch.baseline_error_code,
            "error_message": branch.baseline_error_message,
        }

    @router.post("/baseline-migration")
    def migrate_baselines(
        project_id: str,
        session: SessionDependency,
    ) -> object:
        report = BaselineMigrationService(session).migrate_project(project_id)
        return {
            "project_id": report.project_id,
            "branches": [
                {
                    "branch_id": item.branch_id,
                    "status": item.status,
                    "job_id": item.job_id,
                    "error": item.error,
                }
                for item in report.branches
            ],
        }

    @router.post("/{branch_id}/preparation/retry", response_model=BranchPreparationRead)
    def retry(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        branch = _branch(session, project_id, branch_id)
        if branch.baseline_job_id is None:
            raise HTTPException(status_code=409, detail="分支没有基础历史任务")
        try:
            JobService(session).resume(branch.baseline_job_id)
        except InvalidJobTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        branch.baseline_status = "preparing"
        branch.baseline_error_code = None
        branch.baseline_error_message = None
        session.commit()
        return preparation(project_id, branch_id, session)

    @router.get("/{branch_id}/history", response_model=BranchHistoryPageRead)
    def history(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
        before: str | None = None,
        limit: int = Query(default=40, ge=20, le=100),
    ) -> object:
        try:
            return BranchHistoryService(session).page(
                project_id,
                branch_id,
                before=before,
                limit=limit,
            )
        except BranchHistoryCursorError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return router


def _branch(session: Session, project_id: str, branch_id: str) -> Branch:
    branch = session.get(Branch, branch_id)
    if branch is None or branch.project_id != project_id:
        raise HTTPException(status_code=404, detail="时间分支不存在")
    return branch
