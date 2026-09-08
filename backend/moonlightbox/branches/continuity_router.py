from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_schemas import (
    BranchMemoryOverviewRead,
    BranchStateVersionRead,
    JobStatusRead,
    MemoryEpisodeRead,
    MemoryItemRead,
    MigrationRead,
    ReflectionRunRead,
)
from moonlightbox.branches.continuity_service import (
    ContinuityNotReadyError,
    ContinuityService,
)
from moonlightbox.db import Database
from moonlightbox.jobs.service import InvalidJobTransitionError


def create_continuity_router(
    database: Database,
    continuity_repository: BranchContinuityRepository | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/projects/{project_id}/branches/{branch_id}/memory",
        tags=["branch-memory"],
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    def service(session: Session) -> ContinuityService:
        return ContinuityService(session, continuity_repository)

    @router.get("", response_model=BranchMemoryOverviewRead)
    def overview(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).overview(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="分支不存在") from error
        except ContinuityNotReadyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/episodes", response_model=list[MemoryEpisodeRead])
    def episodes(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).episodes(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="分支不存在") from error

    @router.get("/beliefs", response_model=list[MemoryItemRead])
    def beliefs(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).beliefs(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="分支不存在") from error

    @router.get("/reflections", response_model=list[ReflectionRunRead])
    def reflections(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).reflections(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="分支不存在") from error

    @router.get("/versions", response_model=list[BranchStateVersionRead])
    def versions(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).versions(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="分支不存在") from error

    @router.post(
        "/versions/{version_id}/rollback",
        response_model=BranchStateVersionRead,
    )
    def rollback(
        project_id: str,
        branch_id: str,
        version_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).rollback(project_id, branch_id, version_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="状态版本不存在") from error

    @router.post("/jobs/{job_id}/retry", response_model=JobStatusRead)
    def retry_job(
        project_id: str,
        branch_id: str,
        job_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).retry_job(project_id, branch_id, job_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="记忆任务不存在") from error
        except InvalidJobTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/jobs", response_model=list[JobStatusRead])
    def jobs(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).jobs(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="分支不存在") from error

    @router.post("/migrate", response_model=MigrationRead)
    def migrate(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return service(session).migrate(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ContinuityNotReadyError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return router
