from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.jobs.schemas import JobRead
from moonlightbox.jobs.service import (
    InvalidJobTransitionError,
    JobNotFoundError,
    JobService,
)


def create_jobs_router(database: Database) -> APIRouter:
    router = APIRouter(prefix="/api/jobs", tags=["jobs"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    def get_service(session: SessionDependency) -> JobService:
        return JobService(session)

    ServiceDependency = Annotated[JobService, Depends(get_service)]

    @router.get("", response_model=list[JobRead])
    def list_project_jobs(
        project_id: str,
        service: ServiceDependency,
    ) -> object:
        return service.list_for_project(project_id)

    @router.get("/{job_id}", response_model=JobRead)
    def get_job(job_id: str, service: ServiceDependency) -> object:
        try:
            return service.get(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="任务不存在") from error

    @router.post("/{job_id}/cancel", response_model=JobRead)
    def cancel_job(job_id: str, service: ServiceDependency) -> object:
        try:
            return service.cancel(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="任务不存在") from error
        except InvalidJobTransitionError as error:
            raise HTTPException(status_code=409, detail="任务当前状态不可取消") from error

    @router.post("/{job_id}/resume", response_model=JobRead)
    def resume_job(job_id: str, service: ServiceDependency) -> object:
        try:
            return service.resume(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="任务不存在") from error
        except InvalidJobTransitionError as error:
            raise HTTPException(status_code=409, detail="任务当前状态不可恢复") from error

    return router
