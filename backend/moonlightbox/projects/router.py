from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.projects.schemas import ProjectCreate, ProjectRead
from moonlightbox.projects.service import ProjectNotFoundError, ProjectService


def create_projects_router(database: Database, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/projects", tags=["projects"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    def get_service(session: SessionDependency) -> ProjectService:
        return ProjectService(
            session,
            (settings.data_dir, settings.chroma_dir, settings.model_dir),
        )

    ServiceDependency = Annotated[ProjectService, Depends(get_service)]

    @router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
    def create_project(
        payload: ProjectCreate,
        service: ServiceDependency,
    ) -> object:
        return service.create(payload.name)

    @router.get("", response_model=list[ProjectRead])
    def list_projects(service: ServiceDependency) -> object:
        return service.list()

    @router.get("/{project_id}", response_model=ProjectRead)
    def get_project(
        project_id: str,
        service: ServiceDependency,
    ) -> object:
        try:
            return service.get(project_id)
        except ProjectNotFoundError as error:
            raise HTTPException(status_code=404, detail="项目不存在") from error

    @router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_project(
        project_id: str,
        service: ServiceDependency,
    ) -> Response:
        try:
            service.delete(project_id)
        except ProjectNotFoundError as error:
            raise HTTPException(status_code=404, detail="项目不存在") from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
