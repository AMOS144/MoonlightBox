from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from moonlightbox.imports.models import Participant

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
    def list_projects(service: ServiceDependency, session: SessionDependency) -> object:
        projects = service.list()
        # 列表批量读取真实人物身份，不为每张卡请求完整流程/背景。
        targets = {p.project_id: p for p in session.scalars(select(Participant).where(
            Participant.project_id.in_([p.id for p in projects]), Participant.role == "target",
        ).order_by(Participant.id))}
        return [ProjectRead.model_validate(p).model_copy(update={
            "target_name": targets[p.id].name if p.id in targets else None,
            "target_avatar_asset_id": targets[p.id].avatar_asset_id if p.id in targets else None,
        }) for p in projects]

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

    @router.get("/{project_id}/journey")
    def get_journey(project_id: str, session: SessionDependency, service: ServiceDependency):
        from .journey import project_journey

        try:
            service.get(project_id)
        except ProjectNotFoundError as error:
            raise HTTPException(status_code=404, detail="项目不存在") from error
        return project_journey(session, project_id)

    return router
