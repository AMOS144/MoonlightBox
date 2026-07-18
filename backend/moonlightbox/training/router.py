from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.evaluation.recommendation import NoQualifiedModelError
from moonlightbox.training.registry import ModelRegistry
from moonlightbox.training.schemas import ModelVersionCreate, ModelVersionRead


def create_models_router(database: Database) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["models"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.post("", response_model=ModelVersionRead, status_code=status.HTTP_201_CREATED)
    def create_model_version(
        project_id: str,
        payload: ModelVersionCreate,
        session: SessionDependency,
    ) -> object:
        return ModelRegistry(session).create(project_id=project_id, **payload.model_dump())

    @router.get("", response_model=list[ModelVersionRead])
    def list_model_versions(project_id: str, session: SessionDependency) -> object:
        return ModelRegistry(session).list(project_id)

    @router.post("/recommend", response_model=ModelVersionRead)
    def recommend_model_version(
        project_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return ModelRegistry(session).recommend(project_id)
        except NoQualifiedModelError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return router
