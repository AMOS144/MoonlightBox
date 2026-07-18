from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from moonlightbox.branches.generation import (
    BranchGenerator,
    GeneratorUnavailableError,
)
from moonlightbox.branches.schemas import (
    BranchCreate,
    BranchMessageCreate,
    BranchMessageRead,
    BranchRead,
)
from moonlightbox.branches.service import BranchNotFoundError, BranchService
from moonlightbox.db import Database


def create_branches_router(
    database: Database,
    generator: BranchGenerator,
) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/branches", tags=["branches"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.post("", response_model=BranchRead, status_code=status.HTTP_201_CREATED)
    def create_branch(
        project_id: str,
        payload: BranchCreate,
        session: SessionDependency,
    ) -> object:
        return BranchService(session, generator).create(project_id, payload)

    @router.get("", response_model=list[BranchRead])
    def list_branches(project_id: str, session: SessionDependency) -> object:
        return BranchService(session, generator).list_branches(project_id)

    @router.get("/{branch_id}/messages", response_model=list[BranchMessageRead])
    def list_messages(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(session, generator).messages(project_id, branch_id)
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error

    @router.post(
        "/{branch_id}/messages",
        response_model=BranchMessageRead,
        status_code=status.HTTP_201_CREATED,
    )
    def generate_message(
        project_id: str,
        branch_id: str,
        payload: BranchMessageCreate,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(session, generator).add_user_message(
                project_id, branch_id, payload.content
            )
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except GeneratorUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return router
