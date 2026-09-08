from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from moonlightbox.agent.acceptance import ReplayObservation
from moonlightbox.agent.activation_service import (
    AcceptanceRejectedError,
    AcceptanceReportNotFoundError,
    BranchScopeError,
    ModelScopeError,
    SubjectAgentActivationService,
)
from moonlightbox.agent.schemas import (
    AcceptanceEvaluateRequest,
    SubjectAgentAcceptanceReportRead,
    SubjectAgentActivateRequest,
    SubjectAgentModeRead,
)
from moonlightbox.db import Database


def create_subject_agent_activation_router(database: Database) -> APIRouter:
    """创建严格按项目、分支和模型隔离的主体 Agent 激活路由。"""

    router = APIRouter(
        prefix="/api/projects/{project_id}/branches/{branch_id}/subject-agent",
        tags=["subject-agent-activation"],
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    def translate_scope_error(error: Exception) -> HTTPException:
        if isinstance(error, BranchScopeError):
            return HTTPException(status_code=404, detail=str(error))
        if isinstance(error, AcceptanceReportNotFoundError):
            return HTTPException(status_code=404, detail=str(error))
        return HTTPException(status_code=409, detail=str(error))

    @router.get(
        "/acceptance/latest",
        response_model=SubjectAgentAcceptanceReportRead,
    )
    def get_latest_report(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return SubjectAgentActivationService(session).latest_bound_report(
                project_id,
                branch_id,
            )
        except (BranchScopeError, ModelScopeError, AcceptanceReportNotFoundError) as error:
            raise translate_scope_error(error) from error

    @router.post(
        "/acceptance/evaluate",
        response_model=SubjectAgentAcceptanceReportRead,
        status_code=status.HTTP_201_CREATED,
    )
    def evaluate(
        project_id: str,
        branch_id: str,
        payload: AcceptanceEvaluateRequest,
        session: SessionDependency,
    ) -> object:
        observations = [
            ReplayObservation(**observation.model_dump())
            for observation in payload.observations
        ]
        try:
            return SubjectAgentActivationService(session).evaluate_and_save(
                project_id,
                branch_id,
                payload.model_version_id,
                observations,
                direct_lora_p95=payload.direct_lora_p95,
            )
        except (BranchScopeError, ModelScopeError) as error:
            raise translate_scope_error(error) from error

    @router.post("/activate", response_model=SubjectAgentModeRead)
    def activate(
        project_id: str,
        branch_id: str,
        payload: SubjectAgentActivateRequest,
        session: SessionDependency,
    ) -> object:
        try:
            return SubjectAgentActivationService(session).activate_branch(
                project_id,
                branch_id,
                payload.model_version_id,
            )
        except (
            BranchScopeError,
            ModelScopeError,
            AcceptanceRejectedError,
        ) as error:
            raise translate_scope_error(error) from error

    @router.post("/rollback", response_model=SubjectAgentModeRead)
    def rollback(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return SubjectAgentActivationService(session).rollback_branch(
                project_id,
                branch_id,
            )
        except BranchScopeError as error:
            raise translate_scope_error(error) from error

    return router
