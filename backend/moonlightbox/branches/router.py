from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from moonlightbox.agent.service import BranchScopeError, ShadowCognitionService
from moonlightbox.branches.baseline_boundary import BaselineBoundaryError
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.generation import (
    BranchGenerator,
    GenerationFailedError,
    GeneratorUnavailableError,
)
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.models import Branch
from moonlightbox.branches.reviewer import ReplyReviewer
from moonlightbox.branches.schemas import (
    BranchCreate,
    BranchMessageCreate,
    BranchMessageRead,
    BranchRead,
    ConversationActorStateRead,
    ConversationTypingUpdate,
    SituationalStateRead,
    SituationalStateWrite,
)
from moonlightbox.branches.service import (
    ActiveModelUnavailableError,
    BranchBaselineNotReadyError,
    BranchModelNotAcceptedError,
    BranchNotFoundError,
    BranchReadOnlyError,
    BranchService,
    CandidateModelNotReadyError,
    stage_candidate_branch,
)
from moonlightbox.branches.situational_state import active_situational_state
from moonlightbox.db import Database


def create_branches_router(
    database: Database,
    generator: BranchGenerator,
    reviewer: ReplyReviewer | None = None,
    memory_repository: ProjectMemoryRepository | None = None,
    continuity_repository: BranchContinuityRepository | None = None,
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
        try:
            return BranchService(
                session, generator, reviewer, memory_repository, continuity_repository
            ).create(project_id, payload)
        except BaselineBoundaryError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "baseline_boundary_ambiguous", "message": str(error)},
            ) from error
        except BranchModelNotAcceptedError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "model_not_accepted", "message": str(error)},
            ) from error

    @router.get("", response_model=list[BranchRead])
    def list_branches(project_id: str, session: SessionDependency) -> object:
        return BranchService(
            session, generator, reviewer, memory_repository, continuity_repository
        ).list_branches(project_id)

    @router.post("/{branch_id}/upgrade", response_model=BranchRead)
    def upgrade_branch(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(
                session, generator, reviewer, memory_repository, continuity_repository
            ).upgrade(project_id, branch_id)
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except BaselineBoundaryError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "baseline_boundary_ambiguous", "message": str(error)},
            ) from error
        except ActiveModelUnavailableError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post(
        "/{branch_id}/candidate-model/{model_version_id}",
        response_model=BranchRead,
    )
    def stage_candidate_model_branch(
        project_id: str,
        branch_id: str,
        model_version_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return stage_candidate_branch(
                session,
                project_id=project_id,
                source_branch_id=branch_id,
                model_version_id=model_version_id,
            )
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except (BaselineBoundaryError, CandidateModelNotReadyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/{branch_id}/messages", response_model=list[BranchMessageRead])
    def list_messages(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(
                session, generator, reviewer, memory_repository, continuity_repository
            ).messages(project_id, branch_id)
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except BranchBaselineNotReadyError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "branch_baseline_not_ready", "message": str(error)},
            ) from error

    @router.post(
        "/{branch_id}/messages",
        response_model=BranchMessageRead,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def generate_message(
        project_id: str,
        branch_id: str,
        payload: BranchMessageCreate,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(
                session, generator, reviewer, memory_repository, continuity_repository
            ).add_user_message(
                project_id,
                branch_id,
                payload.content,
                payload.client_message_id,
            )
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except BranchReadOnlyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except BranchBaselineNotReadyError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "branch_baseline_not_ready", "message": str(error)},
            ) from error
        except GeneratorUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except GenerationFailedError as error:
            session.rollback()
            raise HTTPException(
                status_code=503,
                detail={"code": "generation_failed", "message": str(error)},
            ) from error

    @router.get(
        "/{branch_id}/conversation-state",
        response_model=ConversationActorStateRead,
    )
    def get_conversation_state(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(
                session, generator, reviewer, memory_repository, continuity_repository
            ).conversation_state(project_id, branch_id)
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except BranchBaselineNotReadyError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "branch_baseline_not_ready", "message": str(error)},
            ) from error

    @router.put(
        "/{branch_id}/typing",
        response_model=ConversationActorStateRead,
    )
    def update_typing(
        project_id: str,
        branch_id: str,
        payload: ConversationTypingUpdate,
        session: SessionDependency,
    ) -> object:
        try:
            return BranchService(
                session, generator, reviewer, memory_repository, continuity_repository
            ).set_user_typing(project_id, branch_id, payload.typing)
        except BranchNotFoundError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except BranchBaselineNotReadyError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "branch_baseline_not_ready", "message": str(error)},
            ) from error

    @router.put(
        "/{branch_id}/situational-state",
        response_model=SituationalStateRead,
    )
    def update_situational_state(
        project_id: str,
        branch_id: str,
        payload: SituationalStateWrite,
        session: SessionDependency,
    ) -> object:
        try:
            event = ShadowCognitionService(session).append_situational_observation(
                project_id=project_id,
                branch_id=branch_id,
                values=dict(payload.values),
                occurred_at=datetime.now(UTC),
                valid_until=payload.valid_until,
                source="user_configured",
                idempotency_key=payload.idempotency_key,
                confidence=payload.confidence,
            )
            branch = session.get(Branch, branch_id)
            if branch is None:
                raise BranchScopeError("project/branch 不匹配")
            state = active_situational_state(branch.state_snapshot)
            if state is None:
                raise ValueError("情境状态写入后没有有效槽位")
            return {"event_id": event.id, "state": state}
        except BranchScopeError as error:
            raise HTTPException(status_code=404, detail="时间分支不存在") from error
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_situational_state", "message": str(error)},
            ) from error

    return router
