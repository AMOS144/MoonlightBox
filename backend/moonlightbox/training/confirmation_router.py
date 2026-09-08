from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.training.confirmation import (
    ConfirmationService,
    TimelineConfirmationError,
)
from moonlightbox.training.schemas import ConfirmAndTrainRead, ConfirmAndTrainRequest


def create_training_confirmation_router(
    database: Database,
    settings: Settings,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/projects/{project_id}/events",
        tags=["training"],
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.post("/confirm-and-train", response_model=ConfirmAndTrainRead)
    def confirm_and_train(
        project_id: str,
        payload: ConfirmAndTrainRequest,
        session: SessionDependency,
    ) -> ConfirmAndTrainRead:
        try:
            result = ConfirmationService(session).confirm(
                project_id=project_id,
                analysis_run_id=payload.analysis_run_id,
                event_revisions=[
                    (item.event_id, item.revision_number) for item in payload.event_revisions
                ],
                config=settings.training_config_snapshot(),
            )
        except TimelineConfirmationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return ConfirmAndTrainRead(
            confirmation_id=result.confirmation.id,
            training_job_id=result.job.id,
            status=result.job.status,
        )

    return router
