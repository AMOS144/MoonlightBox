from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.events.schemas import (
    EventNodeRead,
    EventRevisionRequest,
    ReviewedEvent,
)
from moonlightbox.events.service import EventNotFoundError, EventService


def create_events_router(database: Database) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/events", tags=["events"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.post("", response_model=EventNodeRead, status_code=status.HTTP_201_CREATED)
    def create_event(
        project_id: str,
        payload: ReviewedEvent,
        session: SessionDependency,
    ) -> object:
        return EventService(session).create(
            project_id,
            payload,
            analysis_version="manual-v1",
            prompt_version="manual-v1",
        )

    @router.get("", response_model=list[EventNodeRead])
    def list_events(
        project_id: str,
        session: SessionDependency,
        lane: Literal["relationship", "shared_experience"] | None = None,
    ) -> object:
        return EventService(session).list_read(project_id, lane=lane)

    @router.patch("/{event_id}", response_model=EventNodeRead)
    def revise_event(
        project_id: str,
        event_id: str,
        payload: EventRevisionRequest,
        session: SessionDependency,
    ) -> object:
        service = EventService(session)
        try:
            event = service.revise(
                event_id,
                payload.changes,
                payload.reason,
                project_id=project_id,
            )
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail="事件节点不存在") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return event

    return router
