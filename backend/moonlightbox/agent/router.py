from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent.models import (
    AgentGoal,
    AgentIntention,
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
    PrivateCognitionNote,
)
from moonlightbox.agent.schemas import (
    CognitionStateRead,
    CognitiveCycleRead,
    PerceptionEventRead,
)
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database


def create_cognition_router(database: Database) -> APIRouter:
    """创建严格按项目和分支隔离的只读认知审计路由。"""

    router = APIRouter(
        prefix="/api/projects/{project_id}/branches/{branch_id}/cognition",
        tags=["cognition"],
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]
    Limit = Annotated[int, Query(ge=1, le=100)]

    def require_branch(session: Session, project_id: str, branch_id: str) -> None:
        branch_id_in_scope = session.scalar(
            select(Branch.id).where(
                Branch.id == branch_id,
                Branch.project_id == project_id,
            )
        )
        if branch_id_in_scope is None:
            raise HTTPException(status_code=404, detail="时间分支不存在")

    @router.get("/state", response_model=CognitionStateRead)
    def get_state(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
    ) -> object:
        require_branch(session, project_id, branch_id)
        mental_state = session.scalar(
            select(MentalStateVersion).where(
                MentalStateVersion.project_id == project_id,
                MentalStateVersion.branch_id == branch_id,
                MentalStateVersion.is_current.is_(True),
            )
        )
        active_goals = list(
            session.scalars(
                select(AgentGoal)
                .where(
                    AgentGoal.project_id == project_id,
                    AgentGoal.branch_id == branch_id,
                    AgentGoal.status == "active",
                )
                .order_by(AgentGoal.priority.desc(), AgentGoal.created_at.desc())
            )
        )
        active_intentions = list(
            session.scalars(
                select(AgentIntention)
                .where(
                    AgentIntention.project_id == project_id,
                    AgentIntention.branch_id == branch_id,
                    AgentIntention.status == "active",
                )
                .order_by(AgentIntention.created_at.desc())
            )
        )
        next_wakeup = session.scalar(
            select(AgentWakeup)
            .where(
                AgentWakeup.project_id == project_id,
                AgentWakeup.branch_id == branch_id,
                AgentWakeup.status == "scheduled",
            )
            .order_by(AgentWakeup.wake_at.asc(), AgentWakeup.id.asc())
            .limit(1)
        )
        return {
            "mental_state": mental_state,
            "active_goals": active_goals,
            "active_intentions": active_intentions,
            "next_wakeup": next_wakeup,
        }

    @router.get("/cycles", response_model=list[CognitiveCycleRead])
    def list_cycles(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
        limit: Limit = 50,
    ) -> object:
        require_branch(session, project_id, branch_id)
        cycles = list(
            session.scalars(
                select(CognitiveCycle)
                .where(
                    CognitiveCycle.project_id == project_id,
                    CognitiveCycle.branch_id == branch_id,
                )
                .order_by(CognitiveCycle.created_at.desc(), CognitiveCycle.id.desc())
                .limit(limit)
            )
        )
        note_ids = [cycle.private_note_id for cycle in cycles if cycle.private_note_id]
        notes = (
            {
                note.id: note
                for note in session.scalars(
                    select(PrivateCognitionNote).where(
                        PrivateCognitionNote.project_id == project_id,
                        PrivateCognitionNote.branch_id == branch_id,
                        PrivateCognitionNote.id.in_(note_ids),
                    )
                )
            }
            if note_ids
            else {}
        )
        return [
            {
                "id": cycle.id,
                "trigger_event_id": cycle.trigger_event_id,
                "input_cutoff_at": cycle.input_cutoff_at,
                "starting_state_version_id": cycle.starting_state_version_id,
                "memory_ids": cycle.memory_ids,
                "goal_ids": cycle.goal_ids,
                "structured_changes": cycle.structured_changes,
                "final_decision": cycle.final_decision,
                "evidence": cycle.evidence,
                "status": cycle.status,
                "model_version_id": cycle.model_version_id,
                "model_protocol_version": cycle.model_protocol_version,
                "created_at": cycle.created_at,
                "started_at": cycle.started_at,
                "completed_at": cycle.completed_at,
                "private_note": notes.get(cycle.private_note_id),
            }
            for cycle in cycles
        ]

    @router.get("/events", response_model=list[PerceptionEventRead])
    def list_events(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
        limit: Limit = 50,
    ) -> object:
        require_branch(session, project_id, branch_id)
        return list(
            session.scalars(
                select(PerceptionEvent)
                .where(
                    PerceptionEvent.project_id == project_id,
                    PerceptionEvent.branch_id == branch_id,
                )
                .order_by(
                    PerceptionEvent.occurred_at.desc(),
                    PerceptionEvent.created_at.desc(),
                    PerceptionEvent.id.desc(),
                )
                .limit(limit)
            )
        )

    return router
