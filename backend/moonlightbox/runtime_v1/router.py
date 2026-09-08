"""Runtime v1 HTTP API；请求只写队列，Cycle 由显式 worker/处理接口执行。"""

from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database

from .executor import RuntimeExecutor, RuntimeWorldUnavailableError
from .schemas import MemoryRecord
from .service import RuntimeService


class RuntimeMessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=10000)
    idempotency_key: str = Field(min_length=1, max_length=160)
    # 仅用于前端把乐观气泡与服务端落盘消息去重；幂等键仍是 RuntimeEvent 的唯一键。
    client_message_id: str | None = Field(default=None, min_length=1, max_length=64)
    occurred_at: datetime | None = None


class RuntimeClockAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume"]


class RuntimeMemoryConfirm(BaseModel):
    """用户明确确认后写入 branch 范围记忆的最小请求。"""

    model_config = ConfigDict(extra="forbid")
    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=120)
    object: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] = Field(min_length=1, max_length=20)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    supersedes_id: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


def create_runtime_router(database: Database, settings: Settings) -> APIRouter:
    router = APIRouter(
        prefix="/api/projects/{project_id}/branches/{branch_id}/runtime", tags=["runtime-v1"]
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]
    @router.post("/bootstrap", status_code=status.HTTP_201_CREATED)
    def bootstrap(project_id: str, branch_id: str, session: SessionDependency) -> dict[str, object]:
        try:
            result = RuntimeService(session).bootstrap(project_id, branch_id)
            return {
                "snapshot_id": result["snapshot"].id,
                "state": result["state"].state,
                "clock": result["clock"].model_dump(mode="json"),
            }
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeWorldUnavailableError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/snapshot")
    def snapshot(project_id: str, branch_id: str, session: SessionDependency) -> object:
        try:
            row = RuntimeService(session).get_snapshot(project_id, branch_id)
            return {
                "id": row.id,
                "branch_id": row.branch_id,
                "graph_version_id": row.graph_version_id,
                "profile_id": row.profile_id,
                "source_node_id": row.source_node_id,
                "cutoff_at": row.cutoff_at,
                "timezone": row.timezone,
                "snapshot_mode": row.snapshot_mode,
                "source_message_ids": row.source_message_ids,
                "profile": row.profile,
                "routine_profile": row.routine_profile,
                "compiler_version": row.compiler_version,
                "created_at": row.created_at,
            }
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/state")
    def state(project_id: str, branch_id: str, session: SessionDependency) -> dict[str, object]:
        try:
            return RuntimeService(session).get_state(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/clock")
    def get_clock(project_id: str, branch_id: str, session: SessionDependency) -> object:
        try:
            return RuntimeService(session).get_clock(project_id, branch_id).model_dump(mode="json")
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/day-plan")
    def get_day_plan(project_id: str, branch_id: str, session: SessionDependency) -> object:
        try:
            plan = RuntimeService(session).get_plan(project_id, branch_id)
            return {"branch_id": branch_id, "date": plan.plan_date, "blocks": plan.blocks}
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/messages", status_code=status.HTTP_202_ACCEPTED)
    def enqueue_message(
        project_id: str, branch_id: str, payload: RuntimeMessageCreate, session: SessionDependency
    ) -> dict[str, object]:
        try:
            result = RuntimeService(session).submit_user_message(
                project_id=project_id,
                branch_id=branch_id,
                content=payload.content,
                idempotency_key=payload.idempotency_key,
                client_message_id=payload.client_message_id,
                occurred_at=payload.occurred_at,
            )
            return {
                "event_id": result["event"].id,
                "message_id": result["message"].id,
                "message": _branch_message_payload(result["message"]),
            }
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeWorldUnavailableError as error:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(error)) from error

    # events 是 messages 的语义别名，方便 Worker/客户端按设计文档命名调用。
    @router.post("/events", status_code=status.HTTP_202_ACCEPTED)
    def enqueue_event(
        project_id: str, branch_id: str, payload: RuntimeMessageCreate, session: SessionDependency
    ) -> dict[str, object]:
        return enqueue_message(project_id, branch_id, payload, session)

    @router.post("/memories/confirm", status_code=status.HTTP_201_CREATED)
    def confirm_memory(
        project_id: str,
        branch_id: str,
        payload: RuntimeMemoryConfirm,
        session: SessionDependency,
    ) -> object:
        """显式确认分支事实；Director 和 PersonaActor 都没有此 API 的等价工具。"""

        try:
            RuntimeService(session).get_snapshot(project_id, branch_id)
            record = MemoryRecord(
                id=str(uuid4()),
                scope="branch",
                branch_id=branch_id,
                subject=payload.subject,
                predicate=payload.predicate,
                object=payload.object,
                summary=payload.summary,
                status="confirmed",
                source_ids=payload.source_ids,
                valid_from=payload.valid_from,
                valid_to=payload.valid_to,
                supersedes_id=payload.supersedes_id,
                confidence=payload.confidence,
            )
            RuntimeExecutor(session).confirm_memory(record)
            session.commit()
            return record.model_dump(mode="json")
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            session.rollback()
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.post("/clock")
    def change_clock(
        project_id: str, branch_id: str, payload: RuntimeClockAction, session: SessionDependency
    ) -> object:
        try:
            return RuntimeService(session).change_clock(
                project_id, branch_id, action=payload.action
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return router


def _branch_message_payload(message: object) -> dict[str, object]:
    """Runtime 队列响应复用现有聊天列表契约，页面无需等待 Worker 才显示用户原文。"""

    fields = (
        "id",
        "branch_id",
        "sequence",
        "role",
        "content",
        "type",
        "media_asset_id",
        "turn_id",
        "bubble_index",
        "delay_ms",
        "generation_status",
        "generation_metadata",
        "client_message_id",
        "observed_at",
        "expression_plan_id",
        "actor_intent",
        "is_proactive",
        "created_at",
    )
    return {field: getattr(message, field) for field in fields}
