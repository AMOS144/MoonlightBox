"""Runtime v1 HTTP API；请求只写队列，Cycle 由显式 worker/处理接口执行。"""

from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService

from .executor import RuntimeExecutor, RuntimeWorldUnavailableError
from .schemas import MemoryRecord
from .service import RuntimeService
from .source_history import SourceHistoryError, SourceHistoryService


class RuntimeMessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=10000)
    idempotency_key: str = Field(min_length=1, max_length=160)
    # 仅用于前端把乐观气泡与服务端落盘消息去重；幂等键仍是 RuntimeEvent 的唯一键。
    client_message_id: str | None = Field(default=None, min_length=1, max_length=64)
    occurred_at: datetime | None = None


class BranchChatMessageCreate(BaseModel):
    """面向聊天界面的通用消息契约。

    前端只知道自己在向一个时间分支发送消息；事件入队、世界快照初始化和
    Director/PersonaActor 的编排均是后端 Runtime 的实现细节。因此这里不把
    ``runtime``、``event`` 或内部队列字段暴露到常规聊天路由中。
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=10000)
    # 客户端若提供稳定 ID，重试同一条发送时会复用同一 RuntimeEvent。
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


class RuntimeBranchCreate(BaseModel):
    """创建 Runtime 分支所需的最小、可审计输入。"""

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(min_length=1)
    preview_hash: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=255)
    publication_id: str | None = None


def create_runtime_router(database: Database, settings: Settings) -> APIRouter:
    router = APIRouter(tags=["runtime-v1"])
    # 这是产品层的分支/聊天 API。它刻意不带 ``runtime`` 路径，让聊天前端继续
    # 保持普通微信式会话的心智模型；实际编排由下方 RuntimeService 完成。
    public_branch_router = APIRouter(
        prefix="/api/projects/{project_id}/branches", tags=["branches"]
    )
    branch_router = APIRouter(
        prefix="/api/projects/{project_id}/runtime/branches", tags=["runtime-v1"]
    )
    cycle_router = APIRouter(
        prefix="/api/projects/{project_id}/branches/{branch_id}/runtime", tags=["runtime-v1"]
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    def preparation_payload(project_id, branch_id, session):
        from .db_models import RuntimeInitializationRow, RuntimeSnapshotRow

        branch = RuntimeService(session)._branch(project_id, branch_id)
        initialization = session.get(RuntimeInitializationRow, branch_id)
        snapshot = session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
        )
        bootstrap = session.scalar(
            select(Job).where(Job.dedupe_key == f"runtime-v1-cycle:bootstrap:{branch_id}")
        )
        understanding_started = initialization is not None and initialization.status != "pending"
        state = (
            "ready"
            if branch.lifecycle_status not in {"preparing", "prepare_failed"}
            else "failed"
            if branch.lifecycle_status == "prepare_failed"
            else "preparing"
        )
        # 透出失败任务的真实错误码（如 quota_exhausted），前端据此给出可操作提示。
        failure_code = (
            (initialization.error_code if initialization else None)
            or (bootstrap.error_code if bootstrap is not None and bootstrap.status == "failed" else None)
            or "preparation_failed"
        )
        return {
            "branch_id": branch_id,
            "title": branch.title,
            "origin_time": branch.origin_time,
            "background_mode": snapshot.snapshot_mode if snapshot else None,
            "background": snapshot.profile if snapshot else None,
            "status": state,
            # 没有可测总工作量就不编造百分比，前端显示业务阶段即可。
            "progress": 1 if state == "ready" else None,
            "stage": "completed"
            if state == "ready"
            else "initializing_director"
            if understanding_started
            else "planning",
            "message_count": 0,
            "event_count": 0,
            "error_code": failure_code if state == "failed" else None,
            "error_message": "分支准备未完成，已有进展保留。" if state == "failed" else None,
        }

    @public_branch_router.get("/{branch_id}/preparation")
    def get_preparation(project_id: str, branch_id: str, session: SessionDependency):
        try:
            return preparation_payload(project_id, branch_id, session)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @public_branch_router.post("/{branch_id}/preparation/retry")
    def retry_preparation(project_id: str, branch_id: str, session: SessionDependency):
        try:
            branch = RuntimeService(session)._branch(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if branch.lifecycle_status != "prepare_failed":
            return preparation_payload(project_id, branch_id, session)
        job = session.scalar(
            select(Job).where(Job.dedupe_key == f"runtime-v1-cycle:bootstrap:{branch_id}")
        )
        if job is None or job.status not in {"failed", "interrupted"}:
            raise HTTPException(status_code=409, detail="准备任务仍在运行或不存在")
        branch.lifecycle_status = "preparing"
        JobService(session).resume(job.id)
        return preparation_payload(project_id, branch_id, session)

    @public_branch_router.post("", status_code=status.HTTP_201_CREATED)
    def create_public_branch(
        project_id: str, payload: RuntimeBranchCreate, session: SessionDependency
    ) -> dict[str, object]:
        """创建时间分支；前端无需知道其底层由 Runtime 驱动。"""

        try:
            branch = RuntimeService(session).create_branch(
                project_id=project_id,
                investigation_id=payload.investigation_id,
                preview_hash=payload.preview_hash,
                title=payload.title,
                publication_id=payload.publication_id,
            )
            return _branch_payload(branch)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @public_branch_router.get("")
    def list_public_branches(
        project_id: str, session: SessionDependency
    ) -> list[dict[str, object]]:
        return [_branch_payload(item) for item in RuntimeService(session).list_branches(project_id)]

    @public_branch_router.get("/{branch_id}/messages")
    def list_public_messages(
        project_id: str, branch_id: str, session: SessionDependency
    ) -> list[dict[str, object]]:
        try:
            return [
                _branch_message_payload(item)
                for item in RuntimeService(session).list_messages(project_id, branch_id)
            ]
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @public_branch_router.get("/{branch_id}/history")
    def list_source_history(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
        before: str | None = None,
        limit: int = Query(default=40, ge=20, le=100),
    ) -> dict[str, object]:
        """读取起点节点及其之前的导入消息，供普通聊天页面滚动展示。"""

        try:
            page = SourceHistoryService(session).page(
                project_id, branch_id, before=before, limit=limit
            )
            return {
                "items": [
                    {
                        "id": item.id,
                        "source_id": item.source_id,
                        "role": item.role,
                        "content": item.content,
                        "type": item.type,
                        "media_asset_id": item.media_asset_id,
                        "timestamp": item.timestamp,
                    }
                    for item in page.items
                ],
                "next_cursor": page.next_cursor,
                "has_more": page.has_more,
                "manifest_id": page.manifest_id,
            }
        except SourceHistoryError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @public_branch_router.get("/{branch_id}/clock")
    def get_public_branch_clock(
        project_id: str, branch_id: str, session: SessionDependency
    ) -> dict[str, object]:
        """为聊天界面提供当前分支时间，不暴露 Runtime 内部锚点和调度细节。"""

        try:
            clock = RuntimeService(session).get_clock(project_id, branch_id)
            from .collaboration.plans import local_time, shared_plan_window

            now = local_time(clock.now(), clock.timezone)
            today = shared_plan_window(session, branch_id, now, clock.timezone)[
                now.date().isoformat()
            ]
            return {
                # 当前钟面由后端计算，浏览器不自行重建锚点公式，避免客户端时钟漂移。
                "virtual_now": now,
                "day_plan_preparation": today["preparation"],
                "has_day_plan": today["status"] == "available",
                "timezone": clock.timezone,
                "status": clock.status,
                "time_scale": clock.time_scale,
            }
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @public_branch_router.post("/{branch_id}/messages", status_code=status.HTTP_202_ACCEPTED)
    def send_public_message(
        project_id: str,
        branch_id: str,
        payload: BranchChatMessageCreate,
        session: SessionDependency,
    ) -> dict[str, object]:
        """普通聊天发送入口；内部再将消息可靠写入 Runtime 事件队列。"""

        return _enqueue_message(
            session=session,
            project_id=project_id,
            branch_id=branch_id,
            content=payload.content,
            idempotency_key=payload.client_message_id or str(uuid4()),
            client_message_id=payload.client_message_id,
            occurred_at=payload.occurred_at,
        )

    @branch_router.post("", status_code=status.HTTP_201_CREATED)
    def create_branch(
        project_id: str, payload: RuntimeBranchCreate, session: SessionDependency
    ) -> dict[str, object]:
        try:
            branch = RuntimeService(session).create_branch(
                project_id=project_id,
                investigation_id=payload.investigation_id,
                preview_hash=payload.preview_hash,
                title=payload.title,
                publication_id=payload.publication_id,
            )
            return _branch_payload(branch)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @branch_router.get("")
    def list_branches(project_id: str, session: SessionDependency) -> list[dict[str, object]]:
        return [_branch_payload(item) for item in RuntimeService(session).list_branches(project_id)]

    @branch_router.get("/{branch_id}/messages")
    def list_messages(
        project_id: str, branch_id: str, session: SessionDependency
    ) -> list[dict[str, object]]:
        try:
            return [
                _branch_message_payload(item)
                for item in RuntimeService(session).list_messages(project_id, branch_id)
            ]
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @cycle_router.post("/bootstrap", status_code=status.HTTP_201_CREATED)
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

    @cycle_router.get("/snapshot")
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

    @cycle_router.get("/state")
    def state(project_id: str, branch_id: str, session: SessionDependency) -> dict[str, object]:
        try:
            return RuntimeService(session).get_state(project_id, branch_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @cycle_router.get("/clock")
    def get_clock(project_id: str, branch_id: str, session: SessionDependency) -> object:
        try:
            return RuntimeService(session).get_clock(project_id, branch_id).model_dump(mode="json")
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @cycle_router.get("/day-plan")
    def get_day_plan(project_id: str, branch_id: str, session: SessionDependency) -> object:
        try:
            plan = RuntimeService(session).get_plan(project_id, branch_id)
            return {"branch_id": branch_id, "date": plan.plan_date, "blocks": plan.blocks}
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @cycle_router.get("/traces")
    def traces(
        project_id: str,
        branch_id: str,
        session: SessionDependency,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """仅供本地体验期诊断：按 Cycle 返回模型、工具与提交审计信息。"""

        try:
            return [
                _cycle_trace_payload(item)
                for item in RuntimeService(session).list_cycle_traces(
                    project_id, branch_id, limit=limit
                )
            ]
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @cycle_router.post("/messages", status_code=status.HTTP_202_ACCEPTED)
    def enqueue_message(
        project_id: str, branch_id: str, payload: RuntimeMessageCreate, session: SessionDependency
    ) -> dict[str, object]:
        return _enqueue_message(
            session=session,
            project_id=project_id,
            branch_id=branch_id,
            content=payload.content,
            idempotency_key=payload.idempotency_key,
            client_message_id=payload.client_message_id,
            occurred_at=payload.occurred_at,
        )

    def _enqueue_message(
        *,
        session: Session,
        project_id: str,
        branch_id: str,
        content: str,
        idempotency_key: str,
        client_message_id: str | None,
        occurred_at: datetime | None,
    ) -> dict[str, object]:
        try:
            result = RuntimeService(session).submit_user_message(
                project_id=project_id,
                branch_id=branch_id,
                content=content,
                idempotency_key=idempotency_key,
                client_message_id=client_message_id,
                occurred_at=occurred_at,
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
    @cycle_router.post("/events", status_code=status.HTTP_202_ACCEPTED)
    def enqueue_event(
        project_id: str, branch_id: str, payload: RuntimeMessageCreate, session: SessionDependency
    ) -> dict[str, object]:
        return enqueue_message(project_id, branch_id, payload, session)

    @cycle_router.post("/memories/confirm", status_code=status.HTTP_201_CREATED)
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

    @cycle_router.post("/clock")
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

    router.include_router(public_branch_router)
    router.include_router(branch_router)
    router.include_router(cycle_router)
    return router


def _branch_payload(branch: object) -> dict[str, object]:
    fields = (
        "id",
        "project_id",
        "origin_event_id",
        "origin_boundary",
        "title",
        "origin_time",
        "lifecycle_status",
        "created_at",
    )
    return {
        **{field: getattr(branch, field) for field in fields},
        "baseline_status": "ready"
        if branch.lifecycle_status not in {"preparing", "prepare_failed"}
        else "failed"
        if branch.lifecycle_status == "prepare_failed"
        else "preparing",
    }


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
        "is_proactive",
        "created_at",
    )
    return {field: getattr(message, field) for field in fields}


def _cycle_trace_payload(trace: object) -> dict[str, object]:
    fields = (
        "id",
        "project_id",
        "branch_id",
        "job_id",
        "cycle_key",
        "status",
        "stage",
        "trigger_event_ids",
        "wakeup_ids",
        "virtual_now",
        "packet",
        "planner_steps",
        "planner_proposal",
        "director_steps",
        "actor_steps",
        "director_decision",
        "committed_decision",
        "outcome",
        "error_code",
        "error_message",
        "started_at",
        "completed_at",
    )
    return {field: getattr(trace, field) for field in fields}
