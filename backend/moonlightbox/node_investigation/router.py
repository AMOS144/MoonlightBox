"""工作台 HTTP 只做用户操作和任务排队，模型调用不占住请求连接。"""

from zoneinfo import ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.tool_errors import ToolInputError
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import InvalidJobTransitionError

from .models import NodeInvestigation
from .schemas import ConfirmArgs, ContextArgs, InputArgs, PreviewArgs, ScopeArgs, StartArgs
from .store import InvestigationStore, activity, queue_turn


def create_node_investigation_router(database):
    router = APIRouter(
        prefix="/api/projects/{project_id}/node-investigations", tags=["node-investigation"]
    )

    def call(fn):
        try:
            return fn()
        except (ToolInputError, InvalidJobTransitionError, ZoneInfoNotFoundError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    def store(project_id, investigation_id):
        return InvestigationStore(database.engine, project_id, investigation_id)

    @router.get("")
    def current(project_id: str):
        with Session(database.engine) as session:
            latest = session.scalar(
                select(NodeInvestigation.id)
                .where(NodeInvestigation.project_id == project_id)
                .order_by(NodeInvestigation.created_at.desc())
            )
        return call(lambda: store(project_id, latest).view()) if latest else None

    @router.post("")
    def start(project_id: str, args: StartArgs):
        return call(lambda: InvestigationStore(database.engine, project_id).start(args.timezone))

    @router.get("/{investigation_id}")
    def read(project_id: str, investigation_id: str):
        return call(lambda: store(project_id, investigation_id).view())

    @router.post("/{investigation_id}/inputs")
    def user_input(project_id: str, investigation_id: str, args: InputArgs):
        return call(lambda: store(project_id, investigation_id).add_input(args))

    @router.post("/{investigation_id}/scope")
    def scope(project_id: str, investigation_id: str, args: ScopeArgs):
        return call(lambda: store(project_id, investigation_id).set_scope(args))

    @router.post("/{investigation_id}/pause")
    def pause(project_id: str, investigation_id: str):
        workspace = store(project_id, investigation_id)

        def apply(state, session):
            state["status"] = "paused"
            job = session.get(Job, state.get("job_id"))
            if job and job.status == "running":
                job.status = "cancelling"
            elif job and job.status == "queued":
                job.status = "cancelled"
            activity(
                state, "paused", "用户暂停调查；正在进行的同步操作将在退出后停止，候选与进度保留"
            )

        call(lambda: workspace.mutate(apply))
        return workspace.view()

    @router.post("/{investigation_id}/resume")
    def resume(project_id: str, investigation_id: str):
        workspace = store(project_id, investigation_id)
        row = call(workspace.get)
        with Session(database.engine) as session:
            job = session.get(Job, row.state.get("job_id"))
            if job and job.status == "cancelling":
                raise HTTPException(status_code=409, detail="旧执行仍在退出，请稍后继续")
            if job and job.status in {"failed", "interrupted"}:
                # 相同 Job/turn 恢复同一 Controller 检查点，不清空已有调查和累计预算。
                def retry(state, transaction):
                    current = transaction.get(Job, state["job_id"])
                    if current.status not in {"failed", "interrupted"}:
                        raise ToolInputError("任务状态已改变，请刷新")
                    current.status = "queued"
                    current.error_code = current.error_message = None
                    state.update(status="queued", error=None)

                call(lambda: workspace.mutate(retry))
            elif (job and job.status == "cancelled") or row.state["status"] not in {
                "queued",
                "running",
            }:
                call(
                    lambda: workspace.mutate(
                        lambda state, s: queue_turn(s, workspace.id, project_id, state)
                    )
                )
        return workspace.view()

    @router.post("/{investigation_id}/context")
    def context(project_id: str, investigation_id: str, args: ContextArgs):
        workspace = store(project_id, investigation_id)

        def load():
            rows = workspace.source_rows(workspace.get().state["message_ids"])
            positions = {m.id: i for i, (m, _) in enumerate(rows)}
            if any(ref not in positions for ref in args.message_refs):
                raise ToolInputError("引用不属于本次记录")
            indexes = {
                i
                for ref in args.message_refs
                for i in range(
                    max(0, positions[ref] - args.radius),
                    min(len(rows), positions[ref] + args.radius + 1),
                )
            }
            return {"messages": [workspace.message_json(*rows[i], i) for i in sorted(indexes)]}

        return call(load)

    @router.get("/{investigation_id}/messages")
    def messages(
        project_id: str,
        investigation_id: str,
        cursor: int = Query(0, ge=0),
        limit: int = Query(40, ge=1, le=100),
    ):
        workspace = store(project_id, investigation_id)

        def load():
            rows = workspace.source_rows(workspace.get().state["message_ids"])
            if cursor > len(rows):
                raise ToolInputError("消息页超出记录范围")
            end = min(len(rows), cursor + limit)
            return {
                "messages": [workspace.message_json(*rows[i], i) for i in range(cursor, end)],
                "next_cursor": end,
                "total": len(rows),
            }

        return call(load)

    @router.post("/{investigation_id}/candidates/{candidate_id}/preview")
    def preview(project_id: str, investigation_id: str, candidate_id: str, args: PreviewArgs):
        return call(lambda: store(project_id, investigation_id).preview(candidate_id, args))

    @router.post("/{investigation_id}/candidates/{candidate_id}/confirm")
    def confirm(project_id: str, investigation_id: str, candidate_id: str, args: ConfirmArgs):
        return call(lambda: store(project_id, investigation_id).confirm(candidate_id, args))

    return router
