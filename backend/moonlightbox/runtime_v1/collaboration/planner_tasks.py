"""平级计划任务：Job/检查点保存工作，RuntimeEvent 交付问题或回执。

规划调查不占 Director 执行权。需要对方意见时立即结束本次 Job，
收到回答才从保存的规划工作继续；不是等待中持锁调用另一个 Agent。
"""

from datetime import UTC, datetime
from time import time

from langchain_core.messages import messages_from_dict, messages_to_dict
from sqlalchemy import select

from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService

from ..branch_models import Branch
from ..context import ContextAssembler
from ..db_models import RuntimeCycleTraceRow, RuntimeEventRow
from ..event_queue import RuntimeEventQueue
from ..executor import RuntimeExecutor
from .graph import build_collaboration_graph
from .messages import discussion_payload, shared_message
from .persistence import branch_writer, collaboration_checkpointer


def make_dispatch(session, branch_id, packet, decision):
    if decision.plan_request:
        return {"request": decision.plan_request.model_dump(mode="json")}
    read_ids = {item["id"] for item in packet.trigger.get("input_events", [])}
    read_ids.update(
        item["source_event_id"] for item in packet.current.get("pending_planner_questions", [])
    )
    requests = list(
        session.scalars(
            select(RuntimeEventRow).where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.id.in_(read_ids),
                RuntimeEventRow.event_type == "planner_question",
            )
        )
    )
    ref = decision.peer_reply.request_ref
    if ref:
        requests = [event for event in requests if event.id == ref]
    if len(requests) != 1:
        raise ValueError("请用 request_ref 指明本轮已收到的计划协作问题")
    parent = session.get(Job, requests[0].payload["planner_job_id"])
    if (
        not parent
        or not (parent.checkpoint or {}).get("planner_state")
        or not parent.checkpoint.get("awaiting_answer")
    ):
        raise ValueError("计划工作尚未保存，不能回复不存在的协作问题")
    return {
        "resume": parent.checkpoint["planner_state"],
        "answer": decision.peer_reply.model_dump(mode="json"),
        "parent_job_id": parent.id,
    }


def pending_questions(session, branch_id):
    """已消费的问题仍可能在等用户补充；不能因一次聊天结束而丢掉协商。"""
    result = []
    for event in session.scalars(
        select(RuntimeEventRow).where(
            RuntimeEventRow.branch_id == branch_id,
            RuntimeEventRow.event_type == "planner_question",
        )
    ):
        parent = session.get(Job, event.payload.get("planner_job_id"))
        if parent and (parent.checkpoint or {}).get("awaiting_answer"):
            result.append(
                {
                    "source_event_id": event.id,
                    "discussion": event.payload.get("discussion", []),
                }
            )
    return result


def enqueue_dispatch(session, branch_id, cycle_key, dispatch, messages):
    if dispatch.get("parent_job_id"):
        parent = session.get(Job, dispatch["parent_job_id"])
        parent.checkpoint = {**parent.checkpoint, "awaiting_answer": False}
    return JobService(session).enqueue_unique(
        "runtime-v1-cycle",
        {
            "project_id": session.get(Branch, branch_id).project_id,
            "branch_id": branch_id,
            "planner_task": dispatch,
            "discussion": messages_to_dict(messages),
        },
        dedupe_key=f"planner:{cycle_key}",
        commit=False,
    )


def run_planner_task(runtime, job):
    try:
        return _run_planner_task(runtime, job)
    except Exception as error:
        runtime.session.rollback()
        trace = runtime.session.scalar(
            select(RuntimeCycleTraceRow)
            .where(RuntimeCycleTraceRow.job_id == job.id)
            .order_by(RuntimeCycleTraceRow.started_at.desc())
            .limit(1)
        )
        if trace and trace.status != "succeeded":
            trace.status, trace.stage = "failed", "failed"
            trace.error_code = getattr(error, "code", type(error).__name__)
            trace.error_message = str(error)[:500]
            trace.completed_at = datetime.now(UTC)
            runtime.session.commit()
        raise


def _run_planner_task(runtime, job):
    """只运行 Planner/Executor；到 Director 的路由变成持久化交付后让出。"""
    from ..service import RuntimeModelExecutionError
    from .nodes import RuntimePeerNodes

    session = runtime.session
    branch_id = job.payload["branch_id"]
    with branch_writer(session, f"planner:{branch_id}") as acquired:
        if not acquired:
            raise RuntimeModelExecutionError("branch_writer_busy", "Planner 正在处理上一项工作")
        branch = session.get(Branch, branch_id)
        bootstrap = RuntimeExecutor(session).bootstrap(
            project_id=branch.project_id, branch_id=branch_id
        )
        packet = ContextAssembler(session).assemble(
            branch_id=branch_id,
            trigger={"type": "plan_request", "payload": {}},
            clock=bootstrap["clock"],
            snapshot=bootstrap["snapshot"],
        )
        trace = RuntimeCycleTraceRow(
            project_id=branch.project_id,
            branch_id=branch_id,
            job_id=job.id,
            cycle_key=f"planner:{job.id}",
            trigger_event_ids=[],
            wakeup_ids=[],
            virtual_now=packet.virtual_now,
            started_at=datetime.now(UTC),
        )
        session.add(trace)
        session.commit()
        nodes = RuntimePeerNodes(
            runtime,
            project_id=branch.project_id,
            branch=branch,
            bootstrap=bootstrap,
            packet=packet,
            trace=trace,
            input_revision=branch.runtime_input_revision,
            cycle_key=f"planner:{job.id}",
            trigger_ids=[],
            planning_only=False,
        )
        dispatch = job.payload["planner_task"]
        initial = dict(dispatch.get("resume", {}))
        messages = messages_from_dict(job.payload.get("discussion", []))
        if dispatch.get("answer"):
            messages.append(
                shared_message(
                    "director",
                    "day_planner",
                    "answer",
                    dispatch["answer"]["content"],
                    initial.get("task_id", f"planner:{job.id}"),
                )
            )
        initial.update(
            branch_id=branch_id,
            cycle_id=f"planner:{job.id}",
            task_id=initial.get("task_id", f"planner:{job.id}"),
            input_revision=branch.runtime_input_revision,
            started_at=time(),
            route="day_planner",
            status="running",
            messages=messages,
            request=dispatch.get("request") or initial.get("request"),
            decision=None,
            proposal=None,
            result={},
        )
        with collaboration_checkpointer(session) as saver:
            graph = build_collaboration_graph(
                director=nodes.director,
                day_planner=nodes.day_planner,
                executor=nodes.executor_node,
                checkpointer=saver,
                input_is_current=nodes.current,
                yield_director=True,
            )
            config = {"configurable": {"thread_id": f"runtime-planner:{job.id}"}}
            saved = graph.get_state(config)
            if saved.values and not saved.next and saved.values.get("route") in {"director", "end"}:
                state = saved.values  # 交付前崩溃：复用结果，不重复调查/提交。
            else:
                if saved.values:
                    from .graph import resume_window

                    # Job 批准恢复，继续原节点和累计消耗，只重开本次墙钟窗口。
                    graph.update_state(config, resume_window())
                    if not saved.next:
                        initial = {**saved.values, **resume_window()}
                state = graph.invoke(None if saved.next else initial, config, durability="sync")
        serial = {key: value for key, value in state.items() if key != "messages"}
        question = bool(state.get("request"))
        job.checkpoint = {
            **(job.checkpoint or {}),
            "planner_state": serial,
            "awaiting_answer": question,
        }
        if question:
            from ..plan_status import set_preparation

            target = state["request"].get("target_date") or nodes.packet.virtual_now.date()
            if isinstance(target, str):
                target = datetime.fromisoformat(target).date()
            set_preparation(
                session,
                branch_id,
                target,
                "blocked",
                job_id=job.id,
                owner_key=nodes.cycle_key,
                error_code="awaiting_peer_reply",
            )
        new_messages = [
            message
            for message in state.get("messages", [])
            if message.id not in {m.id for m in messages}
        ]
        # 单纯消化通知不制造自动回复；否则两个 Agent 会互相确认、无限唤醒。
        if new_messages or state.get("receipt") or question:
            RuntimeEventQueue(session).enqueue(
                project_id=branch.project_id,
                branch_id=branch_id,
                event_type="planner_question" if question else "planner_notification",
                occurred_at=runtime.get_clock(branch.project_id, branch_id).now(),
                idempotency_key=f"planner-delivery:{job.id}",
                payload={
                    "planner_job_id": job.id,
                    "discussion": discussion_payload(
                        [
                            message
                            for message in state.get("messages", [])
                            if message.id not in {m.id for m in messages}
                        ]
                    ),
                    "receipt": state.get("receipt"),
                },
            )
        for event_id in dispatch.get("input_ids", []):
            event = session.get(RuntimeEventRow, event_id)
            if event and event.branch_id == branch_id and event.event_type == "planner_inbox":
                event.status = "completed"
                event.completed_at = datetime.now(UTC)
        trace.status, trace.stage = "succeeded", "delivered"
        trace.completed_at = datetime.now(UTC)
        session.commit()
