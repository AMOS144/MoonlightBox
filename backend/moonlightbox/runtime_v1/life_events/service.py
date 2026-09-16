"""DayPlan 生活任务：复用平级 LangGraph、统一 Controller、Job 恢复及公共消息。"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from time import time

from langchain_core.messages import messages_from_dict
from opentelemetry import trace
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.cancellation import job_signal
from moonlightbox.agent_runtime.resilience import ExecutionInterrupted
from moonlightbox.jobs.models import Job

from ..branch_models import Branch
from ..collaboration.graph import build_collaboration_graph, resume_window
from ..collaboration.messages import discussion_payload, shared_message
from ..collaboration.persistence import branch_writer, collaboration_checkpointer
from ..collaboration.transport import peer_history
from ..context import ContextAssembler
from ..context_views import director_context_payload
from ..db_models import (
    RuntimeClockRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeSnapshotRow,
)
from ..executor import RuntimeExecutor
from ..tools.memory_search import build_memory_search_tool
from ..tools.recent_life_events import build_recent_life_events_tool, recent_life_events
from ..tools.send_agent_message import build_send_agent_message_tool
from .commit import batch_requests, schedule_followup
from .models import LifeOpportunityRow
from .scheduling import branch_now


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def total_usage(meters):
    return {
        k: sum(m.get(k, 0) for m in meters.values())
        for k in ("model_steps", "tool_calls", "provider_tokens")
    }


def persist_usage(session, opportunity_id, thread, usage):
    if not usage:
        return
    row = session.get(LifeOpportunityRow, opportunity_id)
    meters = dict(row.work.get("usage_by_thread", {}))
    meters[thread] = {k: max(v, meters.get(thread, {}).get(k, 0)) for k, v in usage.items()}
    row.work = {**row.work, "usage_by_thread": meters}
    session.commit()


class LifeEventNodes:
    def __init__(self, runtime, row, job):
        self.runtime, self.session, self.row, self.job = runtime, runtime.session, row, job
        self.branch = self.session.get(Branch, row.branch_id)
        self.revision = self.branch.runtime_input_revision
        self.worker_token = job.worker_token
        self.cancelled = job_signal(self.session.get_bind(), job.id, job.worker_token)

    def current(self):
        with Session(self.session.get_bind()) as reader:
            branch = reader.get(Branch, self.row.branch_id)
            job = reader.get(Job, self.job.id)
            clock = reader.get(RuntimeClockRow, self.row.branch_id)
            return (
                branch
                and branch.runtime_input_revision == self.revision
                and clock
                and clock.status == "running"
                and job
                and job.status == "running"
                and job.worker_token == self.worker_token
            )

    def now(self):
        return branch_now(self.session.get(RuntimeClockRow, self.row.branch_id), datetime.now(UTC))

    def message(self, sender, recipient, kind, content, payload=None):
        message = shared_message(sender, recipient, kind, content, f"life:{self.row.id}", payload)
        message.id = fingerprint([self.job.id, sender, kind, content, payload])
        return message

    def context(self, state):
        now = self.now()
        snapshot = self.session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == self.row.branch_id)
        )
        clock = self.runtime.get_clock(self.branch.project_id, self.row.branch_id)
        packet = ContextAssembler(self.session).assemble(
            branch_id=self.row.branch_id,
            trigger={"type": "system", "payload": {"reason": "life_progression"}},
            clock=clock,
            snapshot=snapshot,
            now=now,
        )
        packet.current["collaboration"] = peer_history(
            self.session,
            self.row.branch_id,
            recipient="day_planner",
            current=discussion_payload(state.get("messages", [])),
        )
        result = director_context_payload(packet)
        requests = batch_requests(self.session, self.row)
        earliest = min(
            [
                now.date() - timedelta(days=1),
                *[
                    datetime.fromisoformat(r["recovery"]["from"]).date()
                    for r in requests
                    if r.get("recovery")
                ],
            ]
        )
        plans = list(
            self.session.scalars(
                select(RuntimeDayPlanRow).where(
                    RuntimeDayPlanRow.branch_id == self.row.branch_id,
                    RuntimeDayPlanRow.plan_date >= earliest.isoformat(),
                )
            )
        )
        self.row.work = {**self.row.work, "plan_versions": {p.plan_date: p.version for p in plans}}
        parent_ids = [r["parent_event_id"] for r in requests if r.get("parent_event_id")]
        parents = list(
            self.session.scalars(
                select(RuntimeLifeEventRow).where(
                    RuntimeLifeEventRow.branch_id == self.row.branch_id,
                    RuntimeLifeEventRow.id.in_(parent_ids),
                )
            )
        )
        result.update(
            virtual_now=now.isoformat(),
            life_requests=requests,
            plans_by_date={p.plan_date: {"blocks": p.blocks} for p in plans},
            earlier_events=[{"source_id": p.id, "event": p.payload} for p in parents],
            candidate=state.get("life", {}).get("candidate"),
            candidate_status="not_occurred",
            last_rejection=state.get("life", {}).get("last_rejection"),
            recent_simulated_events=recent_life_events(self.session, self.row.branch_id, now),
        )
        span = trace.get_current_span()
        span.set_attribute("moonlightbox.life.opportunity_id", self.row.id)
        span.set_attribute("moonlightbox.life.batch_size", len(requests))
        span.set_attribute("moonlightbox.life.policy_version", self.row.policy.get("version", ""))
        tools = [
            build_send_agent_message_tool(
                self.session,
                branch_id=self.row.branch_id,
                sender="day_planner",
                task_id=f"life:{self.row.id}:{self.row.step}",
                cancellation_requested=self.cancelled,
            ),
            build_memory_search_tool(
                self.session, branch_id=self.row.branch_id, snapshot_id=snapshot.id, as_of=now
            ),
            build_recent_life_events_tool(self.session, self.row.branch_id, now),
        ]
        self.session.commit()
        return dict(
            context=result,
            tools=tools,
            branch_id=self.row.branch_id,
            project_id=self.branch.project_id,
            owner_id=self.row.id,
            input_revision=self.revision,
            resolver=self.runtime._runtime_input_revision_resolver(self.row.branch_id),
            cancellation_requested=self.cancelled,
            validate_plan_sources=lambda refs: RuntimeExecutor(
                self.session
            )._validate_plan_evidence_sources(self.row.branch_id, refs),
        )

    def planner(self, state):
        value = self.runtime.planner.advance_life(**self.context(state))
        life = {**state.get("life", {}), "candidate": value.model_dump(mode="json")}
        route = {"no_event": "end", "discuss": "director", "submit_event": "executor"}[
            value.outcome
        ]
        return {
            "life": life,
            "route": route,
            "status": "no_event" if value.outcome == "no_event" else "running",
            "messages": [
                self.message(
                    "day_planner",
                    "executor" if route == "executor" else "director",
                    "submit" if route == "executor" else "proposal",
                    value.reason,
                    {"candidate": life["candidate"], "status": "not_occurred"},
                )
            ],
        }

    def executor(self, state):
        if not self.current():
            raise ExecutionInterrupted("cancelled" if self.cancelled() else "stale_input_revision")
        try:
            with self.session.begin_nested():
                # 与写入同一事务确认租约；失效任务不能提交。
                job = self.session.scalar(
                    select(Job).where(
                        Job.id == self.job.id,
                        Job.status == "running",
                        Job.worker_token == self.worker_token,
                    )
                )
                if not job:
                    raise ExecutionInterrupted("cancelled")
                event = RuntimeExecutor(self.session).commit_simulated_event(
                    self.row,
                    decision=state["life"]["candidate"],
                    input_revision=self.revision,
                    now=self.now(),
                )
            self.session.commit()
        except ValueError as error:
            self.session.rollback()
            return {
                "life": {**state["life"], "last_rejection": str(error)},
                "route": "day_planner",
                "messages": [
                    self.message(
                        "executor",
                        "day_planner",
                        "commit_result",
                        str(error),
                        {"status": "rejected"},
                    )
                ],
            }
        return {
            "route": "end",
            "status": "committed",
            "life": state["life"],
            "messages": [
                self.message(
                    "executor",
                    "day_planner",
                    "commit_result",
                    "事件与计划已提交，通知已入队",
                    {"event_id": event.id},
                )
            ],
        }


def process_life_step(runtime, job):
    session = runtime.session
    row = session.get(LifeOpportunityRow, job.payload["life_opportunity_id"])
    if row and row.kind == "legacy":
        raise ExecutionInterrupted("legacy_life_work_requires_review")
    if not row or row.job_id != job.id:
        return
    # 已提交但 graph/消费回执尚未写完时仍需恢复，不能提前 return 丢批次。
    if row.status not in {"active", "committed", "no_event"}:
        return
    with branch_writer(session, f"planner:{row.branch_id}") as acquired:
        if not acquired:
            raise ExecutionInterrupted("branch_busy")
        nodes = LifeEventNodes(runtime, row, job)
        if not nodes.current():
            raise ExecutionInterrupted("cancelled" if nodes.cancelled() else "branch_busy")
        if row.work.get("awaiting_director_event_id"):
            return
        with collaboration_checkpointer(session) as saver:
            graph = build_collaboration_graph(
                director=None,
                day_planner=nodes.planner,
                executor=nodes.executor,
                checkpointer=saver,
                input_is_current=nodes.current,
                yield_director=True,
                usage_observer=lambda thread, usage: persist_usage(session, row.id, thread, usage),
            )
            config = {"configurable": {"thread_id": f"runtime-life-job:{job.id}"}}
            saved = graph.get_state(config)
            if saved.values and not saved.next and saved.values.get("route") in {"end", "director"}:
                state = saved.values
            else:
                initial = dict(row.work.get("state", {}))
                initial.update(
                    branch_id=row.branch_id,
                    cycle_id=f"life:{row.id}:step:{row.step}",
                    task_id=f"life:{row.id}",
                    input_revision=nodes.revision,
                    route="day_planner",
                    status="running",
                    started_at=time(),
                    life=initial.get("life", {}),
                    messages=messages_from_dict(row.work.get("peer_messages", [])),
                    usage=total_usage(row.work.get("usage_by_thread", {})),
                )
                if saved.values:
                    graph.update_state(config, resume_window())
                state = graph.invoke(None if saved.next else initial, config, durability="sync")
        row = session.get(LifeOpportunityRow, row.id)
        serial = {
            k: v
            for k, v in state.items()
            if k not in {"messages", "tasks", "proposals", "commits", "delivery"}
        }
        row.work = {**row.work, "state": serial}
        now = nodes.now()
        if state.get("route") == "director":
            from .delivery import request_advice

            request_advice(session, row, serial, now)
        else:
            if state.get("status") == "no_event":
                row.status = "no_event"
                candidate = state.get("life", {}).get("candidate", {})
                if candidate.get("follow_up_at"):
                    # 无新事件但仍需跟进已有事件，不凭空创建已发生内容。
                    parent = next(
                        (
                            r["parent_event_id"]
                            for r in batch_requests(session, row)
                            if r.get("parent_event_id")
                        ),
                        None,
                    )
                    if parent:
                        schedule_followup(
                            session, row, datetime.fromisoformat(candidate["follow_up_at"]), parent
                        )
            for key in row.work.get("batch_ids", []):
                item = session.get(LifeOpportunityRow, key)
                if item and item.id != row.id and item.status == "batched":
                    item.status, item.event_id = row.status, row.event_id
        # 读取过的来信才完成；在途新增消息留给下一次。
        for key in row.work.get("reply_event_ids", []):
            event = session.get(RuntimeEventRow, key)
            if event and event.branch_id == row.branch_id:
                event.status, event.completed_at = "completed", datetime.now(UTC)
        if row.work.get("completed_job_id") != job.id:
            row.step += 1
            row.work = {**row.work, "completed_job_id": job.id}
        session.commit()
