"""领域服务与协作图的适配层。只有 Executor 节点提交领域副作用。"""

import hashlib
import json
from datetime import UTC, timedelta
from time import time

from langchain_core.messages import HumanMessage
from sqlalchemy import select, update

from moonlightbox.jobs.models import Job
from moonlightbox.observability.phoenix import chain_span, record_span_attributes

from ..branch_models import Branch, BranchMessage
from ..context import ContextAssembler
from ..db_models import RuntimeDayPlanRow, RuntimeEventRow, RuntimeLifeEventRow, RuntimeLifeStateRow
from ..executor import RuntimeExecutor
from ..plan_status import set_preparation
from ..schemas import DayPlanProposal, LifeDecision, PeerReply, PlanRevisionRequest
from ..tools.director_context import build_director_context_tools
from ..tools.memory_search import build_memory_search_tool
from ..tools.recent_life_events import build_recent_life_events_tool
from ..tools.style_examples import StyleService
from .graph import build_collaboration_graph, resume_window
from .messages import discussion_payload, shared_message
from .persistence import collaboration_checkpointer
from .plans import local_time, shared_plan_window


class RuntimePeerNodes:
    """服务对象仅放在节点闭包中，绝不写入 checkpoint。"""

    def __init__(
        self,
        service,
        *,
        project_id,
        branch,
        bootstrap,
        packet,
        trace,
        input_revision,
        cycle_key,
        trigger_ids,
        planning_only,
    ):
        self.service = service
        self.session = service.session
        self.project_id = project_id
        self.branch = branch
        self.branch_id = branch.id
        self.bootstrap = bootstrap
        self.packet = packet
        self.trace = trace
        self.input_revision = input_revision
        self.cycle_key = cycle_key
        self.task_id = cycle_key
        self.trigger_ids = trigger_ids
        self.planning_only = planning_only
        self.executor = RuntimeExecutor(self.session)
        self.result = None
        self.incoming_discussion = []
        job = self.session.get(Job, trace.job_id) if trace.job_id else None
        self.worker_token = job.worker_token if job else None

    def current(self):
        from sqlalchemy.orm import Session

        with Session(self.session.get_bind()) as check:
            revision = check.scalar(
                select(Branch.runtime_input_revision).where(Branch.id == self.branch_id)
            )
            job = check.get(Job, self.trace.job_id) if self.trace.job_id else None
            return revision == self.input_revision and (
                job is None or (job.status == "running" and job.worker_token == self.worker_token)
            )

    def cancelled(self):
        from moonlightbox.agent_runtime.cancellation import job_signal

        return bool(
            self.trace.job_id
            and job_signal(
                self.session.get_bind(),
                self.trace.job_id,
                self.worker_token,
            )()
        )

    def message(self, sender, recipient, kind, content, payload=None):
        return shared_message(sender, recipient, kind, content, self.task_id, payload)

    def refresh(self, state):
        self.session.expire_all()
        self.bootstrap["clock"] = self.service.get_clock(self.project_id, self.branch_id)
        self.packet = ContextAssembler(self.session).assemble(
            branch_id=self.branch_id,
            trigger=self.packet.trigger,
            clock=self.bootstrap["clock"],
            snapshot=self.bootstrap["snapshot"],
            now=local_time(self.bootstrap["clock"].now(), self.packet.timezone),
        )
        self.packet.current["collaboration"] = discussion_payload(state.get("messages", []))
        from .planner_tasks import pending_questions

        self.packet.current["pending_planner_questions"] = pending_questions(
            self.session, self.branch_id
        )
        self.session.commit()

    def director(self, state):
        self.refresh(state)
        from ..conversation_index import search_branch

        query = self.packet.trigger.get("payload", {}).get("content")
        if query:
            self.packet.memory["retrieved_records"] = [
                search_branch(
                    self.session, self.branch_id, str(query), self.packet.virtual_now, limit=6
                )
            ]
        self.packet.current["submission_error"] = state.get("expression_error")
        tool = build_memory_search_tool(
            self.session,
            branch_id=self.branch_id,
            snapshot_id=self.bootstrap["snapshot"].id,
            as_of=self.packet.virtual_now,
            as_of_resolver=lambda: self.packet.virtual_now,
        )

        def receive_inputs(seen, restored_packet=None):
            def persist_read_boundary():
                # 失败终态需要知道本轮实际读过哪些输入，不依赖 Trace，也不误伤后到消息。
                job = self.session.get(Job, self.trace.job_id) if self.trace.job_id else None
                if job and job.status == "running" and job.worker_token == self.worker_token:
                    checkpoint = dict(job.checkpoint or {})
                    checkpoint["runtime_cycle"] = {
                        **checkpoint.get("runtime_cycle", {}),
                        "input_event_ids": list(self.trigger_ids),
                    }
                    job.checkpoint = checkpoint
                    self.session.commit()

            # 已读游标仅在 Agent 思考边界推进；到达不等于完成，事件仍留在队列中。
            self.trigger_ids[:] = list(dict.fromkeys([*self.trigger_ids, *seen]))
            persist_read_boundary()
            if restored_packet is not None:
                self.packet = restored_packet
                self.sync_discussion(state)
                return self.packet, self.trigger_ids
            clock = self.service.get_clock(self.project_id, self.branch_id)
            from ..input_delivery import deliver_wakeups

            deliver_wakeups(self.session, self.project_id, self.branch_id, clock)
            now = local_time(clock.now(), self.packet.timezone)
            self.session.expire_all()
            incoming = list(
                self.session.scalars(
                    select(RuntimeEventRow)
                    .where(
                        RuntimeEventRow.branch_id == self.branch_id,
                        RuntimeEventRow.status == "queued",
                        RuntimeEventRow.event_type != "planner_inbox",
                        RuntimeEventRow.occurred_at <= now.astimezone(UTC),
                    )
                    .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
                )
            )
            new_ids = {e.id for e in incoming} - set(self.trigger_ids)
            if not new_ids and self.packet.trigger.get("input_events") is not None:
                self.session.commit()
                return self.packet, list(self.trigger_ids)
            # 只刷新动态输入/当前时间，不换背景，也不清掉本轮已检索资料与协作草稿。
            fresh = ContextAssembler(self.session).assemble(
                branch_id=self.branch_id,
                trigger=self.packet.trigger,
                clock=clock,
                snapshot=self.bootstrap["snapshot"],
                now=now,
            )
            self.packet.branch = fresh.branch
            self.packet.virtual_now = fresh.virtual_now
            self.packet.generated_at = fresh.generated_at
            self.packet.current.update(
                {
                    key: value
                    for key, value in fresh.current.items()
                    if key not in {"collaboration", "expression_style_profile"}
                }
            )
            self.trigger_ids[:] = list(
                dict.fromkeys([*self.trigger_ids, *(e.id for e in incoming)])
            )
            persist_read_boundary()
            rows = list(
                self.session.scalars(
                    select(RuntimeEventRow)
                    .where(
                        RuntimeEventRow.id.in_(self.trigger_ids),
                        RuntimeEventRow.branch_id == self.branch_id,
                    )
                    .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
                )
            )
            # 保留触发原文、来源和时间；不能只把用户气泡放进窗口而漏掉 DayPlan 通知。
            self.packet.trigger = {
                **self.packet.trigger,
                "input_events": [
                    {
                        "id": row.id,
                        "type": row.event_type,
                        "occurred_at": (
                            row.occurred_at
                            if row.occurred_at.tzinfo
                            else row.occurred_at.replace(tzinfo=UTC)
                        ).isoformat(),
                        "payload": row.payload,
                    }
                    for row in rows
                ],
            }
            self.sync_discussion(state)
            # 对话窗口也绑定同一读取集合，不能偷看到 SELECT 边界以后到达的消息。
            # 已消费但仍待回复的旧消息必须保留；只隐藏未进入此次读取集合的新输入。
            unread_messages = {
                row.payload.get("branch_message_id")
                for row in self.session.scalars(
                    select(RuntimeEventRow).where(
                        RuntimeEventRow.branch_id == self.branch_id,
                        RuntimeEventRow.status.in_(["queued", "claimed"]),
                        RuntimeEventRow.id.not_in(self.trigger_ids),
                    )
                )
            }
            window = self.packet.branch["working_window"]
            window["messages"] = [
                item for item in window["messages"] if item["source_id"] not in unread_messages
            ]
            visible = {item["source_id"] for item in window["messages"]}
            window["pending_message_refs"] = [
                ref for ref in window["pending_message_refs"] if ref in visible
            ]
            self.session.commit()
            return self.packet, list(self.trigger_ids)

        from ..style_history import validate_asset
        from ..tools.send_agent_message import build_send_agent_message_tool

        run = self.service.director.run_with_trace(
            self.packet,
            receive_inputs=receive_inputs,
            asset_validator=lambda ref: validate_asset(
                self.session, self.branch, self.bootstrap["snapshot"], ref
            ),
            cancellation_requested=self.cancelled,
            search_tool=tool,
            context_tools=[
                *build_director_context_tools(self.session, self.packet),
                build_send_agent_message_tool(
                    self.session,
                    branch_id=self.branch_id,
                    sender="director",
                    task_id=self.trace.job_id or self.cycle_key,
                    cancellation_requested=self.cancelled,
                ),
            ],
            skill_tools=[
                StyleService(self.session).tool(
                    branch_id=self.branch_id,
                    model_version_id=self.branch.model_version_id,
                )
            ],
            recent_life_tool=build_recent_life_events_tool(
                self.session, self.branch_id, lambda: self.packet.virtual_now
            ),
            owner_id=self.trace.id,
            project_id=self.project_id,
            input_revision=self.input_revision,
            input_revision_resolver=self.service._runtime_input_revision_resolver(self.branch_id),
            reference_validator=self.validate_references,
        )
        if run.terminal_reason not in {"decision", "success"}:
            self.fail(run.terminal_reason, "Director 未产生可执行决策")
        decision = run.decision
        self.trace.outcome = {**self.trace.outcome, "director_terminal_reason": run.terminal_reason}
        self.session.commit()
        data = decision.model_dump(mode="json")
        accepted_inputs = {
            "input_event_ids": list(self.trigger_ids),
            "input_packet": self.packet.model_dump(mode="json"),
        }
        dispatch = None
        if decision.plan_request or decision.peer_reply:
            from .planner_tasks import make_dispatch

            dispatch = make_dispatch(self.session, self.branch_id, self.packet, decision)
        return {
            **accepted_inputs,
            "decision": data,
            "expected_state_version": self.packet.current["life_state"]["version"],
            "planner_dispatch": dispatch,
            "route": "executor",
            "expression_dependencies": self.expression_dependencies(),
            "actor_message": decision.reply.model_dump(mode="json") if decision.reply else None,
            "expression_assets": run.asset_sources or {},
            "messages": [
                *self.incoming_discussion,
                self.message(
                    "director",
                    "executor",
                    "request",
                    decision.communication_intent or decision.private_reason,
                    data,
                ),
            ],
        }

    def validate_references(self, value):
        error = self.executor.validate_decision_references(
            self.branch_id, value, self.packet.virtual_now
        )
        if error:
            return error
        if value.peer_reply:
            from .planner_tasks import make_dispatch

            try:
                make_dispatch(self.session, self.branch_id, self.packet, value)
            except ValueError as error:
                return str(error)
        return None

    def sync_discussion(self, state):
        """从本次已接收快照恢复共享发言，重放使用稳定 ID，不查询后来输入。"""
        known = {message.id for message in state.get("messages", [])}
        received = []
        for event in self.packet.trigger.get("input_events", []):
            for index, envelope in enumerate(event["payload"].get("discussion", [])):
                message = HumanMessage(
                    id=f"delivery:{event['id']}:{index}",
                    content=json.dumps(envelope, ensure_ascii=False),
                )
                if message.id not in known:
                    known.add(message.id)
                    received.append(message)
        self.incoming_discussion = received
        from .transport import peer_history

        self.packet.current["collaboration"] = peer_history(
            self.session,
            self.branch_id,
            recipient="director",
            current=discussion_payload([*state.get("messages", []), *received]),
        )

    def day_planner(self, state):
        # 普通维护和生活推进共享 Planner 执行权；planner_task 已在外层持有同一锁。
        from .persistence import branch_writer

        job = self.session.get(Job, self.trace.job_id) if self.trace.job_id else None
        if self.planning_only or (job and job.payload.get("planner_task")):
            return self._day_planner(state)
        with branch_writer(self.session, f"planner:{self.branch_id}") as acquired:
            if not acquired:
                self.fail("branch_busy", "DayPlan 正在处理另一项生活工作")
            return self._day_planner(state)

    def _day_planner(self, state):
        self.refresh(state)
        request = (
            PlanRevisionRequest.model_validate_json(json.dumps(state["request"]))
            if state.get("request")
            else None
        )
        target = (
            request.target_date
            if request and request.target_date
            else self.packet.virtual_now.date()
        )
        if target not in {
            self.packet.virtual_now.date(),
            self.packet.virtual_now.date() + timedelta(days=1),
        }:
            self.fail("invalid_plan_date", "不能改写过去或窗口外日期")
        plan = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == self.branch_id,
                RuntimeDayPlanRow.plan_date == target.isoformat(),
            )
        )
        mode = (
            "revision"
            if plan is not None and (plan.generation_metadata or {}).get("status") == "agent"
            else "initial"
        )
        base_version = plan.version if plan else 0
        job = self.session.get(Job, self.trace.job_id) if self.trace.job_id else None
        inbox_task = bool(job and job.payload.get("planner_task", {}).get("input_ids"))
        if not inbox_task:
            set_preparation(
                self.session,
                self.branch_id,
                target,
                "running",
                job_id=self.trace.job_id,
                owner_key=self.cycle_key,
            )
        self.session.commit()
        try:
            proposal, reason, work = self.service._run_day_planner(
                cancellation_requested=self.cancelled,
                branch_id=self.branch_id,
                snapshot=self.bootstrap["snapshot"],
                timezone=self.packet.timezone,
                virtual_now=self.packet.virtual_now,
                mode=mode,
                request=request,
                trace=self.trace,
                input_revision=self.input_revision,
                collaboration=discussion_payload(state.get("messages", [])),
                working_state=state.get("planner_work"),
            )
        except Exception as error:
            self.session.rollback()
            if not inbox_task:
                set_preparation(
                    self.session,
                    self.branch_id,
                    target,
                    "retryable_failed",
                    error_code=getattr(error, "code", "planner_interrupted"),
                )
            self.session.commit()
            raise
        if reason == "completed_without_change":
            from ..plan_status import preparation

            parent_id = job.payload.get("planner_task", {}).get("parent_job_id") if job else None
            if mode == "revision" and preparation(plan).get("owner_key") in {
                self.cycle_key,
                f"planner:{parent_id}",
            }:
                # 原调查已决定无需修改：释放它自己的等待状态，不覆盖别的任务准备状态。
                set_preparation(self.session, self.branch_id, target, "ready")
                self.session.commit()
            return {
                "planner_work": work,
                "proposal": None,
                "request": None,
                "route": "director",
                "messages": [],
            }
        if isinstance(proposal, PeerReply):
            if proposal.kind == "answer" and mode == "revision" and not self.planning_only:
                set_preparation(self.session, self.branch_id, target, "ready")
                self.session.commit()
            if self.planning_only:
                set_preparation(
                    self.session,
                    self.branch_id,
                    target,
                    "blocked",
                    error_code="planning_requires_input",
                )
                self.session.commit()
                self.fail("planning_requires_input", proposal.content)
            history = discussion_payload(state.get("messages", []))
            if any(
                item["sender"] == "day_planner"
                and item["kind"] == proposal.kind
                and item["content"] == proposal.content
                and item["task_id"] == self.task_id
                for item in history
            ):
                self.fail("collaboration_stalled", "同一问题被重复提出，已有问答已保存在共享记录")
            return {
                "request": None if proposal.kind == "answer" else state.get("request"),
                "planner_work": work,
                "proposal": None,
                "route": "director",
                "messages": [
                    self.message("day_planner", "director", proposal.kind, proposal.content)
                ],
            }
        if proposal is None:
            if not inbox_task:
                set_preparation(
                    self.session, self.branch_id, target, "retryable_failed", error_code=str(reason)
                )
            self.session.commit()
            # 失败 Job 从同一请求/检查点恢复，不能伪装成功并丢掉用户的修订意图。
            self.fail(str(reason or "planner_unavailable"), "DayPlan 未生成可提交计划")
        return {
            "planner_work": work,
            "proposal": proposal.model_dump(mode="json"),
            "plan_mode": mode,
            "date_features": self.trace.outcome.get(f"planner_{mode}_date_features", {}),
            "base_plan_version": base_version,
            "route": "executor",
            "messages": [
                self.message(
                    "day_planner",
                    "executor",
                    "submit",
                    proposal.private_reason or "提交目标日期计划",
                    {"mode": mode},
                )
            ],
        }

    def expression_dependencies(self):
        """只比较有语义的时间边界，不因墙钟前进一秒就重跑决策。"""
        now = local_time(self.bootstrap["clock"].now(), self.packet.timezone)
        plans = self.packet.current.get("day_plans", {})
        blocks = plans.get(now.date().isoformat(), {}).get("blocks", [])
        active = [
            b for b in blocks if b.get("start", "") <= now.strftime("%H:%M") < b.get("end", "")
        ]
        return hashlib.sha256(
            json.dumps(
                {
                    "state": self.packet.current["life_state"]["version"],
                    "plans": {day: plan.get("plan_version") for day, plan in plans.items()},
                    "date": now.date().isoformat(),
                    "active": active,
                    "snapshot": self.bootstrap["snapshot"].profile,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    def executor_node(self, state):
        # 外层图恢复到提交节点时，消费集合来自被接受的决定，而不是重查当前队列。
        self.trigger_ids[:] = state.get("input_event_ids", self.trigger_ids)
        self.refresh(state)
        if not self.current():
            self.fail("stale_input_revision", "分支执行条件已变更或任务租约已失效")
        locked = self.session.execute(
            update(Branch)
            .where(
                Branch.id == self.branch_id, Branch.runtime_input_revision == self.input_revision
            )
            .values(runtime_input_revision=self.input_revision)
        )
        if locked.rowcount != 1:
            self.session.rollback()
            self.fail("stale_input_revision", "提交事务中分支执行代际已改变")
        if self.trace.job_id:
            fence = self.session.scalar(
                select(Job.id).where(
                    Job.id == self.trace.job_id,
                    Job.worker_token == self.worker_token,
                    Job.status == "running",
                )
            )
            if fence is None:
                self.session.rollback()
                self.fail("worker_lease_lost", "提交事务中 Worker 租约已失效")
        if state.get("proposal") is not None:
            proposal = DayPlanProposal.model_validate_json(json.dumps(state["proposal"]))
            mode = state["plan_mode"]
            # 正式提交时间重新读取，不能拿调查开始时间绕过已开始块锁定。
            fresh_clock = self.service.get_clock(self.project_id, self.branch_id)
            now = local_time(fresh_clock.now(), fresh_clock.timezone)
            receipt_key = hashlib.sha256(
                (self.cycle_key + json.dumps(state["proposal"], sort_keys=True)).encode()
            ).hexdigest()
            try:
                plan = self.executor.commit_day_plan_proposal(
                    branch_id=self.branch_id,
                    proposal=proposal,
                    virtual_now=now,
                    mode=mode,
                    expected_version=state["base_plan_version"],
                    idempotency_key=receipt_key,
                    date_features=state.get("date_features"),
                )
                self.trace.planner_proposal = state["proposal"]
                if self.planning_only:
                    from .transport import enqueue_peer_message

                    enqueue_peer_message(
                        self.session,
                        branch=self.branch,
                        sender="executor",
                        recipient="director",
                        content=(
                            f"{plan.plan_date} 的计划已提交，版本 {plan.version}。"
                            "请按需考虑安排变化，不必告知用户。"
                        ),
                        task_id=self.cycle_key,
                        occurred_at=fresh_clock.now(),
                        key=f"plan-notice:{receipt_key}",
                        payload={
                            "status": "committed",
                            "plan_date": plan.plan_date,
                            "plan_version": plan.version,
                            "key": receipt_key,
                        },
                    )
                self.session.commit()  # 独立日程不会随之后的表达失败一起回滚。
            except ValueError as error:
                self.session.rollback()
                expired = proposal.plan_date < now.date()
                set_preparation(
                    self.session,
                    self.branch_id,
                    proposal.plan_date,
                    "blocked" if expired else "running",
                    error_code="plan_commit_rejected",
                    owner_key=self.cycle_key,
                )
                self.session.commit()
                if expired:
                    self.fail("plan_commit_rejected", str(error))
                return {
                    "proposal": None,
                    "route": "day_planner",
                    "messages": [
                        self.message(
                            "executor",
                            "day_planner",
                            "commit_result",
                            str(error),
                            {"status": "rejected"},
                        ),
                    ],
                }
            receipt = {
                "status": "committed",
                "plan_date": plan.plan_date,
                "plan_version": plan.version,
                "key": receipt_key,
            }
            receipt = (
                (plan.generation_metadata or {})
                .get("commit_receipts", {})
                .get(receipt_key, receipt)
            )
            self.trace.outcome = {**self.trace.outcome, f"planner_{mode}": "committed"}
            self.session.commit()
            if proposal.plan_date == now.date():
                self.service._apply_plan_boundary(self.branch_id, now, plan.blocks)
                self.session.commit()
            life = self.session.scalar(
                select(RuntimeLifeStateRow).where(
                    RuntimeLifeStateRow.branch_id == self.branch_id,
                    RuntimeLifeStateRow.is_current.is_(True),
                )
            )
            window = shared_plan_window(self.session, self.branch_id, now, self.packet.timezone)
            return {
                "proposal": None,
                "request": None,
                "receipt": receipt,
                "expected_state_version": life.version,
                "day_plans": window,
                "route": "executor" if self.planning_only else "director",
                "decision": LifeDecision(action="wait", private_reason="日程维护已完成").model_dump(
                    mode="json"
                )
                if self.planning_only
                else None,
                "messages": [
                    self.message(
                        "executor", "day_planner", "commit_result", "计划提交成功", receipt
                    ),
                    self.message(
                        "day_planner",
                        "director",
                        "notification",
                        "计划已生效，请以共享新版本判断",
                        receipt,
                    ),
                ],
            }
        decision = LifeDecision.model_validate_json(json.dumps(state["decision"]))
        from ..expression_contracts import ExpressionResult

        actor = (
            ExpressionResult.model_validate(state["actor_message"])
            if state.get("actor_message")
            else None
        )
        if (
            actor is not None
            and state.get("expression_dependencies") != self.expression_dependencies()
        ):
            self.session.rollback()
            return {"route": "director", "actor_message": None}
        if actor is not None:
            from ..style_history import validate_asset

            try:
                for part in actor.messages:
                    if part.kind == "sticker":
                        validate_asset(
                            self.session, self.branch, self.bootstrap["snapshot"], part.asset_ref
                        )
            except ValueError as error:
                self.session.rollback()
                return {
                    "route": "director",
                    "actor_message": None,
                    "expression_error": str(error),
                }
        self.result = self.executor.commit(
            project_id=self.project_id,
            branch_id=self.branch_id,
            decision=decision.model_copy(update={"plan_request": None, "peer_reply": None}),
            expected_version=state.get(
                "expected_state_version", self.packet.current["life_state"]["version"]
            ),
            virtual_now=self.packet.virtual_now,
            trigger_event_ids=self.trigger_ids,
            expression_result=actor,
            asset_sources=state.get("expression_assets", {}),
            idempotency_key=self.cycle_key,
        )
        if state.get("planner_dispatch"):
            from .planner_tasks import enqueue_dispatch

            enqueue_dispatch(
                self.session,
                self.branch_id,
                self.cycle_key,
                state["planner_dispatch"],
                state.get("messages", []),
            )
        # 与业务回复同事务确认实际读到的事件；最后一次思考后到达的输入不在此集合中。
        from ..input_delivery import consume_inputs

        consume_inputs(self.session, self.branch_id, self.trigger_ids)
        self.trace.trigger_event_ids = list(self.trigger_ids)
        self.session.commit()
        return {
            "route": "end",
            "status": "waiting" if state.get("request") else "completed",
            "result": {"event_id": self.result["event"].id},
            "messages": [self.message("executor", "broadcast", "commit_result", "本轮行为已提交")],
        }

    def run(self):
        with (
            chain_span(
                "runtime.collaboration.cycle",
                attributes={
                    "moonlightbox.branch.id": self.branch_id,
                    "moonlightbox.cycle.key": self.cycle_key,
                    "moonlightbox.input_revision": self.input_revision,
                },
            ) as span,
            collaboration_checkpointer(self.session) as saver,
        ):
            snapshot = self.bootstrap["snapshot"]
            record_span_attributes(
                span,
                {
                    "moonlightbox.background.profile_id": snapshot.profile_id or "",
                    "moonlightbox.background.graph_version_id": snapshot.graph_version_id or "",
                    "moonlightbox.background.profile_schema_version": snapshot.profile.get(
                        "profile_schema_version", "legacy_unversioned"
                    ),
                },
            )
            graph = build_collaboration_graph(
                director=self.director,
                day_planner=self.day_planner,
                executor=self.executor_node,
                checkpointer=saver,
                input_is_current=self.current,
            )
            thread_id = (
                f"runtime-maintenance:{self.branch_id}:{self.cycle_key}"
                if self.planning_only
                else f"runtime-peer-v1:{self.branch_id}"
            )
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 160,
            }
            saved = graph.get_state(config)
            pending_request = None  # 等待对方的工作在独立 Planner Job 内，不占据聊天图。
            if pending_request and not self.planning_only:
                self.task_id = saved.values.get("task_id", self.cycle_key)
            initial = {
                "schema_version": "peer-v4-director-skill",
                "branch_id": self.branch_id,
                "cycle_id": self.cycle_key,
                "task_id": self.task_id,
                "input_revision": self.input_revision,
                "route": "day_planner" if self.planning_only else "director",
                "status": "running",
                "handoffs": 0,
                "started_at": time(),
                "seen_requests": [],
                "usage": {},
                "result": {},
                "request": (
                    {
                        "target_date": self.packet.trigger["payload"]["target_date"],
                        "reason": "维护下一天计划",
                    }
                    if self.planning_only
                    and self.packet.trigger.get("payload", {}).get("target_date")
                    else pending_request
                ),
                "proposal": None,
                "receipt": None,
                "decision": None,
                "actor_message": None,
                "expression_assets": {},
                "expression_error": None,
                "expected_state_version": self.packet.current["life_state"]["version"],
                "messages": [
                    self.message(
                        "system",
                        "day_planner" if self.planning_only else "director",
                        "event",
                        "日程维护" if self.planning_only else "处理分支输入",
                        self.packet.trigger,
                    )
                ],
                "day_plans": self.packet.current["day_plans"],
                "planner_work": saved.values.get("planner_work") if pending_request else None,
            }
            same_cycle = (
                saved.values
                and saved.values.get("schema_version") == "peer-v4-director-skill"
                and saved.values.get("cycle_id") == self.cycle_key
                and saved.values.get("input_revision") == self.input_revision
            )
            if same_cycle and saved.values.get("status") in {"completed", "waiting"}:
                state = saved.values
            else:
                if same_cycle:
                    resumed = resume_window()
                    if saved.next:
                        graph.update_state(config, resumed)
                    else:
                        initial = {**saved.values, **resumed, "status": "running"}
                state = graph.invoke(
                    None if same_cycle and saved.next else initial, config, durability="sync"
                )
            if self.result is None and state.get("result"):
                event = self.session.get(RuntimeLifeEventRow, state["result"]["event_id"])
                life = self.session.scalar(
                    select(RuntimeLifeStateRow).where(
                        RuntimeLifeStateRow.branch_id == self.branch_id,
                        RuntimeLifeStateRow.is_current.is_(True),
                    )
                )
                message = self.session.scalar(
                    select(BranchMessage).where(
                        BranchMessage.branch_id == self.branch_id, BranchMessage.turn_id == event.id
                    )
                )
                self.result = {"event": event, "state": life, "message": message}
        return state

    @staticmethod
    def fail(code, message):
        from ..service import RuntimeModelExecutionError

        raise RuntimeModelExecutionError(code, message)
