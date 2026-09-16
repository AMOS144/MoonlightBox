"""Runtime v1 应用服务：把队列、上下文、LangGraph 和 Executor 串成一轮 Cycle。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime import AgentLoopController
from moonlightbox.agent_runtime.contracts import ChatModel
from moonlightbox.observability.phoenix import record_span_output
from moonlightbox.observability.runtime import runtime_cycle_span

from .branch_models import Branch, BranchMessage
from .clock import pause_clock, resume_clock
from .context import ContextAssembler
from .day_planner import DayPlanAgent
from .db_models import (
    RuntimeClockRow,
    RuntimeCycleTraceRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeStateRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from .director import DirectorAgent
from .executor import RuntimeExecutor
from .plan_context import PlanContextAssembler
from .schemas import ContextPacket, LifeDecision, PlanRevisionRequest, VirtualClock
from .tools.plan_constraints import build_plan_constraints_tool
from .tools.plan_memory import build_plan_memory_tool
from .tools.routine_evidence import build_routine_evidence_tool
from .work_calendar import UnverifiedWorkCalendar, WorkCalendar

_CLAIM_LEASE = timedelta(minutes=5)


class RuntimeModelExecutionError(RuntimeError):
    """模型不可用时中止本轮，令事件保持可重试而不是静默完成。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class RuntimeInputSupersededError(RuntimeError):
    """分支控制条件已变更，旧 Cycle 不得继续提交；普通新消息不属于此类。"""

    code = "stale_input_revision"


class RuntimeService:
    """面向 HTTP/Worker 的门面；旧 branches 服务不会直接调用模型。"""

    def __init__(
        self,
        session: Session,
        *,
        director_model: ChatModel | None = None,
        actor_model: ChatModel | None = None,
        planner_model: ChatModel | None = None,
        work_calendar: WorkCalendar | None = None,
    ) -> None:
        self.session = session
        # Director 和 DayPlan 共享同一套无状态 Harness；每次调用都是
        # 独立角色执行；协作图注入恢复检查点，完整调用链由 Phoenix 记录。
        self.agent_loop_controller = AgentLoopController()
        self.director = DirectorAgent(director_model, controller=self.agent_loop_controller)
        # Planner 与 Director 使用同一认知模型配置；二者靠 Prompt、工具集合和输出
        # schema 隔离职责，而不是悄悄降级到 LoRA 或另一套小模型。
        self.planner = DayPlanAgent(
            planner_model or director_model,
            controller=self.agent_loop_controller,
        )
        self.work_calendar = work_calendar or UnverifiedWorkCalendar()

    def bootstrap(self, project_id: str, branch_id: str) -> dict[str, Any]:
        result = RuntimeExecutor(self.session).bootstrap(project_id=project_id, branch_id=branch_id)
        self.session.commit()
        return result

    def refresh_published_background(self, project_id: str, branch_id: str) -> dict:
        """显式升级 latest_profile 体验绑定，历史内容留档，旧执行因输入版本变化失效。"""
        from .background_binding import refresh_published_background
        from .collaboration.persistence import branch_writer

        with branch_writer(self.session, branch_id) as acquired:
            if not acquired:
                raise RuntimeModelExecutionError(
                    "branch_writer_busy", "分支正在执行，请完成后更新背景"
                )
            try:
                binding = refresh_published_background(self.session, project_id, branch_id)
                self.session.commit()
                return binding
            except Exception:
                self.session.rollback()
                raise

    def create_branch(
        self,
        *,
        project_id: str,
        investigation_id: str,
        preview_hash: str,
        title: str,
        publication_id: str | None = None,
    ) -> Branch:
        """只从获批节点版本创建冻结分支，任务准备完成以前不允许聊天。"""
        from moonlightbox.jobs.service import JobService
        from moonlightbox.world.models import (
            PersonWorldProfile,
            WorldGraphVersion,
            WorldPublication,
        )
        from moonlightbox.world.person_world.node_review import runtime_node_projection
        from moonlightbox.world.person_world.node_scope import NodeCompilationScope
        from moonlightbox.world.person_world.publication import active_publication

        from .executor import _profile_payload, _runtime_routine_profile

        publication = (
            self.session.get(WorldPublication, publication_id)
            if publication_id
            else active_publication(self.session, project_id, preview_hash)
        )
        if (
            publication is None
            or publication.project_id != project_id
            or publication.node_boundary_hash != preview_hash
            or publication.status != "active"
        ):
            raise ValueError("该起点尚未审核发布，请先完成节点画像审核")
        profile = self.session.get(PersonWorldProfile, publication.profile_id)
        graph = self.session.get(WorldGraphVersion, publication.graph_version_id)
        if (
            profile is None
            or graph is None
            or profile.project_id != project_id
            or graph.project_id != project_id
            or profile.graph_version_id != graph.id
            or profile.node_boundary_hash != preview_hash
        ):
            raise ValueError("发布版本的图谱、画像与起点不一致")
        scope = NodeCompilationScope.restore(
            self.session, graph=graph, envelope=profile.generation_summary.get("node_scope")
        )
        if scope.boundary["investigation_id"] != investigation_id:
            raise ValueError("发布版本不属于指定起点调查")
        if not title.strip():
            raise ValueError("分支名称不能为空")
        origin = datetime.fromisoformat(scope.boundary["cutoff_at"]).astimezone(UTC)
        branch = Branch(
            project_id=project_id,
            title=title.strip(),
            origin_time=origin,
            origin_boundary={**scope.boundary, "publication_id": publication.id},
            lifecycle_status="preparing",
        )
        self.session.add(branch)
        self.session.flush()
        binding = {
            "publication_id": publication.id,
            "profile_id": profile.id,
            "graph_version_id": graph.id,
            "profile_schema_version": "v3",
            "temporal_scope": {
                "source_version": scope.source_version,
                "included_count": scope.boundary["included_count"],
            },
        }
        self.session.add(
            RuntimeSnapshotRow(
                branch_id=branch.id,
                graph_version_id=graph.id,
                profile_id=profile.id,
                cutoff_at=origin,
                timezone=scope.boundary["timezone"],
                snapshot_mode="historical_cutoff",
                source_message_ids=list(scope.message_ids[: scope.boundary["included_count"]]),
                profile={
                    **runtime_node_projection(_profile_payload(profile)),
                    "_runtime_binding": binding,
                },
                routine_profile=runtime_node_projection(_runtime_routine_profile(profile)),
                compiler_version=profile.compiler_version,
            )
        )
        self.session.add(
            RuntimeClockRow(
                branch_id=branch.id,
                virtual_anchor=origin,
                wall_anchor=datetime.now(UTC),
                timezone=scope.boundary["timezone"],
                status="paused",
                time_scale=1.0,
            )
        )
        JobService(self.session).enqueue_unique(
            "runtime-v1-cycle",
            {"project_id": project_id, "branch_id": branch.id, "bootstrap": True},
            dedupe_key=f"runtime-v1-cycle:bootstrap:{branch.id}",
            commit=False,
        )
        self.session.commit()
        return branch

    def list_branches(self, project_id: str) -> list[Branch]:
        return list(
            self.session.scalars(
                select(Branch)
                .where(Branch.project_id == project_id)
                .order_by(Branch.created_at.desc())
            )
        )

    def list_messages(self, project_id: str, branch_id: str) -> list[BranchMessage]:
        self._branch(project_id, branch_id)
        return list(
            self.session.scalars(
                select(BranchMessage)
                .where(BranchMessage.branch_id == branch_id)
                .order_by(BranchMessage.sequence, BranchMessage.created_at)
            )
        )

    def list_cycle_traces(
        self, project_id: str, branch_id: str, *, limit: int = 20
    ) -> list[RuntimeCycleTraceRow]:
        """读取本地诊断账本；它不作为模型上下文或人物记忆的输入。"""

        self._branch(project_id, branch_id)
        return list(
            self.session.scalars(
                select(RuntimeCycleTraceRow)
                .where(RuntimeCycleTraceRow.branch_id == branch_id)
                .order_by(RuntimeCycleTraceRow.started_at.desc(), RuntimeCycleTraceRow.id.desc())
                .limit(max(1, min(limit, 100)))
            )
        )

    def submit_user_message(
        self,
        *,
        project_id: str,
        branch_id: str,
        content: str,
        idempotency_key: str,
        client_message_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        """先把用户原文写入 BranchMessage，再排入 RuntimeEvent。"""
        branch = self._branch(project_id, branch_id)
        # 入队前冻结背景并初始化计划/状态；如果人物世界尚未完成，直接返回明确错误，
        if branch.lifecycle_status in {"preparing", "prepare_failed"}:
            raise ValueError("分支日程尚未准备完成，请先完成或重试准备")
        # 不留下一个将由 Worker 无限重试的空背景事件。
        bootstrap = RuntimeExecutor(self.session).bootstrap(
            project_id=project_id, branch_id=branch_id
        )
        existing_event = self.session.scalar(
            select(RuntimeEventRow).where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.idempotency_key == idempotency_key,
            )
        )
        if existing_event is not None:
            return {
                "event": existing_event,
                "message": self.session.get(
                    BranchMessage, existing_event.payload.get("branch_message_id")
                ),
            }
        # BranchMessage 的 observed_at 属于分支时间轴。若此处误写服务器墙上时间，
        # 从旧节点创建的分支会永远在 working_window 之外看不到刚收到的用户消息。
        virtual_occurred_at = _as_utc(occurred_at or bootstrap["clock"].now()).astimezone(UTC)
        sequence = self.session.scalar(
            select(func.max(BranchMessage.sequence)).where(BranchMessage.branch_id == branch_id)
        )
        message = BranchMessage(
            branch_id=branch_id,
            sequence=int(sequence) + 1 if sequence is not None else 0,
            role="user",
            content=content,
            type="text",
            turn_id=str(uuid4()),
            bubble_index=0,
            generation_status="completed",
            generation_metadata={"runtime_v1": True, "input_status": "pending"},
            client_message_id=client_message_id or idempotency_key[:64],
            observed_at=virtual_occurred_at,
        )
        self.session.add(message)
        self.session.flush()
        # 普通消息只追加输入，不改变执行代际，不作废正在思考或准备提交的决定。
        # 新消息有自己的事件与 pending 状态，不能被本轮旧决定顺手确认。
        from .conversation_maintenance import enqueue_maintenance

        event = RuntimeExecutor(self.session).append_user_event(
            project_id=project_id,
            branch_id=branch_id,
            content=content,
            idempotency_key=idempotency_key,
            occurred_at=virtual_occurred_at,
        )
        event.payload = {"content": content, "branch_message_id": message.id}
        # 这里只存消息和事件；统一节拍负责空闲分支投递，不能每条消息启动一个 Director。
        enqueue_maintenance(self.session, branch_id, message.id)
        self.session.commit()
        return {"event": event, "message": message}

    def _finish_branch_preparation(self, project_id, branch_id, bootstrap, job_id):
        from moonlightbox.agent_runtime.cancellation import job_signal
        from moonlightbox.jobs.models import Job

        from .db_models import RuntimeInitializationRow

        job = self.session.get(Job, job_id) if job_id else None
        signal = job_signal(self.session.get_bind(), job.id, job.worker_token) if job else None
        if self.session.get(RuntimeInitializationRow, branch_id) is not None:
            self.director.initialize_branch(
                self.session,
                project_id=project_id,
                branch_id=branch_id,
                bootstrap=bootstrap,
                cancellation_requested=signal,
            )
        if signal and signal():
            raise RuntimeModelExecutionError("cancelled", "准备任务已取消，暂不开放聊天")
        branch = self._branch(project_id, branch_id)
        branch.lifecycle_status = "active"
        row = self.session.get(RuntimeClockRow, branch_id)
        row.wall_anchor, row.status = datetime.now(UTC), "running"
        self.session.commit()

    def process_next(
        self,
        *,
        project_id: str,
        branch_id: str,
        job_id: str | None = None,
        prepare_branch: bool = False,
        plan_date: str | None = None,
    ) -> dict[str, Any] | None:
        """同一分支一个 Director；规划调查使用独立执行权，提交依靠事务 CAS。"""
        from .collaboration.persistence import branch_writer

        role = "planner" if prepare_branch or plan_date else "director"
        with branch_writer(self.session, f"{role}:{branch_id}") as acquired:
            if not acquired:
                raise RuntimeModelExecutionError(
                    "branch_writer_busy", "该分支已有运行中的协作，稍后重试"
                )
            try:
                return self._process_next(
                    project_id=project_id,
                    branch_id=branch_id,
                    job_id=job_id,
                    prepare_branch=prepare_branch,
                    plan_date=plan_date,
                )
            except Exception:
                # 背景装载失败可能发生在 Cycle Trace 创建之前，也必须结束准备态。
                self.session.rollback()
                if prepare_branch:
                    branch = self._branch(project_id, branch_id)
                    branch.lifecycle_status = "prepare_failed"
                    self.session.commit()
                raise

    def _process_next(
        self,
        *,
        project_id: str,
        branch_id: str,
        job_id: str | None = None,
        prepare_branch: bool = False,
        plan_date: str | None = None,
    ) -> dict[str, Any] | None:
        """领取该分支最早事件并运行一次；模型调用期间不持有数据库写锁。"""
        self.recover_stale(project_id, branch_id)
        branch = self._branch(project_id, branch_id)
        from moonlightbox.jobs.models import Job

        from .db_models import RuntimeLifeEventRow

        owner_job = self.session.get(Job, job_id) if job_id else None
        resume_cycle = (owner_job.checkpoint or {}).get("runtime_cycle") if owner_job else None
        if resume_cycle and not prepare_branch:
            committed = self.session.scalar(
                select(RuntimeLifeEventRow).where(
                    RuntimeLifeEventRow.branch_id == branch_id,
                    RuntimeLifeEventRow.idempotency_key == resume_cycle["key"],
                )
            )
            if committed:
                return {"replayed": True, "event": committed}
        from .db_models import RuntimeInitializationRow

        if (
            prepare_branch
            and branch.lifecycle_status == "preparing"
            and self.session.get(RuntimeInitializationRow, branch_id) is None
        ):
            self.session.add(
                RuntimeInitializationRow(
                    branch_id=branch_id, input_hash="", status="pending", work={}
                )
            )
            self.session.commit()
        bootstrap = RuntimeExecutor(self.session).bootstrap(
            project_id=project_id, branch_id=branch_id
        )
        # 一个 Cycle 内所有组件必须使用同一 virtual_now，不能每一步重新取墙上时间。
        if prepare_branch:
            row = self.session.get(RuntimeClockRow, branch_id)
            paused = pause_clock(bootstrap["clock"])
            row.virtual_anchor, row.wall_anchor, row.status = (
                paused.virtual_anchor,
                paused.wall_anchor,
                "paused",
            )
            bootstrap["clock"] = paused
            self.session.commit()
        from .collaboration.plans import local_time

        virtual_now = local_time(bootstrap["clock"].now(), bootstrap["clock"].timezone)
        if bootstrap["clock"].status != "running" and not (prepare_branch or plan_date):
            return None
        from .input_delivery import deliver_wakeups

        if not (prepare_branch or plan_date):
            deliver_wakeups(self.session, project_id, branch_id, bootstrap["clock"])
        planner_pending = RuntimeExecutor(self.session).day_plan_needs_generation(bootstrap["plan"])
        if (
            prepare_branch
            and self.session.get(RuntimeInitializationRow, branch_id) is not None
            and (bootstrap["plan"].generation_metadata or {}).get("status") == "agent"
        ):
            self._finish_branch_preparation(project_id, branch_id, bootstrap, job_id)
            return {"initialized": True}  # 初始化重试复用已完成计划。
        queued_events = list(
            self.session.scalars(
                select(RuntimeEventRow)
                .where(
                    RuntimeEventRow.branch_id == branch_id,
                    RuntimeEventRow.status == "queued",
                    RuntimeEventRow.event_type != "planner_inbox",
                    RuntimeEventRow.occurred_at <= virtual_now.astimezone(UTC),
                )
                .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
            )
        )
        events = [] if prepare_branch or plan_date else queued_events
        if resume_cycle:
            # 稳定任务身份不随后来输入改变；下一次思考再把新增队列合并进来。
            original_ids = resume_cycle["trigger_ids"]
            self.session.execute(
                update(RuntimeEventRow)
                .where(
                    RuntimeEventRow.branch_id == branch_id,
                    RuntimeEventRow.id.in_(original_ids),
                    RuntimeEventRow.status == "claimed",
                )
                .values(status="queued", claimed_at=None)
            )
            events = list(
                self.session.scalars(
                    select(RuntimeEventRow)
                    .where(
                        RuntimeEventRow.branch_id == branch_id,
                        RuntimeEventRow.id.in_(original_ids),
                        RuntimeEventRow.status == "queued",
                    )
                    .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
                )
            )
        wakeups = list(
            self.session.scalars(
                select(RuntimeWakeupRow)
                .where(
                    RuntimeWakeupRow.branch_id == branch_id,
                    RuntimeWakeupRow.status == "scheduled",
                    RuntimeWakeupRow.wake_at <= virtual_now.astimezone(UTC),
                )
                .order_by(RuntimeWakeupRow.wake_at, RuntimeWakeupRow.id)
            )
        )
        if prepare_branch or plan_date:
            wakeups = []
        if not events and not wakeups and not prepare_branch and not plan_date:
            self.session.commit()
            return None
        # Compare-and-set 领取：先看到 queued 不等于拥有它，不能让两个 Worker 同时进模型。
        claimed_events: list[RuntimeEventRow] = []
        for item in events:
            claimed = cast(
                Any,
                self.session.execute(
                    update(RuntimeEventRow)
                    .where(RuntimeEventRow.id == item.id, RuntimeEventRow.status == "queued")
                    .values(status="claimed", claimed_at=datetime.now(UTC))
                ),
            )
            if claimed.rowcount == 1:
                claimed_events.append(item)
        claimed_wakeups: list[RuntimeWakeupRow] = []
        for wakeup in wakeups:
            claimed = cast(
                Any,
                self.session.execute(
                    update(RuntimeWakeupRow)
                    .where(RuntimeWakeupRow.id == wakeup.id, RuntimeWakeupRow.status == "scheduled")
                    .values(status="executing", executing_at=datetime.now(UTC))
                ),
            )
            if claimed.rowcount == 1:
                claimed_wakeups.append(wakeup)
        if not claimed_events and not claimed_wakeups and not prepare_branch and not plan_date:
            self.session.rollback()
            return None
        events = claimed_events
        wakeups = claimed_wakeups
        planning_only = (
            prepare_branch or bool(plan_date) or (not events and not wakeups and planner_pending)
        )
        if not events and not wakeups:
            # 分支创建后的首个 Job 不一定伴随聊天或 Wakeup。仍要以一条可审计的
            # 系统触发运行 Planner，但不能伪造 RuntimeEvent 行。
            trigger_ids = [f"plan:{branch_id}:{plan_date or virtual_now.date().isoformat()}"]
            trigger_payload = {
                "type": "system",
                "id": trigger_ids[0],
                "occurred_at": virtual_now.isoformat(),
                "payload": {"reason": "day_plan_generation", "target_date": plan_date},
                "additional_triggers": [],
            }
        else:
            trigger_ids, trigger_payload = _merge_triggers(events, wakeups)
        cycle_key = f"cycle:{branch_id}:{','.join(sorted(trigger_ids))}"
        if resume_cycle:
            cycle_key = resume_cycle["key"]
            trigger_ids = list(resume_cycle["trigger_ids"])
            trigger_payload = resume_cycle["trigger"]
        elif owner_job:
            owner_job.checkpoint = {
                **(owner_job.checkpoint or {}),
                "runtime_cycle": {
                    "key": cycle_key,
                    "trigger_ids": list(trigger_ids),
                    "trigger": trigger_payload,
                },
            }
        # 领取状态先提交，避免一次远端模型推理长期占住 SQLite 写事务。乐观锁会
        # 防止其后到达的实时 Cycle 被旧结果覆盖。
        self.session.commit()
        trace = RuntimeCycleTraceRow(
            project_id=project_id,
            branch_id=branch_id,
            job_id=job_id,
            cycle_key=cycle_key,
            trigger_event_ids=trigger_ids,
            wakeup_ids=[item.id for item in wakeups],
            virtual_now=virtual_now,
            started_at=datetime.now(UTC),
        )
        self.session.add(trace)
        self.session.commit()
        trace_id = trace.id
        claimed_event_ids = [item.id for item in claimed_events]
        claimed_wakeup_ids = [item.id for item in claimed_wakeups]
        started = monotonic()
        with runtime_cycle_span(
            cycle_id=trace_id,
            cycle_key=cycle_key,
            project_id=project_id,
            branch_id=branch_id,
            input_value={
                "trigger_event_ids": trigger_ids,
                "wakeup_ids": claimed_wakeup_ids,
                "virtual_now": virtual_now,
                "job_id": job_id,
            },
        ) as cycle_span:
            try:
                cycle_input_revision = self._runtime_input_revision(branch_id)
                if cycle_input_revision is None:
                    raise RuntimeInputSupersededError("分支已不存在，不能提交 Runtime 结果")
                bootstrap = RuntimeExecutor(self.session).bootstrap(
                    project_id=project_id, branch_id=branch_id
                )
                clock = bootstrap["clock"]
                # 初始计划由 Agent 提交后，才可以用其真实块推进 LifeState；绝不能让
                # Director 读取 bootstrap 阶段臆造的“睡眠/工作”默认骨架。
                self._apply_plan_boundary(branch_id, virtual_now, bootstrap["plan"].blocks)
                self._expire_state(branch_id, virtual_now, bootstrap["plan"].blocks)
                bootstrap["state"] = (
                    self.session.scalar(
                        select(RuntimeLifeStateRow).where(
                            RuntimeLifeStateRow.branch_id == branch_id,
                            RuntimeLifeStateRow.is_current.is_(True),
                        )
                    )
                    or bootstrap["state"]
                )
                self.session.flush()
                # 确定性的时间边界推进单独提交；等待 Agent 时不持有写事务。
                self.session.commit()
                packet = ContextAssembler(self.session).assemble(
                    branch_id=branch_id,
                    trigger=trigger_payload,
                    clock=clock,
                    snapshot=bootstrap["snapshot"],
                    now=virtual_now,
                )
                trace.packet = packet.model_dump(mode="json")
                from .collaboration.nodes import RuntimePeerNodes

                self.session.commit()
                nodes = RuntimePeerNodes(
                    self,
                    project_id=project_id,
                    branch=branch,
                    bootstrap=bootstrap,
                    packet=packet,
                    trace=trace,
                    input_revision=cycle_input_revision,
                    cycle_key=cycle_key,
                    trigger_ids=trigger_ids,
                    planning_only=planning_only,
                )
                collaboration = nodes.run()
                packet = nodes.packet
                decision = LifeDecision.model_validate_json(json.dumps(collaboration["decision"]))
                trace.director_decision = decision.model_dump(mode="json")
                trace.committed_decision = decision.model_dump(mode="json")
                result = nodes.result
                if result is None:
                    raise RuntimeModelExecutionError(
                        "missing_commit_result", "协作结束但没有提交回执"
                    )
                if prepare_branch:
                    prepared_plan = self.get_plan(project_id, branch_id)
                    if (prepared_plan.generation_metadata or {}).get("status") != "agent":
                        raise RuntimeModelExecutionError(
                            "preparation_incomplete", "还没有已提交的当天计划"
                        )
                    self._finish_branch_preparation(project_id, branch_id, bootstrap, job_id)
                # 消费确认已在 Executor 与业务回执同事务完成，不在这里批量扫尾。
                trace.status = "succeeded"
                trace.stage = "committed"
                trace.completed_at = datetime.now(UTC)
                trace.outcome = {
                    **trace.outcome,
                    "elapsed_ms": int((monotonic() - started) * 1000),
                    "life_event_id": result["event"].id,
                    "message_id": result["message"].id if result["message"] is not None else None,
                    "state_version": result["state"].version,
                }
                # 包括最终 COMMIT 本身也必须落在恢复边界内。
                self.session.commit()
            except Exception as error:
                logging.getLogger(__name__).exception("Runtime cycle %s 失败", trace_id)
                # flush 失败后 Session 已不可写，必须先回滚，再读取已提交记录恢复状态。
                # 不使用失效的 ORM 对象，不让第二个 PendingRollbackError 掩盖原始异常。
                self.session.rollback()
                trace = self.session.get(RuntimeCycleTraceRow, trace_id)
                assert trace is not None
                trace.status = "failed"
                trace.stage = "failed"
                trace.error_code = _safe_error_code(error) or type(error).__name__
                trace.error_message = str(error)[:500]
                trace.completed_at = datetime.now(UTC)
                trace.outcome = {
                    **trace.outcome,
                    "elapsed_ms": int((monotonic() - started) * 1000),
                }
                for event_id in claimed_event_ids:
                    item = self.session.get(RuntimeEventRow, event_id)
                    if item is not None and item.status == "claimed":
                        item.status = "queued"
                        item.claimed_at = None
                for wakeup_id in claimed_wakeup_ids:
                    wakeup = self.session.get(RuntimeWakeupRow, wakeup_id)
                    if wakeup is not None and wakeup.status == "executing":
                        wakeup.status = "scheduled"
                        wakeup.executing_at = None
                self.session.commit()
                raise
            record_span_output(cycle_span, trace.outcome)
        return {"packet": packet, "decision": decision, "trace": trace, **result}

    def recover_stale(self, project_id: str, branch_id: str) -> None:
        """仅回收超出租约的领取状态，绝不抢占正在运行的另一台 Worker。"""
        self._branch(project_id, branch_id)
        stale_before = datetime.now(UTC) - _CLAIM_LEASE
        self.session.query(RuntimeEventRow).filter(
            RuntimeEventRow.branch_id == branch_id,
            RuntimeEventRow.status == "claimed",
            (RuntimeEventRow.claimed_at.is_(None)) | (RuntimeEventRow.claimed_at < stale_before),
        ).update({"status": "queued", "claimed_at": None}, synchronize_session=False)
        self.session.query(RuntimeWakeupRow).filter(
            RuntimeWakeupRow.branch_id == branch_id,
            RuntimeWakeupRow.status == "executing",
            (RuntimeWakeupRow.executing_at.is_(None))
            | (RuntimeWakeupRow.executing_at < stale_before),
        ).update({"status": "scheduled", "executing_at": None}, synchronize_session=False)
        self.session.flush()

    def get_state(self, project_id: str, branch_id: str) -> dict[str, Any]:
        self._branch(project_id, branch_id)
        row = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id, RuntimeLifeStateRow.is_current.is_(True)
            )
        )
        return (
            row.state if row is not None else self.bootstrap(project_id, branch_id)["state"].state
        )

    def get_snapshot(self, project_id: str, branch_id: str) -> RuntimeSnapshotRow:
        self._branch(project_id, branch_id)
        snapshot = self.session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
        )
        if snapshot is None:
            snapshot = self.bootstrap(project_id, branch_id)["snapshot"]
        return snapshot

    def get_clock(self, project_id: str, branch_id: str) -> VirtualClock:
        self._branch(project_id, branch_id)
        row = self.session.get(RuntimeClockRow, branch_id, populate_existing=True)
        if row is None:
            self.bootstrap(project_id, branch_id)
            row = self.session.get(RuntimeClockRow, branch_id)
        assert row is not None
        return VirtualClock(
            branch_id=branch_id,
            virtual_anchor=row.virtual_anchor,
            wall_anchor=row.wall_anchor,
            time_scale=row.time_scale,
            status=cast(Any, row.status),
            timezone=row.timezone,
        )

    def get_plan(self, project_id: str, branch_id: str) -> RuntimeDayPlanRow:
        self._branch(project_id, branch_id)
        bootstrap = self.bootstrap(project_id, branch_id)
        return cast(RuntimeDayPlanRow, bootstrap["plan"])

    def change_clock(self, project_id: str, branch_id: str, *, action: str) -> VirtualClock:
        self._branch(project_id, branch_id)
        row = self.session.get(RuntimeClockRow, branch_id)
        if row is None:
            row = RuntimeExecutor(self.session).bootstrap(
                project_id=project_id, branch_id=branch_id
            )["clock"]
            row = self.session.get(RuntimeClockRow, branch_id)
        assert row is not None
        clock = VirtualClock(
            branch_id=branch_id,
            virtual_anchor=row.virtual_anchor,
            wall_anchor=row.wall_anchor,
            time_scale=row.time_scale,
            status=cast(Any, row.status),
            timezone=row.timezone,
        )
        changed = (
            pause_clock(clock)
            if action == "pause"
            else resume_clock(clock)
            if action == "resume"
            else None
        )
        if changed is None:
            raise ValueError("action 必须是 pause 或 resume")
        row.virtual_anchor, row.wall_anchor, row.status = (
            changed.virtual_anchor,
            changed.wall_anchor,
            changed.status,
        )
        # 暂停/恢复会改变提交边界，正在推理的旧输入不能继续提交。
        self.session.execute(
            update(Branch)
            .where(Branch.id == branch_id)
            .values(runtime_input_revision=Branch.runtime_input_revision + 1)
        )
        self.session.commit()
        return changed

    def _branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self.session.scalar(
            select(Branch).where(Branch.id == branch_id, Branch.project_id == project_id)
        )
        if branch is None:
            raise LookupError("时间分支不存在")
        return branch

    def _run_day_planner(
        self,
        *,
        branch_id: str,
        snapshot: RuntimeSnapshotRow,
        timezone: str,
        virtual_now: datetime,
        mode: str,
        request: PlanRevisionRequest | None,
        trace: RuntimeCycleTraceRow,
        input_revision: int | None = None,
        collaboration: list[dict] | None = None,
        working_state: dict | None = None,
        cancellation_requested=None,
    ):
        """运行规划领域调查，只返回提案或追问；写入只能发生在图的 Executor 节点。"""

        effective_input_revision = (
            input_revision
            if input_revision is not None
            else self._runtime_input_revision(branch_id)
        )
        if effective_input_revision is None:
            raise RuntimeInputSupersededError("分支已不存在，不能提交 DayPlan")
        target_date = virtual_now.date()
        if request is not None and request.target_date is not None:
            target_date = request.target_date
        context = PlanContextAssembler(
            self.session,
            work_calendar=self.work_calendar,
        ).assemble(
            branch_id=branch_id,
            snapshot=snapshot,
            target_date=target_date,
            timezone=timezone,
            virtual_now=virtual_now,
            mode=mode,
            request=request,
        )
        context.collaboration = list(collaboration or [])
        from moonlightbox.jobs.models import Job

        from .collaboration.transport import peer_history
        from .tools.send_agent_message import build_send_agent_message_tool

        context.collaboration = peer_history(
            self.session, branch_id, recipient="day_planner", current=context.collaboration
        )
        job = self.session.get(Job, trace.job_id) if trace.job_id else None
        context.allow_no_change = bool(job and job.payload.get("planner_task", {}).get("input_ids"))
        context.working_state = dict(working_state or {})
        # 日历来源和版本必须进入 Cycle Trace；否则事后只能看到“安排了工作”，
        # 却无法解释当时为何把自然周末认定为法定工作日。
        trace.outcome = {
            **trace.outcome,
            f"planner_{mode}_date_features": context.date_features,
            "planner_background_binding": context.origin_projection.get("background_binding", {}),
        }
        trace.stage = "planner"
        self.session.commit()
        from .tools.recent_life_events import build_recent_life_events_tool

        run = self.planner.run_with_trace(
            context,
            cancellation_requested=cancellation_requested,
            tools=[
                build_send_agent_message_tool(
                    self.session,
                    branch_id=branch_id,
                    sender="day_planner",
                    task_id=trace.job_id or trace.cycle_key,
                    cancellation_requested=cancellation_requested,
                ),
                build_recent_life_events_tool(self.session, branch_id, virtual_now),
                build_plan_constraints_tool(context),
                build_routine_evidence_tool(
                    self.session,
                    snapshot=snapshot,
                    context=context,
                ),
                build_plan_memory_tool(
                    self.session,
                    branch_id=branch_id,
                    snapshot=snapshot,
                    virtual_now=virtual_now,
                ),
            ],
            owner_id=trace.id,
            project_id=trace.project_id,
            input_revision=effective_input_revision,
            input_revision_resolver=self._runtime_input_revision_resolver(branch_id),
        )
        if run.terminal_reason in {"cancelled", "stale_input_revision"}:
            raise RuntimeInputSupersededError("分支执行条件已变更或任务已取消，不能提交 DayPlan")
        if run.completion is not None:
            return run.completion, "completed_without_change", run.working_state
        if run.reply is not None:
            return run.reply, "peer_reply", run.working_state
        if run.proposal is None:
            return None, run.validation_error or run.terminal_reason, run.working_state
        return run.proposal, "proposed", run.working_state

    def _runtime_input_revision(self, branch_id: str) -> int | None:
        """从数据库读取版本而非复用 ORM 缓存，供跨 Worker 的 stale 检查使用。"""

        value = self.session.scalar(
            select(Branch.runtime_input_revision).where(Branch.id == branch_id)
        )
        return int(value) if isinstance(value, int) else None

    def _runtime_input_revision_resolver(self, branch_id: str) -> Callable[[], int | None]:
        def resolve():
            # 独立只读 Session，避免 flush 当前 Cycle，也避免复用旧读取快照。
            with Session(self.session.get_bind()) as reader:
                return reader.scalar(
                    select(Branch.runtime_input_revision).where(Branch.id == branch_id)
                )

        return resolve

    def _apply_plan_boundary(
        self, branch_id: str, virtual_now: datetime, blocks: list[dict[str, Any]]
    ) -> None:
        """计划块切换由代码推进，避免 Director 伪造当前活动。"""
        current = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
        )
        if current is None:
            return
        block = next(
            (
                item
                for item in blocks
                if item.get("start", "") <= virtual_now.strftime("%H:%M") < item.get("end", "")
            ),
            None,
        )
        block_id = block.get("id") if block else None
        if block_id == (current.state or {}).get("current_plan_block_id"):
            return
        next_version = current.version + 1
        valid_until = _plan_block_end(virtual_now, block)
        state = {
            **(current.state or {}),
            "virtual_now": virtual_now.isoformat(),
            "current_plan_block_id": block_id,
            "activity": block.get("activity") if block else "unknown",
            "location_role": block.get("location_role") if block else None,
            "availability": block.get("default_availability", "unknown") if block else "unknown",
            "version": next_version,
            "last_transition_at": virtual_now.isoformat(),
            "reason": "plan_boundary",
            "source_event_ids": [],
            "field_sources": {
                **dict((current.state or {}).get("field_sources") or {}),
                "activity": [],
                "location_role": [],
                "availability": [],
                "current_plan_block_id": [],
            },
            "previous_version_id": current.id,
            "valid_until": valid_until.isoformat() if valid_until is not None else None,
        }
        RuntimeExecutor(self.session).retire_current_state(
            current,
            expected_version=current.version,
        )
        self.session.add(
            RuntimeLifeStateRow(
                branch_id=branch_id,
                version=next_version,
                state=state,
                virtual_now=virtual_now,
                current_plan_block_id=block_id,
                availability=state["availability"],
                energy=state.get("energy", "unknown"),
                activity=state["activity"],
                location_role=state["location_role"],
                social_context=state.get("social_context"),
                mood=state.get("mood"),
                attention=state.get("attention"),
                current_goal=state.get("current_goal"),
                valid_until=valid_until,
                reason="plan_boundary",
                source_event_ids=[],
                field_sources={
                    "activity": [],
                    "location_role": [],
                    "availability": [],
                    "current_plan_block_id": [],
                },
                previous_version_id=current.id,
                is_current=True,
            )
        )

    def _expire_state(
        self,
        branch_id: str,
        virtual_now: datetime,
        blocks: list[dict[str, Any]],
    ) -> None:
        """过期短期字段回落到当前 DayPlan，而不是删除原始 LifeEvent。"""

        current = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
        )
        if current is None or current.valid_until is None:
            return
        expiry = current.valid_until
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        now = virtual_now if virtual_now.tzinfo is not None else virtual_now.replace(tzinfo=UTC)
        if expiry > now:
            return
        next_version = current.version + 1
        block = next(
            (
                item
                for item in blocks
                if item.get("start", "") <= virtual_now.strftime("%H:%M") < item.get("end", "")
            ),
            None,
        )
        state = {
            **(current.state or {}),
            "virtual_now": virtual_now.isoformat(),
            "activity": block.get("activity") if block else "unknown",
            "location_role": block.get("location_role") if block else None,
            "availability": block.get("default_availability", "unknown") if block else "unknown",
            "energy": "unknown",
            "mood": None,
            "attention": None,
            "valid_until": None,
            "version": next_version,
            "last_transition_at": virtual_now.isoformat(),
            "reason": "expiry",
            "source_event_ids": [],
            "previous_version_id": current.id,
            "field_sources": {
                **dict((current.state or {}).get("field_sources") or {}),
                "activity": [],
                "location_role": [],
                "availability": [],
                "energy": [],
                "mood": [],
                "attention": [],
            },
        }
        RuntimeExecutor(self.session).retire_current_state(
            current,
            expected_version=current.version,
        )
        self.session.add(
            RuntimeLifeStateRow(
                branch_id=branch_id,
                version=next_version,
                state=state,
                virtual_now=virtual_now,
                current_plan_block_id=state.get("current_plan_block_id"),
                availability=state["availability"],
                energy="unknown",
                activity=state["activity"],
                location_role=state["location_role"],
                social_context=state.get("social_context"),
                mood=None,
                attention=None,
                current_goal=state.get("current_goal"),
                valid_until=None,
                reason="expiry",
                source_event_ids=[],
                field_sources=dict(state.get("field_sources") or {}),
                previous_version_id=current.id,
                is_current=True,
            )
        )


def _packet_source_ids(packet: ContextPacket) -> set[str]:
    messages = packet.branch.get("working_window", {}).get("messages", [])
    return {str(item.get("source_id")) for item in messages if item.get("source_id")}


def _trace_error_code(steps: list[dict[str, object]]) -> str | None:
    """从受控 Trace 读取模型适配器的稳定诊断码。"""

    for step in reversed(steps):
        code = step.get("error_code")
        if isinstance(code, str) and code:
            return code
    return None


def _safe_error_code(error: Exception) -> str | None:
    if isinstance(error, SQLAlchemyError):
        return type(error).__name__
    code = getattr(error, "code", None)
    return code if isinstance(code, str) and code else None


def _merge_triggers(events, wakeups):
    """按接入顺序合并，不替 Director 预设哪类输入更重要。"""
    rows = [
        (_as_utc(e.received_at), e.id, e.event_type, e.occurred_at, dict(e.payload or {}))
        for e in events
    ]
    rows += [
        (datetime.now(UTC), w.id, w.trigger_type, w.wake_at, {"reason": w.reason}) for w in wakeups
    ]
    rows.sort(key=lambda item: (item[0], item[1]))
    inputs = [
        {"id": ident, "type": kind, "occurred_at": _as_utc(at).isoformat(), "payload": payload}
        for _, ident, kind, at, payload in rows
    ]
    return [row["id"] for row in inputs], {
        **inputs[0],
        "additional_triggers": inputs[1:],
    }


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _plan_block_end(now: datetime, block: dict[str, Any] | None) -> datetime | None:
    if block is None or not isinstance(block.get("end"), str):
        return None
    try:
        hour, minute = (int(part) for part in block["end"].split(":", 1))
    except (TypeError, ValueError):
        return None
    if hour == 24 and minute == 0:
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
