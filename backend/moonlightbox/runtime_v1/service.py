"""Runtime v1 应用服务：把队列、上下文、LangGraph 和 Executor 串成一轮 Cycle。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.jobs.service import JobService

from .actor import ActorModel, PersonaActor
from .clock import pause_clock, resume_clock
from .context import ContextAssembler
from .db_models import (
    RuntimeClockRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeStateRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from .director import DirectorAgent, DirectorModel
from .executor import RuntimeExecutor
from .memory import MemoryService
from .schemas import ActorMessage, ContextPacket, LifeDecision, VirtualClock
from .style import StyleService

_CLAIM_LEASE = timedelta(minutes=5)


class RuntimeService:
    """面向 HTTP/Worker 的门面；旧 branches 服务不会直接调用模型。"""

    def __init__(
        self,
        session: Session,
        *,
        director_model: DirectorModel | None = None,
        actor_model: ActorModel | None = None,
    ) -> None:
        self.session = session
        self.director = DirectorAgent(director_model)
        self.actor = PersonaActor(actor_model)

    def bootstrap(self, project_id: str, branch_id: str) -> dict[str, Any]:
        result = RuntimeExecutor(self.session).bootstrap(project_id=project_id, branch_id=branch_id)
        self.session.commit()
        return result

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
        self._branch(project_id, branch_id)
        # 入队前冻结背景并初始化计划/状态；如果人物世界尚未完成，直接返回明确错误，
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
        virtual_occurred_at = occurred_at or bootstrap["clock"].now()
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
            generation_metadata={"runtime_v1": True},
            client_message_id=client_message_id or idempotency_key[:64],
            observed_at=virtual_occurred_at,
        )
        self.session.add(message)
        self.session.flush()
        event = RuntimeExecutor(self.session).append_user_event(
            project_id=project_id,
            branch_id=branch_id,
            content=content,
            idempotency_key=idempotency_key,
            occurred_at=virtual_occurred_at,
        )
        event.payload = {"content": content, "branch_message_id": message.id}
        from .jobs import RUNTIME_CYCLE_JOB_KIND

        JobService(self.session).enqueue_unique(
            RUNTIME_CYCLE_JOB_KIND,
            {"project_id": project_id, "branch_id": branch_id, "event_id": event.id},
            dedupe_key=f"{RUNTIME_CYCLE_JOB_KIND}:{event.id}",
            commit=False,
        )
        self.session.commit()
        return {"event": event, "message": message}

    def process_next(self, *, project_id: str, branch_id: str) -> dict[str, Any] | None:
        """领取该分支最早事件并运行一次；模型调用期间不持有数据库写锁。"""
        self.recover_stale(project_id, branch_id)
        branch = self._branch(project_id, branch_id)
        bootstrap = RuntimeExecutor(self.session).bootstrap(
            project_id=project_id, branch_id=branch_id
        )
        # 一个 Cycle 内所有组件必须使用同一 virtual_now，不能每一步重新取墙上时间。
        virtual_now = bootstrap["clock"].now()
        queued_events = list(
            self.session.scalars(
                select(RuntimeEventRow)
                .where(RuntimeEventRow.branch_id == branch_id, RuntimeEventRow.status == "queued")
                .order_by(RuntimeEventRow.priority.desc(), RuntimeEventRow.occurred_at)
                .limit(16)
            )
        )
        events = queued_events
        if events:
            # 250ms 合并窗口：同一分支同时到达的用户消息只触发一次 Director。
            window_end = events[0].occurred_at + timedelta(milliseconds=250)
            events = [item for item in events if item.occurred_at <= window_end]
        wakeups = list(
            self.session.scalars(
                select(RuntimeWakeupRow)
                .where(
                    RuntimeWakeupRow.branch_id == branch_id,
                    RuntimeWakeupRow.status == "scheduled",
                    RuntimeWakeupRow.wake_at <= virtual_now,
                )
                .order_by(RuntimeWakeupRow.wake_at, RuntimeWakeupRow.id)
                .limit(16)
            )
        )
        if not events and not wakeups:
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
        if not claimed_events and not claimed_wakeups:
            self.session.rollback()
            return None
        events = claimed_events
        wakeups = claimed_wakeups
        trigger_ids, trigger_payload = _merge_triggers(events, wakeups)
        # 领取状态先提交，避免一次远端模型推理长期占住 SQLite 写事务。乐观锁会
        # 防止其后到达的实时 Cycle 被旧结果覆盖。
        self.session.commit()
        bootstrap = RuntimeExecutor(self.session).bootstrap(
            project_id=project_id, branch_id=branch_id
        )
        clock = bootstrap["clock"]
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
        packet = ContextAssembler(
            self.session,
            token_counter=self._director_token_counter(),
        ).assemble(
            branch_id=branch_id,
            trigger=trigger_payload,
            clock=clock,
            snapshot=bootstrap["snapshot"],
            now=virtual_now,
        )
        tool = MemoryService(self.session).tool(
            branch_id=branch_id,
            snapshot_id=bootstrap["snapshot"].id,
            excluded_source_ids=_packet_source_ids(packet),
            as_of=virtual_now,
        )
        decision = self.director.run(packet, search_tool=tool)
        actor_message: ActorMessage | None = None
        expected = int(bootstrap["state"].version)
        executor = RuntimeExecutor(self.session)
        try:
            # 先通过纯代码约束，再允许 Actor 消耗 LoRA/显存生成公开文本。
            executor.validate(
                branch_id=branch_id,
                decision=decision,
                expected_version=expected,
                virtual_now=virtual_now,
            )
        except (RuntimeError, ValueError) as error:
            decision = LifeDecision(
                action="wait",
                private_reason=f"Executor 拒绝 Director 决定：{str(error)[:420]}",
            )
        else:
            if decision.action == "speak":
                actor_message = self.actor.run(
                    packet=packet,
                    intent=decision.communication_intent or "回复用户",
                    content_points=decision.content_points,
                    style_tool=StyleService(self.session).tool(
                        branch_id=branch_id,
                        model_version_id=branch.model_version_id,
                    ),
                )
        try:
            result = executor.commit(
                project_id=project_id,
                branch_id=branch_id,
                decision=decision,
                expected_version=expected,
                virtual_now=virtual_now,
                trigger_event_ids=trigger_ids,
                actor_message=actor_message,
                idempotency_key=f"cycle:{branch_id}:{','.join(sorted(trigger_ids))}",
            )
        except Exception:
            # 模型失败或版本冲突不覆盖有效状态，事件回到队列等待重试。
            for item in claimed_events:
                item.status = "queued"
            for wakeup in claimed_wakeups:
                wakeup.status = "scheduled"
                wakeup.executing_at = None
            self.session.commit()
            raise
        for item in claimed_events:
            item.status = "completed"
            item.completed_at = datetime.now(UTC)
        for wakeup in claimed_wakeups:
            wakeup.status = "completed"
            wakeup.executing_at = None
        self.session.commit()
        return {"packet": packet, "decision": decision, **result}

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
        row = self.session.get(RuntimeClockRow, branch_id)
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
        self.session.commit()
        return changed

    def _branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self.session.scalar(
            select(Branch).where(Branch.id == branch_id, Branch.project_id == project_id)
        )
        if branch is None:
            raise LookupError("时间分支不存在")
        return branch

    def _director_token_counter(self) -> Callable[[str], int] | None:
        """只接受远端 Director 暴露的真实 tokenizer；缺失时安全等待。"""

        counter = getattr(self.director.model, "count_text_tokens", None)
        return counter if callable(counter) else None

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


def _merge_triggers(
    events: list[RuntimeEventRow],
    wakeups: list[RuntimeWakeupRow],
) -> tuple[list[str], dict[str, Any]]:
    """用固定优先级把同一分支已领取的触发合并为一次 Cycle。"""

    candidates: list[tuple[int, datetime, str, str, dict[str, Any]]] = []
    for event in events:
        candidates.append(
            (
                _trigger_priority(event.event_type),
                _as_utc(event.occurred_at),
                event.id,
                event.event_type,
                dict(event.payload or {}),
            )
        )
    for wakeup in wakeups:
        candidates.append(
            (
                _trigger_priority(wakeup.trigger_type),
                _as_utc(wakeup.wake_at),
                wakeup.id,
                wakeup.trigger_type,
                {"reason": wakeup.reason},
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    primary = candidates[0]
    additional = [
        {
            "id": trigger_id,
            "type": trigger_type,
            "occurred_at": occurred_at.isoformat(),
            "payload": payload,
        }
        for _, occurred_at, trigger_id, trigger_type, payload in candidates[1:]
    ]
    return (
        [item[2] for item in candidates],
        {
            "type": primary[3],
            "id": primary[2],
            "occurred_at": primary[1],
            "payload": primary[4],
            "additional_triggers": additional,
        },
    )


def _trigger_priority(trigger_type: str) -> int:
    """用户消息优先于承诺、延迟回复和日程边界，和设计文档保持一致。"""

    return {
        "user_message": 400,
        "commitment_due": 300,
        "commitment": 300,
        "delayed_reply": 200,
        "plan_transition": 100,
    }.get(trigger_type, 0)


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
