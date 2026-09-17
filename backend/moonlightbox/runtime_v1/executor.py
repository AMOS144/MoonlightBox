"""Executor：Runtime 唯一的副作用出口。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from moonlightbox.observability.runtime import observed_stage
from moonlightbox.world.models import (
    PersonWorldProfile,
    WorldGraphVersion,
    WorldPublication,
)

from .branch_models import Branch, BranchMessage
from .clock import create_clock
from .day_plan import (
    _initial_state,
    _reused_block_id,
    _same_plan_block,
    _schedule_next_boundary,
    validate_plan_evidence_sources,
)
from .db_models import (
    RuntimeClockRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeLifeStateRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from .event_queue import RuntimeEventQueue
from .memory import MemoryService
from .memory_validation import (
    validate_decision_references as _check_decision_references,
)
from .memory_validation import (
    validate_memory_sources,
    validate_state_patch_sources,
)
from .profile_projection import (
    _latest_import_node,
    _latest_profile_cutoff,
    _profile_payload,
    _runtime_routine_profile,
    _runtime_time_anchor,
    _seed_world_memory,
)
from .schemas import (
    ActorMessage,
    DayPlanProposal,
    LifeDecision,
    MemoryRecord,
    VirtualClock,
)


class RuntimeWorldUnavailableError(RuntimeError):
    """运行时只能读取已完成的 LightRAG/PWP，不允许用空背景启动人物。"""


class RuntimeExecutor:
    """负责校验、幂等和事务写入，不调用模型也不理解自然语言。"""

    def commit_initial_state(self, branch_id, expected_state_id, value, allowed, now):
        """仅准备态可提交，事务失败不能留下半份初始化结果。"""
        from copy import deepcopy
        from uuid import uuid4

        from .branch_models import BranchMessage
        from .subjective_state import apply_update

        branch = self.session.get(Branch, branch_id)
        current = self._current_state(branch_id)
        if (
            branch.lifecycle_status not in {"preparing", "prepare_failed"}
            or current.id != expected_state_id
        ):
            raise ValueError("初始化起点已变化，不能覆盖当前状态")
        if self.session.scalar(
            select(BranchMessage.id).where(BranchMessage.branch_id == branch_id).limit(1)
        ):
            raise ValueError("已开始聊天，不能重置起点")
        if current.state.get("subjective_state"):
            raise ValueError("已存在心理状态，不能被初始化覆盖")
        entries = [*value.open_conversation_threads, *value.active_commitments]
        if any(set(item.source_refs) - allowed for item in entries):
            raise ValueError("初始化引用越界")
        state = deepcopy(current.state)
        state["subjective_state"] = apply_update(None, value.subjective_state, allowed, now)
        state["open_conversation_threads"] = [
            {**item.model_dump(mode="json"), "id": str(uuid4()), "status": "active"}
            for item in value.open_conversation_threads
        ]
        state["active_commitments"] = [
            {**item.model_dump(mode="json"), "id": str(uuid4()), "status": "active"}
            for item in value.active_commitments
        ]
        state.update(version=current.version + 1, reason="director_initialization")
        payload = {
            column.name: getattr(current, column.name)
            for column in RuntimeLifeStateRow.__table__.columns
            if column.name
            not in {"id", "created_at", "state", "version", "previous_version_id", "reason"}
        }
        changed = self.session.execute(
            update(RuntimeLifeStateRow)
            .where(
                RuntimeLifeStateRow.id == expected_state_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
            .values(is_current=False)
        )
        if changed.rowcount != 1:
            raise ValueError("初始化状态版本已被其他事务更新")
        self.session.add(
            RuntimeLifeStateRow(
                **payload,
                state=state,
                version=state["version"],
                previous_version_id=current.id,
                reason="director_initialization",
            )
        )
        self.session.flush()

    def commit_simulated_event(self, row, **kwargs):
        """模拟生活的唯一副作用入口，调用方负责提交或回滚同一事务。"""
        from .life_events.commit import commit_simulated_event

        return commit_simulated_event(self, row, **kwargs)

    def __init__(self, session: Session) -> None:
        self.session = session

    def bootstrap(self, *, project_id: str, branch_id: str) -> dict[str, Any]:
        branch = self._branch(project_id, branch_id)
        # 已批准起点同时决定时间与时区，不从训练产物推断。
        runtime_anchor, runtime_timezone = _runtime_time_anchor(self.session, branch)
        snapshot = self.session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
        )
        if snapshot is None:
            if branch.origin_boundary:
                raise RuntimeWorldUnavailableError(
                    "历史起点必须先完成独立快照编译，不能使用最新背景"
                )
            publication = self.session.scalar(
                select(WorldPublication)
                .where(
                    WorldPublication.project_id == project_id,
                    WorldPublication.status == "active",
                    WorldPublication.node_boundary_hash.is_(None),
                )
                .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
            )
            if publication is not None:
                published_graph = self.session.get(WorldGraphVersion, publication.graph_version_id)
                profile = (
                    self.session.get(PersonWorldProfile, publication.profile_id)
                    if published_graph is not None and published_graph.status == "ready"
                    else None
                )
            else:
                # 兼容升级前尚未建立 WorldPublication 的 ready 图。
                profile = self.session.scalar(
                    select(PersonWorldProfile)
                    .where(PersonWorldProfile.node_boundary_hash.is_(None))
                    .join(
                        WorldGraphVersion,
                        PersonWorldProfile.graph_version_id == WorldGraphVersion.id,
                    )
                    .where(
                        PersonWorldProfile.project_id == project_id,
                        WorldGraphVersion.status == "ready",
                    )
                    .order_by(PersonWorldProfile.created_at.desc())
                )
            if profile is None:
                raise RuntimeWorldUnavailableError("人物世界尚未就绪，不能初始化 Runtime 分支")
            # 仅兼容没有新起点契约的旧末端分支。新历史分支必须已有独立快照，
            # 上方门禁禁止它进入全量背景兜底；时间查询由冻结范围适配器承担。
            cutoff = _latest_profile_cutoff(self.session, profile, branch.origin_time)
            snapshot = RuntimeSnapshotRow(
                branch_id=branch_id,
                graph_version_id=profile.graph_version_id if profile is not None else None,
                profile_id=profile.id if profile is not None else None,
                source_node_id=_latest_import_node(self.session, project_id),
                cutoff_at=cutoff,
                timezone=runtime_timezone,
                snapshot_mode="latest_profile",
                source_message_ids=list(profile.source_message_ids),
                profile=_profile_payload(profile),
                routine_profile=_runtime_routine_profile(profile),
                compiler_version=profile.compiler_version,
            )
            self.session.add(snapshot)
            self.session.flush()
        _seed_world_memory(self.session, snapshot)
        # 历史已审核记忆由数据库退役迁移一次性导入。Runtime v1 后续只读取
        # 自己的 MemoryRecord / MemoryIndexVersion，绝不再依赖旧连续记忆表。
        memory_service = MemoryService(self.session)
        memory_service.rebuild_index(branch_id, reason="bootstrap")
        clock_row = self.session.get(RuntimeClockRow, branch_id)
        if clock_row is None:
            clock = create_clock(
                branch_id,
                runtime_anchor,
                timezone=runtime_timezone,
            )
            clock_row = RuntimeClockRow(
                branch_id=branch_id,
                virtual_anchor=clock.virtual_anchor,
                wall_anchor=clock.wall_anchor,
                time_scale=clock.time_scale,
                status=clock.status,
                timezone=clock.timezone,
            )
            self.session.add(clock_row)
        else:
            clock = self._clock(clock_row)
        from .collaboration.plans import local_time

        if (
            branch.lifecycle_status in {"preparing", "prepare_failed"}
            and clock_row.status != "paused"
        ):
            from .clock import pause_clock

            clock = pause_clock(clock)
            clock_row.virtual_anchor, clock_row.wall_anchor = (
                clock.virtual_anchor,
                clock.wall_anchor,
            )
            clock_row.status = "paused"
        local_now = local_time(clock.now(), clock.timezone)
        plan = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id,
                RuntimeDayPlanRow.plan_date == local_now.date().isoformat(),
            )
        )
        if plan is None:
            plan = RuntimeDayPlanRow(
                branch_id=branch_id,
                plan_date=local_now.date().isoformat(),
                # 日程内容只能由 DayPlanAgent 提案。这里的空列表是一个“尚未
                # 生成”的持久化槽位，不是睡眠、通勤、工作等固定日程的变体。
                blocks=[],
                generation_metadata={
                    "status": "pending",
                    "planner_version": "runtime-day-plan-v1",
                    "reason": "awaiting_day_plan_agent",
                },
            )
            self.session.add(plan)
            self.session.flush()
        state = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
        )
        if state is None:
            current = _initial_state(branch_id, local_now, plan.blocks)
            state = RuntimeLifeStateRow(
                branch_id=branch_id,
                version=1,
                state=current.model_dump(mode="json"),
                virtual_now=current.virtual_now,
                current_plan_block_id=current.current_plan_block_id,
                availability=current.availability,
                energy=current.energy,
                activity=current.activity,
                location_role=current.location_role,
                social_context=current.social_context,
                mood=current.mood,
                attention=current.attention,
                current_goal=current.current_goal,
                reason="initial",
                source_event_ids=[],
                field_sources={},
                previous_version_id=None,
                is_current=True,
            )
            self.session.add(state)
            self.session.flush()
            # 空的 pending 槽位不安排任何生活边界；Worker 会先运行 Planner，
            # 通过验收后再由 commit_day_plan_proposal 安排首个 Wakeup。
            _schedule_next_boundary(self.session, branch_id, local_now, plan.blocks)
        return {"snapshot": snapshot, "clock": clock, "plan": plan, "state": state}

    def append_user_event(
        self,
        *,
        project_id: str,
        branch_id: str,
        content: str,
        idempotency_key: str,
        occurred_at: datetime | None = None,
    ) -> RuntimeEventRow:
        """经统一队列写入用户触发；HTTP 层不会直接进入 Director。"""

        self._branch(project_id, branch_id)
        return RuntimeEventQueue(self.session).enqueue(
            project_id=project_id,
            branch_id=branch_id,
            event_type="user_message",
            payload={"content": content},
            idempotency_key=idempotency_key,
            occurred_at=occurred_at,
            priority=100,
        )

    @observed_stage("runtime.executor.validate")
    def validate(
        self,
        *,
        branch_id: str,
        decision: LifeDecision,
        expected_version: int,
        virtual_now: datetime,
    ) -> None:
        current = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
        )
        if current is None or current.version != expected_version:
            raise RuntimeError("life_state_version_conflict")
        if decision.action == "speak" and decision.expression_task is None:
            raise ValueError("speak 必须提供 expression_task")
        if decision.next_wakeup_at is not None and decision.next_wakeup_at <= virtual_now:
            raise ValueError("next_wakeup_at 必须晚于 virtual_now")
        if decision.action == "schedule" and decision.next_wakeup_at is None:
            raise ValueError("schedule 必须提供 next_wakeup_at")
        if decision.action != "speak" and decision.speech_mode is not None:
            raise ValueError("只有 speak 可以设置 speech_mode")
        if decision.plan_request is not None:
            target_date = decision.plan_request.target_date or virtual_now.date()
            if target_date != virtual_now.date():
                raise ValueError("当前 Runtime 仅允许修订虚拟当天的 DayPlan")
            self._validate_state_patch_sources(
                branch_id=branch_id,
                source_ids=decision.plan_request.source_event_ids,
            )
        patch = decision.state_patch.model_dump(exclude_none=True)
        if patch.get("activity") and patch.get("activity") != current.activity:
            # 跨越默认生活块的活动变化必须由规划请求或已落库分支事件支撑。
            if decision.plan_request is None and not decision.state_patch.source_event_ids:
                raise ValueError("改变 activity 必须提供 plan_request 或 source_event_ids")

    @observed_stage("runtime.executor.commit")
    def commit(
        self,
        *,
        project_id: str,
        branch_id: str,
        decision: LifeDecision,
        expected_version: int,
        virtual_now: datetime,
        trigger_event_ids: list[str],
        actor_message: ActorMessage | None = None,
        expression_result=None,
        asset_sources=None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = idempotency_key or f"cycle:{branch_id}:{','.join(sorted(trigger_event_ids))}"
        existing = self.session.scalar(
            select(RuntimeLifeEventRow).where(
                RuntimeLifeEventRow.branch_id == branch_id,
                RuntimeLifeEventRow.idempotency_key == key,
            )
        )
        if existing is not None:
            return {
                "event": existing,
                "message": self.session.scalar(
                    select(BranchMessage).where(
                        BranchMessage.branch_id == branch_id, BranchMessage.turn_id == existing.id
                    )
                ),
                "state": self._current_state(branch_id),
            }
        if decision.plan_request is not None or decision.peer_reply is not None:
            raise ValueError("协作请求必须先由图路由处理，不能作为最终行为直接提交")
        reference_error = self.validate_decision_references(branch_id, decision, virtual_now)
        if reference_error:
            raise ValueError(reference_error)
        self.validate(
            branch_id=branch_id,
            decision=decision,
            expected_version=expected_version,
            virtual_now=virtual_now,
        )
        from .expression_contracts import ExpressionResult

        if expression_result is not None:
            if expression_result.status != "ready":
                raise ValueError("未完成的表达不能提交")
            actor_message = expression_result
        elif actor_message is not None:
            # 只为历史内部调用保留文本适配；新模型只输出 messages。
            actor_message = ExpressionResult(
                status="ready",
                messages=[
                    {"kind": "text", "text": text}
                    for text in actor_message.bubbles or [actor_message.text]
                ],
            )
        if actor_message is not None:
            from .style_history import validate_asset

            branch = self._branch(project_id, branch_id)
            snapshot = self.session.scalar(
                select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
            )
            for part in actor_message.messages:
                if part.kind == "sticker":
                    if part.asset_ref not in (asset_sources or {}):
                        raise ValueError("sticker_not_supplied_to_actor")
                    validate_asset(self.session, branch, snapshot, part.asset_ref)
        if decision.action == "speak" and actor_message is None:
            raise ValueError("speak 决定必须包含可提交回复")
        if actor_message is not None and decision.action != "speak":
            raise ValueError("非 speak 决定不能提交回复")
        current = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id, RuntimeLifeStateRow.is_current.is_(True)
            )
        )
        if current is None or current.version != expected_version:
            raise RuntimeError("life_state_version_conflict")
        old = dict(current.state or {})
        patch = decision.state_patch.model_dump(exclude_none=True)
        patch.pop("reason", None)
        patch.pop("source_event_ids", None)
        patch.pop("valid_until", None)
        from .subjective_state import apply_update, commit_memories, resolve_inputs, visible_sources

        allowed = visible_sources(self.session, branch_id, virtual_now)
        decision_ref = str(uuid4())
        if decision.subjective_state_updates is not None:
            if any(
                key in patch
                for key in ("mood", "attention", "current_goal", "open_conversation_threads")
            ):
                raise ValueError("subjective_state_dual_write")
            patch["subjective_state"] = apply_update(
                old.get("subjective_state"),
                decision.subjective_state_updates,
                allowed,
                virtual_now,
                decision_ref=decision_ref,
                has_expression=actor_message is not None,
            )
        commit_memories(
            self.session,
            branch_id,
            decision.memory_proposals,
            allowed,
            virtual_now,
            actor_message is not None,
        )
        resolve_inputs(self.session, branch_id, decision.input_resolutions, virtual_now)
        state: RuntimeLifeStateRow = current
        if patch or decision.action in {"continue_life", "schedule"}:
            self._validate_state_patch_sources(
                branch_id=branch_id,
                source_ids=decision.state_patch.source_event_ids,
            )
            changed_fields = set(patch)
            field_sources = dict(old.get("field_sources") or {})
            for field in changed_fields:
                field_sources[field] = list(
                    decision.state_patch.source_event_ids or trigger_event_ids
                )
            effective_valid_until = _effective_valid_until(
                virtual_now=virtual_now,
                patch=patch,
                requested=decision.state_patch.valid_until,
                current=current.valid_until,
            )
            new_state = {
                **old,
                **patch,
                "virtual_now": virtual_now.isoformat(),
                "version": expected_version + 1,
                "last_transition_at": virtual_now.isoformat(),
                "reason": decision.state_patch.reason or decision.action,
                "source_event_ids": list(dict.fromkeys(trigger_event_ids)),
                "field_sources": field_sources,
                "previous_version_id": current.id,
            }
            new_state["valid_until"] = (
                effective_valid_until.isoformat() if effective_valid_until is not None else None
            )
            self.retire_current_state(current, expected_version=expected_version)
            state = RuntimeLifeStateRow(
                branch_id=branch_id,
                version=expected_version + 1,
                state=new_state,
                virtual_now=virtual_now,
                current_plan_block_id=new_state.get("current_plan_block_id"),
                availability=new_state.get("availability", "unknown"),
                energy=new_state.get("energy", "unknown"),
                activity=new_state.get("activity"),
                location_role=new_state.get("location_role"),
                social_context=new_state.get("social_context"),
                mood=new_state.get("mood"),
                attention=new_state.get("attention"),
                current_goal=new_state.get("current_goal"),
                valid_until=effective_valid_until,
                reason=decision.state_patch.reason or decision.action,
                source_event_ids=trigger_event_ids,
                field_sources=field_sources,
                previous_version_id=current.id,
                is_current=True,
            )
            self.session.add(state)
        life_event = RuntimeLifeEventRow(
            id=decision_ref,
            branch_id=branch_id,
            event_type="decision",
            occurred_at=virtual_now.astimezone(UTC),
            payload=decision.model_dump(mode="json"),
            idempotency_key=key,
        )
        self.session.add(life_event)
        message = None
        if actor_message is not None:
            sequence = self.session.scalar(
                select(func.max(BranchMessage.sequence)).where(BranchMessage.branch_id == branch_id)
            )
            for index, part in enumerate(actor_message.messages):
                item = BranchMessage(
                    branch_id=branch_id,
                    sequence=(int(sequence) + 1 if sequence is not None else 0) + index,
                    role="assistant",
                    turn_id=life_event.id,
                    bubble_index=index,
                    content=part.text if part.kind == "text" else "[表情]",
                    type=part.kind,
                    media_asset_id=part.asset_ref if part.kind == "sticker" else None,
                    generation_status="completed",
                    generation_metadata={
                        "runtime_v1": True,
                        "expression_version": 3,
                        "sticker_usage_refs": (asset_sources or {}).get(part.asset_ref, [])
                        if part.kind == "sticker"
                        else [],
                    },
                    observed_at=virtual_now.astimezone(UTC),
                    is_proactive=decision.speech_mode == "proactive",
                )
                self.session.add(item)
                if message is None:
                    message = item
        if decision.next_wakeup_at is not None:
            self.schedule_wakeup(
                branch_id=branch_id,
                wake_at=decision.next_wakeup_at,
                reason=decision.private_reason or decision.action,
                trigger_type="delayed_reply"
                if decision.speech_mode == "delayed_reply"
                else "system",
                idempotency_key=f"wakeup:{branch_id}:{decision.next_wakeup_at.isoformat()}:{key}",
            )
        # 每个成功的 Cycle 都保证下一个 DayPlan 边界存在；不能只在初始化时
        # 安排一次，否则第一条边界处理完分支就会永久停止自动推进。
        self.ensure_next_plan_boundary(branch_id=branch_id, virtual_now=virtual_now)
        self.session.flush()
        from .conversation_maintenance import enqueue_maintenance

        enqueue_maintenance(self.session, branch_id, life_event.id)
        return {"event": life_event, "message": message, "state": state}

    def schedule_wakeup(
        self,
        *,
        branch_id: str,
        wake_at: datetime,
        reason: str,
        trigger_type: str,
        idempotency_key: str,
        revive_cancelled: bool = False,
    ) -> RuntimeWakeupRow:
        # SQLite 不保存偏移；持久化时间点统一 UTC，本地时区仅用于人物日历。
        wake_at = (wake_at if wake_at.tzinfo else wake_at.replace(tzinfo=UTC)).astimezone(UTC)
        existing = self.session.scalar(
            select(RuntimeWakeupRow).where(
                RuntimeWakeupRow.branch_id == branch_id,
                RuntimeWakeupRow.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            # DayPlan 修订会先取消旧边界，再按新块重建；若恰好仍是同一时刻，
            # 幂等键相同的旧行应被恢复，而不是让分支失去下一条边界唤醒。
            if revive_cancelled and existing.status == "cancelled":
                existing.status = "scheduled"
                existing.wake_at = wake_at
                existing.reason = reason
                existing.trigger_type = trigger_type
                existing.executing_at = None
            return existing
        row = RuntimeWakeupRow(
            branch_id=branch_id,
            wake_at=wake_at,
            reason=reason,
            trigger_type=trigger_type,
            idempotency_key=idempotency_key,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def confirm_memory(self, record: MemoryRecord) -> MemoryRecord:
        """唯一允许写入长期记忆的入口；新事实可显式关闭旧记录。"""
        if record.status != "confirmed":
            raise ValueError("Runtime v1 只接受已确认记忆")
        if record.scope == "branch" and not record.branch_id:
            raise ValueError("branch 记忆必须绑定分支")
        if record.supersedes_id:
            old = self.session.get(RuntimeMemoryRow, record.supersedes_id)
            if old is not None:
                old.status = "superseded"
        self._validate_memory_sources(record)
        MemoryService(self.session).add(record)
        if record.scope == "branch":
            assert record.branch_id is not None
            MemoryService(self.session).rebuild_index(record.branch_id, reason="memory_confirmed")
        self.session.flush()
        return record

    def retire_current_state(
        self,
        current: RuntimeLifeStateRow,
        *,
        expected_version: int,
    ) -> None:
        """通过版本 CAS 退休当前 LifeState，防止并发 Cycle 静默互相覆盖。"""

        result = cast(
            CursorResult[Any],
            self.session.execute(
                update(RuntimeLifeStateRow)
                .where(
                    RuntimeLifeStateRow.id == current.id,
                    RuntimeLifeStateRow.branch_id == current.branch_id,
                    RuntimeLifeStateRow.version == expected_version,
                    RuntimeLifeStateRow.is_current.is_(True),
                )
                .values(is_current=False)
                .execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            raise RuntimeError("life_state_version_conflict")
        # 避免 ORM 在本事务末尾把旧对象缓存的 is_current=True 写回数据库。
        self.session.expire(current)

    def ensure_next_plan_boundary(self, *, branch_id: str, virtual_now: datetime) -> None:
        """为当前分支的当天计划补齐唯一的下一条边界 Wakeup。"""

        plan = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id,
                RuntimeDayPlanRow.plan_date == virtual_now.date().isoformat(),
            )
        )
        if plan is not None:
            _schedule_next_boundary(self.session, branch_id, virtual_now, plan.blocks or [])

    def day_plan_needs_generation(self, plan: RuntimeDayPlanRow) -> bool:
        """判断空闲维护是否到期；聊天本身不承担正常计划准备职责。"""

        from .plan_status import can_schedule

        return can_schedule(plan, datetime.now(UTC))

    def mark_day_plan_unavailable(
        self,
        *,
        plan: RuntimeDayPlanRow,
        reason: str,
        virtual_now: datetime,
    ) -> RuntimeDayPlanRow:
        """记录 Planner 本轮不可用，不伪造固定生活计划。"""

        plan.generation_metadata = {
            **(plan.generation_metadata if isinstance(plan.generation_metadata, dict) else {}),
            "status": "unavailable",
            "planner_version": "runtime-day-plan-v1",
            "reason": reason[:160],
            "updated_at": virtual_now.isoformat(),
        }
        return plan

    @observed_stage("runtime.executor.commit_day_plan")
    def commit_day_plan_proposal(
        self,
        *,
        branch_id: str,
        proposal: DayPlanProposal,
        virtual_now: datetime,
        mode: str,
        date_features: dict[str, Any] | None = None,
        expected_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> RuntimeDayPlanRow:
        """验证并原子替换当天计划；DayPlanAgent 本身永远没有写库能力。"""

        from .collaboration.plans import local_time

        clock_row = self.session.get(RuntimeClockRow, branch_id)
        if clock_row is not None:
            virtual_now = local_time(virtual_now, clock_row.timezone)
        if proposal.plan_date < virtual_now.date() or (
            mode != "life_event"
            and proposal.plan_date
            not in {virtual_now.date(), virtual_now.date() + timedelta(days=1)}
        ):
            raise ValueError("DayPlanProposal 只能写入虚拟当地的今天或明天")
        plan = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id,
                RuntimeDayPlanRow.plan_date == proposal.plan_date.isoformat(),
            )
        )
        if plan is None:
            if expected_version not in {None, 0}:
                raise ValueError("计划版本冲突")
            plan = RuntimeDayPlanRow(
                branch_id=branch_id,
                plan_date=proposal.plan_date.isoformat(),
                blocks=[],
                version=0,
                generation_metadata={"status": "pending"},
            )
            self.session.add(plan)
            self.session.flush()
        metadata = plan.generation_metadata or {}
        if idempotency_key and idempotency_key in metadata.get("commit_keys", []):
            return plan
        if expected_version is not None and plan.version != expected_version:
            raise ValueError("计划版本冲突，必须读取新版本重新判断")
        self._validate_day_plan_proposal(
            branch_id=branch_id,
            plan=plan,
            proposal=proposal,
            virtual_now=virtual_now,
            mode=mode,
        )
        existing_blocks = [dict(item) for item in (plan.blocks or []) if isinstance(item, dict)]
        from .life_events.commit import archive_plan

        archive_plan(self.session, plan)
        # 对语义未变的块复用 ID，特别是 revision 中正在执行的块。否则当前
        # LifeState 会指向一个刚被模型重写掉的 block，并在下轮产生伪边界切换。
        plan.blocks = [
            {
                "id": _reused_block_id(existing_blocks, block.model_dump(mode="json"))
                or str(uuid4()),
                **block.model_dump(mode="json"),
            }
            for block in proposal.blocks
        ]
        plan.version += 1
        plan.generation_metadata = {
            **metadata,
            "preparation": {
                **metadata.get("preparation", {}),
                "status": "ready",
                "retry_at": None,
                "error_code": None,
            },
            "commit_receipts": {
                **metadata.get("commit_receipts", {}),
                **(
                    {
                        idempotency_key: {
                            "plan_date": plan.plan_date,
                            "plan_version": plan.version,
                            "status": "committed",
                            "key": idempotency_key,
                        }
                    }
                    if idempotency_key
                    else {}
                ),
            },
            "commit_keys": [
                *metadata.get("commit_keys", []),
                *([idempotency_key] if idempotency_key else []),
            ],
            "status": "agent",
            "planner_version": "runtime-day-plan-v1",
            "mode": mode,
            "source_ids": list(
                dict.fromkeys(
                    source_id for block in proposal.blocks for source_id in block.evidence_ids
                )
            ),
            "assumptions": proposal.assumptions,
            "reason": proposal.private_reason,
            "date_features": dict(date_features or {}),
            "updated_at": virtual_now.isoformat(),
        }
        if proposal.plan_date != virtual_now.date():
            # 明天的计划只更新明天，绝不能取消今天的 Wakeup 或推进当前活动。
            archive_plan(self.session, plan)
            # 次日首块也需要被唤醒，不能默认早七点才开始生活。
            _schedule_next_boundary(self.session, branch_id, virtual_now, [])
            self.session.flush()
            return plan
        # 旧边界可能已指向被修订掉的时间；先取消再为新计划补一条唯一边界。
        self.session.query(RuntimeWakeupRow).filter(
            RuntimeWakeupRow.branch_id == branch_id,
            RuntimeWakeupRow.status == "scheduled",
            RuntimeWakeupRow.trigger_type == "plan_transition",
            RuntimeWakeupRow.wake_at > virtual_now,
        ).update({"status": "cancelled"}, synchronize_session=False)
        _schedule_next_boundary(self.session, branch_id, virtual_now, plan.blocks)
        archive_plan(self.session, plan)
        self.session.flush()
        return plan

    def _validate_day_plan_proposal(
        self,
        *,
        branch_id: str,
        plan: RuntimeDayPlanRow,
        proposal: DayPlanProposal,
        virtual_now: datetime,
        mode: str,
    ) -> None:
        blocks = proposal.blocks
        from .tools.plan_validation import validate_plan_structure

        validate_plan_structure(proposal)
        requested_sources = {
            source_id for block in blocks for source_id in block.evidence_ids if source_id
        }
        validate_plan_evidence_sources(self.session, branch_id, requested_sources)
        if mode == "revision" and proposal.plan_date == virtual_now.date():
            current_time = virtual_now.strftime("%H:%M")
            existing_locked = [
                item for item in (plan.blocks or []) if str(item.get("start", "")) <= current_time
            ]
            proposed_locked = [block for block in blocks if block.start <= current_time]
            if len(existing_locked) != len(proposed_locked) or any(
                not _same_plan_block(old, new.model_dump(mode="json"))
                for old, new in zip(existing_locked, proposed_locked, strict=True)
            ):
                raise ValueError("计划修订不能改写已经开始的生活块")

    def _validate_plan_evidence_sources(self, branch_id: str, requested_sources: set[str]) -> None:
        validate_plan_evidence_sources(self.session, branch_id, requested_sources)

    def _branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self.session.scalar(
            select(Branch).where(Branch.id == branch_id, Branch.project_id == project_id)
        )
        if branch is None:
            raise LookupError("时间分支不存在")
        return branch

    def _clock(self, row: RuntimeClockRow) -> VirtualClock:
        return VirtualClock(
            branch_id=row.branch_id,
            virtual_anchor=row.virtual_anchor,
            wall_anchor=row.wall_anchor,
            time_scale=row.time_scale,
            status=cast(Any, row.status),
            timezone=row.timezone,
        )

    def _current_state(self, branch_id: str) -> RuntimeLifeStateRow | None:
        return self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id, RuntimeLifeStateRow.is_current.is_(True)
            )
        )

    def _validate_memory_sources(self, record: MemoryRecord) -> None:
        validate_memory_sources(self.session, record)

    @observed_stage("runtime.executor.validate_references")
    def validate_decision_references(self, branch_id, decision, now):
        """检查引用权限，不要求回复对象与完成的消息一一对应。"""
        return _check_decision_references(self.session, branch_id, decision, now)

    def _validate_state_patch_sources(
        self, *, branch_id: str, source_ids: list[str], now=None
    ) -> None:
        validate_state_patch_sources(
            self.session, branch_id=branch_id, source_ids=source_ids, now=now
        )


def _effective_valid_until(
    *,
    virtual_now: datetime,
    patch: dict[str, Any],
    requested: datetime | None,
    current: datetime | None,
) -> datetime | None:
    """给临时状态一个有限寿命；没有字段变化时保留当前状态的到期点。"""

    if requested is not None:
        return requested
    fields = set(patch)
    if fields & {"mood", "energy", "attention", "current_goal"}:
        return virtual_now + timedelta(hours=12)
    if fields & {"activity", "location_role", "social_context", "availability"}:
        return virtual_now + timedelta(hours=4)
    return current
