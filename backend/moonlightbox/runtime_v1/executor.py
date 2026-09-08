"""Executor：Runtime 唯一的副作用出口。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from moonlightbox.events.models import EventNode
from moonlightbox.personas.models import IdentityKernel
from moonlightbox.world.models import ConversationBundle, PersonWorldProfile, WorldGraphVersion

from .branch_models import Branch, BranchMessage
from .clock import create_clock
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
from .planning import generate_day_plan
from .schemas import ActorMessage, LifeDecision, LifeState, MemoryRecord, VirtualClock


class RuntimeWorldUnavailableError(RuntimeError):
    """运行时只能读取已完成的 LightRAG/PWP，不允许用空背景启动人物。"""


class RuntimeExecutor:
    """负责校验、幂等和事务写入，不调用模型也不理解自然语言。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def bootstrap(self, *, project_id: str, branch_id: str) -> dict[str, Any]:
        branch = self._branch(project_id, branch_id)
        # 导入记录一般以 UTC 保存，但 DayPlan 必须按人物对话时区解释“早上/睡觉”。
        # IdentityKernel 是现有训练流程产出的稳定读模型；没有时区证据时保留 UTC，
        # 不擅自把所有历史都当成东八区。
        runtime_anchor, runtime_timezone = _runtime_time_anchor(self.session, branch)
        snapshot = self.session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
        )
        if snapshot is None:
            profile = self.session.scalar(
                select(PersonWorldProfile)
                .join(
                    WorldGraphVersion, PersonWorldProfile.graph_version_id == WorldGraphVersion.id
                )
                .where(
                    PersonWorldProfile.project_id == project_id, WorldGraphVersion.status == "ready"
                )
                .order_by(PersonWorldProfile.created_at.desc())
            )
            if profile is None:
                raise RuntimeWorldUnavailableError("人物世界尚未就绪，不能初始化 Runtime 分支")
            # LightRAG 暂无按时间检索：按当前约定，忽略分支的历史节点，直接把
            # 最新导入对话末端和完整 ready 图谱作为 Snapshot 的唯一背景边界。
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
                routine_profile=profile.routine_summary,
                compiler_version="runtime-v1-latest-graph",
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
        plan = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id,
                RuntimeDayPlanRow.plan_date == clock.now().date().isoformat(),
            )
        )
        if plan is None:
            generated = generate_day_plan(branch_id, clock.now().date(), snapshot.routine_profile)
            plan = RuntimeDayPlanRow(
                branch_id=branch_id,
                plan_date=(generated.plan_date or generated.date or clock.now().date()).isoformat(),
                blocks=[item.model_dump(mode="json") for item in generated.blocks],
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
            current = _initial_state(branch_id, clock.now(), plan.blocks)
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
            _schedule_next_boundary(self.session, branch_id, clock.now(), plan.blocks)
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
        if decision.action == "speak" and (
            not decision.communication_intent or not decision.content_points
        ):
            raise ValueError("speak 必须提供 communication_intent 和 content_points")
        if decision.next_wakeup_at is not None and decision.next_wakeup_at <= virtual_now:
            raise ValueError("next_wakeup_at 必须晚于 virtual_now")
        if decision.action == "schedule" and decision.next_wakeup_at is None:
            raise ValueError("schedule 必须提供 next_wakeup_at")
        if decision.action != "speak" and decision.speech_mode is not None:
            raise ValueError("只有 speak 可以设置 speech_mode")
        if decision.plan_patch is not None:
            plan = self.session.scalar(
                select(RuntimeDayPlanRow).where(
                    RuntimeDayPlanRow.branch_id == branch_id,
                    RuntimeDayPlanRow.plan_date == virtual_now.date().isoformat(),
                )
            )
            target_id = decision.plan_patch.block_id
            if plan is None or not target_id:
                raise ValueError("plan_patch 必须指向当天已有的生活块")
            target = next(
                (item for item in (plan.blocks or []) if item.get("id") == target_id), None
            )
            if target is None:
                raise ValueError("plan_patch 指向的生活块不存在")
        patch = decision.state_patch.model_dump(exclude_none=True)
        if patch.get("activity") and patch.get("activity") != current.activity:
            # 跨越默认生活块的活动变化必须有局部计划覆盖，或引用已经落库的分支事件。
            if decision.plan_patch is None and not decision.state_patch.source_event_ids:
                raise ValueError("改变 activity 必须提供 plan_patch 或 source_event_ids")

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
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.validate(
            branch_id=branch_id,
            decision=decision,
            expected_version=expected_version,
            virtual_now=virtual_now,
        )
        if decision.action == "speak" and actor_message is None:
            raise ValueError("speak 决定必须经过 PersonaActor 生成待提交消息")
        if actor_message is not None and decision.action != "speak":
            raise ValueError("非 speak 决定不能提交 PersonaActor 消息")
        key = idempotency_key or f"cycle:{branch_id}:{','.join(sorted(trigger_event_ids))}"
        existing = self.session.scalar(
            select(RuntimeLifeEventRow).where(RuntimeLifeEventRow.idempotency_key == key)
        )
        if existing is not None:
            return {"event": existing, "message": None, "state": self._current_state(branch_id)}
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
        state: RuntimeLifeStateRow = current
        if (
            patch
            or decision.plan_patch is not None
            or decision.action in {"continue_life", "schedule"}
        ):
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
        if decision.plan_patch is not None:
            plan = self.session.scalar(
                select(RuntimeDayPlanRow).where(
                    RuntimeDayPlanRow.branch_id == branch_id,
                    RuntimeDayPlanRow.plan_date == virtual_now.date().isoformat(),
                )
            )
            if plan is not None:
                blocks = [dict(item) for item in (plan.blocks or [])]
                target_id = decision.plan_patch.block_id
                target = next((item for item in blocks if item.get("id") == target_id), None)
                if target is not None:
                    target.update(
                        decision.plan_patch.model_dump(exclude_none=True, exclude={"block_id"})
                    )
                    plan.blocks = blocks
        life_event = RuntimeLifeEventRow(
            branch_id=branch_id,
            event_type="decision",
            occurred_at=virtual_now,
            payload=decision.model_dump(mode="json"),
            idempotency_key=key,
        )
        self.session.add(life_event)
        message = None
        if actor_message is not None:
            sequence = self.session.scalar(
                select(func.max(BranchMessage.sequence)).where(BranchMessage.branch_id == branch_id)
            )
            message = BranchMessage(
                branch_id=branch_id,
                sequence=(int(sequence) + 1 if sequence is not None else 0),
                role="assistant",
                content=actor_message.text,
                type="text",
                generation_status="completed",
                generation_metadata={"runtime_v1": True, "bubbles": actor_message.bubbles},
                observed_at=virtual_now,
                is_proactive=decision.speech_mode == "proactive",
            )
            self.session.add(message)
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
        return {"event": life_event, "message": message, "state": state}

    def schedule_wakeup(
        self,
        *,
        branch_id: str,
        wake_at: datetime,
        reason: str,
        trigger_type: str,
        idempotency_key: str,
    ) -> RuntimeWakeupRow:
        existing = self.session.scalar(
            select(RuntimeWakeupRow).where(
                RuntimeWakeupRow.branch_id == branch_id,
                RuntimeWakeupRow.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
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

        result = self.session.execute(
            update(RuntimeLifeStateRow)
            .where(
                RuntimeLifeStateRow.id == current.id,
                RuntimeLifeStateRow.branch_id == current.branch_id,
                RuntimeLifeStateRow.version == expected_version,
                RuntimeLifeStateRow.is_current.is_(True),
            )
            .values(is_current=False)
            .execution_options(synchronize_session=False)
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
        """确认记忆前核对来源归属，模型或客户端不能伪造跨分支事实。"""

        source_ids = set(record.source_ids)
        if not source_ids:
            raise ValueError("确认记忆必须携带至少一个 source_id")
        if record.scope == "branch":
            assert record.branch_id is not None
            message_ids = set(
                self.session.scalars(
                    select(BranchMessage.id).where(
                        BranchMessage.branch_id == record.branch_id,
                        BranchMessage.id.in_(source_ids),
                    )
                )
            )
            event_ids = set(
                self.session.scalars(
                    select(RuntimeLifeEventRow.id).where(
                        RuntimeLifeEventRow.branch_id == record.branch_id,
                        RuntimeLifeEventRow.id.in_(source_ids),
                    )
                )
            )
            if source_ids - message_ids - event_ids:
                raise ValueError("branch 记忆含有不属于当前分支的来源")
            return
        assert record.snapshot_id is not None
        snapshot = self.session.get(RuntimeSnapshotRow, record.snapshot_id)
        if snapshot is None:
            raise ValueError("world 记忆引用的快照不存在")
        # world 证据只能来自冻结快照已记录的导入消息，不能引用其他项目的资料。
        allowed = set(snapshot.source_message_ids or [])
        if not source_ids.issubset(allowed):
            raise ValueError("world 记忆含有不属于快照的来源")

    def _validate_state_patch_sources(self, *, branch_id: str, source_ids: list[str]) -> None:
        """StatePatch 可以引用本轮或已落库分支事件，但绝不能越过分支边界。"""

        requested = set(source_ids)
        if not requested:
            return
        known: set[str] = set(
            self.session.scalars(
                select(BranchMessage.id).where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.id.in_(requested),
                )
            )
        )
        known.update(
            self.session.scalars(
                select(RuntimeEventRow.id).where(
                    RuntimeEventRow.branch_id == branch_id,
                    RuntimeEventRow.id.in_(requested),
                )
            )
        )
        known.update(
            self.session.scalars(
                select(RuntimeLifeEventRow.id).where(
                    RuntimeLifeEventRow.branch_id == branch_id,
                    RuntimeLifeEventRow.id.in_(requested),
                )
            )
        )
        if requested - known:
            raise ValueError("state_patch 含有不属于当前分支的来源")


def _latest_profile_cutoff(
    session: Session, profile: PersonWorldProfile | None, fallback: datetime
) -> datetime:
    if profile is None:
        return fallback
    latest = session.scalar(
        select(func.max(ConversationBundle.ended_at)).where(
            ConversationBundle.graph_version_id == profile.graph_version_id
        )
    )
    return latest or profile.created_at or fallback


def _latest_import_node(session: Session, project_id: str) -> str | None:
    """记录当前导入末端节点，而非分支原节点，匹配 v1 的最新完整图谱约定。"""

    sort_time = func.coalesce(EventNode.ended_at, EventNode.started_at, EventNode.created_at)
    return session.scalar(
        select(EventNode.id)
        .where(EventNode.project_id == project_id)
        .order_by(sort_time.desc(), EventNode.id.desc())
        .limit(1)
    )


def _runtime_time_anchor(session: Session, branch: Branch) -> tuple[datetime, str]:
    """从已冻结的人格行为节律取得分支本地时钟，缺失时安全退回 UTC。"""

    origin = branch.origin_time
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=UTC)
    kernel = session.scalar(
        select(IdentityKernel).where(IdentityKernel.model_version_id == branch.model_version_id)
    )
    rhythm = (
        kernel.content.get("behavioral_rhythm", {})
        if kernel is not None and isinstance(kernel.content, dict)
        else {}
    )
    raw_offset = rhythm.get("timezone_offset_minutes") if isinstance(rhythm, dict) else None
    if (
        not isinstance(raw_offset, int)
        or isinstance(raw_offset, bool)
        or not -720 <= raw_offset <= 840
    ):
        return origin.astimezone(UTC), "UTC"
    local_timezone = timezone(timedelta(minutes=raw_offset))
    sign = "+" if raw_offset >= 0 else "-"
    hours, minutes = divmod(abs(raw_offset), 60)
    return (
        origin.astimezone(local_timezone),
        f"UTC{sign}{hours:02d}:{minutes:02d}",
    )


def _profile_payload(profile: PersonWorldProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    return {
        "identity": profile.identity,
        "work_and_education": profile.work_and_education,
        "places": profile.places,
        "social_relationships": profile.social_relationships,
        "preferences": profile.preferences,
        "recurring_activities": profile.recurring_activities,
        "routine_summary": profile.routine_summary,
        "life_phases": profile.life_phases,
        "relationship_with_user": profile.relationship_with_user,
        "important_events": profile.important_events,
    }


def _seed_world_memory(session: Session, snapshot: RuntimeSnapshotRow) -> None:
    """把人物档案编译为统一 world 记忆，供 search_memory 只读检索。"""
    for field in (
        "identity",
        "work_and_education",
        "places",
        "social_relationships",
        "preferences",
        "recurring_activities",
        "routine_summary",
        "relationship_with_user",
        "important_events",
        "life_phases",
    ):
        for summary, source_ids in _profile_statements(snapshot.profile.get(field)):
            session.add(
                RuntimeMemoryRow(
                    id=str(uuid4()),
                    scope="world",
                    snapshot_id=snapshot.id,
                    subject="目标人物",
                    predicate=field,
                    object=str(summary)[:500],
                    summary=str(summary)[:2000],
                    status="confirmed",
                    source_ids=source_ids,
                    confidence=1.0,
                )
            )


def _profile_statements(value: object) -> list[tuple[str, list[str]]]:
    """从编译档案提取最小可检索陈述，保留每条原始 message ID。"""

    if isinstance(value, dict):
        text = value.get("text") or value.get("summary")
        if isinstance(text, str) and text.strip():
            sources = value.get("source_message_ids", [])
            return [
                (
                    text.strip(),
                    [item for item in sources if isinstance(item, str)]
                    if isinstance(sources, list)
                    else [],
                )
            ]
        statements: list[tuple[str, list[str]]] = []
        for nested in value.values():
            statements.extend(_profile_statements(nested))
        return statements
    if isinstance(value, list):
        nested_statements: list[tuple[str, list[str]]] = []
        for nested in value:
            nested_statements.extend(_profile_statements(nested))
        return nested_statements
    if isinstance(value, str) and value.strip():
        return [(value.strip(), [])]
    return []


def _initial_state(branch_id: str, now: datetime, blocks: list[dict[str, Any]]) -> LifeState:
    current = next(
        (
            block
            for block in blocks
            if block.get("start", "") <= now.strftime("%H:%M") < block.get("end", "")
        ),
        None,
    )
    return LifeState(
        branch_id=branch_id,
        virtual_now=now,
        current_plan_block_id=current.get("id") if current else None,
        activity=current.get("activity") if current else None,
        location_role=current.get("location_role") if current else None,
        availability=current.get("default_availability", "unknown") if current else "unknown",
        last_transition_at=now,
        valid_until=_block_end(now, current) if current else None,
        reason="initial",
    )


def _schedule_next_boundary(
    session: Session, branch_id: str, now: datetime, blocks: list[dict[str, Any]]
) -> None:
    next_block = next(
        (block for block in blocks if block.get("start", "") > now.strftime("%H:%M")), None
    )
    if next_block is None:
        wake_at = (now + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
    else:
        wake_at = now.replace(
            hour=int(next_block["start"][:2]),
            minute=int(next_block["start"][3:]),
            second=0,
            microsecond=0,
        )
    RuntimeExecutor(session).schedule_wakeup(
        branch_id=branch_id,
        wake_at=wake_at,
        reason="DayPlan 生活块边界",
        trigger_type="plan_transition",
        idempotency_key=f"plan:{branch_id}:{wake_at.isoformat()}",
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


def _block_end(now: datetime, block: dict[str, Any] | None) -> datetime | None:
    if block is None or not isinstance(block.get("end"), str):
        return None
    try:
        hour, minute = (int(part) for part in block["end"].split(":", 1))
    except (TypeError, ValueError):
        return None
    if hour == 24 and minute == 0:
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
