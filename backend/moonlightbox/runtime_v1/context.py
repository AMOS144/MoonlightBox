"""Runtime ContextAssembler：把数据库对象整理成稳定、可审计的 ContextPacket。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.personas.models import IdentityKernel

from .branch_models import Branch, BranchMessage
from .db_models import (
    RuntimeContextSummaryRow,
    RuntimeDayPlanRow,
    RuntimeLifeEventRow,
    RuntimeLifeStateRow,
    RuntimeMemoryIndexRow,
    RuntimeSnapshotRow,
)
from .schemas import ContextPacket, VirtualClock


class ContextAssembler:
    """只组装领域上下文；容量准入和压缩由统一 Controller 负责。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def assemble(
        self,
        *,
        branch_id: str,
        trigger: dict[str, Any],
        clock: VirtualClock,
        snapshot: RuntimeSnapshotRow,
        now: datetime | None = None,
    ) -> ContextPacket:
        from .collaboration.plans import local_time, shared_plan_window

        virtual_now = local_time(now or clock.now(), clock.timezone)
        state_row = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
        )
        state = (
            state_row.state
            if state_row is not None
            else {
                "branch_id": branch_id,
                "virtual_now": virtual_now.isoformat(),
                "last_transition_at": virtual_now.isoformat(),
                "version": 0,
                "availability": "unknown",
                "energy": "unknown",
            }
        )
        plan_row = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id,
                RuntimeDayPlanRow.plan_date == virtual_now.date().isoformat(),
            )
        )
        blocks = list(plan_row.blocks or []) if plan_row is not None else []
        current_block, next_blocks = _select_blocks(blocks, virtual_now)
        messages = self._working_window(
            branch_id,
            virtual_now,
            active_threads=state.get("open_conversation_threads", []),
        )
        events = list(
            self.session.scalars(
                select(RuntimeLifeEventRow)
                .where(RuntimeLifeEventRow.branch_id == branch_id)
                .order_by(RuntimeLifeEventRow.occurred_at.desc())
                .limit(20)
            )
        )
        branch = self.session.get(Branch, branch_id)
        from .conversation import imported_prefix

        prefix = imported_prefix(self.session, snapshot) if len(messages) < 100 else []
        kernel = (
            self.session.scalar(
                select(IdentityKernel).where(
                    IdentityKernel.model_version_id == branch.model_version_id
                )
            )
            if branch is not None
            else None
        )
        style_profile = (
            kernel.content.get("style_profile", {})
            if kernel is not None and isinstance(kernel.content, dict)
            else {}
        )
        memory_index = self.session.scalar(
            select(RuntimeMemoryIndexRow).where(
                RuntimeMemoryIndexRow.branch_id == branch_id,
                RuntimeMemoryIndexRow.is_current.is_(True),
            )
        )
        packet = ContextPacket(
            generated_at=datetime.now(UTC),
            virtual_now=virtual_now,
            timezone=clock.timezone,
            trigger=trigger,
            origin={
                "cutoff_at": snapshot.cutoff_at,
                "snapshot_id": snapshot.id,
                # 只放稳定身份/关系/规律投影，完整档案仍是只读快照，不会每轮塞进 Prompt。
                "person_world_profile": _origin_projection(snapshot.profile),
                "routine_profile": snapshot.routine_profile,
            },
            current={
                "life_state": state,
                "day_plans": shared_plan_window(
                    self.session, branch_id, virtual_now, clock.timezone
                ),
                "day_plan": {
                    "date": virtual_now.date().isoformat(),
                    "current_block": current_block,
                    "next_blocks": next_blocks[:2],
                },
                "open_commitments": state.get("active_commitments", []),
                "open_conversation_threads": state.get("open_conversation_threads", []),
                "expression_style_profile": style_profile,
            },
            branch={
                "branch_id": branch_id,
                "working_window": {
                    "from_sequence": messages[0]["sequence"] if messages else None,
                    "to_sequence": messages[-1]["sequence"] if messages else None,
                    "messages": messages,
                    "imported_prefix": prefix,
                    "pending_message_refs": [
                        item["source_id"]
                        for item in messages
                        if item.get("input_status") in {"pending", "awaiting_response"}
                    ],
                    "active_thread_ids": state.get("active_thread_ids", []),
                    "unresolved_items": state.get("unresolved_items", []),
                },
                "recent_events": [
                    {
                        "id": event.id,
                        "type": event.event_type,
                        "occurred_at": (
                            event.occurred_at
                            if event.occurred_at.tzinfo
                            else event.occurred_at.replace(tzinfo=UTC)
                        ),
                        "payload": event.payload,
                    }
                    for event in events
                ],
                "overlay_summary": [
                    {
                        "text": row.text,
                        "source_refs": row.source_message_ids,
                        "covered_until": row.covered_until.isoformat(),
                    }
                    for row in self.session.scalars(
                        select(RuntimeContextSummaryRow)
                        .where(
                            RuntimeContextSummaryRow.branch_id == branch_id,
                            RuntimeContextSummaryRow.covered_until <= virtual_now,
                        )
                        .order_by(RuntimeContextSummaryRow.covered_until.desc())
                        .limit(8)
                    )
                ],
            },
            memory={
                "index_version_id": memory_index.id if memory_index is not None else None,
                "index_version": memory_index.version if memory_index is not None else 0,
                "retrieved_records": [],
            },
            budgets={"input_tokens": 0, "compressed": False, "compression_round": 0},
        )
        return packet

    def _working_window(
        self,
        branch_id: str,
        virtual_now: datetime,
        *,
        active_threads: object,
    ) -> list[dict[str, Any]]:
        """取连续近期原文并保护待处理输入；时间间隔不代表话题结束。"""

        observed = func.coalesce(BranchMessage.observed_at, BranchMessage.created_at)
        rows = list(
            self.session.scalars(
                select(BranchMessage)
                .where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.generation_status != "failed",
                    observed <= virtual_now,
                )
                .order_by(BranchMessage.sequence.desc())
                .limit(100)
            )
        )
        # 未处理输入不受普通近期窗口约束；状态由 Executor 与回复事务一起提交。
        pending = list(
            self.session.scalars(
                select(BranchMessage).where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.role == "user",
                    BranchMessage.generation_status != "failed",
                    observed <= virtual_now,
                )
            )
        )
        covered = {
            ref
            for summary in self.session.scalars(
                select(RuntimeContextSummaryRow).where(
                    RuntimeContextSummaryRow.branch_id == branch_id,
                    RuntimeContextSummaryRow.created_by == "director-summary-v2",
                    RuntimeContextSummaryRow.covered_until <= virtual_now,
                )
            )
            for ref in summary.source_message_ids
        }
        # 未被成功摘要覆盖的旧原文仍需可见，不能因为后台任务失败而静默跳过。
        pending.extend(
            self.session.scalars(
                select(BranchMessage).where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.generation_status != "failed",
                    observed <= virtual_now,
                )
            )
        )
        seen_ids = {row.id for row in rows}
        for row in pending:
            if row.id not in seen_ids and (
                row.id not in covered
                or (row.generation_metadata or {}).get("input_status")
                in {"pending", "awaiting_response"}
            ):
                rows.append(row)
                seen_ids.add(row.id)
        thread_ids = _active_thread_ids(active_threads)
        if thread_ids:
            extra = list(
                self.session.scalars(
                    select(BranchMessage)
                    .where(
                        BranchMessage.branch_id == branch_id,
                        BranchMessage.turn_id.in_(thread_ids),
                        BranchMessage.generation_status != "failed",
                    )
                    .order_by(BranchMessage.sequence.desc())
                    .limit(6)
                )
            )
            seen = {row.id for row in rows}
            rows.extend(row for row in extra if row.id not in seen)
        # 活动话题最多额外六条；这是文档定义的例外，不能被普通 16 条窗口截断。
        rows = sorted(rows, key=lambda item: item.sequence)
        return [
            {
                "source_id": row.id,
                "sequence": row.sequence,
                "role": row.role,
                "content": row.content,
                "type": row.type,
                "asset_ref": row.media_asset_id,
                "usage_refs": (row.generation_metadata or {}).get("sticker_usage_refs", []),
                "occurred_at": (
                    (row.observed_at or row.created_at)
                    if (row.observed_at or row.created_at).tzinfo
                    else (row.observed_at or row.created_at).replace(tzinfo=UTC)
                ),
                "input_status": (row.generation_metadata or {}).get("input_status", "legacy")
                if row.role == "user"
                else None,
            }
            for row in rows
        ]


def _select_blocks(
    blocks: list[dict[str, Any]], at: datetime
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    local = at.strftime("%H:%M")
    current = None
    future: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("start", "") <= local < block.get("end", ""):
            current = block
        elif block.get("start", "") > local:
            future.append(block)
    return current, future


def _active_thread_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    thread_ids: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw = item.get("turn_id") or item.get("thread_id")
        if isinstance(raw, str) and raw:
            thread_ids.append(raw)
    return list(dict.fromkeys(thread_ids))


def _origin_projection(profile: object) -> dict[str, object]:
    """保持 ContextPacket 固定区域，防止全量 PWP 挤占当前触发上下文。"""

    if not isinstance(profile, dict):
        return {}
    if profile.get("profile_schema_version") == "v3":
        from moonlightbox.world.person_world.contracts.world_dimensions import SECTION_DIMENSIONS

        return {
            key: profile[key]
            for key in ("profile_schema_version", "overview", *SECTION_DIMENSIONS)
            if key in profile
        }
    keys = (
        "profile_schema_version",
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
    )
    return {key: profile[key] for key in keys if key in profile}


def _latest_occurred_at(values: list[object]) -> datetime:
    """兼容 JSON 往返后的 ISO 时间；无法解析时只使用当前审计时间。"""

    parsed: list[datetime] = []
    for value in values:
        if isinstance(value, datetime):
            parsed.append(value)
        elif isinstance(value, str):
            try:
                parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
            except ValueError:
                continue
    return max(parsed, default=datetime.now(UTC))
