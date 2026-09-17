"""DayPlan 块的运行时辅助：初始状态、边界唤醒与计划证据校验。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .branch_models import BranchMessage
from .db_models import RuntimeDayPlanRow, RuntimeEventRow, RuntimeLifeEventRow, RuntimeSnapshotRow
from .schemas import LifeState


def validate_plan_evidence_sources(
    session: Session, branch_id: str, requested_sources: set[str]
) -> None:
    if not requested_sources:
        return
    snapshot = session.scalar(
        select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
    )
    from .snapshot_sources import frozen_source_ids

    allowed = frozen_source_ids(session, snapshot) if snapshot is not None else set()
    allowed.update(
        session.scalars(
            select(BranchMessage.id).where(
                BranchMessage.branch_id == branch_id,
                BranchMessage.id.in_(requested_sources),
            )
        )
    )
    allowed.update(
        session.scalars(
            select(RuntimeEventRow.id).where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.id.in_(requested_sources),
            )
        )
    )
    allowed.update(
        session.scalars(
            select(RuntimeLifeEventRow.id).where(
                RuntimeLifeEventRow.branch_id == branch_id,
                RuntimeLifeEventRow.id.in_(requested_sources),
            )
        )
    )
    if requested_sources - allowed:
        raise ValueError("DayPlanProposal 含有不属于当前分支或冻结 Snapshot 的证据")


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
    # 没有经过 DayPlanAgent 验收的块时，不能偷偷安排一个固定的次日 07:00 唤醒。
    # 等 Agent 正式提交计划后，此函数会由 Executor 再次调用。
    next_block = next(
        (block for block in blocks if block.get("start", "") > now.strftime("%H:%M")), None
    )
    if next_block is None:
        tomorrow = (now + timedelta(days=1)).date().isoformat()
        next_plan = session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id, RuntimeDayPlanRow.plan_date == tomorrow
            )
        )
        if (
            not next_plan
            or not next_plan.blocks
            or (next_plan.generation_metadata or {}).get("status") != "agent"
        ):
            return  # 未就绪由独立准备扫描恢复，不臆造默认起床时刻。
        first = next_plan.blocks[0]["start"]
        wake_at = (now + timedelta(days=1)).replace(
            hour=int(first[:2]), minute=int(first[3:]), second=0, microsecond=0
        )
    else:
        wake_at = now.replace(
            hour=int(next_block["start"][:2]),
            minute=int(next_block["start"][3:]),
            second=0,
            microsecond=0,
        )
    from .executor import RuntimeExecutor

    RuntimeExecutor(session).schedule_wakeup(
        branch_id=branch_id,
        wake_at=wake_at,
        reason="DayPlan 生活块边界",
        trigger_type="plan_transition",
        idempotency_key=f"plan:{branch_id}:{wake_at.isoformat()}",
        revive_cancelled=True,
    )


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


def _minutes(value: str) -> int:
    """将已经过 schema 校验的 HH:MM 转为日内分钟，24:00 合法。"""

    hour, minute = (int(part) for part in value.split(":", 1))
    return hour * 60 + minute


def _same_plan_block(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """修订时只比较运行时语义字段；数据库 ID 和审计证据允许自然更新。"""

    fields = ("start", "end", "activity", "location_role", "default_availability")
    return all(old.get(field) == new.get(field) for field in fields)


def _reused_block_id(existing: list[dict[str, Any]], proposed: dict[str, Any]) -> str | None:
    for block in existing:
        block_id = block.get("id")
        if isinstance(block_id, str) and _same_plan_block(block, proposed):
            return block_id
    return None
