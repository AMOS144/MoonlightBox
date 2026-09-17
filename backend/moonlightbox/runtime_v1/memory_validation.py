"""记忆与决定引用的来源边界校验；模型或客户端不能伪造跨分支事实。"""

from __future__ import annotations

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import Session

from .branch_models import BranchMessage
from .db_models import (
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from .schemas import MemoryRecord


def validate_memory_sources(session: Session, record: MemoryRecord) -> None:
    """确认记忆前核对来源归属，模型或客户端不能伪造跨分支事实。"""

    source_ids = set(record.source_ids)
    if not source_ids:
        raise ValueError("确认记忆必须携带至少一个 source_id")
    if record.scope == "branch":
        assert record.branch_id is not None
        message_ids = set(
            session.scalars(
                select(BranchMessage.id).where(
                    BranchMessage.branch_id == record.branch_id,
                    BranchMessage.id.in_(source_ids),
                )
            )
        )
        event_ids = set(
            session.scalars(
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
    snapshot = session.get(RuntimeSnapshotRow, record.snapshot_id)
    if snapshot is None:
        raise ValueError("world 记忆引用的快照不存在")
    # world 证据只能来自冻结快照已记录的导入消息，不能引用其他项目的资料。
    allowed = set(snapshot.source_message_ids or [])
    if not source_ids.issubset(allowed):
        raise ValueError("world 记忆含有不属于快照的来源")


def validate_decision_references(session: Session, branch_id, decision, now):
    """检查引用权限，不要求回复对象与完成的消息一一对应。"""
    from .snapshot_sources import frozen_source_ids
    from .subjective_state import visible_sources

    try:
        validate_state_patch_sources(
            session, branch_id=branch_id, source_ids=decision.state_patch.source_event_ids, now=now
        )
        seen = set()
        for item in decision.input_resolutions:
            if item.message_ref in seen:
                raise ValueError("input_resolutions 含有重复 message_ref")
            seen.add(item.message_ref)
            row = session.get(BranchMessage, item.message_ref)
            if row is None or row.branch_id != branch_id or row.role != "user":
                raise ValueError(f"input_resolutions 无效用户消息引用：{item.message_ref}")
            occurred = row.observed_at or row.created_at
            if occurred.tzinfo is None:
                occurred = occurred.replace(tzinfo=UTC)
            if occurred > now:
                raise ValueError("input_resolutions 不能处理未来消息")
            old = (row.generation_metadata or {}).get("input_status")
            if old in {"completed", "no_response_needed"} and old != item.status:
                raise ValueError(f"消息已经处理完成：{item.message_ref}")
        if decision.expression_task:
            snapshot = session.scalar(
                select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
            )
            messages = set(
                session.scalars(
                    select(BranchMessage.id).where(
                        BranchMessage.branch_id == branch_id,
                        BranchMessage.generation_status != "failed",
                    )
                )
            )
            if snapshot:
                messages.update(frozen_source_ids(session, snapshot))
            allowed = messages & visible_sources(session, branch_id, now)
            invalid = set(decision.expression_task.respond_to_refs) - allowed
            if invalid:
                raise ValueError(
                    "expression_task.respond_to_refs 无效消息引用：" + ", ".join(sorted(invalid))
                )
    except ValueError as error:
        return str(error)
    return None


def validate_state_patch_sources(
    session: Session, *, branch_id: str, source_ids: list[str], now=None
) -> None:
    """StatePatch 可以引用本轮或已落库分支事件，但绝不能越过分支边界。"""

    requested = set(source_ids)
    if not requested:
        return
    known: set[str] = set(
        session.scalars(
            select(BranchMessage.id).where(
                BranchMessage.branch_id == branch_id,
                BranchMessage.id.in_(requested),
            )
        )
    )
    known.update(
        session.scalars(
            select(RuntimeEventRow.id).where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.id.in_(requested),
            )
        )
    )
    known.update(
        session.scalars(
            select(RuntimeLifeEventRow.id).where(
                RuntimeLifeEventRow.branch_id == branch_id,
                RuntimeLifeEventRow.id.in_(requested),
            )
        )
    )
    known.update(
        session.scalars(
            select(RuntimeWakeupRow.id).where(
                RuntimeWakeupRow.branch_id == branch_id,
                RuntimeWakeupRow.id.in_(requested),
                RuntimeWakeupRow.status != "cancelled",
            )
        )
    )
    if now is not None:
        from .subjective_state import visible_sources

        known.intersection_update(visible_sources(session, branch_id, now))
    if requested - known:
        raise ValueError("state_patch 含有不属于当前分支的来源")
