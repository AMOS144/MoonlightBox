"""处境状态提交辅助函数；不定义事件到情绪的规则，不创建独立心理事实库。"""

from copy import deepcopy
from uuid import uuid4

from sqlalchemy import select

from .branch_models import BranchMessage
from .db_models import RuntimeEventRow, RuntimeLifeEventRow, RuntimeMemoryRow, RuntimeWakeupRow


def visible_sources(session, branch_id, now):
    """语义来源限定在本分支、当前虚拟时间之前；不允许跨分支引用。"""
    from sqlalchemy import func

    messages = session.scalars(
        select(BranchMessage.id).where(
            BranchMessage.branch_id == branch_id,
            BranchMessage.generation_status != "failed",
            func.coalesce(BranchMessage.observed_at, BranchMessage.created_at) <= now,
        )
    )
    events = session.scalars(
        select(RuntimeLifeEventRow.id).where(
            RuntimeLifeEventRow.branch_id == branch_id,
            RuntimeLifeEventRow.occurred_at <= now,
        )
    )
    triggers = session.scalars(
        select(RuntimeEventRow.id).where(
            RuntimeEventRow.branch_id == branch_id,
            RuntimeEventRow.occurred_at <= now,
        )
    )
    from .db_models import RuntimeSnapshotRow
    from .snapshot_sources import frozen_source_ids

    snapshot = session.scalar(
        select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
    )
    historical = frozen_source_ids(session, snapshot) if snapshot else set()
    wakeups = session.scalars(
        select(RuntimeWakeupRow.id).where(
            RuntimeWakeupRow.branch_id == branch_id,
            RuntimeWakeupRow.wake_at <= now,
            RuntimeWakeupRow.status != "cancelled",
        )
    )
    return set(messages) | set(events) | set(triggers) | set(wakeups) | historical


def apply_update(old, update, allowed, now, *, decision_ref=None, has_expression=False):
    result = deepcopy(old or {"mood": None, "concerns": []})
    if update.mood is not None:
        result["mood"] = update.mood
    concerns = {item["concern_ref"]: item for item in result.get("concerns", [])}
    for item in update.concerns:
        if item.depends_on_expression and not has_expression:
            raise ValueError("concern_requires_expression")
        action_ref = decision_ref if item.action_ref == "this_decision" else item.action_ref
        if action_ref and action_ref != decision_ref and action_ref not in allowed:
            raise ValueError("concern_action_out_of_scope")
        if not set(item.source_refs).issubset(allowed):
            raise ValueError("subjective_source_out_of_scope")
        if item.concern_ref is not None and item.concern_ref not in concerns:
            raise ValueError("unknown_concern_ref")
        ref = item.concern_ref or str(uuid4())
        concerns[ref] = {
            **item.model_dump(mode="json"),
            "concern_ref": ref,
            "action_ref": action_ref,
            "updated_at": now.isoformat(),
        }
    result["concerns"] = list(concerns.values())
    result["updated_at"] = now.isoformat()
    return result


def commit_memories(session, branch_id, proposals, allowed, now, has_expression):
    """候选只进入分支账本；推断不伪装成用户确认，不改原始图谱。"""
    for proposal in proposals:
        if not set(proposal.source_refs).issubset(allowed):
            raise ValueError("memory_source_out_of_scope")
        if proposal.depends_on_expression and not has_expression:
            raise ValueError("memory_requires_expression")
        if proposal.supersedes_ref:
            prior = session.get(RuntimeMemoryRow, proposal.supersedes_ref)
            if prior is None or prior.branch_id != branch_id or prior.scope != "branch":
                raise ValueError("memory_supersession_out_of_scope")
            if prior.status not in {"asserted", "confirmed"}:
                raise ValueError("memory_already_superseded")
            prior.status = "superseded"
        session.add(
            RuntimeMemoryRow(
                branch_id=branch_id,
                scope="branch",
                subject=proposal.subject,
                predicate=proposal.basis,
                object="",
                summary=proposal.summary,
                status="asserted",
                confidence=1.0,
                source_ids=proposal.source_refs,
                valid_from=now,
                supersedes_id=proposal.supersedes_ref,
            )
        )
    if proposals:
        from .memory import MemoryService

        session.flush()
        MemoryService(session).rebuild_index(branch_id, reason="director_memory_commit")


def resolve_inputs(session, branch_id, resolutions, now):
    from datetime import UTC

    for resolution in resolutions:
        row = session.get(BranchMessage, resolution.message_ref)
        if row is None or row.branch_id != branch_id or row.role != "user":
            raise ValueError("input_resolution_out_of_scope")
        occurred = row.observed_at or row.created_at
        old_status = (row.generation_metadata or {}).get("input_status", "legacy")
        if old_status in {"completed", "no_response_needed"} and old_status != resolution.status:
            raise ValueError("input_already_resolved")
        if (
            occurred.replace(tzinfo=UTC) > now.astimezone(UTC)
            if occurred.tzinfo is None
            else occurred > now
        ):
            raise ValueError("input_resolution_from_future")
        row.generation_metadata = {
            **(row.generation_metadata or {}),
            "input_status": resolution.status,
            "processed_at": now.isoformat(),
        }
