"""Runtime ContextAssembler：把数据库对象整理成稳定、可审计的 ContextPacket。"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import IdentityKernel
from moonlightbox.branches.models import Branch, BranchMessage

from .config import (
    DIRECTOR_SYSTEM_PROMPT,
    INPUT_HARD_LIMIT,
    INPUT_SOFT_LIMIT,
    INPUT_TARGET_AFTER_COMPACT,
    WORKING_WINDOW_HOURS,
    WORKING_WINDOW_MESSAGES,
)
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
    """只负责读数据和预算裁剪，不替模型决定事实。"""

    def __init__(
        self, session: Session, *, token_counter: Callable[[str], int] | None = None
    ) -> None:
        self.session = session
        # 生产路径由 Linux 人格服务提供当前 Director 模型的 tokenizer。不能再用
        # 字符数估计，否则中英文混排和工具结果会绕过真实上下文窗口。
        self.token_counter = token_counter

    def assemble(
        self,
        *,
        branch_id: str,
        trigger: dict[str, Any],
        clock: VirtualClock,
        snapshot: RuntimeSnapshotRow,
        now: datetime | None = None,
    ) -> ContextPacket:
        virtual_now = now or clock.now()
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
                    "active_thread_ids": state.get("active_thread_ids", []),
                    "unresolved_items": state.get("unresolved_items", []),
                },
                "recent_events": [
                    {
                        "id": event.id,
                        "type": event.event_type,
                        "occurred_at": event.occurred_at,
                        "payload": event.payload,
                    }
                    for event in events
                ],
                "overlay_summary": None,
            },
            memory={
                "index_version_id": memory_index.id if memory_index is not None else None,
                "index_version": memory_index.version if memory_index is not None else 0,
                "retrieved_records": [],
            },
            budgets={"input_tokens": 0, "compressed": False, "compression_round": 0},
        )
        return self._fit_budget(packet)

    def _working_window(
        self,
        branch_id: str,
        virtual_now: datetime,
        *,
        active_threads: object,
    ) -> list[dict[str, Any]]:
        """按虚拟时间取最近窗口，并补回未关闭话题的少量上下文。"""

        start = virtual_now - timedelta(hours=WORKING_WINDOW_HOURS)
        observed = func.coalesce(BranchMessage.observed_at, BranchMessage.created_at)
        rows = list(
            self.session.scalars(
                select(BranchMessage)
                .where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.generation_status != "failed",
                    observed >= start,
                    observed <= virtual_now,
                )
                .order_by(BranchMessage.sequence.desc())
                .limit(WORKING_WINDOW_MESSAGES)
            )
        )
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
                "occurred_at": row.observed_at or row.created_at,
            }
            for row in rows
        ]

    def _fit_budget(self, packet: ContextPacket) -> ContextPacket:
        payload = packet.model_dump(mode="json")
        used = self._count_payload(payload)
        if used is None:
            packet.budgets = {
                "input_tokens": 0,
                "compressed": False,
                "compression_round": 0,
                "budget_exceeded": True,
                "tokenizer_unavailable": True,
            }
            return packet
        if used <= INPUT_SOFT_LIMIT:
            packet.budgets = {
                "input_tokens": used,
                "compressed": False,
                "compression_round": 0,
                "token_count_source": "director_tokenizer",
            }
            return packet

        # 压缩完全由代码完成。以下字段是本轮安全边界，绝不能为了节省 token 被删掉：
        # trigger、LifeState、当前 block、未完成承诺、开放话题最后消息和最近六轮对话。
        branch = payload["branch"]
        messages = branch["working_window"].get("messages", [])
        dropped = messages[:-6]
        branch["working_window"]["messages"] = messages[-6:]
        branch["overlay_summary"] = self._persist_summary(
            branch_id=str(branch.get("branch_id", "")),
            messages=dropped,
            compression_round=1,
        )
        round_count = 1
        used = self._count_payload(payload)
        if used is None:
            packet.budgets = {
                "input_tokens": 0,
                "compressed": True,
                "compression_round": round_count,
                "budget_exceeded": True,
                "tokenizer_unavailable": True,
            }
            return packet

        if used > INPUT_TARGET_AFTER_COMPACT:
            # 第二轮只裁减远期档案和已完成决策的载荷；未关闭事实仍保留原文与来源。
            payload["origin"]["person_world_profile"] = _compact_origin(
                payload["origin"].get("person_world_profile", {})
            )
            payload["branch"]["recent_events"] = payload["branch"].get("recent_events", [])[:6]
            round_count = 2
            used = self._count_payload(payload)
            if used is None:
                packet.budgets = {
                    "input_tokens": 0,
                    "compressed": True,
                    "compression_round": round_count,
                    "budget_exceeded": True,
                    "tokenizer_unavailable": True,
                }
                return packet

        packet = ContextPacket.model_validate(payload)
        packet.budgets = {
            "input_tokens": used,
            "compressed": True,
            "compression_round": round_count,
            "token_count_source": "director_tokenizer",
        }
        if used > INPUT_HARD_LIMIT:
            # 硬上限前不允许发起模型请求，Director 会产生可审计的安全等待。
            packet.budgets["budget_exceeded"] = True
        elif used > INPUT_TARGET_AFTER_COMPACT:
            packet.budgets["target_exceeded"] = True
        return packet

    def _count_payload(self, payload: dict[str, Any]) -> int | None:
        """计算实际 Director 请求的输入 token，不把字符数冒充 token。"""

        if self.token_counter is None:
            return None
        serialized = (
            DIRECTOR_SYSTEM_PROMPT
            + "\n<runtime_context>"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "</runtime_context>"
        )
        try:
            return max(1, int(self.token_counter(serialized)))
        except Exception:
            # tokenizer 服务不可用时必须走 Director 的安全 wait，而非乐观估算。
            return None

    def _persist_summary(
        self,
        *,
        branch_id: str,
        messages: list[dict[str, Any]],
        compression_round: int,
    ) -> str | None:
        """保存可追溯的确定性窗口摘要；绝不调用模型补写新事实。"""

        if not branch_id or not messages:
            return None
        source_message_ids = [
            str(item.get("source_id"))
            for item in messages
            if isinstance(item.get("source_id"), str) and item["source_id"]
        ]
        if not source_message_ids:
            return None
        latest = self.session.scalar(
            select(RuntimeContextSummaryRow)
            .where(RuntimeContextSummaryRow.branch_id == branch_id)
            .order_by(RuntimeContextSummaryRow.created_at.desc())
            .limit(1)
        )
        if latest is not None and list(latest.source_message_ids or []) == source_message_ids:
            return latest.text
        occurred_values = [item.get("occurred_at") for item in messages]
        covered_until = _latest_occurred_at(occurred_values)
        excerpts = []
        for item in messages:
            role = str(item.get("role", "unknown"))
            content = str(item.get("content", "")).replace("\n", " ").strip()
            source_id = str(item.get("source_id", ""))
            if content:
                excerpts.append(f"{role}: {content[:180]} [source:{source_id}]")
        text = "已压缩的早期对话摘录（仅保留原文片段，不新增事实）：" + " | ".join(excerpts)
        row = RuntimeContextSummaryRow(
            branch_id=branch_id,
            covered_until=covered_until,
            source_message_ids=source_message_ids,
            source_event_ids=[],
            text=text[:4000],
            unresolved_items=[],
            compression_round=compression_round,
        )
        self.session.add(row)
        return row.text


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
    keys = (
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


def _compact_origin(profile: object) -> dict[str, object]:
    """第二轮压缩只保留每个档案栏目最靠前的少量带来源陈述。"""

    if not isinstance(profile, dict):
        return {}
    compact: dict[str, object] = {}
    for key, value in profile.items():
        if isinstance(value, list):
            compact[key] = value[:3]
        elif isinstance(value, dict):
            compact[key] = {
                nested_key: nested_value[:3] if isinstance(nested_value, list) else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            compact[key] = value
    return compact
