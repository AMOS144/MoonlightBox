"""DayPlanAgent 专用的上下文组装器。

规划上下文与 Director 的 ``ContextPacket`` 故意分离：计划不需要整段聊天原文、
表达风格或临时情绪，只需要日期、已确认约束和冻结背景中的作息证据。
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db_models import RuntimeDayPlanRow, RuntimeLifeStateRow, RuntimeSnapshotRow
from .schemas import PlanRevisionRequest, StrictModel
from .snapshot_sources import profile_entries
from .work_calendar import UnverifiedWorkCalendar, WorkCalendar


class DayPlanContext(StrictModel):
    protocol_version: str = "runtime-day-plan-context-v1"
    generated_at: datetime
    virtual_now: datetime
    branch_id: str
    target_date: date
    timezone: str
    mode: str
    allow_no_change: bool = False
    request: dict[str, Any]
    hard_constraints: dict[str, Any]
    prior_plan: dict[str, Any] | None = None
    origin_projection: dict[str, Any]
    date_features: dict[str, Any]
    branch_evidence: dict[str, Any]
    evidence_requirements: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, int]
    day_plans: dict[str, Any] = Field(default_factory=dict)
    collaboration: list[dict[str, Any]] = Field(default_factory=list)
    working_state: dict[str, Any] = Field(default_factory=dict)
    subjective_state: dict[str, Any] = Field(default_factory=dict)


class PlanContextAssembler:
    """只读取已持久化事实并标记来源，不替 Planner 决定日程内容。"""

    def __init__(self, session: Session, *, work_calendar: WorkCalendar | None = None) -> None:
        self.session = session
        # 生产 Worker 显式注入在线日历；没有注入时只给出自然星期并标记未验证，
        # 避免测试、脚本或降级路径悄悄把周一至周五当成法定工作日。
        self.work_calendar = work_calendar or UnverifiedWorkCalendar()

    def assemble(
        self,
        *,
        branch_id: str,
        snapshot: RuntimeSnapshotRow,
        target_date: date,
        timezone: str,
        virtual_now: datetime,
        mode: str,
        request: PlanRevisionRequest | None = None,
    ) -> DayPlanContext:
        from .collaboration.plans import local_time, shared_plan_window

        virtual_now = local_time(virtual_now, timezone)
        plan = self.session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id,
                RuntimeDayPlanRow.plan_date == target_date.isoformat(),
            )
        )
        state = self.session.scalar(
            select(RuntimeLifeStateRow).where(
                RuntimeLifeStateRow.branch_id == branch_id,
                RuntimeLifeStateRow.is_current.is_(True),
            )
        )
        state_payload = dict(state.state or {}) if state is not None else {}
        current_time = virtual_now.strftime("%H:%M")
        locked_blocks = []
        if mode == "revision" and target_date == virtual_now.date() and plan is not None:
            # 已开始块不能被 Planner 重写；否则一个迟到的模型调用会倒改“现在”。
            locked_blocks = [
                dict(block)
                for block in (plan.blocks or [])
                if isinstance(block, dict) and str(block.get("start", "")) <= current_time
            ]
        profile = snapshot.profile if isinstance(snapshot.profile, dict) else {}
        routine = snapshot.routine_profile if isinstance(snapshot.routine_profile, dict) else {}
        date_features = self.work_calendar.day_features(target_date)
        return DayPlanContext(
            subjective_state=state_payload.get("subjective_state", {}),
            day_plans=shared_plan_window(self.session, branch_id, virtual_now, timezone),
            generated_at=datetime.now(UTC),
            virtual_now=virtual_now,
            branch_id=branch_id,
            target_date=target_date,
            timezone=timezone,
            mode=mode,
            request=(
                request.model_dump(mode="json")
                if request is not None
                else {"reason": "为新日期建立初始生活计划", "source_event_ids": []}
            ),
            hard_constraints={
                "active_commitments": state_payload.get("active_commitments", []),
                "locked_blocks": locked_blocks,
                "current_life_state": {
                    key: state_payload.get(key)
                    for key in (
                        "current_plan_block_id",
                        "activity",
                        "location_role",
                        "availability",
                        "valid_until",
                    )
                },
            },
            prior_plan=_prior_plan(plan, mode=mode, date_features=date_features),
            # Snapshot 是资料数据，而不是 system prompt；只投影出排程有关字段。
            origin_projection={
                "background_binding": {
                    "profile_id": snapshot.profile_id,
                    "graph_version_id": snapshot.graph_version_id,
                    "profile_schema_version": profile.get(
                        "profile_schema_version", "legacy_unversioned"
                    ),
                    **profile.get("_runtime_binding", {}),
                },
                "snapshot_id": snapshot.id,
                "cutoff_at": snapshot.cutoff_at,
                "snapshot_mode": snapshot.snapshot_mode,
                "work_and_education": profile.get("work_and_education", []),
                "places": profile.get("places", []),
                "recurring_activities": profile.get("recurring_activities", []),
                "routine_summary": routine,
                "profile_references": profile_entries(snapshot),
                "relationship_with_user": profile.get("relationship_with_user", []),
                # v3 的现实条件与动机仍来自分支绑定的 Snapshot，不查询候选画像。
                **(
                    {
                        "life_context": profile.get("life_context", {}),
                        "practices": profile.get("practices", {}),
                        "agency": profile.get("agency", {}),
                    }
                    if profile.get("profile_schema_version") == "v3"
                    else {}
                ),
            },
            date_features=date_features,
            branch_evidence={
                "accepted_plan_overrides": [],
                "recent_plan_feedback": [],
                "request_source_event_ids": request.source_event_ids if request is not None else [],
            },
            evidence_requirements={
                # 仅给调查方向，不将是否调用过指定工具作为提交条件。
                "suggested_topics": []
                if profile.get("profile_schema_version") == "v3"
                else _required_topics(
                    source_message_ids=snapshot.source_message_ids,
                    date_features=date_features,
                )
            },
            budget={"max_blocks": 24, "minimum_granularity_minutes": 15},
        )


def _required_topics(*, source_message_ids: list[Any], date_features: dict[str, Any]) -> list[str]:
    if not any(isinstance(item, str) and item for item in source_message_ids):
        return []
    common = ["sleep", "meal", "leisure"]
    if date_features.get("calendar_verified") is True and date_features.get("is_workday") is True:
        return ["work", "commute", *common]
    return common


def _prior_plan(
    plan: RuntimeDayPlanRow | None,
    *,
    mode: str,
    date_features: dict[str, Any],
) -> dict[str, Any] | None:
    """旧计划只是修订参考；日历依据变化时不能继续锚定未发生部分。"""

    if plan is None or mode != "revision":
        return None
    metadata = plan.generation_metadata if isinstance(plan.generation_metadata, dict) else {}
    previous_features = metadata.get("date_features")
    current_verified = date_features.get("calendar_verified") is True
    previous_verified = (
        isinstance(previous_features, dict) and previous_features.get("calendar_verified") is True
    )
    same_day_type = (
        previous_verified
        and isinstance(previous_features, dict)
        and previous_features.get("is_workday") == date_features.get("is_workday")
        and previous_features.get("day_type") == date_features.get("day_type")
    )
    reusable = not current_verified or same_day_type
    return {
        "reuse_policy": "reference" if reusable else "locked_only",
        "reason": (
            "日期依据与当前在线日历一致，可把未发生块作为参考"
            if reusable
            else "旧计划缺少可验证日历依据或日期类型已经变化，只保留 locked_blocks"
        ),
        "blocks": list(plan.blocks or []) if reusable else [],
    }
