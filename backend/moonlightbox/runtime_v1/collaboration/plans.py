"""双方共用的三天已提交计划投影。缺失不代表空闲。"""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from ..db_models import RuntimeDayPlanRow
from ..plan_status import preparation


def local_time(value: datetime, timezone: str) -> datetime:
    """先还原存储时间点，再按分支时区解释日期和时刻。"""
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    zone = (
        datetime.fromisoformat("2000-01-01T00:00:00" + timezone[3:]).tzinfo
        if timezone.startswith(("UTC+", "UTC-"))
        else ZoneInfo(timezone)
    )
    return aware.astimezone(zone)


def shared_plan_window(session, branch_id: str, now: datetime, timezone: str) -> dict:
    today = local_time(now, timezone).date()
    dates = [(today + timedelta(days=offset)).isoformat() for offset in (-1, 0, 1)]
    rows = {
        row.plan_date: row
        for row in session.scalars(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id, RuntimeDayPlanRow.plan_date.in_(dates)
            )
        )
    }
    result = {}
    for day in dates:
        row = rows.get(day)
        status = (row.generation_metadata or {}).get("status") if row else None
        available = row is not None and status == "agent"
        result[day] = {
            "date": day,
            "status": "available"
            if available
            else "unavailable"
            if status == "unavailable"
            else "missing",
            "plan_version": row.version if row else None,
            "preparation": preparation(row),
            "missing_plan_policy": "判断未知；可回复说明正在准备，不得视为空闲或虚构安排",
            "blocks": list(row.blocks or []) if available else [],
            "read_only": day < today.isoformat(),
        }
    return result
