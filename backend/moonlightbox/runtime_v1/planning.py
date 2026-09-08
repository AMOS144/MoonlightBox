"""基于人物规律生成粗粒度 DayPlan。"""

from datetime import date
from uuid import uuid4

from .schemas import Availability, DayPlan, DayPlanBlock


def generate_day_plan(
    branch_id: str,
    plan_date: date,
    routine_profile: dict[str, object] | None = None,
) -> DayPlan:
    """每天只生成少量生活块，具体活动仍可被分支局部覆盖。

    v1 使用稳定的默认骨架；routine_profile 作为文本提示附着在活动名称上，
    不把没有时间证据的背景强行解释成精确日历。
    """
    routine = routine_profile or {}
    weekday = plan_date.weekday() < 5
    work_hint = _routine_hint(routine, "workdays" if weekday else "weekends")
    blocks = [
        # 用 24:00 表示当天结束，避免午夜出现没有任何生活块的空档。
        _block("00:00", "07:00", "睡眠", "home", "asleep"),
        _block("07:00", "08:00", "起床与早餐", "home", "available"),
        _block(
            "08:00",
            "09:00",
            "通勤" if weekday else "个人时间",
            "transit" if weekday else "home",
            "busy" if weekday else "available",
        ),
        _block(
            "09:00",
            "12:00",
            f"工作{work_hint}" if weekday else "个人时间",
            "work" if weekday else "home",
            "busy" if weekday else "available",
        ),
        _block("12:00", "13:30", "午餐与休息", "outside", "available"),
        _block(
            "13:30",
            "18:00",
            f"工作{work_hint}" if weekday else "个人时间",
            "work" if weekday else "home",
            "busy" if weekday else "available",
        ),
        _block(
            "18:00",
            "19:00",
            "通勤" if weekday else "晚餐",
            "transit" if weekday else "home",
            "busy" if weekday else "available",
        ),
        _block("19:00", "23:00", "私人时间", "home", "available"),
        _block("23:00", "24:00", "睡眠", "home", "asleep"),
    ]
    return DayPlan(branch_id=branch_id, plan_date=plan_date, blocks=blocks)


def _block(
    start: str, end: str, activity: str, location: str, availability: Availability
) -> DayPlanBlock:
    return DayPlanBlock(
        id=str(uuid4()),
        start=start,
        end=end,
        activity=activity,
        location_role=location,
        default_availability=availability,
    )


def _routine_hint(profile: dict[str, object], key: str) -> str:
    value = profile.get(key)
    if isinstance(value, list) and value:
        return "（按人物规律）"
    return ""
