"""虚拟时钟：所有 Runtime 模型共享同一个确定的分支时间。"""

from datetime import UTC, datetime

from .schemas import VirtualClock


def create_clock(
    branch_id: str,
    virtual_anchor: datetime,
    *,
    wall_anchor: datetime | None = None,
    timezone: str = "UTC",
) -> VirtualClock:
    """创建 1:1 时钟；分支创建时不让模型参与时间初始化。"""
    wall = wall_anchor or datetime.now(UTC)
    return VirtualClock(
        branch_id=branch_id,
        virtual_anchor=virtual_anchor,
        wall_anchor=wall,
        time_scale=1.0,
        status="running",
        timezone=timezone,
    )


def pause_clock(clock: VirtualClock, *, wall_now: datetime | None = None) -> VirtualClock:
    """暂停时重新锚定已流逝的虚拟时间，保证恢复时不会跳变。"""
    now = clock.now(wall_now)
    wall = wall_now or datetime.now(UTC)
    return clock.model_copy(update={"virtual_anchor": now, "wall_anchor": wall, "status": "paused"})


def resume_clock(clock: VirtualClock, *, wall_now: datetime | None = None) -> VirtualClock:
    """从暂停点恢复 1:1 运行。"""
    wall = wall_now or datetime.now(UTC)
    return clock.model_copy(update={"wall_anchor": wall, "status": "running"})


def set_time_scale(
    clock: VirtualClock,
    time_scale: float,
    *,
    wall_now: datetime | None = None,
) -> VirtualClock:
    """改变倍率时创建新锚点，避免旧事件的时间解释发生变化。"""
    if time_scale <= 0:
        raise ValueError("time_scale 必须大于 0")
    now = clock.now(wall_now)
    wall = wall_now or datetime.now(UTC)
    return clock.model_copy(
        update={
            "virtual_anchor": now,
            "wall_anchor": wall,
            "time_scale": time_scale,
        }
    )
