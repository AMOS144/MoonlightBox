"""Runtime v1：事件驱动的分支生活运行时。

该包与早期 cognition/continuity 实现隔离，所有新表、Prompt 和工具 schema
都集中在 runtime_v1 中，便于后续替换和排查。
"""

from .db_models import (
    RuntimeClockRow,
    RuntimeContextSummaryRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeLifeStateRow,
    RuntimeMemoryIndexRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from .event_queue import RuntimeEventQueue
from .schemas import (
    ActorMessage,
    ContextPacket,
    DayPlan,
    DayPlanBlock,
    LifeDecision,
    LifeState,
    MemoryRecord,
    OriginWorldSnapshot,
    VirtualClock,
)
from .service import RuntimeService

__all__ = [
    "RuntimeDayPlanRow",
    "RuntimeEventRow",
    "RuntimeLifeEventRow",
    "RuntimeLifeStateRow",
    "RuntimeMemoryRow",
    "RuntimeMemoryIndexRow",
    "RuntimeSnapshotRow",
    "RuntimeWakeupRow",
    "RuntimeClockRow",
    "RuntimeContextSummaryRow",
    "RuntimeService",
    "ActorMessage",
    "ContextPacket",
    "DayPlan",
    "DayPlanBlock",
    "LifeDecision",
    "LifeState",
    "MemoryRecord",
    "OriginWorldSnapshot",
    "VirtualClock",
    "RuntimeEventQueue",
]
