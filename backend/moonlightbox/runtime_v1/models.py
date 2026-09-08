"""Runtime v1 兼容导出：领域模型与 ORM 模型分开定义，但统一从此处发现。"""

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
from .schemas import (
    ActorMessage,
    ContextPacket,
    DayPlan,
    DayPlanBlock,
    LifeDecision,
    LifeState,
    MemoryEvidence,
    MemoryRecord,
    MergedTrigger,
    OriginWorldSnapshot,
    VirtualClock,
    Wakeup,
)

# 设计文档中的语义名称；实际 ORM 类名带 Row 后缀，避免与领域 schema 混淆。
LifeStateVersion = RuntimeLifeStateRow
OriginWorldSnapshotRow = RuntimeSnapshotRow
DayPlanRow = RuntimeDayPlanRow
EventQueueRow = RuntimeEventRow

__all__ = [
    "ActorMessage",
    "ContextPacket",
    "DayPlan",
    "DayPlanBlock",
    "LifeDecision",
    "LifeState",
    "MemoryEvidence",
    "MemoryRecord",
    "MergedTrigger",
    "OriginWorldSnapshot",
    "VirtualClock",
    "Wakeup",
    "RuntimeClockRow",
    "RuntimeContextSummaryRow",
    "RuntimeDayPlanRow",
    "RuntimeEventRow",
    "RuntimeLifeEventRow",
    "RuntimeLifeStateRow",
    "RuntimeMemoryRow",
    "RuntimeMemoryIndexRow",
    "RuntimeSnapshotRow",
    "RuntimeWakeupRow",
    "LifeStateVersion",
    "OriginWorldSnapshotRow",
    "DayPlanRow",
    "EventQueueRow",
]
