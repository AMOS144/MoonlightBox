"""Runtime v1 的领域契约。

这些 Pydantic 模型是 Director、PersonaActor、Executor 和 API 之间唯一共享的
数据边界。数据库 JSON 字段在进入运行时后也必须先通过这里的校验。
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Availability = Literal["available", "busy", "resting", "asleep", "unknown"]
Energy = Literal["low", "medium", "high", "unknown"]
EventType = Literal[
    "user_message",
    "plan_transition",
    "delayed_reply",
    "commitment_due",
    "system",
]


class RuntimeEvent(StrictModel):
    id: str
    branch_id: str
    event_type: EventType
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=160)
    status: Literal["queued", "claimed", "completed", "cancelled", "invalidated"] = "queued"


class OriginWorldSnapshot(StrictModel):
    """分支起点的只读世界资料；v1 的 cutoff 是当前导入图谱末时刻。"""

    id: str
    branch_id: str
    source_graph_version_id: str | None = None
    source_profile_id: str | None = None
    source_node_id: str | None = None
    cutoff_at: datetime
    timezone: str = "UTC"
    snapshot_mode: Literal["latest_profile", "historical_cutoff"] = "latest_profile"
    source_message_ids: list[str] = Field(default_factory=list)
    person_world_profile: dict[str, Any] = Field(default_factory=dict)
    routine_profile: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    compiler_version: str = "runtime-v1"


class Wakeup(StrictModel):
    id: str
    branch_id: str
    wake_at: datetime
    reason: str = Field(min_length=1, max_length=200)
    trigger_type: Literal[
        "plan_transition", "delayed_reply", "commitment", "user_message", "system"
    ]
    idempotency_key: str = Field(min_length=1, max_length=160)
    status: Literal["scheduled", "executing", "completed", "cancelled"] = "scheduled"


class VirtualClock(StrictModel):
    branch_id: str
    virtual_anchor: datetime
    wall_anchor: datetime
    time_scale: float = Field(default=1.0, gt=0)
    status: Literal["running", "paused"] = "running"
    timezone: str = "UTC"

    def now(self, wall_now: datetime | None = None) -> datetime:
        """根据墙上时间计算虚拟时间；LLM 永远不能修改这个结果。"""
        if self.status == "paused":
            return self.virtual_anchor
        current_wall = wall_now or datetime.now(self.virtual_anchor.tzinfo)
        # SQLite 读取 DateTime(timezone=True) 时可能丢失 tzinfo，统一按 UTC 解释。
        anchor = self.wall_anchor
        if anchor.tzinfo is None and current_wall.tzinfo is not None:
            anchor = anchor.replace(tzinfo=current_wall.tzinfo)
        elif anchor.tzinfo is not None and current_wall.tzinfo is None:
            current_wall = current_wall.replace(tzinfo=anchor.tzinfo)
        elapsed = current_wall - anchor
        return self.virtual_anchor + elapsed * self.time_scale


class DayPlanBlock(StrictModel):
    id: str
    start: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(pattern=r"^(([01]\d|2[0-3]):[0-5]\d|24:00)$")
    activity: str = Field(min_length=1, max_length=160)
    location_role: str | None = Field(default=None, max_length=160)
    default_availability: Availability = "unknown"


class DayPlan(StrictModel):
    branch_id: str
    date: Date | None = None
    plan_date: Date | None = None
    blocks: list[DayPlanBlock] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def normalize_date(self) -> DayPlan:
        if self.date is None and self.plan_date is None:
            raise ValueError("DayPlan 必须提供 date")
        if self.date is None:
            self.date = self.plan_date
        if self.plan_date is None:
            self.plan_date = self.date
        return self

    def current_block(self, at: datetime) -> DayPlanBlock | None:
        """返回包含当前本地时间的生活块，跨午夜块按普通字符串规则处理。"""
        local = at.astimezone(at.tzinfo).strftime("%H:%M") if at.tzinfo else at.strftime("%H:%M")
        for block in self.blocks:
            if block.start <= local < block.end:
                return block
        return None


class LifeState(StrictModel):
    branch_id: str
    virtual_now: datetime
    current_plan_block_id: str | None = None
    location_role: str | None = None
    activity: str | None = None
    social_context: str | None = None
    availability: Availability = "unknown"
    energy: Energy = "unknown"
    mood: str | None = None
    attention: str | None = None
    current_goal: str | None = None
    open_conversation_threads: list[dict[str, Any]] = Field(default_factory=list)
    active_commitments: list[dict[str, Any]] = Field(default_factory=list)
    last_transition_at: datetime
    valid_until: datetime | None = None
    reason: str = "initial"
    source_event_ids: list[str] = Field(default_factory=list)
    field_sources: dict[str, list[str]] = Field(default_factory=dict)
    previous_version_id: str | None = None
    version: int = 1


class MemoryRecord(StrictModel):
    id: str
    scope: Literal["world", "branch"]
    branch_id: str | None = None
    snapshot_id: str | None = None
    subject: str
    predicate: str
    object: str
    summary: str
    status: Literal["asserted", "confirmed", "superseded", "rejected"] = "asserted"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)
    supersedes_id: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_scope(self) -> MemoryRecord:
        if self.scope == "branch" and not self.branch_id:
            raise ValueError("branch 记忆必须绑定 branch_id")
        if self.scope == "world" and not self.snapshot_id:
            raise ValueError("world 记忆必须绑定 snapshot_id")
        return self


class MemoryEvidence(StrictModel):
    record_id: str
    scope: Literal["world", "branch"]
    summary: str
    source_ids: list[str] = Field(default_factory=list)
    occurred_at: datetime | None = None
    valid_until: datetime | None = None
    certainty: Literal["observed", "approved", "inferred", "unknown"] = "unknown"
    original: str | None = None


class WorkingMessage(StrictModel):
    source_id: str
    sequence: int
    role: Literal["user", "assistant", "system"]
    content: str
    occurred_at: datetime | None = None


class ContextPacket(StrictModel):
    protocol_version: Literal["runtime-context-v1"] = "runtime-context-v1"
    generated_at: datetime
    virtual_now: datetime
    timezone: str
    trigger: dict[str, Any]
    origin: dict[str, Any]
    current: dict[str, Any]
    branch: dict[str, Any]
    memory: dict[str, Any] = Field(default_factory=lambda: {"retrieved_records": []})
    budgets: dict[str, Any] = Field(default_factory=dict)


class StatePatch(StrictModel):
    activity: str | None = Field(default=None, max_length=160)
    location_role: str | None = Field(default=None, max_length=160)
    social_context: str | None = Field(default=None, max_length=160)
    availability: Availability | None = None
    energy: Energy | None = None
    mood: str | None = Field(default=None, max_length=160)
    attention: str | None = Field(default=None, max_length=300)
    current_goal: str | None = Field(default=None, max_length=300)
    open_conversation_threads: list[dict[str, Any]] | None = None
    active_commitments: list[dict[str, Any]] | None = None
    valid_until: datetime | None = None
    reason: str = Field(default="", max_length=200)
    source_event_ids: list[str] = Field(default_factory=list, max_length=20)


class PlanPatch(StrictModel):
    block_id: str | None = None
    start: str | None = None
    end: str | None = None
    activity: str | None = Field(default=None, max_length=160)
    location_role: str | None = Field(default=None, max_length=160)
    default_availability: Availability | None = None


class LifeDecision(StrictModel):
    action: Literal["speak", "wait", "continue_life", "schedule"] = "wait"
    state_patch: StatePatch = Field(default_factory=StatePatch)
    plan_patch: PlanPatch | None = None
    speech_mode: Literal["reply", "proactive", "delayed_reply"] | None = None
    communication_intent: str | None = Field(default=None, max_length=500)
    content_points: list[str] = Field(default_factory=list, max_length=8)
    next_wakeup_at: datetime | None = None
    private_reason: str = Field(default="", max_length=500)


class ActorMessage(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    bubbles: list[str] = Field(default_factory=list, max_length=8)
    style_applied: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def ensure_bubbles(self) -> ActorMessage:
        if not self.bubbles:
            self.bubbles = [self.text]
        return self


class SearchMemoryArgs(StrictModel):
    scope: Literal["branch", "world", "both"]
    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=8, ge=1, le=12)
    include_original: bool = False


class GetStyleExamplesArgs(StrictModel):
    # 版本由 StyleService 绑定当前分支；模型没有跨模型读取的能力。
    model_version_id: str | None = Field(default=None, max_length=128)
    # 由 PersonaActor 根据本轮 runtime_context 组织。它用于 LightRAG 的语义检索，
    # 不是把相邻历史消息硬拼成「提问—回复」对。
    situation: str = Field(min_length=8, max_length=1200)
    intent: str = Field(min_length=2, max_length=160)
    speech_mode: Literal["reply", "proactive", "delayed_reply"]
    limit: int = Field(default=4, ge=1, le=6)


class MergedTrigger(StrictModel):
    branch_id: str
    trigger_ids: list[str]
    primary_trigger: EventType
    input_cutoff_at: datetime
    priority: Literal["realtime", "commitment", "plan", "proactive"]
    additional_triggers: list[dict[str, Any]] = Field(default_factory=list)
