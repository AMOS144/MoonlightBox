"""Runtime v1 的领域契约。

这些 Pydantic 模型是 Director、PersonaActor、Executor 和 API 之间唯一共享的
数据边界。数据库 JSON 字段在进入运行时后也必须先通过这里的校验。
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as Date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .director_contracts import (
    ExpressionTask,
    InputResolution,
    MemoryProposal,
    SubjectiveStateUpdate,
)
from .expression_contracts import ExpressionResult


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
        # SQLite 的 DateTime(timezone=True) 会丢失 tzinfo。运行时锚点统一按 UTC
        # 解释，不能因为部署机器恰好在东八区而把经过时间多算八小时。
        virtual_anchor = _as_utc(self.virtual_anchor)
        if self.status == "paused":
            return virtual_anchor
        current_wall = _as_utc(wall_now or datetime.now(UTC))
        elapsed = current_wall - _as_utc(self.wall_anchor)
        return virtual_anchor + elapsed * self.time_scale


def _as_utc(value: datetime) -> datetime:
    """把 SQLite 读出的朴素时间还原为系统约定的 UTC 时间点。"""

    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class DayPlanBlock(StrictModel):
    id: str
    start: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(pattern=r"^(([01]\d|2[0-3]):[0-5]\d|24:00)$")
    activity: str = Field(min_length=1, max_length=160)
    location_role: str | None = Field(default=None, max_length=160)
    default_availability: Availability = "unknown"
    # 计划块本身也保留证据边界。它们只用于审计和后续修订，不会直接进入聊天提示词。
    basis: Literal[
        "branch_commitment",
        "snapshot_routine",
        "historical_pattern",
        "profile_inference",
        "date_feature",
        "simulation_assumption",
        "fallback",
    ] = "fallback"
    evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    confidence: Literal["confirmed", "inferred", "fallback"] = "fallback"
    assumption: str | None = Field(default=None, max_length=500)


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
    basis: str | None = None
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
    activity: str | None = Field(
        default=None,
        max_length=160,
        description="此刻实际活动的变化；null 不更新，不把未来计划提前当现状",
    )
    location_role: str | None = Field(
        default=None, max_length=160, description="当前位置或场所角色的变化；null 不更新"
    )
    social_context: str | None = Field(
        default=None, max_length=160, description="当前身边的人或社交环境变化；null 不更新"
    )
    availability: Availability | None = Field(
        default=None, description="当前能否回应：available/busy/resting/asleep/unknown；null 不更新"
    )
    energy: Energy | None = Field(default=None, description="当前精力状态；null 不更新")
    mood: str | None = Field(
        default=None,
        max_length=160,
        description="历史兼容字段，新 Director 使用 subjective_state_updates.mood",
    )
    attention: str | None = Field(
        default=None, max_length=300, description="历史兼容字段，新 Director 通过关切表达关注点"
    )
    current_goal: str | None = Field(
        default=None,
        max_length=300,
        description="历史兼容字段，新 Director 通过关切 intention 表达意图",
    )
    open_conversation_threads: list[dict[str, Any]] | None = Field(
        default=None, description="历史兼容字段，不由新工具直接覆写"
    )
    active_commitments: list[dict[str, Any]] | None = Field(
        default=None, description="需要更新时提交完整有效承诺列表；null 不更新，不通过遗漏清空承诺"
    )
    valid_until: datetime | None = Field(
        default=None, description="带时区 ISO 虚拟时间，表示当前状态预计有效到何时；不是完成回执"
    )
    reason: str = Field(default="", max_length=200, description="这次生活状态变化的简短原因")
    source_event_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="支持变化的当前已读事件引用；没有则 []，不能使用未读或不存在的事件",
    )


class PlanPatch(StrictModel):
    """已废弃的旧契约。

    DayPlan 不再由 Director 直接局部改写。保留该类型仅为了让旧 Trace 能被读取；
    新的 LifeDecision 一律使用 ``plan_request``，再交由 DayPlanAgent 生成提案。
    """

    block_id: str | None = None
    start: str | None = None
    end: str | None = None
    activity: str | None = Field(default=None, max_length=160)
    location_role: str | None = Field(default=None, max_length=160)
    default_availability: Availability | None = None


class PlanRevisionRequest(StrictModel):
    """Director 交给 DayPlanAgent 的受限计划请求，而不是可直接落库的 patch。"""

    target_date: Date | None = Field(
        default=None, description="要修改的分支本地日期 YYYY-MM-DD；省略由工作流使用当前目标日期"
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="具体需要调整什么以及原因，保留已确认约束；不是对 Planner 的泛化提问",
    )
    source_event_ids: list[str] = Field(
        default_factory=list, max_length=20, description="引发调整的已读事件引用，不编造"
    )
    affected_start: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="受影响本地时间段起点 HH:MM；不清楚 null",
    )
    affected_end: str | None = Field(
        default=None,
        pattern=r"^(([01]\d|2[0-3]):[0-5]\d|24:00)$",
        description="受影响本地时间段终点 HH:MM，日末可 24:00；不清楚 null",
    )


class DayPlanProposalBlock(StrictModel):
    """Planner 的纯提案块；ID 只能由 Executor 在通过校验后分配。"""

    start: str = Field(
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="分支本地时间 HH:MM，按15分钟对齐，全天首块从00:00开始",
    )
    end: str = Field(
        pattern=r"^(([01]\d|2[0-3]):[0-5]\d|24:00)$",
        description="本地结束时间 HH:MM，日末可24:00；晚于start，接下一块start，按15分钟对齐",
    )
    activity: str = Field(
        min_length=1,
        max_length=160,
        description="具体要做的事情；不把性格分析、动机解释写进活动名称",
    )
    location_role: str | None = Field(
        default=None,
        max_length=160,
        description="具体场所或场所角色；无依据可 null，不编造精确地址",
    )
    default_availability: Availability = Field(
        default="unknown",
        description="该块默认回应状态：available 可回应，busy 忙，resting 休息，asleep 睡眠，unknown 未知",
    )
    basis: Literal[
        "branch_commitment",
        "snapshot_routine",
        "historical_pattern",
        "profile_inference",
        "simulation_assumption",
        "fallback",
    ] = Field(
        description="依据类型：branch_commitment 为分支承诺；snapshot_routine/historical_pattern 为已有规律；profile_inference 为画像支持的推断；simulation_assumption 为使模拟可执行而补齐的安排；fallback 为缺少个体资料的兜底。不得为了通过校验更换事实依据。"
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=24,
        description='只填当前材料或工具提供的引用。simulation_assumption/fallback 必须填空数组 []（不是空字符串、null、[""] 或占位符）；其他 basis 至少一个有效引用。通勤存在的证据不等于精确通勤时长的证据。',
        examples=[[]],
    )
    confidence: Literal["confirmed", "inferred", "fallback"] = Field(
        default="inferred",
        description="simulation_assumption 固定 inferred；fallback 固定 fallback；其他依据用 confirmed 或 inferred，匹配实际确定程度。",
    )
    assumption: str | None = Field(
        default=None,
        max_length=500,
        description="模拟补齐时必填非空理由，说明哪些时间或安排是估计；其他依据可为 null。不能用说明文字替代正确填写 basis/evidence_ids。",
    )

    @field_validator("evidence_ids")
    @classmethod
    def evidence_matches_basis(cls, value, info):
        # 将错误定位到具体字段，回执才能回显实际数组，而不是整块提案。
        basis = info.data.get("basis")
        if basis in {"simulation_assumption", "fallback"} and value:
            raise ValueError(
                f"{basis} 的 evidence_ids 必须是空数组 []；删除此字段中的引用，不要更换 basis 来绕过校验"
            )
        if basis and basis not in {"simulation_assumption", "fallback"} and not value:
            raise ValueError(
                "此 basis 需要已提供的真实引用；若安排来自模拟估计，明确选择 simulation_assumption，不编造 evidence_ids"
            )
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_matches_basis(cls, value, info):
        basis = info.data.get("basis")
        expected = {"simulation_assumption": "inferred", "fallback": "fallback"}.get(basis)
        if expected and value != expected:
            raise ValueError(f"basis={basis} 时 confidence 必须是 {expected}")
        return value

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> DayPlanProposalBlock:
        """提案阶段就拒绝“声称有依据但没有来源”的计划块。"""

        if self.basis == "simulation_assumption":
            if not self.assumption or not self.assumption.strip():
                raise ValueError("simulation_assumption 必须填写 assumption，说明模拟安排的理由")
            if self.evidence_ids or self.confidence != "inferred":
                raise ValueError("模拟补齐不冒充事实：evidence_ids=[]，confidence=inferred")
        elif self.basis == "fallback":
            if self.evidence_ids or self.confidence != "fallback":
                raise ValueError("fallback 块必须使用空 evidence_ids 和 fallback confidence")
        elif not self.evidence_ids or self.confidence == "fallback":
            raise ValueError(
                "此块声称有依据，但缺少 evidence_ids 或 confidence 不可用。"
                "真实安排引用事件，画像推断引用 profile_references；"
                "若只是模拟时间安排，请主动改为 simulation_assumption，"
                "填写 assumption 说明，不要编造来源。"
            )
        return self


class DayPlanProposal(StrictModel):
    """DayPlanAgent 的唯一输出；它不包含数据库 ID 或可执行副作用。"""

    plan_date: Date = Field(
        description="本任务指定的分支本地日期 YYYY-MM-DD；不能改用墙上日期或 UTC 日期"
    )
    blocks: list[DayPlanProposalBlock] = Field(
        min_length=1,
        max_length=24,
        description="按时间顺序提交全天完整计划，从 00:00 连续覆盖到 24:00，无空档、重叠、倒序，按15分钟对齐；原样保留上下文给出的 locked_blocks",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="明确区分模拟选择、估算和已知资料；无额外假设填 []，不是事实记忆",
    )
    private_reason: str = Field(
        default="",
        max_length=500,
        description="供协作方理解本次排程选择的简明理由，不作为用户聊天内容",
    )


class PeerReply(StrictModel):
    """协作中的具体追问/回答，不含角色、路由 ID 或写库权限。"""

    kind: Literal["clarification", "answer", "objection"] = Field(
        description="clarification 提出影响安排的问题；answer 回答收到的问题；objection 说明具体冲突；均不表示计划已生效"
    )
    content: str = Field(
        min_length=1,
        max_length=2000,
        description="清楚说明问题、回答或异议及其影响；不夹带路由命令或数据库写入操作",
    )
    request_ref: str | None = Field(default=None, description="回答哪个协作输入的 source_event_id")


class DayPlanTurn(StrictModel):
    """规划节点可交付提案，也可先提出影响修改结果的问题。"""

    proposal: DayPlanProposal | None = Field(
        default=None,
        description="完整日程提案；与 reply、completion 三选一；接受提案不等于 Executor 已提交",
    )
    reply: PeerReply | None = Field(
        default=None,
        description=(
            "需要对方答复时填写 clarification/objection；已有答案可用 answer。"
            "与 proposal、completion 三选一"
        ),
    )
    completion: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description=(
            "仅当 allow_no_change 为 true：简述已理解消息、无需改计划的原因，"
            "不自动向对方回信。与 proposal、reply 三选一"
        ),
    )

    @model_validator(mode="after")
    def exactly_one(self):
        if sum(value is not None for value in (self.proposal, self.reply, self.completion)) != 1:
            raise ValueError("proposal、reply、completion 必须且只能填写一项")
        return self


class LifeDecision(StrictModel):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        """历史表达任务只读兼容；当前模型直接提交实际回复。"""
        schema = super().model_json_schema(*args, **kwargs)
        for name in ("communication_intent", "content_points", "expression_task"):
            schema.get("properties", {}).pop(name, None)
        # 历史状态仍可读取，但新 Director 不再通过旧接口写入心理状态。
        properties = schema.get("$defs", {}).get("StatePatch", {}).get("properties", {})
        for name in ("mood", "attention", "current_goal", "open_conversation_threads"):
            properties.pop(name, None)
        return schema

    action: Literal["speak", "wait", "continue_life", "schedule"] = Field(
        description="speak 交付实际回复；wait 暂不表达；continue_life 继续生活；schedule 安排明确唤醒。speak 必须有 ready 的 reply，schedule 必须有 next_wakeup_at。"
    )
    state_patch: StatePatch = Field(
        default_factory=StatePatch,
        description="本轮需要更新的生活状态字段；不更新的字段省略或 null，心理变化放 subjective_state_updates",
    )
    # Director 只能提出“需要重新规划”，不能再把某一个 block 直接写进 DayPlan。
    plan_request: PlanRevisionRequest | None = Field(
        default=None,
        description="确需改日程时提交给同级 DayPlan 的请求；只是判断已有计划时不需要，不意味着日程已修改",
    )
    peer_reply: PeerReply | None = Field(
        default=None,
        description="回答或澄清已收到的 DayPlan 协作问题；用 request_ref 关联，不虚构已提交回执",
    )
    speech_mode: Literal["reply", "proactive", "delayed_reply"] | None = Field(
        default=None, description="speak 的表达类型：接话、主动发起或延迟接话；非表达动作可 null"
    )
    communication_intent: str | None = Field(default=None, max_length=500)
    content_points: list[str] = Field(default_factory=list, max_length=8)
    next_wakeup_at: datetime | None = Field(
        default=None,
        description="带时区的 ISO 虚拟时间点，必须晚于当前虚拟时间；schedule 必填，不是现实墙上时间",
    )
    private_reason: str = Field(
        default="", max_length=500, description="本轮决定的简明原因，供协作和审查，不发送给用户"
    )
    subjective_state_updates: SubjectiveStateUpdate | None = Field(
        default=None, description="本轮心理状态增量；null 不更新，不能通过省略清空旧关切"
    )
    expression_task: ExpressionTask | None = None
    reply: ExpressionResult | None = Field(
        default=None,
        description="speak 必须填写 ready 的实际文字/表情消息；其他动作填 null。不是表达意图或候选草稿。",
    )
    memory_proposals: list[MemoryProposal] = Field(
        default_factory=list,
        description="本轮值得持久记忆的新内容或修订；无则 []，来源与主体必须正确",
    )
    input_resolutions: list[InputResolution] = Field(
        default_factory=list,
        description="对当前已读待处理消息逐条交付状态；completed 必须由本次真实回复处理，不能把未读新输入标记完成",
    )

    @model_validator(mode="after")
    def validate_action(self):
        refs = [item.message_ref for item in self.input_resolutions]
        if len(refs) != len(set(refs)):
            raise ValueError("input_resolutions 同一 message_ref 只能提交一次处理状态")
        if self.reply is not None:
            if self.action != "speak" or self.reply.status != "ready":
                raise ValueError("reply 只用于 speak，必须包含 ready 的实际回复")
            if self.expression_task is None:
                self.expression_task = ExpressionTask(
                    purpose="直接回复",
                    speech_mode=self.speech_mode or "reply",
                    must_convey=[],
                    respond_to_refs=[
                        item.message_ref
                        for item in self.input_resolutions
                        if item.status == "completed"
                    ],
                )
        if self.expression_task is not None:
            if self.action != "speak":
                raise ValueError("expression_task 只用于 speak")
            self.speech_mode = self.expression_task.speech_mode
            self.communication_intent = self.expression_task.purpose
            self.content_points = self.expression_task.must_convey
        if self.action == "speak" and self.expression_task is None:
            if not self.communication_intent or not self.content_points:
                raise ValueError("speak 必须给出 expression_task")
            self.expression_task = ExpressionTask(
                purpose=self.communication_intent,
                must_convey=self.content_points,
                speech_mode=self.speech_mode or "reply",
            )
        if self.action == "schedule" and self.next_wakeup_at is None:
            raise ValueError("schedule 必须给出 next_wakeup_at")
        return self


class ActorMessage(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    bubbles: list[str] = Field(default_factory=list, max_length=8)
    style_applied: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def ensure_bubbles(self) -> ActorMessage:
        if not self.text.strip() or any(not bubble.strip() for bubble in self.bubbles):
            raise ValueError("Actor 消息和气泡不能只有空白")
        if not self.bubbles:
            self.bubbles = [self.text]
        return self


class MergedTrigger(StrictModel):
    branch_id: str
    trigger_ids: list[str]
    primary_trigger: EventType
    input_cutoff_at: datetime
    priority: Literal["realtime", "commitment", "plan", "proactive"]
    additional_triggers: list[dict[str, Any]] = Field(default_factory=list)
