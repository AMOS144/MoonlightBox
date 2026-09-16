"""概率只描述机会；事件影响由 Agent 提出，执行层不分配剧情预算。"""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from ..schemas import DayPlanProposal, StrictModel


class EventOpportunity(StrictModel):
    direction: Literal["autonomy", "competence", "relatedness"] = Field(
        description="自主、胜任或关系体验方向；不指定具体剧情或人物情绪"
    )
    intensity: float = Field(
        ge=0, le=1, allow_inf_nan=False, description="连续机会强度，不是影响预算"
    )


class EventOccurrence(StrictModel):
    summary: str = Field(min_length=1, description="本次发生的具体内容，不是未来必然发生的剧情")
    details: str = Field(default="", description="必要情境和细节")
    participants: list[str] = Field(default_factory=list, description="实际参与本次模拟事件的人")
    occurred_at: datetime | None = Field(
        default=None,
        description="通常省略，由后端用提交时刻绑定；离线补充才填写恢复区间内带时区的时间",
    )


class EventCharacteristics(StrictModel):
    novelty: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown", description="对人物而言的新颖程度，不能判断时为 unknown"
    )
    disruption: str = Field(description="具体描述对日程的影响，可无影响、跨块或较长，不是预算等级")
    goal_importance: str = Field(default="unknown", description="与人物目标的关联，由当前处境判断")


class EventImpact(StrictModel):
    estimated_added_minutes: int = Field(ge=0, description="估计新增耗时，不是实际已花费时间")
    scope: str = Field(default="", description="预计影响范围，可跨生活块或日期")
    affected_commitments: list[str] = Field(
        default_factory=list, description="可能受影响的约定；提及不代表对方已同意修改"
    )
    expected_long_term_consequences: list[str] = Field(
        default_factory=list, description="预计后续影响，不作为已发生事实"
    )


class EventHandling(StrictModel):
    available_options: list[str] = Field(default_factory=list, description="当前考虑的应对方式")
    chosen_intent: str = Field(description="此刻准备如何应对")
    unresolved_questions: list[str] = Field(
        default_factory=list, description="仍需了解或后续推进的问题"
    )
    actual_outcome: str | None = Field(
        default=None, description="仅在后续推进已有事件时描述此刻已发生的结果；新起因不提前写完成"
    )


class SimulatedEventProposal(StrictModel):
    occurrence: EventOccurrence = Field(description="具体发生内容；讨论中的候选尚未成为经历")
    situation: str = Field(description="事件发生时的人物处境和相关活动")
    characteristics: EventCharacteristics = Field(description="新颖程度与目标关联，不限定影响等级")
    impact: EventImpact = Field(description="估计的耗时和影响范围，不将预期写为已完成")
    appraisal: str | None = Field(
        default=None, description="结合人物当前处境的理解，不由种子指定情绪"
    )
    handling: EventHandling = Field(description="当下应对意图、未决问题与可确认的结果")
    assumptions: list[str] = Field(
        default_factory=list, description="模拟补齐的假设，不冒充真实历史"
    )


class LifeAdvanceDecision(StrictModel):
    outcome: Literal["no_event", "discuss", "submit_event"] = Field(
        description=(
            "no_event 消化机会但不发生事件；discuss 保存工作等待协作；submit_event 提交此刻经历"
        )
    )
    reason: str = Field(min_length=1, description="选择这个结果的原因")
    event: SimulatedEventProposal | None = Field(
        default=None, description="要讨论或提交的事件；no_event 时省略"
    )
    plan_proposals: list[DayPlanProposal] = Field(
        default_factory=list,
        description="确需修改的各日期完整计划，日期是业务目标而非内部 ID；不改则 []",
    )
    question: str | None = Field(
        default=None, description="discuss 时要问 Director 的具体问题；已用通信工具发出可省略"
    )
    follow_up_at: datetime | None = Field(
        default=None,
        description="需要后续推进时填带时区的未来虚拟时间，不保证届时完成，也不改变随机节拍",
    )

    @model_validator(mode="after")
    def check_outcome(self):
        if self.outcome == "no_event" and (self.event or self.plan_proposals):
            raise ValueError("no_event 不能携带事件或计划")
        if self.outcome == "submit_event" and self.event is None:
            raise ValueError("提交需要具体事件")
        if self.plan_proposals and not self.event:
            raise ValueError("计划修改必须关联事件候选")
        dates = [p.plan_date for p in self.plan_proposals]
        if len(dates) != len(set(dates)):
            raise ValueError("同一日期只能提交一份计划")
        return self
