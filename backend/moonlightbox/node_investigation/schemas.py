"""模型只填写调查语义；作用域、ID、版本、审批与精确时间由宿主绑定。"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArgs(Strict):
    pass


class ReadArgs(Strict):
    cursor: int = Field(
        ge=0,
        description="使用概览的 cursor 或上一页 next_cursor。可重读，不能跳过未读内容",
    )
    limit: int = Field(default=80, ge=1, le=160, description="连续消息条数上限；页边界不是经历边界")


class ContextArgs(Strict):
    message_refs: list[str] = Field(
        min_length=1, max_length=12, description="原文工具返回的真实消息引用"
    )
    radius: int = Field(
        default=8, ge=0, le=30, description="每个引用前后补读的消息数，不推进顺序阅读进度"
    )


class SearchArgs(Strict):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="用自然语言描述要寻找的经历、互动或疑点；可包含时间线索，不是正则",
    )
    limit: int = Field(default=8, ge=1, le=20, description="最多检索片段数；原文可另行展开")


class WorkArgs(Strict):
    understanding: str = Field(
        max_length=16000, description="当前对时间线的理解，保留跨页尚未结束的经历"
    )
    open_questions: list[str] = Field(
        default_factory=list, max_length=40, description="尚待定位或澄清的问题"
    )
    next_steps: list[str] = Field(
        default_factory=list, max_length=40, description="后续调查方向；不得冒充已执行动作"
    )


class Boundary(Strict):
    label: str = Field(
        min_length=1, max_length=200, description="面向用户的起点说明，例如出发前、恢复联系后"
    )
    kind: Literal["message"] = Field(description="使用真实消息精确定位起点前后")
    message_ref: str | None = Field(
        default=None, description="message 类型必填，从工具返回的原始消息引用中选择"
    )
    side: Literal["before", "after"] = Field(
        default="before", description="包含该条消息选 after，不包含选 before；同秒消息保持稳定顺序"
    )
    time_hint: str | None = Field(
        default=None, max_length=500, description="第一版不使用空档边界，必须为空"
    )

    @model_validator(mode="after")
    def check_kind(self):
        if not self.message_ref or self.time_hint:
            raise ValueError("message 边界需要 message_ref，不填写 time_hint")
        return self


class CandidateArgs(Strict):
    candidate_ref: str | None = Field(
        default=None, description="更新已有经历时使用工具返回的候选引用；新经历省略"
    )
    title: str = Field(
        min_length=1, max_length=200, description="具体经历的简短标题，不限定事件分类"
    )
    summary: str = Field(
        min_length=1,
        max_length=4000,
        description="经历的前后联系；写给用户直接阅读的 2–4 句简述，从用户视角叙述，不展开推理过程；区分发生时间与后来提及的时间",
    )
    occurrence: str = Field(
        max_length=800, description="经历实际发生的时间或范围，可未知，不用提及时间代替"
    )
    why_branch: str = Field(
        max_length=2000,
        description="为什么从这里继续会形成有意义的另一条生活路径；面向用户的一两句说明",
    )
    source_refs: list[str] = Field(
        default_factory=list,
        max_length=60,
        description="支持理解的原文引用；用户补充或合理推测不必伪造消息",
    )
    user_input_refs: list[str] = Field(
        default_factory=list, max_length=30, description="所依据的用户补充 ID，来自工作台输入"
    )
    interpretation: str = Field(
        default="",
        max_length=3000,
        description="综合上下文的理解，仅作调查留存，不直接展示给用户；不要把猜测写成已确认经历",
    )
    uncertainties: list[str] = Field(
        default_factory=list, max_length=20, description="会影响理解或起点选择的未决问题"
    )
    boundaries: list[Boundary] = Field(
        default_factory=list,
        max_length=1,
        description="唯一的可选消息起点；用户只选择候选，不再编辑候选内部边界",
    )
    status: Literal["investigating", "waiting", "ready", "excluded"] = Field(
        default="investigating", description="调查状态，不是用户确认状态；ready 需有边界"
    )
    change_reason: str = Field(
        min_length=1, max_length=1000, description="本次新增或修改的公开理由，供工作台呈现"
    )


class TurnResult(Strict):
    outcome: Literal["waiting_for_user", "completed"] = Field(
        description="第一版必须提交 completed；waiting_for_user 仅用于兼容旧检查点"
    )
    summary: str = Field(
        min_length=1, max_length=6000, description="本轮调查得到的理解及局限，用用户读得懂的话说明"
    )
    question: str | None = Field(
        default=None,
        max_length=2000,
        description="第一版不向用户提问，必须为空；字段仅用于兼容旧检查点",
    )

    @model_validator(mode="after")
    def check_question(self):
        if (self.outcome == "waiting_for_user") != bool(self.question and self.question.strip()):
            raise ValueError("等待用户时必须有一个问题；完成时不填写 question")
        return self


class StartArgs(Strict):
    timezone: str = "Asia/Shanghai"


class ScopeArgs(Strict):
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("开始日期不能晚于结束日期")
        return self


class InputArgs(Strict):
    request_id: str = Field(min_length=1, max_length=80)
    text: str = Field(default="", max_length=8000)
    kind: Literal["supplement", "answer", "skip", "unknown", "correction"] = "supplement"
    question_id: str | None = None
    message_refs: list[str] = Field(default_factory=list, max_length=12)


class PreviewArgs(Strict):
    candidate_revision: int = Field(ge=1)
    option_index: int = Field(ge=0)
    exact_time: str | None = None


class ConfirmArgs(Strict):
    preview_hash: str
    continue_investigating: bool
