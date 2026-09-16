"""Director 的处境、表达与记忆契约；语义由模型判断，执行结果由回执确认。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Appraisal(Contract):
    interpretation: str = Field(
        description="人物如何理解当前处境，用自然语言概括，不写系统调度分析"
    )
    stakes: list[str] = Field(
        default_factory=list, description="此事牵涉的需要、目标或关系；无明确事项填 []"
    )
    expected_consequences: str | None = Field(
        default=None, description="人物预期的后果，不当作已经发生的结果；未形成预期填 null"
    )
    causal_understanding: str | None = Field(
        default=None, description="人物对原因、责任和可控性的理解；允许保留不确定"
    )
    coping_potential: str | None = Field(
        default=None, description="人物认为自己有哪些应对资源、选择或限制"
    )
    open_questions: list[str] = Field(
        default_factory=list, description="仍影响判断的未解问题；无则 []，不是强制追问用户"
    )


class Concern(Contract):
    concern_ref: str | None = Field(
        default=None,
        description="更新已有关切时使用当前状态给出的引用；新关切填 null，不自行生成 ID",
    )
    focus: str = Field(description="这件关切具体关于什么，不是泛化性格标签")
    subjects: list[Literal["user", "target", "other"]] = Field(
        description="关切涉及的人，target 为被扮演人物，user 为聊天用户"
    )
    appraisal: Appraisal = Field(description="人物对这件事的当前评估，解释感受与行动倾向")
    feelings: str | None = Field(
        default=None, description="此事引起的具体感受；无变化或不清楚可 null"
    )
    action_tendency: str | None = Field(
        default=None, description="倾向靠近、回避、解释、等待等，不表示行动已执行"
    )
    intention: str | None = Field(
        default=None, description="人物目前想做什么；不冒充已提交的安排或已经完成的动作"
    )
    action_ref: str | None = Field(
        default=None, description="已有行动回执引用；本次动作可填 this_decision"
    )
    depends_on_expression: bool = Field(
        default=False, description="此状态变化是否仅在本次回复实际发送后才成立；非表达动作填 false"
    )
    status: Literal["active", "parked", "resolved"] = Field(
        default="active",
        description="active 正在关切，parked 暂放，resolved 已解决；不能把暂时没谈当作已解决",
    )
    latest_change: str | None = Field(default=None, description="相较已有关切具体发生了什么变化")
    resume_when: str | None = Field(
        default=None, description="暂放后应在什么条件下重新关注，不是可执行的定时器"
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description="当前输入或工具提供的来源引用；不编造引用，无可定位来源填 []",
    )


class SubjectiveStateUpdate(Contract):
    """只提交变化；引用已有事项进行修订，不用重新生成整个人物状态。"""

    mood: str | None = Field(
        default=None, description="整体心境的本轮变化；null 表示不更新，不是清空"
    )
    concerns: list[Concern] = Field(
        default_factory=list,
        description="只提交新关切或发生变化的关切，按 concern_ref 更新；[] 表示没有变化，不删除旧关切",
    )


class ExpressionTask(Contract):
    purpose: str = Field(
        min_length=1, description="表达目的；历史表达任务兼容字段，当前 Director 提交实际 reply"
    )
    speech_mode: Literal["reply", "proactive", "delayed_reply"] = Field(
        default="reply", description="reply 接话、proactive 主动表达、delayed_reply 延迟接话"
    )
    respond_to_refs: list[str] = Field(
        default_factory=list, description="本次表达处理的已读用户消息引用，不包括尚未读到的新消息"
    )
    must_convey: list[str] = Field(
        default_factory=list, description="必须传达的具体内容要点，不是逐字台词"
    )
    optional_content: list[str] = Field(
        default_factory=list, description="可选补充内容，不强制塞入全部要点"
    )
    stance: str | None = Field(default=None, description="这次互动的态度与立场")
    interaction_style: str | None = Field(default=None, description="适合当前关系和情境的互动方式")
    feelings: str | None = Field(default=None, description="当前感受如何影响表达，不重做心理判断")
    boundaries: list[str] = Field(default_factory=list, description="不能越过的表达和事实边界")
    context_refs: list[str] = Field(
        default_factory=list, description="理解表达需要的已有上下文引用，不编造"
    )

    @model_validator(mode="before")
    @classmethod
    def historical_fields(cls, data):
        """只在读取旧任务时转换；新 Schema 不暴露重复字段。"""
        if isinstance(data, dict):
            data = dict(data)
            for old, new in (("content_points", "must_convey"), ("source_refs", "context_refs")):
                if old in data:
                    data.setdefault(new, data.pop(old))
        return data


class MemoryProposal(Contract):
    subject: Literal["user", "target", "relationship", "other"] = Field(
        description="事实主体，不能用发送者代替被描述的人"
    )
    summary: str = Field(min_length=1, description="值得后续记住的具体内容，保留时间条件与不确定性")
    basis: Literal["user_report", "observed", "inferred", "simulation"] = Field(
        description="user_report 用户自述；observed 已观察事实；inferred 推断；simulation 模拟经历，不混作真实历史"
    )
    source_refs: list[str] = Field(
        min_length=1, description="至少一个当前已提供的来源引用，不能自行创造证据"
    )
    depends_on_expression: bool = Field(
        default=False, description="仅在本轮表达实际发出后成立的承诺或关系变化填 true"
    )
    supersedes_ref: str | None = Field(
        default=None, description="明确取代的已有记忆引用；新记忆填 null"
    )


class InputResolution(Contract):
    message_ref: str = Field(
        description="当前已读待处理用户消息的引用；不是任意事件 ID，不能填写未读或不存在的消息"
    )
    status: Literal["awaiting_response", "completed", "no_response_needed"] = Field(
        description="awaiting_response 仍待回复；completed 本轮实际回复已处理；no_response_needed 判断无需回复。每条消息只能提交一次状态"
    )


class OpenTopic(Contract):
    topic: str = Field(description="历史末端尚未结束的话题名称")
    summary: str = Field(description="接续话题需要知道的进展，不复述整段聊天")
    resume_when: str = Field(description="何时或什么情况下适合继续，不自动创建唤醒")
    source_refs: list[str] = Field(
        default_factory=list, description="历史材料中的来源消息引用；不编造"
    )


class InitialCommitment(Contract):
    participants: list[str] = Field(
        description="已确认承诺的参与者，不把旁观者或消息发送者自动视为参与方"
    )
    description: str = Field(description="历史末端仍有效的具体承诺；区分建议、愿望和确认")
    timing: str = Field(description="原材料支持的时间表达，不知道精确时间就保留模糊描述")
    conditions: str = Field(description="承诺成立的条件或限制；无则空字符串")
    source_refs: list[str] = Field(default_factory=list, description="支持承诺的历史消息引用")


class InitialState(Contract):
    subjective_state: SubjectiveStateUpdate = Field(
        description="仅建立分支起点心理状态；关切均为新项，不引用旧动作或依赖本轮表达"
    )
    open_conversation_threads: list[OpenTopic] = Field(
        description="从历史尾部整理仍开放的话题；没有则 []"
    )
    active_commitments: list[InitialCommitment] = Field(
        description="起点仍有效的已确认承诺；没有则 []，不自动生成新约定"
    )


class InvestigationWork(Contract):
    read_ranges: list[str] = Field(
        default_factory=list, description="已调查的历史范围和引用，供恢复后避免重复读取"
    )
    tentative_understanding: str = Field(
        default="", description="当前暂定理解，不是已提交的起点状态"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="恢复后还要调查的具体问题；每次保存完整笔记，[] 表示无待查问题",
    )
