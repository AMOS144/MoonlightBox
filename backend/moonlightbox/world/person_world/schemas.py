"""PersonWorldAgent 内部结构化协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ``ProfileFactPath`` 是 v2 唯一的可编辑事实位置。它不是模型从文本猜出来的标签：
# 页面从已持久化的 v2 栏目路径传入，服务端再用 Scope 内稳定 Claim 绑定具体事实。
ProfileFactPath = Literal[
    "identity.identifiers",
    "identity.self_descriptions",
    "identity.self_narratives",
    "life_context.work_and_learning",
    "life_context.home_and_care",
    "life_context.places_and_environment",
    "life_context.functional_context",
    "social_world.ties",
    "agency.preferences",
    "agency.values_and_interpretations",
    "agency.goals_and_commitments",
    "practices.recurring_activities",
    "practices.temporal_rhythms",
    "life_course.episodes",
    "life_course.transitions",
    "life_course.trajectories",
    "relationship_with_user.standing",
    "relationship_with_user.interaction_observations",
    "relationship_with_user.interaction_patterns",
    "relationship_with_user.history",
]

# 只读旧档案在完成 v2 重编译前仍可能出现在界面。保留这个输入兼容类型是为了让用户可
# 发起“不带 Claim 的暂定纠正”，而不是把旧字段当作新版事实写入路径。
LegacyProfileSection = Literal[
    "identity.names",
    "identity.aliases",
    "identity.self_descriptions",
    "identity.roles",
    "work_and_education",
    "places",
    "social_relationships",
    "preferences",
    "recurring_activities",
    "routine_summary.workdays",
    "routine_summary.weekends",
    "routine_summary.other_patterns",
    "life_phases",
    "relationship_with_user.overview",
    "relationship_with_user.changes_over_time",
    "important_events",
    "unresolved_candidates",
]
ProfileSection = ProfileFactPath | LegacyProfileSection
SubjectKind = Literal[
    "target_person",
    "user",
    "third_person",
    "organization",
    "place",
    "general_topic",
    "unknown",
]
AssertionKind = Literal[
    "self_fact",
    "other_person_fact",
    "general_rule",
    "plan",
    "desire",
    "report",
    "joke",
    "question",
    "unknown",
]
TemporalStatus = Literal[
    "current", "past", "planned", "recurring", "one_off", "timeless", "unknown"
]
Derivation = Literal["direct", "summarized", "inferred"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SelectedProfileStatement(StrictModel):
    """用户在 Profile 界面明确圈选、要求本轮 Agent 核对的陈述。"""

    # 新版页面传入稳定 Claim ID；旧 Profile 没有可追溯 Claim 时保留 None，作为
    # provisional scope，不能由后端根据展示文案自行猜测关联到哪条事实。
    claim_id: str | None = Field(default=None, max_length=36)
    section: str
    entry_id: str | None = None
    module_id: str | None = None
    field_path: str | None = None
    section_label: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1000)
    source_message_ids: list[str] = Field(default_factory=list, max_length=80)


class EvidenceProvenance(StrictModel):
    """一条原始消息进入栏目窗口的单次检索来源。"""

    query_id: str = Field(min_length=1, max_length=100)
    reference_rank: int = Field(ge=0)
    document_rank: int = Field(ge=0)
    context_window_id: str = Field(min_length=1, max_length=120)


class EvidenceMessage(StrictModel):
    message_id: str
    document_id: str
    bundle_id: str
    ordinal: int
    timestamp: datetime
    participant_id: str
    participant_name: str
    participant_role: Literal["self", "target", "other"]
    kind: str
    content: str
    is_primary_match: bool
    source_period: Literal["before", "after", "unknown"] | None = None
    mapping_method: str | None = None
    # 这几项仅记录 LightRAG → 原始消息的定位 provenance，不承载事实语义。
    retrieval_query_ids: list[str] = Field(default_factory=list, max_length=30)
    reference_ranks: list[int] = Field(default_factory=list, max_length=30)
    document_ranks: list[int] = Field(default_factory=list, max_length=30)
    context_window_ids: list[str] = Field(default_factory=list, max_length=30)
    retrieval_provenance: list[EvidenceProvenance] = Field(default_factory=list, max_length=60)


class RevisionUnderstanding(StrictModel):
    # 修改范围也是确认内容，不能在用户确认后再由另一模型自行扩大。
    affected_profile_sections: list[
        Literal[
            "identity",
            "life_context",
            "social_world",
            "agency",
            "practices",
            "life_course",
            "relationship_with_user",
        ]
    ] = Field(
        default_factory=list, description="用户确认拟修改的七栏目键；不因发现相关内容自动扩大范围"
    )
    # 默认只改 Profile。心理理解不能自动成为图谱事实；图修改还需独立批准。
    graph_change_requested: bool = Field(
        default=False,
        description="仅用户明确希望纠正图谱事实时为 true；心理理解或展示改写默认 false，仍需独立批准",
    )
    wrong_interpretation: str = Field(
        min_length=1, max_length=2000, description="当前档案哪里理解有误，明确针对用户选中内容"
    )
    corrected_interpretation: str = Field(
        min_length=1, max_length=2000, description="拟改成的具体理解，供用户确认，不宣称已修改"
    )
    affected_dimensions: list[
        Literal["subject", "predicate", "object", "time", "speech_act", "recurrence", "wording"]
    ] = Field(
        min_length=1,
        max_length=7,
        description="改动涉及主体、谓词、对象、时间、言语行为、重复性或展示措辞；至少一项",
    )
    source_message_ids: list[str] = Field(
        default_factory=list,
        max_length=80,
        description="实际提供的原文消息 UUID；没有可定位来源填 []，不编造",
    )
    open_question: str | None = Field(
        default=None,
        max_length=1000,
        description="提交 understanding 时必须 null；仍需问用户时改用 question 回合",
    )
    summary_for_user: str = Field(
        min_length=1,
        max_length=2000,
        description="直接向用户说明我们理解将如何修改及影响，等待确认",
    )


class RevisionQuestionOption(StrictModel):
    """确有互斥理解时供用户选择的一个选项。"""

    id: str = Field(
        min_length=1, max_length=80, description="本问题中唯一的选项标识，供用户回答关联"
    )
    label: str = Field(min_length=1, max_length=500, description="用户能看懂的简短选项文本")
    effect: str = Field(min_length=1, max_length=1000, description="选择此项将怎样改变修改结果")
    recommended: bool = Field(default=False, description="是否推荐此解释；不是用户已经选择")


class RevisionQuestionTurn(StrictModel):
    """探索期唯一允许等待用户的回合：恰好一个决定性问题。"""

    premise: str = Field(
        min_length=1, max_length=1500, description="提问前说明当前理解与不确定之处"
    )
    decision_key: Literal[
        "subject", "predicate", "object", "time", "recurrence", "speech_act", "wording", "scope"
    ] = Field(description="此问题要确定的修改维度；scope 表示需确认影响范围")
    question: str = Field(
        min_length=1,
        max_length=1200,
        description="本轮唯一一个会改变修改结果的问题，不一次堆多个问题",
    )
    options: list[RevisionQuestionOption] = Field(
        default_factory=list,
        max_length=3,
        description="确有互斥解释时提供最多三个选项；开放问题可 []，不强迫用户接受预设",
    )
    source_message_ids: list[str] = Field(
        default_factory=list, max_length=80, description="提问涉及的已提供原文 UUID；无则 []"
    )


class RevisionScopeProposalItem(StrictModel):
    """一个结构相关候选的展示序号；真正 Claim 身份由服务端 Snapshot 保存。"""

    candidate_item: int = Field(
        ge=1, le=60, description="当前系统给出的相关候选展示序号，不是 Claim UUID，不能猜测"
    )
    reason: str = Field(min_length=1, max_length=1000, description="为什么此候选可能与本次纠正有关")
    effect: str = Field(
        min_length=1, max_length=1000, description="纳入范围将改变什么，等待用户决定"
    )


class RevisionScopeProposal(StrictModel):
    """探索中发现的可选范围，必须等待用户明确纳入或排除。"""

    summary: str = Field(
        min_length=1, max_length=1500, description="拟扩展范围的原因与边界，不直接执行"
    )
    items: list[RevisionScopeProposalItem] = Field(
        min_length=1, max_length=8, description="从已提供候选中提出需要用户确认的条目，至少一项"
    )


class RevisionAgentTurn(StrictModel):
    """探索 Agent 的判别联合，服务端据此保证一个回合只做一件事。"""

    kind: Literal["question", "scope_proposal", "understanding"] = Field(
        description="本回合类型；只填写同名 payload，其他两项必须 null；question 等待用户，其他类型提交理解或范围草案，不构成批准"
    )
    question: RevisionQuestionTurn | None = Field(
        default=None, description="kind=question 时填写一个影响修改结果的明确问题；否则 null"
    )
    scope_proposal: RevisionScopeProposal | None = Field(
        default=None,
        description="kind=scope_proposal 时提出需要用户确认纳入的范围；不能自动扩大修改范围",
    )
    understanding: RevisionUnderstanding | None = Field(
        default=None,
        description="kind=understanding 时总结待确认理解，open_question 必须 null；不代表用户已批准 Profile 或 Graph Patch",
    )

    @model_validator(mode="after")
    def validate_variant(self) -> RevisionAgentTurn:
        if (
            self.kind == "question"
            and self.question is not None
            and self.scope_proposal is None
            and self.understanding is None
        ):
            return self
        if (
            self.kind == "scope_proposal"
            and self.scope_proposal is not None
            and self.question is None
            and self.understanding is None
        ):
            return self
        if (
            self.kind == "understanding"
            and self.understanding is not None
            and self.question is None
            and self.scope_proposal is None
        ):
            if self.understanding.open_question is None:
                return self
        raise ValueError("RevisionAgentTurn 的 kind 必须与唯一 payload 对应")


class GraphOperationDraft(StrictModel):
    operation_id: str = Field(
        min_length=1, max_length=80, description="本提案内唯一的操作标签，不是已有数据库 ID"
    )
    operation_type: Literal[
        "CREATE_ENTITY",
        "UPDATE_ENTITY",
        "DELETE_ENTITY",
        "CREATE_RELATION",
        "UPDATE_RELATION",
        "DELETE_RELATION",
        "MERGE_ENTITIES",
    ] = Field(
        description="实体操作填写 entity_name；关系操作填写 source_entity/target_entity；合并填写 source_entities/target_entity；只交待批准提案"
    )
    source_entity: str | None = Field(
        default=None, max_length=500, description="关系起点的已有实体完整名称；非关系操作 null"
    )
    target_entity: str | None = Field(
        default=None,
        max_length=500,
        description="关系终点或合并后的目标实体完整名称；其他操作 null",
    )
    source_entities: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="合并源实体名称；非合并操作 []，不扩展用户批准范围",
    )
    entity_name: str | None = Field(
        default=None, max_length=500, description="实体增删改针对的完整名称；关系操作 null"
    )
    before_description: str | None = Field(
        default=None, max_length=4000, description="实际读取的修改前描述；未知 null，不编造旧值"
    )
    after_description: str | None = Field(
        default=None,
        max_length=4000,
        description="拟写入的完整实体描述，不是修改指令；删除操作 null",
    )
    entity_type: str | None = Field(
        default=None,
        max_length=120,
        description="拟创建或更新实体类型，沿用图谱约定；其他操作 null",
    )
    relation_description: str | None = Field(
        default=None, max_length=4000, description="拟写入的完整关系描述；删除或实体操作 null"
    )
    relation_keywords: str | None = Field(
        default=None, max_length=1000, description="关系内容关键词；无关系写入时 null"
    )
    reason: str = Field(
        min_length=1, max_length=2000, description="此操作如何落实已确认纠正及必要性"
    )
    source_message_ids: list[str] = Field(
        default_factory=list,
        max_length=80,
        description="相关已读取的原文消息 UUID；没有则 []，不用实体名替代",
    )
    precondition_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="只沿用系统提供的精确前置状态 hash；未提供填 null，不计算或猜测",
    )
    cascade: bool = Field(
        default=False, description="仅用户批准删除实体及其附带关系时为 true；不能借此扩大删除范围"
    )


class GraphPatchDraft(StrictModel):
    """只有 Profile Patch 批准后才能生成的第二阶段图谱提案。"""

    graph_operations: list[GraphOperationDraft] = Field(
        default_factory=list,
        max_length=40,
        description="需要用户最终确认的有序图谱操作提案；只覆盖已批准的纠正范围，提交不会执行",
    )
    regression_queries: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="用于验证纠正是否生效及是否损伤相关事实的具体查询；无图修改可 []",
    )


class RegressionCheck(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    passed: bool
    explanation: str = Field(min_length=1, max_length=1500)
    # 下列字段由验证 Job 从实际 Sidecar 调用确定性回填；模型只判断是否通过及其理由。
    # 它们保留在 ChangeSet 执行结果中，供审核界面展示“预期 / 实际 / 来源”。
    expected_change: str = Field(default="", max_length=1500)
    actual_context_excerpt: str = Field(default="", max_length=4000)
    source_references: list[str] = Field(default_factory=list, max_length=30)


class RegressionValidationBatch(StrictModel):
    checks: list[RegressionCheck] = Field(default_factory=list, max_length=30)


class RegressionAssessmentCheck(StrictModel):
    """模型只负责对实际检索结果做判定，不能伪造展示用的预期、实际或来源。"""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="原样使用已批准且实际调用 read_regression_query 的查询，不改写",
    )
    passed: bool = Field(description="实际检索结果是否符合预期纠正；不能仅因请求成功就判通过")
    explanation: str = Field(
        min_length=1,
        max_length=1500,
        description="结合实际检索结果解释通过或失败，不伪造工具未返回的事实",
    )


class RegressionAssessmentBatch(StrictModel):
    checks: list[RegressionAssessmentCheck] = Field(
        default_factory=list,
        max_length=30,
        description="覆盖本任务全部批准查询，每项一次；每项须先读取真实候选图结果",
    )
