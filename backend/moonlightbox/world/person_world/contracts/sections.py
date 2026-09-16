"""七个 Section Agent 的专属结构化输出模型。

模型只表达各栏目的事实类型；字段存在性、证据归属和跨栏目唯一性由后端 Contract 校验。
它们不会把所有栏目压回一个泛化的 ``AtomicClaimDraft`` 输出。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from ..schemas import AssertionKind, Derivation, StrictModel, TemporalStatus

PrimaryDomain = Literal[
    "identity",
    "life_context",
    "social_world",
    "agency",
    "practices",
    "life_course",
    "relationship_with_user",
]
ResolutionBasis = Literal[
    "first_person_speaker",
    "second_person_addressee",
    "named_reference",
    "quoted_context",
    "context_resolved",
    "unknown",
]
RelationshipDirection = Literal["target_to_user", "user_to_target", "mutual"]
RecurrenceBasis = Literal["explicit_statement", "observed_pattern"]
Sensitivity = Literal["ordinary", "restricted"]


class SubjectBinding(StrictModel):
    grammatical_subject: str | None = Field(default=None, max_length=500)
    resolved_participant_id: str | None = Field(default=None, max_length=36)
    resolution_basis: ResolutionBasis = "unknown"


class EvidenceBackedFact(StrictModel):
    """各栏目事实共同的证据与时间信封，不作为独立 Agent 输出列表使用。"""

    statement: str = Field(min_length=1, max_length=1200)
    # 原子 Claim 落库后由 Coordinator 回写；模型不能自行编造。它是用户圈选和修订
    # Scope 的稳定事实身份，而不是展示文案的哈希或文本匹配结果。
    claim_id: str | None = Field(default=None, min_length=36, max_length=36)
    speaker_message_id: str = Field(min_length=1, max_length=36)
    evidence_message_ids: list[str] = Field(min_length=1, max_length=30)
    subject_binding: SubjectBinding
    assertion_kind: AssertionKind = "unknown"
    # ``direct`` 表示结论可由一条已核对原话直接转述；``inferred`` 表示 Agent 在
    # 多条已读上下文之间作出的受约束归纳。两者必须让审核者能够一眼区分，不能把
    # 推断伪装成当事人的原话。
    derivation: Derivation = "direct"
    inference_rationale: str | None = Field(default=None, max_length=1200)
    temporal_status: TemporalStatus = "unknown"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    qualifiers: list[str] = Field(default_factory=list, max_length=20)
    related_fact_ids: list[str] = Field(default_factory=list, max_length=20)
    sensitivity: Sensitivity = "ordinary"

    @model_validator(mode="after")
    def require_auditable_inference(self) -> EvidenceBackedFact:
        """推断只允许建立在多条真实消息上，约束结构而不解析聊天正文。"""

        if self.derivation != "inferred":
            return self
        if len(set(self.evidence_message_ids)) < 2:
            raise ValueError("上下文推断至少需要两条不同的证据消息")
        if self.inference_rationale is None or not self.inference_rationale.strip():
            raise ValueError("上下文推断必须说明证据之间的推理路径")
        return self


class IdentifierFact(EvidenceBackedFact):
    identifier: str = Field(min_length=1, max_length=500)
    identifier_kind: Literal["name", "alias", "self_reference"]


class SelfDescriptionFact(EvidenceBackedFact):
    description: str = Field(min_length=1, max_length=1000)


class SelfNarrativeFact(EvidenceBackedFact):
    narrative: str = Field(min_length=1, max_length=1200)
    expressed_meaning: str = Field(min_length=1, max_length=1000)


class IdentitySectionResult(StrictModel):
    section: Literal["identity"] = "identity"
    identifiers: list[IdentifierFact] = Field(default_factory=list, max_length=24)
    self_descriptions: list[SelfDescriptionFact] = Field(default_factory=list, max_length=24)
    self_narratives: list[SelfNarrativeFact] = Field(default_factory=list, max_length=24)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class WorkLearningFact(EvidenceBackedFact):
    fact_type: Literal["role", "employer_association", "work_condition", "education_history"]
    detail: str = Field(min_length=1, max_length=1200)


class HomeCareFact(EvidenceBackedFact):
    detail: str = Field(min_length=1, max_length=1200)


class PlaceEnvironmentFact(EvidenceBackedFact):
    place_name: str = Field(min_length=1, max_length=500)
    association: Literal["lives", "works", "studies", "care", "access", "other"]


class FunctionalContextFact(EvidenceBackedFact):
    detail: str = Field(min_length=1, max_length=1200)


class LifeContextSectionResult(StrictModel):
    section: Literal["life_context"] = "life_context"
    work_and_learning: list[WorkLearningFact] = Field(default_factory=list, max_length=32)
    home_and_care: list[HomeCareFact] = Field(default_factory=list, max_length=24)
    places_and_environment: list[PlaceEnvironmentFact] = Field(default_factory=list, max_length=24)
    functional_context: list[FunctionalContextFact] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class SocialTieFact(EvidenceBackedFact):
    other_party: str = Field(min_length=1, max_length=500)
    relation_type: str = Field(min_length=1, max_length=160)
    direction: Literal["target_to_other", "other_to_target", "mutual"]
    relation_dimension: Literal["structure", "quality", "function"]


class SocialWorldSectionResult(StrictModel):
    section: Literal["social_world"] = "social_world"
    ties: list[SocialTieFact] = Field(default_factory=list, max_length=40)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class PreferenceFact(EvidenceBackedFact):
    preference: str = Field(min_length=1, max_length=1000)


class ValueInterpretationFact(EvidenceBackedFact):
    interpretation: str = Field(min_length=1, max_length=1200)


class GoalCommitmentFact(EvidenceBackedFact):
    goal: str = Field(min_length=1, max_length=1200)
    status: Literal["wish", "planned", "committed", "completed", "cancelled", "unknown"]


class AgencySectionResult(StrictModel):
    section: Literal["agency"] = "agency"
    preferences: list[PreferenceFact] = Field(default_factory=list, max_length=32)
    values_and_interpretations: list[ValueInterpretationFact] = Field(
        default_factory=list, max_length=32
    )
    goals_and_commitments: list[GoalCommitmentFact] = Field(default_factory=list, max_length=32)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class RecurringActivityFact(EvidenceBackedFact):
    activity: str = Field(min_length=1, max_length=1000)
    recurrence_basis: RecurrenceBasis
    occurrence_dates: list[str] = Field(default_factory=list, max_length=60)
    condition: str | None = Field(default=None, max_length=500)


class TemporalRhythmFact(RecurringActivityFact):
    time_expression: str = Field(min_length=1, max_length=300)


class PracticesSectionResult(StrictModel):
    section: Literal["practices"] = "practices"
    recurring_activities: list[RecurringActivityFact] = Field(default_factory=list, max_length=32)
    temporal_rhythms: list[TemporalRhythmFact] = Field(default_factory=list, max_length=32)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class EpisodeFact(EvidenceBackedFact):
    event: str = Field(min_length=1, max_length=1200)


class TransitionFact(EvidenceBackedFact):
    before: str = Field(min_length=1, max_length=1000)
    after: str = Field(min_length=1, max_length=1000)
    before_evidence_message_ids: list[str] = Field(min_length=1, max_length=30)
    after_evidence_message_ids: list[str] = Field(min_length=1, max_length=30)


class TrajectoryFact(EvidenceBackedFact):
    trajectory: str = Field(min_length=1, max_length=1200)
    observation_window: str = Field(min_length=1, max_length=500)


class LifeCourseSectionResult(StrictModel):
    section: Literal["life_course"] = "life_course"
    episodes: list[EpisodeFact] = Field(default_factory=list, max_length=32)
    transitions: list[TransitionFact] = Field(default_factory=list, max_length=24)
    trajectories: list[TrajectoryFact] = Field(default_factory=list, max_length=24)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class RelationshipStandingFact(EvidenceBackedFact):
    relationship: str = Field(min_length=1, max_length=1200)
    direction: RelationshipDirection
    # 关系位置、边界/承诺与单方体验不能混为“关系好坏”。该枚举只约束结构位置，
    # 不替 Agent 判断一句话究竟是否足以证明亲密或冲突。
    relationship_dimension: Literal["structure", "boundary", "commitment", "quality"]


class InteractionObservationFact(EvidenceBackedFact):
    """一条方向明确的互动观察，不把单次互动夸大成关系标签或长期模式。"""

    observation: str = Field(min_length=1, max_length=1200)
    direction: RelationshipDirection


class InteractionPatternFact(EvidenceBackedFact):
    pattern: str = Field(min_length=1, max_length=1200)
    direction: RelationshipDirection
    recurrence_basis: RecurrenceBasis
    occurrence_dates: list[str] = Field(default_factory=list, max_length=60)


class RelationshipHistoryFact(EvidenceBackedFact):
    before: str = Field(min_length=1, max_length=1000)
    after: str = Field(min_length=1, max_length=1000)
    direction: RelationshipDirection
    before_evidence_message_ids: list[str] = Field(min_length=1, max_length=30)
    after_evidence_message_ids: list[str] = Field(min_length=1, max_length=30)


class RelationshipWithUserSectionResult(StrictModel):
    section: Literal["relationship_with_user"] = "relationship_with_user"
    standing: list[RelationshipStandingFact] = Field(default_factory=list, max_length=32)
    interaction_observations: list[InteractionObservationFact] = Field(
        default_factory=list, max_length=48
    )
    interaction_patterns: list[InteractionPatternFact] = Field(default_factory=list, max_length=32)
    history: list[RelationshipHistoryFact] = Field(default_factory=list, max_length=24)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


type SectionResult = (
    IdentitySectionResult
    | LifeContextSectionResult
    | SocialWorldSectionResult
    | AgencySectionResult
    | PracticesSectionResult
    | LifeCourseSectionResult
    | RelationshipWithUserSectionResult
)

_SECTION_RESULT_MODELS: dict[str, type[SectionResult]] = {
    "identity": IdentitySectionResult,
    "life_context": LifeContextSectionResult,
    "social_world": SocialWorldSectionResult,
    "agency": AgencySectionResult,
    "practices": PracticesSectionResult,
    "life_course": LifeCourseSectionResult,
    "relationship_with_user": RelationshipWithUserSectionResult,
}


def section_result_model(section: str) -> type[SectionResult]:
    try:
        return _SECTION_RESULT_MODELS[section]
    except KeyError as error:
        raise ValueError(f"未知 PersonWorld 顶层栏目: {section}") from error
