"""PersonWorldProfile v2 的七栏目读模型。"""

from __future__ import annotations

from pydantic import Field

from ..schemas import StrictModel
from .sections import (
    EpisodeFact,
    FunctionalContextFact,
    GoalCommitmentFact,
    HomeCareFact,
    IdentifierFact,
    InteractionObservationFact,
    InteractionPatternFact,
    PlaceEnvironmentFact,
    PreferenceFact,
    RecurringActivityFact,
    RelationshipHistoryFact,
    RelationshipStandingFact,
    SelfDescriptionFact,
    SelfNarrativeFact,
    SocialTieFact,
    TemporalRhythmFact,
    TrajectoryFact,
    TransitionFact,
    ValueInterpretationFact,
    WorkLearningFact,
)


class IdentityProfileV2(StrictModel):
    identifiers: list[IdentifierFact] = Field(default_factory=list)
    self_descriptions: list[SelfDescriptionFact] = Field(default_factory=list)
    self_narratives: list[SelfNarrativeFact] = Field(default_factory=list)


class LifeContextProfileV2(StrictModel):
    work_and_learning: list[WorkLearningFact] = Field(default_factory=list)
    home_and_care: list[HomeCareFact] = Field(default_factory=list)
    places_and_environment: list[PlaceEnvironmentFact] = Field(default_factory=list)
    functional_context: list[FunctionalContextFact] = Field(default_factory=list)


class SocialWorldProfileV2(StrictModel):
    ties: list[SocialTieFact] = Field(default_factory=list)


class AgencyProfileV2(StrictModel):
    preferences: list[PreferenceFact] = Field(default_factory=list)
    values_and_interpretations: list[ValueInterpretationFact] = Field(default_factory=list)
    goals_and_commitments: list[GoalCommitmentFact] = Field(default_factory=list)


class PracticesProfileV2(StrictModel):
    recurring_activities: list[RecurringActivityFact] = Field(default_factory=list)
    temporal_rhythms: list[TemporalRhythmFact] = Field(default_factory=list)


class LifeCourseProfileV2(StrictModel):
    episodes: list[EpisodeFact] = Field(default_factory=list)
    transitions: list[TransitionFact] = Field(default_factory=list)
    trajectories: list[TrajectoryFact] = Field(default_factory=list)


class RelationshipWithUserProfileV2(StrictModel):
    standing: list[RelationshipStandingFact] = Field(default_factory=list)
    interaction_observations: list[InteractionObservationFact] = Field(default_factory=list)
    interaction_patterns: list[InteractionPatternFact] = Field(default_factory=list)
    history: list[RelationshipHistoryFact] = Field(default_factory=list)


class PersonWorldProfileV2(StrictModel):
    """只保存人物事实；调查覆盖和错误信息属于独立审计报告。"""

    identity: IdentityProfileV2 = Field(default_factory=IdentityProfileV2)
    life_context: LifeContextProfileV2 = Field(default_factory=LifeContextProfileV2)
    social_world: SocialWorldProfileV2 = Field(default_factory=SocialWorldProfileV2)
    agency: AgencyProfileV2 = Field(default_factory=AgencyProfileV2)
    practices: PracticesProfileV2 = Field(default_factory=PracticesProfileV2)
    life_course: LifeCourseProfileV2 = Field(default_factory=LifeCourseProfileV2)
    relationship_with_user: RelationshipWithUserProfileV2 = Field(
        default_factory=RelationshipWithUserProfileV2
    )
