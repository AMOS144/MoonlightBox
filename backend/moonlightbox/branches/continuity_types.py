from typing import Literal

from pydantic import BaseModel, Field, field_validator

RELATIONSHIP_STATE_AXES = frozenset(
    {
        "trust",
        "intimacy",
        "reciprocity",
        "safety",
        "conflict",
        "boundary_pressure",
    }
)
EMOTIONAL_TENDENCY_AXES = frozenset(
    {
        "warmth",
        "anger",
        "sadness",
        "anxiety",
        "disappointment",
        "hope",
        "guardedness",
        "calm",
    }
)

MemoryKind = Literal[
    "fact",
    "experience",
    "self_narrative",
    "belief",
    "reflection",
]
EvidenceStance = Literal["support", "oppose"]


class MemoryCandidate(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=1, le=10)
    evidence_message_ids: tuple[str, ...] = Field(min_length=1)
    stance: EvidenceStance = "support"
    source_role: Literal["user", "digital_human", "interaction"] = "interaction"


class StateDeltaProposal(BaseModel):
    relationship_delta: dict[str, float] = Field(default_factory=dict)
    emotional_delta: dict[str, float] = Field(default_factory=dict)
    user_model_updates: dict[str, str] = Field(default_factory=dict)
    supporting_candidate_indexes: tuple[int, ...] = ()

    @field_validator("relationship_delta", mode="before")
    @classmethod
    def _known_relationship_axes(cls, value: object) -> object:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): item
            for key, item in value.items()
            if str(key) in RELATIONSHIP_STATE_AXES
        }

    @field_validator("emotional_delta", mode="before")
    @classmethod
    def _known_emotional_axes(cls, value: object) -> object:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): item
            for key, item in value.items()
            if str(key) in EMOTIONAL_TENDENCY_AXES
        }

    @field_validator("user_model_updates", mode="before")
    @classmethod
    def _user_model_requires_belief_aggregation(cls, value: object) -> dict[str, str]:
        # Online free-form overwrite is intentionally disabled. Observations
        # about the user must travel through evidence-backed beliefs first.
        del value
        return {}


class MemoryProposal(BaseModel):
    candidates: tuple[MemoryCandidate, ...] = ()
    state_delta: StateDeltaProposal = Field(default_factory=StateDeltaProposal)


class MemoryReviewResult(BaseModel):
    verdict: Literal["approve", "reject"]
    approved_candidate_indexes: tuple[int, ...] = ()
    approved_state_delta: StateDeltaProposal | None = None
    rejected_reasons: tuple[str, ...] = ()
