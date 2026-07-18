from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StateTransition:
    at: datetime
    relationship_status: str
    emotions: dict[str, str]
    open_loops: list[str]
    evidence_ids: list[str]


@dataclass(frozen=True)
class StateSnapshot:
    at: datetime
    relationship_status: str
    emotions: dict[str, str]
    open_loops: list[str]
    evidence_ids: list[str]


class StateBuilder:
    def build(
        self,
        transitions: list[StateTransition],
        cutoff: datetime,
    ) -> StateSnapshot:
        valid = sorted(
            (transition for transition in transitions if transition.at <= cutoff),
            key=lambda transition: transition.at,
        )
        if not valid:
            return StateSnapshot(cutoff, "未知", {}, [], [])
        latest = valid[-1]
        return StateSnapshot(
            at=cutoff,
            relationship_status=latest.relationship_status,
            emotions=latest.emotions,
            open_loops=latest.open_loops,
            evidence_ids=latest.evidence_ids,
        )
