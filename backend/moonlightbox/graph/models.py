from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TemporalFact:
    id: str
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: datetime | None
    evidence_ids: list[str]


@dataclass(frozen=True)
class EventEdge:
    source_id: str
    target_id: str
    relation: str
    evidence_ids: list[str]
