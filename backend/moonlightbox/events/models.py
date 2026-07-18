from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Episode:
    message_ids: list[str]
    started_at: datetime
    ended_at: datetime
    boundary_reasons: list[str]
