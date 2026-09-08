from dataclasses import dataclass
from datetime import datetime

from moonlightbox.graph.models import TemporalFact
from moonlightbox.imports.types import ImportedMessage


@dataclass(frozen=True)
class TemporalContext:
    messages: list[ImportedMessage]
    facts: list[TemporalFact]
    cutoff: datetime


class TemporalContextBuilder:
    def build(
        self,
        messages: list[ImportedMessage],
        facts: list[TemporalFact],
        cutoff: datetime,
    ) -> TemporalContext:
        valid_messages = sorted(
            (message for message in messages if message.timestamp <= cutoff),
            key=lambda message: message.timestamp,
        )
        valid_facts = [
            fact
            for fact in facts
            if fact.valid_from <= cutoff and (fact.valid_to is None or cutoff < fact.valid_to)
        ]
        return TemporalContext(valid_messages, valid_facts, cutoff)
