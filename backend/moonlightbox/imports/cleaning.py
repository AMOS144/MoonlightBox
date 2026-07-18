from dataclasses import dataclass
from datetime import timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind


@dataclass
class TrainingUnit:
    sender: str
    content: str
    source_ids: list[str]


@dataclass
class CleanResult:
    training_units: list[TrainingUnit]


def clean_messages(messages: list[ImportedMessage]) -> CleanResult:
    units: list[TrainingUnit] = []
    previous_message: ImportedMessage | None = None

    for current in messages:
        can_merge = (
            previous_message is not None
            and current.kind is MessageKind.TEXT
            and previous_message.kind is MessageKind.TEXT
            and current.sender == previous_message.sender
            and len(current.content) <= 8
            and len(previous_message.content) <= 8
            and current.timestamp - previous_message.timestamp <= timedelta(seconds=30)
        )
        if can_merge:
            units[-1].content = f"{units[-1].content}\n{current.content}"
            units[-1].source_ids.append(current.source_id)
        else:
            units.append(
                TrainingUnit(
                    sender=current.sender,
                    content=current.content,
                    source_ids=[current.source_id],
                )
            )
        previous_message = current

    return CleanResult(training_units=units)
