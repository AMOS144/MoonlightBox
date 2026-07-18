from datetime import timedelta

from moonlightbox.events.models import Episode
from moonlightbox.imports.types import ImportedMessage


def segment(
    messages: list[ImportedMessage],
    gap: timedelta,
    semantic_boundaries: set[str] | None = None,
) -> list[Episode]:
    if not messages:
        return []

    ordered = sorted(messages, key=lambda message: message.timestamp)
    boundaries = semantic_boundaries or set()
    episodes: list[Episode] = []
    current: list[ImportedMessage] = [ordered[0]]
    current_reasons: list[str] = []

    for previous, message in zip(ordered, ordered[1:], strict=False):
        reasons: list[str] = []
        if message.timestamp - previous.timestamp > gap:
            reasons.append("time_gap")
        if message.source_id in boundaries:
            reasons.append("semantic_change")

        if reasons:
            episodes.append(_episode(current, current_reasons))
            current = [message]
            current_reasons = reasons
        else:
            current.append(message)

    episodes.append(_episode(current, current_reasons))
    return episodes


def _episode(messages: list[ImportedMessage], reasons: list[str]) -> Episode:
    return Episode(
        message_ids=[message.source_id for message in messages],
        started_at=messages[0].timestamp,
        ended_at=messages[-1].timestamp,
        boundary_reasons=reasons,
    )
