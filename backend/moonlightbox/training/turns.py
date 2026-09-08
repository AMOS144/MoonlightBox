from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Literal

from moonlightbox.imports.types import ImportedMessage, MessageKind


@dataclass(frozen=True)
class TurnBubble:
    content: str
    delay_ms: int
    source_id: str
    timestamp: datetime
    kind: Literal["text", "sticker", "emoji"] = "text"
    asset_id: str | None = None


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    started_at: datetime
    ended_at: datetime
    bubbles: tuple[TurnBubble, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class TurnGroupingProfile:
    sender: str
    threshold_seconds: float
    sample_count: int


def build_conversation_turns(
    messages: list[ImportedMessage],
) -> tuple[list[ConversationTurn], dict[str, TurnGroupingProfile]]:
    eligible = sorted(
        (message for message in messages if _is_training_message(message)),
        key=lambda message: message.timestamp,
    )
    profiles = _build_profiles(eligible)
    turns: list[ConversationTurn] = []
    current: list[ImportedMessage] = []
    previous_before_turn: ImportedMessage | None = None
    for message in eligible:
        if current and _starts_new_turn(current[-1], message, profiles):
            turns.extend(_to_turns(current, previous_before_turn))
            previous_before_turn = current[-1]
            current = []
        current.append(message)
    if current:
        turns.extend(_to_turns(current, previous_before_turn))
    return turns, profiles


def _build_profiles(
    messages: list[ImportedMessage],
) -> dict[str, TurnGroupingProfile]:
    gaps: dict[str, list[float]] = {}
    for previous, current in zip(messages, messages[1:], strict=False):
        if previous.sender != current.sender:
            continue
        gap = max(0.0, (current.timestamp - previous.timestamp).total_seconds())
        gaps.setdefault(current.sender, []).append(gap)
    senders = {message.sender for message in messages}
    profiles: dict[str, TurnGroupingProfile] = {}
    for sender in senders:
        samples = gaps.get(sender, [])
        threshold = 30.0 if len(samples) < 3 else median(samples) * 2
        profiles[sender] = TurnGroupingProfile(
            sender=sender,
            threshold_seconds=max(3.0, min(180.0, threshold)),
            sample_count=len(samples),
        )
    return profiles


def _starts_new_turn(
    previous: ImportedMessage,
    current: ImportedMessage,
    profiles: dict[str, TurnGroupingProfile],
) -> bool:
    if previous.sender != current.sender:
        return True
    gap = (current.timestamp - previous.timestamp).total_seconds()
    return gap > profiles[current.sender].threshold_seconds


def _to_turns(
    messages: list[ImportedMessage],
    previous_message: ImportedMessage | None,
) -> list[ConversationTurn]:
    return [_to_turn(messages, previous_message)]


def _to_turn(
    messages: list[ImportedMessage],
    previous_message: ImportedMessage | None,
) -> ConversationTurn:
    bubbles = tuple(
        TurnBubble(
            content=message.content,
            delay_ms=_bubble_delay_ms(messages, index, previous_message),
            source_id=message.source_id,
            timestamp=message.timestamp,
            kind=_bubble_kind(message),
            asset_id=_asset_id(message),
        )
        for index, message in enumerate(messages)
    )
    return ConversationTurn(
        speaker=messages[0].sender,
        started_at=messages[0].timestamp,
        ended_at=messages[-1].timestamp,
        bubbles=bubbles,
        source_ids=tuple(message.source_id for message in messages),
    )


def _is_training_message(message: ImportedMessage) -> bool:
    return message.kind is MessageKind.TEXT or (
        message.kind is MessageKind.STICKER and _asset_id(message) is not None
    )


def _asset_id(message: ImportedMessage) -> str | None:
    value = message.raw.get("media_asset_id", message.raw.get("asset_id"))
    return value if isinstance(value, str) and value else None


def _bubble_kind(message: ImportedMessage) -> Literal["text", "sticker", "emoji"]:
    if message.kind is MessageKind.TEXT:
        return "text"
    raw_kind = message.raw.get("asset_kind")
    if raw_kind == "emoji" or "<emoji" in message.content:
        return "emoji"
    return "sticker"


def _bubble_delay_ms(
    messages: list[ImportedMessage],
    index: int,
    previous_message: ImportedMessage | None,
) -> int:
    if index == 0 and previous_message is None:
        return 0
    previous = previous_message if index == 0 else messages[index - 1]
    if previous is None:
        return 0
    return max(
        0,
        int((messages[index].timestamp - previous.timestamp).total_seconds() * 1000),
    )
