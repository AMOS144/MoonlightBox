from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MessageKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    STICKER = "sticker"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ImportedMessage:
    source_id: str
    timestamp: datetime
    sender: str
    kind: MessageKind
    content: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ImportError:
    line: int
    code: str
    message: str


@dataclass
class ImportResult:
    messages: list[ImportedMessage] = field(default_factory=list)
    errors: list[ImportError] = field(default_factory=list)
