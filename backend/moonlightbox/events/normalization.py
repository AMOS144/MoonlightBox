import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from xml.etree import ElementTree

from moonlightbox.imports.types import ImportedMessage, MessageKind


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    """可安全用于语义分析的规范化消息。"""

    source_id: str
    timestamp: datetime
    sender: str
    kind: MessageKind
    content: str


_MEDIA_LABELS: dict[MessageKind, str] = {
    MessageKind.IMAGE: "[图片]",
    MessageKind.AUDIO: "[语音]",
    MessageKind.VIDEO: "[视频]",
    MessageKind.FILE: "[文件]",
}

_SYSTEM_PROMPT_PATTERNS = (
    re.compile(r"你撤回了一条消息"),
    re.compile(r'["“][^"”\r\n]+["”]撤回了一条消息'),
    re.compile(r"以下为新消息"),
)

_PROTOCOL_QUERY_PATTERN = re.compile(
    r"(?:room_type|red_dot)=[^&\s]+(?:&(?:room_type|red_dot)=[^&\s]+)*"
)


def normalize_message(message: ImportedMessage) -> NormalizedMessage | None:
    """规范化单条消息；没有可靠语义的消息返回 ``None``。"""

    media_label = _MEDIA_LABELS.get(message.kind)
    if media_label is not None:
        return _copy_with_content(message, media_label)

    if message.kind is MessageKind.SYSTEM:
        return None

    content = message.content.strip()
    if not content:
        return None
    if _is_known_emoji(content):
        return _copy_with_content(message, "[表情]")
    if _is_system_prompt(content) or _is_protocol_noise(content):
        return None

    return _copy_with_content(message, content)


def normalize_messages(messages: Iterable[ImportedMessage]) -> list[NormalizedMessage]:
    """按输入顺序规范化消息并过滤不可用内容。"""

    normalized: list[NormalizedMessage] = []
    for message in messages:
        result = normalize_message(message)
        if result is not None:
            normalized.append(result)
    return normalized


def _copy_with_content(message: ImportedMessage, content: str) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=message.source_id,
        timestamp=message.timestamp,
        sender=message.sender,
        kind=message.kind,
        content=content,
    )


def _is_protocol_noise(content: str) -> bool:
    lowered = content.casefold()
    if re.fullmatch(r"wxpay://\S+", lowered) is not None:
        return True
    if _PROTOCOL_QUERY_PATTERN.fullmatch(lowered) is not None:
        return True
    return _xml_root_tag(content) is not None


def _is_system_prompt(content: str) -> bool:
    return any(pattern.fullmatch(content) is not None for pattern in _SYSTEM_PROMPT_PATTERNS)


def _is_known_emoji(content: str) -> bool:
    return content in {"[表情]", "[动画表情]"} or _xml_root_tag(content) == "emoji"


def _xml_root_tag(content: str) -> str | None:
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return None
    return root.tag.rsplit("}", maxsplit=1)[-1].casefold()
