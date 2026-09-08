"""聊天 Bundle 构建：按时间和长度切分会话，不做语义阈值判断。"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class WorldMessage:
    id: str
    import_id: str
    participant_id: str
    participant_name: str
    participant_role: str
    timestamp: datetime
    kind: str
    content: str


@dataclass(frozen=True)
class BundleMessage:
    message: WorldMessage
    is_carry_in: bool


@dataclass(frozen=True)
class ConversationBundleDocument:
    document_id: str
    source_name: str
    ordinal: int
    started_at: datetime
    ended_at: datetime
    content: str
    content_hash: str
    messages: tuple[BundleMessage, ...]

    @property
    def primary_message_count(self) -> int:
        return sum(not item.is_carry_in for item in self.messages)

    @property
    def carry_in_message_count(self) -> int:
        return sum(item.is_carry_in for item in self.messages)


def source_fingerprint(messages: list[WorldMessage]) -> str:
    """根据原始消息生成指纹，判断建图输入是否变化。"""
    serialized = json.dumps(
        [
            {
                "id": item.id,
                "import_id": item.import_id,
                "timestamp": item.timestamp.isoformat(),
                "participant": item.participant_name,
                "role": item.participant_role,
                "kind": item.kind,
                "content": item.content,
            }
            for item in messages
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def build_conversation_bundles(
    messages: list[WorldMessage],
    *,
    project_id: str,
    session_gap: timedelta = timedelta(hours=6),
    max_characters: int = 12_000,
    carry_in_turns: int = 3,
) -> list[ConversationBundleDocument]:
    """按简单时间/长度规则构建会话窗口，并携带前一窗口的少量对话。"""

    if session_gap <= timedelta(0):
        raise ValueError("session_gap 必须大于零")
    if max_characters <= 0:
        raise ValueError("max_characters 必须大于零")
    if carry_in_turns < 0:
        raise ValueError("carry_in_turns 不能为负数")
    ordered = sorted(messages, key=lambda item: (item.timestamp, item.id))
    if not ordered:
        return []

    primary_groups: list[list[WorldMessage]] = []
    current: list[WorldMessage] = []
    current_size = 0
    for message in ordered:
        rendered_size = len(render_world_message(message)) + (1 if current else 0)
        gap_boundary = bool(current and message.timestamp - current[-1].timestamp > session_gap)
        length_boundary = bool(current and current_size + rendered_size > max_characters)
        if gap_boundary or length_boundary:
            primary_groups.append(current)
            current = []
            current_size = 0
        current.append(message)
        current_size += len(render_world_message(message)) + (1 if current_size else 0)
    if current:
        primary_groups.append(current)

    documents: list[ConversationBundleDocument] = []
    previous_primary: list[WorldMessage] = []
    for ordinal, primary in enumerate(primary_groups):
        carry = _last_turns(previous_primary, carry_in_turns) if ordinal else []
        combined = [
            *(BundleMessage(message=item, is_carry_in=True) for item in carry),
            *(BundleMessage(message=item, is_carry_in=False) for item in primary),
        ]
        content = "\n".join(render_world_message(item.message) for item in combined)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        identity = "\x1f".join(
            [project_id, str(ordinal), *(item.message.id for item in combined), content_hash]
        )
        document_id = f"bundle_{hashlib.sha256(identity.encode()).hexdigest()[:32]}"
        documents.append(
            ConversationBundleDocument(
                document_id=document_id,
                source_name=f"{document_id}.txt",
                ordinal=ordinal,
                started_at=primary[0].timestamp,
                ended_at=primary[-1].timestamp,
                content=content,
                content_hash=content_hash,
                messages=tuple(combined),
            )
        )
        previous_primary = primary
    return documents


def render_world_message(message: WorldMessage) -> str:
    """把消息渲染为 LightRAG 能直接阅读的自然语言对话行。"""
    content = " ".join(message.content.split())
    return f"{message.timestamp:%Y-%m-%d %H:%M} {message.participant_name}：{content}"


def _last_turns(messages: list[WorldMessage], count: int) -> list[WorldMessage]:
    if count == 0 or not messages:
        return []
    turns: list[list[WorldMessage]] = []
    for message in messages:
        if turns and turns[-1][-1].participant_name == message.participant_name:
            turns[-1].append(message)
        else:
            turns.append([message])
    return [item for turn in turns[-count:] for item in turn]
