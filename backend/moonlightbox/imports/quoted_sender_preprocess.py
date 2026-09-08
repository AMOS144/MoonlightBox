"""Deterministic preprocessing for WeChat quoted-message display names."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import ParticipantAlias
from moonlightbox.world.bundles import WorldMessage

_QUOTE_LINE = re.compile(
    r"^\s*[>＞]\s*(?P<alias>[^<>\[\]【】\n:：]{1,80})\s*[:：]\s*(?P<body>.+?)\s*$",
    re.MULTILINE,
)
_TECHNICAL_SENDER_ID = re.compile(r"^wxid_[A-Za-z0-9_-]+$", re.IGNORECASE)
# 微信导出正文中可能出现短账号 ID；这些是技术标识，不是人物名称。
_TECHNICAL_ID = re.compile(
    r"(?<![A-Za-z0-9_])wxid_[A-Za-z0-9_-]+(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_TECHNICAL_PARTICIPANT_NAME = re.compile(
    r"^(?:wxid_[A-Za-z0-9_-]+|[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)$",
    re.IGNORECASE,
)
_TECHNICAL_QUOTE_ALIAS = re.compile(
    r"(?m)^(?P<prefix>\s*[>＞]\s*)"
    r"(?P<alias>wxid_[A-Za-z0-9_-]+|(?=[A-Za-z][A-Za-z0-9_-]{7,}\s*[:：])"
    r"[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)\s*[:：]\s*",
    re.IGNORECASE,
)
_LONG_MACHINE_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{128,}(?![A-Za-z0-9+/=_-])")
_XML_TAG = re.compile(r"<[^>]{1,1000}>")
_XML_ATTRIBUTE = re.compile(r"\b(?:aeskey|cdn\w+|md5|appid|bufid|clientmsgid)\s*=\s*[^\s>]+", re.IGNORECASE)


@dataclass(frozen=True)
class QuotedSenderObservation:
    alias: str
    status: str
    participant_id: str | None
    participant_name: str | None
    participant_role: str | None
    quoted_message_ids: tuple[str, ...]
    message_ids: tuple[str, ...]
    samples: tuple[str, ...]


def preprocess_quoted_senders(
    messages: list[WorldMessage],
) -> tuple[QuotedSenderObservation, ...]:
    """Resolve only explicit quote lines; ordinary prose is ignored.

    A quote is resolved when its body exactly matches an imported message and
    all matches belong to one participant. No fuzzy matching or lexical
    name-list is used.
    """

    by_content: dict[str, list[WorldMessage]] = {}
    for message in messages:
        by_content.setdefault(_normalize(message.content), []).append(message)

    grouped: dict[str, dict[str, object]] = {}
    for message in messages:
        for match in _QUOTE_LINE.finditer(message.content):
            alias = match.group("alias").strip()
            body = _normalize(match.group("body"))
            if not alias or not body or _TECHNICAL_SENDER_ID.fullmatch(alias):
                continue
            row = grouped.setdefault(
                alias,
                {"message_ids": [], "samples": [], "matches": []},
            )
            row["message_ids"].append(message.id)  # type: ignore[union-attr]
            if len(row["samples"]) < 5:  # type: ignore[arg-type]
                row["samples"].append(message.content[:240])  # type: ignore[union-attr]
            source_rows = by_content.get(body, [])
            source_participants = {item.participant_id for item in source_rows}
            # An individual quote contributes identity evidence only when its
            # exact body belongs to one participant. Repeated short messages
            # sent by both sides are deliberately ignored as ambiguous.
            if len(source_participants) == 1:
                row["matches"].extend(source_rows)  # type: ignore[union-attr]

    observations: list[QuotedSenderObservation] = []
    for alias, row in sorted(grouped.items()):
        matches = row["matches"]  # type: ignore[assignment]
        participant_ids = {item.participant_id for item in matches}
        if len(participant_ids) == 1 and matches:
            source = matches[0]
            status = "resolved"
            participant_id = source.participant_id
            participant_name = source.participant_name
            participant_role = source.participant_role
            quoted_ids = tuple(dict.fromkeys(item.id for item in matches))
        elif matches:
            status = "conflicting"
            participant_id = participant_name = participant_role = None
            quoted_ids = tuple(dict.fromkeys(item.id for item in matches))
        else:
            status = "unresolved"
            participant_id = participant_name = participant_role = None
            quoted_ids = ()
        observations.append(
            QuotedSenderObservation(
                alias=alias,
                status=status,
                participant_id=participant_id,
                participant_name=participant_name,
                participant_role=participant_role,
                quoted_message_ids=quoted_ids,
                message_ids=tuple(dict.fromkeys(row["message_ids"])),  # type: ignore[arg-type]
                samples=tuple(row["samples"]),  # type: ignore[arg-type]
            )
        )
    return tuple(observations)


def _normalize(value: str) -> str:
    return " ".join(value.split())


def persist_quoted_sender_aliases(
    session: Session,
    *,
    project_id: str,
    observations: tuple[QuotedSenderObservation, ...],
) -> None:
    """Idempotently archive quote-display names without changing messages."""

    now = datetime.now(UTC)
    existing = {
        item.alias: item
        for item in session.scalars(
            select(ParticipantAlias).where(ParticipantAlias.project_id == project_id)
        )
    }
    for observation in observations:
        record = existing.get(observation.alias)
        if record is None:
            record = ParticipantAlias(project_id=project_id, alias=observation.alias)
            session.add(record)
        record.participant_id = observation.participant_id
        record.origin = "quoted_sender"
        record.resolution_status = observation.status
        record.source_message_ids = list(observation.message_ids)
        record.quoted_message_ids = list(observation.quoted_message_ids)
        record.updated_at = now


def build_clean_world_messages(
    messages: list[WorldMessage],
    observations: tuple[QuotedSenderObservation, ...],
) -> list[WorldMessage]:
    """Create Bundle-only normalized messages while preserving database text."""

    resolved = {
        item.alias: item.participant_name
        for item in observations
        if item.status == "resolved" and item.participant_name
    }
    quoted_aliases = {item.alias for item in observations}
    cleaned: list[WorldMessage] = []
    for message in messages:
        content = _remove_technical_payload(message.content)
        for alias, label in resolved.items():
            content = re.sub(
                rf"(^\s*[>＞]\s*){re.escape(alias)}(\s*[:：])",
                rf"\g<1>{label}\g<2>",
                content,
                flags=re.MULTILINE,
            )
        # 无法精确回指原消息时，也不能把微信引用显示名泄漏给
        # LightRAG；只保留被引用正文，避免 AMOS/Feather 这类昵称
        # 被误当作人物节点。已解析的引用仍保留真实发送者名称。
        for alias in quoted_aliases - resolved.keys():
            content = re.sub(
                rf"(^\s*[>＞]\s*){re.escape(alias)}\s*[:：]\s*",
                r"\g<1>",
                content,
                flags=re.MULTILINE,
            )
        cleaned.append(
            replace(
                message,
                # 技术账号 ID 不能作为人物实体进入 Bundle；原始 sender
                # 仍保留在导入数据库中，LightRAG 只看到稳定的角色标签。
                participant_name=_clean_participant_name(
                    message.participant_name, message.participant_role
                ),
                content=content,
            )
        )
    return cleaned


def _remove_technical_payload(content: str) -> str:
    # 先处理引用署名中的裸 ID（例如 zrkzb5hf22），再处理正文中的
    # wxid_账号。正常英文昵称没有数字，不会被此规则删除。
    cleaned = _TECHNICAL_QUOTE_ALIAS.sub(r"\g<prefix>", content)
    cleaned = _TECHNICAL_ID.sub("", cleaned)
    cleaned = _LONG_MACHINE_TOKEN.sub("[技术载荷已省略]", cleaned)
    cleaned = _XML_ATTRIBUTE.sub("", cleaned)
    cleaned = _XML_TAG.sub("", cleaned)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def _clean_participant_name(name: str, role: str) -> str:
    if not _TECHNICAL_PARTICIPANT_NAME.fullmatch(name.strip()):
        return name
    return {
        "self": "用户",
        "target": "目标人物",
        "other": "其他参与者",
    }.get(role, "参与者")
