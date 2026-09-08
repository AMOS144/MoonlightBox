"""Cutoff-safe authentic dialogue examples for cognition and expression."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy.orm import Session

from moonlightbox.branches.history import BranchHistoryService
from moonlightbox.branches.models import Branch
from moonlightbox.imports.models import Message
from moonlightbox.imports.wechat_rendering import normalize_wechat_display

_TOKEN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


@dataclass(frozen=True)
class AuthenticDialogueExample:
    """One real other-person prompt followed by the subject's real bubbles."""

    other_messages: tuple[str, ...]
    person_messages: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "other_messages": list(self.other_messages),
            "person_messages": list(self.person_messages),
        }

    def as_prompt_text(self) -> str:
        other = " / ".join(self.other_messages)
        person = " / ".join(self.person_messages)
        return f"对方：{other}\n本人：{person}"


def select_authentic_dialogue_examples(
    session: Session,
    branch: Branch,
    query: str,
    *,
    limit: int = 6,
) -> tuple[AuthenticDialogueExample, ...]:
    """Select real baseline exchanges without crossing the branch boundary."""

    rows = BranchHistoryService(session).authentic_example_rows(
        branch,
        message_limit=500,
    )
    return rank_authentic_dialogue_examples(
        extract_authentic_dialogue_examples(rows),
        query,
        limit=limit,
    )


def extract_authentic_dialogue_examples(
    rows: list[tuple[Message, str]],
) -> tuple[AuthenticDialogueExample, ...]:
    """Turn consecutive baseline bubbles into self->target exchange pairs."""

    groups: list[tuple[str, tuple[str, ...]]] = []
    current_role: str | None = None
    current_messages: list[str] = []
    for message, role in rows:
        if role not in {"self", "target"}:
            continue
        if role != current_role:
            if current_role is not None and current_messages:
                groups.append((current_role, tuple(current_messages)))
            current_role = role
            current_messages = []
        content = _display_content(message)
        if content is None:
            continue
        current_messages.append(content)
    if current_role is not None and current_messages:
        groups.append((current_role, tuple(current_messages)))

    examples: list[AuthenticDialogueExample] = []
    for index, (role, messages) in enumerate(groups):
        if role != "target" or index == 0:
            continue
        previous_role, previous_messages = groups[index - 1]
        if previous_role != "self":
            continue
        examples.append(
            AuthenticDialogueExample(
                other_messages=previous_messages[-4:],
                person_messages=messages[:5],
            )
        )
    return tuple(examples)


def rank_authentic_dialogue_examples(
    examples: tuple[AuthenticDialogueExample, ...],
    query: str,
    *,
    limit: int = 6,
) -> tuple[AuthenticDialogueExample, ...]:
    """Blend lexical relevance with recent representative behavior."""

    if limit <= 0 or not examples:
        return ()
    normalized_query = _normalized(query)
    query_grams = _bigrams(normalized_query)

    def score(item: tuple[int, AuthenticDialogueExample]) -> tuple[float, int]:
        index, example = item
        other = _normalized("".join(example.other_messages))
        if not normalized_query or not other:
            relevance = 0.0
        else:
            grams = _bigrams(other)
            overlap = len(query_grams & grams) / max(1, len(query_grams | grams))
            containment = 1.0 if other in normalized_query or normalized_query in other else 0.0
            shared_chars = len(set(normalized_query) & set(other)) / max(
                1, len(set(normalized_query) | set(other))
            )
            phrase_overlap = len(_ngrams(normalized_query, 3) & _ngrams(other, 3))
            relevance = (
                containment * 2.0
                + overlap * 1.5
                + shared_chars * 0.35
                + phrase_overlap * 0.8
            )
        return relevance, index

    ranked = sorted(enumerate(examples), key=score, reverse=True)
    selected_indexes = [
        index
        for item in ranked
        if score(item)[0] >= 0.12
        for index in [item[0]]
    ]
    return tuple(examples[index] for index in selected_indexes[:limit])


def _display_content(message: Message) -> str | None:
    kind, content = normalize_wechat_display(message.kind, message.content)
    if kind == "quote":
        try:
            quote = json.loads(content)
        except json.JSONDecodeError:
            quote = None
        if isinstance(quote, dict):
            text = quote.get("text")
            quoted = quote.get("quoted_content")
            if isinstance(text, str) and text.strip():
                content = text
                if isinstance(quoted, str) and quoted.strip():
                    content += f"（回复：{quoted.strip()}）"
    value = " ".join(content.split()).strip()
    if (
        not value
        or len(value) > 240
        or value in {"[图片]", "[表情]", "[语音]", "[视频]", "[文件]"}
        or not _is_clean_chat_text(value)
    ):
        return None
    return value


def _is_clean_chat_text(value: str) -> bool:
    """拒绝解密乱码、私用区字符和无意义单字样本。"""

    if any(unicodedata.category(character) in {"Cc", "Cs", "Co", "Cn"} for character in value):
        return False
    normalized = _normalized(value)
    if len(normalized) >= 2:
        return True
    return normalized in {"嗯", "哦", "啊", "好", "行", "对", "不", "哈"}


def _normalized(value: str) -> str:
    return "".join(_TOKEN.findall(value)).lower()


def _bigrams(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _ngrams(value: str, size: int) -> set[str]:
    if len(value) < size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}
