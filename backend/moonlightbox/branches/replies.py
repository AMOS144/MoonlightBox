import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from moonlightbox.training.bubble_protocol import (
    CompactProtocolNormalizationError,
    ProtocolBubble,
    normalize_mixed_compact_protocol,
    parse_bubble_protocol,
)


class ReplyStructureError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        attempted_invalid_sticker_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempted_invalid_sticker_ids = attempted_invalid_sticker_ids


class _BubblePayload(BaseModel):
    type: Literal["text", "sticker", "emoji"] = "text"
    content: str | None = Field(default=None, max_length=200)
    asset_id: str | None = None
    delay_ms: int = 0

    @model_validator(mode="after")
    def validate_payload(self) -> "_BubblePayload":
        if self.type == "text" and not (self.content or "").strip():
            raise ValueError("文字气泡不能为空")
        if self.type in {"sticker", "emoji"} and not self.asset_id:
            raise ValueError("表情气泡缺少资产")
        return self


class _ReplyPayload(BaseModel):
    bubbles: list[_BubblePayload] = Field(min_length=1, max_length=4)


@dataclass(frozen=True)
class GeneratedBubble:
    content: str | None
    delay_ms: int
    type: Literal[
        "text",
        "sticker",
        "emoji",
        "quote",
        "call",
        "reaction",
        "retract",
        "image",
        "audio",
        "video",
    ] = "text"
    asset_id: str | None = None


@dataclass(frozen=True)
class GeneratedReplyTurn:
    bubbles: tuple[GeneratedBubble, ...]
    raw_output: str
    degraded: bool = False
    normalization: str | None = None
    policy_appended: bool = False
    attempted_invalid_sticker_ids: tuple[str, ...] = ()


def parse_reply_turn(
    raw_output: str,
    *,
    allowed_sticker_ids: tuple[str, ...] | None = None,
    normalize_compact: bool = False,
) -> GeneratedReplyTurn:
    if normalize_compact:
        return _parse_compact_reply_turn(raw_output, allowed_sticker_ids)
    cleaned = _strip_code_fence(raw_output)
    if any(tag in cleaned for tag in ("<bubble", "<sticker", "<delay")):
        try:
            protocol_bubbles = parse_bubble_protocol(cleaned)
        except ValueError as error:
            raise ReplyStructureError("回复结构无效") from error
        return _build_generated_turn(
            tuple(
                GeneratedBubble(
                    content=bubble.value if bubble.kind == "text" else None,
                    delay_ms=bubble.delay_ms,
                    type=bubble.kind,
                    asset_id=bubble.value if bubble.kind != "text" else None,
                )
                for bubble in protocol_bubbles
            ),
            raw_output,
            allowed_sticker_ids=allowed_sticker_ids,
        )
    try:
        payload = _ReplyPayload.model_validate(_extract_json_payload(cleaned))
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise ReplyStructureError("回复结构无效") from error
    return _build_generated_turn(
        tuple(
            GeneratedBubble(
                content=(item.content or "").strip() or None,
                delay_ms=item.delay_ms,
                type=item.type,
                asset_id=item.asset_id,
            )
            for item in payload.bubbles
        ),
        raw_output,
        allowed_sticker_ids=allowed_sticker_ids,
    )


def parse_persona_text_turn(raw_output: str) -> GeneratedReplyTurn:
    """Turn natural persona text into application bubbles deterministically."""

    cleaned = _strip_code_fence(raw_output)
    if not cleaned.strip():
        raise ReplyStructureError("纯文本回复为空")
    if any(marker in cleaned for marker in ("<bubble", "<delay", "<sticker", "```")):
        raise ReplyStructureError("纯文本回复包含机器协议")
    if re.search(r"!\[[^\]]*\]\(https?://", cleaned):
        raise ReplyStructureError("纯文本回复包含 Markdown 媒体伪造")
    stripped = cleaned.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        raise ReplyStructureError("纯文本回复包含结构化对象")
    lines = tuple(line.strip() for line in cleaned.splitlines() if line.strip())
    # Real persona turns contain as many as ten consecutive short bubbles.
    # Preserve that rhythm; the four-bubble cap remains for legacy protocols.
    if not lines or len(lines) > 10:
        raise ReplyStructureError("纯文本回复气泡数量无效")
    return _build_generated_turn(
        tuple(
            GeneratedBubble(
                content=line,
                delay_ms=0 if index == 0 else 800,
            )
            for index, line in enumerate(lines)
        ),
        raw_output,
        preserve_text=True,
    )


def _build_generated_turn(
    source_bubbles: tuple[GeneratedBubble, ...],
    raw_output: str,
    *,
    allowed_sticker_ids: tuple[str, ...] | None = None,
    normalization: str | None = None,
    attempted_invalid_sticker_ids: tuple[str, ...] = (),
    preserve_text: bool = False,
) -> GeneratedReplyTurn:
    bubbles: list[GeneratedBubble] = []
    previous = ""
    for index, item in enumerate(source_bubbles):
        raw_content = item.content or ""
        content = (
            raw_content
            if preserve_text and raw_content.strip()
            else raw_content.strip() or None
        )
        if item.type == "text" and content is None:
            raise ReplyStructureError("文字气泡不能为空")
        if item.type in {"sticker", "emoji"} and not item.asset_id:
            raise ReplyStructureError("表情气泡缺少资产")
        if (
            item.type == "sticker"
            and allowed_sticker_ids is not None
            and item.asset_id not in allowed_sticker_ids
        ):
            raise ReplyStructureError("模型 sticker ID 不属于本轮 Top-K")
        if content and (content == previous or _is_repetitive(content)):
            raise ReplyStructureError("回复包含空内容或复读")
        bubbles.append(
            GeneratedBubble(
                content=content,
                delay_ms=(
                    0
                    if index == 0
                    else (
                        item.delay_ms
                        if preserve_text
                        else max(100, min(5000, item.delay_ms))
                    )
                ),
                type=item.type,
                asset_id=item.asset_id,
            )
        )
        previous = content or ""
    maximum_bubbles = 10 if preserve_text else 4
    if len(bubbles) > maximum_bubbles:
        raise ReplyStructureError("回复气泡数量超限")
    if sum(len(bubble.content or "") for bubble in bubbles) > 500:
        raise ReplyStructureError("回复总长度超限")
    return GeneratedReplyTurn(
        bubbles=tuple(bubbles),
        raw_output=raw_output,
        normalization=normalization,
        attempted_invalid_sticker_ids=attempted_invalid_sticker_ids,
    )


def _parse_compact_reply_turn(
    raw_output: str,
    allowed_sticker_ids: tuple[str, ...] | None,
) -> GeneratedReplyTurn:
    cleaned = raw_output.strip()
    normalization: str | None = None
    attempted_invalid: tuple[str, ...] = ()
    try:
        protocol_bubbles = parse_bubble_protocol(cleaned)
    except ValueError:
        try:
            normalized = normalize_mixed_compact_protocol(
                raw_output,
                allowed_sticker_ids=allowed_sticker_ids,
            )
        except CompactProtocolNormalizationError as error:
            raise ReplyStructureError(
                str(error),
                attempted_invalid_sticker_ids=(
                    error.attempted_invalid_sticker_ids
                ),
            ) from error
        except ValueError as error:
            raise ReplyStructureError(
                str(error),
                attempted_invalid_sticker_ids=_audit_invalid_sticker_attempts(
                    raw_output,
                    allowed_sticker_ids,
                ),
            ) from error
        protocol_bubbles = normalized.bubbles
        attempted_invalid = normalized.attempted_invalid_sticker_ids
        normalization = "mixed-compact-v1"
    else:
        protocol_bubbles, attempted_invalid = _drop_invalid_protocol_stickers(
            protocol_bubbles,
            allowed_sticker_ids,
        )
        if attempted_invalid:
            normalization = "mixed-compact-v1"
    return _build_generated_turn(
        tuple(
            GeneratedBubble(
                content=bubble.value if bubble.kind == "text" else None,
                delay_ms=bubble.delay_ms,
                type=bubble.kind,
                asset_id=bubble.value if bubble.kind != "text" else None,
            )
            for bubble in protocol_bubbles
        ),
        raw_output,
        allowed_sticker_ids=allowed_sticker_ids,
        normalization=normalization,
        attempted_invalid_sticker_ids=attempted_invalid,
        preserve_text=True,
    )


def _drop_invalid_protocol_stickers(
    bubbles: tuple[ProtocolBubble, ...],
    allowed_sticker_ids: tuple[str, ...] | None,
) -> tuple[tuple[ProtocolBubble, ...], tuple[str, ...]]:
    if allowed_sticker_ids is None:
        return bubbles, ()
    allowed = set(allowed_sticker_ids)
    invalid = tuple(
        bubble.value
        for bubble in bubbles
        if bubble.kind == "sticker" and bubble.value not in allowed
    )
    filtered = tuple(
        bubble
        for bubble in bubbles
        if bubble.kind != "sticker" or bubble.value in allowed
    )
    if not filtered and invalid:
        raise ReplyStructureError(
            "删除越界 sticker 后回复为空",
            attempted_invalid_sticker_ids=invalid,
        )
    return filtered, invalid


def _audit_invalid_sticker_attempts(
    raw_output: str,
    allowed_sticker_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """只从拒绝输出中提取审计 ID，不把未知语法转换为合法气泡。"""

    if allowed_sticker_ids is None:
        return ()
    allowed = set(allowed_sticker_ids)
    attempted = (
        *re.findall(r"<sticker(?: [^>]*)?>([^<>\r\n]+)</sticker>", raw_output),
        *re.findall(r"\[sticker资产:([^\]\r\n]+)\]", raw_output),
    )
    return tuple(asset_id for asset_id in attempted if asset_id not in allowed)


def fallback_reply_turn(content: str = "你说哪个呀？") -> GeneratedReplyTurn:
    return GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
        raw_output="",
        degraded=True,
    )


def validate_reply_sticker_ids(
    reply: GeneratedReplyTurn,
    allowed_sticker_ids: tuple[str, ...],
) -> None:
    """拒绝模型绕过本轮 context-only Top-K 的 sticker。"""

    allowed = set(allowed_sticker_ids)
    if any(
        bubble.type == "sticker" and bubble.asset_id not in allowed
        for bubble in reply.bubbles
    ):
        raise ReplyStructureError("模型 sticker ID 不属于本轮 Top-K")


def drop_invalid_sticker_bubbles(
    reply: GeneratedReplyTurn,
    allowed_sticker_ids: tuple[str, ...],
) -> GeneratedReplyTurn:
    """删除越界 sticker 气泡，保留其它合法内容并记录违规尝试。"""

    allowed = set(allowed_sticker_ids)
    invalid = tuple(
        bubble.asset_id or ""
        for bubble in reply.bubbles
        if bubble.type == "sticker" and bubble.asset_id not in allowed
    )
    if not invalid:
        return reply
    bubbles = tuple(
        bubble
        for bubble in reply.bubbles
        if bubble.type != "sticker" or bubble.asset_id in allowed
    )
    if not bubbles:
        raise ReplyStructureError("删除越界 sticker 后回复为空")
    return GeneratedReplyTurn(
        bubbles=bubbles,
        raw_output=reply.raw_output,
        degraded=reply.degraded,
        normalization=reply.normalization or "mixed-compact-v1",
        policy_appended=reply.policy_appended,
        attempted_invalid_sticker_ids=(
            *reply.attempted_invalid_sticker_ids,
            *invalid,
        ),
    )


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_payload(value: str) -> object:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "bubbles" in payload:
            return payload
    raise json.JSONDecodeError("未找到回复 JSON", value, 0)


def _is_repetitive(content: str) -> bool:
    segments = [
        segment.strip() for segment in re.split(r"[，。！？!?；;\n]+", content) if segment.strip()
    ]
    if len(segments) >= 3 and len(set(segments)) <= len(segments) / 2:
        return True
    compact = re.sub(r"\s+", "", content)
    for width in range(1, min(20, len(compact) // 3) + 1):
        chunk = compact[:width]
        repeated = chunk * (len(compact) // width)
        if len(repeated) >= len(compact) * 0.8 and compact.startswith(repeated):
            return True
    return False
