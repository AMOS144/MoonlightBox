import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import escape, unescape
from typing import Literal


@dataclass(frozen=True)
class ProtocolBubble:
    kind: Literal["text", "sticker", "emoji"]
    value: str
    delay_ms: int


@dataclass(frozen=True)
class NormalizedProtocolTurn:
    bubbles: tuple[ProtocolBubble, ...]
    attempted_invalid_sticker_ids: tuple[str, ...] = ()


class CompactProtocolNormalizationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        attempted_invalid_sticker_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempted_invalid_sticker_ids = attempted_invalid_sticker_ids


def private_chat_instruction() -> str:
    """训练和运行时共用的风格条件，不激活通用助手/角色扮演语域。"""

    return (
        "当前是两个人之间的私人微信聊天。直接延续对话，保持平时的短句、"
        "措辞和语气；不要解释、总结或提供客服式建议。"
    )


def persona_text_instruction() -> str:
    """Keep machine serialization outside the persona language model."""

    return (
        "只输出你此刻真正会发出的聊天文字本身；不要输出 JSON、XML、Markdown、"
        "标签、字段名、解释或内心分析。需要连续发多条消息时，每条独占一行。"
    )


def persona_style_transfer_instruction(
    style_examples: tuple[str, ...] = (),
) -> str:
    """Shared prompt for turning a trusted content draft into persona speech."""

    instruction = (
        "把内容草稿改写成这个人在私人微信里真实会发送的表达。"
        "不能新增、删除或改变事实、否定、数量、人物和态度；"
        "只改变措辞、语气、标点和消息分段。"
        + persona_text_instruction()
    )
    if not style_examples:
        return instruction
    examples = "\n\n".join(style_examples[:6])
    return (
        instruction
        + "\n以下是分支边界前这个真人实际发过的对话，只用于学习其称呼、口癖、"
        "句长和连续发多条消息的节奏；不得照抄其中的具体事实、人物、地点或事件：\n"
        + examples
    )


def prompt_is_persona_style_transfer(system_prompt: str) -> bool:
    """Identify an already-style-transfer request and prevent a nested rewrite."""

    return system_prompt.startswith("把内容草稿改写成这个人在私人微信里真实会发送的表达。")


def evidence_grounded_style_fallback_lines(
    draft: str,
    rewritten: str,
    style_examples: tuple[str, ...],
) -> tuple[str, ...]:
    """Add only example-backed rhythm when LoRA returns the draft verbatim."""

    if _style_comparison_text(draft) != _style_comparison_text(rewritten):
        return tuple(line.strip() for line in rewritten.splitlines() if line.strip())
    parts = [
        part.strip()
        for part in re.split(r"[，。；;]+", draft)
        if part.strip()
    ]
    if not parts:
        return (draft.strip(),) if draft.strip() else ()
    interjection = _example_backed_interjection(style_examples)
    if interjection is not None and interjection not in parts:
        parts.insert(0, interjection)
    return tuple(parts)


def _example_backed_interjection(style_examples: tuple[str, ...]) -> str | None:
    for example in style_examples:
        person = example.partition("本人：")[2]
        for bubble in person.split("/"):
            candidate = bubble.strip()
            if re.fullmatch(r"[啊哎唉呀哈嘿嗯哦啦诶]{1,6}", candidate):
                return candidate
    return None


def _style_comparison_text(value: str) -> str:
    return re.sub(r"\s|[，。！？!?；;、]", "", value)


def proactive_chat_instruction() -> str:
    """Shared condition for a genuine target-initiated private-chat turn."""

    return (
        "现在没有新的对方消息；只有确实想联系时才自然发消息，"
        "不要提及系统、任务或等待。"
    )


def prompt_uses_persona_text_protocol(system_prompt: str) -> bool:
    return system_prompt.rfind("聊天文字本身") > system_prompt.rfind("紧凑气泡协议")


_COMPACT_ELEMENT = re.compile(
    r"<bubble>(?P<bubble>[^<>]*)</bubble>"
    r'|<sticker(?: kind="(?P<kind>emoji|sticker)")?>(?P<sticker>[^<>]*)</sticker>'
    r"|<delay>(?P<delay>[^<>]*)</delay>"
    r"|^(?P<legacy_sticker>\[sticker资产:"
    r"(?P<legacy_sticker_id>[A-Za-z0-9_-]+)\])$",
    flags=re.MULTILINE,
)


def compact_protocol_instruction(
    allowed_sticker_ids: tuple[str, ...] = (),
) -> str:
    """返回训练、运行时和验收共用的紧凑协议说明。"""

    grammar = (
        "回复必须且只能使用高保真紧凑气泡协议，不要输出 JSON、Markdown、解释或游离文本。"
        'BNF：回合 ::= (文字气泡 | sticker气泡) 延迟 { (文字气泡 | sticker气泡) 延迟 }；'
        '文字气泡 ::= "<bubble>" 文本 "</bubble>"；'
        '延迟 ::= "<delay>" 非负整数 "</delay>"。'
        "每个内容标签后必须立即跟一个延迟标签；延迟使用符合真实聊天节奏的毫秒数。"
    )
    if not allowed_sticker_ids:
        return grammar + "本轮禁止输出 sticker。"
    return (
        grammar
        + "sticker气泡必须由 <sticker>、下列真实标识符之一、</sticker> 依次拼接；"
        + "本轮 sticker 仅允许以下资产 ID："
        + "、".join(allowed_sticker_ids)
        + "；禁止输出其他 sticker。"
    )


def compact_retry_instruction(
    allowed_sticker_ids: tuple[str, ...] = (),
) -> str:
    """返回不会作为对话内容注入的紧凑协议重试约束。"""

    return (
        "上一个输出格式无效，请完全重新生成且不要增加新事实。"
        "严禁使用 [bubble]、[delay] 或 [sticker资产:...] 方括号伪标签。"
        "越界 sticker ID 必须省略且不得替换；若省略后无内容，请重新规划文字回复。"
        + compact_protocol_instruction(allowed_sticker_ids)
    )


def allowed_sticker_ids_from_prompt(
    system_prompt: str,
) -> tuple[str, ...] | None:
    """从紧凑协议提示中提取本轮 context-only sticker 白名单。"""

    matches = tuple(
        re.finditer(
            r"本轮 sticker 仅允许(?:以下历史资产 ID|以下资产 ID)：([^；\n]+)",
            system_prompt,
        )
    )
    allowed_match = matches[-1] if matches else None
    ban_index = system_prompt.rfind("本轮禁止输出 sticker")
    if ban_index >= 0 and (
        allowed_match is None or ban_index > allowed_match.start()
    ):
        return ()
    if allowed_match is None:
        return None
    return tuple(
        item.strip()
        for item in re.split(r"[、,]", allowed_match.group(1))
        if item.strip()
    )


def prompt_uses_compact_protocol(system_prompt: str) -> bool:
    """按最后出现的协议指令判断当前回复格式。"""

    compact_index = system_prompt.rfind("紧凑气泡协议")
    legacy_index = max(
        system_prompt.rfind("气泡 JSON"),
        system_prompt.rfind("气泡JSON"),
    )
    return compact_index >= 0 and compact_index > legacy_index


def serialize_bubble_protocol(bubbles: tuple[ProtocolBubble, ...]) -> str:
    if not bubbles:
        raise ValueError("气泡协议不能为空")
    parts: list[str] = []
    for bubble in bubbles:
        if bubble.delay_ms < 0:
            raise ValueError("气泡延迟不能为负数")
        tag = "bubble" if bubble.kind == "text" else "sticker"
        attributes = ' kind="emoji"' if bubble.kind == "emoji" else ""
        if bubble.kind in {"sticker", "emoji"} and not bubble.value:
            raise ValueError("资产气泡必须包含资产 ID")
        parts.append(
            f"<{tag}{attributes}>{escape(bubble.value)}</{tag}>"
            f"<delay>{bubble.delay_ms}</delay>"
        )
    return "".join(parts)


def parse_bubble_protocol(content: str) -> tuple[ProtocolBubble, ...]:
    try:
        root = ET.fromstring(f"<turn>{content}</turn>")
    except ET.ParseError as error:
        raise ValueError("气泡协议格式无效") from error
    if _has_non_whitespace(root.text):
        raise ValueError("气泡协议不能包含游离文本")
    children = list(root)
    if not children or len(children) % 2:
        raise ValueError("气泡协议必须包含成对的内容与延迟标签")
    bubbles: list[ProtocolBubble] = []
    for index in range(0, len(children), 2):
        value_node = children[index]
        delay_node = children[index + 1]
        if value_node.tag not in {"bubble", "sticker"} or delay_node.tag != "delay":
            raise ValueError("气泡协议标签顺序无效")
        if delay_node.attrib:
            raise ValueError("延迟标签不允许属性")
        if value_node.tag == "bubble" and value_node.attrib:
            raise ValueError("文字气泡不允许属性")
        if value_node.tag == "sticker" and (
            set(value_node.attrib) - {"kind"}
            or value_node.attrib.get("kind", "sticker") not in {"emoji", "sticker"}
        ):
            raise ValueError("资产气泡 kind 属性无效")
        if list(value_node) or list(delay_node):
            raise ValueError("气泡协议不允许嵌套标签")
        if _has_non_whitespace(value_node.tail) or _has_non_whitespace(delay_node.tail):
            raise ValueError("气泡协议不能包含游离文本")
        value = value_node.text or ""
        kind: Literal["text", "sticker", "emoji"] = "text"
        if value_node.tag == "sticker":
            kind = (
                "emoji"
                if value_node.attrib.get("kind") == "emoji"
                else "sticker"
            )
        if kind in {"sticker", "emoji"} and not value:
            raise ValueError("资产气泡必须包含资产 ID")
        try:
            delay_ms = int(delay_node.text or "")
        except ValueError as error:
            raise ValueError("气泡延迟必须是整数") from error
        if delay_ms < 0:
            raise ValueError("气泡延迟不能为负数")
        bubbles.append(ProtocolBubble(kind=kind, value=value, delay_ms=delay_ms))
    return tuple(bubbles)


def normalize_plain_text_lines(content: str) -> tuple[str, ...]:
    """仅把无协议结构的自然多行文本确定性拆成文字气泡。"""

    if not content.strip():
        raise ValueError("纯文本回复不能为空")
    if any(character in content for character in "<>{}[]") or "```" in content:
        raise ValueError("纯文本回复包含协议或代码围栏结构")
    lines = tuple(line for line in content.splitlines() if line.strip())
    if not lines:
        raise ValueError("纯文本回复不能为空")
    return lines


def normalize_mixed_compact_protocol(
    content: str,
    *,
    allowed_sticker_ids: tuple[str, ...] | None,
) -> NormalizedProtocolTurn:
    """只修复可证明安全的紧凑协议外壳与缺失延迟。"""

    cleaned = _strip_single_markdown_fence(content)
    cleaned = re.sub(r"^\s*assistant:\s*", "", cleaned, count=1)
    if re.match(r"^\s*assistant:", cleaned):
        raise ValueError("紧凑协议包含重复 assistant 前缀")
    if not cleaned.strip():
        raise ValueError("紧凑协议回复不能为空")

    entries: list[ProtocolBubble | None] = []
    delay_assigned: list[bool] = []
    attempted_invalid: list[str] = []
    pending_delay_ms: int | None = None
    cursor = 0
    for match in _COMPACT_ELEMENT.finditer(cleaned):
        _append_safe_plain_text(
            cleaned[cursor : match.start()],
            entries,
            delay_assigned,
        )
        cursor = match.end()
        if match.group("bubble") is not None:
            value = unescape(match.group("bubble"))
            if not value.strip():
                raise ValueError("文字气泡不能为空")
            entries.append(ProtocolBubble("text", value, pending_delay_ms or 0))
            delay_assigned.append(pending_delay_ms is not None)
            pending_delay_ms = None
            continue
        if match.group("sticker") is not None:
            value = unescape(match.group("sticker"))
            if not value:
                raise ValueError("资产气泡必须包含资产 ID")
            kind: Literal["sticker", "emoji"] = (
                "emoji" if match.group("kind") == "emoji" else "sticker"
            )
            if (
                kind == "sticker"
                and allowed_sticker_ids is not None
                and value not in allowed_sticker_ids
            ):
                attempted_invalid.append(value)
                entries.append(None)
            else:
                entries.append(ProtocolBubble(kind, value, pending_delay_ms or 0))
            delay_assigned.append(pending_delay_ms is not None)
            pending_delay_ms = None
            continue
        if match.group("legacy_sticker") is not None:
            value = match.group("legacy_sticker_id") or ""
            if (
                allowed_sticker_ids is not None
                and value not in allowed_sticker_ids
            ):
                attempted_invalid.append(value)
                entries.append(None)
            else:
                entries.append(
                    ProtocolBubble("sticker", value, pending_delay_ms or 0)
                )
            delay_assigned.append(pending_delay_ms is not None)
            pending_delay_ms = None
            continue

        delay_text = match.group("delay") or ""
        try:
            delay_ms = int(delay_text)
        except ValueError as error:
            raise ValueError("气泡延迟必须是整数") from error
        if delay_ms < 0:
            raise ValueError("气泡延迟不能为负数")
        if not entries or delay_assigned[-1]:
            if pending_delay_ms is not None:
                raise ValueError("连续延迟标签没有对应的内容气泡")
            pending_delay_ms = delay_ms
            continue
        if entries[-1] is not None:
            bubble = entries[-1]
            entries[-1] = ProtocolBubble(
                kind=bubble.kind,
                value=bubble.value,
                delay_ms=delay_ms,
            )
        delay_assigned[-1] = True

    _append_safe_plain_text(cleaned[cursor:], entries, delay_assigned)
    if pending_delay_ms is not None:
        raise ValueError("延迟标签没有对应的内容气泡")
    bubbles = tuple(entry for entry in entries if entry is not None)
    if not bubbles:
        if attempted_invalid:
            raise CompactProtocolNormalizationError(
                "删除越界 sticker 后回复为空",
                attempted_invalid_sticker_ids=tuple(attempted_invalid),
            )
        raise ValueError("紧凑协议回复不能为空")
    return NormalizedProtocolTurn(
        bubbles=bubbles,
        attempted_invalid_sticker_ids=tuple(attempted_invalid),
    )


def _strip_single_markdown_fence(content: str) -> str:
    stripped = content.strip()
    if "```" not in stripped:
        return content.strip("\r\n")
    match = re.fullmatch(
        r"```[A-Za-z0-9_-]*[ \t]*\n(?P<body>.*)\n?```",
        stripped,
        flags=re.DOTALL,
    )
    if match is None or "```" in match.group("body"):
        raise ValueError("紧凑协议代码围栏无效")
    return match.group("body")


def _append_safe_plain_text(
    value: str,
    entries: list[ProtocolBubble | None],
    delay_assigned: list[bool],
) -> None:
    if not value.strip():
        return
    if any(character in value for character in "<>{}[]") or "```" in value:
        raise ValueError("紧凑协议包含半截、未知标签或结构残片")
    for line in value.splitlines():
        if not line.strip():
            continue
        entries.append(ProtocolBubble("text", line, 0))
        delay_assigned.append(False)


def _has_non_whitespace(value: str | None) -> bool:
    return bool(value and value.strip())
