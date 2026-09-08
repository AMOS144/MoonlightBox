from datetime import datetime

import pytest
from moonlightbox.imports.types import ImportedMessage, MessageKind


def make_message(
    content: str,
    *,
    kind: MessageKind = MessageKind.TEXT,
    source_id: str = "message-001",
) -> ImportedMessage:
    return ImportedMessage(
        source_id=source_id,
        timestamp=datetime(2026, 7, 18, 12, 0),
        sender="甲",
        kind=kind,
        content=content,
        raw={},
    )


def test_normalize_message_preserves_identity_and_trims_text() -> None:
    from moonlightbox.events.normalization import normalize_message

    source = make_message("  普通聊天内容\n")

    normalized = normalize_message(source)

    assert normalized is not None
    assert normalized.source_id == source.source_id
    assert normalized.timestamp == source.timestamp
    assert normalized.sender == source.sender
    assert normalized.kind == source.kind
    assert normalized.content == "普通聊天内容"


@pytest.mark.parametrize(
    "content",
    [
        "",
        " \n\t ",
        "<voipmsg><room_type>1</room_type></voipmsg>",
        '<msg><appmsg appid="wx123"><title>位置共享</title></appmsg></msg>',
        "<appmsg><title>无法识别的协议</title></appmsg>",
        "<wrapper><secret>不应保留内部文本</secret></wrapper>",
        "wxpay://c2cbizmessagehandler/hongbao/receivehongbao",
        "room_type=1&red_dot=0",
    ],
)
def test_normalize_message_discards_empty_and_protocol_noise(content: str) -> None:
    from moonlightbox.events.normalization import normalize_message

    assert normalize_message(make_message(content, kind=MessageKind.UNKNOWN)) is None


@pytest.mark.parametrize(
    "content",
    [
        "你撤回了一条消息",
        '"甲"撤回了一条消息',
        "以下为新消息",
    ],
)
def test_normalize_message_discards_system_prompts(content: str) -> None:
    from moonlightbox.events.normalization import normalize_message

    assert normalize_message(make_message(content, kind=MessageKind.TEXT)) is None


@pytest.mark.parametrize(
    "content",
    [
        "为什么显示‘以下为新消息’？",
        "你说我撤回了一条消息？",
        "<这不是协议>",
        "普通文字 <b>没有闭合",
        "文档里写着 room_type=1，但这是正常讨论",
    ],
)
def test_normalize_message_preserves_text_that_only_mentions_protocol_or_prompt(
    content: str,
) -> None:
    from moonlightbox.events.normalization import normalize_message

    normalized = normalize_message(make_message(content))

    assert normalized is not None
    assert normalized.content == content


@pytest.mark.parametrize("content", ["[表情]", "[动画表情]", "<emoji md5='abc' />"])
def test_normalize_message_replaces_known_emoji_with_stable_label(content: str) -> None:
    from moonlightbox.events.normalization import normalize_message

    normalized = normalize_message(make_message(content, kind=MessageKind.UNKNOWN))

    assert normalized is not None
    assert normalized.kind is MessageKind.UNKNOWN
    assert normalized.content == "[表情]"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (MessageKind.IMAGE, "[图片]"),
        (MessageKind.AUDIO, "[语音]"),
        (MessageKind.VIDEO, "[视频]"),
        (MessageKind.FILE, "[文件]"),
    ],
)
def test_normalize_message_replaces_known_media_with_stable_label(
    kind: MessageKind,
    expected: str,
) -> None:
    from moonlightbox.events.normalization import normalize_message

    normalized = normalize_message(make_message("<msg>协议原文</msg>", kind=kind))

    assert normalized is not None
    assert normalized.kind == kind
    assert normalized.content == expected


def test_normalize_messages_filters_noise_without_reordering() -> None:
    from moonlightbox.events.normalization import normalize_messages

    normalized = normalize_messages(
        [
            make_message("第一条", source_id="m-2"),
            make_message("<msg>协议</msg>", kind=MessageKind.UNKNOWN, source_id="noise"),
            make_message("第二条", source_id="m-1"),
        ]
    )

    assert [message.source_id for message in normalized] == ["m-2", "m-1"]
