def test_wechat_call_xml_becomes_human_readable_call_card() -> None:
    from moonlightbox.imports.wechat_rendering import normalize_wechat_display

    kind, content = normalize_wechat_display(
        "unknown",
        """
        <voipmsg type="VoIPBubbleMsg"><VoIPBubbleMsg>
        <msg><![CDATA[对方无应答]]></msg><room_type>1</room_type>
        </VoIPBubbleMsg></voipmsg>
        """,
    )

    assert kind == "call"
    assert content == "视频通话 · 对方无应答"


def test_quoted_voice_xml_is_replaced_without_losing_leading_text() -> None:
    import json

    from moonlightbox.imports.wechat_rendering import normalize_wechat_display

    kind, content = normalize_wechat_display(
        "unknown",
        '哈哈 不知道为何却觉得非常高兴\n> AMOS：<msg><voicemsg voicelength="2511"/></msg>',
    )

    assert kind == "quote"
    assert json.loads(content) == {
        "text": "哈哈 不知道为何却觉得非常高兴",
        "quoted_content": "[语音]",
    }


def test_plain_quote_hides_protocol_sender_and_returns_structured_quote() -> None:
    import json

    from moonlightbox.imports.wechat_rendering import normalize_wechat_display

    kind, content = normalize_wechat_display(
        "unknown",
        "正是这样\n> wxid_m6t7zrkzb5hf22：仍在加班中么",
    )

    assert kind == "quote"
    assert json.loads(content) == {
        "text": "正是这样",
        "quoted_content": "仍在加班中么",
    }


def test_recalled_message_xml_becomes_system_message() -> None:
    from moonlightbox.imports.wechat_rendering import normalize_wechat_display

    kind, content = normalize_wechat_display(
        "unknown",
        '<?xml version="1.0"?><sysmsg type="revokemsg"><revokemsg>'
        '<content>"Feather" 撤回了一条消息</content>'
        "</revokemsg></sysmsg>",
    )

    assert kind == "system"
    assert content == "撤回了一条消息"
