from pathlib import Path


def test_parse_wxecho_csv() -> None:
    from moonlightbox.imports.types import MessageKind
    from moonlightbox.imports.wxecho_csv import WxechoCsvImporter

    result = WxechoCsvImporter().parse(Path("tests/fixtures/wxecho_sample.csv"))

    assert result.messages[0].sender == "甲"
    assert result.messages[0].kind is MessageKind.TEXT
    assert result.messages[0].content == "今晚有空吗"
    assert result.errors == []


def test_invalid_timestamp_is_reported_without_losing_valid_rows(tmp_path: Path) -> None:
    from moonlightbox.imports.wxecho_csv import WxechoCsvImporter

    source = tmp_path / "chat.csv"
    source.write_text(
        "时间,发送者,类型,内容\n"
        "错误时间,甲,文本,无法解析\n"
        '2026-01-01 20:00:00,乙,文本,"带,逗号的消息"\n',
        encoding="utf-8",
    )

    result = WxechoCsvImporter().parse(source)

    assert [message.content for message in result.messages] == ["带,逗号的消息"]
    assert result.errors[0].line == 2
    assert result.errors[0].code == "invalid_row"
