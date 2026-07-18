import json
from pathlib import Path


def test_json_and_txt_match_csv_messages(tmp_path: Path) -> None:
    from moonlightbox.imports.wxecho_csv import WxechoCsvImporter
    from moonlightbox.imports.wxecho_json import WxechoJsonImporter
    from moonlightbox.imports.wxecho_txt import WxechoTxtImporter

    json_path = tmp_path / "chat.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "time": "2026-01-01 20:00:00",
                    "sender": "甲",
                    "type": 1,
                    "type_name": "文本",
                    "content": "今晚有空吗",
                    "server_id": 1,
                },
                {
                    "time": "2026-01-01 20:00:12",
                    "sender": "乙",
                    "type": 1,
                    "type_name": "文本",
                    "content": "晚一点可以",
                    "server_id": 2,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    txt_path = tmp_path / "chat.txt"
    txt_path.write_text(
        "微信聊天记录: 测试\n"
        "总消息数: 2\n"
        "============================================================\n\n"
        "[2026-01-01 20:00:00] 甲: 今晚有空吗\n"
        "[2026-01-01 20:00:12] 乙: 晚一点可以\n",
        encoding="utf-8",
    )

    csv_messages = WxechoCsvImporter().parse(
        Path("tests/fixtures/wxecho_sample.csv")
    ).messages
    json_messages = WxechoJsonImporter().parse(json_path).messages
    txt_messages = WxechoTxtImporter().parse(txt_path).messages

    expected = [(message.sender, message.content) for message in csv_messages]
    assert [(message.sender, message.content) for message in json_messages] == expected
    assert [(message.sender, message.content) for message in txt_messages] == expected
