from datetime import datetime, timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind


def message(content: str, second: int) -> ImportedMessage:
    return ImportedMessage(
        source_id=str(second),
        timestamp=datetime(2026, 1, 1, 20, 0, 0) + timedelta(seconds=second),
        sender="甲",
        kind=MessageKind.TEXT,
        content=content,
        raw={},
    )


def test_consecutive_short_replies_preserve_style() -> None:
    from moonlightbox.imports.cleaning import clean_messages

    cleaned = clean_messages(
        [
            message("猜字顶", second=0),
            message("的", second=6),
        ]
    )

    assert cleaned.training_units[0].content == "猜字顶\n的"
    assert cleaned.training_units[0].source_ids == ["0", "6"]


def test_sensitive_values_are_redacted_without_changing_source() -> None:
    from moonlightbox.imports.redaction import redact_text

    source = "联系我 13812345678 或 test@example.com"

    redacted = redact_text(source)

    assert redacted == "联系我 [手机号] 或 [邮箱]"
    assert source == "联系我 13812345678 或 test@example.com"
