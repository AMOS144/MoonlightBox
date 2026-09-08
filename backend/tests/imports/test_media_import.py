from pathlib import Path

import pytest
from moonlightbox.imports.types import ImportedMessage, MessageKind


def test_media_matcher_links_sticker_by_exported_filename() -> None:
    from moonlightbox.imports.media import MediaFile, match_message_media

    message = ImportedMessage(
        source_id="m1",
        timestamp=__import__("datetime").datetime(2026, 1, 1),
        sender="她",
        kind=MessageKind.UNKNOWN,
        content="[表情]",
        raw={"path": "Emoji/abc.gif"},
    )

    matched = match_message_media(
        message,
        [MediaFile(relative_path="Emoji/abc.gif", stored_path=Path("/tmp/abc.gif"))],
    )

    assert matched is not None


def test_media_path_rejects_parent_traversal() -> None:
    from moonlightbox.imports.media import normalize_media_path

    with pytest.raises(ValueError):
        normalize_media_path("../secret.gif")
