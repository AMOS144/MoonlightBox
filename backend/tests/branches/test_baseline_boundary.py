from datetime import UTC, datetime

from moonlightbox.imports.models import Message


def test_semantic_digest_ignores_media_but_detects_content_changes() -> None:
    from moonlightbox.branches.baseline_boundary import semantic_message_digest

    message = Message(
        id="message-1",
        project_id="project-1",
        import_id="import-1",
        participant_id="participant-1",
        source_id="10",
        timestamp=datetime.now(UTC),
        kind="text",
        content="原始内容",
        raw={},
        media_asset_id=None,
    )
    original = semantic_message_digest([(message, "target")])
    message.media_asset_id = "asset-1"
    media_enriched = semantic_message_digest([(message, "target")])
    message.content = "修改后的内容"
    changed = semantic_message_digest([(message, "target")])

    assert original == media_enriched
    assert changed != original
