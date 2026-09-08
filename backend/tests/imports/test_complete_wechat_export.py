import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.imports.service import ImportService
from moonlightbox.imports.types import MessageKind
from moonlightbox.imports.wxecho_json import WxechoJsonImporter
from moonlightbox.media.models import MediaAsset
from moonlightbox.projects.models import Project
from sqlalchemy.orm import Session


def test_complete_export_json_preserves_normalized_kind_and_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "chat.json"
    source.write_text(
        json.dumps(
            [
                {
                    "time": "2026-04-18 20:00:00",
                    "sender": "洪欣羽",
                    "type_name": "通话",
                    "kind": "system",
                    "content": "语音通话 02:03",
                    "media_path": "media/call-cover.jpg",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    message = WxechoJsonImporter().parse(source).messages[0]

    assert message.kind is MessageKind.SYSTEM
    assert message.content == "语音通话 02:03"
    assert message.raw["media_path"] == "media/call-cover.jpg"


def test_complete_export_packages_latest_messages_media_and_avatars(
    tmp_path: Path,
) -> None:
    from moonlightbox.imports.complete_export import CompleteWechatExporter

    database = Database(f"sqlite:///{tmp_path / 'export.db'}")
    Project.metadata.create_all(database.engine)
    data_dir = tmp_path / "data"
    media_dir = data_dir / "projects/project-1/media"
    media_dir.mkdir(parents=True)
    sticker_path = media_dir / "sticker.gif"
    avatar_path = media_dir / "avatar.jpg"
    sticker_path.write_bytes(b"GIF89a-sticker")
    avatar_path.write_bytes(b"\xff\xd8\xff-avatar")
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="洪欣羽"))
        session.flush()
        source = ImportSource(
            id="import-1",
            project_id="project-1",
            preview_id="preview-1",
            source_path="/old/chat.csv",
            message_count=1,
            confirmed_at=now,
        )
        sticker = MediaAsset(
            id="sticker-1",
            project_id="project-1",
            kind="sticker",
            sha256="a" * 64,
            relative_path="projects/project-1/media/sticker.gif",
            mime_type="image/gif",
            source_key="wechat-sticker:original",
        )
        avatar = MediaAsset(
            id="avatar-1",
            project_id="project-1",
            kind="avatar",
            sha256="b" * 64,
            relative_path="projects/project-1/media/avatar.jpg",
            mime_type="image/jpeg",
            source_key="wechat-avatar:target",
        )
        session.add_all([source, sticker, avatar])
        session.flush()
        target = Participant(
            id="target-1",
            project_id="project-1",
            name="小肥入",
            role="target",
            avatar_asset_id=avatar.id,
        )
        session.add(target)
        session.flush()
        session.add(
            Message(
                id="message-1",
                project_id="project-1",
                import_id=source.id,
                participant_id=target.id,
                source_id="csv:2",
                timestamp=now,
                kind="sticker",
                content="[表情]",
                raw={},
                media_asset_id=sticker.id,
            )
        )
        session.commit()

        destination = tmp_path / "洪欣羽-完整导出"
        report = CompleteWechatExporter(data_dir).export(
            session,
            project_id="project-1",
            destination=destination,
        )

    chat_path = destination / "chat.json"
    payload = json.loads(chat_path.read_text(encoding="utf-8"))
    assert payload[0]["kind"] == "sticker"
    assert payload[0]["sender"] == "洪欣羽"
    assert payload[0]["media_path"].startswith("media/")
    archive_path = destination / "assets.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read(payload[0]["media_path"]) == b"GIF89a-sticker"
        assert archive.read("avatars/洪欣羽-avatar.jpg") == b"\xff\xd8\xff-avatar"
    assert not (destination / "media").exists()
    preview = ImportService(tmp_path / "preview-data").preview(
        "project-preview",
        "chat.json",
        chat_path.read_bytes(),
        [("assets.zip", archive_path.read_bytes())],
    )
    assert preview.media_file_count == 2
    assert preview.linked_sticker_count == 1
    assert report.message_count == 1
    assert report.media_count == 1
    assert report.avatar_count == 1
    database.close()
