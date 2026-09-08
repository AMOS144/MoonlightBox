import hashlib
import io
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from moonlightbox.imports.base import ChatImporter
from moonlightbox.imports.media import (
    MediaFile,
    match_message_media,
    match_participant_avatar,
    normalize_media_path,
)
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.imports.schemas import (
    ImportConfirm,
    ImportConfirmRead,
    ImportErrorRead,
    ImportPreviewRead,
    MessageSample,
)
from moonlightbox.imports.wechat_rendering import normalize_wechat_display
from moonlightbox.imports.wxecho_csv import WxechoCsvImporter
from moonlightbox.imports.wxecho_json import WxechoJsonImporter
from moonlightbox.imports.wxecho_txt import WxechoTxtImporter
from moonlightbox.media.service import MediaStore


class UnsupportedImportFormatError(ValueError):
    pass


class InvalidMediaArchiveError(ValueError):
    pass


class ImportService:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._importers: list[ChatImporter] = [
            WxechoCsvImporter(),
            WxechoJsonImporter(),
            WxechoTxtImporter(),
        ]

    def preview(
        self,
        project_id: str,
        filename: str,
        content: bytes,
        media_files: list[tuple[str, bytes]] | None = None,
    ) -> ImportPreviewRead:
        preview_id = str(uuid4())
        suffix = Path(filename).suffix.lower()
        preview_dir = self._data_dir / "projects" / project_id / "imports" / preview_id
        preview_dir.mkdir(parents=True, exist_ok=False)
        source_path = preview_dir / f"source{suffix}"
        source_path.write_bytes(content)
        stored_media: list[MediaFile] = []
        for relative_path, media_content in _expand_media_files(media_files or []):
            normalized = normalize_media_path(relative_path)
            stored_path = preview_dir / "media" / normalized
            stored_path.parent.mkdir(parents=True, exist_ok=True)
            stored_path.write_bytes(media_content)
            stored_media.append(MediaFile(relative_path=normalized, stored_path=stored_path))

        importer = self._select_importer(source_path)
        result = importer.parse(source_path)
        messages = result.messages
        timestamps = [message.timestamp for message in messages]

        return ImportPreviewRead(
            id=preview_id,
            message_count=len(messages),
            participants=list(dict.fromkeys(message.sender for message in messages)),
            time_range=(min(timestamps), max(timestamps)) if timestamps else None,
            kind_counts=dict(Counter(message.kind.value for message in messages)),
            sample_messages=[
                MessageSample(
                    timestamp=message.timestamp,
                    sender=message.sender,
                    kind=message.kind.value,
                    content=message.content,
                )
                for message in messages[:20]
            ],
            errors=[
                ImportErrorRead(line=error.line, code=error.code, message=error.message)
                for error in result.errors
            ],
            media_file_count=len(stored_media),
            linked_sticker_count=sum(
                message.content == "[表情]"
                and match_message_media(message, stored_media) is not None
                for message in messages
            ),
            unlinked_sticker_count=sum(
                message.content == "[表情]" and match_message_media(message, stored_media) is None
                for message in messages
            ),
            deduplicated_asset_count=len(
                {hashlib.sha256(item.stored_path.read_bytes()).hexdigest() for item in stored_media}
            ),
            failure_reasons={
                "missing_strong_media_reference": sum(
                    message.content == "[表情]"
                    and match_message_media(message, stored_media) is None
                    for message in messages
                )
            },
        )

    def confirm(
        self,
        session: Session,
        project_id: str,
        preview_id: str,
        request: ImportConfirm,
    ) -> ImportConfirmRead:
        existing = session.scalar(
            select(ImportSource).where(
                ImportSource.project_id == project_id,
                ImportSource.preview_id == preview_id,
            )
        )
        if existing is not None:
            return ImportConfirmRead(
                import_id=existing.id,
                message_count=existing.message_count,
                created=False,
            )

        preview_dir = self._data_dir / "projects" / project_id / "imports" / preview_id
        source_paths = list(preview_dir.glob("source.*"))
        if len(source_paths) != 1:
            raise FileNotFoundError(preview_id)

        source_path = source_paths[0]
        same_source = self._find_same_source(session, project_id, source_path)
        if same_source is not None:
            return ImportConfirmRead(
                import_id=same_source.id,
                message_count=same_source.message_count,
                created=False,
            )

        result = self._select_importer(source_path).parse(source_path)
        participant_names = set(message.sender for message in result.messages)
        selected = {request.self_participant, request.target_participant}
        if len(selected) != 2 or not selected.issubset(participant_names):
            raise ValueError("角色映射必须对应两个不同的聊天参与者")

        source = ImportSource(
            project_id=project_id,
            preview_id=preview_id,
            source_path=str(source_path),
            message_count=len(result.messages),
            confirmed_at=datetime.now(UTC),
        )
        session.add(source)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(ImportSource).where(
                    ImportSource.project_id == project_id,
                    ImportSource.preview_id == preview_id,
                )
            )
            if existing is None:
                raise
            return ImportConfirmRead(
                import_id=existing.id,
                message_count=existing.message_count,
                created=False,
            )

        participants: dict[str, Participant] = {}
        for name in participant_names:
            role = "self" if name == request.self_participant else "target"
            if name not in selected:
                role = "other"
            participant = Participant(project_id=project_id, name=name, role=role)
            session.add(participant)
            participants[name] = participant
        session.flush()

        media_files = (
            [
                MediaFile(
                    relative_path=path.relative_to(preview_dir / "media").as_posix(),
                    stored_path=path,
                )
                for path in (preview_dir / "media").rglob("*")
                if path.is_file()
            ]
            if (preview_dir / "media").is_dir()
            else []
        )
        media_store = MediaStore(self._data_dir)
        for name, participant in participants.items():
            avatar_file = match_participant_avatar(name, media_files)
            if avatar_file is None:
                continue
            avatar = media_store.save(
                session,
                project_id,
                kind="avatar",
                filename=avatar_file.relative_path,
                content=avatar_file.stored_path.read_bytes(),
                source_key=avatar_file.relative_path,
            )
            participant.avatar_asset_id = avatar.id
        for message in result.messages:
            matched = match_message_media(message, media_files)
            asset = None
            kind, rendered_content = normalize_wechat_display(
                message.kind.value,
                message.content,
            )
            if matched is not None:
                asset_kind = (
                    "sticker"
                    if message.content == "[表情]"
                    else message.kind.value
                    if message.kind.value in {"image", "audio", "video"}
                    else "image"
                )
                asset = media_store.save(
                    session,
                    project_id,
                    kind=asset_kind,
                    filename=matched.relative_path,
                    content=matched.stored_path.read_bytes(),
                    source_key=matched.relative_path,
                )
                if asset_kind == "sticker":
                    kind = "sticker"
            session.add(
                Message(
                    project_id=project_id,
                    import_id=source.id,
                    participant_id=participants[message.sender].id,
                    source_id=message.source_id,
                    timestamp=message.timestamp,
                    kind=kind,
                    content=rendered_content,
                    raw=message.raw,
                    media_asset_id=asset.id if asset else None,
                )
            )
        session.commit()
        return ImportConfirmRead(
            import_id=source.id,
            message_count=source.message_count,
            created=True,
        )

    @staticmethod
    def _find_same_source(
        session: Session,
        project_id: str,
        source_path: Path,
    ) -> ImportSource | None:
        source_hash = _file_hash(source_path)
        imports = session.scalars(select(ImportSource).where(ImportSource.project_id == project_id))
        for imported in imports:
            existing_path = Path(imported.source_path)
            if existing_path.is_file() and _file_hash(existing_path) == source_hash:
                return imported
        return None

    def _select_importer(self, path: Path) -> ChatImporter:
        importer, confidence = max(
            ((candidate, candidate.sniff(path)) for candidate in self._importers),
            key=lambda item: item[1],
        )
        if confidence < 0.5:
            raise UnsupportedImportFormatError(path.suffix)
        return importer


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expand_media_files(
    media_files: list[tuple[str, bytes]],
) -> list[tuple[str, bytes]]:
    expanded: list[tuple[str, bytes]] = []
    for relative_path, content in media_files:
        if Path(relative_path).suffix.lower() != ".zip":
            expanded.append((relative_path, content))
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) > 5000:
                    raise InvalidMediaArchiveError("媒体压缩包文件数量超过 5000")
                total_size = sum(item.file_size for item in members)
                if total_size > 1024 * 1024 * 1024:
                    raise InvalidMediaArchiveError("媒体压缩包解压后超过 1GB")
                seen: set[str] = set()
                for item in members:
                    if item.flag_bits & 0x1:
                        raise InvalidMediaArchiveError("媒体压缩包不能包含加密文件")
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise InvalidMediaArchiveError("媒体压缩包不能包含符号链接")
                    if item.filename.startswith(("/", "\\")):
                        raise InvalidMediaArchiveError("媒体压缩包包含不安全路径")
                    normalized = normalize_media_path(item.filename)
                    if normalized in seen:
                        raise InvalidMediaArchiveError("媒体压缩包包含重复路径")
                    seen.add(normalized)
                    expanded.append((normalized, archive.read(item)))
        except (zipfile.BadZipFile, RuntimeError) as error:
            raise InvalidMediaArchiveError("媒体压缩包无效") from error
    return expanded
