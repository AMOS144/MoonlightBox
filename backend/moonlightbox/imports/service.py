from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.base import ChatImporter
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.imports.schemas import (
    ImportConfirm,
    ImportConfirmRead,
    ImportErrorRead,
    ImportPreviewRead,
    MessageSample,
)
from moonlightbox.imports.wxecho_csv import WxechoCsvImporter
from moonlightbox.imports.wxecho_json import WxechoJsonImporter
from moonlightbox.imports.wxecho_txt import WxechoTxtImporter


class UnsupportedImportFormatError(ValueError):
    pass


class ImportService:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._importers: list[ChatImporter] = [
            WxechoCsvImporter(),
            WxechoJsonImporter(),
            WxechoTxtImporter(),
        ]

    def preview(self, project_id: str, filename: str, content: bytes) -> ImportPreviewRead:
        preview_id = str(uuid4())
        suffix = Path(filename).suffix.lower()
        preview_dir = self._data_dir / "projects" / project_id / "imports" / preview_id
        preview_dir.mkdir(parents=True, exist_ok=False)
        source_path = preview_dir / f"source{suffix}"
        source_path.write_bytes(content)

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
        session.flush()

        participants: dict[str, Participant] = {}
        for name in participant_names:
            role = "self" if name == request.self_participant else "target"
            if name not in selected:
                role = "other"
            participant = Participant(project_id=project_id, name=name, role=role)
            session.add(participant)
            participants[name] = participant
        session.flush()

        session.add_all(
            [
                Message(
                    project_id=project_id,
                    import_id=source.id,
                    participant_id=participants[message.sender].id,
                    source_id=message.source_id,
                    timestamp=message.timestamp,
                    kind=message.kind.value,
                    content=message.content,
                    raw=message.raw,
                )
                for message in result.messages
            ]
        )
        session.commit()
        return ImportConfirmRead(
            import_id=source.id,
            message_count=source.message_count,
            created=True,
        )

    def _select_importer(self, path: Path) -> ChatImporter:
        importer, confidence = max(
            ((candidate, candidate.sniff(path)) for candidate in self._importers),
            key=lambda item: item[1],
        )
        if confidence < 0.5:
            raise UnsupportedImportFormatError(path.suffix)
        return importer
