import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.imports.models import Message, Participant


class BaselineBoundaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedBaselineBoundary:
    project_id: str
    import_id: str
    event_id: str
    message_id: str
    timestamp: datetime
    source_id: str
    inclusive: bool = True


class BaselineBoundaryResolver:
    def __init__(self, session: Session) -> None:
        self._session = session

    def resolve(
        self,
        project_id: str,
        event_id: str,
    ) -> ResolvedBaselineBoundary:
        event = self._session.get(EventNode, event_id)
        if event is None or event.project_id != project_id:
            raise BaselineBoundaryError("节点不存在")
        revision = self._session.scalar(
            select(AnalysisRevision)
            .where(
                AnalysisRevision.event_id == event.id,
                AnalysisRevision.run_id.is_not(None),
            )
            .order_by(AnalysisRevision.revision_number.desc())
        )
        import_id: str | None = None
        if revision is not None and revision.run_id is not None:
            run = self._session.get(AnalysisRun, revision.run_id)
            if run is not None and run.project_id == project_id:
                candidate = self._session.scalar(
                    select(Message).where(
                        Message.project_id == project_id,
                        Message.import_id == run.import_id,
                        Message.source_id == event.end_message_id,
                    )
                )
                if candidate is not None:
                    import_id = run.import_id
        if import_id is None:
            candidates = list(
                self._session.scalars(
                    select(Message.import_id)
                    .where(
                        Message.project_id == project_id,
                        Message.source_id == event.end_message_id,
                    )
                    .distinct()
                )
            )
            if len(candidates) != 1:
                raise BaselineBoundaryError("节点无法唯一关联导入记录")
            import_id = candidates[0]
        boundary = self._session.scalar(
            select(Message).where(
                Message.project_id == project_id,
                Message.import_id == import_id,
                Message.source_id == event.end_message_id,
            )
        )
        if boundary is None:
            raise BaselineBoundaryError("节点终点消息不存在")
        return ResolvedBaselineBoundary(
            project_id=project_id,
            import_id=import_id,
            event_id=event.id,
            message_id=boundary.id,
            timestamp=boundary.timestamp,
            source_id=boundary.source_id,
            inclusive=True,
        )

    def messages_before(
        self,
        boundary: ResolvedBaselineBoundary,
    ) -> list[tuple[Message, str]]:
        boundary_condition = or_(
            Message.timestamp < boundary.timestamp,
            and_(
                Message.timestamp == boundary.timestamp,
                Message.source_id < boundary.source_id,
            ),
            and_(
                Message.timestamp == boundary.timestamp,
                Message.source_id == boundary.source_id,
                Message.id < boundary.message_id,
            ),
        )
        if boundary.inclusive:
            boundary_condition = or_(
                boundary_condition,
                Message.id == boundary.message_id,
            )
        return cast(
            list[tuple[Message, str]],
            self._session.execute(
                select(Message, Participant.role)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.project_id == boundary.project_id,
                    Message.import_id == boundary.import_id,
                    Participant.role.in_(("self", "target")),
                    boundary_condition,
                )
                .order_by(Message.timestamp, Message.source_id, Message.id)
            ).all(),
        )


def semantic_message_digest(
    rows: Iterable[tuple[Message, str]],
) -> str:
    digest = hashlib.sha256()
    for message, role in rows:
        payload = {
            "id": message.id,
            "source_id": message.source_id,
            "timestamp": message.timestamp.isoformat(),
            "role": role,
            "kind": message.kind,
            "content": message.content,
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
