from datetime import datetime

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from moonlightbox.branches.baseline_models import (
    BranchBaselineEventSnapshot,
    BranchBaselineManifest,
)
from moonlightbox.branches.context import ContextMemory
from moonlightbox.branches.embeddings import TextEmbedder
from moonlightbox.branches.memory_index import (
    ChromaProjectMemoryIndex,
    MemoryDocument,
)
from moonlightbox.branches.models import Branch
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import Message, Participant


class ProjectMemoryRepository:
    def __init__(self, chroma_dir: str, embedder: TextEmbedder) -> None:
        self._chroma_dir = chroma_dir
        self._embedder = embedder
        self._fingerprints: dict[str, tuple[int, int]] = {}

    def rebuild_project(self, session: Session, project_id: str) -> int:
        documents = _exchange_documents(session, project_id)
        documents.extend(_event_documents(session, project_id))
        self._index(project_id).replace(documents)
        self._fingerprints[project_id] = _project_fingerprint(session, project_id)
        return len(documents)

    def retrieve(
        self,
        session: Session,
        branch: Branch,
        query_text: str,
    ) -> tuple[ContextMemory, ...]:
        fingerprint = _project_fingerprint(session, branch.project_id)
        if self._fingerprints.get(branch.project_id) != fingerprint:
            self.rebuild_project(session, branch.project_id)
        manifest = session.get(BranchBaselineManifest, branch.baseline_manifest_id)
        cutoff = manifest.boundary_timestamp if manifest is not None else branch.origin_time
        exchange_ids = {
            document.resource_id
            for document in _exchange_documents(
                session,
                branch.project_id,
                cutoff=cutoff,
                import_id=manifest.import_id if manifest is not None else None,
                boundary=(
                    (
                        manifest.boundary_timestamp,
                        manifest.boundary_source_id,
                        manifest.boundary_message_id,
                    )
                    if manifest is not None
                    else None
                ),
            )
        }
        event_ids = (
            {
                f"event:{event_id}"
                for event_id in session.scalars(
                    select(BranchBaselineEventSnapshot.source_event_id).where(
                        BranchBaselineEventSnapshot.manifest_id == manifest.id
                    )
                )
            }
            if manifest is not None
            else {
                document.resource_id
                for document in _event_documents(
                    session, branch.project_id, cutoff=branch.origin_time
                )
            }
        )
        index = self._index(branch.project_id)
        exchange_results = index.query(
            query_text,
            resource_type="exchange",
            allowed_ids=exchange_ids,
            cutoff=cutoff,
        )
        event_results = index.query(
            query_text,
            resource_type="event",
            allowed_ids=event_ids,
            cutoff=cutoff,
        )
        return tuple(
            ContextMemory(
                resource_id=resource_id,
                resource_type="exchange",
                content=content,
                authority="conversation_record",
                evidence_ids=(resource_id.removeprefix("exchange:"),),
            )
            for resource_id, content, _score in exchange_results
        ) + tuple(
            ContextMemory(
                resource_id=resource_id,
                resource_type="event",
                content=content,
                authority="verified_history",
                evidence_ids=(resource_id.removeprefix("event:"),),
            )
            for resource_id, content, _score in event_results
        )

    def _index(self, project_id: str) -> ChromaProjectMemoryIndex:
        return ChromaProjectMemoryIndex(
            chroma_dir=self._chroma_dir,
            project_id=project_id,
            embedder=self._embedder,
        )


def _project_fingerprint(session: Session, project_id: str) -> tuple[int, int]:
    message_count = (
        session.scalar(select(func.count(Message.id)).where(Message.project_id == project_id)) or 0
    )
    event_count = (
        session.scalar(
            select(func.count(EventNode.id)).where(
                EventNode.project_id == project_id,
                EventNode.status == "active",
            )
        )
        or 0
    )
    return int(message_count), int(event_count)


def _exchange_documents(
    session: Session,
    project_id: str,
    *,
    cutoff: datetime | None = None,
    import_id: str | None = None,
    boundary: tuple[datetime, str, str] | None = None,
) -> list[MemoryDocument]:
    conditions = [
        Message.project_id == project_id,
        Message.kind == "text",
        Participant.role.in_(("self", "target")),
    ]
    if cutoff is not None:
        conditions.append(Message.timestamp <= cutoff)
    if import_id is not None:
        conditions.append(Message.import_id == import_id)
    if boundary is not None:
        at, source_id, row_id = boundary
        conditions.append(
            or_(
                Message.timestamp < at,
                and_(Message.timestamp == at, Message.source_id < source_id),
                and_(
                    Message.timestamp == at,
                    Message.source_id == source_id,
                    Message.id < row_id,
                ),
            )
        )
    rows = list(
        session.execute(
            select(Message, Participant.role)
            .join(Participant, Participant.id == Message.participant_id)
            .where(*conditions)
            .order_by(Message.timestamp, Message.source_id)
        )
    )
    documents: list[MemoryDocument] = []
    for index, (message, role) in enumerate(rows):
        if role != "self":
            continue
        replies: list[Message] = []
        for reply, reply_role in rows[index + 1 :]:
            if reply_role == "self":
                break
            if reply_role == "target" and reply.content.strip():
                replies.append(reply)
            if len(replies) == 4:
                break
        if not replies:
            continue
        documents.append(
            MemoryDocument(
                resource_id=f"exchange:{message.id}",
                resource_type="exchange",
                content=(
                    f"用户：{message.content}\n"
                    f"目标：{' / '.join(reply.content for reply in replies)}"
                ),
                started_at=message.timestamp,
                ended_at=replies[-1].timestamp,
            )
        )
    return documents


def _event_documents(
    session: Session,
    project_id: str,
    *,
    cutoff: datetime | None = None,
) -> list[MemoryDocument]:
    events = list(
        session.scalars(
            select(EventNode).where(
                EventNode.project_id == project_id,
                EventNode.status == "active",
            )
        )
    )
    documents: list[MemoryDocument] = []
    for event in events:
        started_at = event.started_at or event.created_at
        ended_at = event.ended_at or started_at
        if cutoff is not None and ended_at > cutoff:
            continue
        documents.append(
            MemoryDocument(
                resource_id=f"event:{event.id}",
                resource_type="event",
                content=(f"标题：{event.title}\n摘要：{event.summary}\n主题：{event.topic}"),
                started_at=started_at,
                ended_at=ended_at,
            )
        )
    return documents
