from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.events.models import AnalysisRevision, EventNode
from moonlightbox.events.schemas import ReviewedEvent

EDITABLE_FIELDS = {
    "type",
    "start_message_id",
    "end_message_id",
    "before_state",
    "after_state",
    "emotion_labels",
    "topic",
    "conflict_level",
    "importance",
    "reason",
    "evidence_ids",
    "status",
}


class EventNotFoundError(LookupError):
    pass


class EventService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        project_id: str,
        reviewed: ReviewedEvent,
        analysis_version: str,
        prompt_version: str,
    ) -> EventNode:
        event = EventNode(project_id=project_id, **reviewed.model_dump())
        self._session.add(event)
        self._session.flush()
        self._add_revision(
            event,
            revision_number=1,
            action_reason="自动分析",
            analysis_version=analysis_version,
            prompt_version=prompt_version,
        )
        self._session.commit()
        self._session.refresh(event)
        return event

    def revise(
        self,
        event_id: str,
        changes: dict[str, object],
        action_reason: str,
        project_id: str | None = None,
    ) -> EventNode:
        event = self._get(event_id, project_id)
        invalid = set(changes) - EDITABLE_FIELDS
        if invalid:
            raise ValueError(f"不可修改字段：{', '.join(sorted(invalid))}")
        for field, value in changes.items():
            setattr(event, field, value)

        history = self.revisions(event_id)
        latest = history[-1]
        self._add_revision(
            event,
            revision_number=latest.revision_number + 1,
            action_reason=action_reason,
            analysis_version=latest.analysis_version,
            prompt_version=latest.prompt_version,
        )
        self._session.commit()
        self._session.refresh(event)
        return event

    def revisions(self, event_id: str) -> list[AnalysisRevision]:
        return list(
            self._session.scalars(
                select(AnalysisRevision)
                .where(AnalysisRevision.event_id == event_id)
                .order_by(AnalysisRevision.revision_number)
            )
        )

    def list(self, project_id: str) -> list[EventNode]:
        return list(
            self._session.scalars(
                select(EventNode)
                .where(EventNode.project_id == project_id)
                .order_by(EventNode.created_at)
            )
        )

    def _get(self, event_id: str, project_id: str | None = None) -> EventNode:
        event = self._session.get(EventNode, event_id)
        if event is None or (project_id is not None and event.project_id != project_id):
            raise EventNotFoundError(event_id)
        return event

    def _add_revision(
        self,
        event: EventNode,
        revision_number: int,
        action_reason: str,
        analysis_version: str,
        prompt_version: str,
    ) -> None:
        self._session.add(
            AnalysisRevision(
                event_id=event.id,
                revision_number=revision_number,
                snapshot=_snapshot(event),
                action_reason=action_reason,
                analysis_version=analysis_version,
                prompt_version=prompt_version,
            )
        )


def _snapshot(event: EventNode) -> dict[str, object]:
    return {field: getattr(event, field) for field in sorted(EDITABLE_FIELDS)}
