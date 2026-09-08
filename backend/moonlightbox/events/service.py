from __future__ import annotations

import builtins
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.events.models import (
    AnalysisRevision,
    AnalysisRun,
    EventCandidate,
    EventNode,
)
from moonlightbox.events.normalization import normalize_message
from moonlightbox.events.ranking import RankedCandidate
from moonlightbox.events.runs import AnalysisRunService
from moonlightbox.events.schemas import (
    EventEvidenceSummary,
    EventNodeRead,
    EventScoreComponents,
    ReviewedEvent,
    V2EventScoreComponents,
)
from moonlightbox.events.v3_ranking import V3RankedCandidate
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.jobs.service import JobLeaseLostError, JobService

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
V3_SNAPSHOT_FIELDS = {
    "lane",
    "event_status",
    "title",
    "summary",
    "started_at",
    "ended_at",
    "source_lanes",
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

    def publish_v2(
        self,
        *,
        project_id: str,
        run_id: str,
        candidates: Sequence[RankedCandidate],
        lease_token: str | None = None,
        lease_owner: str | None = None,
        job_id: str | None = None,
        job_worker_token: str | None = None,
        now: datetime | None = None,
    ) -> list[EventNode]:
        """在同一事务中替换自动节点、发布 V2 节点并完成分析运行。"""

        run = self._session.get(AnalysisRun, run_id)
        if run is None or run.project_id != project_id:
            raise ValueError("分析运行不属于指定项目")
        existing = self._published_events(run_id)
        if run.status == "succeeded":
            return existing
        if run.status != "running":
            raise ValueError(f"{run.status} 状态不能发布 V2 节点")
        if run.completed_windows != run.total_windows:
            raise ValueError("仍有窗口未完成，不能发布 V2 节点")
        AnalysisRunService(self._session).assert_lease_if_required(
            run,
            lease_token=lease_token,
            lease_owner=lease_owner,
            now=now or datetime.now(UTC),
        )
        if (job_id is None) != (job_worker_token is None):
            raise ValueError("job_id 与 job_worker_token 必须同时提供")

        try:
            self._supersede_automatic_nodes(
                project_id,
                run,
                replacement_version="hybrid-v2",
                source_versions={"heuristic-v1", "hybrid-v2"},
            )
            created = [
                self._create_published_event(project_id, candidate, run) for candidate in candidates
            ]
            AnalysisRunService(self._session).succeed(
                run.id,
                lease_token=lease_token,
                lease_owner=lease_owner,
                now=now,
            )
            if job_id is not None and job_worker_token is not None:
                completed = JobService(self._session).succeed_in_transaction(
                    job_id,
                    token=job_worker_token,
                    checkpoint={
                        "stage": "completed",
                        "run_id": run.id,
                        "event_count": len(created),
                        "progress": 1.0,
                    },
                )
                if not completed:
                    raise JobLeaseLostError("job_lease_lost")
            self._session.commit()
            return created
        except Exception:
            self._session.rollback()
            raise

    def publish_v3(
        self,
        *,
        project_id: str,
        run_id: str,
        candidates: Sequence[V3RankedCandidate],
        lease_token: str | None = None,
        lease_owner: str | None = None,
        job_id: str | None = None,
        job_worker_token: str | None = None,
        now: datetime | None = None,
    ) -> list[EventNode]:
        """在同一事务中发布 V3 节点并完成运行与 Job。"""

        run = self._session.get(AnalysisRun, run_id)
        if run is None or run.project_id != project_id:
            raise ValueError("分析运行不属于指定项目")
        existing = self._published_events(run_id)
        if run.status == "succeeded":
            return existing
        if run.analysis_version != "hybrid-v3":
            raise ValueError("只有 hybrid-v3 运行可以发布 V3 节点")
        if run.status != "running":
            raise ValueError(f"{run.status} 状态不能发布 V3 节点")
        if run.completed_windows != run.total_windows:
            raise ValueError("仍有槽位未完成，不能发布 V3 节点")
        checked_at = now or datetime.now(UTC)
        AnalysisRunService(self._session).assert_lease_if_required(
            run,
            lease_token=lease_token,
            lease_owner=lease_owner,
            now=checked_at,
        )
        if (job_id is None) != (job_worker_token is None):
            raise ValueError("job_id 与 job_worker_token 必须同时提供")

        try:
            self._supersede_automatic_nodes(
                project_id,
                run,
                replacement_version="hybrid-v3",
                source_versions={"heuristic-v1", "hybrid-v2", "hybrid-v3"},
            )
            created = [
                self._create_published_v3_event(project_id, candidate, run)
                for candidate in candidates
            ]
            AnalysisRunService(self._session).succeed(
                run.id,
                lease_token=lease_token,
                lease_owner=lease_owner,
                now=checked_at,
            )
            if job_id is not None and job_worker_token is not None:
                completed = JobService(self._session).succeed_in_transaction(
                    job_id,
                    token=job_worker_token,
                    checkpoint={
                        "stage": "completed",
                        "run_id": run.id,
                        "event_count": len(created),
                        "progress": 1.0,
                    },
                )
                if not completed:
                    raise JobLeaseLostError("job_lease_lost")
            self._session.commit()
            return created
        except Exception:
            self._session.rollback()
            raise

    def _published_events(self, run_id: str) -> list[EventNode]:
        return list(
            self._session.scalars(
                select(EventNode)
                .join(
                    AnalysisRevision,
                    AnalysisRevision.event_id == EventNode.id,
                )
                .where(
                    AnalysisRevision.run_id == run_id,
                    AnalysisRevision.action_reason.in_({"V2 自动发布", "V3 自动发布"}),
                )
                .order_by(EventNode.created_at, EventNode.id)
            )
        )

    def _supersede_automatic_nodes(
        self,
        project_id: str,
        run: AnalysisRun,
        *,
        replacement_version: str,
        source_versions: set[str],
    ) -> None:
        revision_counts = (
            select(
                AnalysisRevision.event_id,
                func.count(AnalysisRevision.id).label("revision_count"),
            )
            .group_by(AnalysisRevision.event_id)
            .subquery()
        )
        rows = self._session.execute(
            select(EventNode, AnalysisRevision)
            .join(
                AnalysisRevision,
                AnalysisRevision.event_id == EventNode.id,
            )
            .join(
                revision_counts,
                revision_counts.c.event_id == EventNode.id,
            )
            .where(
                EventNode.project_id == project_id,
                EventNode.status == "active",
                revision_counts.c.revision_count == 1,
                AnalysisRevision.revision_number == 1,
                AnalysisRevision.analysis_version.in_(source_versions),
            )
            .order_by(EventNode.id)
        )
        for event, _ in rows:
            event.status = "superseded"
            self._add_revision(
                event,
                revision_number=2,
                action_reason=(
                    "V3 自动替换" if replacement_version == "hybrid-v3" else "V2 自动替换"
                ),
                analysis_version=replacement_version,
                prompt_version=run.prompt_version,
                model=run.model,
                run_id=run.id,
            )

    def _create_published_event(
        self,
        project_id: str,
        candidate: RankedCandidate,
        run: AnalysisRun,
    ) -> EventNode:
        source = candidate.candidate
        event = EventNode(
            project_id=project_id,
            type=source.type,
            start_message_id=source.start_message_id,
            end_message_id=source.end_message_id,
            before_state=source.before_state,
            after_state=source.after_state,
            emotion_labels=source.emotion_labels,
            topic=source.topic,
            conflict_level=source.conflict_level,
            importance=candidate.scores["total"],
            reason=source.reason,
            evidence_ids=source.evidence_ids,
            status="active",
        )
        self._session.add(event)
        self._session.flush()
        self._add_revision(
            event,
            revision_number=1,
            action_reason="V2 自动发布",
            analysis_version="hybrid-v2",
            prompt_version=run.prompt_version,
            model=run.model,
            run_id=run.id,
            candidate_id=candidate.source_candidate_id,
        )
        return event

    def _create_published_v3_event(
        self,
        project_id: str,
        candidate: V3RankedCandidate,
        run: AnalysisRun,
    ) -> EventNode:
        source = candidate.candidate
        event = EventNode(
            project_id=project_id,
            lane=source.lane,
            event_status=source.event_status,
            type=source.type,
            title=source.title,
            summary=source.summary,
            display_summary=candidate.display_summary,
            summary_status=(
                "ready" if candidate.display_summary is not None else "pending"
            ),
            summary_model=candidate.summary_model,
            started_at=candidate.started_at,
            ended_at=candidate.ended_at,
            source_lanes=list(candidate.source_lanes),
            start_message_id=source.start_message_id,
            end_message_id=source.end_message_id,
            before_state=source.before_state,
            after_state=source.after_state,
            emotion_labels=source.emotion_labels,
            topic=source.topic,
            conflict_level=source.conflict_level,
            importance=(
                candidate.global_importance
                if candidate.global_importance is not None
                else candidate.scores.total
            ),
            reason=candidate.global_reason or source.reason,
            evidence_ids=source.evidence_ids,
            status="active",
        )
        self._session.add(event)
        self._session.flush()
        source_candidate_id = (
            candidate.source_candidate_ids[0] if candidate.source_candidate_ids else None
        )
        self._add_revision(
            event,
            revision_number=1,
            action_reason="V3 自动发布",
            analysis_version="hybrid-v3",
            prompt_version=run.prompt_version,
            model=run.model,
            run_id=run.id,
            candidate_id=source_candidate_id,
            snapshot_extra={
                "scores": asdict(candidate.scores),
                "source_candidate_ids": list(candidate.source_candidate_ids),
                "source_scores": [
                    {"candidate_id": source_id, "scores": asdict(scores)}
                    for source_id, scores in candidate.source_scores
                ],
                "source_reviews": [
                    {"candidate_id": source_id, "review": asdict(review)}
                    for source_id, review in candidate.source_reviews
                ],
                "global_selection": (
                    {
                        "relative_importance": candidate.global_importance,
                        "reason": candidate.global_reason,
                    }
                    if candidate.global_importance is not None
                    else None
                ),
                "display_summary": candidate.display_summary,
                "summary_model": candidate.summary_model,
            },
        )
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
            model=latest.model,
            run_id=latest.run_id,
            candidate_id=latest.candidate_id,
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

    def list(
        self,
        project_id: str,
        *,
        lane: str | None = None,
    ) -> list[EventNode]:
        statement = select(EventNode).where(EventNode.project_id == project_id)
        if lane is not None:
            if lane not in {"relationship", "shared_experience"}:
                raise ValueError("lane 必须是 relationship 或 shared_experience")
        events = list(self._session.scalars(statement.order_by(EventNode.created_at)))
        if lane is None:
            return events
        return [event for event in events if lane in event.source_lanes]

    def list_read(
        self,
        project_id: str,
        *,
        lane: str | None = None,
    ) -> builtins.list[EventNodeRead]:
        """批量加载节点解释信息，查询次数不随节点数量增长。"""

        events = self.list(project_id, lane=lane)
        if not events:
            return []
        event_ids = [event.id for event in events]
        revision_rows = self._session.execute(
            select(AnalysisRevision, AnalysisRun.import_id, EventCandidate)
            .outerjoin(AnalysisRun, AnalysisRun.id == AnalysisRevision.run_id)
            .outerjoin(
                EventCandidate,
                EventCandidate.id == AnalysisRevision.candidate_id,
            )
            .where(AnalysisRevision.event_id.in_(event_ids))
            .order_by(
                AnalysisRevision.event_id,
                AnalysisRevision.revision_number.desc(),
            )
        )
        revision_by_event: dict[
            str,
            tuple[AnalysisRevision, str | None, EventCandidate | None],
        ] = {}
        for revision, import_id, candidate in revision_rows:
            current = revision_by_event.get(revision.event_id)
            if (
                current is None
                or (current[2] is None and candidate is not None)
                or (
                    current[2] is None and current[0].run_id is None and revision.run_id is not None
                )
            ):
                revision_by_event[revision.event_id] = (
                    revision,
                    import_id,
                    candidate,
                )

        evidence_ids = {evidence_id for event in events for evidence_id in event.evidence_ids}
        message_rows = (
            list(
                self._session.execute(
                    select(Message, Participant.name)
                    .join(Participant, Participant.id == Message.participant_id)
                    .where(
                        Message.project_id == project_id,
                        Message.source_id.in_(evidence_ids),
                    )
                )
            )
            if evidence_ids
            else []
        )
        messages_by_import_and_source = {
            (message.import_id, message.source_id): (message, sender)
            for message, sender in message_rows
        }
        messages_by_source: dict[str, list[tuple[Message, str]]] = {}
        for message, sender in message_rows:
            messages_by_source.setdefault(message.source_id, []).append((message, sender))

        result: list[EventNodeRead] = []
        for event in events:
            revision_entry = revision_by_event.get(event.id)
            revision = revision_entry[0] if revision_entry is not None else None
            import_id = revision_entry[1] if revision_entry is not None else None
            candidate = revision_entry[2] if revision_entry is not None else None
            result.append(
                EventNodeRead.model_validate(event).model_copy(
                    update={
                        "score_components": _score_components(
                            candidate,
                            revision,
                        ),
                        "evidence_summaries": _evidence_summaries(
                            event.evidence_ids,
                            import_id=import_id,
                            messages_by_import_and_source=messages_by_import_and_source,
                            messages_by_source=messages_by_source,
                        ),
                        "analysis_version": (
                            revision.analysis_version if revision is not None else None
                        ),
                        "prompt_version": (
                            revision.prompt_version if revision is not None else None
                        ),
                        "model": revision.model if revision is not None else None,
                        "revision_number": (
                            revision.revision_number if revision is not None else None
                        ),
                        "analysis_run_id": (revision.run_id if revision is not None else None),
                    }
                )
            )
        return result

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
        model: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
        snapshot_extra: dict[str, object] | None = None,
    ) -> None:
        snapshot = _snapshot(event)
        if snapshot_extra:
            snapshot.update(snapshot_extra)
        self._session.add(
            AnalysisRevision(
                event_id=event.id,
                revision_number=revision_number,
                snapshot=snapshot,
                action_reason=action_reason,
                analysis_version=analysis_version,
                prompt_version=prompt_version,
                model=model,
                run_id=run_id,
                candidate_id=candidate_id,
            )
        )


def _snapshot(event: EventNode) -> dict[str, object]:
    fields = EDITABLE_FIELDS | V3_SNAPSHOT_FIELDS
    return {field: _json_snapshot_value(getattr(event, field)) for field in sorted(fields)}


def _json_snapshot_value(value: object) -> object:
    """将快照值转换为 SQLite JSON 可稳定保存的结构。"""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_snapshot_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_snapshot_value(item) for key, item in value.items()}
    return value


def _score_components(
    candidate: EventCandidate | None,
    revision: AnalysisRevision | None,
) -> EventScoreComponents | V2EventScoreComponents | None:
    if revision is not None and revision.analysis_version == "hybrid-v3":
        snapshot_scores = revision.snapshot.get("scores")
        scores = (
            snapshot_scores
            if isinstance(snapshot_scores, dict)
            else candidate.scores
            if candidate is not None
            else None
        )
        v3_required = (
            "event_significance",
            "relationship_impact",
            "evidence_quality",
            "persistence",
            "type_support",
            "model_confidence",
        )
        if not isinstance(scores, dict) or any(key not in scores for key in v3_required):
            return None
        return EventScoreComponents.model_validate({key: scores[key] for key in v3_required})
    if candidate is None:
        return None
    v2_required = (
        "state_change_strength",
        "persistence",
        "evidence_quality",
        "model_confidence",
    )
    if any(key not in candidate.scores for key in v2_required):
        return None
    return V2EventScoreComponents(
        state_change_strength=candidate.scores["state_change_strength"],
        persistence=candidate.scores["persistence"],
        evidence_quality=candidate.scores["evidence_quality"],
        model_confidence=candidate.scores["model_confidence"],
    )


def _evidence_summaries(
    evidence_ids: Sequence[str],
    *,
    import_id: str | None,
    messages_by_import_and_source: dict[tuple[str, str], tuple[Message, str]],
    messages_by_source: dict[str, list[tuple[Message, str]]],
) -> list[EventEvidenceSummary]:
    summaries: list[EventEvidenceSummary] = []
    for evidence_id in evidence_ids:
        row = (
            messages_by_import_and_source.get((import_id, evidence_id))
            if import_id is not None
            else None
        )
        if row is None and import_id is None:
            matches = messages_by_source.get(evidence_id, [])
            row = matches[0] if len(matches) == 1 else None
        if row is None:
            continue
        message, sender = row
        try:
            kind = MessageKind(message.kind)
        except ValueError:
            kind = MessageKind.UNKNOWN
        normalized = normalize_message(
            ImportedMessage(
                source_id=message.source_id,
                timestamp=message.timestamp,
                sender=sender,
                kind=kind,
                content=message.content,
                raw={},
            )
        )
        if normalized is None:
            continue
        summaries.append(
            EventEvidenceSummary(
                message_id=normalized.source_id,
                sender=normalized.sender,
                timestamp=normalized.timestamp,
                content=normalized.content,
            )
        )
    return summaries
