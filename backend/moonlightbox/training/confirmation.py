import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from moonlightbox.training.models import TimelineConfirmation

TRAINING_JOB_KIND = "digital_human_training_v1"
DEFAULT_TRAINING_CONFIG: dict[str, object] = {
    "base_model": "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "iterations": 600,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 512,
    "maximum_event_ratio": 0.25,
    "style_transfer_ratio": 0.5,
    "recency_window_days": 10,
    "recency_multiplier": 2,
    "runtime_style_transfer_enabled": False,
    "context_turns": 12,
    "training_protocol_version": (
        "persona-plain-text-private-chat-v19-runtime-style-transfer-aligned"
    ),
    "checkpoint_selection_version": "person-identity-held-out-v11-role-separated",
    "reply_protocol_version": "persona-text-v1",
    "memory_protocol_version": "evidence-layered-temporal-v3-one-pass-preference",
    # Direct service callers remain backward compatible. The application settings
    # snapshot enables this for new production confirmations.
    "memory_preference_training_enabled": False,
    "memory_preference_iterations": 40,
    "behavioral_timezone_offset_minutes": 480,
}


@dataclass(frozen=True)
class ConfirmationResult:
    confirmation: TimelineConfirmation
    job: Job


class TimelineConfirmationError(ValueError):
    pass


def confirmation_fingerprint(
    *,
    project_id: str,
    import_id: str,
    analysis_run_id: str,
    active_revisions: Sequence[tuple[str, int]],
    rejected_event_ids: Sequence[str],
    config: dict[str, object],
) -> str:
    payload = {
        "project_id": project_id,
        "import_id": import_id,
        "analysis_run_id": analysis_run_id,
        "active_revisions": sorted(
            [[event_id, revision] for event_id, revision in active_revisions]
        ),
        "rejected_event_ids": sorted(rejected_event_ids),
        "config": config,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConfirmationService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def confirm(
        self,
        *,
        project_id: str,
        analysis_run_id: str,
        event_revisions: Sequence[tuple[str, int]],
        config: dict[str, object] | None = None,
    ) -> ConfirmationResult:
        run = self._session.get(AnalysisRun, analysis_run_id)
        if (
            run is None
            or run.project_id != project_id
            or run.analysis_version != "hybrid-v3"
            or run.status != "succeeded"
        ):
            raise TimelineConfirmationError("只能确认已完成的 V3 时间轴")
        events = list(
            self._session.scalars(
                select(EventNode)
                .join(
                    AnalysisRevision,
                    AnalysisRevision.event_id == EventNode.id,
                )
                .where(
                    EventNode.project_id == project_id,
                    AnalysisRevision.run_id == run.id,
                )
                .distinct()
            )
        )
        active_events = {event.id: event for event in events if event.status == "active"}
        if not active_events:
            raise TimelineConfirmationError("至少需要保留一个有效节点")
        requested = dict(event_revisions)
        if len(requested) != len(event_revisions) or set(requested) != set(active_events):
            raise TimelineConfirmationError("有效节点集合已经变化，请刷新后重试")

        revision_snapshots: list[dict[str, object]] = []
        for event_id in sorted(active_events):
            latest = self._session.scalar(
                select(AnalysisRevision)
                .where(
                    AnalysisRevision.event_id == event_id,
                    AnalysisRevision.run_id == run.id,
                )
                .order_by(AnalysisRevision.revision_number.desc())
                .limit(1)
            )
            if latest is None or latest.revision_number != requested[event_id]:
                raise TimelineConfirmationError("节点修订已经变化，请刷新后重试")
            event = active_events[event_id]
            revision_snapshots.append(
                {
                    "event_id": event.id,
                    "revision_number": latest.revision_number,
                    "lane": event.lane,
                    "event_status": event.event_status,
                    "title": event.title,
                    "summary": event.summary,
                    "before_state": event.before_state,
                    "after_state": event.after_state,
                    "evidence_ids": list(event.evidence_ids),
                    "started_at": _json_value(event.started_at),
                    "ended_at": _json_value(event.ended_at),
                }
            )

        rejected_ids = sorted(event.id for event in events if event.status == "rejected")
        resolved_config = dict(DEFAULT_TRAINING_CONFIG)
        if config:
            resolved_config.update(config)
        fingerprint = confirmation_fingerprint(
            project_id=project_id,
            import_id=run.import_id,
            analysis_run_id=run.id,
            active_revisions=list(requested.items()),
            rejected_event_ids=rejected_ids,
            config=resolved_config,
        )
        confirmation = self._session.scalar(
            select(TimelineConfirmation).where(
                TimelineConfirmation.confirmation_fingerprint == fingerprint
            )
        )
        if confirmation is None:
            confirmation = TimelineConfirmation(
                project_id=project_id,
                import_id=run.import_id,
                analysis_run_id=run.id,
                confirmation_fingerprint=fingerprint,
                active_event_ids=sorted(active_events),
                rejected_event_ids=rejected_ids,
                event_revision_snapshots=revision_snapshots,
                config_snapshot=resolved_config,
                status="confirmed",
            )
            self._session.add(confirmation)
            try:
                self._session.commit()
            except IntegrityError:
                self._session.rollback()
                confirmation = self._session.scalar(
                    select(TimelineConfirmation).where(
                        TimelineConfirmation.confirmation_fingerprint == fingerprint
                    )
                )
                if confirmation is None:
                    raise
            self._session.refresh(confirmation)

        job = JobService(self._session).enqueue_unique(
            TRAINING_JOB_KIND,
            {
                "project_id": project_id,
                "import_id": run.import_id,
                "analysis_run_id": run.id,
                "confirmation_id": confirmation.id,
                "confirmation_fingerprint": fingerprint,
            },
            dedupe_key=f"training:{fingerprint}",
        )
        if confirmation.training_job_id != job.id:
            confirmation.training_job_id = job.id
            self._session.commit()
            self._session.refresh(confirmation)
        return ConfirmationResult(confirmation=confirmation, job=job)


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    return value
