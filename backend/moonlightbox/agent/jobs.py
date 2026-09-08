"""主体认知周期的后台任务处理。"""

from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent.extraction import COGNITIVE_EXTRACTION_JOB_KIND
from moonlightbox.agent.models import CognitiveCycle, PerceptionEvent
from moonlightbox.agent.service import ShadowCognitionService
from moonlightbox.agent.types import CognitionDraft, CognitionRequest
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService

COGNITIVE_CYCLE_JOB_KIND = "subject_cognitive_cycle"
_RETRY_DELAYS = (timedelta(minutes=1), timedelta(minutes=5), timedelta(minutes=30))


class CognitionGenerator(Protocol):
    """可注入的认知草稿生成器协议。"""

    def generate(self, request: CognitionRequest) -> CognitionDraft:
        """根据只读请求生成认知草稿。"""


def create_cognitive_cycle_handler(generator: CognitionGenerator) -> JobHandler:
    """创建不泄漏内部异常且不产生外显行为的任务处理器。"""

    def handle(job_service: JobService, job: Job) -> None:
        cycle_id = job.payload.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise JobHandlerError(
                "invalid_subject_cognition_job",
                "主体认知任务参数无效",
            )

        session = job_service.session
        cognition = ShadowCognitionService(session)
        try:
            cognition.mark_cycle_running(cycle_id)
            if cognition.invalidate_if_stale(cycle_id):
                return
            request = cognition.build_request(cycle_id)
            draft = generator.generate(request)

            session.expire_all()
            if cognition.invalidate_if_stale(cycle_id):
                return

            cycle = cognition.get_cycle(cycle_id)
            note = cognition.persist_shadow_result(cycle_id, draft, commit=False)
            cycle.structured_changes = {
                **cycle.structured_changes,
                "extraction_status": "pending",
            }
            message_id = job.payload.get("message_id")
            if isinstance(message_id, str) and message_id:
                from moonlightbox.branches.actor import (
                    BRANCH_CONVERSATION_JOB_KIND,
                )

                JobService(session).enqueue_unique(
                    BRANCH_CONVERSATION_JOB_KIND,
                    {
                        "project_id": cycle.project_id,
                        "branch_id": cycle.branch_id,
                        "message_id": message_id,
                        "cognitive_cycle_id": cycle.id,
                    },
                    dedupe_key=(
                        f"{BRANCH_CONVERSATION_JOB_KIND}:cognitive-ready:{cycle.id}"
                    ),
                    commit=False,
                )
            elif draft.expression_decision.express:
                trigger = session.get(PerceptionEvent, cycle.trigger_event_id)
                if trigger is not None and trigger.event_type == "elapsed_time":
                    from moonlightbox.branches.actor import (
                        BRANCH_CONVERSATION_JOB_KIND,
                    )

                    JobService(session).enqueue_unique(
                        BRANCH_CONVERSATION_JOB_KIND,
                        {
                            "project_id": cycle.project_id,
                            "branch_id": cycle.branch_id,
                            "proactive": True,
                            "cognitive_cycle_id": cycle.id,
                        },
                        dedupe_key=(
                            f"{BRANCH_CONVERSATION_JOB_KIND}:cognitive-proactive:"
                            f"{cycle.id}"
                        ),
                        commit=False,
                    )
            JobService(session).enqueue_unique(
                COGNITIVE_EXTRACTION_JOB_KIND,
                {
                    "project_id": cycle.project_id,
                    "branch_id": cycle.branch_id,
                    "cycle_id": cycle.id,
                    "note_id": note.id,
                },
                dedupe_key=f"{COGNITIVE_EXTRACTION_JOB_KIND}:{cycle.id}",
                commit=False,
            )
            suggestion = draft.suggested_next_wakeup
            if suggestion is not None:
                cognition.schedule_wakeup(
                    project_id=cycle.project_id,
                    branch_id=cycle.branch_id,
                    wake_at=suggestion.wake_at,
                    reason=suggestion.reason,
                    idempotency_key=suggestion.idempotency_key,
                    event_id=cycle.trigger_event_id,
                    model_version_id=cycle.model_version_id,
                    model_protocol_version=cycle.model_protocol_version,
                    evidence={"source_cycle_id": cycle.id, "shadow_mode": True},
                    commit=False,
                )
            session.commit()
        except Exception as error:
            session.rollback()
            try:
                cognition.mark_cycle_failed(cycle_id)
            except Exception:
                session.rollback()
            raise JobHandlerError(
                "subject_cognition_failed",
                "主体认知任务处理失败",
            ) from error

    return handle


def resume_retryable_cognitive_jobs(
    session: Session,
    *,
    now: datetime | None = None,
) -> int:
    """Requeue transient cognition failures with bounded durable backoff."""

    current = now or datetime.now(UTC)
    resumed = 0
    jobs = list(
        session.scalars(
            select(Job).where(
                Job.kind == COGNITIVE_CYCLE_JOB_KIND,
                Job.status == "failed",
                Job.error_code == "subject_cognition_failed",
            )
        )
    )
    for job in jobs:
        attempt = _retry_attempt(job.payload)
        if attempt >= len(_RETRY_DELAYS):
            continue
        updated_at = job.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        if updated_at + _RETRY_DELAYS[attempt] > current:
            continue
        cycle_id = job.payload.get("cycle_id")
        cycle = session.get(CognitiveCycle, cycle_id) if isinstance(cycle_id, str) else None
        if cycle is None or cycle.status != "failed":
            continue
        cycle.status = "pending"
        cycle.started_at = None
        cycle.completed_at = None
        job.payload = {**job.payload, "automatic_retry_count": attempt + 1}
        session.commit()
        JobService(session).resume(job.id)
        resumed += 1
    return resumed


def _retry_attempt(payload: dict[str, object]) -> int:
    value = payload.get("automatic_retry_count", 0)
    return int(value) if isinstance(value, int | float) and value >= 0 else 0
