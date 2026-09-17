import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.events.cloud_client import NodeAnalysisCloudError
from moonlightbox.events.models import AnalysisRun
from moonlightbox.events.normalization import NormalizedMessage, normalize_messages
from moonlightbox.events.ranking import (
    DEFAULT_MAXIMUM_NODES,
    DEFAULT_THRESHOLD,
    RankableCandidate,
    RankedCandidate,
    merge_ranked_candidates,
    rank_scored_candidates,
    score_candidate,
)
from moonlightbox.events.reviewer import (
    CandidateEvidenceUnavailableError,
    EventCandidate,
    EventCandidateReview,
    TwoStageEventReviewer,
)
from moonlightbox.events.runs import AnalysisRunLeaseLostError, AnalysisRunService
from moonlightbox.events.schemas import EventCandidateUpsert
from moonlightbox.events.service import EventService
from moonlightbox.events.validation import ValidationContext, validate_candidate
from moonlightbox.events.windowing import (
    AnalysisWindow,
    MessageSession,
    PersistenceContext,
    build_analysis_windows,
    get_persistence_context,
    split_sessions,
)
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.jobs.service import JobLeaseLostError


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """V2 流水线的稳定配置快照。"""

    session_gap: timedelta = timedelta(hours=6)
    character_budget: int = 12000
    overlap_messages: int = 8
    persistence_session_limit: int = 3
    persistence_character_budget: int = 12000
    evidence_alignment_threshold: float = 0.7
    acceptance_threshold: float = DEFAULT_THRESHOLD
    maximum_nodes: int = DEFAULT_MAXIMUM_NODES

    def __post_init__(self) -> None:
        if self.session_gap <= timedelta(0):
            raise ValueError("session_gap 必须大于零")
        if self.character_budget <= 0:
            raise ValueError("character_budget 必须大于零")
        if self.overlap_messages < 0:
            raise ValueError("overlap_messages 不能为负数")
        if not 0 <= self.persistence_session_limit <= 3:
            raise ValueError("persistence_session_limit 必须在 0 到 3 之间")
        if self.persistence_character_budget <= 0:
            raise ValueError("persistence_character_budget 必须大于零")
        if not 0 <= self.evidence_alignment_threshold <= 1:
            raise ValueError("evidence_alignment_threshold 必须在 0 到 1 之间")
        if not 0 <= self.acceptance_threshold <= 1:
            raise ValueError("acceptance_threshold 必须在 0 到 1 之间")
        if self.maximum_nodes < 0:
            raise ValueError("maximum_nodes 不能为负数")

    def snapshot(self) -> dict[str, object]:
        return {
            "session_gap_seconds": self.session_gap.total_seconds(),
            "character_budget": self.character_budget,
            "overlap_messages": self.overlap_messages,
            "persistence_session_limit": self.persistence_session_limit,
            "persistence_character_budget": self.persistence_character_budget,
            "evidence_alignment_threshold": self.evidence_alignment_threshold,
            "acceptance_threshold": self.acceptance_threshold,
            "maximum_nodes": self.maximum_nodes,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> "PipelineConfig":
        """从已校验的 Job 快照重建流水线配置。"""

        return cls(
            session_gap=timedelta(seconds=cast(float, snapshot["session_gap_seconds"])),
            character_budget=cast(int, snapshot["character_budget"]),
            overlap_messages=cast(int, snapshot["overlap_messages"]),
            persistence_session_limit=cast(
                int,
                snapshot["persistence_session_limit"],
            ),
            persistence_character_budget=cast(
                int,
                snapshot["persistence_character_budget"],
            ),
            evidence_alignment_threshold=cast(
                float,
                snapshot["evidence_alignment_threshold"],
            ),
            acceptance_threshold=cast(float, snapshot["acceptance_threshold"]),
            maximum_nodes=cast(int, snapshot["maximum_nodes"]),
        )


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: str
    event_count: int


class PipelineExecutionError(RuntimeError):
    """窗口处理或发布失败，运行状态已持久化为 failed。"""


class PipelineCancelledError(RuntimeError):
    """任务取消或 Worker 停止，运行状态已安全中断。"""


@dataclass(frozen=True, slots=True)
class _SafeFailure:
    category: str
    persisted_message: str
    public_message: str


class EventV2Pipeline:
    """串联规范化、两阶段复核、持久化、全局排名与原子发布。"""

    def __init__(
        self,
        session: Session,
        reviewer: TwoStageEventReviewer,
        *,
        config: PipelineConfig | None = None,
        worker_id: str | None = None,
        lease_duration: timedelta = timedelta(minutes=2),
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        job_id: str | None = None,
        job_worker_token: str | None = None,
    ) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration 必须大于零")
        self._session = session
        self._reviewer = reviewer
        self._config = config or PipelineConfig()
        self._worker_id = worker_id or f"pipeline-{uuid4()}"
        self._lease_duration = lease_duration
        self._progress_callback = progress_callback
        self._should_cancel = should_cancel or (lambda: False)
        if (job_id is None) != (job_worker_token is None):
            raise ValueError("job_id 与 job_worker_token 必须同时提供")
        self._job_id = job_id
        self._job_worker_token = job_worker_token

    def run(
        self,
        *,
        project_id: str,
        import_id: str,
        prompt_version: str,
        model: str,
        audit_config: Mapping[str, object] | None = None,
    ) -> PipelineResult:
        self._check_cancelled()
        self._report_progress({"stage": "loading_messages", "progress": 0.05})
        imported_messages = self._load_messages(project_id, import_id)
        messages = normalize_messages(imported_messages)
        sessions = split_sessions(messages, gap=self._config.session_gap)
        windows = build_analysis_windows(
            sessions,
            character_budget=self._config.character_budget,
            overlap_messages=self._config.overlap_messages,
        )
        run_service = AnalysisRunService(self._session)
        run = run_service.get_or_create(
            project_id=project_id,
            import_id=import_id,
            analysis_version="hybrid-v2",
            prompt_version=prompt_version,
            model=model,
            config=audit_config or self._config.snapshot(),
            window_ids=[window.window_id for window in windows],
        )
        if run.status == "succeeded":
            return PipelineResult(
                run_id=run.id,
                event_count=len(EventService(self._session)._published_events(run.id)),
            )
        lease = run_service.acquire_lease(
            run.id,
            owner=self._worker_id,
            duration=self._lease_duration,
        )
        if run.status == "queued":
            run_service.start(run.id)
        elif run.status in {"failed", "interrupted"}:
            run_service.resume(run.id)
        elif run.status != "running":
            raise PipelineExecutionError(f"{run.status} 状态不能执行流水线")
        self._session.commit()
        self._report_window_progress(run)

        try:
            self._process_unfinished_windows(
                run,
                project_id=project_id,
                messages=messages,
                sessions=sessions,
                windows=windows,
                lease_token=lease.token,
            )
            run_service.heartbeat_lease(
                run.id,
                token=lease.token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            self._session.commit()
            self._check_cancelled()
            self._report_progress(
                {
                    "stage": "ranking",
                    "run_id": run.id,
                    "progress": 0.85,
                }
            )
            ranked = self._rank_persisted_candidates(run, messages)
            run_service.heartbeat_lease(
                run.id,
                token=lease.token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            self._report_progress(
                {
                    "stage": "publishing",
                    "run_id": run.id,
                    "progress": 0.95,
                }
            )
            self._check_cancelled()
            published = EventService(self._session).publish_v2(
                project_id=project_id,
                run_id=run.id,
                candidates=ranked,
                lease_token=lease.token,
                lease_owner=self._worker_id,
                job_id=self._job_id,
                job_worker_token=self._job_worker_token,
            )
            return PipelineResult(run_id=run.id, event_count=len(published))
        except JobLeaseLostError as error:
            self._interrupt_run(run.id, lease_token=lease.token)
            raise PipelineCancelledError("Job 已取消或租约已被接管") from error
        except PipelineCancelledError:
            self._interrupt_run(run.id, lease_token=lease.token)
            raise
        except Exception as error:
            failure = safe_failure(error)
            self._mark_failed(run.id, failure, lease_token=lease.token)
            raise PipelineExecutionError(failure.public_message) from error

    def _process_unfinished_windows(
        self,
        run: AnalysisRun,
        *,
        project_id: str,
        messages: Sequence[NormalizedMessage],
        sessions: Sequence[MessageSession],
        windows: Sequence[AnalysisWindow],
        lease_token: str,
    ) -> None:
        run_service = AnalysisRunService(self._session)
        messages_by_id = {message.source_id: message for message in messages}
        ownership = {message.source_id: project_id for message in messages}
        for window_index in run_service.unfinished_window_indexes(run.id):
            self._check_cancelled()
            run_service.heartbeat_lease(
                run.id,
                token=lease_token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            self._session.commit()
            window = windows[window_index]
            persistence = get_persistence_context(
                window,
                sessions,
                max_sessions=self._config.persistence_session_limit,
            )
            denominator = max(run.total_windows, 1)
            window_base = 0.1 + 0.7 * window_index / denominator
            self._report_progress(
                {
                    "stage": "extracting_candidates",
                    "run_id": run.id,
                    "completed_windows": window_index,
                    "total_windows": run.total_windows,
                    "progress": window_base,
                }
            )
            extraction = self._reviewer.extract_candidates(window, run_id=run.id)
            self._report_progress(
                {
                    "stage": "reviewing_persistence",
                    "run_id": run.id,
                    "completed_windows": window_index,
                    "total_windows": run.total_windows,
                    "progress": window_base + 0.35 / denominator,
                }
            )
            upserts: list[EventCandidateUpsert] = []
            for candidate in _unique_candidates(extraction.candidates):
                self._check_cancelled()
                upserts.append(
                    self._review_and_validate(
                        candidate,
                        run=run,
                        project_id=project_id,
                        window=window,
                        persistence=persistence,
                        messages_by_id=messages_by_id,
                        ownership=ownership,
                    )
                )
            # 云端处理可能超过 TTL；token 未被接管时在写入前原子续租。
            run_service.heartbeat_lease(
                run.id,
                token=lease_token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            run_service.record_window_results(
                run.id,
                window_index=window_index,
                window_id=window.window_id,
                candidates=upserts,
                lease_token=lease_token,
                lease_owner=self._worker_id,
            )
            # 每个窗口单独提交，失败恢复时 checkpoint 与候选不会丢失。
            self._session.commit()
            refreshed = run_service.get(run.id)
            self._report_window_progress(refreshed)

    def _report_window_progress(self, run: AnalysisRun) -> None:
        denominator = max(run.total_windows, 1)
        window_progress = run.completed_windows / denominator
        self._report_progress(
            {
                "stage": "windows",
                "run_id": run.id,
                "completed_windows": run.completed_windows,
                "total_windows": run.total_windows,
                "progress": 0.1 + 0.7 * window_progress,
            }
        )

    def _report_progress(self, checkpoint: dict[str, object]) -> None:
        if self._progress_callback is not None:
            self._progress_callback(checkpoint)

    def _check_cancelled(self) -> None:
        if self._should_cancel():
            raise PipelineCancelledError("事件分析已取消")

    def _interrupt_run(self, run_id: str, *, lease_token: str) -> None:
        self._session.rollback()
        run = self._session.get(AnalysisRun, run_id)
        if run is None or run.status != "running":
            return
        try:
            AnalysisRunService(self._session).interrupt(
                run_id,
                error_category="worker_interrupted",
                error_message="事件分析已取消或 Worker 正在退出",
                lease_token=lease_token,
                lease_owner=self._worker_id,
            )
        except AnalysisRunLeaseLostError:
            self._session.rollback()
            return
        self._session.commit()

    def _review_and_validate(
        self,
        candidate: EventCandidate,
        *,
        run: AnalysisRun,
        project_id: str,
        window: AnalysisWindow,
        persistence: PersistenceContext,
        messages_by_id: Mapping[str, NormalizedMessage],
        ownership: Mapping[str, str],
    ) -> EventCandidateUpsert:
        self._check_cancelled()
        try:
            review = self._reviewer.review_candidate(
                candidate,
                window,
                persistence,
                run_id=run.id,
            )
        except CandidateEvidenceUnavailableError:
            return EventCandidateUpsert(
                candidate_key=candidate.candidate_key,
                raw_payload=candidate.model_dump(mode="json"),
                review_payload=None,
                status="rejected",
                rejection_reason="unknown_message_id",
                scores={},
            )
        self._check_cancelled()
        validation = validate_candidate(
            candidate,
            review,
            ValidationContext(
                expected_project_id=project_id,
                messages=messages_by_id,
                message_project_ids=ownership,
                analysis_window=window,
                persistence_context=persistence,
            ),
            evidence_alignment_threshold=self._config.evidence_alignment_threshold,
        )
        rankable = _rankable(candidate, review, messages_by_id, window, persistence)
        scored = score_candidate(rankable)
        accepted = (
            validation.accepted and scored.scores["total"] >= self._config.acceptance_threshold
        )
        reasons = list(validation.rejection_reasons)
        if validation.accepted and not accepted:
            reasons.append("below_threshold")
        return EventCandidateUpsert(
            candidate_key=candidate.candidate_key,
            raw_payload=candidate.model_dump(mode="json"),
            review_payload=review.model_dump(mode="json"),
            status="accepted" if accepted else "rejected",
            rejection_reason=None if accepted else ",".join(reasons),
            scores=scored.scores,
        )

    def _rank_persisted_candidates(
        self,
        run: AnalysisRun,
        messages: Sequence[NormalizedMessage],
    ) -> list[RankedCandidate]:
        messages_by_id = {message.source_id: message for message in messages}
        ranked_candidates: list[RankedCandidate] = []
        for stored in AnalysisRunService(self._session).candidates(run.id):
            if stored.status != "accepted" or stored.review_payload is None:
                continue
            candidate = EventCandidate.model_validate(stored.raw_payload)
            review = EventCandidateReview.model_validate(stored.review_payload)
            start_message = messages_by_id[candidate.start_message_id]
            end_message = messages_by_id[candidate.end_message_id]
            ranked_candidates.append(
                RankedCandidate(
                    candidate=candidate,
                    review=review,
                    started_at=start_message.timestamp,
                    ended_at=end_message.timestamp,
                    valid_follow_up_ids=tuple(review.evidence_ids),
                    scores=dict(stored.scores),
                    source_candidate_id=stored.id,
                )
            )
        scored = rank_scored_candidates(
            ranked_candidates,
            threshold=self._config.acceptance_threshold,
            maximum_nodes=None,
        )
        order = {message.source_id: index for index, message in enumerate(messages)}
        return merge_ranked_candidates(
            scored,
            evidence_order=order,
        )[: self._config.maximum_nodes]

    def _mark_failed(
        self,
        run_id: str,
        failure: _SafeFailure,
        *,
        lease_token: str,
    ) -> None:
        self._session.rollback()
        run = self._session.get(AnalysisRun, run_id)
        if run is None or run.status != "running":
            return
        try:
            AnalysisRunService(self._session).fail(
                run_id,
                error_category=failure.category,
                error_message=failure.persisted_message,
                lease_token=lease_token,
                lease_owner=self._worker_id,
            )
        except AnalysisRunLeaseLostError:
            self._session.rollback()
            return
        self._session.commit()

    def _load_messages(
        self,
        project_id: str,
        import_id: str,
    ) -> list[ImportedMessage]:
        imported = self._session.get(ImportSource, import_id)
        if imported is None or imported.project_id != project_id:
            raise ValueError("导入批次不属于指定项目")
        rows = self._session.execute(
            select(Message, Participant.name)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.import_id == import_id,
                Message.project_id == project_id,
            )
            .order_by(Message.timestamp, Message.source_id)
        )
        loaded: list[ImportedMessage] = []
        for message, sender in rows:
            try:
                kind = MessageKind(message.kind)
            except ValueError:
                kind = MessageKind.UNKNOWN
            loaded.append(
                ImportedMessage(
                    source_id=message.source_id,
                    timestamp=message.timestamp,
                    sender=sender,
                    kind=kind,
                    content=message.content,
                    raw=message.raw,
                )
            )
        return loaded


def _rankable(
    candidate: EventCandidate,
    review: EventCandidateReview,
    messages: Mapping[str, NormalizedMessage],
    window: AnalysisWindow,
    persistence: PersistenceContext,
) -> RankableCandidate:
    start_message = messages.get(candidate.start_message_id)
    end_message = messages.get(candidate.end_message_id)
    # 无效边界仅用于生成拒绝审计分项，不能让单个候选中断整个窗口。
    started_at = (
        start_message.timestamp if start_message is not None else window.messages[0].timestamp
    )
    ended_at = end_message.timestamp if end_message is not None else window.messages[-1].timestamp
    return RankableCandidate(
        candidate=candidate,
        review=review,
        started_at=started_at,
        ended_at=ended_at,
        valid_follow_up_ids=_valid_follow_up_ids(
            candidate,
            window,
            persistence,
        ),
    )


def _valid_follow_up_ids(
    candidate: EventCandidate,
    window: AnalysisWindow,
    persistence: PersistenceContext,
) -> tuple[str, ...]:
    same_window: tuple[NormalizedMessage, ...] = ()
    for index, message in enumerate(window.messages):
        if message.source_id == candidate.end_message_id:
            same_window = window.messages[index + 1 :]
            break
    return tuple(message.source_id for message in same_window + persistence.messages)


def _unique_candidates(
    candidates: Sequence[EventCandidate],
) -> tuple[EventCandidate, ...]:
    """同窗口重复键只处理一次，避免重复云端复核与数据库冲突。"""

    by_key: dict[str, EventCandidate] = {}
    for candidate in candidates:
        by_key.setdefault(candidate.candidate_key, candidate)
    return tuple(by_key[key] for key in sorted(by_key))


def safe_failure(error: Exception) -> _SafeFailure:
    if isinstance(error, NodeAnalysisCloudError):
        allowed_diagnostic = {
            key: value
            for key, value in error.diagnostic.items()
            if key in {"configuration", "exception_type", "http_status"}
            and isinstance(value, (str, int))
        }
        payload = {
            "code": error.code.value,
            "diagnostic": allowed_diagnostic,
        }
        return _SafeFailure(
            category=f"cloud:{error.code.value}",
            persisted_message=json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            public_message="节点分析云端服务失败",
        )
    return _SafeFailure(
        category=type(error).__name__[:64],
        persisted_message="分析执行失败",
        public_message="分析执行失败",
    )
