from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from typing import Protocol, cast
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.events.models import AnalysisRun, EventNode
from moonlightbox.events.normalization import NormalizedMessage, normalize_messages
from moonlightbox.events.local_summary import LocalSummaryError
from moonlightbox.events.pipeline import (
    PipelineCancelledError,
    PipelineExecutionError,
    PipelineResult,
    safe_failure,
)
from moonlightbox.events.runs import (
    AnalysisRunLeaseLostError,
    AnalysisRunService,
    build_lane_slot_ids,
    parse_lane_slot_id,
)
from moonlightbox.events.schemas import EventCandidateUpsert
from moonlightbox.events.service import EventService
from moonlightbox.events.v3_ranking import (
    DEFAULT_MAXIMUM_NODES,
    DEFAULT_THRESHOLD,
    V3RankableCandidate,
    V3RankedCandidate,
    rank_candidates,
    score_candidate,
)
from moonlightbox.events.v3_reviewer import (
    V3CandidateBatchStructureError,
    V3CandidateIntegrityError,
    V3CandidateReview,
    V3EventCandidate,
    V3ExtractionResult,
    V3GlobalSelectionResult,
    V3PromptBudgetExceededError,
    V3ReviewStructureError,
)
from moonlightbox.events.v3_types import EventLane
from moonlightbox.events.v3_validation import validate_v3_candidate
from moonlightbox.events.validation import ValidationContext
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

V3_ANALYSIS_VERSION = "hybrid-v3"


class V3Reviewer(Protocol):
    def extract_candidates(
        self,
        window: AnalysisWindow,
        *,
        lane: EventLane,
        run_id: str | None = None,
    ) -> V3ExtractionResult: ...

    def review_candidate(
        self,
        candidate: V3EventCandidate,
        window: AnalysisWindow,
        *,
        run_id: str | None = None,
    ) -> V3CandidateReview: ...


class V3GlobalSelector(Protocol):
    def select_globally(
        self,
        *,
        candidates: Sequence[Mapping[str, object]],
        rejected_examples: Sequence[Mapping[str, object]],
        maximum_nodes: int,
        run_id: str | None = None,
    ) -> V3GlobalSelectionResult: ...


class V3NarrativeSummarizer(Protocol):
    def summarize(
        self,
        candidate: V3RankedCandidate,
        messages: Sequence[NormalizedMessage],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class V3PipelineConfig:
    """V3 流水线可审计且可恢复的稳定配置。"""

    session_gap: timedelta = timedelta(hours=6)
    character_budget: int = 12000
    overlap_messages: int = 8
    persistence_session_limit: int = 3
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
            "acceptance_threshold": self.acceptance_threshold,
            "maximum_nodes": self.maximum_nodes,
            "weights": {
                "event_significance": 0.25,
                "relationship_impact": 0.20,
                "evidence_quality": 0.25,
                "persistence": 0.10,
                "type_support": 0.10,
                "model_confidence": 0.10,
            },
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> V3PipelineConfig:
        """从已由 Job schema 校验的快照重建配置。"""

        return cls(
            session_gap=timedelta(seconds=cast(float, snapshot["session_gap_seconds"])),
            character_budget=cast(int, snapshot["character_budget"]),
            overlap_messages=cast(int, snapshot["overlap_messages"]),
            persistence_session_limit=cast(
                int,
                snapshot["persistence_session_limit"],
            ),
            acceptance_threshold=cast(
                float,
                snapshot["acceptance_threshold"],
            ),
            maximum_nodes=cast(int, snapshot["maximum_nodes"]),
        )


class EventV3Pipeline:
    """串联双通道提取、事实校验、软评分、合并与原子发布。"""

    def __init__(
        self,
        session: Session,
        reviewer: V3Reviewer,
        *,
        global_selector: V3GlobalSelector | None = None,
        narrative_summarizer: V3NarrativeSummarizer | None = None,
        config: V3PipelineConfig | None = None,
        worker_id: str | None = None,
        lease_duration: timedelta = timedelta(minutes=2),
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        job_id: str | None = None,
        job_worker_token: str | None = None,
    ) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration 必须大于零")
        if (job_id is None) != (job_worker_token is None):
            raise ValueError("job_id 与 job_worker_token 必须同时提供")
        self._session = session
        self._reviewer = reviewer
        self._global_selector = global_selector
        self._narrative_summarizer = narrative_summarizer
        self._config = config or V3PipelineConfig()
        self._worker_id = worker_id or f"pipeline-v3-{uuid4()}"
        self._lease_duration = lease_duration
        self._progress_callback = progress_callback
        self._should_cancel = should_cancel or (lambda: False)
        self._job_id = job_id
        self._job_worker_token = job_worker_token

    def run(
        self,
        *,
        project_id: str,
        import_id: str,
        relationship_prompt_version: str,
        shared_experience_prompt_version: str,
        global_selection_prompt_version: str | None = None,
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
        slot_ids = build_lane_slot_ids([window.window_id for window in windows])
        prompt_version = (
            f"relationship={relationship_prompt_version};"
            f"shared_experience={shared_experience_prompt_version};"
            f"global_selection={global_selection_prompt_version or 'disabled'}"
        )
        config_snapshot = dict(audit_config or self._config.snapshot())
        config_snapshot.update(
            {
                "relationship_prompt_version": relationship_prompt_version,
                "shared_experience_prompt_version": (shared_experience_prompt_version),
                "global_selection_prompt_version": global_selection_prompt_version,
            }
        )
        run_service = AnalysisRunService(self._session)
        run = run_service.get_or_create(
            project_id=project_id,
            import_id=import_id,
            analysis_version=V3_ANALYSIS_VERSION,
            prompt_version=prompt_version,
            model=model,
            config=config_snapshot,
            window_ids=slot_ids,
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
            raise PipelineExecutionError(f"{run.status} 状态不能执行 V3 流水线")
        self._session.commit()
        self._report_slot_progress(run)

        try:
            self._process_unfinished_slots(
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
            self._report_progress({"stage": "ranking", "run_id": run.id, "progress": 0.88})
            ranked = self._rank_persisted_candidates(
                run,
                messages,
                project_id=project_id,
            )
            run_service.heartbeat_lease(
                run.id,
                token=lease.token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            self._report_progress({"stage": "publishing", "run_id": run.id, "progress": 0.96})
            self._check_cancelled()
            published = EventService(self._session).publish_v3(
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
            self._mark_failed(run.id, failure.category, failure.persisted_message, lease.token)
            raise PipelineExecutionError(failure.public_message) from error

    def _process_unfinished_slots(
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
        windows_by_id = {window.window_id: window for window in windows}
        messages_by_id = {message.source_id: message for message in messages}
        ownership = {message.source_id: project_id for message in messages}
        for slot_index in run_service.unfinished_window_indexes(run.id):
            self._check_cancelled()
            run_service.heartbeat_lease(
                run.id,
                token=lease_token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            self._session.commit()
            slot_id = run.window_ids[slot_index]
            window_id, lane = parse_lane_slot_id(slot_id)
            window = windows_by_id[window_id]
            persistence = get_persistence_context(
                window,
                sessions,
                max_sessions=self._config.persistence_session_limit,
            )
            denominator = max(run.total_windows, 1)
            progress = 0.1 + 0.72 * slot_index / denominator
            self._report_progress(
                {
                    "stage": "extracting_candidates",
                    "lane": lane,
                    "run_id": run.id,
                    "completed_windows": slot_index,
                    "total_windows": run.total_windows,
                    "progress": progress,
                }
            )
            try:
                extraction = self._reviewer.extract_candidates(
                    window,
                    lane=lane,
                    run_id=run.id,
                )
            except V3CandidateBatchStructureError as error:
                extraction = V3ExtractionResult(
                    candidates=(),
                    rejected_candidates=error.diagnostics,
                )
                self._report_progress(
                    {
                        "stage": "extraction_batch_rejected",
                        "lane": lane,
                        "run_id": run.id,
                        "completed_windows": slot_index,
                        "total_windows": run.total_windows,
                        "rejected_candidates": error.rejected_count,
                        "progress": progress,
                    }
                )
            upserts: list[EventCandidateUpsert] = []
            for candidate in _unique_candidates(extraction.candidates):
                self._check_cancelled()
                try:
                    upsert = self._review_candidate(
                        candidate,
                        run=run,
                        project_id=project_id,
                        messages_by_id=messages_by_id,
                        ownership=ownership,
                        window=window,
                        persistence=persistence,
                    )
                except (
                    ValidationError,
                    V3CandidateIntegrityError,
                    V3PromptBudgetExceededError,
                    V3ReviewStructureError,
                ) as error:
                    self._report_progress(
                        {
                            "stage": "candidate_review_rejected",
                            "lane": lane,
                            "run_id": run.id,
                            "completed_windows": slot_index,
                            "total_windows": run.total_windows,
                            "error_type": type(error).__name__,
                            "progress": progress,
                        }
                    )
                    continue
                upserts.append(upsert)
            run_service.heartbeat_lease(
                run.id,
                token=lease_token,
                owner=self._worker_id,
                duration=self._lease_duration,
            )
            run_service.record_window_results(
                run.id,
                window_index=slot_index,
                window_id=slot_id,
                candidates=upserts,
                lease_token=lease_token,
                lease_owner=self._worker_id,
            )
            self._session.commit()
            self._report_slot_progress(run_service.get(run.id))

    def _review_candidate(
        self,
        candidate: V3EventCandidate,
        *,
        run: AnalysisRun,
        project_id: str,
        messages_by_id: Mapping[str, NormalizedMessage],
        ownership: Mapping[str, str],
        window: AnalysisWindow,
        persistence: PersistenceContext,
    ) -> EventCandidateUpsert:
        review = self._reviewer.review_candidate(
            candidate,
            window,
            run_id=run.id,
        )
        context = ValidationContext(
            expected_project_id=project_id,
            messages=messages_by_id,
            message_project_ids=ownership,
            analysis_window=window,
            persistence_context=persistence,
        )
        validation = validate_v3_candidate(candidate, review, context)
        if validation.is_valid:
            rankable = _rankable(
                candidate,
                review,
                messages_by_id,
                window,
                persistence,
            )
            return EventCandidateUpsert(
                candidate_key=candidate.candidate_key,
                raw_payload=candidate.model_dump(mode="json"),
                review_payload=review.model_dump(mode="json"),
                status="accepted",
                scores=asdict(score_candidate(rankable).scores),
            )
        return EventCandidateUpsert(
            candidate_key=candidate.candidate_key,
            raw_payload=candidate.model_dump(mode="json"),
            review_payload=review.model_dump(mode="json"),
            status="rejected",
            rejection_reason=",".join(validation.rejection_reasons),
            scores={},
        )

    def _rank_persisted_candidates(
        self,
        run: AnalysisRun,
        messages: Sequence[NormalizedMessage],
        *,
        project_id: str,
    ) -> list[V3RankedCandidate]:
        messages_by_id = {message.source_id: message for message in messages}
        rankables: list[V3RankableCandidate] = []
        for stored in AnalysisRunService(self._session).candidates(run.id):
            if stored.status != "accepted" or stored.review_payload is None:
                continue
            candidate = V3EventCandidate.model_validate(stored.raw_payload)
            review = V3CandidateReview.model_validate(stored.review_payload)
            start = messages_by_id[candidate.start_message_id]
            end = messages_by_id[candidate.end_message_id]
            valid_follow_up_ids = tuple(
                evidence_id
                for evidence_id in review.evidence_ids
                if evidence_id in messages_by_id
                and (
                    messages_by_id[evidence_id].timestamp,
                    evidence_id,
                )
                > (end.timestamp, candidate.end_message_id)
            )
            rankables.append(
                V3RankableCandidate(
                    candidate=candidate,
                    review=review,
                    started_at=start.timestamp,
                    ended_at=end.timestamp,
                    valid_follow_up_ids=valid_follow_up_ids,
                    source_lane=candidate.lane,
                    source_candidate_id=stored.id,
                )
            )
        ranked = rank_candidates(
            rankables,
            # The global model is an additional relative-importance judge, not
            # a bypass around the deterministic local quality floor.
            threshold=self._config.acceptance_threshold,
            maximum_nodes=(
                None
                if self._global_selector is not None
                else self._config.maximum_nodes
            ),
        )
        if self._global_selector is None or not ranked:
            return ranked
        payloads = [
            _global_candidate_payload(candidate, messages_by_id)
            for candidate in ranked
        ]
        selection = self._global_selector.select_globally(
            candidates=payloads,
            rejected_examples=self._rejected_examples(project_id),
            maximum_nodes=self._config.maximum_nodes,
            run_id=run.id,
        )
        ranked_by_key = {
            candidate.candidate.candidate_key: candidate
            for candidate in ranked
        }
        selected = [
            replace(
                ranked_by_key[item.candidate_key],
                global_importance=item.relative_importance,
                global_reason=item.reason,
            )
            for item in selection.selected
        ]
        if self._narrative_summarizer is None:
            return selected
        summarized: list[V3RankedCandidate] = []
        for selected_candidate in selected:
            node_messages = _messages_in_candidate_range(selected_candidate, messages)
            try:
                display_summary = self._narrative_summarizer.summarize(
                    selected_candidate,
                    node_messages,
                )
            except LocalSummaryError:
                # Narrative summaries are an optional display enrichment. A
                # 本地摘要模型缺失不能使已经验收的节点失效。
                # event candidates or prevent publication of the analysis run.
                return selected
            summarized.append(
                replace(
                    selected_candidate,
                    display_summary=display_summary,
                    summary_model=str(
                        getattr(
                            self._narrative_summarizer,
                            "model_path",
                            "local",
                        )
                    ),
                )
            )
        return summarized

    def _rejected_examples(self, project_id: str) -> list[dict[str, object]]:
        rows = list(
            self._session.scalars(
                select(EventNode)
                .where(
                    EventNode.project_id == project_id,
                    EventNode.status == "rejected",
                )
                .order_by(EventNode.created_at.desc())
                .limit(12)
            )
        )
        return [
            {
                "title": row.title,
                "summary": row.summary,
                "reason": row.reason,
                "type": row.type,
            }
            for row in rows
        ]

    def _report_slot_progress(self, run: AnalysisRun) -> None:
        denominator = max(run.total_windows, 1)
        self._report_progress(
            {
                "stage": "slots",
                "run_id": run.id,
                "completed_windows": run.completed_windows,
                "total_windows": run.total_windows,
                "progress": 0.1 + 0.72 * run.completed_windows / denominator,
            }
        )

    def _report_progress(self, checkpoint: dict[str, object]) -> None:
        if self._progress_callback is not None:
            self._progress_callback(checkpoint)

    def _check_cancelled(self) -> None:
        if self._should_cancel():
            raise PipelineCancelledError("V3 事件分析已取消")

    def _interrupt_run(self, run_id: str, *, lease_token: str) -> None:
        self._session.rollback()
        run = self._session.get(AnalysisRun, run_id)
        if run is None or run.status != "running":
            return
        try:
            AnalysisRunService(self._session).interrupt(
                run_id,
                error_category="worker_interrupted",
                error_message="V3 事件分析已取消或 Worker 正在退出",
                lease_token=lease_token,
                lease_owner=self._worker_id,
            )
        except AnalysisRunLeaseLostError:
            self._session.rollback()
            return
        self._session.commit()

    def _mark_failed(
        self,
        run_id: str,
        category: str,
        message: str,
        lease_token: str,
    ) -> None:
        self._session.rollback()
        run = self._session.get(AnalysisRun, run_id)
        if run is None or run.status != "running":
            return
        try:
            AnalysisRunService(self._session).fail(
                run_id,
                error_category=category,
                error_message=message,
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
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    messages: Mapping[str, NormalizedMessage],
    window: AnalysisWindow,
    persistence: PersistenceContext,
) -> V3RankableCandidate:
    start = messages.get(candidate.start_message_id)
    end = messages.get(candidate.end_message_id)
    started_at = start.timestamp if start is not None else window.messages[0].timestamp
    ended_at = end.timestamp if end is not None else window.messages[-1].timestamp
    follow_up_messages: list[NormalizedMessage] = []
    found_end = False
    for message in window.messages:
        if found_end:
            follow_up_messages.append(message)
        elif message.source_id == candidate.end_message_id:
            found_end = True
    follow_up_messages.extend(persistence.messages)
    valid_ids = {message.source_id for message in follow_up_messages}
    return V3RankableCandidate(
        candidate=candidate,
        review=review,
        started_at=started_at,
        ended_at=ended_at,
        valid_follow_up_ids=tuple(
            evidence_id for evidence_id in review.evidence_ids if evidence_id in valid_ids
        ),
        source_lane=candidate.lane,
    )


def _unique_candidates(
    candidates: Sequence[V3EventCandidate],
) -> tuple[V3EventCandidate, ...]:
    by_key: dict[str, V3EventCandidate] = {}
    for candidate in candidates:
        by_key.setdefault(candidate.candidate_key, candidate)
    return tuple(by_key[key] for key in sorted(by_key))


def _global_candidate_payload(
    ranked: V3RankedCandidate,
    messages: Mapping[str, NormalizedMessage],
) -> dict[str, object]:
    source = ranked.candidate
    evidence = [
        {
            "message_id": evidence_id,
            "sender": messages[evidence_id].sender,
            "content": messages[evidence_id].content[:300],
            "timestamp": messages[evidence_id].timestamp.isoformat(),
        }
        for evidence_id in source.evidence_ids
        if evidence_id in messages
    ]
    return {
        "candidate_key": source.candidate_key,
        "lane": source.lane,
        "type": source.type,
        "title": source.title,
        "event_status": source.event_status,
        "summary": source.summary,
        "reason": source.reason,
        "started_at": ranked.started_at.isoformat(),
        "ended_at": ranked.ended_at.isoformat(),
        "source_lanes": list(ranked.source_lanes),
        "before_state": source.before_state,
        "after_state": source.after_state,
        "topic": source.topic,
        "scores": asdict(ranked.scores),
        "review": {
            "facts_supported": ranked.review.facts_supported,
            "occurrence_supported": ranked.review.occurrence_supported,
            "bilateral_confirmation": ranked.review.bilateral_confirmation,
            "reason": ranked.review.reason,
        },
        "evidence": evidence,
    }


def _messages_in_candidate_range(
    candidate: V3RankedCandidate,
    messages: Sequence[NormalizedMessage],
) -> tuple[NormalizedMessage, ...]:
    start_id = candidate.candidate.start_message_id
    end_id = candidate.candidate.end_message_id
    positions = {
        message.source_id: index
        for index, message in enumerate(messages)
        if message.source_id in {start_id, end_id}
    }
    if start_id not in positions or end_id not in positions:
        raise ValueError("节点摘要范围消息不存在")
    start = min(positions[start_id], positions[end_id])
    end = max(positions[start_id], positions[end_id])
    return tuple(messages[start : end + 1])
