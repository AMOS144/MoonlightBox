import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from moonlightbox.events.config import (
    AnalysisConfigIntegrityError,
    AnalysisWindowManifestIntegrityError,
    config_fingerprint,
    ensure_config_integrity,
    ensure_window_manifest_integrity,
    normalize_config,
    normalize_window_ids,
    window_manifest_fingerprint,
)
from moonlightbox.events.config import (
    InvalidWindowManifestError as InvalidWindowManifestError,
)
from moonlightbox.events.models import AnalysisRun, EventCandidate
from moonlightbox.events.schemas import EventCandidateUpsert
from moonlightbox.events.v3_reviewer import (
    V3CandidateReview,
    V3EventCandidate,
    stable_v3_candidate_key,
)
from moonlightbox.events.v3_types import EventLane
from moonlightbox.imports.models import ImportSource

LANE_SLOT_SEPARATOR = "::"
EVENT_LANES: tuple[EventLane, ...] = (
    "relationship",
    "shared_experience",
)
V3_ANALYSIS_VERSIONS = frozenset({"hybrid-v3"})

TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "cancelled"},
    "running": {"interrupted", "failed", "succeeded", "cancelled"},
    "interrupted": {"running", "cancelled"},
    "failed": {"running", "cancelled"},
}


class AnalysisRunNotFoundError(LookupError):
    pass


class InvalidAnalysisRunTransitionError(ValueError):
    pass


class InvalidAnalysisCheckpointError(ValueError):
    pass


class AnalysisRunManifestError(ValueError):
    pass


class AnalysisImportMismatchError(ValueError):
    pass


class AnalysisRunConcurrencyError(RuntimeError):
    pass


class AnalysisDatabaseBusyError(RuntimeError):
    pass


class AnalysisRunAlreadyRunningError(RuntimeError):
    pass


class AnalysisRunLeaseLostError(RuntimeError):
    pass


def build_lane_slot_ids(window_ids: Sequence[str]) -> list[str]:
    """按窗口及固定通道顺序构造 V3 处理槽位。"""

    normalized = normalize_window_ids(window_ids)
    if any("::" in window_id for window_id in normalized):
        raise InvalidWindowManifestError(
            f"物理 window_id 不能包含保留分隔符 {LANE_SLOT_SEPARATOR!r}"
        )
    return [
        f"{window_id}{LANE_SLOT_SEPARATOR}{lane}"
        for window_id in normalized
        for lane in EVENT_LANES
    ]


def parse_lane_slot_id(slot_id: str) -> tuple[str, EventLane]:
    """严格解析 V3 槽位，拒绝伪造通道及多重分隔符。"""

    if slot_id.count(LANE_SLOT_SEPARATOR) != 1:
        raise InvalidWindowManifestError("lane slot 必须包含且仅包含一个保留分隔符")
    window_id, lane = slot_id.split(LANE_SLOT_SEPARATOR)
    if not window_id.strip() or lane not in EVENT_LANES:
        raise InvalidWindowManifestError("lane slot 包含无效 window_id 或 lane")
    if lane == "relationship":
        return window_id, "relationship"
    return window_id, "shared_experience"


def resolve_lane_slot_id(slot_id: str) -> tuple[str, EventLane]:
    """解析槽位并返回物理窗口 ID 与事件通道。"""

    return parse_lane_slot_id(slot_id)


def _uses_lane_slots(analysis_version: str) -> bool:
    return analysis_version in V3_ANALYSIS_VERSIONS


@dataclass(frozen=True, slots=True)
class AnalysisRunLease:
    owner: str
    token: str
    expires_at: datetime


class AnalysisRunService:
    """管理分析运行状态与候选持久化，不替调用者提交事务。"""

    def __init__(
        self,
        session: Session,
        *,
        lock_retry_attempts: int = 3,
        lock_retry_delay: float = 0.01,
    ) -> None:
        if lock_retry_attempts < 1:
            raise ValueError("lock_retry_attempts 必须至少为 1")
        if lock_retry_delay < 0:
            raise ValueError("lock_retry_delay 不能为负数")
        self._session = session
        self._lock_retry_attempts = lock_retry_attempts
        self._lock_retry_delay = lock_retry_delay

    def get_or_create(
        self,
        *,
        project_id: str,
        import_id: str,
        analysis_version: str,
        prompt_version: str,
        model: str,
        config: Mapping[str, object],
        window_ids: Sequence[str],
    ) -> AnalysisRun:
        window_manifest = normalize_window_ids(window_ids)
        if _uses_lane_slots(analysis_version):
            for slot_id in window_manifest:
                parse_lane_slot_id(slot_id)
        for attempt in range(self._lock_retry_attempts):
            try:
                return self._get_or_create_once(
                    project_id=project_id,
                    import_id=import_id,
                    analysis_version=analysis_version,
                    prompt_version=prompt_version,
                    model=model,
                    config=config,
                    window_ids=window_manifest,
                )
            except OperationalError as error:
                if not self._is_database_locked(error):
                    raise
                if attempt + 1 == self._lock_retry_attempts:
                    raise AnalysisDatabaseBusyError(
                        f"数据库持续锁定，已重试 {self._lock_retry_attempts} 次"
                    ) from error
                if self._lock_retry_delay:
                    time.sleep(self._lock_retry_delay)
        raise AssertionError("锁重试循环不应执行到此处")

    def _get_or_create_once(
        self,
        *,
        project_id: str,
        import_id: str,
        analysis_version: str,
        prompt_version: str,
        model: str,
        config: Mapping[str, object],
        window_ids: Sequence[str],
    ) -> AnalysisRun:
        with self._session.no_autoflush:
            return self._get_or_create_without_autoflush(
                project_id=project_id,
                import_id=import_id,
                analysis_version=analysis_version,
                prompt_version=prompt_version,
                model=model,
                config=config,
                window_ids=window_ids,
            )

    def _get_or_create_without_autoflush(
        self,
        *,
        project_id: str,
        import_id: str,
        analysis_version: str,
        prompt_version: str,
        model: str,
        config: Mapping[str, object],
        window_ids: Sequence[str],
    ) -> AnalysisRun:
        imported = self._session.get(ImportSource, import_id)
        if imported is None or imported.project_id != project_id:
            raise AnalysisImportMismatchError("导入批次不属于指定项目")

        config_snapshot = normalize_config(config)
        fingerprint = config_fingerprint(config_snapshot)
        manifest_fingerprint = window_manifest_fingerprint(window_ids)
        now = datetime.now(UTC)
        connection = self._session.connection()
        with self._session.no_autoflush, connection.begin_nested():
            self._session.execute(
                sqlite_insert(AnalysisRun)
                .values(
                    id=str(uuid4()),
                    project_id=project_id,
                    import_id=import_id,
                    analysis_version=analysis_version,
                    prompt_version=prompt_version,
                    model=model,
                    config=config_snapshot,
                    config_fingerprint=fingerprint,
                    window_ids=list(window_ids),
                    window_manifest_fingerprint=manifest_fingerprint,
                    status="queued",
                    total_windows=len(window_ids),
                    completed_windows=0,
                    checkpoint=0,
                    error_category=None,
                    error_message=None,
                    created_at=now,
                    updated_at=now,
                    completed_at=None,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        "import_id",
                        "analysis_version",
                        "prompt_version",
                        "model",
                        "config_fingerprint",
                    ]
                )
            )
        run = self._session.scalar(
            select(AnalysisRun)
            .where(
                AnalysisRun.import_id == import_id,
                AnalysisRun.analysis_version == analysis_version,
                AnalysisRun.prompt_version == prompt_version,
                AnalysisRun.model == model,
                AnalysisRun.config_fingerprint == fingerprint,
            )
            .execution_options(populate_existing=True)
        )
        if run is None:
            raise AnalysisRunConcurrencyError("幂等创建后未找到分析运行")
        self._verify_config(run, config_snapshot)
        self._verify_window_manifest(run, window_ids)
        return run

    @staticmethod
    def _is_database_locked(error: OperationalError) -> bool:
        message = str(error.orig).lower()
        return "database is locked" in message or "database table is locked" in message

    @staticmethod
    def _verify_config(
        run: AnalysisRun,
        expected_snapshot: Mapping[str, object],
    ) -> None:
        ensure_config_integrity(run.config, run.config_fingerprint)
        if run.config != expected_snapshot:
            raise AnalysisConfigIntegrityError("配置快照与配置指纹不一致")

    @staticmethod
    def _verify_window_manifest(
        run: AnalysisRun,
        expected_window_ids: Sequence[str],
    ) -> None:
        ensure_window_manifest_integrity(
            run.window_ids,
            run.window_manifest_fingerprint,
            run.total_windows,
        )
        if run.window_ids != list(expected_window_ids):
            raise AnalysisWindowManifestIntegrityError("窗口清单与已有运行不一致")

    def get(self, run_id: str) -> AnalysisRun:
        run = self._session.get(AnalysisRun, run_id)
        if run is None:
            raise AnalysisRunNotFoundError(run_id)
        return run

    def acquire_lease(
        self,
        run_id: str,
        *,
        owner: str,
        duration: timedelta,
        now: datetime | None = None,
    ) -> AnalysisRunLease:
        if not owner.strip():
            raise ValueError("lease owner 不能为空")
        if duration <= timedelta(0):
            raise ValueError("lease duration 必须大于零")
        acquired_at = now or datetime.now(UTC)
        expires_at = acquired_at + duration
        token = str(uuid4())
        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(AnalysisRun)
                .where(
                    AnalysisRun.id == run_id,
                    AnalysisRun.status.not_in({"succeeded", "cancelled"}),
                    (
                        (AnalysisRun.lease_token.is_(None))
                        | (AnalysisRun.lease_expires_at <= acquired_at)
                    ),
                )
                .values(
                    lease_owner=owner,
                    lease_token=token,
                    lease_expires_at=expires_at,
                    updated_at=acquired_at,
                )
                .execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            run = self.get(run_id)
            self._session.refresh(run)
            if (
                run.lease_token is not None
                and run.lease_expires_at is not None
                and _as_utc(run.lease_expires_at) > _as_utc(acquired_at)
            ):
                raise AnalysisRunAlreadyRunningError("analysis_run_already_running")
            raise InvalidAnalysisRunTransitionError(f"{run.status} 状态不能获取 lease")
        run = self.get(run_id)
        self._session.refresh(run)
        return AnalysisRunLease(owner=owner, token=token, expires_at=expires_at)

    def heartbeat_lease(
        self,
        run_id: str,
        *,
        token: str,
        owner: str | None = None,
        duration: timedelta,
        now: datetime | None = None,
    ) -> AnalysisRunLease:
        if duration <= timedelta(0):
            raise ValueError("lease duration 必须大于零")
        heartbeat_at = now or datetime.now(UTC)
        expires_at = heartbeat_at + duration
        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(AnalysisRun)
                .where(
                    AnalysisRun.id == run_id,
                    AnalysisRun.lease_token == token,
                    AnalysisRun.lease_expires_at > heartbeat_at,
                    *((AnalysisRun.lease_owner == owner,) if owner is not None else ()),
                )
                .values(lease_expires_at=expires_at, updated_at=heartbeat_at)
                .execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            raise AnalysisRunLeaseLostError("analysis_run_lease_lost")
        run = self.get(run_id)
        self._session.refresh(run)
        return AnalysisRunLease(
            owner=run.lease_owner or "",
            token=token,
            expires_at=expires_at,
        )

    def release_lease(
        self,
        run_id: str,
        *,
        token: str,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> None:
        released_at = now or datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(AnalysisRun)
                .where(
                    AnalysisRun.id == run_id,
                    AnalysisRun.lease_token == token,
                    AnalysisRun.lease_expires_at > released_at,
                    *((AnalysisRun.lease_owner == owner,) if owner is not None else ()),
                )
                .values(
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
                .execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            raise AnalysisRunLeaseLostError("analysis_run_lease_lost")

    def assert_lease(
        self,
        run_id: str,
        *,
        token: str,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> AnalysisRun:
        checked_at = now or datetime.now(UTC)
        run = self._session.scalar(
            select(AnalysisRun)
            .where(
                AnalysisRun.id == run_id,
                AnalysisRun.lease_token == token,
                AnalysisRun.lease_expires_at > checked_at,
                *((AnalysisRun.lease_owner == owner,) if owner is not None else ()),
            )
            .execution_options(populate_existing=True)
        )
        if run is None:
            raise AnalysisRunLeaseLostError("analysis_run_lease_lost")
        return run

    def start(self, run_id: str) -> AnalysisRun:
        return self._transition(run_id, "running", {"queued"})

    def resume(self, run_id: str) -> AnalysisRun:
        return self._transition(
            run_id,
            "running",
            {"interrupted", "failed"},
            validate_manifest=True,
        )

    def interrupt(
        self,
        run_id: str,
        error_category: str,
        error_message: str,
        *,
        lease_token: str | None = None,
        lease_owner: str | None = None,
    ) -> AnalysisRun:
        return self._transition(
            run_id,
            "interrupted",
            {"running"},
            error_category=error_category,
            error_message=error_message,
            lease_token=lease_token,
            lease_owner=lease_owner,
        )

    def fail(
        self,
        run_id: str,
        error_category: str,
        error_message: str,
        *,
        lease_token: str | None = None,
        lease_owner: str | None = None,
        now: datetime | None = None,
    ) -> AnalysisRun:
        return self._transition(
            run_id,
            "failed",
            {"running"},
            error_category=error_category,
            error_message=error_message,
            lease_token=lease_token,
            lease_owner=lease_owner,
            now=now,
        )

    def cancel(self, run_id: str) -> AnalysisRun:
        return self._transition(
            run_id,
            "cancelled",
            {"queued", "running", "interrupted", "failed"},
        )

    def succeed(
        self,
        run_id: str,
        *,
        lease_token: str | None = None,
        lease_owner: str | None = None,
        now: datetime | None = None,
    ) -> AnalysisRun:
        return self._transition(
            run_id,
            "succeeded",
            {"running"},
            require_complete=True,
            validate_manifest=True,
            lease_token=lease_token,
            lease_owner=lease_owner,
            now=now,
        )

    def record_window_results(
        self,
        run_id: str,
        *,
        window_index: int,
        window_id: str,
        candidates: Sequence[EventCandidateUpsert],
        lease_token: str | None = None,
        lease_owner: str | None = None,
        now: datetime | None = None,
    ) -> list[EventCandidate]:
        run = self.get(run_id)
        self._session.refresh(run)
        checked_at = now or datetime.now(UTC)
        self.assert_lease_if_required(
            run,
            lease_token=lease_token,
            lease_owner=lease_owner,
            now=checked_at,
        )
        self._ensure_run_manifest(run)
        if run.status != "running":
            raise InvalidAnalysisRunTransitionError(f"{run.status} 状态不能写入窗口结果")
        if not window_id:
            raise ValueError("window_id 不能为空")
        if not 0 <= window_index < run.total_windows:
            raise InvalidAnalysisCheckpointError("window_index 超出运行窗口范围")
        if window_id != run.window_ids[window_index]:
            raise InvalidAnalysisCheckpointError("window_id 与窗口清单不匹配")
        if window_index > run.completed_windows:
            raise InvalidAnalysisCheckpointError("不能跳过尚未完成的窗口")
        should_advance = window_index == run.completed_windows
        if _uses_lane_slots(run.analysis_version):
            _, lane = resolve_lane_slot_id(window_id)
            self._validate_candidate_lanes(candidates, expected_lane=lane)

        keys = [candidate.candidate_key for candidate in candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("同一窗口内 candidate_key 不能重复")

        written_at = checked_at
        with self._session.begin_nested():
            if candidates:
                rows = [
                    {
                        "id": str(uuid4()),
                        "run_id": run_id,
                        "window_id": window_id,
                        "created_at": written_at,
                        "updated_at": written_at,
                        **candidate.model_dump(),
                    }
                    for candidate in candidates
                ]
                candidate_insert = sqlite_insert(EventCandidate).values(rows)
                self._session.execute(
                    candidate_insert.on_conflict_do_update(
                        index_elements=["run_id", "window_id", "candidate_key"],
                        set_={
                            "raw_payload": candidate_insert.excluded.raw_payload,
                            "review_payload": candidate_insert.excluded.review_payload,
                            "status": candidate_insert.excluded.status,
                            "rejection_reason": (candidate_insert.excluded.rejection_reason),
                            "scores": candidate_insert.excluded.scores,
                            "updated_at": written_at,
                        },
                    )
                )

            if should_advance:
                next_window_index = window_index + 1
                progress_result = cast(
                    CursorResult[Any],
                    self._session.execute(
                        update(AnalysisRun)
                        .where(
                            AnalysisRun.id == run_id,
                            AnalysisRun.status == "running",
                            AnalysisRun.completed_windows == window_index,
                            AnalysisRun.checkpoint == window_index,
                            *(
                                (
                                    AnalysisRun.lease_token == lease_token,
                                    AnalysisRun.lease_expires_at > written_at,
                                    *(
                                        (AnalysisRun.lease_owner == lease_owner,)
                                        if lease_owner is not None
                                        else ()
                                    ),
                                )
                                if lease_token is not None
                                else ()
                            ),
                        )
                        .values(
                            completed_windows=next_window_index,
                            checkpoint=next_window_index,
                            updated_at=written_at,
                        )
                        .execution_options(synchronize_session=False)
                    ),
                )
                if progress_result.rowcount != 1:
                    current = self._session.scalar(
                        select(AnalysisRun)
                        .where(AnalysisRun.id == run_id)
                        .execution_options(populate_existing=True)
                    )
                    if current is None:
                        raise AnalysisRunNotFoundError(run_id)
                    if lease_token is not None and (
                        current.lease_token != lease_token
                        or current.lease_expires_at is None
                        or _as_utc(current.lease_expires_at) <= _as_utc(written_at)
                        or (lease_owner is not None and current.lease_owner != lease_owner)
                    ):
                        raise AnalysisRunLeaseLostError("analysis_run_lease_lost")
                    if current.status != "running":
                        raise AnalysisRunConcurrencyError("并发操作已改变分析运行状态")
                    if not (
                        current.completed_windows == current.checkpoint
                        and current.completed_windows > window_index
                    ):
                        raise AnalysisRunConcurrencyError("分析窗口进度发生并发冲突")

        self._session.refresh(run)
        if not keys:
            return []
        loaded = self._session.scalars(
            select(EventCandidate)
            .where(
                EventCandidate.run_id == run_id,
                EventCandidate.window_id == window_id,
                EventCandidate.candidate_key.in_(keys),
            )
            .execution_options(populate_existing=True)
        )
        by_key = {candidate.candidate_key: candidate for candidate in loaded}
        return [by_key[key] for key in keys]

    @staticmethod
    def _validate_candidate_lanes(
        candidates: Sequence[EventCandidateUpsert],
        *,
        expected_lane: EventLane,
    ) -> None:
        """写入前校验 V3 候选及复核载荷均属于当前槽位。"""

        for candidate in candidates:
            expected_prefix = f"candidate-{expected_lane}-"
            if not candidate.candidate_key.startswith(expected_prefix):
                raise InvalidAnalysisCheckpointError("候选 candidate_key lane 与窗口槽位不一致")
            if candidate.raw_payload.get("lane") != expected_lane:
                raise InvalidAnalysisCheckpointError("候选 raw_payload lane 与窗口槽位不一致")
            payload_key = candidate.raw_payload.get("candidate_key")
            try:
                validated = V3EventCandidate.model_validate(candidate.raw_payload)
            except (TypeError, ValueError):
                raise InvalidAnalysisCheckpointError(
                    "候选 raw_payload 无法通过 V3 结构校验"
                ) from None
            stable_key = stable_v3_candidate_key(validated)
            if payload_key != stable_key or candidate.candidate_key != stable_key:
                raise InvalidAnalysisCheckpointError("候选 candidate_key 与完整规范化载荷不一致")
            if candidate.review_payload is not None:
                try:
                    V3CandidateReview.model_validate(candidate.review_payload)
                except (TypeError, ValueError):
                    raise InvalidAnalysisCheckpointError(
                        "候选 review_payload 无法通过 V3 结构校验"
                    ) from None

    def unfinished_window_indexes(self, run_id: str) -> list[int]:
        run = self.get(run_id)
        self._session.refresh(run)
        self._ensure_run_manifest(run)
        return list(range(run.checkpoint, run.total_windows))

    def candidates(self, run_id: str) -> list[EventCandidate]:
        self.get(run_id)
        return list(
            self._session.scalars(
                select(EventCandidate)
                .where(EventCandidate.run_id == run_id)
                .order_by(
                    EventCandidate.window_id.asc(),
                    EventCandidate.candidate_key.asc(),
                )
            )
        )

    def _transition(
        self,
        run_id: str,
        target: str,
        allowed_sources: set[str],
        *,
        error_category: str | None = None,
        error_message: str | None = None,
        require_complete: bool = False,
        validate_manifest: bool = False,
        lease_token: str | None = None,
        lease_owner: str | None = None,
        now: datetime | None = None,
    ) -> AnalysisRun:
        run = self.get(run_id)
        if validate_manifest:
            self._session.refresh(run)
            self._ensure_run_manifest(run)
        observed_status = run.status
        if observed_status not in allowed_sources or target not in TRANSITIONS.get(
            observed_status, set()
        ):
            raise InvalidAnalysisRunTransitionError(f"{observed_status} 不能转换为 {target}")

        transitioned_at = now or datetime.now(UTC)
        statement = update(AnalysisRun).where(
            AnalysisRun.id == run_id,
            AnalysisRun.status == observed_status,
        )
        if require_complete:
            statement = statement.where(AnalysisRun.completed_windows == AnalysisRun.total_windows)
        if validate_manifest:
            statement = statement.where(
                AnalysisRun.window_manifest_fingerprint == run.window_manifest_fingerprint,
                AnalysisRun.total_windows == run.total_windows,
                AnalysisRun.completed_windows == run.completed_windows,
                AnalysisRun.checkpoint == run.checkpoint,
            )
        if lease_token is not None:
            statement = statement.where(
                AnalysisRun.lease_token == lease_token,
                AnalysisRun.lease_expires_at > transitioned_at,
                *((AnalysisRun.lease_owner == lease_owner,) if lease_owner is not None else ()),
            )
        result = cast(
            CursorResult[Any],
            self._session.execute(
                statement.values(
                    status=target,
                    error_category=error_category,
                    error_message=error_message,
                    updated_at=transitioned_at,
                    completed_at=(
                        transitioned_at if target in {"succeeded", "failed", "cancelled"} else None
                    ),
                    lease_owner=(
                        None
                        if target in {"succeeded", "failed", "cancelled", "interrupted"}
                        else AnalysisRun.lease_owner
                    ),
                    lease_token=(
                        None
                        if target in {"succeeded", "failed", "cancelled", "interrupted"}
                        else AnalysisRun.lease_token
                    ),
                    lease_expires_at=(
                        None
                        if target in {"succeeded", "failed", "cancelled", "interrupted"}
                        else AnalysisRun.lease_expires_at
                    ),
                ).execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            current = self._session.scalar(
                select(AnalysisRun)
                .where(AnalysisRun.id == run_id)
                .execution_options(populate_existing=True)
            )
            if current is None:
                raise AnalysisRunNotFoundError(run_id)
            if validate_manifest:
                self._ensure_run_manifest(current)
            if current.status != observed_status:
                raise AnalysisRunConcurrencyError(
                    f"运行状态已从 {observed_status} 并发变为 {current.status}"
                )
            if require_complete and current.completed_windows != current.total_windows:
                raise InvalidAnalysisRunTransitionError("仍有窗口未完成，不能标记成功")
            if lease_token is not None:
                raise AnalysisRunLeaseLostError("analysis_run_lease_lost")
            raise AnalysisRunConcurrencyError("运行状态并发更新失败")

        self._session.refresh(run)
        return run

    @staticmethod
    def _ensure_run_manifest(run: AnalysisRun) -> None:
        """验证窗口清单指纹、总数与 checkpoint 单调不变量。"""

        try:
            ensure_window_manifest_integrity(
                run.window_ids,
                run.window_manifest_fingerprint,
                run.total_windows,
            )
        except AnalysisWindowManifestIntegrityError:
            raise AnalysisRunManifestError("分析运行窗口清单完整性校验失败") from None
        if not (0 <= run.checkpoint == run.completed_windows <= run.total_windows):
            raise AnalysisRunManifestError("分析运行 checkpoint 与窗口清单不一致")

    def assert_lease_if_required(
        self,
        run: AnalysisRun,
        *,
        lease_token: str | None,
        lease_owner: str | None = None,
        now: datetime,
    ) -> None:
        if lease_token is not None:
            self.assert_lease(
                run.id,
                token=lease_token,
                owner=lease_owner,
                now=now,
            )
        elif run.lease_token is not None:
            raise AnalysisRunLeaseLostError("analysis_run_lease_required")


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
