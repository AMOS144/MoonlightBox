from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event as ThreadEvent
from time import sleep
from typing import Any

import pytest
from moonlightbox.db import Database
from moonlightbox.events.models import (
    AnalysisRevision,
    AnalysisRun,
    EventCandidate,
    EventNode,
)
from moonlightbox.events.reviewer import TwoStageEventReviewer
from moonlightbox.events.schemas import ReviewedEvent
from moonlightbox.events.service import EventService
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from sqlalchemy import func, select
from sqlalchemy.orm import Session


class FakeCloudClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create_structured_completion(
        self,
        *,
        response_model: type[Any],
        **metadata: object,
    ) -> Any:
        self.calls.append(metadata)
        if not self.responses:
            raise AssertionError("发生了计划外的云端调用")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response)


class BlockingCloudClient:
    def __init__(self, entered: ThreadEvent, release: ThreadEvent) -> None:
        self.entered = entered
        self.release = release

    def create_structured_completion(
        self,
        *,
        response_model: type[Any],
        **_: object,
    ) -> Any:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("测试等待首个 worker 超时")
        return response_model.model_validate({"candidates": []})


class SlowCloudClient:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    def create_structured_completion(
        self,
        *,
        response_model: type[Any],
        **_: object,
    ) -> Any:
        sleep(self.delay_seconds)
        return response_model.model_validate({"candidates": []})


def _session_with_messages(
    tmp_path: Path,
    *,
    contents: list[str] | None = None,
) -> tuple[Database, Session]:
    database = Database(f"sqlite:///{tmp_path / 'pipeline.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    session = Session(database.engine)
    session.add(Project(id="project-1", name="V2 流水线测试"))
    session.flush()
    participant = Participant(id="participant-1", project_id="project-1", name="甲", role="self")
    session.add(participant)
    session.add(
        ImportSource(
            id="import-1",
            project_id="project-1",
            preview_id="preview-1",
            source_path="/tmp/chat.json",
            message_count=3,
            confirmed_at=datetime.now(UTC),
        )
    )
    session.flush()
    values = contents or ["我们需要谈谈边界", "这让我很生气", "之后仍然没有解决"]
    for index, content in enumerate(values, start=1):
        session.add(
            Message(
                id=f"row-{index}",
                project_id="project-1",
                import_id="import-1",
                participant_id=participant.id,
                source_id=f"m{index}",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
                kind="text",
                content=content,
                raw={},
            )
        )
    session.commit()
    return database, session


def _candidate(
    *,
    key: str = "ignored",
    start_message_id: str = "m1",
    evidence_ids: list[str] | None = None,
    topic: str = "边界冲突",
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "type": "conflict",
        "start_message_id": start_message_id,
        "end_message_id": "m2",
        "before_state": "正常交流",
        "after_state": "关系紧张",
        "emotion_labels": ["生气"],
        "topic": topic,
        "conflict_level": 4,
        "state_change_strength": 0.9,
        "model_confidence": 0.9,
        "reason": "冲突持续",
        "evidence_ids": evidence_ids or ["m1", "m2"],
    }


def _accepted_review() -> dict[str, object]:
    return {
        "accepted": True,
        "type_supported": True,
        "evidence_alignment": 0.9,
        "decisive_event": False,
        "persistence": 0.9,
        "evidence_ids": ["m3"],
        "reason": "后续仍未解决",
    }


def test_pipeline_runs_fake_cloud_end_to_end_and_publishes(tmp_path: Path) -> None:
    from moonlightbox.events.pipeline import EventV2Pipeline, PipelineConfig

    database, session = _session_with_messages(tmp_path)
    client = FakeCloudClient(
        [
            {"candidates": [_candidate()]},
            _accepted_review(),
        ]
    )
    checkpoints: list[dict[str, object]] = []
    try:
        result = EventV2Pipeline(
            session,
            TwoStageEventReviewer(client),
            config=PipelineConfig(character_budget=1000),
            progress_callback=checkpoints.append,
        ).run(
            project_id="project-1",
            import_id="import-1",
            prompt_version="prompt-v2",
            model="cloud-v2",
        )

        run = session.get(AnalysisRun, result.run_id)
        node = session.scalar(select(EventNode))
        stored_candidate = session.scalar(select(EventCandidate))
        revision = session.scalar(select(AnalysisRevision))
        assert result.event_count == 1
        assert run is not None and run.status == "succeeded"
        assert run.checkpoint == 1
        assert node is not None and node.type == "conflict"
        assert stored_candidate is not None
        assert revision is not None and revision.candidate_id == stored_candidate.id
        assert node.importance == pytest.approx(0.92)
        assert len(client.calls) == 2
        stages = [str(checkpoint["stage"]) for checkpoint in checkpoints]
        expected_stages = [
            "loading_messages",
            "extracting_candidates",
            "reviewing_persistence",
            "ranking",
            "publishing",
        ]
        assert [stage for stage in stages if stage in expected_stages] == expected_stages
    finally:
        session.close()
        database.close()


def test_pipeline_cancellation_interrupts_run_before_candidate_review(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineCancelledError,
        PipelineConfig,
    )

    database, session = _session_with_messages(tmp_path)
    cancelled = ThreadEvent()
    client = FakeCloudClient([{"candidates": [_candidate()]}])
    original_call = client.create_structured_completion

    def cancel_after_extraction(**kwargs: object) -> Any:
        result = original_call(**kwargs)
        cancelled.set()
        return result

    client.create_structured_completion = cancel_after_extraction  # type: ignore[method-assign]
    try:
        with pytest.raises(PipelineCancelledError):
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(client),
                config=PipelineConfig(character_budget=1000),
                should_cancel=cancelled.is_set,
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

        run = session.scalar(select(AnalysisRun))
        assert run is not None
        assert run.status == "interrupted"
        assert run.checkpoint == 0
        assert run.lease_token is None
        assert len(client.calls) == 1
    finally:
        session.close()
        database.close()


def test_pipeline_rejects_one_invalid_candidate_and_continues(tmp_path: Path) -> None:
    from moonlightbox.events.pipeline import EventV2Pipeline, PipelineConfig

    database, session = _session_with_messages(tmp_path)
    client = FakeCloudClient(
        [
            {
                "candidates": [
                    _candidate(),
                    _candidate(
                        key="bad",
                        start_message_id="unknown",
                        evidence_ids=["unknown", "m2"],
                    ),
                ]
            },
            _accepted_review(),
            _accepted_review(),
        ]
    )
    try:
        result = EventV2Pipeline(
            session,
            TwoStageEventReviewer(client),
            config=PipelineConfig(character_budget=1000),
        ).run(
            project_id="project-1",
            import_id="import-1",
            prompt_version="prompt-v2",
            model="cloud-v2",
        )

        stored = list(session.scalars(select(EventCandidate).order_by(EventCandidate.status)))
        assert result.event_count == 1
        assert {candidate.status for candidate in stored} == {
            "accepted",
            "rejected",
        }
        rejected = next(item for item in stored if item.status == "rejected")
        assert "unknown_message_id" in (rejected.rejection_reason or "")
    finally:
        session.close()
        database.close()


def test_window_failure_marks_run_failed_without_publishing(tmp_path: Path) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, session = _session_with_messages(tmp_path)
    old = EventService(session).create(
        "project-1",
        ReviewedEvent(
            type="long_pause",
            start_message_id="m1",
            end_message_id="m2",
            before_state="中断",
            after_state="恢复",
            emotion_labels=["间隔"],
            topic="旧节点",
            conflict_level=1,
            importance=0.8,
            reason="旧启发式节点",
            evidence_ids=["m1", "m2"],
        ),
        analysis_version="heuristic-v1",
        prompt_version="none",
    )
    try:
        with pytest.raises(PipelineExecutionError, match="分析执行失败"):
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(FakeCloudClient([RuntimeError("云端不可用")])),
                config=PipelineConfig(character_budget=1000),
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

        run = session.scalar(select(AnalysisRun))
        session.refresh(old)
        assert run is not None and run.status == "failed"
        assert run.checkpoint == 0
        assert old.status == "active"
        assert session.scalar(select(func.count()).select_from(EventNode)) == 1
    finally:
        session.close()
        database.close()


def test_later_window_failure_preserves_checkpoint_and_active_nodes(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, session = _session_with_messages(tmp_path, contents=["甲", "乙", "丙"])
    old = EventService(session).create(
        "project-1",
        ReviewedEvent(
            type="long_pause",
            start_message_id="m1",
            end_message_id="m2",
            before_state="中断",
            after_state="恢复",
            emotion_labels=["间隔"],
            topic="必须保持不变的旧节点",
            conflict_level=1,
            importance=0.8,
            reason="旧启发式节点",
            evidence_ids=["m1", "m2"],
        ),
        analysis_version="heuristic-v1",
        prompt_version="none",
    )
    old_snapshot = {
        column.name: getattr(old, column.name) for column in EventNode.__table__.columns
    }
    old_revision_ids = [revision.id for revision in EventService(session).revisions(old.id)]
    try:
        with pytest.raises(PipelineExecutionError, match="分析执行失败"):
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(
                    FakeCloudClient(
                        [
                            {"candidates": []},
                            RuntimeError("第二窗口失败"),
                        ]
                    )
                ),
                config=PipelineConfig(character_budget=2),
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

        run = session.scalar(select(AnalysisRun))
        session.expire_all()
        unchanged = session.get(EventNode, old.id)
        assert run is not None and run.status == "failed"
        assert run.checkpoint == 1
        assert unchanged is not None
        assert {
            column.name: getattr(unchanged, column.name) for column in EventNode.__table__.columns
        } == old_snapshot
        assert [
            revision.id for revision in EventService(session).revisions(old.id)
        ] == old_revision_ids
        assert session.scalar(select(func.count()).select_from(EventNode)) == 1
    finally:
        session.close()
        database.close()


def test_unknown_failure_is_redacted_in_database_and_public_error(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, session = _session_with_messages(tmp_path)
    secret = "api-key=sk-secret prompt=私密提示 response=完整响应正文"
    try:
        with pytest.raises(PipelineExecutionError) as captured:
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(FakeCloudClient([RuntimeError(secret)])),
                config=PipelineConfig(character_budget=1000),
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

        run = session.scalar(select(AnalysisRun))
        assert run is not None
        assert run.error_category == "RuntimeError"
        assert run.error_message == "分析执行失败"
        assert str(captured.value) == "分析执行失败"
        assert secret not in f"{run.error_category}{run.error_message}{captured.value}"
    finally:
        session.close()
        database.close()


def test_cloud_failure_persists_only_stable_code_and_diagnostic(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.cloud_client import (
        NodeAnalysisCloudError,
        NodeAnalysisCloudErrorCode,
    )
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, session = _session_with_messages(tmp_path)
    cloud_error = NodeAnalysisCloudError(
        NodeAnalysisCloudErrorCode.AUTHENTICATION,
        attempts=1,
        context={"prompt": "私密提示", "api_key": "sk-secret"},
        diagnostic={"http_status": 401},
    )
    try:
        with pytest.raises(PipelineExecutionError) as captured:
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(FakeCloudClient([cloud_error])),
                config=PipelineConfig(character_budget=1000),
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

        run = session.scalar(select(AnalysisRun))
        persisted = f"{run.error_category}{run.error_message}"  # type: ignore[union-attr]
        assert run is not None
        assert run.error_category == "cloud:authentication"
        assert run.error_message == ('{"code":"authentication","diagnostic":{"http_status":401}}')
        assert str(captured.value) == "节点分析云端服务失败"
        assert "私密提示" not in persisted
        assert "sk-secret" not in persisted
    finally:
        session.close()
        database.close()


def test_resume_skips_completed_windows(tmp_path: Path) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, session = _session_with_messages(tmp_path, contents=["甲", "乙", "丙"])
    config = PipelineConfig(character_budget=2)
    failing_client = FakeCloudClient(
        [
            {"candidates": []},
            RuntimeError("第二窗口失败"),
        ]
    )
    try:
        with pytest.raises(PipelineExecutionError):
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(failing_client),
                config=config,
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )
        run = session.scalar(select(AnalysisRun))
        assert run is not None and run.checkpoint == 1

        resumed_client = FakeCloudClient([{"candidates": []}])
        result = EventV2Pipeline(
            session,
            TwoStageEventReviewer(resumed_client),
            config=config,
        ).run(
            project_id="project-1",
            import_id="import-1",
            prompt_version="prompt-v2",
            model="cloud-v2",
        )

        assert result.event_count == 0
        assert len(resumed_client.calls) == 1
        assert session.get(AnalysisRun, run.id).status == "succeeded"  # type: ignore[union-attr]
    finally:
        session.close()
        database.close()


def test_pipeline_rejects_second_worker_before_cloud_processing(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.pipeline import EventV2Pipeline, PipelineConfig
    from moonlightbox.events.runs import AnalysisRunAlreadyRunningError

    database, setup = _session_with_messages(tmp_path)
    setup.close()
    entered = ThreadEvent()
    release = ThreadEvent()

    def run_first_worker() -> None:
        with Session(database.engine) as session:
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(BlockingCloudClient(entered, release)),
                config=PipelineConfig(character_budget=1000),
                worker_id="worker-first",
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_first_worker)
            assert entered.wait(timeout=5)
            second_client = FakeCloudClient([])
            with Session(database.engine) as second:
                with pytest.raises(
                    AnalysisRunAlreadyRunningError,
                    match="already_running",
                ):
                    EventV2Pipeline(
                        second,
                        TwoStageEventReviewer(second_client),
                        config=PipelineConfig(character_budget=1000),
                        worker_id="worker-second",
                    ).run(
                        project_id="project-1",
                        import_id="import-1",
                        prompt_version="prompt-v2",
                        model="cloud-v2",
                    )
            assert second_client.calls == []
            release.set()
            future.result(timeout=5)
    finally:
        release.set()
        database.close()


def test_slow_window_cannot_renew_expired_uncontested_lease(tmp_path: Path) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, session = _session_with_messages(tmp_path)
    try:
        with pytest.raises(PipelineExecutionError):
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(SlowCloudClient(0.05)),
                config=PipelineConfig(character_budget=1000),
                worker_id="slow-worker",
                lease_duration=timedelta(milliseconds=10),
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

        run = session.scalar(select(AnalysisRun))
        assert run is not None and run.status == "running"
        assert run.checkpoint == 0
    finally:
        session.close()
        database.close()


def test_reclaimed_slow_window_rejects_old_token_and_new_worker_continues(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.pipeline import (
        EventV2Pipeline,
        PipelineConfig,
        PipelineExecutionError,
    )

    database, setup = _session_with_messages(tmp_path)
    setup.close()
    entered = ThreadEvent()
    release = ThreadEvent()

    def run_slow_worker() -> None:
        with Session(database.engine) as session:
            EventV2Pipeline(
                session,
                TwoStageEventReviewer(BlockingCloudClient(entered, release)),
                config=PipelineConfig(character_budget=1000),
                worker_id="worker-old",
                lease_duration=timedelta(milliseconds=20),
            ).run(
                project_id="project-1",
                import_id="import-1",
                prompt_version="prompt-v2",
                model="cloud-v2",
            )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            stale_future = executor.submit(run_slow_worker)
            assert entered.wait(timeout=5)
            sleep(0.06)
            with Session(database.engine) as replacement:
                result = EventV2Pipeline(
                    replacement,
                    TwoStageEventReviewer(FakeCloudClient([{"candidates": []}])),
                    config=PipelineConfig(character_budget=1000),
                    worker_id="worker-new",
                    lease_duration=timedelta(minutes=1),
                ).run(
                    project_id="project-1",
                    import_id="import-1",
                    prompt_version="prompt-v2",
                    model="cloud-v2",
                )
                assert result.event_count == 0
            release.set()
            with pytest.raises(PipelineExecutionError, match="分析执行失败"):
                stale_future.result(timeout=5)

        with Session(database.engine) as verification:
            run = verification.scalar(select(AnalysisRun))
            assert run is not None and run.status == "succeeded"
            assert run.checkpoint == 1
            assert verification.scalar(select(func.count()).select_from(EventCandidate)) == 0
    finally:
        release.set()
        database.close()


def test_persisted_candidate_ranking_builds_persistence_context_once_per_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.pipeline as pipeline_module

    database, session = _session_with_messages(tmp_path)
    context_calls = 0
    original = pipeline_module.get_persistence_context

    def counted(*args: object, **kwargs: object):
        nonlocal context_calls
        context_calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline_module, "get_persistence_context", counted)
    candidates = [
        _candidate(key=f"candidate-{index}", topic=f"边界冲突-{index}") for index in range(100)
    ]
    client = FakeCloudClient(
        [{"candidates": candidates}] + [_accepted_review() for _ in candidates]
    )
    try:
        pipeline_module.EventV2Pipeline(
            session,
            TwoStageEventReviewer(client),
            config=pipeline_module.PipelineConfig(character_budget=1000),
        ).run(
            project_id="project-1",
            import_id="import-1",
            prompt_version="prompt-v2",
            model="cloud-v2",
        )

        # 窗口处理构造一次，持久候选重建最多再构造一次。
        assert context_calls <= 2
    finally:
        session.close()
        database.close()
