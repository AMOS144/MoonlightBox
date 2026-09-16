import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from typing import Any

import httpx
from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.config import config_fingerprint
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.events.runs import AnalysisRunService
from moonlightbox.imports.analysis_job import (
    ANALYSIS_JOB_KIND,
    ANALYSIS_PROMPT_VERSION,
    analysis_dedupe_key,
    build_analysis_job_snapshot,
    create_event_analysis_v2_handler,
)
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.jobs.service import JobService
from moonlightbox.worker import Worker, recover_interrupted_jobs
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified


class FakeCloudClient:
    def create_structured_completion(
        self,
        *,
        user_content: str,
        response_model: type[BaseModel],
        **_: Any,
    ) -> BaseModel:
        payload = json.loads(user_content)
        if response_model.__name__ == "EventCandidateBatch":
            message_ids = [message["id"] for message in payload["消息"]]
            if len(message_ids) < 2:
                return response_model.model_validate({"candidates": []})
            return response_model.model_validate(
                {
                    "candidates": [
                        {
                            "candidate_key": "由本地重算",
                            "type": "conflict",
                            "start_message_id": message_ids[0],
                            "end_message_id": message_ids[1],
                            "before_state": "关系稳定",
                            "after_state": "发生冲突",
                            "emotion_labels": ["失望"],
                            "topic": "联系频率",
                            "conflict_level": 4,
                            "state_change_strength": 0.95,
                            "model_confidence": 0.95,
                            "reason": "关系状态发生变化",
                            "evidence_ids": message_ids[:2],
                        }
                    ]
                }
            )
        persistence_ids = [
            message["id"] for message in payload["同窗口后续消息"] + payload["持续性上下文"]
        ]
        return response_model.model_validate(
            {
                "accepted": True,
                "type_supported": True,
                "evidence_alignment": 0.95,
                "decisive_event": False,
                "persistence": 0.9,
                "evidence_ids": persistence_ids[-1:],
                "reason": "后续消息持续支持该变化",
            }
        )


class BlockingFakeCloudClient(FakeCloudClient):
    def __init__(self, entered: Event, release: Event) -> None:
        self._entered = entered
        self._release = release

    def create_structured_completion(
        self,
        *,
        user_content: str,
        response_model: type[BaseModel],
        **metadata: Any,
    ) -> BaseModel:
        self._entered.set()
        if not self._release.wait(timeout=5):
            raise TimeoutError("等待取消测试释放云端调用超时")
        return super().create_structured_completion(
            user_content=user_content,
            response_model=response_model,
            **metadata,
        )


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path,
        "database_url": f"sqlite:///{tmp_path / 'analysis-job.db'}",
        "chroma_dir": tmp_path / "chroma",
        "model_dir": tmp_path / "models",
        "auto_create_schema": True,
        "node_analysis_enabled": True,
        "node_analysis_api_key": "test-key",
    }
    values.update(overrides)
    return Settings(**values)


def _enqueue_import(settings: Settings) -> tuple[str, str, str]:
    with TestClient(create_app(settings)) as client:
        project_id = client.post("/api/projects", json={"name": "V2 Worker"}).json()["id"]
        content = (
            "时间,发送者,类型,内容\n"
            "2026-01-01 20:00:00,甲,文本,我们需要谈谈\n"
            "2026-01-01 20:01:00,乙,文本,你总是忽略我\n"
            "2026-01-01 20:02:00,甲,文本,那先冷静一下\n"
            "2026-01-02 20:00:00,乙,文本,我还是很难过\n"
        ).encode()
        preview_id = client.post(
            f"/api/projects/{project_id}/imports/preview",
            files={"file": ("chat.csv", content, "text/csv")},
        ).json()["id"]
        confirmed = client.post(
            f"/api/projects/{project_id}/imports/{preview_id}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        ).json()
    import_id = confirmed["import_id"]
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        service = JobService(session)
        assert confirmed["analysis_job_id"] is None  # 正常导入不再启动旧事件评分器。
        snapshot = build_analysis_job_snapshot(settings)
        job = service.enqueue_unique(
            ANALYSIS_JOB_KIND,
            {
                "project_id": project_id,
                "import_id": import_id,
                "analysis_config": snapshot,
                "config_fingerprint": config_fingerprint(snapshot),
            },
            dedupe_key=analysis_dedupe_key(import_id, snapshot),
        )
    database.close()
    return project_id, import_id, job.id


def test_worker_runs_complete_v2_pipeline_with_fake_cloud(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    project_id, import_id, job_id = _enqueue_import(settings)
    database = Database(settings.database_url)
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(settings, cloud_client=FakeCloudClient()),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        run = session.query(AnalysisRun).filter_by(import_id=import_id).one()
        assert job.status == "succeeded", (
            job.error_code,
            job.error_message,
            job.checkpoint,
            run.status,
            run.error_category,
            run.error_message,
        )
        event = session.query(EventNode).filter_by(project_id=project_id).one()
        revision = session.query(AnalysisRevision).filter_by(event_id=event.id).one()
        assert job.progress == 1.0
        assert job.checkpoint == {
            "stage": "completed",
            "run_id": run.id,
            "event_count": 1,
            "progress": 1.0,
        }
        assert run.status == "succeeded"
        assert revision.analysis_version == "hybrid-v2"
    database.close()


def test_worker_uses_enqueued_model_snapshot_and_local_secret_only(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    api_settings = _settings(
        tmp_path,
        node_analysis_model="api-model-a",
        node_analysis_endpoint="https://api.deepseek.com/chat/completions",
        node_analysis_api_key="api-side-secret",
        node_analysis_timeout_seconds=17,
        node_analysis_max_retries=0,
        node_analysis_backoff_seconds=0.25,
        node_analysis_max_backoff_seconds=7,
        node_analysis_max_retry_after_seconds=19,
        node_analysis_response_format="json_object",
        node_analysis_thinking_mode="disabled",
        node_analysis_max_output_tokens=4096,
    )
    project_id, import_id, job_id = _enqueue_import(api_settings)
    worker_settings = _settings(
        tmp_path,
        node_analysis_model="worker-model-b",
        node_analysis_endpoint="https://worker.example.test/v1/chat/completions",
        node_analysis_api_key="worker-local-secret",
        node_analysis_timeout_seconds=91,
        node_analysis_max_retries=5,
        node_analysis_backoff_seconds=2,
        node_analysis_max_backoff_seconds=80,
        node_analysis_max_retry_after_seconds=800,
        node_analysis_response_format="json_schema",
        node_analysis_thinking_mode="default",
        node_analysis_max_output_tokens=16384,
    )
    requested_models: list[str] = []
    authorization_headers: list[str] = []
    requested_urls: list[str] = []
    request_timeouts: list[dict[str, float]] = []
    request_bodies: list[dict[str, Any]] = []

    def handle_cloud_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        authorization_headers.append(request.headers["Authorization"])
        requested_urls.append(str(request.url))
        request_timeouts.append(request.extensions["timeout"])
        request_bodies.append(body)
        user_payload = json.loads(body["messages"][1]["content"].split("\n\n", 1)[0])
        if "消息" in user_payload:
            message_ids = [message["id"] for message in user_payload["消息"]]
            parsed = {"candidates": []}
            if len(message_ids) >= 2:
                parsed["candidates"] = [
                    {
                        "candidate_key": "由本地重算",
                        "type": "conflict",
                        "start_message_id": message_ids[0],
                        "end_message_id": message_ids[1],
                        "before_state": "关系稳定",
                        "after_state": "发生冲突",
                        "emotion_labels": ["失望"],
                        "topic": "联系频率",
                        "conflict_level": 4,
                        "state_change_strength": 0.95,
                        "model_confidence": 0.95,
                        "reason": "关系状态发生变化",
                        "evidence_ids": message_ids[:2],
                    }
                ]
        else:
            persistence_ids = [
                message["id"]
                for message in (user_payload["同窗口后续消息"] + user_payload["持续性上下文"])
            ]
            parsed = {
                "accepted": True,
                "type_supported": True,
                "evidence_alignment": 0.95,
                "decisive_event": False,
                "persistence": 0.9,
                "evidence_ids": persistence_ids[:1],
                "reason": "后续仍在处理该冲突",
            }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"parsed": parsed}}]},
        )

    transport_client = httpx.Client(transport=httpx.MockTransport(handle_cloud_request))
    monkeypatch.setattr(
        "moonlightbox.events.cloud_client.httpx.Client",
        lambda **_: transport_client,
    )
    database = Database(worker_settings.database_url)
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(worker_settings),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        run = session.query(AnalysisRun).filter_by(import_id=import_id).one()
        revision = (
            session.query(AnalysisRevision)
            .join(EventNode, EventNode.id == AnalysisRevision.event_id)
            .filter(EventNode.project_id == project_id)
            .one()
        )
        assert job.status == "succeeded"
        assert requested_models == ["api-model-a"] * 3
        assert request_timeouts == [{"connect": 17, "read": 17, "write": 17, "pool": 17}] * 3
        assert requested_urls == [
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/chat/completions",
        ]
        assert authorization_headers == [
            "Bearer worker-local-secret",
            "Bearer worker-local-secret",
            "Bearer worker-local-secret",
        ]
        assert run.model == "api-model-a"
        assert revision.model == "api-model-a"
        assert run.config == job.payload["analysis_config"]
        assert run.config_fingerprint == job.payload["config_fingerprint"]
        assert job.payload["analysis_config"]["timeout_seconds"] == 17
        assert job.payload["analysis_config"]["max_retries"] == 0
        assert job.payload["analysis_config"]["response_format"] == "json_object"
        assert job.payload["analysis_config"]["thinking_mode"] == "disabled"
        assert job.payload["analysis_config"]["max_output_tokens"] == 4096
        assert all(
            body["response_format"] == {"type": "json_object"}
            and body["thinking"] == {"type": "disabled"}
            and body["max_tokens"] == 4096
            for body in request_bodies
        )
        assert "api-side-secret" not in json.dumps(job.payload)
        assert "worker-local-secret" not in json.dumps(job.payload)
    database.close()


def test_worker_uses_snapshot_zero_retries_instead_of_local_retry_policy(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    api_settings = _settings(
        tmp_path,
        node_analysis_timeout_seconds=17,
        node_analysis_max_retries=0,
    )
    _, _, job_id = _enqueue_import(api_settings)
    worker_settings = _settings(
        tmp_path,
        node_analysis_timeout_seconds=99,
        node_analysis_max_retries=4,
    )
    attempts: list[dict[str, float]] = []

    def fail_server(request: httpx.Request) -> httpx.Response:
        attempts.append(request.extensions["timeout"])
        return httpx.Response(500)

    transport_client = httpx.Client(transport=httpx.MockTransport(fail_server))
    monkeypatch.setattr(
        "moonlightbox.events.cloud_client.httpx.Client",
        lambda **_: transport_client,
    )
    database = Database(worker_settings.database_url)
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(worker_settings),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        failed = JobService(session).get(job_id)
        assert failed.status == "failed"
        assert failed.error_code == "node_analysis_server"
    assert attempts == [{"connect": 17, "read": 17, "write": 17, "pool": 17}]
    database.close()


def test_worker_honors_disabled_snapshot_even_when_locally_enabled(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    api_settings = _settings(tmp_path, node_analysis_enabled=False)
    _, _, job_id = _enqueue_import(api_settings)
    worker_settings = _settings(tmp_path, node_analysis_enabled=True)
    request_count = 0

    def unexpected_request(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200)

    transport_client = httpx.Client(transport=httpx.MockTransport(unexpected_request))
    monkeypatch.setattr(
        "moonlightbox.events.cloud_client.httpx.Client",
        lambda **_: transport_client,
    )
    database = Database(worker_settings.database_url)
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(worker_settings),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        failed = JobService(session).get(job_id)
        assert failed.status == "failed"
        assert failed.error_code == "node_analysis_disabled"
    assert request_count == 0
    database.close()


def test_worker_rejects_tampered_snapshot_fingerprint(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _, _, job_id = _enqueue_import(settings)
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        job.payload["analysis_config"]["model"] = "tampered-model"
        flag_modified(job, "payload")
        session.commit()
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(settings, cloud_client=FakeCloudClient()),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        failed = JobService(session).get(job_id)
        assert failed.status == "failed"
        assert failed.error_code == "invalid_analysis_config"
    database.close()


def test_missing_cloud_key_fails_with_safe_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path, node_analysis_api_key=None)
    _, _, job_id = _enqueue_import(settings)
    database = Database(settings.database_url)
    registry = JobRegistry()
    registry.register(ANALYSIS_JOB_KIND, create_event_analysis_v2_handler(settings))

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        assert job.status == "failed"
        assert job.error_code == "node_analysis_missing_api_key"
        assert job.error_message == "节点分析云端客户端缺少 API key"
        assert "test-key" not in (job.error_message or "")
    database.close()


def test_deepseek_invalid_response_fails_without_publishing_nodes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    settings = _settings(
        tmp_path,
        node_analysis_endpoint="https://api.deepseek.com/chat/completions",
        node_analysis_model="deepseek-v4-flash",
        node_analysis_response_format="json_object",
        node_analysis_thinking_mode="disabled",
        node_analysis_max_output_tokens=4096,
        node_analysis_max_retries=0,
    )
    project_id, import_id, job_id = _enqueue_import(settings)
    transport_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"candidates":['}}]},
            )
        )
    )
    monkeypatch.setattr(
        "moonlightbox.events.cloud_client.httpx.Client",
        lambda **_: transport_client,
    )
    database = Database(settings.database_url)
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(settings),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        run = session.query(AnalysisRun).filter_by(import_id=import_id).one()
        assert job.status == "failed"
        assert job.error_code == "node_analysis_invalid_response"
        assert run.status == "failed"
        assert session.query(EventNode).filter_by(project_id=project_id).count() == 0
    database.close()


def test_running_v2_job_cancel_interrupts_pipeline_and_stays_cancelled(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _, import_id, job_id = _enqueue_import(settings)
    database = Database(settings.database_url)
    entered = Event()
    release = Event()
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(
            settings,
            cloud_client=BlockingFakeCloudClient(entered, release),
        ),
    )
    worker = Worker(database, registry)
    thread = Thread(target=worker.run_once)
    thread.start()
    assert entered.wait(timeout=2)

    with Session(database.engine) as session:
        JobService(session).cancel(job_id)
    # 这个假客户端故意阻塞且不支持取消：请求取消不能冒充执行已退出。
    with Session(database.engine) as session:
        assert JobService(session).get(job_id).status == "cancelling"
        assert session.query(AnalysisRun).filter_by(import_id=import_id).one().status == "running"
    assert thread.is_alive()
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        run = session.query(AnalysisRun).filter_by(import_id=import_id).one()
        assert job.status == "cancelled"
        assert run.status == "interrupted"
        assert run.checkpoint == 0
        assert run.lease_token is None
    database.close()


def test_worker_stop_interrupts_inflight_v2_job(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _, import_id, job_id = _enqueue_import(settings)
    database = Database(settings.database_url)
    entered = Event()
    release = Event()
    stop_event = Event()
    registry = JobRegistry()
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(
            settings,
            cloud_client=BlockingFakeCloudClient(entered, release),
            should_stop=stop_event.is_set,
        ),
    )
    worker = Worker(database, registry, stop_event=stop_event)
    thread = Thread(target=worker.run_once)
    thread.start()
    assert entered.wait(timeout=2)

    stop_event.set()
    interrupted_before_release = False
    deadline = monotonic() + 1
    while monotonic() < deadline:
        with Session(database.engine) as session:
            if JobService(session).get(job_id).status == "interrupted":
                interrupted_before_release = True
                break
        sleep(0.01)
    release.set()
    thread.join(timeout=2)

    assert interrupted_before_release
    assert not thread.is_alive()
    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        run = session.query(AnalysisRun).filter_by(import_id=import_id).one()
        assert job.status == "interrupted"
        assert job.checkpoint is not None
        assert run.status == "interrupted"
        assert run.lease_token is None
    database.close()


def test_startup_recovery_releases_analysis_lease_and_keeps_checkpoint(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    project_id, import_id, job_id = _enqueue_import(settings)
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        job_service = JobService(session)
        running_job = job_service.start(
            job_id,
            now=datetime.now(UTC) - timedelta(minutes=5),
            lease_duration=timedelta(seconds=1),
        )
        assert running_job.worker_token is not None
        job_service.checkpoint(
            job_id,
            {"stage": "windows", "completed_windows": 1, "progress": 0.45},
            token=running_job.worker_token,
        )
        run_service = AnalysisRunService(session)
        run = run_service.get_or_create(
            project_id=project_id,
            import_id=import_id,
            analysis_version="hybrid-v2",
            prompt_version=ANALYSIS_PROMPT_VERSION,
            model=settings.node_analysis_model,
            config={},
            window_ids=["window-1", "window-2"],
        )
        lease = run_service.acquire_lease(
            run.id,
            owner=f"job-{job_id}",
            duration=timedelta(minutes=2),
        )
        run_service.start(run.id)
        other_run = run_service.get_or_create(
            project_id=project_id,
            import_id=import_id,
            analysis_version="hybrid-v2",
            prompt_version=ANALYSIS_PROMPT_VERSION,
            model="other-model",
            config={"variant": "other"},
            window_ids=["window-other"],
        )
        run_service.acquire_lease(
            other_run.id,
            owner="job-unrelated",
            duration=timedelta(minutes=2),
        )
        run_service.start(other_run.id)
        session.commit()
        run_id = run.id
        other_run_id = other_run.id

    assert recover_interrupted_jobs(database) == 1

    with Session(database.engine) as session:
        recovered_job = JobService(session).get(job_id)
        recovered_run = session.get(AnalysisRun, run_id)
        unrelated_run = session.get(AnalysisRun, other_run_id)
        assert recovered_job.status == "queued"
        assert recovered_job.checkpoint == {
            "stage": "windows",
            "completed_windows": 1,
            "progress": 0.45,
        }
        assert recovered_run is not None
        assert recovered_run.status == "interrupted"
        assert recovered_run.lease_token is None
        assert recovered_run.lease_owner is None
        assert recovered_run.lease_expires_at is None
        assert unrelated_run is not None
        assert unrelated_run.status == "running"
        assert unrelated_run.lease_owner == "job-unrelated"
        assert lease.token
    database.close()
