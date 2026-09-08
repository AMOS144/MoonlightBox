from threading import Barrier, Thread
from typing import Any

from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.analysis_job import (
    V3_ANALYSIS_JOB_KIND,
    V3_GLOBAL_SELECTION_PROMPT_VERSION,
    V3_RELATIONSHIP_PROMPT_VERSION,
    V3_SHARED_EXPERIENCE_PROMPT_VERSION,
    analysis_dedupe_key,
    build_analysis_job_snapshot,
)
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from sqlalchemy.orm import Session


def _settings(**overrides: Any) -> Settings:
    return Settings.model_construct(**overrides)


def _preview_import(client: TestClient, project_id: str) -> str:
    chat = (
        "时间,发送者,类型,内容\n"
        "2026-01-01 20:00:00,甲,文本,我们明天见\n"
        "2026-01-01 20:01:00,乙,文本,好呀\n"
        "2026-01-03 20:00:00,甲,文本,算了，不要再联系了\n"
        "2026-01-03 20:01:00,乙,文本,知道了\n"
    ).encode()
    return client.post(
        f"/api/projects/{project_id}/imports/preview",
        files={"file": ("chat.csv", chat, "text/csv")},
    ).json()["id"]


def test_confirm_import_only_enqueues_v3_job(client: TestClient) -> None:
    project = client.post("/api/projects", json={"name": "自动分析"}).json()
    preview_id = _preview_import(client, project["id"])
    confirmed = client.post(
        f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
        json={"self_participant": "乙", "target_participant": "甲"},
    )

    assert confirmed.status_code == 201
    job_id = confirmed.json()["analysis_job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["kind"] == V3_ANALYSIS_JOB_KIND
    assert job["status"] == "queued"
    assert job["progress"] == 0.0
    snapshot = job["payload"]["analysis_config"]
    assert snapshot["analysis_version"] == "hybrid-v3"
    assert snapshot["model"]
    assert snapshot["endpoint"].startswith("http")
    assert snapshot["enabled"] is False
    assert snapshot["timeout_seconds"] == 60.0
    assert snapshot["max_retries"] == 3
    assert snapshot["backoff_seconds"] == 0.5
    assert snapshot["max_backoff_seconds"] == 60.0
    assert snapshot["max_retry_after_seconds"] == 3600.0
    assert snapshot["response_format"] == "json_schema"
    assert snapshot["thinking_mode"] == "disabled"
    assert snapshot["max_output_tokens"] == 8192
    assert snapshot["pipeline"] == {
        "session_gap_seconds": 21600.0,
        "character_budget": 12000,
        "overlap_messages": 8,
        "persistence_session_limit": 3,
        "acceptance_threshold": 0.55,
        "maximum_nodes": 25,
        "relationship_prompt_version": V3_RELATIONSHIP_PROMPT_VERSION,
        "shared_experience_prompt_version": (V3_SHARED_EXPERIENCE_PROMPT_VERSION),
            "global_selection_prompt_version": V3_GLOBAL_SELECTION_PROMPT_VERSION,
        "weights": {
            "event_significance": 0.25,
            "relationship_impact": 0.2,
            "evidence_quality": 0.25,
            "persistence": 0.1,
            "type_support": 0.1,
            "model_confidence": 0.1,
        },
    }
    assert job["payload"]["config_fingerprint"]
    assert "api_key" not in str(job["payload"]).lower()
    assert "secretstr" not in str(job["payload"]).lower()


def test_dedupe_key_changes_when_snapshot_model_changes() -> None:
    first = _settings(node_analysis_model="model-a")
    second = _settings(node_analysis_model="model-b")

    first_snapshot = build_analysis_job_snapshot(first)
    second_snapshot = build_analysis_job_snapshot(second)

    assert analysis_dedupe_key("import-1", first_snapshot) != analysis_dedupe_key(
        "import-1",
        second_snapshot,
    )


def test_dedupe_key_changes_when_prompt_version_changes() -> None:
    current_snapshot = build_analysis_job_snapshot(_settings())
    previous_snapshot = {
        **current_snapshot,
        "prompt_version": "event-analysis-v2.1",
    }

    assert analysis_dedupe_key(
        "import-1",
        current_snapshot,
    ) != analysis_dedupe_key("import-1", previous_snapshot)


def test_dedupe_key_changes_when_cloud_execution_snapshot_changes() -> None:
    first = build_analysis_job_snapshot(
        _settings(node_analysis_timeout_seconds=17, node_analysis_max_retries=0)
    )
    second = build_analysis_job_snapshot(
        _settings(node_analysis_timeout_seconds=18, node_analysis_max_retries=0)
    )

    assert analysis_dedupe_key("import-1", first) != analysis_dedupe_key(
        "import-1",
        second,
    )


def test_dedupe_key_changes_when_structured_output_snapshot_changes() -> None:
    baseline = build_analysis_job_snapshot(_settings())
    variants = [
        build_analysis_job_snapshot(_settings(node_analysis_response_format="json_object")),
        build_analysis_job_snapshot(_settings(node_analysis_thinking_mode="default")),
        build_analysis_job_snapshot(_settings(node_analysis_max_output_tokens=4096)),
    ]

    baseline_key = analysis_dedupe_key("import-1", baseline)

    assert all(analysis_dedupe_key("import-1", variant) != baseline_key for variant in variants)


def test_repeated_confirm_returns_same_v2_job(
    client: TestClient,
) -> None:
    project = client.post("/api/projects", json={"name": "幂等分析"}).json()
    preview_id = _preview_import(client, project["id"])
    first = client.post(
        f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
        json={"self_participant": "乙", "target_participant": "甲"},
    ).json()
    second = client.post(
        f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
        json={"self_participant": "乙", "target_participant": "甲"},
    ).json()

    assert second["analysis_job_id"] == first["analysis_job_id"]
    assert client.get(f"/api/jobs/{first['analysis_job_id']}").json()["status"] == "queued"


def test_concurrent_confirm_creates_one_deduplicated_v3_job(
    client: TestClient,
    settings: Settings,
) -> None:
    project = client.post("/api/projects", json={"name": "并发幂等"}).json()
    preview_id = _preview_import(client, project["id"])
    barrier = Barrier(2)
    responses: list[tuple[int, dict[str, object]]] = []

    def confirm() -> None:
        barrier.wait()
        response = client.post(
            f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        )
        responses.append((response.status_code, response.json()))

    threads = [Thread(target=confirm), Thread(target=confirm)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(responses) == 2
    assert {status_code for status_code, _ in responses}.issubset({200, 201})
    job_ids = {str(body["analysis_job_id"]) for _, body in responses}
    assert len(job_ids) == 1
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        assert session.query(Job).filter_by(kind=V3_ANALYSIS_JOB_KIND).count() == 1
    database.close()


def test_repeated_confirm_resumes_failed_v2_job(
    client: TestClient,
    settings: Settings,
) -> None:
    project = client.post("/api/projects", json={"name": "失败恢复"}).json()
    preview_id = _preview_import(client, project["id"])
    first = client.post(
        f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
        json={"self_participant": "乙", "target_participant": "甲"},
    ).json()
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        service = JobService(session)
        running = service.start(first["analysis_job_id"])
        assert running.worker_token is not None
        service.fail(
            first["analysis_job_id"],
            "test_failure",
            "测试失败",
            token=running.worker_token,
        )
    database.close()

    second = client.post(
        f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
        json={"self_participant": "乙", "target_participant": "甲"},
    ).json()

    assert second["analysis_job_id"] == first["analysis_job_id"]
    resumed = client.get(f"/api/jobs/{first['analysis_job_id']}").json()
    assert resumed["status"] == "queued"
    assert resumed["error_code"] is None
