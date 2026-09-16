from threading import Barrier, Thread
from typing import Any

from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.analysis_job import (
    V3_ANALYSIS_JOB_KIND,
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


def test_confirm_import_does_not_enqueue_retired_analysis(client: TestClient) -> None:
    project = client.post("/api/projects", json={"name": "导入"}).json()
    preview_id = _preview_import(client, project["id"])
    confirmed = client.post(
        f"/api/projects/{project['id']}/imports/{preview_id}/confirm",
        json={"self_participant": "乙", "target_participant": "甲"},
    )
    assert confirmed.status_code == 201
    assert confirmed.json()["analysis_job_id"] is None
    jobs = client.get("/api/jobs", params={"project_id": project["id"]}).json()
    assert all(j["kind"] not in {"event_analysis_v2", V3_ANALYSIS_JOB_KIND} for j in jobs)


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
        build_analysis_job_snapshot(_settings(node_analysis_thinking_mode="disabled")),
        build_analysis_job_snapshot(_settings(node_analysis_max_output_tokens=4096)),
    ]

    baseline_key = analysis_dedupe_key("import-1", baseline)

    assert all(analysis_dedupe_key("import-1", variant) != baseline_key for variant in variants)


def test_repeated_confirm_does_not_create_legacy_job(
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
    assert first["analysis_job_id"] is None


def test_concurrent_confirm_does_not_create_retired_v3_jobs(
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
        assert session.query(Job).filter_by(kind=V3_ANALYSIS_JOB_KIND).count() == 0
    database.close()


def test_repeated_confirm_does_not_resume_retired_job(
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
        old = service.enqueue(
            V3_ANALYSIS_JOB_KIND, {"project_id": project["id"], "import_id": first["import_id"]}
        )
        old_id = old.id
        running = service.start(old_id)
        assert running.worker_token is not None
        service.fail(
            old_id,
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
    assert first["analysis_job_id"] is None
    retired = client.get(f"/api/jobs/{old_id}").json()
    assert retired["status"] == "failed"
    assert retired["error_code"] == "test_failure"
