import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.imports.analysis_job import (
    V3_ANALYSIS_JOB_KIND,
    V3AnalysisJobSnapshot,
    build_v3_analysis_job_snapshot,
    create_event_analysis_v3_handler,
    v3_analysis_dedupe_key,
)
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.jobs.service import JobService
from moonlightbox.worker import Worker
from pydantic import BaseModel
from sqlalchemy.orm import Session


class FakeV3CloudClient:
    def __init__(self) -> None:
        self.operations: list[str] = []

    def create_structured_completion(
        self,
        *,
        user_content: str,
        response_model: type[BaseModel],
        operation_id: str | None = None,
        **_: Any,
    ) -> BaseModel:
        self.operations.append(operation_id or "")
        payload = json.loads(user_content)
        if response_model.__name__ == "RawV3EventCandidateBatch":
            if payload["元数据"]["lane"] != "shared_experience":
                return response_model.model_validate({"candidates": []})
            message_ids = [message["id"] for message in payload["消息"]]
            if len(message_ids) < 3:
                return response_model.model_validate({"candidates": []})
            return response_model.model_validate(
                {
                    "candidates": [
                        {
                            "candidate_key": None,
                            "lane": "shared_experience",
                            "type": "travel",
                            "title": "共同旅行",
                            "event_status": "occurred",
                            "start_message_id": message_ids[0],
                            "end_message_id": message_ids[1],
                            "summary": "双方完成了一次共同旅行",
                            "before_state": None,
                            "after_state": None,
                            "emotion_labels": ["开心"],
                            "topic": "旅行",
                            "conflict_level": 0,
                            "event_significance": 0.9,
                            "relationship_impact": 0.8,
                            "model_confidence": 0.9,
                            "reason": "包含到达与事后体验",
                            "evidence_ids": message_ids[:2],
                        }
                    ]
                }
            )
        if response_model.__name__ == "V3GlobalSelectionResult":
            candidate = payload["候选"][0]
            return response_model.model_validate(
                {
                    "selected": [
                        {
                            "candidate_key": candidate["candidate_key"],
                            "relative_importance": 0.88,
                            "reason": "全局比较后仍具有持续叙事价值",
                        }
                    ]
                }
            )
        message_ids = [message["id"] for message in payload["消息"]]
        return response_model.model_validate(
            {
                "facts_supported": True,
                "occurrence_supported": True,
                "bilateral_confirmation": True,
                "evidence_alignment": 0.95,
                "persistence": 0.4,
                "type_support": 0.95,
                "relationship_impact": 0.8,
                "event_significance": 0.9,
                "model_confidence": 0.9,
                "evidence_ids": message_ids[-1:],
                "reason": "事后消息证明旅行已经发生",
            }
        )


class FakeNarrativeSummarizer:
    model_path = "local-test"

    def summarize(self, candidate: object, messages: object) -> str:
        del candidate, messages
        return "那次共同旅行让双方开始期待下一次一起出发。"


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path,
        "database_url": f"sqlite:///{tmp_path / 'v3-analysis-job.db'}",
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
        project_id = client.post(
            "/api/projects",
            json={"name": "V3 Worker"},
        ).json()["id"]
        content = (
            "时间,发送者,类型,内容\n"
            "2026-05-01 10:00:00,甲,文本,我们到酒店了\n"
            "2026-05-01 10:01:00,乙,文本,这次旅行很开心\n"
            "2026-05-01 10:02:00,甲,文本,下次还一起去\n"
        ).encode()
        preview_id = client.post(
            f"/api/projects/{project_id}/imports/preview",
            files={"file": ("chat.csv", content, "text/csv")},
        ).json()["id"]
        confirmed = client.post(
            f"/api/projects/{project_id}/imports/{preview_id}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        ).json()
    assert confirmed["analysis_job_id"] is None
    # 只验证历史 Worker 兼容，显式构造历史任务；不恢复导入自动评分路径。
    from moonlightbox.imports.analysis_job import config_fingerprint

    database = Database(settings.database_url)
    with Session(database.engine) as session:
        snapshot = build_v3_analysis_job_snapshot(settings)
        job = JobService(session).enqueue_unique(
            V3_ANALYSIS_JOB_KIND,
            {
                "project_id": project_id,
                "import_id": confirmed["import_id"],
                "analysis_config": snapshot,
                "config_fingerprint": config_fingerprint(snapshot),
            },
            dedupe_key=v3_analysis_dedupe_key(confirmed["import_id"], snapshot),
        )
        job_id = job.id
    database.close()
    return project_id, confirmed["import_id"], job_id


def test_v3_snapshot_and_dedupe_include_both_prompts_and_weights(
    tmp_path: Path,
) -> None:
    snapshot = build_v3_analysis_job_snapshot(_settings(tmp_path))
    validated = V3AnalysisJobSnapshot.model_validate(snapshot)
    changed = json.loads(json.dumps(snapshot))
    changed["pipeline"]["relationship_prompt_version"] = "changed"

    assert validated.analysis_version == "hybrid-v3"
    assert validated.pipeline.acceptance_threshold == 0.55
    assert (
        validated.pipeline.global_selection_prompt_version == "event-analysis-v3-global-selection-1"
    )
    assert validated.pipeline.weights["evidence_quality"] == 0.25
    assert v3_analysis_dedupe_key(
        "import-1",
        snapshot,
    ) != v3_analysis_dedupe_key("import-1", changed)
    assert "api_key" not in json.dumps(snapshot).lower()


def test_worker_runs_v3_dual_channel_pipeline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    project_id, import_id, job_id = _enqueue_import(settings)
    fake = FakeV3CloudClient()
    database = Database(settings.database_url)
    registry = JobRegistry()
    registry.register(
        V3_ANALYSIS_JOB_KIND,
        create_event_analysis_v3_handler(
            settings,
            cloud_client=fake,
            narrative_summarizer=FakeNarrativeSummarizer(),  # type: ignore[arg-type]
        ),
    )

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        run = session.query(AnalysisRun).filter_by(import_id=import_id).one()
        event = session.query(EventNode).filter_by(project_id=project_id).one()
        revision = session.query(AnalysisRevision).filter_by(event_id=event.id).one()
        assert job.kind == V3_ANALYSIS_JOB_KIND
        assert job.status == "succeeded"
        assert run.analysis_version == "hybrid-v3"
        assert run.total_windows == 2
        assert event.lane == "shared_experience"
        assert event.type == "travel"
        assert revision.analysis_version == "hybrid-v3"
        assert fake.operations == [
            "v3_relationship_extraction",
            "v3_shared_experience_extraction",
            "v3_shared_experience_review",
            "v3_global_selection",
        ]
    database.close()
