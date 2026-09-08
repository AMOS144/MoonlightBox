import json
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.imports.analysis import analyze_import
from moonlightbox.imports.analysis_job import (
    V3_ANALYSIS_JOB_KIND,
    V3_RELATIONSHIP_PROMPT_VERSION,
    V3_SHARED_EXPERIENCE_PROMPT_VERSION,
    create_event_analysis_v3_handler,
)
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.worker import Worker
from sqlalchemy.orm import Session


class FakeNarrativeSummarizer:
    model_path = "local-test"

    def summarize(self, candidate: object, messages: object) -> str:
        del candidate, messages
        return "那次停止联系让两个人的关系突然停了下来，也留下了需要面对的隔阂。"


def test_import_confirm_worker_publishes_safe_idempotent_v3_events(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'e2e.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
        node_analysis_enabled=True,
        node_analysis_api_key="test-key",
        node_analysis_endpoint="https://node-analysis.invalid/v1/chat/completions",
    )
    requests: list[dict[str, Any]] = []

    def handle_node_analysis(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        response_name = body["response_format"]["json_schema"]["name"]
        user_payload = json.loads(body["messages"][-1]["content"])
        if response_name == "RawV3EventCandidateBatch":
            message_ids = [message["id"] for message in user_payload["消息"]]
            lane = user_payload["元数据"]["lane"]
            payload = {
                "candidates": (
                    [
                        {
                            "candidate_key": None,
                            "lane": "relationship",
                            "type": "conflict",
                            "title": "停止联系引发关系冲突",
                            "event_status": "occurred",
                            "start_message_id": "csv:2",
                            "end_message_id": "csv:3",
                            "summary": "双方谈话后决定暂停联系",
                            "before_state": "愿意沟通",
                            "after_state": "停止联系",
                            "emotion_labels": ["失望"],
                            "topic": "联系边界",
                            "conflict_level": 4,
                            "event_significance": 0.98,
                            "relationship_impact": 0.95,
                            "model_confidence": 0.97,
                            "reason": "双方明确停止联系并在后续持续受影响",
                            "evidence_ids": ["csv:2", "csv:3"],
                        }
                    ]
                    if lane == "relationship" and {"csv:2", "csv:3"}.issubset(message_ids)
                    else []
                )
            }
        elif response_name == "V3GlobalSelectionResult":
            candidate = user_payload["候选"][0]
            payload = {
                "selected": [
                    {
                        "candidate_key": candidate["candidate_key"],
                        "relative_importance": 0.96,
                        "reason": "全局比较后该冲突显著改变关系轨迹",
                    }
                ]
            }
        else:
            message_ids = [message["id"] for message in user_payload["消息"]]
            payload = {
                "facts_supported": True,
                "occurrence_supported": True,
                "bilateral_confirmation": True,
                "evidence_alignment": 0.95,
                "persistence": 0.95,
                "type_support": 0.95,
                "relationship_impact": 0.95,
                "event_significance": 0.98,
                "model_confidence": 0.97,
                "evidence_ids": message_ids[:2],
                "reason": "后续消息仍在处理停止联系造成的关系变化",
            }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"parsed": payload}}]},
        )

    database = Database(settings.database_url)
    transport_client = httpx.Client(transport=httpx.MockTransport(handle_node_analysis))
    node_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint=settings.node_analysis_endpoint,
        model=settings.node_analysis_model,
        api_key=settings.node_analysis_api_key,
        client=transport_client,
    )
    registry = JobRegistry()
    registry.register(
        V3_ANALYSIS_JOB_KIND,
        create_event_analysis_v3_handler(
            settings,
            cloud_client=node_client,
            narrative_summarizer=FakeNarrativeSummarizer(),  # type: ignore[arg-type]
        ),
    )
    worker = Worker(database, registry)

    with TestClient(create_app(settings)) as client:
        project = client.post("/api/projects", json={"name": "端到端"}).json()
        content = (
            "时间,发送者,类型,内容\n"
            "2026-01-01 20:00:00,甲,文本,我们需要认真谈谈\n"
            "2026-01-01 20:01:00,乙,文本,那就先不要联系了\n"
            "2026-01-02 20:00:00,甲,文本,我仍在想昨天停止联系的事\n"
            "2026-01-02 20:01:00,乙,文本,<msg><appmsg>wxid_secret</appmsg></msg>\n"
        ).encode()
        preview = client.post(
            f"/api/projects/{project['id']}/imports/preview",
            files={"file": ("chat.csv", content, "text/csv")},
        ).json()
        confirmed = client.post(
            f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        )
        assert confirmed.status_code == 201
        confirmation = confirmed.json()
        job_id = confirmation["analysis_job_id"]
        queued_job = client.get(f"/api/jobs/{job_id}").json()
        assert queued_job["kind"] == V3_ANALYSIS_JOB_KIND
        assert queued_job["status"] == "queued"

        # 仅构造迁移前已有 heuristic-v1 自动节点，核心流程仍由 API 入队和 Worker 执行。
        with Session(database.engine) as session:
            migration_legacy_result = analyze_import(
                session,
                project_id=project["id"],
                import_id=confirmation["import_id"],
            )
            assert migration_legacy_result.event_count == 1

        assert worker.run_once() is True
        completed_job = client.get(f"/api/jobs/{job_id}").json()
        with Session(database.engine) as session:
            analysis_run = session.query(AnalysisRun).filter_by(project_id=project["id"]).one()
        assert completed_job["status"] == "succeeded", (
            completed_job["error_code"],
            completed_job["error_message"],
            completed_job["checkpoint"],
            analysis_run.error_category,
            analysis_run.error_message,
        )
        events = client.get(f"/api/projects/{project['id']}/events").json()

        assert len(events) <= 25
        legacy_event = next(event for event in events if event["status"] == "superseded")
        v3_events = [
            event
            for event in events
            if event["analysis_version"] == "hybrid-v3" and event["status"] == "active"
        ]
        assert legacy_event["status"] == "superseded"
        assert len(v3_events) == 1
        v3_event = v3_events[0]
        assert set(v3_event["score_components"]) == {
            "event_significance",
            "relationship_impact",
            "evidence_quality",
            "persistence",
            "type_support",
            "model_confidence",
        }
        assert v3_event["lane"] == "relationship"
        assert v3_event["event_status"] == "occurred"
        assert v3_event["title"] == "停止联系引发关系冲突"
        assert v3_event["source_lanes"] == ["relationship"]
        assert v3_event["evidence_summaries"]
        assert V3_RELATIONSHIP_PROMPT_VERSION in v3_event["prompt_version"]
        assert V3_SHARED_EXPERIENCE_PROMPT_VERSION in v3_event["prompt_version"]
        serialized = json.dumps(v3_event, ensure_ascii=False)
        assert "<msg" not in serialized
        assert "appmsg" not in serialized
        assert "wxid_secret" not in serialized

        request_count = len(requests)
        assert worker.run_once() is False
        replay = client.post(
            f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        )
        assert replay.status_code == 200
        assert replay.json()["analysis_job_id"] == job_id
        assert worker.run_once() is False
        replayed_events = client.get(f"/api/projects/{project['id']}/events").json()
        assert replayed_events == events
        assert len(requests) == request_count

        rejected = client.patch(
            f"/api/projects/{project['id']}/events/{v3_event['id']}",
            json={
                "changes": {"status": "rejected"},
                "reason": "端到端人工标记误报",
            },
        )
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "rejected"

    with Session(database.engine) as session:
        run = session.query(AnalysisRun).filter_by(project_id=project["id"]).one()
        assert run.status == "succeeded"
        assert session.query(EventNode).filter_by(project_id=project["id"]).count() == 2
        legacy_revisions = (
            session.query(AnalysisRevision)
            .filter_by(event_id=legacy_event["id"])
            .order_by(AnalysisRevision.revision_number)
            .all()
        )
        assert [revision.analysis_version for revision in legacy_revisions] == [
            "heuristic-v1",
            "hybrid-v3",
        ]
    transport_client.close()
    database.close()
