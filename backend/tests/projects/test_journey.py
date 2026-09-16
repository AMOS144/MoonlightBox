"""流程投影冒烟：Job 成功不能吞掉待办，已有分支不受新调查影响。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.node_investigation.models import NodeInvestigation
from moonlightbox.projects.journey import project_journey
from moonlightbox.projects.models import Project
from moonlightbox.projects.router import create_projects_router
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from datetime import UTC, datetime
from moonlightbox.runtime_v1.branch_models import Branch, BranchMessage
from moonlightbox.events.models import EventNode
from moonlightbox.training.models import ModelVersion


def test_world_backoff_keeps_journey_polling(tmp_path):
    from moonlightbox.jobs.runtime_recovery import WORLD_KIND
    from moonlightbox.jobs.service import JobService

    db = Database(f"sqlite:///{tmp_path / 'waiting.db'}")
    db.create_schema()
    with Session(db.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.commit()
        service = JobService(session)
        job = service.start(service.enqueue(WORLD_KIND, {"project_id": "p"}).id)
        service.fail(job.id, "lightrag_timeout", "超时", token=job.worker_token)
        result = project_journey(session, "p")
        assert result["processing"] is True
        assert result["world_build"]["recovery"]["status"] == "waiting"
        service.cancel(job.id)
        result = project_journey(session, "p")
        assert result["processing"] is False
        assert result["world_build"]["status"] == "cancelled"
    db.close()


def test_projection_is_read_only_and_preserves_user_work(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'journey.db'}")
    db.create_schema()
    with Session(db.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.commit()
        first = project_journey(session, "p")
        assert first["next_action"]["action"] == "import"
        assert not first["publication"]
        session.add(
            NodeInvestigation(
                id="n",
                project_id="p",
                dataset_version="v1",
                state={
                    "status": "waiting_for_user",
                    "pending_question": {"id": "q", "text": "那几天见面了吗？"},
                },
            )
        )
        session.add(
            Job(kind="node_investigation_turn", payload={"project_id": "p"}, status="succeeded")
        )
        session.commit()
        before = session.scalar(select(func.count()).select_from(Job))
        result = project_journey(session, "p")
        assert result["processing"] is False
        nodes = next(stage for stage in result["stages"] if stage["key"] == "nodes")
        assert nodes["state"] == "processing"
        assert not any(t["object_id"] == "n" for t in result["tasks"])
        assert not result["historical_branch_creation"]["available"]
        assert session.scalar(select(func.count()).select_from(Job)) == before
        assert not session.new and not session.dirty
    db.close()


def test_missing_project_is_not_empty_success(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'http.db'}")
    db.create_schema()
    app = FastAPI()
    settings = Settings.model_construct(data_dir=tmp_path, chroma_dir=tmp_path, model_dir=tmp_path)
    app.include_router(create_projects_router(db, settings))
    with TestClient(app) as client:
        assert client.get("/api/projects/missing/journey").status_code == 404
    db.close()


def test_previews_only_expose_committed_messages_in_project(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'previews.db'}")
    db.create_schema()
    now = datetime(2026, 5, 8, tzinfo=UTC)
    with Session(db.engine) as session:
        for project_id in ("p", "other"):
            session.add(Project(id=project_id, name="项目"))
            session.flush()
            session.add(ModelVersion(id=f"m-{project_id}", project_id=project_id,
                                     base_model="test", adapter_path="none", dataset_hash="h", metrics={}))
            session.add(EventNode(id=f"e-{project_id}", project_id=project_id, type="origin",
                                  start_message_id="1", end_message_id="1", emotion_labels=[],
                                  topic="", conflict_level=0, importance=0, reason="", evidence_ids=[]))
            session.flush()
            session.add(Branch(id=project_id, project_id=project_id, origin_event_id=f"e-{project_id}",
                               model_version_id=f"m-{project_id}", title="聊天", origin_time=now))
            session.flush()
            session.add(BranchMessage(branch_id=project_id, sequence=1, role="target", content="已发送",
                                      observed_at=now, generation_status="completed"))
        session.add(BranchMessage(branch_id="p", sequence=2, role="target", content="未发送草稿",
                                  generation_status="failed"))
        session.commit()
        result = project_journey(session, "p")
        assert len(result["branches"]) == 1
        assert result["branches"][0]["latest_message"]["text"] == "已发送"
        session.add(BranchMessage(branch_id="p", sequence=3, role="target", content="内部资源路径",
                                  type="sticker", generation_status="completed"))
        session.commit()
        assert project_journey(session, "p")["branches"][0]["latest_message"]["text"] == "[表情包]"
        assert not session.new and not session.dirty
    db.close()
