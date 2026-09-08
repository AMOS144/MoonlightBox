from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.worker import Worker
from moonlightbox.world.client import (
    LightRAGMetadata,
    LightRAGReference,
    LightRAGRetrieval,
)
from moonlightbox.world.jobs import WORLD_BUILD_JOB_KIND, create_world_build_handler
from moonlightbox.world.schemas import WorldProfileDraft


class FakeLightRAG:
    def __init__(self) -> None:
        self.documents: list[object] = []

    def index_documents(self, workspace: str, documents: list[object]) -> LightRAGMetadata:
        assert workspace.startswith("world_")
        self.documents.extend(documents)
        return LightRAGMetadata(
            lightrag_version="1.5.6",
            embedding_model="embedding-test",
            embedding_dimension=3,
            extraction_model="extract-test",
            chunking_strategy="fixed_token",
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            entity_prompt_version="person-world-v1",
        )

    def query(self, workspace: str, query: str, **_: Any) -> LightRAGRetrieval:
        document = self.documents[0]
        return LightRAGRetrieval(
            context=f"{query}\n小月说自己在新公司上班。",
            references=[LightRAGReference(file_path=document.source)],  # type: ignore[attr-defined]
            metadata=self.index_documents_metadata,
        )

    @property
    def index_documents_metadata(self) -> LightRAGMetadata:
        return LightRAGMetadata(
            lightrag_version="1.5.6",
            embedding_model="embedding-test",
            embedding_dimension=3,
            extraction_model="extract-test",
            chunking_strategy="fixed_token",
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            entity_prompt_version="person-world-v1",
        )


class FakeCompiler:
    def create_structured_completion(self, **_: Any) -> WorldProfileDraft:
        source = self.source
        statement = {
            "text": "在新公司工作",
            "source_status": "direct",
            "source_document_ids": [source],
        }
        return WorldProfileDraft.model_validate(
            {
                "identity": {
                    "names": [
                        {
                            "text": "小月",
                            "source_status": "direct",
                            "source_document_ids": [source],
                        }
                    ],
                    "aliases": [],
                    "self_descriptions": [],
                    "roles": [statement],
                },
                "work_and_education": [statement],
                "places": [],
                "social_relationships": [],
                "preferences": [],
                "recurring_activities": [],
                "routine_summary": {
                    "workdays": [],
                    "weekends": [],
                    "other_patterns": [],
                },
                "life_phases": [],
                "relationship_with_user": {"overview": [], "changes_over_time": []},
                "important_events": [],
                "unresolved_candidates": [],
            }
        )

    source: str = ""


def test_import_builds_world_before_enqueuing_node_analysis(tmp_path: Path) -> None:
    settings = Settings.model_construct(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
        lightrag_enabled=True,
    )
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/projects", json={"name": "人物世界"}).json()
        preview = client.post(
            f"/api/projects/{project['id']}/imports/preview",
            files={
                "file": (
                    "chat.csv",
                    (
                        "时间,发送者,类型,内容\n"
                        "2026-01-01 20:00:00,我,文本,新工作怎么样\n"
                        "2026-01-01 20:01:00,小月,文本,刚去新公司好多事\n"
                    ).encode(),
                    "text/csv",
                )
            },
        ).json()
        confirmed = client.post(
            f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
            json={"self_participant": "我", "target_participant": "小月"},
        ).json()

        assert confirmed["world_job_id"]
        assert confirmed["analysis_job_id"] is None
        world_job_id = confirmed["world_job_id"]
        replayed = client.post(
            f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
            json={"self_participant": "我", "target_participant": "小月"},
        ).json()
        assert replayed["world_job_id"] == world_job_id

        database = Database(settings.database_url)
        fake_lightrag = FakeLightRAG()
        fake_compiler = FakeCompiler()
        registry = JobRegistry()
        registry.register(
            WORLD_BUILD_JOB_KIND,
            create_world_build_handler(
                settings,
                lightrag_client=fake_lightrag,  # type: ignore[arg-type]
                compiler_client=fake_compiler,
            ),
        )

        # Set the source name after bundles are indexed but before compilation.
        original_query = fake_lightrag.query

        def query_and_capture(workspace: str, query: str, **kwargs: Any) -> LightRAGRetrieval:
            fake_compiler.source = fake_lightrag.documents[0].source  # type: ignore[attr-defined]
            return original_query(workspace, query, **kwargs)

        fake_lightrag.query = query_and_capture  # type: ignore[method-assign]
        assert Worker(database, registry, allowed_kinds={WORLD_BUILD_JOB_KIND}).run_once()

        world_job = client.get(f"/api/jobs/{world_job_id}").json()
        assert world_job["status"] == "succeeded"
        analysis_job_id = world_job["checkpoint"]["analysis_job_id"]
        analysis_job = client.get(f"/api/jobs/{analysis_job_id}").json()
        assert analysis_job["kind"] == "event_analysis_v3"
        assert analysis_job["status"] == "queued"

        profile = client.get(f"/api/projects/{project['id']}/world-profile")
        assert profile.status_code == 200
        body = profile.json()
        assert body["identity"]["names"][0]["text"] == "小月"
        assert body["work_and_education"][0]["text"] == "在新公司工作"
        assert body["graph"]["lightrag_version"] == "1.5.6"
        assert body["source_message_ids"]

        # 前端在编译期间通过世界状态接口读取 Worker checkpoint，而不是
        # 根据本地计时模拟百分比。
        session_iterator = database.session()
        session = next(session_iterator)
        try:
            session.add(
                Job(
                    kind=WORLD_BUILD_JOB_KIND,
                    payload={"project_id": project["id"]},
                    status="running",
                    progress=0.75,
                    checkpoint={
                        "stage": "compiling_profile",
                        "progress": 0.75,
                        "completed_questions": 4,
                        "question_count": 8,
                    },
                    worker_token="test-worker-token",
                    lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
                )
            )
            session.commit()
        finally:
            session_iterator.close()
        graph_status = client.get(
            f"/api/projects/{project['id']}/world-profile/status"
        )
        assert graph_status.status_code == 200
        build_progress = graph_status.json()["build_progress"]
        assert build_progress["job_id"]
        assert {
            key: value for key, value in build_progress.items() if key != "job_id"
        } == {
            "status": "running",
            "stage": "compiling_profile",
            "progress": 0.75,
            "completed_questions": 4,
            "question_count": 8,
            "indexed_bundles": None,
            "bundle_count": None,
        }

        source_name = fake_lightrag.documents[0].source  # type: ignore[attr-defined]
        document_id = source_name.removesuffix(".txt")
        source = client.get(
            f"/api/projects/{project['id']}/world-profile/sources/{document_id}"
        ).json()
        assert [message["id"] for message in source["messages"]] == body["source_message_ids"]
        assert all("message:" not in document.text for document in fake_lightrag.documents)  # type: ignore[attr-defined]
        database.close()
