import json
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

    def list_entities(self, _workspace: str) -> list[object]:
        return []

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
    def __init__(self) -> None:
        self.planned_sections: list[str] = []
        self.finalized_sections: list[str] = []

    def create_agent_chat_model(self):
        from langchain_core.messages import AIMessage

        compiler = self

        class NativeModel:
            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages):
                if not any(item.type == "tool" for item in messages):
                    task = json.loads(
                        next(item.content for item in messages if item.type == "human")
                    )
                    compiler.planned_sections.append(task["section"])
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_world",
                                "args": {"question": "小月的生活和工作背景"},
                                "id": "search-1",
                            }
                        ],
                    )
                from moonlightbox.world.person_world.contracts.profile_v3 import (
                    SECTION_RESULT_MODELS,
                )

                task = json.loads(next(item.content for item in messages if item.type == "human"))
                result = compiler.create_structured_completion(
                    response_model=SECTION_RESULT_MODELS[task["section"]],
                    user_content=json.dumps({"task": task}),
                )
                from profile_submission_helpers import section_input
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_section",
                            "id": "submit",
                            "args": {"result": section_input(result.model_dump(mode="json"))},
                        }
                    ],
                )

        return NativeModel()

    def create_structured_completion(self, **kwargs: Any) -> Any:
        response_model = kwargs["response_model"]
        payload = kwargs.get("user_content", "{}")
        if response_model.__name__.endswith("_result_v3"):
            from moonlightbox.world.person_world.contracts.profile_v3 import SECTION_MODELS

            section = json.loads(payload)["task"]["section"]
            self.finalized_sections.append(section)
            values = SECTION_MODELS[section]().model_dump(mode="json")
            values["summary"] = "小月的生活理解"
            result = response_model(
                **values,
                **(
                    {
                        "module_assessment": {
                            "status": "not_identified",
                            "explanation": "此集成测试只验证导入到批准链路",
                        },
                    }
                    if section == "life_context"
                    else {}
                ),
            )
            if section == "identity":
                result.names_and_self_reference.description = "小月"
                result.names_and_self_reference.status = "described"
                result.overview = "小月正在适应新公司的工作。"
            elif section == "life_context":
                result.primary_engagements.description = "小月在新公司工作"
                result.primary_engagements.status = "described"
            return result
        raise AssertionError(f"Unexpected retired output model: {response_model}")


def test_import_builds_world_before_enqueuing_node_analysis(tmp_path: Path) -> None:
    settings = Settings.model_construct(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
        lightrag_enabled=True,
        # 此断言使用共享的顺序记录 Fake；并发行为由 Coordinator 专属测试覆盖。
        person_world_section_concurrency=1,
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
        assert world_job["status"] == "succeeded", world_job
        assert world_job["checkpoint"]["stage"] == "world_ready"
        assert world_job["checkpoint"]["analysis_job_id"]

        profile = client.get(f"/api/projects/{project['id']}/world-profile")
        assert profile.status_code == 404
        candidate = client.get(f"/api/projects/{project['id']}/world-agent/draft")
        assert candidate.status_code == 404
        analysis_job = client.get(
            f"/api/jobs/{world_job['checkpoint']['analysis_job_id']}"
        ).json()
        assert analysis_job["kind"] == "node_investigation_turn"
        assert analysis_job["status"] == "queued"
        assert fake_compiler.planned_sections == []
        assert fake_compiler.finalized_sections == []
        graph = client.get(f"/api/projects/{project['id']}/world-profile/status").json()
        assert graph["status"] == "ready"
        assert graph["lightrag_version"] == "1.5.6"

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
        graph_status = client.get(f"/api/projects/{project['id']}/world-profile/status")
        assert graph_status.status_code == 200
        build_progress = graph_status.json()["build_progress"]
        assert build_progress["job_id"]
        assert {key: value for key, value in build_progress.items() if key != "job_id"} == {
            "status": "running",
            "stage": "compiling_profile",
            "progress": 0.75,
            "completed_questions": 4,
            "question_count": 8,
            "indexed_bundles": None,
            "bundle_count": None,
        }

        assert all("message:" not in document.text for document in fake_lightrag.documents)  # type: ignore[attr-defined]
        database.close()
