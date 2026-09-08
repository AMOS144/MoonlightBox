from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.branches.continuity_models import IdentityKernel
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class FakeGenerator:
    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        del model_version_id, system_prompt, messages
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(type="text", content="好"),),
            raw_output="{}",
        )


def test_memory_api_migrates_reads_and_rolls_back(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'api.db'}"
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=database_url,
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings, generator=FakeGenerator())) as client:
        project = client.post("/api/projects", json={"name": "记忆 API"}).json()
        database = Database(database_url)
        with Session(database.engine) as session:
            source = ImportSource(
                project_id=project["id"],
                preview_id="preview-1",
                source_path="/tmp/chat.txt",
                message_count=1,
                confirmed_at=datetime.now(UTC),
            )
            target = Participant(
                project_id=project["id"],
                name="她",
                role="target",
            )
            session.add_all([source, target])
            session.flush()
            session.add(
                Message(
                    project_id=project["id"],
                    import_id=source.id,
                    participant_id=target.id,
                    source_id="m1",
                    timestamp=datetime.now(UTC),
                    kind="text",
                    content="好呀",
                    raw={},
                )
            )
            session.commit()
        event = client.post(
            f"/api/projects/{project['id']}/events",
            json={
                "type": "relationship",
                "start_message_id": "m1",
                    "end_message_id": "m1",
                "before_state": "普通",
                "after_state": "愿意交流",
                "emotion_labels": [],
                "topic": "交流",
                "conflict_level": 0,
                "importance": 0.8,
                "reason": "起点",
                "evidence_ids": ["m1"],
            },
        ).json()
        model = client.post(
            f"/api/projects/{project['id']}/models",
            json={
                "base_model": "qwen",
                "adapter_path": "/tmp/adapter",
                "dataset_hash": "hash",
                "metrics": {},
            },
        ).json()
        with Session(database.engine) as session:
            version = session.get(ModelVersion, model["id"])
            assert version is not None
            version.active = True
            session.add(
                IdentityKernel(
                    project_id=project["id"],
                    model_version_id=model["id"],
                    schema_version="subject-persona-v2",
                    content={"persona": "她"},
                    evidence_message_ids=["m1"],
                    field_evidence={"values": ["m1"]},
                    field_confidence={"values": 0.8},
                    acceptance_report_id="accepted-report",
                    content_hash="accepted-kernel",
                    locked_at=datetime.now(UTC),
                )
            )
            session.commit()
        branch = client.post(
            f"/api/projects/{project['id']}/branches",
            json={
                "origin_event_id": event["id"],
                "model_version_id": model["id"],
                "title": "测试分支",
                "origin_time": datetime.now(UTC).isoformat(),
            },
        ).json()
        database.close()

        migrated = client.post(
            f"/api/projects/{project['id']}/branches/{branch['id']}/memory/migrate"
        )
        overview = client.get(
            f"/api/projects/{project['id']}/branches/{branch['id']}/memory"
        )
        versions = client.get(
            f"/api/projects/{project['id']}/branches/{branch['id']}/memory/versions"
        )
        version_id = versions.json()[0]["id"]
        rollback = client.post(
            f"/api/projects/{project['id']}/branches/{branch['id']}"
            f"/memory/versions/{version_id}/rollback"
        )
        wrong_project = client.get(
            f"/api/projects/not-this-project/branches/{branch['id']}/memory"
        )

    assert migrated.status_code == 200
    assert overview.status_code == 200
    assert overview.json()["identity_kernel"]["content"]["persona"] == "她"
    assert overview.json()["current_state"]["relationship_state"] == {}
    assert overview.json()["growth_health"] == {
        "status": "insufficient",
        "processed_episode_count": 0,
        "observation_span_days": 0.0,
        "state_version_count": 1,
        "approved_memory_count": 0,
        "approved_reflection_count": 0,
        "successful_cognitive_cycle_count": 0,
        "failed_cognitive_cycle_count": 0,
        "evidence_coverage_rate": 1.0,
        "duplicate_lineage_count": 0,
        "unmet_requirements": [
            "需要至少 20 个已处理互动 episode",
            "需要覆盖至少 14 天真实互动",
            "需要至少 5 个可追溯状态版本",
            "需要至少 5 条已批准长期记忆",
            "需要至少 1 条跨 episode 阶段反思",
            "需要至少 5 次成功的主体认知周期",
        ],
    }
    assert rollback.status_code == 200
    assert rollback.json()["version"] == 2
    assert wrong_project.status_code == 404
