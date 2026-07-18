from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings


class FakeGenerator:
    def generate(
        self,
        _model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        assert "2026-02-01" in system_prompt
        return f"模拟回复：{messages[-1]['content']}"


def test_branch_conversation_is_created_generated_and_replayed(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'branches.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings, generator=FakeGenerator())) as client:
        project = client.post("/api/projects", json={"name": "时间分支"}).json()
        event = client.post(
            f"/api/projects/{project['id']}/events",
            json={
                "type": "reconciliation",
                "start_message_id": "m1",
                "end_message_id": "m2",
                "before_state": "疏远",
                "after_state": "朋友",
                "emotion_labels": ["期待"],
                "topic": "重新联系",
                "conflict_level": 1,
                "importance": 0.8,
                "reason": "恢复交流",
                "evidence_ids": ["m1", "m2"],
            },
        ).json()
        model = client.post(
            f"/api/projects/{project['id']}/models",
            json={
                "base_model": "qwen",
                "adapter_path": "/models/test",
                "dataset_hash": "hash",
                "metrics": {},
            },
        ).json()
        created = client.post(
            f"/api/projects/{project['id']}/branches",
            json={
                "origin_event_id": event["id"],
                "model_version_id": model["id"],
                "title": "如果那天我道歉了",
                "origin_time": datetime(2026, 2, 1).isoformat(),
                "state_snapshot": {"relationship_status": "朋友"},
            },
        )
        assert created.status_code == 201
        branch_id = created.json()["id"]

        generated = client.post(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages",
            json={"content": "对不起"},
        )
        replayed = client.get(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages"
        )

    assert generated.status_code == 201
    assert [message["role"] for message in replayed.json()] == ["user", "assistant"]
    assert replayed.json()[1]["content"] == "模拟回复：对不起"
