from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings


class FakeGenerator:
    def generate(
        self,
        _model_version_id: str,
        _system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        return f"收到：{messages[-1]['content']}"


def test_import_to_timeline_branch_flow(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'e2e.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings, generator=FakeGenerator())) as client:
        project = client.post("/api/projects", json={"name": "端到端"}).json()
        with Path("tests/fixtures/wxecho_sample.csv").open("rb") as upload:
            preview = client.post(
                f"/api/projects/{project['id']}/imports/preview",
                files={"file": ("chat.csv", upload, "text/csv")},
            ).json()
        confirmed = client.post(
            f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        )
        event = client.post(
            f"/api/projects/{project['id']}/events",
            json={
                "type": "reconciliation",
                "start_message_id": "1",
                "end_message_id": "2",
                "before_state": "疏远",
                "after_state": "重新交流",
                "emotion_labels": ["期待"],
                "topic": "再次联系",
                "conflict_level": 1,
                "importance": 0.8,
                "reason": "重新开始对话",
                "evidence_ids": ["1", "2"],
            },
        ).json()
        model = client.post(
            f"/api/projects/{project['id']}/models",
            json={
                "base_model": "qwen",
                "adapter_path": "/models/test",
                "dataset_hash": "test-hash",
                "metrics": {
                    "blind_win_rate": 0.7,
                    "style_score": 0.8,
                    "safety_score": 0.9,
                },
            },
        ).json()
        branch = client.post(
            f"/api/projects/{project['id']}/branches",
            json={
                "origin_event_id": event["id"],
                "model_version_id": model["id"],
                "title": "重新回答",
                "origin_time": "2026-01-01T20:00:00",
                "state_snapshot": {"relationship_status": "重新交流"},
            },
        ).json()
        reply = client.post(
            f"/api/projects/{project['id']}/branches/{branch['id']}/messages",
            json={"content": "我们再聊聊吧"},
        )

    assert confirmed.status_code == 201
    assert reply.status_code == 201
    assert reply.json()["content"] == "收到：我们再聊聊吧"
