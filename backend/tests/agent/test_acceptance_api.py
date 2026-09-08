from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.branches.models import Branch
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def _seed(session: Session) -> None:
    session.add_all(
        [
            Project(id="project-1", name="项目一"),
            Project(id="project-2", name="项目二"),
        ]
    )
    session.flush()
    session.add_all(
        [
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="test",
                adapter_path="/model-1",
                dataset_hash="model-1",
                metrics={},
                active=True,
            ),
            ModelVersion(
                id="model-2",
                project_id="project-1",
                base_model="test",
                adapter_path="/model-2",
                dataset_hash="model-2",
                metrics={},
            ),
            ModelVersion(
                id="model-other",
                project_id="project-2",
                base_model="test",
                adapter_path="/model-other",
                dataset_hash="model-other",
                metrics={},
            ),
        ]
    )
    session.add_all(
        [
            EventNode(
                id="event-1",
                project_id="project-1",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            ),
            EventNode(
                id="event-other",
                project_id="project-2",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            Branch(
                id="branch-1",
                project_id="project-1",
                origin_event_id="event-1",
                model_version_id="model-1",
                title="分支一",
                origin_time=NOW,
                state_snapshot={},
            ),
            Branch(
                id="branch-other",
                project_id="project-2",
                origin_event_id="event-other",
                model_version_id="model-other",
                title="其他分支",
                origin_time=NOW,
                state_snapshot={},
            ),
        ]
    )
    session.commit()


def _payload(*, count: int = 20, model_id: str = "model-1") -> dict[str, object]:
    return {
        "model_version_id": model_id,
        "direct_lora_p95": 100,
        "observations": [
            {
                "source": "historical_replay",
                "cutoff": (NOW + timedelta(seconds=index)).isoformat(),
                "expected_express": index % 2 == 0,
                "predicted_express": index % 2 == 0,
                "extraction_succeeded": True,
                "fact_safe": True,
                "latency_ms": 100,
            }
            for index in range(count)
        ],
    }


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )


def test_acceptance_api_evaluates_latest_activates_and_rolls_back(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        database = Database(settings.database_url)
        with Session(database.engine) as session:
            _seed(session)

        insufficient = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/acceptance/evaluate",
            json=_payload(count=19),
        )
        passed = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/acceptance/evaluate",
            json=_payload(),
        )
        latest = client.get(
            "/api/projects/project-1/branches/branch-1/subject-agent/acceptance/latest"
        )
        activated = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/activate",
            json={"model_version_id": "model-1"},
        )
        rolled_back = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/rollback"
        )
        database.close()

    assert insufficient.status_code == 201
    assert insufficient.json()["passed"] is False
    assert "insufficient_samples" in insufficient.json()["failure_reasons"]
    assert passed.status_code == 201
    assert passed.json()["passed"] is True
    assert latest.status_code == 200
    assert latest.json()["id"] == passed.json()["id"]
    assert activated.status_code == 200
    assert activated.json()["subject_agent_mode"] == "active"
    assert rolled_back.status_code == 200
    assert rolled_back.json()["subject_agent_mode"] == "shadow"


def test_acceptance_api_rejects_cross_scope_and_wrong_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        database = Database(settings.database_url)
        with Session(database.engine) as session:
            _seed(session)

        wrong_branch = client.post(
            "/api/projects/project-1/branches/branch-other/subject-agent/acceptance/evaluate",
            json=_payload(),
        )
        wrong_model = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/acceptance/evaluate",
            json=_payload(model_id="model-2"),
        )
        activation_without_report = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/activate",
            json={"model_version_id": "model-1"},
        )
        database.close()

    assert wrong_branch.status_code == 404
    assert wrong_model.status_code == 409
    assert activation_without_report.status_code == 409


def test_acceptance_api_rejects_observation_without_strict_cutoff(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    payload = _payload()
    observations = payload["observations"]
    assert isinstance(observations, list)
    observations[0]["cutoff"] = "2026-07-23T12:00:00"

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/projects/project-1/branches/branch-1/subject-agent/acceptance/evaluate",
            json=payload,
        )

    assert response.status_code == 422
