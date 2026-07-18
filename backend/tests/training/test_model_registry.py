from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.db import Database
from sqlalchemy.orm import Session


def test_registry_marks_only_qualified_model_as_recommended(tmp_path: Path) -> None:
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'models.db'}")
    ModelVersion.metadata.create_all(database.engine)

    with Session(database.engine) as session:
        registry = ModelRegistry(session)
        registry.create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/a",
            dataset_hash="hash-a",
            metrics={"blind_win_rate": 0.9, "style_score": 0.9, "safety_score": 0.2},
        )
        expected = registry.create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/b",
            dataset_hash="hash-b",
            metrics={"blind_win_rate": 0.6, "style_score": 0.8, "safety_score": 0.9},
        )

        selected = registry.recommend("project-1")

    assert selected.id == expected.id
    assert selected.recommended is True


def test_model_versions_are_exposed_and_recommended_by_api(
    client: TestClient,
) -> None:
    project = client.post("/api/projects", json={"name": "模型项目"}).json()
    payload = {
        "base_model": "qwen",
        "adapter_path": "/models/a",
        "dataset_hash": "hash-a",
        "metrics": {
            "blind_win_rate": 0.7,
            "style_score": 0.8,
            "safety_score": 0.9,
        },
    }
    created = client.post(
        f"/api/projects/{project['id']}/models",
        json=payload,
    )
    recommended = client.post(
        f"/api/projects/{project['id']}/models/recommend",
    )

    assert created.status_code == 201
    assert recommended.status_code == 200
    assert recommended.json()["recommended"] is True
