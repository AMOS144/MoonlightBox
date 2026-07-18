from fastapi.testclient import TestClient
from moonlightbox.config import Settings


def test_create_and_read_project(client: TestClient) -> None:
    created = client.post("/api/projects", json={"name": "测试项目"})

    assert created.status_code == 201
    project_id = created.json()["id"]

    loaded = client.get(f"/api/projects/{project_id}")

    assert loaded.status_code == 200
    assert loaded.json()["name"] == "测试项目"
    assert loaded.json()["status"] == "created"


def test_list_projects_newest_first(client: TestClient) -> None:
    client.post("/api/projects", json={"name": "第一个项目"})
    client.post("/api/projects", json={"name": "第二个项目"})

    response = client.get("/api/projects")

    assert response.status_code == 200
    assert [project["name"] for project in response.json()] == [
        "第二个项目",
        "第一个项目",
    ]


def test_delete_project(client: TestClient) -> None:
    created = client.post("/api/projects", json={"name": "待删除项目"})
    project_id = created.json()["id"]

    deleted = client.delete(f"/api/projects/{project_id}")

    assert deleted.status_code == 204
    assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_delete_project_removes_database_rows_and_local_artifacts(
    client: TestClient,
    settings: Settings,
) -> None:
    project = client.post("/api/projects", json={"name": "彻底删除"}).json()
    project_id = project["id"]
    client.post(
        f"/api/projects/{project_id}/events",
        json={
            "type": "conflict",
            "start_message_id": "m1",
            "end_message_id": "m2",
            "before_state": "朋友",
            "after_state": "疏远",
            "emotion_labels": ["失望"],
            "topic": "冲突",
            "conflict_level": 3,
            "importance": 0.8,
            "reason": "测试",
            "evidence_ids": ["m1"],
        },
    )
    artifact_dirs = [
        settings.data_dir / "projects" / project_id,
        settings.chroma_dir / "projects" / project_id,
        settings.model_dir / "projects" / project_id,
    ]
    for directory in artifact_dirs:
        directory.mkdir(parents=True)
        (directory / "artifact").write_text("private", encoding="utf-8")

    response = client.delete(f"/api/projects/{project_id}")

    assert response.status_code == 204
    assert client.get(f"/api/projects/{project_id}/events").json() == []
    assert all(not directory.exists() for directory in artifact_dirs)
