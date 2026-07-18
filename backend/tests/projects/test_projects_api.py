from fastapi.testclient import TestClient


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
