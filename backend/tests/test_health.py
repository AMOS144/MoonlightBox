from fastapi.testclient import TestClient


def test_health_returns_runtime_status(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "mlx_available": False,
    }
