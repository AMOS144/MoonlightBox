"""图谱快照 API：把 sidecar 的 LightRAG 图裁剪成前端可视化需要的字段。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from moonlightbox.api import create_app
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource
from moonlightbox.world.client import (
    LightRAGGraphEdge,
    LightRAGGraphNode,
    LightRAGSidecarClient,
    LightRAGSidecarError,
    LightRAGSidecarGraph,
)
from moonlightbox.world.models import WorldGraphVersion


def _settings(tmp_path: Path) -> Settings:
    return Settings.model_construct(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
        lightrag_enabled=True,
    )


def _add_graph(database_url: str, project_id: str) -> None:
    database = Database(database_url)
    iterator = database.session()
    session = next(iterator)
    try:
        session.add(
            ImportSource(
                id="import",
                project_id=project_id,
                preview_id="preview",
                source_path="chat.csv",
                message_count=2,
                confirmed_at=datetime.now(UTC),
            )
        )
        session.add(
            WorldGraphVersion(
                id="graph",
                project_id=project_id,
                trigger_import_id="import",
                workspace_key="world_project",
                status="ready",
                source_fingerprint="x",
                config_fingerprint="x",
                source_import_ids=["import"],
                compiler_version="test",
            )
        )
        session.commit()
    finally:
        iterator.close()


def test_graph_snapshot_requires_built_world(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.get("/api/projects/project/world-profile/graph")
        assert response.status_code == 404


def test_graph_snapshot_trims_node_and_edge_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/projects", json={"name": "图谱"}).json()
        _add_graph(settings.database_url, project["id"])

        def fake_get_graph(self: LightRAGSidecarClient, workspace: str) -> LightRAGSidecarGraph:
            assert workspace == "world_project"
            return LightRAGSidecarGraph(
                nodes=[
                    LightRAGGraphNode(
                        id="小月",
                        labels=["小月"],
                        properties={"entity_type": "person", "description": "在新公司工作", "rank": 3},
                    ),
                    LightRAGGraphNode(id="新公司", labels=["新公司"], properties={}),
                ],
                edges=[
                    LightRAGGraphEdge(
                        source="小月",
                        target="新公司",
                        properties={"keywords": "工作", "weight": 1.5},
                    )
                ],
                is_truncated=False,
            )

        monkeypatch.setattr(LightRAGSidecarClient, "get_graph", fake_get_graph)
        response = client.get(f"/api/projects/{project['id']}/world-profile/graph")

        assert response.status_code == 200
        assert response.json() == {
            "nodes": [
                {"id": "小月", "entity_type": "person", "description": "在新公司工作"},
                {"id": "新公司", "entity_type": None, "description": None},
            ],
            "edges": [{"source": "小月", "target": "新公司", "keywords": "工作"}],
            "truncated": False,
        }


def test_graph_snapshot_maps_sidecar_failure_to_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/projects", json={"name": "图谱"}).json()
        _add_graph(settings.database_url, project["id"])

        def failing_get_graph(self: LightRAGSidecarClient, workspace: str) -> LightRAGSidecarGraph:
            raise LightRAGSidecarError("lightrag_unavailable", "LightRAG 服务暂时不可用")

        monkeypatch.setattr(LightRAGSidecarClient, "get_graph", failing_get_graph)
        response = client.get(f"/api/projects/{project['id']}/world-profile/graph")

        assert response.status_code == 503
        assert response.json()["detail"] == "LightRAG 服务暂时不可用"
