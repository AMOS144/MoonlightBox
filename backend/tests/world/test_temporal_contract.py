"""后端时间检索责任边界：内部边界、同秒顺序及可辨认错误。"""

import json
from datetime import datetime

import httpx
import pytest

from moonlightbox.world.bundles import WorldMessage, chronological_source_order
from moonlightbox.world.client import LightRAGSidecarClient, LightRAGSidecarError
from moonlightbox.world.person_world.investigation_artifacts import InvestigationArtifactStore
from moonlightbox.world.person_world.tools.graph_query import build_graph_query_tools


def test_chronology_matches_import_source_order_not_uuid():
    def message(key, source):
        return WorldMessage(
            id=key,
            source_id=source,
            import_id="import",
            participant_id="p",
            participant_name="人",
            participant_role="target",
            timestamp=datetime(2026, 1, 1),
            kind="text",
            content="内容",
        )

    assert chronological_source_order([message("a", "10"), message("z", "2")]) == {"z": 0, "a": 1}


def test_temporal_failure_is_not_empty_retrieval():
    def handler(request):
        assert json.loads(request.content)["temporal"]["included_count"] == 3
        return httpx.Response(409, json={"detail": {"code": "temporal_index_missing"}})

    client = LightRAGSidecarClient(
        "http://sidecar", "secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(LightRAGSidecarError) as error:
        client.query(
            "world",
            "生活情况",
            temporal={"source_version": "v1", "included_count": 3, "period": "before"},
        )
    assert error.value.code == "temporal_index_missing"
    assert error.value.status_code == 409


def test_model_only_selects_question_and_period():
    tools = build_graph_query_tools(
        object(),
        workspace="world",
        allowed_document_names=set(),
        artifacts=InvestigationArtifactStore(),
        top_k=8,
        chunk_top_k=8,
        max_total_tokens=16000,
        temporal_scope={"source_version": "v1", "included_count": 3},
    )
    schema = tools["search_world"].args_schema.model_json_schema()
    assert set(schema["properties"]) == {"question", "period"}
    with pytest.raises(ValueError):
        tools["search_world"].args_schema.model_validate(
            {"question": "她目前生活情况如何？", "included_count": 100}
        )


def test_section_work_survives_checkpoint_and_compaction():
    from langchain_core.messages import HumanMessage
    from moonlightbox.world.person_world.section_agent_v3 import _compact_native_context
    from moonlightbox.world.person_world.tools.section_work import build_section_work_tool

    artifacts = InvestigationArtifactStore()
    build_section_work_tool(artifacts).invoke({"draft_notes": "起点仍在旧岗位", "review_questions": ["何时换岗？"], "findings": "后段说明后来才换岗", "next_action": "回查前段"})
    restored = InvestigationArtifactStore()
    restored.restore(artifacts.snapshot())
    assert restored.section_work == artifacts.section_work
    messages, _ = _compact_native_context([HumanMessage(content="任务")], source_refs=[], unresolved=[], section_work=restored.section_work)
    assert any("起点仍在旧岗位" in str(message.content) for message in messages)


def test_coordinator_rejects_cross_node_resume():
    from types import SimpleNamespace
    from moonlightbox.world.person_world.coordinator_v3 import PersonWorldCoordinatorV3

    coordinator = object.__new__(PersonWorldCoordinatorV3)
    coordinator.node_scope = SimpleNamespace(envelope=lambda: {"preview_hash": "new"})
    with pytest.raises(ValueError, match="跨节点"):
        coordinator._restore_run(SimpleNamespace(state={"node_scope": {"preview_hash": "old"}}))
