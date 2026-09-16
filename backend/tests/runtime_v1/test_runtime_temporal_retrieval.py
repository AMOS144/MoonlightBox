"""Runtime 历史工具不透传全图 context；边界不能被虚拟日期或模型参数扩大。"""

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from moonlightbox.agent_runtime.tool_errors import ToolServiceError
from moonlightbox.runtime_v1.snapshot_sources import temporal_retrieval_scope
from moonlightbox.runtime_v1.tools.routine_evidence import _temporal_result
from moonlightbox.world.client import LightRAGReference


def test_missing_historical_scope_fails_closed():
    with pytest.raises(ToolServiceError, match="冻结检索边界"):
        temporal_retrieval_scope(Mock(), NS(snapshot_mode="historical_cutoff", profile={}), NS())
    assert temporal_retrieval_scope(Mock(), NS(snapshot_mode="latest_profile"), NS()) is None


def test_runtime_scope_is_fixed_and_checks_complete_prefix():
    graph = NS(source_fingerprint="s", project_id="p", source_import_ids=["i"])
    snapshot = NS(
        snapshot_mode="historical_cutoff",
        source_message_ids=["a", "b"],
        profile={
            "_runtime_binding": {
                "temporal_scope": {
                    "source_version": "s",
                    "included_count": 2,
                    "access": "compiler",
                    "period": "after",
                }
            }
        },
    )
    with (
        patch("moonlightbox.world.jobs.load_world_messages", return_value=[]),
        patch("moonlightbox.world.bundles.source_fingerprint", return_value="s"),
        patch(
            "moonlightbox.world.bundles.chronological_source_order",
            return_value={"a": 0, "b": 1, "c": 2},
        ),
    ):
        scope = temporal_retrieval_scope(Mock(), snapshot, graph)
        assert scope["access"] == "runtime" and scope["period"] == "before"
        snapshot.source_message_ids = ["b", "a"]
        with pytest.raises(ToolServiceError, match="前缀"):
            temporal_retrieval_scope(Mock(), snapshot, graph)


def test_runtime_does_not_pass_through_global_text_and_marks_partial():
    scope = {"access": "runtime", "period": "before", "included_count": 0}
    result = NS(
        references=[],
        context="后来换工作了",
        global_graph_clues={"future": "secret"},
        temporal_diagnostics={**scope, "historical_raw_coverage_complete": False},
    )
    session = Mock()
    session.execute.return_value.all.return_value = []
    with patch(
        "moonlightbox.runtime_v1.tools.routine_evidence.frozen_message_ids", return_value=set()
    ):
        output = _temporal_result(session, NS(), NS(project_id="p"), "问题", result, scope)
    assert output["context"] == "[]"
    assert output["retrieval_status"] == "partial"
    assert "global_graph_clues" not in output


@pytest.mark.parametrize(
    "relation,period,key",
    [("mixed", "after", "a"), ("before", "after", "a"), ("before", "before", "future")],
)
def test_runtime_rejects_untrusted_reference(relation, period, key):
    scope = {"access": "runtime"}
    result = NS(
        temporal_diagnostics=scope,
        references=[
            LightRAGReference(
                file_path="doc",
                boundary_relation=relation,
                messages=[{"message_id": key, "period": period}],
            )
        ],
    )
    with patch(
        "moonlightbox.runtime_v1.tools.routine_evidence.frozen_message_ids", return_value={"a"}
    ):
        with pytest.raises(ToolServiceError):
            _temporal_result(Mock(), NS(), NS(), "问题", result, scope)


def test_history_page_uses_prefix_order_and_not_event_node():
    from datetime import datetime

    from moonlightbox.runtime_v1.source_history import SourceHistoryService

    session = Mock()
    session.get.return_value = NS(project_id="p")

    def message(key, source):
        return NS(
            id=key,
            source_id=source,
            content=key,
            kind="text",
            media_asset_id=None,
            timestamp=datetime(2026, 1, 1),
        )

    a, b = message("a", "2"), message("b", "10")
    session.execute.return_value.all.return_value = [(b, "target"), (a, "self")]
    snapshot = NS(id="s", graph_version_id="g", source_message_ids=["a", "b"])
    with patch("moonlightbox.runtime_v1.snapshot_sources.temporal_retrieval_scope"):
        page = SourceHistoryService(session)._historical_page(
            NS(id="branch", project_id="p"),
            snapshot,
            before=None,
            limit=1,
        )
    assert [item.id for item in page.items] == ["b"]
    assert page.has_more and page.next_cursor
