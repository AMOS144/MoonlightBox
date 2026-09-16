"""统一历史工具冒烟：跨起点展开、融合、故障和越界引用。"""

# ruff: noqa: F811
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.tools import ToolException
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.runtime_v1.branch_models import BranchMessage
from moonlightbox.runtime_v1.conversation_history import ConversationHistory
from moonlightbox.runtime_v1.db_models import RuntimeSnapshotRow
from moonlightbox.runtime_v1.tools.director_context import ReadConversation
from moonlightbox.runtime_v1.tools.memory_search import MemorySearchQuery
from test_peer_collaboration import session  # noqa: F401


def history(session):
    now = datetime.now(UTC)
    session.add(
        ImportSource(
            id="i",
            project_id="p",
            preview_id="x",
            source_path="test",
            message_count=2,
            confirmed_at=now,
        )
    )
    session.add(Participant(id="person", project_id="p", name="目标", role="target"))
    session.flush()
    for i in range(2):
        session.add(
            Message(
                id=f"old{i}",
                project_id="p",
                import_id="i",
                participant_id="person",
                source_id=str(i),
                timestamp=now - timedelta(days=1),
                kind="text",
                content=f"以前的聊天{i}",
                raw={},
            )
        )
        session.add(
            BranchMessage(
                id=f"new{i}",
                branch_id="b",
                sequence=i,
                role="user",
                content=f"后来的聊天{i}",
                observed_at=now,
            )
        )
    snapshot = session.get(RuntimeSnapshotRow, "snapshot")
    snapshot.source_message_ids = ["old0", "old1"]
    snapshot.cutoff_at = now
    session.commit()
    return ConversationHistory(session, "b", snapshot, now)


def test_read_crosses_boundary_without_source_selector(session):
    service = history(session)
    assert service.read(around_ref="new0", limit=3)["source_ids"] == ["old1", "new0", "new1"]
    first = service.read(limit=2)
    assert first["source_ids"] == ["new0", "new1"]
    second = service.read(before_ref=first["next_before_ref"], limit=2)
    assert second["source_ids"] == ["old0", "old1"]
    assert second["next_before_ref"] is None
    with pytest.raises(ValueError, match="out_of_scope"):
        service.read(around_ref="another-branch-message")
    assert not {"source", "scope", "before_sequence", "import_before_ref"} & set(
        ReadConversation.model_fields
    )
    assert "scope" not in MemorySearchQuery.model_fields


def test_search_fuses_both_sources_and_reports_partial_failure(session, monkeypatch):
    service = history(session)
    monkeypatch.setattr(
        "moonlightbox.runtime_v1.conversation_index.search_branch",
        lambda *a, **kw: {
            "status": "ready",
            "data": [{"source_ref": "new0"}, {"source_ref": "new0"}],
        },
    )
    monkeypatch.setattr(
        "moonlightbox.runtime_v1.tools.routine_evidence.query_frozen_history",
        lambda *a, **kw: {
            "retrieval_status": "ok",
            "messages": [{"source_id": "old1"}, {"source_id": "outside"}],
        },
    )
    result = service.search("以前聊过什么")
    assert set(result["source_ids"]) == {"new0", "old1"}
    assert result["status"] == "ready"
    assert all("content" in item and "role" in item for item in result["data"])

    def unavailable(*args, **kwargs):
        raise ToolException("offline")

    monkeypatch.setattr(
        "moonlightbox.runtime_v1.tools.routine_evidence.query_frozen_history", unavailable
    )
    result = service.search("以前聊过什么")
    assert result["source_ids"] == ["new0"]
    assert result["status"] == "partial"
