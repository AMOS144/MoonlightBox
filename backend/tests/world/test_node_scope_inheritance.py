"""范围继承烟测：恢复、跨节点拒绝、节点工具及草稿重试。"""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import pytest
from moonlightbox.config import Settings
from moonlightbox.db import Base
from moonlightbox.model_registry import register_models
from moonlightbox.world.models import (
    PersonWorldAgentRun,
    PersonWorldProfile,
    PersonWorldProfileDraft,
    PersonWorldSectionTask,
    WorldGraphVersion,
)
from moonlightbox.world.person_world.contracts.profile_v3 import PersonWorldProfileV3
from moonlightbox.world.person_world.node_scope import inherited_node_scope
from moonlightbox.world.person_world.section_retry_v3 import retry_v3
from moonlightbox.world.person_world.tools.current_profile import build_current_profile_tools
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


@pytest.mark.parametrize("second", [{}, {"node_scope": {"preview_hash": "other"}}])
def test_missing_or_other_scope_cannot_fall_back_to_global(second):
    with pytest.raises(ValueError, match="节点范围不一致"):
        inherited_node_scope(Mock(), Mock(), {"node_scope": {"preview_hash": "a"}}, second)


def test_node_current_profile_does_not_load_global_publication():
    session = Mock()
    tools = build_current_profile_tools(
        session,
        project_id="p",
        graph_version_id="g",
        node_scope={"preview_hash": "n"},
        profile_snapshot={"agency": {"summary": "起点的目标"}},
    )
    value = tools["get_current_profile_section"].invoke({"section": "agency"})
    assert value["content"] == {"summary": "起点的目标"}
    session.scalar.assert_not_called()


def test_restored_scope_is_frozen_and_not_current_selection():
    from moonlightbox.world.person_world.node_scope import NodeCompilationScope, same_node_scope

    boundary = {
        "included_count": 1,
        "cutoff_at": "2026-05-01T10:00:00+08:00",
        "timezone": "Asia/Shanghai",
        "last_included_ref": "a",
        "first_excluded_ref": "b",
        "preview_hash": "approved-old",
    }
    expected = NodeCompilationScope("g", boundary, "s", ("a", "b"))
    graph = NS(id="g", project_id="p", source_import_ids=[], source_fingerprint="s")
    with (
        patch("moonlightbox.world.jobs.load_world_messages", return_value=[]),
        patch("moonlightbox.world.bundles.source_fingerprint", return_value="s"),
        patch(
            "moonlightbox.world.person_world.node_scope.chronological_source_order",
            return_value={"a": 0, "b": 1},
        ),
    ):
        restored = inherited_node_scope(
            Mock(), graph, {"node_scope": expected.envelope()}, {"node_scope": expected.envelope()}
        )
        assert restored.envelope() == expected.envelope()
        assert same_node_scope(
            expected.envelope(), {**expected.envelope(), "graph_version_id": "child"}
        )
        with pytest.raises(ValueError):
            NodeCompilationScope.restore(
                Mock(), graph=graph, envelope={**expected.envelope(), "included_count": 2}
            )


def test_node_retry_updates_own_draft_not_global_profile(tmp_path):
    register_models()
    engine = create_engine(f"sqlite:///{tmp_path}/retry.db")
    Base.metadata.create_all(engine)
    scope = Mock()
    scope.boundary = {"timezone": "Asia/Shanghai"}
    with Session(engine) as session:
        graph = WorldGraphVersion(
            id="g",
            project_id="p",
            trigger_import_id="i",
            workspace_key="w",
            status="ready",
            source_fingerprint="s",
            config_fingerprint="c",
            compiler_version="v3",
            source_import_ids=[],
        )
        run = PersonWorldAgentRun(
            id="r",
            project_id="p",
            graph_version_id="g",
            mode="node_compile",
            status="awaiting_review",
            prompt_version="v3",
            state={"node_scope": {"id": "node"}},
        )
        draft = PersonWorldProfileDraft(
            id="d",
            project_id="p",
            graph_version_id="g",
            agent_run_id="r",
            status="awaiting_review",
            profile_schema_version="v3",
            claim_ids=[],
            payload={},
            profile_v3=PersonWorldProfileV3(subject_participant_id="target").model_dump(
                mode="json"
            ),
            generation_summary={"node_scope": {"id": "node"}},
        )
        task = PersonWorldSectionTask(agent_run_id="r", section="agency", status="researching")
        session.add_all([graph, run, draft, task])
        session.commit()
        fake = Mock(_batch=AsyncMock(), _enrich=AsyncMock())
        with (
            patch(
                "moonlightbox.world.person_world.node_scope.inherited_node_scope",
                return_value=scope,
            ),
            patch(
                "moonlightbox.world.person_world.section_retry_v3.PersonWorldCoordinatorV3",
                return_value=fake,
            ) as factory,
        ):
            retry_v3(
                NS(session=session),
                run=run,
                task=task,
                graph=graph,
                profile=draft,
                draft=draft,
                target=NS(id="target", name="目标"),
                user=NS(id="user", name="用户"),
                compiler=Mock(),
                sidecar=Mock(),
                settings=Settings(),
                resume_key="job:retry",
            )
            assert factory.call_args.kwargs["node_scope"] is scope
        session.refresh(draft)
        assert draft.generation_summary["candidate_revision"] == 2
        assert draft.generation_summary["node_scope"] == {"id": "node"}
        assert graph.status == "ready"
        assert session.scalar(select(PersonWorldProfile)) is None
    engine.dispose()
