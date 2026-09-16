from moonlightbox.db import Base
from moonlightbox.world.models import (
    PersonWorldProfile,
    WorldGraphVersion,
    WorldPublication,
)
from moonlightbox.world.person_world.tools.current_profile import build_current_profile_tools
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def test_candidate_recompile_reads_published_v2_profile_as_its_baseline() -> None:
    """候选 Graph 没有 Profile 时，不能让栏目 Agent 丢掉当前已发布档案。"""

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        base_graph = _graph("graph-base", status="ready")
        candidate_graph = _graph("graph-candidate", status="candidate")
        profile = PersonWorldProfile(
            id="profile-base",
            project_id="project-1",
            subject_person_id="target-1",
            graph_version_id=base_graph.id,
            identity={},
            work_and_education=[],
            places=[],
            social_relationships=[],
            preferences=[],
            recurring_activities=[],
            routine_summary={},
            life_phases=[],
            relationship_with_user={},
            important_events=[],
            unresolved_candidates=[],
            source_message_ids=[],
            retrieval_manifest=[],
            compiler_version="person-world-agent-v3",
            profile_schema_version="v2",
            profile_v2={
                "identity": {"identifiers": [{"statement": "洪欣羽自称小羽"}]},
            },
        )
        session.add_all(
            [
                base_graph,
                candidate_graph,
                profile,
                WorldPublication(
                    project_id="project-1",
                    graph_version_id=base_graph.id,
                    profile_id=profile.id,
                    correction_head_hash="h" * 64,
                    status="active",
                    published_by="local_user",
                ),
            ]
        )
        session.flush()

        tools = build_current_profile_tools(
            session,
            project_id="project-1",
            graph_version_id=candidate_graph.id,
        )
        payload = tools["get_current_profile_section"].invoke({"section": "identity"})

        assert payload["profile_id"] == profile.id
        assert payload["profile_graph_version_id"] == base_graph.id
        assert payload["content"] == {"identifiers": [{"statement": "洪欣羽自称小羽"}]}


def _graph(graph_id: str, *, status: str) -> WorldGraphVersion:
    return WorldGraphVersion(
        id=graph_id,
        project_id="project-1",
        trigger_import_id="import-1",
        workspace_key=f"workspace-{graph_id}",
        status=status,
        source_fingerprint=("s" if graph_id == "graph-base" else "t") * 64,
        config_fingerprint=("c" if graph_id == "graph-base" else "d") * 64,
        source_import_ids=["import-1"],
        compiler_version="person-world-agent-v3",
    )
