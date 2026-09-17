from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.world.compiler import COMPILER_VERSION, CompiledWorldProfile
from moonlightbox.world.models import PersonWorldProfile, WorldGraphVersion


def persist_profile(
    session: Session,
    *,
    graph: WorldGraphVersion,
    subject_person_id: str,
    compiled: CompiledWorldProfile,
    agent_run_id: str | None = None,
    generation_summary: dict[str, object] | None = None,
    profile_v2: dict[str, object] | None = None,
    profile_v3: dict[str, object] | None = None,
    profile_schema_version: str = "v1",
    investigation_report: dict[str, object] | None = None,
) -> PersonWorldProfile:
    node_hash = ((generation_summary or {}).get("node_scope") or {}).get("preview_hash")
    existing = session.scalar(
        select(PersonWorldProfile).where(
            PersonWorldProfile.node_boundary_hash == node_hash,
            PersonWorldProfile.graph_version_id == graph.id,
            *([PersonWorldProfile.agent_run_id == agent_run_id] if node_hash else []),
        )
    )
    draft = compiled.draft.model_dump(mode="json")
    values = {
        "node_boundary_hash": node_hash,
        "identity": draft["identity"],
        "work_and_education": draft["work_and_education"],
        "places": draft["places"],
        "social_relationships": draft["social_relationships"],
        "preferences": draft["preferences"],
        "recurring_activities": draft["recurring_activities"],
        "routine_summary": draft["routine_summary"],
        "life_phases": draft["life_phases"],
        "relationship_with_user": draft["relationship_with_user"],
        "important_events": draft["important_events"],
        "unresolved_candidates": draft["unresolved_candidates"],
        "source_message_ids": list(compiled.source_message_ids),
        "retrieval_manifest": [
            {
                "question": item.question,
                "mode": item.mode,
                "document_ids": list(item.document_ids),
            }
            for item in compiled.retrieval_manifest
        ],
        "compiler_version": COMPILER_VERSION,
        "agent_run_id": agent_run_id,
        "generation_summary": generation_summary or {},
        "profile_v2": profile_v2 or {},
        "profile_v3": profile_v3 or {},
        "profile_schema_version": profile_schema_version,
        "investigation_report": investigation_report or {},
    }
    if existing is None:
        existing = PersonWorldProfile(
            project_id=graph.project_id,
            subject_person_id=subject_person_id,
            graph_version_id=graph.id,
            **values,
        )
        session.add(existing)
    else:
        for key, value in values.items():
            setattr(existing, key, value)
    session.flush()
    return existing
