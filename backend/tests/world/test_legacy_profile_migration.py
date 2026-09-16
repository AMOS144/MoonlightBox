from types import SimpleNamespace

from moonlightbox.world.person_world.migration import audit_legacy_profile


def test_v1_profile_is_only_mapped_as_reverification_candidates() -> None:
    profile = SimpleNamespace(
        id="legacy-profile-1",
        profile_schema_version="v1",
        identity={
            "names": [{"text": "洪欣羽", "source_message_ids": ["message-name"]}],
            "aliases": [],
            "self_descriptions": [],
            "roles": [{"text": "设计师", "source_message_ids": ["message-role"]}],
        },
        work_and_education=[],
        places=[],
        social_relationships=[],
        preferences=[],
        recurring_activities=[],
        routine_summary={"workdays": [{"text": "十点上班"}]},
        life_phases=[],
        relationship_with_user={"overview": [], "changes_over_time": []},
        important_events=[],
        unresolved_candidates=[{"text": "可能在某公司工作"}],
    )

    audit = audit_legacy_profile(profile)  # type: ignore[arg-type]

    assert audit is not None
    candidates = [item.as_dict() for item in audit.candidates]
    assert {
        "legacy_path": "identity.names",
        "legacy_index": 0,
        "candidate_v2_paths": ["identity.identifiers"],
        "source_message_ids": ["message-name"],
        "state": "requires_reverification",
    } in candidates
    role = next(item for item in candidates if item["legacy_path"] == "identity.roles")
    assert role["state"] == "ambiguous_destination"
    assert "life_context.work_and_learning" in role["candidate_v2_paths"]
    assert audit.unresolved_questions


def test_v2_profile_is_not_reclassified_as_legacy_data() -> None:
    profile = SimpleNamespace(profile_schema_version="v2")

    assert audit_legacy_profile(profile) is None  # type: ignore[arg-type]
