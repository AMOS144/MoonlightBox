from types import SimpleNamespace

from moonlightbox.runtime_v1.profile_projection import (
    _profile_payload,
    _profile_statements,
    _runtime_routine_profile,
)


def test_runtime_snapshot_projection_reads_v2_profile_not_legacy_compatibility_fields() -> None:
    profile = SimpleNamespace(
        profile_schema_version="v2",
        profile_v2={
            "identity": {"identifiers": []},
            "life_context": {
                "work_and_learning": [
                    {
                        "statement": "目标人物在新公司工作",
                        "evidence_message_ids": ["message-work"],
                    }
                ],
                "places_and_environment": [],
            },
            "social_world": {"ties": []},
            "agency": {
                "preferences": [],
                "values_and_interpretations": [],
                "goals_and_commitments": [],
            },
            "practices": {
                "recurring_activities": [],
                "temporal_rhythms": [
                    {
                        "statement": "工作日早上通常较忙",
                        "evidence_message_ids": ["message-rhythm"],
                    }
                ],
            },
            "life_course": {"episodes": [], "transitions": [], "trajectories": []},
            "relationship_with_user": {
                "standing": [],
                "interaction_observations": [],
                "interaction_patterns": [],
                "history": [],
            },
        },
        # 如果 Runtime 意外读取这里，测试会立即暴露。
        identity={"names": [{"text": "遗留名字"}]},
        work_and_education=[{"text": "遗留工作"}],
        places=[],
        social_relationships=[],
        preferences=[],
        recurring_activities=[],
        routine_summary={"other_patterns": [{"text": "遗留规律"}]},
        relationship_with_user={},
        important_events=[],
        life_phases=[],
    )

    payload = _profile_payload(profile)

    assert payload["profile_schema_version"] == "v2"
    assert payload["work_and_education"][0]["statement"] == "目标人物在新公司工作"
    assert payload["work_and_education"][0]["statement"] != "遗留工作"
    assert _profile_statements(payload["work_and_education"]) == [
        ("目标人物在新公司工作", ["message-work"])
    ]
    assert (
        _runtime_routine_profile(profile)["other_patterns"][0]["statement"]
        == "工作日早上通常较忙"
    )
