"""表达资料的编译契约、审核定位、版本隔离和渐进注入冒烟。"""

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from moonlightbox.agent_runtime.skills import build_skill_tool
from moonlightbox.runtime_v1.context_views import actor_context_payload, director_context_payload
from moonlightbox.runtime_v1.expression_profile import expression_material
from moonlightbox.runtime_v1.schemas import ContextPacket
from moonlightbox.world.person_world.context_module_snapshots import assign_entry_ids
from moonlightbox.world.person_world.contracts.display import profile_editor_catalog
from moonlightbox.world.person_world.contracts.profile_v3 import (
    EXPRESSION_FIELDS,
    SECTION_RESULT_MODELS,
    PersonWorldProfileV3,
)
from moonlightbox.world.person_world.review.profile_v3 import validate_selection
from pydantic import ValidationError


def profile(word):
    value = PersonWorldProfileV3(subject_participant_id="target").model_dump(mode="json")
    entry = value["relationship_with_user"]["expression_profile"]["forms_of_address"]
    entry.update(
        status="described",
        description="亲近时使用昵称",
        patterns=[
            {
                "form": word,
                "use_when": "轻松接话时偶尔使用",
                "avoid_when": "认真分歧时不用",
                "reference_message_ids": [],
            }
        ],
    )
    return value


def test_compilation_requires_material_and_review_can_select_it():
    value = profile("测试昵称")
    section = value["relationship_with_user"]
    parsed = SECTION_RESULT_MODELS["relationship_with_user"].model_validate(section)
    assert set(parsed.expression_profile.model_dump()) == {"schema_version", *EXPRESSION_FIELDS}
    missing = deepcopy(section)
    del missing["expression_profile"]
    with pytest.raises(ValidationError):
        SECTION_RESULT_MODELS["relationship_with_user"].model_validate(missing)
    # 历史持久化模型仍可读取；默认未知不是新生成完成。
    old = PersonWorldProfileV3.model_validate({**value, "relationship_with_user": missing})
    assert expression_material(old.model_dump())["status"] == "unknown"
    assign_entry_ids(value)
    selected = validate_selection(
        value,
        {
            "section": "relationship_with_user",
            "field_path": "expression_profile.forms_of_address",
        },
    )
    assert selected["id"] and selected["patterns"][0]["form"] == "测试昵称"
    assert profile_editor_catalog()["expression_fields"] == EXPRESSION_FIELDS


def test_skill_asset_is_frozen_isolated_and_not_part_of_native_schema():
    root = Path(__file__).parents[2] / "moonlightbox/runtime_v1/skills"
    a, b = profile("甲的昵称"), profile("乙的昵称")
    first = build_skill_tool(root, skill_contexts={"speaking": expression_material(a)})
    second = build_skill_tool(root, skill_contexts={"speaking": expression_material(b)})
    assert convert_to_openai_tool(first) == convert_to_openai_tool(second)
    assert first.metadata != second.metadata
    a["relationship_with_user"]["expression_profile"] = {}
    loaded = first.invoke({"skill": "speaking"})["contents"]
    assert "甲的昵称" in loaded and "乙的昵称" not in loaded
    assert "{{ expression_profile }}" not in loaded
    assert (
        loaded.index("不必每次都解释")
        < loaded.index("甲的昵称")
        < loaded.index("## 接住此刻的交流")
    )
    assert loaded.count("甲的昵称") == 1
    absent = build_skill_tool(root).invoke({"skill": "speaking"})["contents"]
    assert "not_provided" in absent and "{{ expression_profile }}" not in absent
    assert "乙的昵称" in second.invoke({"skill": "speaking"})["contents"]
    assert expression_material({})["status"] == "not_compiled"


def test_director_defers_details_but_actor_receives_same_material():
    value = profile("独有昵称")
    now = datetime.now(UTC)
    packet = ContextPacket(
        generated_at=now,
        virtual_now=now,
        timezone="Asia/Shanghai",
        trigger={},
        origin={"person_world_profile": value},
        current={"life_state": {}},
        branch={},
    )
    view = director_context_payload(packet)
    assert (
        "expression_profile" not in view["origin"]["person_world_profile"]["relationship_with_user"]
    )
    assert actor_context_payload(packet)["expression_style_profile"] == expression_material(value)
    assert packet.origin["person_world_profile"] == value
