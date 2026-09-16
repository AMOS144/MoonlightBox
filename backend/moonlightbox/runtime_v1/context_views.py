"""把内部 Runtime 状态投影成各 Agent 真正需要看到的上下文。

数据库主键、分支身份和版本号留在 Python/Executor 内部。Agent 只看到做当前决策所需
的语义字段与可引用来源，因而不需要靠 Prompt 反复提醒“不要输出内部 ID”。
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, timedelta, timezone, tzinfo
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .persona_context import PersonaContextAssembler
from .plan_context import DayPlanContext
from .schemas import ContextPacket

_INTERNAL_KEYS = {
    "profile_id",
    "graph_version_id",
    "publication_id",
    "job_id",
    "owner_key",
    "branch_id",
    "snapshot_id",
    "model_version_id",
    "index_version_id",
    "index_version",
    "current_plan_block_id",
    "previous_version_id",
    "version",
    "id",
}


def director_context_payload(packet: ContextPacket) -> dict[str, Any]:
    """Director 看到完整语义状态，但看不到持久化实现细节。"""

    return director_context_payload_dict(packet.model_dump(mode="json"))


def director_context_payload_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """供 token 预算在 ``ContextPacket`` 回填前计算同一份模型输入。"""

    trigger = _trigger_view(payload.get("trigger", {}))
    current = _without_internal_keys(payload.get("current", {}))
    origin = _without_internal_keys(payload.get("origin", {}))
    # 具体表达资料按需从 speaking 读取，不在每轮认知上下文中重复铺开。
    relation = origin.get("person_world_profile", {}).get("relationship_with_user", {})
    if isinstance(relation, dict):
        if "expression_profile" in relation:
            # 新资料已有明确归属，不能同时携带旧 IdentityKernel 风格形成两套冲突输入。
            current.pop("expression_style_profile", None)
        relation.pop("expression_profile", None)
    current.pop("collaboration", None)
    subjective = current.get("life_state", {}).pop("subjective_state", {})
    branch = payload.get("branch", {})
    message_refs = {
        item.get("source_id") for item in branch.get("working_window", {}).get("messages", [])
    }
    if trigger.get("payload", {}).get("source_message_id") in message_refs:
        trigger["payload"].pop("content", None)
    branch_view = {
        "working_window": _without_internal_keys(branch.get("working_window", {})),
        "recent_events": [
            _event_view(item) for item in branch.get("recent_events", []) if isinstance(item, dict)
        ],
        "overlay_summary": branch.get("overlay_summary"),
    }
    return {
        "protocol_version": payload.get("protocol_version"),
        "virtual_now": payload.get("virtual_now"),
        "timezone": payload.get("timezone"),
        "trigger": trigger,
        "origin": origin,
        "current": current,
        "subjective_state": subjective,
        "collaboration": deepcopy(payload.get("current", {}).get("collaboration", [])),
        "branch": branch_view,
        "memory": {
            "retrieved_records": _without_internal_keys(
                payload.get("memory", {}).get("retrieved_records", [])
            )
        },
    }


def actor_context_payload(packet: ContextPacket, **kwargs) -> dict[str, Any]:
    """PersonaActor 只看表达所需材料，不接触完整的规划与审计状态。"""

    from .expression_profile import expression_material

    view = director_context_payload(packet)
    material = expression_material(packet.origin.get("person_world_profile"))
    if material["status"] != "not_compiled":
        # Actor 仍承担最终措辞，消费同一版本资产，不让本次接线只影响 Director。
        view["current"]["expression_style_profile"] = material
    return PersonaContextAssembler().assemble(view, **kwargs)


def day_plan_context_payload(context: DayPlanContext) -> dict[str, Any]:
    """DayPlanAgent 看到当地日期和语义计划，不看到数据库行、分支或快照 ID。"""

    raw = context.model_dump(mode="json")
    request = _without_internal_keys(raw["request"])
    # Director 的请求字段叫 target_date；Planner 的公开上下文统一称 plan_date，
    # 避免让模型在两个相同日期字段之间猜测。
    request.pop("target_date", None)
    return {
        "protocol_version": raw["protocol_version"],
        "plan_date": raw["target_date"],
        "local_now": _local_now(context),
        "mode": raw["mode"],
        "allow_no_change": raw["allow_no_change"],
        "day_plans": deepcopy(raw["day_plans"]),
        "collaboration": deepcopy(raw["collaboration"]),
        "working_state": deepcopy(raw["working_state"]),
        "subjective_state": deepcopy(raw.get("subjective_state", {})),
        "request": request,
        "hard_constraints": _without_internal_keys(raw["hard_constraints"]),
        "prior_plan": _without_internal_keys(raw.get("prior_plan")),
        # Origin 只提供语义提示。文档名、旧编译器生成的伪消息 ID 和摘录不进入
        # Planner；需要引用的真实证据统一由 analyze_routine_evidence 返回。
        "origin": _origin_hint_view(raw["origin_projection"]),
        "date_features": deepcopy(raw["date_features"]),
        "branch_evidence": _without_internal_keys(raw["branch_evidence"]),
        "evidence_requirements": deepcopy(raw.get("evidence_requirements", {})),
        "budget": deepcopy(raw["budget"]),
    }


def _trigger_view(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = cast(dict[str, Any], _without_internal_keys(value))
    if isinstance(value.get("id"), str):
        result["source_event_id"] = value["id"]
    payload = value.get("payload")
    if isinstance(payload, dict):
        clean_payload = cast(dict[str, Any], _without_internal_keys(payload))
        if isinstance(payload.get("branch_message_id"), str):
            clean_payload["source_message_id"] = payload["branch_message_id"]
        result["payload"] = clean_payload
    result["additional_triggers"] = [
        _event_view(item) for item in value.get("additional_triggers", []) if isinstance(item, dict)
    ]
    if "input_events" in value:
        result["input_events"] = [
            _trigger_view(item) for item in value["input_events"] if isinstance(item, dict)
        ]
    return result


def _event_view(value: dict[str, Any]) -> dict[str, Any]:
    result = cast(dict[str, Any], _without_internal_keys(value))
    if isinstance(value.get("id"), str):
        result["source_event_id"] = value["id"]
    return result


def _without_internal_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_internal_keys(item)
            for key, item in value.items()
            if key not in _INTERNAL_KEYS and key != "branch_message_id"
        }
    if isinstance(value, list):
        return [_without_internal_keys(item) for item in value]
    return deepcopy(value)


def _origin_hint_view(value: Any) -> Any:
    """去掉无法直接审计的 Profile 来源字段，只保留用于发起检索的语义提示。"""

    omitted = {
        "source_document_ids",
        "source_message_ids",
        "evidence_message_ids",
        "evidence_quotes",
        "speaker_message_id",
        "subject_binding",
    }
    if isinstance(value, dict):
        return {
            key: _origin_hint_view(item)
            for key, item in value.items()
            if key not in _INTERNAL_KEYS and key not in omitted
        }
    if isinstance(value, list):
        return [_origin_hint_view(item) for item in value]
    return deepcopy(value)


def _local_now(context: DayPlanContext) -> str:
    """在代码层完成时区换算，Planner 不需要接触 timezone 配置本身。"""

    value = context.virtual_now
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    match = re.fullmatch(r"UTC([+-])(\d{2}):(\d{2})", context.timezone)
    if match is not None:
        minutes = int(match.group(2)) * 60 + int(match.group(3))
        if match.group(1) == "-":
            minutes = -minutes
        zone: tzinfo = timezone(timedelta(minutes=minutes))
    else:
        try:
            zone = ZoneInfo(context.timezone)
        except ZoneInfoNotFoundError:
            zone = UTC
    return aware.astimezone(zone).isoformat()
