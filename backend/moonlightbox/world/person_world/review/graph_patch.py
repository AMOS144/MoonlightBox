"""Graph Patch 提案生成与变更集校验。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.tasks import run_submission_task
from moonlightbox.world.models import (
    PersonWorldProfile,
    PersonWorldRevisionSession,
    WorldGraphChangeSet,
    WorldGraphVersion,
)

from ..graph_executor import (
    canonical_change_set_hash,
    canonical_graph_state_hash,
    operation_precondition_state,
)
from ..schemas import GraphPatchDraft
from .payloads import (
    _affected_entities,
    _affected_relations,
    _graph_state_description,
    _profile_payload,
)


def generate_graph_patch(
    service,
    revision: PersonWorldRevisionSession,
    change_set: WorldGraphChangeSet,
) -> None:
    """仅在 Profile Patch 获批后生成最小、可独立审核的 Graph Patch。"""

    from .service import RevisionStateError

    service._ensure_base_is_current(revision)
    if not isinstance(revision.understanding_payload, dict):
        raise RevisionStateError("纠正会话缺少已确认理解")
    graph = service.session.get(WorldGraphVersion, change_set.base_graph_version_id)
    if graph is None:
        raise RevisionStateError("基础图版本不存在")
    profile = (
        service.session.get(PersonWorldProfile, revision.base_profile_id)
        if revision.base_profile_id
        else None
    )
    source_ids = [
        value
        for value in revision.understanding_payload.get("source_message_ids", [])
        if isinstance(value, str)
    ]
    from . import service as service_module

    context = service_module._source_message_context(
        service.session, revision.project_id, source_ids
    )
    if (
        profile is not None
        and profile.profile_schema_version == "v3"
        and not revision.understanding_payload.get("graph_change_requested", False)
    ):
        change_set.graph_operations = []
        change_set.regression_queries = []
        change_set.canonical_payload_hash = canonical_change_set_hash(
            profile_patch=change_set.profile_patch,
            graph_operations=[],
            regression_queries=[],
            base_graph_version_id=change_set.base_graph_version_id,
            revision=change_set.revision,
        )
        return
    from pathlib import Path

    from langchain_core.tools import StructuredTool

    from moonlightbox.agent_runtime.contracts import RegisteredTool, ToolContract
    from moonlightbox.agent_runtime.tool_execution import serial

    def search_graph(question: str):
        """检索当前待纠正图谱的相关实体、关系与上下文；只读。"""
        from ..node_scope import inherited_node_scope

        scope = inherited_node_scope(
            service.session,
            graph,
            revision.scope,
            profile.generation_summary if profile is not None else {},
        )
        result = service.lightrag.query(
            graph.workspace_key,
            question,
            mode="mix",
            top_k=16,
            chunk_top_k=8,
            max_total_tokens=6000,
            **({"temporal": {**scope.retrieval_scope(), "period": "before"}} if scope else {}),
        )
        return {
            "context": result.context,
            "references": [item.model_dump(mode="json") for item in result.references],
            "temporal_diagnostics": getattr(result, "temporal_diagnostics", None),
            "global_graph_clues": getattr(result, "global_graph_clues", None),
        }

    from ..tools.graph_query import GraphPatchSearchArgs

    tool = StructuredTool.from_function(search_graph, args_schema=GraphPatchSearchArgs)

    def validate_patch(result, submission_context):
        for operation in result.graph_operations:
            if not set(operation.source_message_ids).issubset(source_ids):
                return "source_message_ids 只能引用当前确认范围提供的消息。"
        return None

    from .graph_input import GraphPatchInput, bind_graph_patch

    draft = run_submission_task(
        compiler=service.compiler,
        name="graph_patch",
        system_prompt=Path(__file__).parents[1].joinpath("prompts/graph_patch.md").read_text(),
        payload={
            "understanding": revision.understanding_payload,
            "approved_profile_patch": change_set.profile_patch,
            "current_profile": _profile_payload(profile),
            "source_messages": context,
        },
        result_model=GraphPatchDraft,
        input_model=GraphPatchInput,
        result_adapter=bind_graph_patch,
        owner_id=f"{revision.id}:{change_set.revision}",
        session=service.session,
        project_id=revision.project_id,
        input_revision=revision.input_revision,
        input_revision_resolver=lambda: service.session.scalar(
            select(PersonWorldRevisionSession.input_revision).where(
                PersonWorldRevisionSession.id == revision.id,
                PersonWorldRevisionSession.status != "cancelled",
            )
        ),
        validator=validate_patch,
        tools=(
            RegisteredTool(
                tool,
                ToolContract(
                    name=tool.name,
                    execution=serial("revision_graph_read", reason="共享图谱只读客户端"),
                    timeout_seconds=300,
                    max_result_chars=32000,
                ),
            ),
        ),
    )
    allowed_ids = set(source_ids)
    operations: list[dict[str, object]] = []
    for operation in draft.graph_operations:
        sanitized = operation.model_copy(
            update={
                "source_message_ids": [
                    item for item in operation.source_message_ids if item in allowed_ids
                ]
            }
        )
        # 模型只描述意图；服务端读取当前图状态后签发 precondition hash，
        # 防止用户批准后的 workspace 被其他操作悄然改写。
        precondition = operation_precondition_state(
            service.lightrag, graph.workspace_key, sanitized
        )
        # 用户审核的 diff 必须展示服务端刚读取到的图状态，而不是让模型凭记忆
        # 填“旧描述”。这两个展示字段不参与 precondition hash，也不作为后续查询
        # 条件；真正执行仍由同一个 precondition hash 守护。
        before_description = _graph_state_description(precondition)
        updates: dict[str, object] = {}
        if before_description and sanitized.before_description is None:
            updates["before_description"] = before_description
        if (
            sanitized.operation_type.endswith("RELATION")
            and sanitized.relation_description
            and sanitized.after_description is None
        ):
            updates["after_description"] = sanitized.relation_description
        if updates:
            sanitized = sanitized.model_copy(update=updates)
        sanitized = sanitized.model_copy(
            update={"precondition_hash": canonical_graph_state_hash(precondition)}
        )
        operations.append(sanitized.model_dump(mode="json"))
    regression_queries = [
        {"query": item, "expected_change": "按用户确认的纠正回答"}
        for item in draft.regression_queries
    ]
    change_set.graph_operations = operations
    change_set.affected_entities = _affected_entities(operations)
    change_set.affected_relations = _affected_relations(operations)
    change_set.regression_queries = regression_queries
    change_set.canonical_payload_hash = canonical_change_set_hash(
        profile_patch=change_set.profile_patch,
        graph_operations=operations,
        regression_queries=regression_queries,
        base_graph_version_id=change_set.base_graph_version_id,
        revision=change_set.revision,
    )


def load_change_set(
    session: Session,
    revision: PersonWorldRevisionSession,
    change_set_id: str,
    revision_number: int,
    payload_hash: str,
) -> WorldGraphChangeSet:
    from .service import RevisionStateError

    change_set = session.get(WorldGraphChangeSet, change_set_id)
    if (
        change_set is None
        or change_set.revision_session_id != revision.id
        or change_set.project_id != revision.project_id
    ):
        raise RevisionStateError("变更集不存在")
    if (
        change_set.revision != revision_number
        or change_set.canonical_payload_hash != payload_hash
    ):
        raise RevisionStateError("变更集已经变化，请重新查看并确认")
    return change_set
