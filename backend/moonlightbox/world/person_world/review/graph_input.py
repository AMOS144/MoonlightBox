"""图谱操作编号与旧状态由后端负责，Agent 只提出要改的内容。"""

from moonlightbox.agent_runtime.input_contracts import input_model
from moonlightbox.agent_runtime.tool_errors import ToolInputError

from ..schemas import GraphOperationDraft, GraphPatchDraft

GraphPatchInput = input_model(
    GraphPatchDraft,
    omit=lambda cls, name, field: (
        cls is GraphOperationDraft
        and name in {"operation_id", "before_description", "precondition_hash"}
    ),
)


def bind_graph_patch(value):
    data = value.model_dump(mode="json")
    for index, operation in enumerate(data["graph_operations"], 1):
        operation["operation_id"] = f"operation-{index}"
        kind = operation["operation_type"]
        required = (
            ("source_entities", "target_entity")
            if kind == "MERGE_ENTITIES"
            else ("source_entity", "target_entity")
            if kind.endswith("RELATION")
            else ("entity_name",)
        )
        for field in required:
            if not operation.get(field):
                raise ToolInputError(
                    f"{kind} 需要 {field}；选择当前已知实体或明确拟创建名称，不填其他操作的字段。",
                    field=f"result.graph_operations.{index - 1}.{field}",
                )
        if operation.get("cascade") and kind != "DELETE_ENTITY":
            raise ToolInputError(
                "cascade 仅用于拟删除实体的附带关系范围；其他操作省略。",
                field=f"result.graph_operations.{index - 1}.cascade",
            )
    return GraphPatchDraft.model_validate(data)
