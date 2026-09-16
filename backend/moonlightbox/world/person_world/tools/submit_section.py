"""人物栏目提交：结构和引用错误在工具回合中修正，不额外启动文本编译。"""

from moonlightbox.agent_runtime.input_contracts import input_model
from moonlightbox.agent_runtime.submission import result_submission_tool

from ..context_module_snapshots import validate_references
from ..contracts.context_modules import ContextModuleBase, ContextModuleRef
from ..contracts.profile_v3 import (
    SECTION_RESULT_MODELS,
    DimensionEntry,
    ModuleAssessment,
)
from ..contracts.psychological_models import ModelDimensionEntry
from ..section_support import _evidence_from_artifacts


def build_submit_section(section, context, tracker, artifacts):
    model = SECTION_RESULT_MODELS[section]
    projected = input_model(
        model,
        omit=lambda cls, name, field: (
            (issubclass(cls, DimensionEntry) and name == "dimension_id")
            or (issubclass(cls, ModelDimensionEntry) and name == "model")
            or name == "schema_version"
            or (issubclass(cls, (ContextModuleBase, ContextModuleRef)) and name == "revision")
            or (cls is ModuleAssessment and name == "status")
        ),
    )

    def bind(value):
        import json

        from moonlightbox.agent_runtime.tool_errors import ToolInputError

        data = value.model_dump(mode="json")
        modules = {item["id"]: item for item in tracker.snapshot.modules}

        def bind_refs(node, path="result"):
            if isinstance(node, dict):
                for index, ref in enumerate(node.get("context_module_refs", [])):
                    module = modules.get(ref["module_id"])
                    if module is None:
                        raise ToolInputError(
                            "module_id 不在当前快照中；请使用模块工具返回的引用。",
                            field=f"{path}.context_module_refs.{index}.module_id",
                        )
                    ref["revision"] = module["revision"]
                for key, child in node.items():
                    bind_refs(child, f"{path}.{key}")
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    bind_refs(child, f"{path}.{index}")

        bind_refs(data)
        for module in data.get("context_modules", []):
            previous = modules.get(module.get("id"))
            module["revision"] = previous["revision"] if previous else 1
        if "module_assessment" in data:
            data["module_assessment"]["status"] = (
                "identified" if data.get("context_modules") else "not_identified"
            )
        return model.model_validate_json(json.dumps(data))

    def validate(result, receipt_context):
        if section == "life_context" and (
            bool(result.context_modules) != (result.module_assessment.status == "identified")
        ):
            return "module_assessment 与 context_modules 不一致，请修正后提交"
        if not _has_materials(context, tracker.snapshot, artifacts, receipt_context.tool_results):
            return "materials_unavailable：尚无可用于形成栏目理解的材料，请先读取或查询"
        try:
            validate_references(
                result.model_dump(mode="json"),
                tracker.snapshot,
                {item.message_id for item in _evidence_from_artifacts(artifacts)}
                | _reference_ids(context.get("previous_section")),
            )
        except ValueError as error:
            return f"引用无效：{error}；请使用实际读取过的引用，或在不需要引用时留空"
        return None

    return result_submission_tool("submit_section", projected, validate, result_adapter=bind)


def _has_materials(context, snapshot, artifacts, tool_results):
    """集中补全可直接使用已完成栏目与固定模块，不强迫每个消费者重复检索原文。"""
    return bool(
        context.get("confirmed_correction")
        or (context.get("previous_section") or {}).get("summary")
        or any(context.get("section_summaries", {}).values())
        or (snapshot.availability == "ready" and snapshot.modules)
        or artifacts.evidence_rows()
        or any(
            isinstance(item, dict)
            and not item.get("error")
            and any(
                item.get(key) for key in ("context", "graph_data", "messages", "values", "modules")
            )
            for item in tool_results
        )
    )


def _reference_ids(value):
    """既有已校验 Profile 的引用可沿用，不强迫每轮重新检索同一批原文。"""
    if isinstance(value, dict):
        return set(value.get("reference_message_ids", [])) | set().union(
            *(_reference_ids(child) for child in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_reference_ids(child) for child in value))
    return set()
