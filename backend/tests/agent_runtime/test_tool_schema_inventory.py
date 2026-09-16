"""逐个检查只读工具输入契约，避免新增字段只有标题而没有用途说明。"""

import importlib
import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel


def argument_models():
    package = Path(__file__).parents[2] / "moonlightbox"
    modules = []
    for folder in ("runtime_v1/tools", "world/person_world/tools"):
        modules.extend(
            "moonlightbox." + str(p.relative_to(package).with_suffix("")).replace("/", ".")
            for p in (package / folder).glob("*.py")
            if p.name != "__init__.py"
        )
    modules += [
        "moonlightbox.agent_runtime.tasks",
        "moonlightbox.agent_runtime.tool_dispatch",
        "moonlightbox.world.merges",
    ]
    for name in modules:
        module = importlib.import_module(name)
        for _, model in inspect.getmembers(module, inspect.isclass):
            if model.__module__ != name or not issubclass(model, BaseModel):
                continue
            if model.__name__.endswith("Args") or model.__name__ in {
                "ReadConversation",
                "ReadConcerns",
                "ReadProfile",
                "SearchConversation",
                "PlanWork",
                "MemorySearchQuery",
            }:
                yield model


@pytest.mark.parametrize("model", list(argument_models()), ids=lambda m: m.__name__)
def test_each_input_field_has_authored_semantics(model):
    for name, field in model.model_json_schema().get("properties", {}).items():
        assert field.get("description"), (
            f"{model.__module__}.{model.__name__}.{name} 缺少字段用途说明"
        )


def submission_models():
    from moonlightbox.runtime_v1.conversation_maintenance import Summary
    from moonlightbox.runtime_v1.director_contracts import InitialState
    from moonlightbox.runtime_v1.expression_contracts import ExpressionResult
    from moonlightbox.runtime_v1.life_events.contracts import LifeAdvanceDecision
    from moonlightbox.runtime_v1.schemas import DayPlanTurn, LifeDecision
    from moonlightbox.world.merges import MergeCandidateResponse
    from moonlightbox.world.person_world.contracts.profile_v3 import SECTION_RESULT_MODELS
    from moonlightbox.world.person_world.schemas import (
        GraphPatchDraft,
        RegressionAssessmentBatch,
        RevisionAgentTurn,
    )

    return [
        DayPlanTurn,
        LifeDecision,
        InitialState,
        ExpressionResult,
        LifeAdvanceDecision,
        Summary,
        MergeCandidateResponse,
        RevisionAgentTurn,
        GraphPatchDraft,
        RegressionAssessmentBatch,
        *SECTION_RESULT_MODELS.values(),
    ]


@pytest.mark.parametrize("model", submission_models(), ids=lambda m: m.__name__)
def test_submission_nested_fields_keep_descriptions(model):
    schema = model.model_json_schema()
    for name, definition in [("root", schema), *schema.get("$defs", {}).items()]:
        for field_name, field in definition.get("properties", {}).items():
            assert field.get("description"), f"{model.__name__}.{name}.{field_name} 缺少说明"
