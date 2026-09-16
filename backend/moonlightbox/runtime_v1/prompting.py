"""显式组装行为与输出契约，记录工具清单，不重复注入工具说明。

工具参数仍由 LangChain 注册。本地预览和运行时使用同一个组装结果。
交付工具由调用方显式指定；Schema 只通过原生工具传输。
"""

import hashlib
import json
from dataclasses import dataclass

from langchain_core.messages import SystemMessage

from moonlightbox.agent_runtime.submission import submission_instruction

from .agent_catalog import _DEFINITION_FILES, load_agent_definition


@dataclass(frozen=True)
class PromptAssembly:
    behavior: str
    source: str
    tool_names: tuple[str, ...]
    output_schema: dict
    submission_tool_name: str

    @property
    def text(self):
        return self.behavior + "\n\n" + submission_instruction(self.submission_tool_name)

    def manifest(self):
        """轻量来源清单进入 Phoenix，不重复记录整份 Schema。"""
        return {
            "behavior_source": self.source,
            "sha256": hashlib.sha256(self.text.encode()).hexdigest(),
            "tools": list(self.tool_names),
            "output_mode": "tool_submission",
            "submission_tool_name": self.submission_tool_name,
            "output_title": self.output_schema.get("title"),
            "capabilities": ["tool_result_paging"]
            if "read_runtime_result" in self.tool_names
            else [],
        }

    def message(self):
        return SystemMessage(
            content=self.text, additional_kwargs={"prompt_manifest": self.manifest()}
        )

    def preview(self):
        return {
            **self.manifest(),
            "behavior": self.behavior,
            "output_schema": self.output_schema,
            "rendered_system": self.text,
        }


def assemble_prompt(behavior, model, *, submission_tool_name, tools, source="inline"):
    names = tuple(getattr(getattr(tool, "tool", tool), "name", tool) for tool in tools)
    # 工具使用说明随 bind_tools 传递，不再将同一说明注入 SystemMessage。
    if submission_tool_name not in names:
        raise ValueError("提交工具未注册")
    return PromptAssembly(behavior, source, names, model.model_json_schema(), submission_tool_name)


def assemble_agent_prompt(name, model, *, tools, submission_tool_name):
    definition = load_agent_definition(name)
    return assemble_prompt(
        definition.system_prompt,
        model,
        tools=tools,
        submission_tool_name=submission_tool_name,
        source="prompts/" + _DEFINITION_FILES[name],
    )


def main():
    """无数据库、无模型请求的预览；声明能力预览不冒充运行时动态工具列表。"""
    import argparse

    from .expression_contracts import ExpressionResult
    from .schemas import DayPlanTurn, LifeDecision

    schemas = {
        "director": LifeDecision,
        "day_planner": DayPlanTurn,
        "persona_actor": ExpressionResult,
    }
    parser = argparse.ArgumentParser(description="预览 Runtime Agent 指令组成")
    parser.add_argument("agent", choices=schemas)
    args = parser.parse_args()
    definition = load_agent_definition(args.agent)
    submissions = {
        "director": "submit_decision",
        "day_planner": "submit_day_plan",
        "persona_actor": "submit_expression",
    }
    tools = (*definition.tool_names, "read_runtime_result", submissions[args.agent])
    result = assemble_agent_prompt(
        args.agent, schemas[args.agent], tools=tools, submission_tool_name=submissions[args.agent]
    ).preview()
    result["preview_scope"] = "声明工具能力；实际运行工具子集以 Phoenix prompt_manifest 为准"
    print(json.dumps(result, ensure_ascii=False, indent=2))
