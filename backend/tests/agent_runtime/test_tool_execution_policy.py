"""执行契约明确能力，声明并行资格不改变当前串行调度。"""

import pytest
from moonlightbox.agent_runtime.contracts import ToolContract
from moonlightbox.agent_runtime.deadlines import bounded_timeout, tool_deadline
from moonlightbox.agent_runtime.policy import DAY_PLAN_RUNTIME_POLICY
from moonlightbox.agent_runtime.resilience import (
    ExecutionInterrupted,
    ResiliencePolicy,
    execution_scope,
)
from moonlightbox.agent_runtime.tool_execution import ToolExecutionPolicy, submission_policy
from moonlightbox.runtime_v1.agent_catalog import load_agent_definition
from moonlightbox.runtime_v1.tools.execution_policy import POLICIES as RUNTIME_POLICIES
from moonlightbox.runtime_v1.tools.plan_work import PlanningToolbox
from moonlightbox.world.person_world.prompt_loader import (
    load_prompt_definition,
    load_section_prompt,
)
from moonlightbox.world.person_world.tools.execution_policy import (
    POLICIES as WORLD_POLICIES,
)
from moonlightbox.world.person_world.tools.execution_policy import (
    execution_policy,
)


def test_unknown_tool_is_serial_and_not_hard_cancellable():
    policy = ToolContract(name="extension").execution
    assert policy.parallelism == "serial"
    assert policy.cancellation == "boundaries"
    assert policy.exclusive_resources == ("execution_state",)
    assert "未审查" in policy.reason


def test_parallel_safety_cannot_conflict_with_exclusive_resources():
    with pytest.raises(ValueError, match="独占资源"):
        ToolExecutionPolicy(parallelism="parallel_safe")
    with pytest.raises(ValueError, match="必须串行"):
        ToolContract(
            name="proposal",
            side_effect="proposal",
            execution=ToolExecutionPolicy(parallelism="parallel_safe", exclusive_resources=()),
        )
    assert submission_policy().parallelism == "serial"
    assert "submission_receipt" in submission_policy().exclusive_resources
    assert "asset_authorizations" in submission_policy("submit_expression").exclusive_resources
    assert "investigation_artifacts" in submission_policy("submit_section").exclusive_resources


def test_declared_runtime_tools_have_explicit_review():
    # 通用入口由 build_execute_tool 合并内部工具契约，不复制一份静态业务策略。
    from moonlightbox.agent_runtime.tool_dispatch import build_execute_tool

    dispatcher = build_execute_tool(())
    assert dispatcher.contract.execution.parallelism == "serial"
    assert "未审查" not in dispatcher.contract.execution.reason
    gateways = {dispatcher.tool.name}
    submissions = {
        "submit_decision",
        "submit_day_plan",
        "submit_expression",
        "submit_life_result",
    }
    for name in (
        "director",
        "day_planner",
        "persona_actor",
        "day_planner_life_events",
    ):
        assert set(load_agent_definition(name).tool_names) <= (
            set(RUNTIME_POLICIES) | submissions | gateways
        )


def test_world_tools_and_same_name_variants_are_reviewed():
    for section in (
        "identity",
        "life_context",
        "social_world",
        "agency",
        "practices",
        "life_course",
        "relationship_with_user",
    ):
        assert set(load_section_prompt(section).tool_names) <= set(WORLD_POLICIES)
    assert set(load_prompt_definition("revision").tool_names) <= set(WORLD_POLICIES)
    assert execution_policy("get_current_profile_section").parallelism == "serial"
    frozen = execution_policy("get_current_profile_section", frozen_section=True)
    assert frozen.parallelism == "parallel_safe"
    assert not frozen.exclusive_resources
    assert "module_read_tracker" in execution_policy("read_context_module").exclusive_resources
    assert "sqlalchemy_session" in execution_policy("analyze_evidence_dates").exclusive_resources


def test_planning_work_is_not_automatically_replayed_as_read_only():
    box = PlanningToolbox(DAY_PLAN_RUNTIME_POLICY)
    registered = box.register([box.work_tool()])
    work = next(item for item in registered if item.tool.name == "update_plan_work")
    assert work.contract.side_effect == "proposal"
    assert work.contract.execution.parallelism == "serial"
    assert "planner_work" in work.contract.execution.exclusive_resources
    assert all(
        "runtime_result_cache" in item.contract.execution.exclusive_resources for item in registered
    )


def test_cooperative_network_boundary_checks_cancellation():
    with execution_scope(ResiliencePolicy(), lambda: 30, lambda: "cancelled"):
        with tool_deadline(20):
            with pytest.raises(ExecutionInterrupted, match="cancelled"):
                bounded_timeout(30)
