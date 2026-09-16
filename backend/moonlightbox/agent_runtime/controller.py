"""基于 LangChain + LangGraph 的统一 Agent Harness。

LangChain 负责模型和工具协议，LangGraph 负责显式状态图与可选恢复检查点。
执行不需要 AgentRun/Step/ToolInvocation 自建账本；真实调用仍由 Phoenix 观测。
Runtime 同级协作通过作用域注入原生 checkpointer，其他调用者维持原执行方式。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from time import monotonic, time
from typing import Any, Literal, TypedDict, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from moonlightbox.observability import (
    AgentExecutionTraceContext,
    agent_execution_span,
    agent_execution_trace_scope,
    record_agent_outcome,
)
from moonlightbox.observability.agent_observer import AgentRunObserver

from .capacity import RequestCapacityExceeded, capacity_scope
from .contracts import (
    AgentBudgetPolicy,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentSpec,
    BudgetLedger,
    ChatModel,
    ProgressDelta,
    RegisteredTool,
    SubmissionContext,
    ToolExecution,
)
from .deadlines import tool_deadline
from .resilience import (
    ExecutionInterrupted,
    classify_failure,
    execution_scope,
    request_timeout,
    run_operation,
)
from .submission import submission_context


class _LoopState(TypedDict, total=False):
    """可序列化的内部图状态；持久化时连同分页工具正文一起恢复。"""

    messages: list[BaseMessage]
    input_revision: int
    output: object
    accepted_value: object
    terminal_status: str
    terminal_reason: str
    accepted_submission: str
    planned_calls: list[dict[str, Any]]
    executions: list[ToolExecution]
    tool_results: list[object]
    seen_progress_keys: list[str]
    no_progress_steps: int
    seen_call_fingerprints: list[str]
    repetition_warned: bool
    progress_revision: int
    source_refs: list[str]
    unresolved: list[str]
    budget: BudgetLedger
    started_wall: float
    execution_attempt: int
    error: dict[str, Any] | None
    work_state: dict[str, Any]
    capacity_recoveries: int
    previous_capacity_estimate: int


@dataclass(slots=True)
class _ExecutionContext:
    spec: AgentSpec[Any]
    request: AgentExecutionRequest
    model: ChatModel
    execution_id: str
    tools: dict[str, RegisteredTool]
    trace_context: AgentExecutionTraceContext
    observer: AgentRunObserver


class AgentLoopController:
    """所有 Agent 共用的单次运行循环。

    Harness 不接收数据库 Session，也不读取、写入或提交数据库。业务事务仍由 Runtime、
    PersonWorld、Revision 各自管理。
    """

    def run[FinalT](
        self,
        *,
        spec: AgentSpec[FinalT],
        request: AgentExecutionRequest,
        model: ChatModel,
    ) -> AgentExecutionResult[FinalT]:
        from .persistence import request_checkpoint
        from .policy import inject_policy

        spec = inject_policy(spec, model)
        with request_checkpoint(request):
            return self._run(spec=spec, request=request, model=model)

    def _run[FinalT](
        self,
        *,
        spec: AgentSpec[FinalT],
        request: AgentExecutionRequest,
        model: ChatModel,
    ) -> AgentExecutionResult[FinalT]:
        context = self._start_context(spec=spec, request=request, model=model)
        with (
            agent_execution_trace_scope(context.trace_context),
            agent_execution_span(spec, request) as phoenix_span,
        ):
            context.observer.record_run_input(
                root_span=phoenix_span,
                messages=request.messages,
            )
            try:
                from .persistence import current_checkpoint

                checkpoint = current_checkpoint()
                graph = self._build_graph(
                    context, checkpointer=checkpoint[0] if checkpoint else None
                )
                config = {"recursion_limit": self._recursion_limit(spec.budget)}
                initial = self._initial_state(context)
                resume_blocked = None
                if checkpoint:
                    from .checkpoint_contract import checkpoint_contract_fingerprint

                    config["configurable"] = {
                        "thread_id": (
                            checkpoint[1]
                            + ":submission-v3:"
                            + spec.prompt_version
                            + ":"
                            + checkpoint_contract_fingerprint(spec)
                        )
                    }
                    saved = graph.get_state(config)
                    if saved.values:
                        if saved.values["input_revision"] != request.input_revision:
                            raise ValueError("stale_input_revision")
                        if spec.restore_work_state and saved.values.get("work_state") is not None:
                            spec.restore_work_state(saved.values["work_state"])
                        if spec.restore_tool_results:
                            spec.restore_tool_results(saved.values.get("tool_results", []))
                        # 恢复时刷新本轮系统指令，保留新提交协议内的调查材料。
                        graph.update_state(
                            config,
                            {
                                # 首个输入节点也可能成为最后存档；完整保留其状态，不能只写消息。
                                **saved.values,
                                "messages": [
                                    *(m for m in request.messages if isinstance(m, SystemMessage)),
                                    *(
                                        m
                                        for m in saved.values["messages"]
                                        if not isinstance(m, SystemMessage)
                                    ),
                                ],
                            },
                        )
                        saved = graph.get_state(config)
                        if spec.resume_messages and saved.values.get("terminal_status") not in {
                            "succeeded",
                            "waiting_for_user",
                        }:
                            graph.update_state(
                                config,
                                {
                                    "messages": spec.resume_messages(
                                        saved.values["messages"], request.messages
                                    )
                                },
                            )
                            saved = graph.get_state(config)
                        initial = None
                        if saved.values.get("terminal_status") not in {
                            "succeeded",
                            "waiting_for_user",
                        }:
                            attempt = saved.values.get("execution_attempt", 1)
                            from .persistence import job_manages_recovery

                            if not job_manages_recovery() and attempt >= spec.budget.max_execution_attempts:
                                resume_blocked = {
                                    **saved.values,
                                    **_guard_terminal("execution_attempt_limit"),
                                    "accepted_value": None,
                                }
                            else:
                                # 工作材料与累计消耗不丢；停机等待不计入新尝试的墙钟。
                                graph.update_state(
                                    config,
                                    {
                                        "started_wall": time(),
                                        "execution_attempt": attempt + 1,
                                        "no_progress_steps": 0,
                                        "repetition_warned": False,
                                    },
                                )
                                saved = graph.get_state(config)
                        if not saved.next and (
                            saved.values.get("terminal_status")
                            not in {"succeeded", "waiting_for_user"}
                            or (
                                spec.submission_tool_name
                                and saved.values.get("accepted_submission")
                                != spec.submission_tool_name
                            )
                            or saved.values.get("terminal_reason")
                            not in {"success", "needs_user_input:submitted"}
                        ):
                            # 显式重试复用调查材料与累计预算，但不把旧失败当成新结果。
                            initial = {
                                **saved.values,
                                "terminal_status": None,
                                "terminal_reason": None,
                                "accepted_value": None,
                                "error": None,
                                "output": None,
                            }
                    result = resume_blocked or (
                        graph.invoke(initial, config, durability="sync")
                        if initial is not None or saved.next
                        else saved.values
                    )
                else:
                    result = graph.invoke(initial, config)
                # 成功检查点也不能绕过当前任务取消；只复用工作结果，不复用交付权限。
                guard_reason = self._input_guard_reason(context)
                if guard_reason is not None:
                    result = {**result, **_guard_terminal(guard_reason), "accepted_value": None}
            except Exception as error:
                reason = _error_code(error) or "controller_graph_error"
                if request.on_status:
                    try:
                        request.on_status(
                            {
                                "status": "failed",
                                "reason": reason,
                                "error": asdict(classify_failure(error)),
                                "execution_id": context.execution_id,
                            }
                        )
                    except Exception:
                        pass
                record_agent_outcome(
                    phoenix_span,
                    status="failed",
                    reason=reason,
                    execution_id=context.execution_id,
                    result=None,
                )
                context.observer.finalize(
                    root_span=phoenix_span,
                    status="failed",
                    reason=reason,
                    result=None,
                )
                raise
            from .persistence import report_budget

            report_budget(result["budget"])
            outcome = AgentExecutionResult(
                execution_id=context.execution_id,
                status=_terminal_status(str(result.get("terminal_status", "failed"))),
                terminal_reason=str(
                    result.get("terminal_reason", "controller_ended_without_terminal")
                ),
                value=cast(FinalT | None, result.get("accepted_value")),
                messages=tuple(result.get("messages", [])),
                error=result.get("error"),
            )
            record_agent_outcome(
                phoenix_span,
                status=outcome.status,
                reason=outcome.terminal_reason,
                execution_id=outcome.execution_id,
                result=outcome.value,
            )
            context.observer.finalize(
                root_span=phoenix_span,
                status=outcome.status,
                reason=outcome.terminal_reason,
                result=outcome.value,
            )
            if request.on_status:
                try:
                    request.on_status(
                        {
                            "status": outcome.status,
                            "reason": outcome.terminal_reason,
                            "error": outcome.error,
                            "execution_id": outcome.execution_id,
                        }
                    )
                except Exception:
                    pass
            return outcome

    @contextmanager
    def _request_scope(self, state, context):
        budget = state.setdefault("budget", BudgetLedger())
        # 旧检查点没有校正字段；从保守初值开始，不丢已有材料和累计用量。
        if not hasattr(budget, "request_calibration"):
            budget.request_calibration = {}
        with (
            execution_scope(
                context.spec.resilience,
                lambda: context.spec.budget.max_wall_seconds - (time() - state["started_wall"]),
                lambda: self._input_guard_reason(context),
                context.request.on_status,
            ),
            capacity_scope(context.spec.budget, budget.request_calibration),
        ):
            yield

    def _invoke_model(self, model, messages, state, context):
        with self._request_scope(state, context):
            if getattr(model, "has_managed_request_boundaries", False):
                # 复合适配器内部可能发多次请求；各请求独立重试，不重放已成功的前半段。
                return model.invoke(messages)
            return run_operation(lambda: model.invoke(messages), kind="model")

    @staticmethod
    def _start_context(
        *,
        spec: AgentSpec[Any],
        request: AgentExecutionRequest,
        model: ChatModel,
    ) -> _ExecutionContext:
        declared = [item for item in spec.tools if item.tool.name == spec.submission_tool_name]
        if len(declared) != 1 or not declared[0].is_submission:
            raise ValueError("Agent 必须显式绑定一个结果提交工具")
        execution_id = str(uuid4())
        trace_context = AgentExecutionTraceContext.from_request(
            execution_id=execution_id,
            spec=spec,
            request=request,
        )
        return _ExecutionContext(
            spec=spec,
            request=request,
            model=model,
            execution_id=execution_id,
            tools={item.tool.name: item for item in spec.tools},
            trace_context=trace_context,
            observer=AgentRunObserver(
                trace_context=trace_context,
                spec=spec,
            ),
        )

    @staticmethod
    def _initial_state(context: _ExecutionContext) -> _LoopState:
        return {
            "input_revision": context.request.input_revision,
            "messages": list(context.request.messages),
            "planned_calls": [],
            "executions": [],
            "tool_results": [],
            "seen_progress_keys": [],
            "no_progress_steps": 0,
            "seen_call_fingerprints": [],
            "repetition_warned": False,
            "progress_revision": 0,
            "source_refs": [],
            "unresolved": [],
            "budget": BudgetLedger(),
            "started_wall": time(),
            "execution_attempt": 1,
        }

    def _build_graph(self, context: _ExecutionContext, checkpointer=None) -> Any:
        graph = StateGraph(_LoopState)
        # Codex 会在发出请求前先根据上下文窗口准备/压缩 history；不能等到第一轮
        # 因上下文过大被供应商拒绝以后才处理。此节点和每轮工具回填后的
        # ``prepare_next_turn`` 复用同一条确定性压缩路径。
        graph.add_node(
            "prepare_initial_context",
            lambda state: self._prepare_context(state, context),
        )
        graph.add_node("guard", lambda state: self._guard(state, context))
        graph.add_node("model_action", lambda state: self._model_action(state, context))
        graph.add_node("plan_tool_batch", lambda state: self._plan_tool_batch(state, context))
        graph.add_node("execute_tool_batch", lambda state: self._execute_tool_batch(state, context))
        graph.add_node("evaluate_progress", lambda state: self._evaluate_progress(state, context))
        graph.add_node("prepare_next_turn", lambda state: self._prepare_next_turn(state, context))
        graph.add_edge(START, "prepare_initial_context")
        graph.add_edge("prepare_initial_context", "guard")
        graph.add_conditional_edges(
            "guard",
            self._route_after_guard,
            {"model": "model_action", "terminal": END},
        )
        graph.add_conditional_edges(
            "model_action",
            self._route_after_model,
            {
                "tools": "plan_tool_batch",
                "guard": "prepare_next_turn",
                "terminal": END,
            },
        )
        graph.add_edge("plan_tool_batch", "execute_tool_batch")
        graph.add_edge("execute_tool_batch", "evaluate_progress")
        graph.add_edge("evaluate_progress", "prepare_next_turn")
        graph.add_conditional_edges(
            "prepare_next_turn",
            self._route_after_prepare,
            {"continue": "guard", "terminal": END},
        )
        return graph.compile(checkpointer=checkpointer)

    def _guard(self, state: _LoopState, context: _ExecutionContext) -> dict[str, Any]:
        reason = self._guard_reason(state, context)
        if reason is None:
            return {}
        # 只有持续重复相同调用及结果才介入；没有新来源 ID 不是结束调查的理由。
        if reason != "repeated_tool_cycle":
            return _guard_terminal(reason)
        if state.get("repetition_warned"):
            return _guard_terminal(reason)
        return {
            "repetition_warned": True,
            "no_progress_steps": 0,
            "messages": [
                *state["messages"],
                HumanMessage(
                    content=(
                        "RUNTIME_LOOP_FEEDBACK：连续重复了相同工具参数与结果。"
                        "请基于已有材料继续推断、修正调用、保存新草稿或提交结果；"
                        "仍可使用全部已声明工具，无须为了新增来源而检索。"
                    )
                ),
            ],
        }

    @staticmethod
    def _route_after_guard(state: _LoopState) -> Literal["model", "terminal"]:
        if state.get("terminal_status"):
            return "terminal"
        return "model"

    def _model_action(self, state: _LoopState, context: _ExecutionContext) -> dict[str, Any]:
        try:
            model: Any = _bounded_model(context.model, state, context)
        except TimeoutError:
            # guard 与模型节点之间也可能刚好到期，仍以正常熔断结果结束。
            return _guard_terminal("wall_deadline")
        visible_tools = [
            item.tool
            for item in context.tools.values()
            if item.contract.side_effect != "executor_only"
        ]
        bind_tools = getattr(model, "bind_tools", None)
        if visible_tools and callable(bind_tools):
            if any((tool.metadata or {}).get("native_strict") is False for tool in visible_tools):
                from langchain_core.utils.function_calling import convert_to_openai_tool

                # 分工具保留 strict 策略：执行入口允许动态参数，其余工具仍使用原约束。
                # 只在此处生成一次实际绑定的 Schema，不把技能依赖变成模型可见工具。
                model = bind_tools(
                    [
                        convert_to_openai_tool(
                            tool, strict=(tool.metadata or {}).get("native_strict", True)
                        )
                        for tool in visible_tools
                    ]
                )
            else:
                try:
                    model = bind_tools(visible_tools, strict=True)
                except TypeError:
                    model = bind_tools(visible_tools)
        with context.observer.model_call(
            phase="decision",
            model_name=_trace_model_name(model),
            messages=list(state["messages"]),
            visible_tools=visible_tools,
        ) as (phoenix_span, observation):
            try:
                state["budget"].model_steps += 1
                output = self._invoke_model(model, list(state["messages"]), state, context)
            except Exception as error:
                if isinstance(error, RequestCapacityExceeded):
                    # 请求尚未离开进程，不消耗模型轮数，也不是供应商调用失败。
                    state["budget"].model_steps -= 1
                    return self._recover_capacity(state, context, error)
                context.observer.record_model_failure(span=phoenix_span, error=error)
                return {**self._model_error(error), "budget": state["budget"]}
            context.observer.record_model_result(
                span=phoenix_span,
                observation=observation,
                output=output,
                model_name=_trace_model_name(model),
            )
        after_return_reason = self._input_guard_reason(context)
        if time() - state["started_wall"] >= context.spec.budget.max_wall_seconds:
            return _guard_terminal("wall_deadline")
        if after_return_reason is not None:
            return _guard_terminal(after_return_reason)
        budget = state["budget"]
        self._record_provider_usage(budget, output)
        tool_calls = _tool_calls(output)
        messages = [
            *state["messages"],
            *([output] if isinstance(output, BaseMessage) else []),
        ]
        update: dict[str, Any] = {
            "capacity_recoveries": 0,
            "previous_capacity_estimate": 0,
            "output": output,
            "messages": messages,
            "budget": budget,
            "work_state": context.spec.snapshot_work_state()
            if context.spec.snapshot_work_state
            else {},
        }
        if not tool_calls:
            # 普通文本永远不是交付。允许模型下一轮改为工具提交，仍受统一预算与无进展保护。
            update["no_progress_steps"] = state.get("no_progress_steps", 0) + 1
            update["messages"] = [
                *messages,
                HumanMessage(
                    content=(
                        f"尚未交付结果，请调用 {context.spec.submission_tool_name}；"
                        "普通文本不算提交。"
                    )
                ),
            ]
        return update

    @staticmethod
    def _route_after_model(
        state: _LoopState,
    ) -> Literal["tools", "guard", "terminal"]:
        if state.get("terminal_status"):
            return "terminal"
        if _tool_calls(state.get("output")):
            return "tools"
        return "guard"

    def _recover_capacity(self, state, context, error):
        """由完整请求准入驱动压缩；每次重新装配、校正和预留输出，禁止无效压缩循环。"""
        count = state.get("capacity_recoveries", 0)
        estimate = error.report["estimated_input_tokens"]
        previous = state.get("previous_capacity_estimate", 0)
        compactor = context.spec.context_compactor
        if compactor is None or count >= 2 or (previous and estimate >= previous):
            return {**self._model_error(error), "budget": state["budget"]}
        reason = self._input_guard_reason(context)
        if reason:
            return _guard_terminal(reason)
        before = list(state["messages"])
        started = monotonic()
        messages, summary = compactor(
            before,
            source_refs=state.get("source_refs", []),
            unresolved=state.get("unresolved", []),
        )
        context.observer.observe_compaction(
            before_messages=before,
            after_messages=messages,
            summary=summary,
            duration_ms=int((monotonic() - started) * 1000),
        )
        if messages == before:
            return {**self._model_error(error), "budget": state["budget"]}
        return {
            "messages": messages,
            "output": None,
            "budget": state["budget"],
            "capacity_recoveries": count + 1,
            "previous_capacity_estimate": estimate,
        }

    @staticmethod
    def _plan_tool_batch(state: _LoopState, context: _ExecutionContext) -> dict[str, Any]:
        planned: list[dict[str, Any]] = []
        calls = _tool_calls(state.get("output"))
        if len(calls) > 1 and any(
            call.get("name") == context.spec.submission_tool_name for call in calls
        ):
            return {
                "planned_calls": [
                    {
                        "call_id": call.get("id") or uuid4().hex,
                        "name": call.get("name", ""),
                        "args": call.get("args", {}),
                        "error": "submission_must_be_separate",
                    }
                    for call in calls
                ]
            }
        policy_limit = context.spec.budget.max_total_tool_calls
        completed_calls = sum(state["budget"].tool_calls_by_name.values())
        remaining_calls = None if policy_limit is None else max(policy_limit - completed_calls, 0)
        for call in _tool_calls(state.get("output")):
            name = str(call.get("name", ""))
            args = call.get("args", {})
            call_id = str(call.get("id") or uuid4().hex)
            if call.get("type") == "invalid_tool_call":
                planned.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": {},
                        "raw_arguments": args,
                        "error": "invalid_tool_arguments",
                        "message": call.get("error") or "参数无法解析，请使用 JSON 对象重新调用",
                    }
                )
                continue
            if not isinstance(args, dict) or name not in context.tools:
                planned.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args if isinstance(args, dict) else {},
                        "error": "undeclared_tool_or_invalid_arguments",
                        "message": "工具未声明或参数无效；请使用本轮提供的工具名称及参数结构",
                    }
                )
                continue
            registered = context.tools[name]
            missing = registered.contract.required_permissions - context.request.scope.permissions
            if registered.contract.side_effect == "executor_only" or missing:
                planned.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args,
                        "error": "tool_not_authorized",
                    }
                )
                continue
            if remaining_calls is not None and remaining_calls <= 0:
                planned.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args,
                        "error": "tool_call_safety_limit",
                    }
                )
                continue
            planned.append({"call_id": call_id, "name": name, "args": args})
            if remaining_calls is not None:
                remaining_calls -= 1
        return {"planned_calls": planned}

    def _execute_tool_batch(
        self,
        state: _LoopState,
        context: _ExecutionContext,
    ) -> dict[str, Any]:
        executions: list[ToolExecution] = []
        tool_messages: list[ToolMessage] = []
        # 当前有意串行。execution.parallelism 是已审查能力，不是自动启动线程的开关；
        # Session、工件和结果投影未完成实例隔离前，不得绕过 exclusive_resources 并发。
        for planned in state.get("planned_calls", []):
            registered = context.tools.get(str(planned.get("name", "")))
            with context.observer.tool_call(
                tool_name=str(planned.get("name", "unknown")),
                arguments=planned.get("args", {}),
                contract=registered.contract if registered is not None else None,
            ) as phoenix_span:
                if (reason := self._input_guard_reason(context)) is not None:
                    execution = _cancelled_execution(planned, reason)
                else:
                    execution = self._execute_one(planned, state, context)
                context.observer.record_tool_result(span=phoenix_span, execution=execution)
            executions.append(execution)
            contract = registered.contract if registered is not None else None
            model_result = (
                contract.model_result_projector(execution.result)
                if contract is not None
                else execution.result
            )
            tool_messages.append(
                ToolMessage(
                    content=_tool_message_content(
                        model_result,
                        limit=contract.max_result_chars if contract is not None else 6_000,
                    ),
                    tool_call_id=execution.call_id,
                    name=execution.tool_name,
                    status="error" if execution.status in {"error", "cancelled"} else "success",
                )
            )
        return {
            "executions": executions,
            "work_state": context.spec.snapshot_work_state()
            if context.spec.snapshot_work_state
            else {},
            "tool_results": [*state.get("tool_results", []), *(item.result for item in executions)],
            "messages": [*state["messages"], *tool_messages],
        }

    def _execute_one(
        self,
        planned: Mapping[str, Any],
        state: _LoopState,
        context: _ExecutionContext,
    ) -> ToolExecution:
        name = str(planned.get("name", "unknown"))
        call_id = str(planned.get("call_id") or uuid4().hex)
        args = planned.get("args")
        normalized_args = args if isinstance(args, dict) else {}
        if "error" in planned or name not in context.tools:
            from .tool_errors import rejected_call_result

            return ToolExecution(
                call_id,
                name,
                normalized_args,
                rejected_call_result(planned, tool_name=name, available_tools=context.tools),
                "error",
                ProgressDelta(summary="调用被拒绝"),
            )
        registered = context.tools[name]
        start = monotonic()
        retries = 0
        attempts = []

        def invoke_tool():
            attempts.append(None)
            with tool_deadline(
                request_timeout(registered.contract.timeout_seconds, model_request=False)
            ):
                return registered.tool.invoke(normalized_args)

        try:
            with (
                self._request_scope(state, context),
                submission_context(self._submission_context(state, context)) as receipt,
            ):
                tool_result = run_operation(
                    invoke_tool,
                    kind="tool:" + name,
                    replay_safe=registered.contract.side_effect == "read_only",
                    timeout_seconds=registered.contract.timeout_seconds,
                )
            retries = max(0, len(attempts) - 1)
            result = registered.contract.result_normalizer(tool_result)
        except Exception as error:
            from .tool_errors import tool_error_result

            failure = classify_failure(error)
            retries = max(0, len(attempts) - 1)
            return ToolExecution(
                call_id,
                name,
                normalized_args,
                tool_error_result(
                    error, tool_name=name, arguments=normalized_args, attempts=len(attempts)
                ),
                "cancelled" if failure.category == "cancellation" else "error",
                ProgressDelta(summary="工具调用失败"),
                retries,
                int((monotonic() - start) * 1000),
            )
        if (reason := self._input_guard_reason(context)) is not None:
            return _cancelled_execution(planned, reason)
        duration = int((monotonic() - start) * 1000)
        delta = registered.contract.progress_evaluator(
            result,
            state_revision=state.get("progress_revision", 0),
            seen_keys=frozenset(state.get("seen_progress_keys", [])),
        )
        return ToolExecution(
            call_id,
            name,
            normalized_args,
            result,
            "error"
            if name == context.spec.submission_tool_name
            and isinstance(result, Mapping)
            and result.get("status") == "rejected"
            else "succeeded"
            if delta.made_progress
            else "empty",
            delta,
            retries,
            duration,
            receipt if registered.is_submission and receipt.value is not None else None,
        )

    @staticmethod
    def _evaluate_progress(
        state: _LoopState,
        context: _ExecutionContext,
    ) -> dict[str, Any]:
        budget = state["budget"]
        seen = set(state.get("seen_progress_keys", []))
        source_refs = list(state.get("source_refs", []))
        made_progress = False
        seen_calls = set(state.get("seen_call_fingerprints", []))
        repeated_batch = bool(state.get("executions"))
        for execution in state.get("executions", []):
            # 不解释正文语义；相同输入得到不同结果、更新草稿、修正参数都允许继续。
            fingerprint = _canonical_hash(
                {
                    "tool": execution.tool_name,
                    "args": execution.args,
                    "result": context.tools[execution.tool_name].contract.comparison_projection(
                        execution.result
                    )
                    if execution.tool_name in context.tools
                    else execution.result,
                    "status": execution.status,
                }
            )
            if fingerprint not in seen_calls:
                repeated_batch = False
            seen_calls.add(fingerprint)
            # 被安全阈值拒绝的“计划调用”没有真正执行，不能反过来把已用额度加一。
            limit_rejection = (
                execution.status == "error"
                and isinstance(execution.result, Mapping)
                and execution.result.get("error") == "tool_call_safety_limit"
            )
            if not limit_rejection:
                budget.tool_calls_by_name[execution.tool_name] = (
                    budget.tool_calls_by_name.get(execution.tool_name, 0) + 1
                )
            # 资源边界必须计算真正回填给模型的投影 ToolMessage，而不是只在 Phoenix 中
            # 可见的完整结果。否则一个存入执行期工件库的大结果会错误地提前结束 Agent。
            contract = context.tools.get(execution.tool_name)
            model_result = (
                contract.contract.model_result_projector(execution.result)
                if contract is not None
                else execution.result
            )
            budget.tool_result_chars += len(
                _tool_message_content(
                    model_result,
                    limit=contract.contract.max_result_chars if contract is not None else 6_000,
                )
            )
            new_keys = execution.progress.new_keys - seen
            if execution.status not in {"error", "cancelled"}:
                source_refs.append(f"tool:{execution.tool_name}")
            if new_keys:
                made_progress = True
                seen.update(new_keys)
                source_refs.extend(sorted(new_keys))
        submission = next(
            (
                item
                for item in state.get("executions", [])
                if item.tool_name == context.spec.submission_tool_name
                and item.status not in {"error", "cancelled"}
                and item.submission is not None
                and item.submission.value is not None
            ),
            None,
        )
        completed = {}
        if submission is not None:
            receipt = submission.submission
            completed = {
                "accepted_value": receipt.value,
                "terminal_status": receipt.status,
                "terminal_reason": "needs_user_input:submitted"
                if receipt.status == "waiting_for_user"
                else "success",
                "accepted_submission": submission.tool_name,
            }
        return {
            **completed,
            "budget": budget,
            "seen_progress_keys": sorted(seen),
            "source_refs": list(dict.fromkeys(source_refs)),
            "seen_call_fingerprints": sorted(seen_calls),
            "no_progress_steps": state.get("no_progress_steps", 0) + 1
            if repeated_batch and not made_progress
            else 0,
            "progress_revision": state.get("progress_revision", 0) + int(made_progress),
        }

    def _prepare_next_turn(
        self,
        state: _LoopState,
        context: _ExecutionContext,
    ) -> dict[str, Any]:
        if state.get("terminal_status"):
            return {}
        return self._prepare_context(state, context)

    def _prepare_context(
        self,
        state: _LoopState,
        context: _ExecutionContext,
    ) -> dict[str, Any]:
        """在每次模型调用前维护可见上下文，而不占用模型输出预算。"""

        guard_reason = self._input_guard_reason(context)
        if guard_reason is not None:
            return _guard_terminal(guard_reason)
        messages = list(state.get("messages", []))
        if context.spec.refresh_inputs is not None:
            messages = context.spec.refresh_inputs(messages)
        local_chars = _request_context_chars(messages, context)
        # 原生客户端已有完整请求准入，不能再用另一套字符阈值每轮提前
        # 丢弃阅读材料。真正超窗由 _recover_capacity 触发压缩并重新准入。
        native_capacity = bool(getattr(context.model, "has_managed_request_capacity", False))
        if (
            not native_capacity
            and local_chars >= context.spec.budget.compaction_threshold_chars
            and context.spec.context_compactor is not None
        ):
            before_messages = list(messages)
            compaction_started = monotonic()
            messages, summary = context.spec.context_compactor(
                messages,
                source_refs=state.get("source_refs", []),
                unresolved=state.get("unresolved", []),
            )
            context.observer.observe_compaction(
                before_messages=before_messages,
                after_messages=messages,
                summary=summary,
                duration_ms=int((monotonic() - compaction_started) * 1000),
            )
            local_chars = _request_context_chars(messages, context)
        state["budget"].local_context_chars = local_chars
        if (
            not native_capacity
            and context.spec.budget.max_context_chars is not None
            and local_chars > context.spec.budget.max_context_chars
        ):
            return _guard_terminal("input_context_limit")
        return {
            "messages": messages,
            "budget": state["budget"],
            "work_state": context.spec.snapshot_work_state()
            if context.spec.snapshot_work_state
            else state.get("work_state", {}),
        }

    @staticmethod
    def _route_after_prepare(state: _LoopState) -> Literal["continue", "terminal"]:
        if state.get("terminal_status"):
            return "terminal"
        return "continue"

    def _guard_reason(
        self,
        state: _LoopState,
        context: _ExecutionContext,
    ) -> str | None:
        input_reason = self._input_guard_reason(context)
        if input_reason is not None:
            return input_reason
        budget = state["budget"]
        policy = context.spec.budget
        if time() - state["started_wall"] >= policy.max_wall_seconds:
            return "wall_deadline"
        if (
            policy.emergency_max_model_steps is not None
            and budget.model_steps >= policy.emergency_max_model_steps
        ):
            return "emergency_model_step_limit"
        if (
            policy.max_total_tool_calls is not None
            and sum(budget.tool_calls_by_name.values()) >= policy.max_total_tool_calls
        ):
            return "tool_call_safety_limit"
        if budget.tool_result_chars >= policy.max_tool_result_chars:
            return "tool_result_context_limit"
        if state.get("no_progress_steps", 0) >= policy.max_unchanged_state_steps:
            return "repeated_tool_cycle"
        return None

    @staticmethod
    def _input_guard_reason(context: _ExecutionContext) -> str | None:
        from .cancellation import cancellation_requested

        try:
            if cancellation_requested():
                return "cancelled"
        except Exception:
            return "input_revision_unavailable"
        cancellation = context.request.cancellation_requested
        if cancellation is not None:
            try:
                if cancellation():
                    return "cancelled"
            except Exception:
                return "input_revision_unavailable"
        resolver = context.request.input_revision_resolver
        if resolver is None:
            return None
        try:
            current_revision = resolver()
        except Exception:
            return "input_revision_unavailable"
        if current_revision == context.request.input_revision:
            return None
        return "stale_input_revision"

    @staticmethod
    def _submission_context(
        state: _LoopState,
        context: _ExecutionContext,
    ) -> SubmissionContext:
        return SubmissionContext(
            tool_results=tuple(state.get("tool_results", [])),
            source_refs=tuple(state.get("source_refs", [])),
            state_revision=state.get("progress_revision", 0),
            input_revision=context.request.input_revision,
        )

    @staticmethod
    def _model_error(error: Exception) -> dict[str, Any]:
        failure = classify_failure(error)
        return {
            "terminal_reason": failure.code,
            # 供应商或模型调用异常没有产生可接受产物，是实际系统失败；证据不足等
            # 领域提案错误在提交工具中返回，不能冒充模型服务故障。
            "terminal_status": "cancelled"
            if failure.code == "cancelled"
            else "stale"
            if failure.code == "stale_input_revision"
            else "blocked"
            if failure.category == "budget"
            else "failed",
            "error": asdict(failure),
        }

    @staticmethod
    def _record_provider_usage(budget: BudgetLedger, output: object) -> None:
        metadata = getattr(output, "response_metadata", {})
        if not isinstance(metadata, Mapping):
            return
        usage = metadata.get("token_usage") or metadata.get("usage")
        if not isinstance(usage, Mapping):
            return
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        if isinstance(input_tokens, int):
            budget.provider_input_tokens = (budget.provider_input_tokens or 0) + input_tokens
        if isinstance(output_tokens, int):
            budget.provider_output_tokens = (budget.provider_output_tokens or 0) + output_tokens

    @staticmethod
    def _recursion_limit(policy: AgentBudgetPolicy) -> int:
        model_limit = policy.emergency_max_model_steps or 64
        return max(32, model_limit * 8 + 16)


def _guard_terminal(reason: str) -> dict[str, Any]:
    status = {
        "stale_input_revision": "stale",
        "input_revision_unavailable": "failed",
        "wall_deadline": "blocked",
        "emergency_model_step_limit": "blocked",
        "tool_call_safety_limit": "blocked",
        "tool_result_context_limit": "blocked",
        "execution_attempt_limit": "blocked",
        "repeated_tool_cycle": "blocked",
        "cancelled": "cancelled",
    }.get(reason, "failed")
    return {
        "terminal_status": status,
        "terminal_reason": reason,
        "error": asdict(classify_failure(ExecutionInterrupted(reason))),
    }


def _tool_calls(output: object) -> list[dict[str, Any]]:
    calls = getattr(output, "tool_calls", [])
    if not isinstance(calls, list):
        return []
    invalid = getattr(output, "invalid_tool_calls", [])
    return [dict(item) for item in [*calls, *invalid] if isinstance(item, Mapping)]


def _content_for_model(value: object) -> object:
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "content": value.content,
            "tool_calls": _tool_calls(value),
            "additional_kwargs": value.additional_kwargs,
        }
    return getattr(value, "content", value)


def _request_context_chars(messages, context):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return len(
        json.dumps(
            {
                "messages": [_content_for_model(item) for item in messages],
                "tools": [convert_to_openai_tool(item.tool) for item in context.spec.tools],
            },
            ensure_ascii=False,
            default=str,
        )
    )


def _error_code(error: Exception) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(code, Enum) and isinstance(code.value, str) and code.value:
        return code.value
    return None


def _trace_model_name(model: object) -> str:
    candidates = [
        getattr(model, "model_name", None),
        getattr(model, "_model", None),
        getattr(getattr(model, "_client", None), "model_name", None),
        getattr(getattr(model, "_client", None), "_model", None),
    ]
    return next((value for value in candidates if isinstance(value, str) and value), "unknown")


def _terminal_status(
    value: str,
) -> Literal["succeeded", "blocked", "waiting_for_user", "cancelled", "stale", "failed"]:
    if value in {"succeeded", "blocked", "waiting_for_user", "cancelled", "stale", "failed"}:
        return cast(
            Literal["succeeded", "blocked", "waiting_for_user", "cancelled", "stale", "failed"],
            value,
        )
    return "failed"


def _tool_message_content(result: object, *, limit: int) -> str:
    """把单次工具结果限制在模型上下文内；完整诊断在 Phoenix 查看。"""

    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) <= limit:
        return serialized
    references: dict[str, object] = {}
    if isinstance(result, Mapping):
        for field_name in (
            "source_ids",
            "message_ids",
            "document_ids",
            "entity_ids",
            "relation_ids",
            "claim_ids",
            "evidence_ids",
            "version",
            "constraint_version",
            "retrieval_version",
            "graph_version",
        ):
            value = result.get(field_name)
            if isinstance(value, (str, int)):
                references[field_name] = value
            elif isinstance(value, list):
                references[field_name] = [item for item in value[:10] if isinstance(item, str)]
    envelope = {
        "truncated": True,
        "result_hash": _canonical_hash(result),
        "references": references,
    }
    envelope_chars = len(json.dumps(envelope, ensure_ascii=False, default=str))
    return json.dumps(
        {**envelope, "summary": serialized[: max(0, limit - envelope_chars - 32)]},
        ensure_ascii=False,
        default=str,
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cancelled_execution(planned: Mapping[str, Any], reason: str) -> ToolExecution:
    args = planned.get("args")
    return ToolExecution(
        call_id=str(planned.get("call_id") or uuid4().hex),
        tool_name=str(planned.get("name", "unknown")),
        args=args if isinstance(args, dict) else {},
        result={
            "error": reason,
            "source_ids": [],
            "failure": asdict(classify_failure(ExecutionInterrupted(reason))),
        },
        status="cancelled",
        progress=ProgressDelta(summary="调用因取消或输入更新被丢弃"),
    )


def _bounded_model(model, state, context):
    remaining = context.spec.budget.max_wall_seconds - (time() - state["started_wall"])
    if remaining <= 0:
        raise TimeoutError("Agent deadline exceeded")
    with_timeout = getattr(model, "with_timeout_seconds", None)
    return with_timeout(max(0.001, remaining)) if callable(with_timeout) else model
