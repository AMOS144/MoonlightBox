"""PersonWorld 纠正阶段的领域适配器。

本模块只把纠正领域的 Prompt、结构化编译器与 LangChain 工具翻译成 ``AgentSpec``。
模型行动、工具批次、去重、checkpoint、取消和终止全部由统一的
``AgentLoopController`` 负责；这里不再维护第二套 LangGraph 子图。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from sqlalchemy import select
from sqlalchemy.orm import object_session

from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.context import retain_recent_turns
from moonlightbox.agent_runtime.contracts import (
    AgentExecutionRequest,
    ProgressDelta,
    RegisteredTool,
    ToolContract,
)
from moonlightbox.agent_runtime.policy import (
    RevisionAgentRuntimePolicy,
    model_resilience,
    revision_budget,
)
from moonlightbox.agent_runtime.submission import submission_instruction
from moonlightbox.world.compiler import AgentCompilerClient
from moonlightbox.world.models import (
    PersonWorldProfile,
    PersonWorldRevisionSession,
    WorldGraphVersion,
)

from ..investigation_artifacts import InvestigationArtifactStore
from ..prompt_loader import PersonWorldPromptDefinition, validate_registered_tools
from ..schemas import RevisionAgentTurn
from ..tools.comparison import comparison_for
from ..tools.execution_policy import execution_policy
from .context import AssembledRevisionContext, RevisionContextAssembler
from .submit_turn import build_submit_revision_turn


@dataclass(frozen=True, slots=True)
class RevisionGraphExecution:
    """一次探索回合的可持久化结果。

    ``snapshot`` 是最终决定用户可见回合时 Agent 实际读取的冻结上下文；工具错误只保留
    有界的安全摘要，不能被解释成“没有历史证据”。
    """

    turn: RevisionAgentTurn
    snapshot: AssembledRevisionContext
    trace: tuple[dict[str, object], ...]
    tool_errors: tuple[dict[str, object], ...]


class RevisionAgentGraph:
    """纠正探索 Agent 的有限调查图。

    Agent 自主选择只读调查工具，通过提交工具交付一个用户可见回合。
    用户批准阶段和图谱写入权限仍由业务服务控制。
    """

    def __init__(
        self,
        *,
        definition: PersonWorldPromptDefinition,
        tools: Mapping[str, BaseTool],
        compiler: AgentCompilerClient,
        context_assembler: RevisionContextAssembler,
        revision: PersonWorldRevisionSession,
        graph: WorldGraphVersion,
        profile: PersonWorldProfile | None,
        current_profile: dict[str, object],
        conversation: list[dict[str, str]],
        selected_statements: list[dict[str, object]],
        explicit_source_ids: list[str],
        on_stage: Callable[[str], None] | None = None,
        runtime_policy: RevisionAgentRuntimePolicy | None = None,
        controller: AgentLoopController | None = None,
        artifacts: InvestigationArtifactStore | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        on_status: Callable[[dict], None] | None = None,
    ) -> None:
        self.definition = definition
        self.compiler = compiler
        self.context_assembler = context_assembler
        self.revision = revision
        self.graph = graph
        self.profile = profile
        self.current_profile = current_profile
        self.conversation = conversation
        self.selected_statements = selected_statements
        self.explicit_source_ids = list(dict.fromkeys(explicit_source_ids))
        # 仅报告无敏感正文的节点名。Job 层据此持久化 checkpoint；它不参与模型判断，
        # 也不把大段 Tool 返回或 LangGraph State 写进 jobs 表。
        self._on_stage = on_stage
        self._cancellation_requested = cancellation_requested
        self._on_status = on_status
        self.runtime_policy = runtime_policy or RevisionAgentRuntimePolicy()
        names = validate_registered_tools(definition, tools)
        self.tools = {name: tools[name] for name in names}
        self.controller = controller or AgentLoopController()
        # 与本次纠正回合绑定的只读检索/原文恢复桥接；不持久化、不跨用户回合复用。
        self.artifacts = artifacts or InvestigationArtifactStore()

    def run(self) -> RevisionGraphExecution:
        """以统一 Harness 执行一次可中断的 Revision 回合。"""

        self._report_stage("context_assembling")
        adapter = _RevisionModelAdapter(self)
        import hashlib

        from ..prompt_loader import load_temporal_protocol
        system = self.definition.system_prompt
        if self.revision.scope.get("node_scope"):
            system += "\n\n" + load_temporal_protocol()
        result = self.controller.run(
            spec=AgentSpec(
                name="person_world.revision",
                prompt_version=hashlib.sha256(system.encode()).hexdigest(),
                submission_tool_name="submit_revision_turn",
                budget=revision_budget(self.runtime_policy),
                resilience=model_resilience(
                    adapter, timeout_seconds=self.runtime_policy.model_request_timeout_seconds
                ),
                tools=tuple(
                    RegisteredTool(
                        tool=tool,
                        contract=ToolContract(
                            name=tool.name,
                            comparison_projection=comparison_for(tool.name, self.artifacts),
                            execution=execution_policy(tool.name),
                            progress_evaluator=_revision_progress,
                        ),
                    )
                    for tool in self.tools.values()
                )
                + (build_submit_revision_turn(),),
                context_compactor=_compact_revision_context,
                snapshot_work_state=adapter.snapshot_work,
                restore_work_state=adapter.restore_work,
            ),
            request=AgentExecutionRequest(
                checkpoint_path=self._checkpoint_path(),
                owner_type="revision",
                cancellation_requested=self._is_cancelled,
                on_status=self._on_status,
                owner_id=self.revision.id,
                project_id=self.revision.project_id,
                input_revision=self.revision.input_revision,
                scope=RunScope(
                    project_id=self.revision.project_id,
                    graph_read_version=self.graph.id,
                    permissions=frozenset({"person_world.revision.read"}),
                ),
                messages=(
                    SystemMessage(
                        content=system
                        + "\n\n"
                        + submission_instruction("submit_revision_turn")
                    ),
                    HumanMessage(content="根据本次纠正范围调查并提交一个用户可见回合。"),
                ),
                input_revision_resolver=self._current_session_revision,
            ),
            model=adapter,
        )
        turn = result.value
        assembled = adapter.assembled
        if not isinstance(turn, RevisionAgentTurn) or not isinstance(
            assembled, AssembledRevisionContext
        ):
            raise RuntimeError("纠正 Agent 没有形成可展示的协议回合")
        assembled = self.context_assembler.persist_snapshot(assembled, self.revision)
        self._report_stage("turn_validating")
        messages = [item for item in result.messages if isinstance(item, (AIMessage, ToolMessage))]
        return RevisionGraphExecution(
            turn=turn,
            snapshot=assembled,
            # 调用细节由 Phoenix 承担；Revision 只持久化用户可见的回合和 Patch 状态。
            trace=(),
            tool_errors=tuple(_tool_errors(messages)),
        )

    def _report_stage(self, stage: str) -> None:
        """把安全阶段交给 Job；Prompt、工具正文和模型思维都不进入 Job checkpoint。"""

        if self._on_stage is not None:
            self._on_stage(stage)

    def _current_session_revision(self) -> int | None:
        """以数据库当前值保护 Revision；不复用可能已过期的 ORM 属性。"""

        session = object_session(self.revision)
        if session is None:
            return None
        value = session.scalar(
            select(PersonWorldRevisionSession.input_revision).where(
                PersonWorldRevisionSession.id == self.revision.id
            )
        )
        return int(value) if isinstance(value, int) else None

    def _checkpoint_path(self):
        from moonlightbox.agent_runtime.persistence import checkpoint_path

        return checkpoint_path(self.context_assembler.session)

    def _is_cancelled(self):
        if self._cancellation_requested and self._cancellation_requested():
            return True
        from sqlalchemy.orm import Session

        session = self.context_assembler.session
        query = select(PersonWorldRevisionSession.status).where(
            PersonWorldRevisionSession.id == self.revision.id,
        )
        if session.get_bind().url.database in {None, "", ":memory:"}:
            # 内存 SQLite 的同一连接不能交给另一个 Session 关闭/回滚。
            return session.scalar(query) in {"cancelled", "stale"}
        with Session(session.get_bind()) as check:
            status = check.scalar(query)
            return status in {"cancelled", "stale"}


class _RevisionModelAdapter:
    """保留纠正上下文装配，调查和交付均由同一个原生工具模型完成。"""

    has_managed_request_boundaries = True

    @property
    def agent_resilience(self):
        return getattr(self.model, "agent_resilience", None)

    def __init__(self, graph):
        self.graph = graph
        self.model = graph.compiler.create_agent_chat_model()
        self.shared = {}

    @property
    def assembled(self):
        return self.shared.get("assembled")

    def snapshot_work(self):
        assembled = self.assembled
        return {
            "artifacts": self.graph.artifacts.snapshot(),
            "assembled": None
            if assembled is None
            else {
                "model_payload": assembled.model_payload,
                "snapshot": {
                    column.name: getattr(assembled.snapshot, column.name)
                    for column in assembled.snapshot.__table__.columns
                },
            },
        }

    def restore_work(self, state):
        from moonlightbox.world.models import PersonWorldRevisionContextSnapshot

        self.graph.artifacts.restore(state["artifacts"])
        assembled = state.get("assembled")
        if assembled is not None:
            self.shared["assembled"] = AssembledRevisionContext(
                snapshot=PersonWorldRevisionContextSnapshot(**assembled["snapshot"]),
                model_payload=assembled["model_payload"],
            )

    def bind_tools(self, tools, **kwargs):
        from copy import copy

        bound = copy(self)
        bound.model = self.model.bind_tools(tools, **kwargs)
        return bound

    def invoke(self, messages):
        self.graph._report_stage("context_assembling")
        searches = _tool_payloads_for_name(messages, "search_world")
        references = [
            reference
            for payload in searches
            if isinstance(payload, dict)
            for reference in payload.get("references", [])
            if isinstance(reference, dict)
        ]
        related_graph_context = "\n\n".join(
            str(payload.get("context", ""))
            for payload in searches
            if isinstance(payload, dict) and isinstance(payload.get("context"), str)
        )
        self.shared["assembled"] = self.graph.context_assembler.assemble(
            revision=self.graph.revision,
            persist=False,
            graph=self.graph.graph,
            profile=self.graph.profile,
            current_profile=self.graph.current_profile,
            conversation=self.graph.conversation,
            selected_statements=self.graph.selected_statements,
            source_messages=_dedupe_rows(_source_rows(messages, self.graph.artifacts)),
            related_graph_context=related_graph_context,
            graph_references=references,
        )

        self.graph._report_stage("agent_deliberating")
        payload = HumanMessage(
            content=json.dumps(
                {
                    "revision_context": self.assembled.model_payload,
                    "explicit_source_ids": self.graph.explicit_source_ids,
                },
                ensure_ascii=False,
            )
        )
        return self.model.invoke([messages[0], payload, *messages[2:]])


def _revision_progress(
    result: object,
    *,
    state_revision: int,
    seen_keys: frozenset[str],
) -> ProgressDelta:
    keys = _revision_result_keys(result)
    new_keys = frozenset(item for item in keys if item not in seen_keys)
    return ProgressDelta(
        new_keys=new_keys,
        summary="获得新的纠正调查材料" if new_keys else "本次工具没有新调查材料",
    )


def _revision_result_keys(value: object) -> set[str]:
    fields = {
        "message_id",
        "source_id",
        "document_id",
        "bundle_id",
        "reference_id",
        "document_name",
        "entity_name",
        "entity_id",
    }
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name in fields and isinstance(item, (str, int)):
                keys.add(f"{name}:{item}")
            elif name in {"message_ids", "primary_message_ids"} and isinstance(item, list):
                keys.update(f"message_id:{value}" for value in item if isinstance(value, str))
            elif isinstance(item, (dict, list)):
                keys.update(_revision_result_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_revision_result_keys(item))
    return keys


def _compact_revision_context(
    messages: list[AnyMessage],
    *,
    source_refs: list[str],
    unresolved: list[str],
) -> tuple[list[AnyMessage], str]:
    summary = HumanMessage(
        content=json.dumps(
            {
                "revision_runtime_summary": {
                    "source_refs": source_refs[-300:],
                    "unresolved": unresolved[-50:],
                    "instruction": "需要原文时使用受范围限制的只读消息工具重新读取。",
                }
            },
            ensure_ascii=False,
        )
    )
    return retain_recent_turns(messages, notes=(summary,)), "纠正调查上下文已压缩并保留来源引用"


def _tool_payloads(messages: list[AnyMessage]) -> list[object]:
    return [
        _parse_tool_content(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    ]


def _tool_payloads_for_name(messages: list[AnyMessage], name: str) -> list[object]:
    return [
        _parse_tool_content(message.content)
        for message in messages
        if isinstance(message, ToolMessage) and message.name == name
    ]


def _parse_tool_content(content: object) -> object:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"tool_error": content[:1000]}


def _source_rows(
    messages: list[AnyMessage],
    artifacts: InvestigationArtifactStore | None = None,
) -> list[dict[str, object]]:
    if artifacts is not None:
        rows = list(artifacts.evidence_rows())
        if rows:
            return rows
    rows: list[dict[str, object]] = []
    for payload in _tool_payloads(messages):
        if not isinstance(payload, list):
            continue
        rows.extend(
            item
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("message_id"), str)
        )
    return rows


def _dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        message_id = row.get("message_id")
        if not isinstance(message_id, str) or message_id in seen:
            continue
        seen.add(message_id)
        result.append(row)
    return result


def _tool_errors(messages: list[AnyMessage]) -> list[dict[str, object]]:
    """把运行时捕获的工具失败限制为操作名和短诊断，避免泄露完整异常。"""

    errors: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = _parse_tool_content(message.content)
        if not isinstance(payload, dict) or not isinstance(payload.get("tool_error"), str):
            continue
        errors.append(
            {
                "tool_name": message.name,
                "diagnostic": payload["tool_error"][:500],
            }
        )
    return errors
