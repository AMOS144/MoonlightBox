"""Runtime v1 最小冒烟测试：只验证关键边界，不依赖真实模型或显卡。"""

import json
from datetime import UTC, datetime, timedelta
from threading import Lock
from time import sleep

import httpx
import moonlightbox.api  # noqa: F401  # 注册全部 ORM 表
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from moonlightbox.config import Settings
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.runtime_v1.agent_catalog import (
    load_agent_definition,
    parse_agent_markdown,
    select_declared_tools,
)
from moonlightbox.runtime_v1.branch_models import Branch
from moonlightbox.runtime_v1.clock import create_clock, pause_clock, resume_clock
from moonlightbox.runtime_v1.cloud_models import RuntimeCloudChatModel, RuntimeCloudClient
from moonlightbox.runtime_v1.context_views import day_plan_context_payload
from moonlightbox.runtime_v1.day_planner import DayPlanAgent
from moonlightbox.runtime_v1.db_models import (
    RuntimeCycleTraceRow,
    RuntimeEventRow,
    RuntimeMemoryIndexRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from moonlightbox.runtime_v1.jobs import enqueue_due_runtime_cycles
from moonlightbox.runtime_v1.memory import MemoryService
from moonlightbox.runtime_v1.plan_context import DayPlanContext
from moonlightbox.runtime_v1.service import RuntimeModelExecutionError, RuntimeService
from moonlightbox.runtime_v1.source_history import SourceHistoryService
from moonlightbox.runtime_v1.tools.memory_search import SearchMemoryArgs
from moonlightbox.runtime_v1.tools.plan_constraints import GetPlanConstraintsArgs
from moonlightbox.runtime_v1.tools.routine_evidence import (
    AnalyzeRoutineEvidenceArgs,
    build_routine_evidence_tool,
)
from moonlightbox.runtime_v1.tools.style_examples import StyleService
from moonlightbox.training.models import ModelVersion
from moonlightbox.world.client import LightRAGMetadata, LightRAGReference, LightRAGRetrieval
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    WorldGraphVersion,
)
from sqlalchemy import select
from sqlalchemy.orm import Session
from submission_helpers import submission_message


def test_clock_smoke() -> None:
    anchor = datetime(2026, 9, 1, 8, tzinfo=UTC)
    clock = create_clock("branch", anchor, wall_anchor=anchor)
    assert clock.now(anchor + timedelta(hours=2)) == datetime(2026, 9, 1, 10, tzinfo=UTC)
    # SQLite 会丢失 tzinfo；读取后的锚点仍须按 UTC 计算，不能受部署机时区影响。
    sqlite_clock = clock.model_copy(
        update={
            "virtual_anchor": anchor.replace(tzinfo=None),
            "wall_anchor": anchor.replace(tzinfo=None),
        }
    )
    assert sqlite_clock.now(anchor + timedelta(hours=2)) == datetime(2026, 9, 1, 10, tzinfo=UTC)
    assert pause_clock(clock, wall_now=anchor + timedelta(hours=1)).status == "paused"
    assert resume_clock(clock, wall_now=anchor + timedelta(hours=1)).status == "running"


def test_agent_prompts_and_native_tool_whitelists_are_separate() -> None:
    """Markdown 只声明能力边界；schema 与运行熔断器都不混入行为 Prompt。"""

    actor = load_agent_definition("persona_actor")
    planner = load_agent_definition("day_planner")

    assert set(actor.tool_names) == {
        "get_style_examples",
        "read_conversation",
        "get_profile_section",
        "submit_expression",
    }
    assert set(planner.tool_names) == {
        "send_agent_message",
        "submit_day_plan",
        "update_plan_work",
        "get_plan_constraints",
        "analyze_routine_evidence",
        "search_plan_memory",
        "get_recent_life_events",
    }
    combined = (
        load_agent_definition("director").system_prompt
        + actor.system_prompt
        + planner.system_prompt
    )
    assert "<tool_protocol>" not in combined
    assert "args_schema" not in combined
    assert "branch_id" not in combined
    assert not hasattr(planner, "timeout_seconds")
    try:
        parse_agent_markdown(
            "persona_actor",
            "---\ndescription: x\ntools: [get_style_examples]\nmax_rounds: 2\n---\n正文",
        )
    except ValueError as error:
        assert "未知配置" in str(error)
    else:
        raise AssertionError("Prompt frontmatter 不得重新声明运行轮数")

    tool = StructuredTool.from_function(
        name="get_style_examples",
        description="测试工具",
        func=lambda **_kwargs: {},
        args_schema=SearchMemoryArgs,
    )
    assert select_declared_tools(actor, [tool]) == [tool]


def test_director_blank_datetime_is_repairable_tool_parameter_error():
    from types import SimpleNamespace

    from moonlightbox.runtime_v1.tools.submit_decision import build_submit_decision_tool
    from pydantic import ValidationError

    tool = build_submit_decision_tool(SimpleNamespace()).tool
    with pytest.raises(ValidationError):
        tool.invoke(
            {
                "result": {
                    "action": "continue_life",
                    "next_wakeup_at": "",
                    "state_patch": {"valid_until": ""},
                }
            }
        )


def test_minimax_m3_runtime_client_uses_compatible_json_transport_and_retries_empty_final() -> None:
    """M3 不接受 response_format；空的最终回答应在有限次数内安全重试。"""

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = "" if len(requests) == 1 else '{"action":"wait"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = RuntimeCloudClient(
        endpoint="https://api.minimax.io/v1/chat/completions",
        model="MiniMax-M3",
        api_key="test-key",
        timeout_seconds=5,
        thinking_mode="disabled",
        max_output_tokens=128,
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )

    assert (
        client.complete(system_prompt="只返回 JSON", messages=[], temperature=0.1).content
        == '{"action":"wait"}'
    )
    assert len(requests) == 2
    body = json.loads(requests[0].content)
    assert "response_format" not in body
    assert body["max_completion_tokens"] == 128
    assert body["reasoning_split"] is True
    assert body["thinking"] == {"type": "disabled"}


def test_cloud_model_sends_native_tools_and_replays_complete_tool_round() -> None:
    """schema、assistant.tool_calls、reasoning_details 与 tool_call_id 都走原生协议。"""

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_details": [{"type": "text", "text": "需要核实"}],
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_memory",
                                        "arguments": '{"scope":"world","query":"周末作息"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = RuntimeCloudClient(
        endpoint="https://api.minimax.io/v1/chat/completions",
        model="MiniMax-M3",
        api_key="test-key",
        timeout_seconds=5,
        thinking_mode="default",
        max_output_tokens=128,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    search = StructuredTool.from_function(
        name="search_memory",
        description="查询记忆证据",
        func=lambda **_kwargs: {},
        args_schema=SearchMemoryArgs,
    )
    model = RuntimeCloudChatModel(client, temperature=0.1).bind_tools([search], strict=True)
    first = model.invoke([SystemMessage(content="Director"), HumanMessage(content="核实周末作息")])

    assert first.tool_calls[0]["name"] == "search_memory"
    assert first.tool_calls[0]["id"] == "call-1"
    assert first.additional_kwargs["reasoning_details"][0]["text"] == "需要核实"
    body = json.loads(requests[0].content)
    assert body["tools"][0]["function"]["name"] == "search_memory"
    assert body["tool_choice"] == "auto"

    # 只验证第二轮请求体；响应仍可再次调用工具，不影响消息回放断言。
    model.invoke(
        [
            SystemMessage(content="Director"),
            HumanMessage(content="核实周末作息"),
            first,
            ToolMessage(
                content='{"source_ids":["message-1"],"data":[]}',
                tool_call_id="call-1",
                name="search_memory",
            ),
        ]
    )
    replay = json.loads(requests[1].content)["messages"]
    assert replay[2]["tool_calls"][0]["id"] == "call-1"
    assert replay[2]["reasoning_details"][0]["text"] == "需要核实"
    assert replay[3]["role"] == "tool"
    assert replay[3]["tool_call_id"] == "call-1"


def test_memory_search_hides_non_auditable_legacy_source_labels(tmp_path) -> None:
    """旧 Profile 摘要可读取，但日期文本不能冒充 DayPlan 的消息证据。"""

    database = Database(f"sqlite:///{tmp_path / 'memory-evidence.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 5, 9, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.flush()
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        session.flush()
        session.add(
            RuntimeSnapshotRow(
                id="s",
                branch_id="b",
                cutoff_at=now,
                source_message_ids=["imported-message"],
                profile={},
                routine_profile={},
            )
        )
        session.add(
            RuntimeMemoryRow(
                id="record",
                scope="world",
                snapshot_id="s",
                subject="目标人物",
                predicate="routine",
                object="周末休息",
                summary="周末会休息",
                status="confirmed",
                source_ids=["imported-message", "2026-05-11 16:58", "相关对话"],
            )
        )
        session.commit()

        result = MemoryService(session).search(
            branch_id="b",
            snapshot_id="s",
            args=SearchMemoryArgs(scope="world", query="周末作息", limit=1),
        )

        assert result["source_ids"] == ["imported-message"]
        assert result["data"][0]["source_ids"] == ["imported-message"]
    database.close()


def test_branch_anchor_uses_selected_node_start_time(tmp_path) -> None:
    """批准边界决定初始钟点，不再从旧事件或训练产物读取时间。"""
    from moonlightbox.runtime_v1.profile_projection import _runtime_time_anchor

    database = Database(f"sqlite:///{tmp_path / 'branch-anchor.db'}")
    Base.metadata.create_all(database.engine)
    node_started_at = datetime(2026, 5, 8, 11, 40, 13, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        branch = Branch(
            project_id="p",
            title="分支",
            origin_time=node_started_at + timedelta(days=3),
            origin_boundary={
                "cutoff_at": node_started_at.isoformat(),
                "timezone": "Asia/Shanghai",
            },
        )
        session.add(branch)
        session.commit()
        session.refresh(branch)
        anchor, timezone = _runtime_time_anchor(session, branch)
        assert anchor == node_started_at
        assert timezone == "Asia/Shanghai"
        assert branch.origin_event_id is None and branch.model_version_id is None
    database.close()


class _FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def bind_tools(self, _tools: list[object], **_kwargs: object) -> "_FakeModel":
        return self

    def invoke(self, _messages: list[object]) -> AIMessage:
        self.calls += 1
        from test_peer_collaboration import with_pending_inputs

        return submission_message(
            content=json.dumps(with_pending_inputs(json.loads(self.content), _messages))
        )

    def count_text_tokens(self, text: str) -> int:
        """测试替身显式提供 tokenizer 契约，生产代码不得退回字符数估算。"""

        return max(1, len(text.split()))


class _SequentialPlannerModel:
    """让冒烟测试覆盖一次受限的 Planner 协议修复循环。"""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def bind_tools(self, _tools: list[object], **_kwargs: object) -> "_SequentialPlannerModel":
        return self

    def invoke(self, _messages: list[object]) -> AIMessage:
        return submission_message(content=self.outputs.pop(0))


class _ToolCallingPlannerModel:
    """先发起原生 tool call，再根据 ToolMessage 返回计划。"""

    def __init__(self) -> None:
        self.calls = 0
        self.saw_tool_message = False

    def bind_tools(self, _tools: list[object], **_kwargs: object) -> "_ToolCallingPlannerModel":
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls += 1
        self.saw_tool_message = any(isinstance(item, ToolMessage) for item in messages)
        if self.calls == 1:
            return submission_message(
                content="",
                tool_calls=[
                    {
                        "name": "get_plan_constraints",
                        "args": {},
                        "id": "constraints-1",
                    }
                ],
            )
        return submission_message(
            content=(
                '{"plan_date":"2026-05-09","blocks":[{"start":"00:00",'
                '"end":"24:00","activity":"个人时间","location_role":null,'
                '"default_availability":"available","basis":"fallback",'
                '"evidence_ids":[],"confidence":"fallback"}]}'
            )
        )


class _TwoToolCallingPlannerModel(_ToolCallingPlannerModel):
    """同一回合请求两个工具，用于锁定 SQLAlchemy Session 的串行边界。"""

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls += 1
        self.saw_tool_message = any(isinstance(item, ToolMessage) for item in messages)
        if self.calls == 1:
            return submission_message(
                content="",
                tool_calls=[
                    {"name": "get_plan_constraints", "args": {}, "id": "constraints-1"},
                    {"name": "analyze_routine_evidence", "args": {}, "id": "routine-1"},
                ],
            )
        return submission_message(
            content=(
                '{"plan_date":"2026-05-09","blocks":[{"start":"00:00",'
                '"end":"24:00","activity":"个人时间","location_role":null,'
                '"default_availability":"available","basis":"fallback",'
                '"evidence_ids":[],"confidence":"fallback"}]}'
            )
        )


class _EvidenceRequiredPlannerModel:
    """先跳过证据，再响应 Runtime 的强制分析提示。"""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(
        self, _tools: list[object], **_kwargs: object
    ) -> "_EvidenceRequiredPlannerModel":
        return self

    def invoke(self, _messages: list[object]) -> AIMessage:
        self.calls += 1
        if self.calls == 2:
            return submission_message(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_routine_evidence",
                        "args": {
                            "question": "目标人物在当前工作阶段通常怎样安排时间？",
                        },
                        "id": "routine-required",
                    }
                ],
            )
        evidence = '["message-1"]' if self.calls >= 3 else "[]"
        basis = "historical_pattern" if self.calls >= 3 else "fallback"
        confidence = "inferred" if self.calls >= 3 else "fallback"
        return submission_message(
            content=(
                '{"plan_date":"2026-05-09","blocks":[{"start":"00:00",'
                '"end":"24:00","activity":"工作日安排","location_role":null,'
                f'"default_availability":"busy","basis":"{basis}",'
                f'"evidence_ids":{evidence},"confidence":"{confidence}"}}]}}'
            )
        )


def test_day_planner_repairs_one_invalid_structured_output() -> None:
    now = datetime(2026, 5, 9, 8, tzinfo=UTC)
    model = _SequentialPlannerModel(
        [
            '{"plan_date":"2026-05-09","blocks":[{"start":"00:00","end":"24:00",'
            '"activity":"臆造规律","location_role":null,"default_availability":"available",'
            '"basis":"snapshot_routine","evidence_ids":["untrusted-source"],'
            '"confidence":"inferred"}]}',
            '已按协议生成：\n```json\n{"plan_date":"2026-05-09","blocks":[{"start":"00:00","end":"24:00",'
            '"activity":"个人时间","location_role":null,"default_availability":"available",'
            '"basis":"fallback","evidence_ids":[],"confidence":"fallback"}]}\n```',
        ]
    )
    run = DayPlanAgent(model).run_with_trace(
        DayPlanContext(
            generated_at=now,
            virtual_now=now,
            branch_id="b",
            target_date=now.date(),
            timezone="UTC",
            mode="initial",
            request={},
            hard_constraints={},
            origin_projection={},
            date_features={},
            branch_evidence={},
            budget={"max_blocks": 12, "minimum_granularity_minutes": 30},
        ),
        tools=[],
    )

    assert run.proposal is not None
    assert run.proposal.plan_date == now.date()
    assert run.terminal_reason == "success"


def test_day_planner_executes_declared_tool_through_langgraph_tool_node() -> None:
    now = datetime(2026, 5, 9, 8, tzinfo=UTC)
    invoked = 0

    def constraints() -> dict[str, object]:
        nonlocal invoked
        invoked += 1
        return {"source_ids": [], "data": {"active_commitments": []}}

    tool = StructuredTool.from_function(
        name="get_plan_constraints",
        description="读取计划约束",
        func=constraints,
        args_schema=GetPlanConstraintsArgs,
    )
    model = _ToolCallingPlannerModel()
    run = DayPlanAgent(model).run_with_trace(
        DayPlanContext(
            generated_at=now,
            virtual_now=now,
            branch_id="b",
            target_date=now.date(),
            timezone="UTC",
            mode="initial",
            request={},
            hard_constraints={},
            origin_projection={},
            date_features={},
            branch_evidence={},
            budget={"max_blocks": 12},
        ),
        tools=[tool],
    )

    assert run.proposal is not None
    assert invoked == 1
    assert model.saw_tool_message is True
    assert run.terminal_reason == "success"


def test_day_planner_serializes_tools_that_share_database_session() -> None:
    now = datetime(2026, 5, 9, 8, tzinfo=UTC)
    lock = Lock()
    active = 0
    maximum_active = 0

    def read_database() -> dict[str, object]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.03)
        with lock:
            active -= 1
        return {"source_ids": [], "data": {}}

    tools = [
        StructuredTool.from_function(
            name=name,
            description="读取同一个数据库会话",
            func=read_database,
            args_schema=GetPlanConstraintsArgs,
        )
        for name in ("get_plan_constraints", "analyze_routine_evidence")
    ]
    run = DayPlanAgent(_TwoToolCallingPlannerModel()).run_with_trace(
        DayPlanContext(
            generated_at=now,
            virtual_now=now,
            branch_id="b",
            target_date=now.date(),
            timezone="UTC",
            mode="initial",
            request={},
            hard_constraints={},
            origin_projection={},
            date_features={},
            branch_evidence={},
            budget={"max_blocks": 12},
        ),
        tools=tools,
    )

    assert run.proposal is not None
    assert maximum_active == 1


def test_day_planner_can_submit_without_forced_retrieval() -> None:
    """调查路径交给 Agent，不能用必须调用指定工具卡住可接受提案。"""

    now = datetime(2026, 5, 9, 8, tzinfo=UTC)
    tool = StructuredTool.from_function(
        name="analyze_routine_evidence",
        description="语义分析作息",
        func=lambda **_kwargs: {
            "tool_name": "analyze_routine_evidence",
            "source_ids": ["message-1"],
            "data": {
                "retrieval_status": "ok",
                "messages": [{"source_id": "message-1", "content": "通常十点上班"}],
            },
        },
        args_schema=AnalyzeRoutineEvidenceArgs,
    )
    model = _EvidenceRequiredPlannerModel()
    run = DayPlanAgent(model).run_with_trace(
        DayPlanContext(
            generated_at=now,
            virtual_now=now,
            branch_id="b",
            target_date=now.date(),
            timezone="UTC",
            mode="initial",
            request={},
            hard_constraints={},
            origin_projection={},
            date_features={"calendar_verified": True, "is_workday": True},
            branch_evidence={},
            evidence_requirements={"required_topics": ["work"]},
            budget={"max_blocks": 24, "minimum_granularity_minutes": 15},
        ),
        tools=[tool],
    )

    assert run.proposal is not None
    assert model.calls == 1
    assert run.terminal_reason == "success"


def test_day_plan_model_context_hides_persistence_ids() -> None:
    now = datetime(2026, 5, 9, 8, tzinfo=UTC)
    context = DayPlanContext(
        generated_at=now,
        virtual_now=now,
        branch_id="internal-branch",
        target_date=now.date(),
        timezone="UTC+08:00",
        mode="revision",
        request={"reason": "调整上午", "target_date": now.date().isoformat()},
        hard_constraints={
            "current_plan": {
                "id": "plan-row",
                "version": 3,
                "blocks": [
                    {
                        "id": "block-row",
                        "start": "00:00",
                        "end": "24:00",
                        "activity": "休息",
                    }
                ],
            },
            "current_life_state": {"current_plan_block_id": "block-row", "activity": "休息"},
        },
        origin_projection={"snapshot_id": "snapshot-row", "routine_summary": {}},
        date_features={"is_weekday": False},
        branch_evidence={},
        budget={"max_blocks": 12},
    )

    payload = day_plan_context_payload(context)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["plan_date"] == "2026-05-09"
    assert payload["local_now"] == "2026-05-09T16:00:00+08:00"
    assert "internal-branch" not in serialized
    assert "plan-row" not in serialized
    assert "block-row" not in serialized
    assert "snapshot-row" not in serialized
    assert "target_date" not in serialized


class _RoleAwarePlannerModel(_FakeModel):
    """同一认知模型在不同 Agent 协议下应返回各自的结构化输出。"""

    def __init__(self, director_content: str, planner_content: str) -> None:
        super().__init__(director_content)
        self.planner_content = planner_content

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls += 1
        first = messages[0] if messages else None
        system = str(getattr(first, "content", ""))
        content = self.content
        if "# DayPlanAgent" not in system:
            value = json.loads(content)
            if value.get("plan_request"):
                # 同级图提交后再次调用 Director；离线模型也必须消费新回执。
                if getattr(self, "requested", False):
                    value["plan_request"] = None
                else:
                    self.requested = True
                content = json.dumps(value)
        from test_peer_collaboration import with_pending_inputs

        return submission_message(
            content=(
                self.planner_content
                if "# DayPlanAgent" in system
                else json.dumps(with_pending_inputs(json.loads(content), messages))
            )
        )


class _UnavailableModel:
    """模拟云端模型不可用，验证用户消息不会被静默标为已处理。"""

    def bind_tools(self, _tools: list[object], **_kwargs: object) -> "_UnavailableModel":
        return self

    def invoke(self, _messages: list[object]) -> AIMessage:
        raise RuntimeError("simulated provider outage")


class _FakeLightRAG:
    """不访问 Sidecar；记录 Runtime 实际提交的动态检索问题。"""

    def __init__(self, context: str) -> None:
        self.context = context
        self.queries: list[str] = []

    def query(self, _workspace: str, query: str, **_kwargs: object) -> LightRAGRetrieval:
        self.queries.append(query)
        return LightRAGRetrieval(
            context=self.context,
            references=[LightRAGReference(file_path="bundle-1.txt", content=self.context)],
            metadata=LightRAGMetadata(
                lightrag_version="test",
                embedding_model="test",
                embedding_dimension=1,
                extraction_model="test",
                chunking_strategy="test",
                chunk_token_size=1,
                chunk_overlap_token_size=0,
                entity_prompt_version="test",
            ),
        )


@pytest.mark.parametrize("graph_status", ["ready", "superseded"])
def test_routine_tool_uses_semantic_windows_and_event_times(tmp_path, graph_status) -> None:
    """工具依赖 LightRAG 语义窗口，不把 17:11 的发送时间误当上班时间。"""

    database = Database(f"sqlite:///{tmp_path / 'routine-evidence.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 5, 12, 8, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        session.add(
            ImportSource(
                id="import",
                project_id="p",
                preview_id="preview",
                source_path="/tmp/chat.txt",
                message_count=3,
                confirmed_at=now,
            )
        )
        session.add_all(
            [
                Participant(id="self", project_id="p", name="我", role="self"),
                Participant(id="target", project_id="p", name="她", role="target"),
            ]
        )
        session.flush()
        messages = [
            Message(
                id="work-end",
                project_id="p",
                import_id="import",
                participant_id="target",
                source_id="1",
                timestamp=datetime(2026, 4, 22, 17, 10),
                kind="text",
                content="下班了",
                raw={},
            ),
            Message(
                id="work-start",
                project_id="p",
                import_id="import",
                participant_id="target",
                source_id="2",
                timestamp=datetime(2026, 4, 22, 17, 11),
                kind="text",
                content="我早上十点才上班",
                raw={},
            ),
            Message(
                id="relative-end",
                project_id="p",
                import_id="import",
                participant_id="target",
                source_id="3",
                timestamp=datetime(2026, 5, 11, 14, 49),
                kind="text",
                content="还要两小时即可下班",
                raw={},
            ),
        ]
        session.add_all(messages)
        session.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="3",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.flush()
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        session.add(
            WorldGraphVersion(
                id="graph",
                project_id="p",
                trigger_import_id="import",
                workspace_key="world_test",
                status=graph_status,
                source_fingerprint="source",
                config_fingerprint="config",
                source_import_ids=["import"],
                compiler_version="test",
            )
        )
        session.flush()
        bundle_content = (
            "2026-04-22 17:10 她：下班了\n"
            "2026-04-22 17:11 她：我早上十点才上班\n"
            "2026-05-11 14:49 她：还要两小时即可下班"
        )
        bundle = ConversationBundle(
            project_id="p",
            graph_version_id="graph",
            document_id="bundle-1",
            source_name="bundle-1.txt",
            ordinal=0,
            started_at=messages[0].timestamp,
            ended_at=messages[-1].timestamp,
            content=bundle_content,
            content_hash="bundle",
            primary_message_count=3,
        )
        session.add(bundle)
        session.flush()
        for ordinal, message in enumerate(messages):
            session.add(
                ConversationBundleMessage(
                    bundle_id=bundle.id,
                    message_id=message.id,
                    ordinal=ordinal,
                )
            )
        snapshot = RuntimeSnapshotRow(
            id="snapshot",
            branch_id="b",
            graph_version_id="graph",
            cutoff_at=now,
            timezone="UTC",
            snapshot_mode="latest_profile",
            source_message_ids=[item.id for item in messages],
            profile={"routine_summary": {"workdays": [{"text": "十点上班"}]}},
            routine_profile={"workdays": [{"text": "十点上班"}]},
            compiler_version="test",
        )
        # 工具本身不读取 Branch，因此这个轻量测试无需构造完整 Runtime 分支。
        session.add(snapshot)
        session.flush()
        context = DayPlanContext(
            generated_at=now,
            virtual_now=now,
            branch_id="b",
            target_date=now.date(),
            timezone="UTC",
            mode="initial",
            request={},
            hard_constraints={},
            origin_projection={},
            date_features={"calendar_verified": True, "is_workday": True},
            branch_evidence={},
            evidence_requirements={"required_topics": ["work"]},
            budget={"max_blocks": 24, "minimum_granularity_minutes": 15},
        )
        sidecar = _FakeLightRAG(bundle_content)
        result = build_routine_evidence_tool(
            session,
            snapshot=snapshot,
            context=context,
            settings=Settings(lightrag_enabled=True),
            lightrag_client=sidecar,  # type: ignore[arg-type]
        ).invoke({"question": "目标人物当前工作时间有什么线索和反例？"})

        assert sidecar.queries == ["目标人物当前工作时间有什么线索和反例？"]
        assert any(item["content"] == "我早上十点才上班" for item in result["data"]["messages"])
        assert "intervals" not in result["data"]  # 代码不能把一次观察平均成长期工作区间。
        assert set(result["source_ids"]) == {"work-start", "work-end", "relative-end"}
    database.close()


def test_runtime_cycle_smoke_without_gpu(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'runtime.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        session.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.flush()
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        # Runtime v1 只允许从已完成的人物世界启动；测试直接放入冻结快照，
        # 避免在冒烟测试中启动 LightRAG 或占用本地模型显存。
        session.add(
            RuntimeSnapshotRow(
                id="snapshot",
                branch_id="b",
                cutoff_at=now,
                timezone="UTC",
                snapshot_mode="latest_profile",
                source_message_ids=[],
                profile={"identity": {"names": []}},
                routine_profile={},
                compiler_version="test",
            )
        )
        session.commit()
        planner_proposal = (
            '{"plan_date":"' + now.date().isoformat() + '","blocks":['
            '{"start":"00:00","end":"08:00","activity":"休息","location_role":"home",'
            '"default_availability":"asleep","basis":"fallback","confidence":"fallback"},'
            '{"start":"08:00","end":"12:00","activity":"个人时间","location_role":"home",'
            '"default_availability":"available","basis":"fallback","confidence":"fallback"},'
            '{"start":"12:00","end":"18:00","activity":"个人时间","location_role":"outside",'
            '"default_availability":"available","basis":"fallback","confidence":"fallback"},'
            '{"start":"18:00","end":"24:00","activity":"休息","location_role":"home",'
            '"default_availability":"resting","basis":"fallback","confidence":"fallback"}'
            '],"assumptions":["测试中的无证据保守安排"],"private_reason":"smoke"}'
        )
        service = RuntimeService(
            session,
            director_model=_RoleAwarePlannerModel(
                '{"action":"speak","reply":{"messages":[{"kind":"text","text":"你好呀"}]},"state_patch":{},"private_reason":"用户消息"}',
                planner_proposal,
            ),
            actor_model=_FakeModel('{"text":"你好呀","bubbles":["你好呀"],"style_applied":[]}'),
        )
        pending_bootstrap = service.bootstrap("p", "b")
        # bootstrap 只能建立空槽位，绝不能在模型前注入固定睡眠/工作/通勤日程。
        assert pending_bootstrap["plan"].blocks == []
        assert pending_bootstrap["plan"].generation_metadata["status"] == "pending"
        assert (
            session.scalar(
                select(RuntimeWakeupRow).where(
                    RuntimeWakeupRow.branch_id == "b",
                    RuntimeWakeupRow.status == "scheduled",
                )
            )
            is None
        )
        # bootstrap 会建立独立 MemoryIndexVersion，不再把旧 BranchStateVersion 当作事实库。
        assert (
            session.scalar(
                select(RuntimeMemoryIndexRow).where(RuntimeMemoryIndexRow.branch_id == "b")
            )
            is not None
        )
        prepared = service.process_next(project_id="p", branch_id="b", prepare_branch=True)
        assert prepared["trace"].planner_proposal["plan_date"] == now.date().isoformat()
        service.submit_user_message(
            project_id="p", branch_id="b", content="在吗", idempotency_key="msg-1"
        )
        result = service.process_next(project_id="p", branch_id="b")
        assert result is not None
        assert result["decision"].action == "speak"
        assert result["message"].content == "你好呀"
        trace = result["trace"]
        assert trace.status == "succeeded"
        assert trace.planner_proposal is None  # 普通聊天不得偷偷规划。
        assert trace.director_decision["action"] == "speak"
        assert trace.outcome["director_terminal_reason"] == "decision"
        plan = service.get_plan("p", "b")
        assert plan.generation_metadata["status"] == "agent"
        assert plan.generation_metadata["date_features"]["calendar_verified"] is False
        assert session.get(RuntimeCycleTraceRow, trace.id) is not None

        # Director 只提交异步请求；同一 Cycle 不等待 Planner 或声称计划已生效。
        revision_service = RuntimeService(
            session,
            director_model=_RoleAwarePlannerModel(
                '{"action":"wait","state_patch":{},"plan_request":{"reason":"用户需要调整今天安排"},'
                '"private_reason":"计划修订"}',
                planner_proposal,
            ),
            actor_model=_FakeModel('{"text":"好的呀","bubbles":["好的呀"],"style_applied":[]}'),
        )
        revision_service.submit_user_message(
            project_id="p", branch_id="b", content="帮我调整今天安排", idempotency_key="msg-plan"
        )
        revision = revision_service.process_next(project_id="p", branch_id="b")
        assert revision is not None
        assert revision["message"] is None
        from moonlightbox.jobs.models import Job
        assert any(job.payload.get("planner_task") for job in session.scalars(
            select(Job).where(Job.status == "queued")
        ))
        revised_plan = revision_service.get_plan("p", "b")
        current_state = revision_service.get_state("p", "b")
        assert current_state["current_plan_block_id"] in {
            block["id"] for block in revised_plan.blocks
        }

        # 同时到期的多个 Wakeup 必须合并为一次 Cycle，并在提交后续接下一条计划边界。
        virtual_now = service.get_clock("p", "b").now()
        first_plan_wakeup = session.scalar(
            select(RuntimeWakeupRow).where(
                RuntimeWakeupRow.branch_id == "b",
                RuntimeWakeupRow.status == "scheduled",
            )
        )
        assert first_plan_wakeup is not None
        first_plan_wakeup.wake_at = virtual_now
        # 模拟已经跨过此前安排的边界，使后续调度必须创建新的标准计划 key。
        first_plan_wakeup.idempotency_key = "smoke-expired-plan"
        due_wakeups = [
            RuntimeWakeupRow(
                branch_id="b",
                wake_at=virtual_now,
                reason="延迟回复",
                trigger_type="delayed_reply",
                idempotency_key="smoke-delayed-1",
            ),
        ]
        session.add_all(due_wakeups)
        session.commit()
        merged = service.process_next(project_id="p", branch_id="b")
        assert merged is not None
        # 同时到期的输入都必须到达；同优先级的主触发顺序不依赖随机 UUID。
        assert {item["type"] for item in [
            merged["packet"].trigger, *merged["packet"].trigger["additional_triggers"]
        ]} == {"delayed_reply", "plan_transition"}
        assert len(merged["packet"].trigger["additional_triggers"]) == 1
        assert all(
            session.get(RuntimeWakeupRow, wakeup.id).status == "completed"
            for wakeup in [first_plan_wakeup, *due_wakeups]
        )
        assert (
            session.scalar(
                select(RuntimeWakeupRow).where(
                    RuntimeWakeupRow.branch_id == "b",
                    RuntimeWakeupRow.status == "scheduled",
                    RuntimeWakeupRow.wake_at > virtual_now,
                )
            )
            is not None
        )

        # Director 的不完整 speak 决定会在 Actor 之前被 Executor 拒绝，不能浪费
        # LoRA 调用，也不能让未校验意图进入公开表达层。
        rejected_actor = _FakeModel('{"text":"不应生成","bubbles":["不应生成"]}')
        rejecting_service = RuntimeService(
            session,
            director_model=_FakeModel('{"action":"speak","state_patch":{}}'),
            actor_model=rejected_actor,
        )
        rejecting_service.submit_user_message(
            project_id="p", branch_id="b", content="测试校验", idempotency_key="msg-invalid"
        )
        with pytest.raises(RuntimeModelExecutionError):
            rejecting_service.process_next(project_id="p", branch_id="b")
        assert rejected_actor.calls == 0

        # 云端 Director 两次调用都失败时，事件必须回到 queued、Trace 标为失败；不能
        # 把用户消息伪装成正常 wait 后直接完成，否则用户看到的就是“永远不回复”。
        unavailable_service = RuntimeService(
            session,
            director_model=_UnavailableModel(),
            actor_model=_FakeModel('{"text":"不应生成","bubbles":["不应生成"]}'),
        )
        pending = unavailable_service.submit_user_message(
            project_id="p", branch_id="b", content="服务恢复后回复", idempotency_key="msg-retry"
        )
        previous_trace_ids = set(session.scalars(select(RuntimeCycleTraceRow.id)))
        try:
            unavailable_service.process_next(project_id="p", branch_id="b")
        except RuntimeModelExecutionError as error:
            assert error.code == "model_error"
        else:
            raise AssertionError("云端模型不可用时必须中止并保留事件")
        pending_event = session.get(RuntimeEventRow, pending["event"].id)
        assert pending_event is not None and pending_event.status == "queued"
        failure_trace = session.scalar(
            select(RuntimeCycleTraceRow)
            .where(
                RuntimeCycleTraceRow.branch_id == "b",
                RuntimeCycleTraceRow.id.not_in(previous_trace_ids),
            )
            .order_by(RuntimeCycleTraceRow.started_at.desc())
        )
        assert failure_trace is not None
        assert failure_trace.status == "failed"
        assert failure_trace.error_code == "model_error"

        # Wakeup 是虚拟时间：暂停后即使墙上时间继续流逝，扫描器也不应创建任务。
        scheduled = session.scalar(
            select(RuntimeWakeupRow).where(
                RuntimeWakeupRow.branch_id == "b",
                RuntimeWakeupRow.status == "scheduled",
            )
        )
        assert scheduled is not None
        scheduled.wake_at = virtual_now + timedelta(hours=1)
        session.commit()
        service.change_clock("p", "b", action="pause")
        assert enqueue_due_runtime_cycles(database, now=datetime.now(UTC) + timedelta(hours=2)) == 0

        # 暂停也停止后台规划；恢复后补齐旧 pending 计划，不偷跑模型。
        service.get_plan("p", "b").generation_metadata = {"status": "pending"}
        session.commit()
        assert enqueue_due_runtime_cycles(database, now=datetime.now(UTC) + timedelta(hours=2)) == 0
        service.change_clock("p", "b", action="resume")
        # 恢复后既补排 pending 计划，也唤起前面失败后仍 queued 的输入事件。
        assert enqueue_due_runtime_cycles(database, now=datetime.now(UTC)) == 2
    database.close()


@pytest.mark.parametrize("graph_status", ["ready", "superseded"])
def test_style_tool_queries_lightrag_without_forcing_history_into_reply_pairs(
    tmp_path, graph_status
) -> None:
    """跨夜的相邻消息可以作为原文证据，但绝不被工具标成一问一答。"""

    database = Database(f"sqlite:///{tmp_path / 'style.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 9, 2, 8, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        session.add(
            ImportSource(
                id="import",
                project_id="p",
                preview_id="preview",
                source_path="/tmp/chat.txt",
                message_count=2,
                confirmed_at=now,
            )
        )
        session.flush()
        session.add_all(
            [
                Participant(id="self", project_id="p", name="我", role="self"),
                Participant(id="target", project_id="p", name="她", role="target"),
            ]
        )
        session.flush()
        session.add_all(
            [
                Message(
                    id="sleep",
                    project_id="p",
                    import_id="import",
                    participant_id="self",
                    source_id="1",
                    timestamp=now - timedelta(hours=10),
                    kind="text",
                    content="我睡觉了，晚安",
                    raw={},
                ),
                Message(
                    id="nightmare",
                    project_id="p",
                    import_id="import",
                    participant_id="target",
                    source_id="2",
                    timestamp=now,
                    kind="text",
                    content="bb 我做噩梦了",
                    raw={},
                ),
            ]
        )
        session.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        # EventNode 外键指向原始消息，先明确写入聊天证据，避免 ORM 无关联对象
        # 在同一次 flush 中选择不稳定的插入顺序。
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="sleep",
                end_message_id="nightmare",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        session.add(
            WorldGraphVersion(
                id="graph",
                project_id="p",
                trigger_import_id="import",
                workspace_key="world_test",
                status=graph_status,
                source_fingerprint="source",
                config_fingerprint="config",
                source_import_ids=["import"],
                compiler_version="test",
            )
        )
        session.flush()
        bundle = ConversationBundle(
            project_id="p",
            graph_version_id="graph",
            document_id="bundle-1",
            source_name="bundle-1.txt",
            ordinal=0,
            started_at=now - timedelta(hours=10),
            ended_at=now,
            content="2026-09-01 22:00 我：我睡觉了，晚安\n2026-09-02 08:00 她：bb 我做噩梦了",
            content_hash="bundle",
            primary_message_count=2,
        )
        session.add(bundle)
        session.flush()
        session.add_all(
            [
                ConversationBundleMessage(bundle_id=bundle.id, message_id="sleep", ordinal=0),
                ConversationBundleMessage(bundle_id=bundle.id, message_id="nightmare", ordinal=1),
                RuntimeSnapshotRow(
                    id="snapshot",
                    branch_id="b",
                    graph_version_id="graph",
                    cutoff_at=now,
                    timezone="UTC",
                    snapshot_mode="latest_profile",
                    source_message_ids=["sleep", "nightmare"],
                    profile={},
                    routine_profile={},
                    compiler_version="test",
                ),
            ]
        )
        session.commit()

        sidecar = _FakeLightRAG(bundle.content)
        result = (
            StyleService(
                session,
                settings=Settings(lightrag_enabled=True),
                lightrag_client=sidecar,  # type: ignore[arg-type]
            )
            .tool(branch_id="b", model_version_id="m")
            .invoke(
                {
                    "situation": "用户早上问候，目标人物准备自然回应。",
                    "intent": "早安问候",
                    "speech_mode": "reply",
                    "limit": 2,
                }
            )
        )

        assert "用户早上问候" in sidecar.queries[0]
        example = result["examples"][0]
        assert example["source_ids"] == ["sleep", "nightmare"]
        assert "prompt" not in example
        assert "text" not in example
    database.close()


def test_source_history_ends_at_branch_origin_event(tmp_path) -> None:
    """聊天页面只展示节点结束消息及其之前的同一批导入记录。"""

    database = Database(f"sqlite:///{tmp_path / 'history.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 9, 2, 8, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        session.add_all(
            [
                ImportSource(
                    id="import",
                    project_id="p",
                    preview_id="preview",
                    source_path="/tmp/chat.txt",
                    message_count=3,
                    confirmed_at=now,
                ),
                Participant(id="self", project_id="p", name="我", role="self"),
                Participant(id="target", project_id="p", name="她", role="target"),
                ModelVersion(
                    id="m",
                    project_id="p",
                    base_model="test",
                    adapter_path="none",
                    dataset_hash="h",
                    metrics={},
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                Message(
                    id="before",
                    project_id="p",
                    import_id="import",
                    participant_id="self",
                    source_id="1",
                    timestamp=now - timedelta(minutes=2),
                    kind="text",
                    content="早安",
                    raw={},
                ),
                Message(
                    id="boundary",
                    project_id="p",
                    import_id="import",
                    participant_id="target",
                    source_id="2",
                    timestamp=now - timedelta(minutes=1),
                    kind="text",
                    content="早呀",
                    raw={},
                ),
                Message(
                    id="after",
                    project_id="p",
                    import_id="import",
                    participant_id="self",
                    source_id="3",
                    timestamp=now,
                    kind="text",
                    content="节点之后的消息",
                    raw={},
                ),
            ]
        )
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="2",
                ended_at=now - timedelta(minutes=1),
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.flush()
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        session.commit()

        page = SourceHistoryService(session).page("p", "b", before=None, limit=20)

        assert [item.id for item in page.items] == ["before", "boundary"]
        assert page.has_more is False
        assert page.next_cursor is None
        # latest_profile 体验起点与旧事件节点不同，界面必须显示至绑定图末端。
        session.add(
            WorldGraphVersion(
                id="graph-latest",
                project_id="p",
                trigger_import_id="import",
                workspace_key="latest",
                status="ready",
                source_fingerprint="s",
                config_fingerprint="c",
                source_import_ids=["import"],
                compiler_version="test",
            )
        )
        session.flush()
        bundle = ConversationBundle(
            project_id="p",
            graph_version_id="graph-latest",
            document_id="latest",
            source_name="latest.txt",
            ordinal=0,
            started_at=now,
            ended_at=now,
            content="末端",
            content_hash="latest",
            primary_message_count=1,
        )
        session.add(bundle)
        session.flush()
        session.add(ConversationBundleMessage(bundle_id=bundle.id, message_id="after", ordinal=0))
        session.add(
            RuntimeSnapshotRow(
                branch_id="b",
                graph_version_id="graph-latest",
                cutoff_at=now,
                timezone="Asia/Shanghai",
                snapshot_mode="latest_profile",
                compiler_version="test",
            )
        )
        session.commit()
        latest = SourceHistoryService(session).page("p", "b", before=None, limit=20)
        assert [item.id for item in latest.items] == ["before", "boundary", "after"]
    database.close()
