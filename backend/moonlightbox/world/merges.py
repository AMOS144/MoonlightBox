"""LightRAG 建图后的别名归一化 workflow。

这个模块只生成待人工审核的提案。真正修改 LightRAG 的 ``amerge_entities``
仍然只由审核接口在用户批准后调用。候选生成使用一个有界 LangGraph 节点，
并把原文定位、图节点详情和 LightRAG 局部检索封装成只读工具。
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from moonlightbox.world.client import LightRAGEntity, LightRAGSidecarClient


class MergeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    entity: str = Field(min_length=1, max_length=500, description="证据关联的候选实体名称")
    basis: str = Field(
        min_length=1,
        max_length=1000,
        description="为何此材料支持或反驳同一人物判断，区分同名和同一人",
    )
    source_document_id: str | None = Field(
        default=None, max_length=180, description="实际提供的原始文档引用；没有则 null"
    )
    message_id: str | None = Field(
        default=None, max_length=80, description="真实消息 UUID；没有则 null，不编造"
    )
    quote: str | None = Field(
        default=None,
        max_length=1200,
        description="相关原话短摘录，必须来自实际材料，不改写后冒充原话",
    )


class LocateMentionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    names: list[str] = Field(
        min_length=1,
        max_length=20,
        description="需要定位的候选人物名或别名；按字面包含检索，不能据此自动认定两人相同",
    )
    limit: int = Field(
        default=80, ge=1, le=200, description="最多返回的原文命中条数，包含发送者和消息引用"
    )


class QueryPersonGraphArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        min_length=1,
        max_length=500,
        description="待调查的人物节点完整名称；返回局部关系线索，不写入或合并图谱",
    )


class MergeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_entities: list[str] = Field(
        min_length=1, max_length=20, description="拟合并的已有源实体名称，不包含无关人物"
    )
    target_entity: str = Field(
        min_length=1, max_length=500, description="拟保留的目标实体名称；只是候选，等待用户审核"
    )
    reason: str = Field(
        default="", max_length=2000, description="说明为何认为这些名称属于同一人物，保留不确定性"
    )
    evidence: list[MergeEvidence] = Field(
        default_factory=list,
        max_length=20,
        description="与候选直接相关的已提供材料；无明确材料可 []，不编造",
    )


class MergeCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proposals: list[MergeCandidate] = Field(
        default_factory=list,
        max_length=100,
        description="本次别名合并候选列表；无可靠候选填 []，提交不会执行合并",
    )


_TECHNICAL_ENTITY_ID = re.compile(
    r"^(?:wxid_[A-Za-z0-9_-]+|[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)$", re.IGNORECASE
)


def is_technical_entity_name(name: str) -> bool:
    value = name.strip()
    return bool(_TECHNICAL_ENTITY_ID.fullmatch(value)) and (
        value.lower().startswith("wxid_") or len(value) >= 8
    )


def person_entities(entities: Iterable[LightRAGEntity]) -> list[LightRAGEntity]:
    """只保留 LightRAG 标为人物的节点，不从普通正文猜测人物。"""
    return [
        item
        for item in entities
        if not is_technical_entity_name(item.entity_name)
        if _is_person_graph_data(item.graph_data)
    ]


def _is_person_graph_data(graph_data: dict[str, object] | None) -> bool:
    if not isinstance(graph_data, dict):
        return False
    for key in ("entity_type", "type", "node_type", "entityType"):
        value = graph_data.get(key)
        if isinstance(value, str) and value.strip().casefold() in {
            "person",
            "human",
            "people",
            "人物",
            "人",
        }:
            return True
    return False


@dataclass(frozen=True)
class AliasAgentMessage:
    message_id: str
    document_id: str | None
    timestamp: str
    participant: str
    content: str
    participant_role: str = "other"


class AliasResolutionTools:
    """Agent 使用的只读工具集合。工具不能写图谱或改变审核状态。"""

    def __init__(
        self,
        *,
        sidecar: LightRAGSidecarClient,
        workspace: str,
        entities: list[LightRAGEntity],
        original_messages: Iterable[AliasAgentMessage] = (),
    ) -> None:
        self.sidecar = sidecar
        self.workspace = workspace
        self.entities = tuple(entities)
        self.original_messages = tuple(original_messages)

    def list_person_nodes(self) -> list[dict[str, object]]:
        return [
            {
                "entity": item.entity_name,
                "description": (
                    str(item.graph_data.get("description", ""))[:450]
                    if isinstance(item.graph_data, dict)
                    else ""
                ),
                "graph_data": dict(item.graph_data or {}),
            }
            for item in self.entities
        ]

    def locate_original_mentions(
        self, names: list[str], limit: int = 80
    ) -> list[dict[str, object]]:
        needles = tuple(dict.fromkeys(item.strip() for item in names if item.strip()))
        matches = [
            {
                "message_id": item.message_id,
                "document_id": item.document_id,
                "timestamp": item.timestamp,
                "participant": item.participant,
                "content": item.content[:400],
            }
            for item in self.original_messages
            if any(needle in item.content or needle == item.participant for needle in needles)
        ]
        return matches[: max(0, limit)]

    def query_lightrag(self, name: str) -> dict[str, object]:
        result = self.sidecar.query(
            self.workspace,
            f"请返回人物节点“{name}”的描述、关系网络、出现过的称呼及相关原文证据。",
            mode="local",
            top_k=12,
            chunk_top_k=8,
            max_total_tokens=6000,
        )
        return {
            "entity": name,
            "context": result.context,
            "references": [item.model_dump(mode="json") for item in result.references],
        }

    def as_langchain_tools(self) -> dict[str, StructuredTool]:
        return {
            "list_person_nodes": StructuredTool.from_function(
                func=lambda: self.list_person_nodes(),
                name="list_person_nodes",
                description="读取已由 LightRAG 标记为人物的全部图节点及描述。",
            ),
            "locate_original_mentions": StructuredTool.from_function(
                func=self.locate_original_mentions,
                args_schema=LocateMentionsArgs,
                name="locate_original_mentions",
                description="在原始聊天消息中定位名称，返回 message_id、Bundle 文档和原文摘录。",
            ),
            "query_lightrag": StructuredTool.from_function(
                func=self.query_lightrag,
                args_schema=QueryPersonGraphArgs,
                name="query_lightrag",
                description="查询一个人物节点的 LightRAG 局部关系和上下文。",
            ),
        }


class AliasResolutionAgent:
    """只读调查与提案提交走统一循环，图谱合并仍须用户批准。"""

    def __init__(self, *, tools, compiler, subject_name, user_name, session=None, project_id=None):
        self.tools, self.compiler = tools, compiler
        self.subject_name, self.user_name = subject_name, user_name
        self.session = session
        self.project_id = project_id

    def run(self):
        from pathlib import Path

        from moonlightbox.agent_runtime.contracts import RegisteredTool, ToolContract
        from moonlightbox.agent_runtime.tasks import run_submission_task
        from moonlightbox.agent_runtime.tool_execution import serial

        allowed = {item.entity_name for item in self.tools.entities}

        def validate(result, context):
            for candidate in result.proposals:
                if candidate.target_entity not in allowed:
                    return "target_entity 必须来自可用人物节点。"
                if any(
                    source not in allowed or source == candidate.target_entity
                    for source in candidate.source_entities
                ):
                    return "source_entities 必须是现有节点，且不能包含 target_entity。"
            return None

        tools = tuple(
            RegisteredTool(
                tool,
                ToolContract(
                    name=tool.name,
                    execution=serial("alias_read_context", reason="共享图谱客户端和冻结消息"),
                    timeout_seconds=300,
                    max_result_chars=32000,
                ),
            )
            for tool in self.tools.as_langchain_tools().values()
        )
        result = run_submission_task(
            compiler=self.compiler,
            name="alias_resolution",
            system_prompt=Path(__file__)
            .with_name("prompts")
            .joinpath("alias_resolution.md")
            .read_text(),
            payload={
                "target": self.subject_name,
                "user": self.user_name,
                "nodes": self.tools.list_person_nodes(),
            },
            result_model=MergeCandidateResponse,
            owner_id=self.tools.workspace,
            session=self.session,
            project_id=self.project_id,
            tools=tools,
            validator=validate,
        )
        return result.proposals


def generate_merge_candidates(
    *,
    sidecar,
    compiler,
    workspace,
    subject_name,
    user_name,
    original_messages=(),
    max_entities=160,
    session=None,
    project_id=None,
):
    entities = person_entities(sidecar.list_entities(workspace))[:max_entities]
    if len(entities) < 2:
        return []
    tools = AliasResolutionTools(
        sidecar=sidecar, workspace=workspace, entities=entities, original_messages=original_messages
    )
    return AliasResolutionAgent(
        tools=tools,
        compiler=compiler,
        subject_name=subject_name,
        user_name=user_name,
        session=session,
        project_id=project_id,
    ).run()
