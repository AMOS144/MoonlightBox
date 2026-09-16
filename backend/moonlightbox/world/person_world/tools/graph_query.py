"""PersonWorldAgent 的 LightRAG 只读查询工具。"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from moonlightbox.world.bundles import match_bundle_document_reference
from moonlightbox.world.client import LightRAGSidecarClient

from ..investigation_artifacts import InvestigationArtifactStore


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def nonblank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("参数不能只有空白；可选筛选不需要时请省略")
        return value


class SearchWorldArgs(_StrictArgs):
    question: str = Field(
        min_length=8,
        max_length=1000,
        description="需要核实的完整人物世界问题，必须包含主体和事实维度。",
    )


class TemporalSearchWorldArgs(SearchWorldArgs):
    period: Literal["before", "after", "all"] = Field(
        default="before",
        description="before 调查起点前材料（包含明确标注的跨界场景）；after 查纯后段以核对解释、变化与反例；all 跨期综合。时间分类以逐消息标记为准。",
    )


class GraphPatchSearchArgs(_StrictArgs):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="围绕用户已确认纠正范围提出的具体图谱问题；读取当前待纠正图，不执行任何增删改",
    )


class ReadRegressionQueryArgs(_StrictArgs):
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="原样复制本任务已批准回归清单中的 query，不改写或新增；返回候选图实际检索结果供判定",
    )


class ListGraphEntitiesArgs(_StrictArgs):
    name_contains: str | None = Field(
        default=None,
        max_length=200,
        description="实体名称的字面包含筛选，不是语义问题或正则；省略列出全部范围内实体",
    )
    limit: int = Field(
        default=100, ge=1, le=500, description="最多返回的实体数；达到上限不代表图谱只有这些实体"
    )


class GetEntityArgs(_StrictArgs):
    entity_name: str = Field(
        min_length=1,
        max_length=500,
        description="图谱查询或实体列表返回的完整实体名称，不是消息 ID",
    )


class GetEntityNeighborhoodArgs(_StrictArgs):
    entity_name: str = Field(
        min_length=1, max_length=500, description="图谱中已有的完整实体名称，作为邻域中心"
    )
    depth: int = Field(
        default=1,
        ge=1,
        le=2,
        description="关系跳数：1 为直接邻居，2 包含邻居的邻居；不代表时间范围",
    )


class GetRelationArgs(_StrictArgs):
    source_entity: str = Field(
        min_length=1, max_length=500, description="待读取关系的起点实体完整名称，来自已有图谱"
    )
    target_entity: str = Field(
        min_length=1,
        max_length=500,
        description="待读取关系的终点实体完整名称；这里只读，不创建关系",
    )


_SEARCH_DESCRIPTION = """只读查询当前 PersonWorld LightRAG 图谱，寻找可能相关的实体、关系和
原始文档引用。结果只是调查线索，不能单独证明人物事实；必须继续用原始消息工具核验。"""

_LIST_DESCRIPTION = """列出当前图谱实体及描述，用于检查图中已经存在的人物、地点、组织、
规律和事件。该工具只读，不会合并或修改节点。"""

_ENTITY_DESCRIPTION = """读取一个图谱实体的当前详情。结果来自派生图谱，修改建议仍需回到
原始消息证据并经过用户审核。"""
_NEIGHBORHOOD_DESCRIPTION = """读取图谱中一个实体的局部邻域快照。它只用于定位已有的候选
实体和关系；返回内容来自派生图谱，不能独立证明人物事实。"""
_RELATION_DESCRIPTION = """读取两个已知实体之间的当前图谱关系。它只说明图中目前有什么，
不判定关系内容真伪，所有结论仍要回到原始消息。"""


def build_graph_query_tools(
    client: LightRAGSidecarClient,
    *,
    workspace: str,
    allowed_document_names: set[str],
    artifacts: InvestigationArtifactStore,
    top_k: int,
    chunk_top_k: int,
    max_total_tokens: int,
    temporal_scope: dict[str, object] | None = None,
) -> dict[str, StructuredTool]:
    """把 workspace 和来源白名单绑定在闭包中，模型不能跨项目查询。"""

    def search_world(**kwargs: Any) -> dict[str, object]:
        args = (
            TemporalSearchWorldArgs if temporal_scope is not None else SearchWorldArgs
        ).model_validate(kwargs)
        retrieval = client.query(
            workspace,
            args.question,
            mode="mix",
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_total_tokens=max_total_tokens,
            **(
                {"temporal": {**temporal_scope, "period": args.period}}
                if temporal_scope is not None
                else {}
            ),
        )
        references: list[dict[str, object]] = []
        document_ranks: dict[str, int] = {}
        for reference_rank, reference in enumerate(retrieval.references, start=1):
            document_name = match_bundle_document_reference(
                reference.file_path, allowed_document_names
            )
            if document_name is None:
                continue
            document_rank = document_ranks.setdefault(document_name, len(document_ranks) + 1)
            references.append(
                {
                    "document_name": document_name,
                    "reference_id": reference.reference_id,
                    "chunk_content": reference.content,
                    "reference_rank": reference_rank,
                    "document_rank": document_rank,
                    **(
                        {
                            key: value
                            for key, value in reference.model_dump().items()
                            if key
                            in {
                                "chunk_id",
                                "assigned_period",
                                "boundary_relation",
                                "messages",
                                "content_hash",
                            }
                        }
                        if temporal_scope is not None
                        else {}
                    ),
                }
            )
        artifact = artifacts.add_retrieval(
            question=args.question,
            mode="mix",
            references=references,
        )
        return {
            "retrieval_id": artifact.retrieval_id,
            "question": args.question,
            "mode": "mix",
            "context": retrieval.context,
            "references": list(artifact.references),
            **(
                {
                    "temporal_diagnostics": retrieval.temporal_diagnostics,
                    "global_graph_clues": retrieval.global_graph_clues,
                }
                if temporal_scope is not None
                else {}
            ),
        }

    def list_entities(**kwargs: Any) -> list[dict[str, object]]:
        args = ListGraphEntitiesArgs.model_validate(kwargs)
        needle = args.name_contains.casefold() if args.name_contains else None
        rows: list[dict[str, object]] = []
        for entity in client.list_entities(workspace):
            if needle is not None and needle not in entity.entity_name.casefold():
                continue
            rows.append(
                {
                    "entity_name": entity.entity_name,
                    "graph_data": dict(entity.graph_data or {}),
                    **({"scope": "global_retrospective"} if temporal_scope is not None else {}),
                }
            )
            if len(rows) >= args.limit:
                break
        return rows

    def get_entity(**kwargs: Any) -> dict[str, object]:
        args = GetEntityArgs.model_validate(kwargs)
        entity = client.get_entity(workspace, args.entity_name)
        return {
            "entity_name": args.entity_name,
            "graph_data": entity or {},
            **({"scope": "global_retrospective"} if temporal_scope is not None else {}),
        }

    def get_entity_neighborhood(**kwargs: Any) -> dict[str, object]:
        args = GetEntityNeighborhoodArgs.model_validate(kwargs)
        # Sidecar 的实体详情包含其当前图数据。LightRAG 暂未提供一个可以安全分页的
        # 全图邻居 API，因此 depth 目前只作为请求的审计边界返回，不能伪造递归结果。
        entity = client.get_entity(workspace, args.entity_name)
        return {
            "entity_name": args.entity_name,
            "requested_depth": args.depth,
            **({"scope": "global_retrospective"} if temporal_scope is not None else {}),
            "graph_data": entity or {},
        }

    def get_relation(**kwargs: Any) -> dict[str, object]:
        args = GetRelationArgs.model_validate(kwargs)
        relation = client.get_relation(workspace, args.source_entity, args.target_entity)
        return {
            "source_entity": args.source_entity,
            "target_entity": args.target_entity,
            **({"scope": "global_retrospective"} if temporal_scope is not None else {}),
            "graph_data": relation,
        }

    return {
        "search_world": StructuredTool.from_function(
            name="search_world",
            description=_SEARCH_DESCRIPTION,
            func=search_world,
            args_schema=cast(
                Any, TemporalSearchWorldArgs if temporal_scope is not None else SearchWorldArgs
            ),
        ),
        "list_graph_entities": StructuredTool.from_function(
            name="list_graph_entities",
            description=_LIST_DESCRIPTION,
            func=list_entities,
            args_schema=cast(Any, ListGraphEntitiesArgs),
        ),
        "get_graph_entity": StructuredTool.from_function(
            name="get_graph_entity",
            description=_ENTITY_DESCRIPTION,
            func=get_entity,
            args_schema=cast(Any, GetEntityArgs),
        ),
        "get_entity_neighborhood": StructuredTool.from_function(
            name="get_entity_neighborhood",
            description=_NEIGHBORHOOD_DESCRIPTION,
            func=get_entity_neighborhood,
            args_schema=cast(Any, GetEntityNeighborhoodArgs),
        ),
        "get_relation": StructuredTool.from_function(
            name="get_relation",
            description=_RELATION_DESCRIPTION,
            func=get_relation,
            args_schema=cast(Any, GetRelationArgs),
        ),
    }
