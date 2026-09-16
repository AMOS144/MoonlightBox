"""把 LightRAG 文档引用还原为带 UUID 的 SQL 原始消息。"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.world.models import ConversationBundle, ConversationBundleMessage

from ..investigation_artifacts import InvestigationArtifactStore


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LocateSourceMessagesArgs(_StrictArgs):
    # ``retrieval_id`` 只能由上一跳 search_world 的运行内工件生成。模型不会看到
    # chunk，也不能伪造数据库文档名来扩大恢复范围。
    retrieval_id: str = Field(
        min_length=1,
        max_length=100,
        description="本次 search_world 返回的 retrieval_id；不是文档名、消息 UUID 或 evidence_set_id",
    )
    context_turns: int = Field(
        default=4,
        ge=1,
        le=10,
        description="命中位置两侧补充的上下文范围；仍受原 Bundle 和消息上限限制",
    )
    limit: int = Field(
        default=160,
        ge=1,
        le=300,
        description="最多还原的消息数；更多正文通过返回的证据集合分页读取",
    )


class GetMessageContextArgs(_StrictArgs):
    message_ids: list[str] = Field(
        min_length=1,
        max_length=40,
        description="原文工具实际返回的消息 UUID 数组；不可用序号、文档名或臆造 ID 替代",
    )
    context_turns: int = Field(
        default=5, ge=1, le=12, description="在各命中消息前后读取的同 Bundle 上下文范围"
    )


_LOCATE_DESCRIPTION = """把 LightRAG 命中的 Bundle 和 chunk 精确还原为 SQL 原始消息 UUID，
并携带相邻对话。该工具不根据关键词或正则判定事实，只负责恢复可审计上下文。"""

_CONTEXT_DESCRIPTION = """按真实消息 UUID 读取同一 Bundle 中的前后连续对话，用于判断代词、
称呼、转述、反问、时间和事实主体。"""


def build_source_message_tools(
    session: Session,
    *,
    project_id: str,
    graph_version_id: str,
    artifacts: InvestigationArtifactStore,
    message_periods: dict[str, str] | None = None,
) -> dict[str, StructuredTool]:
    def locate_messages(**kwargs: Any) -> dict[str, object]:
        args = LocateSourceMessagesArgs.model_validate(kwargs)
        retrieval = artifacts.get_retrieval(args.retrieval_id)
        if retrieval is None:
            from moonlightbox.agent_runtime.tool_errors import ToolInputError

            raise ToolInputError(
                "retrieval_id 不属于当前调查；先调用 search_world 并原样使用它返回的 retrieval_id。无效引用不代表没有证据。",
                field="retrieval_id",
            )
        references = list(retrieval.references[:30])
        document_names = [
            str(item["document_name"])
            for item in references
            if isinstance(item.get("document_name"), str)
        ]
        if not document_names:
            return {
                "retrieval_id": args.retrieval_id,
                "evidence_set_id": None,
                "message_count": 0,
                "primary_message_ids": [],
                "message_ids": [],
                "negative_evidence": False,
                "retrieval_status": "empty",
                "message": "本次检索没有可还原引用，不代表目标人物没有相关经历或事实",
            }
        bundles = list(
            session.scalars(
                select(ConversationBundle)
                .where(
                    ConversationBundle.project_id == project_id,
                    ConversationBundle.graph_version_id == graph_version_id,
                    ConversationBundle.source_name.in_(document_names),
                )
                .order_by(ConversationBundle.ordinal.asc())
            )
        )
        output: list[dict[str, object]] = []
        seen: set[str] = set()
        document_rank_by_name = {
            str(item["document_name"]): int(item.get("document_rank", index + 1))
            for index, item in enumerate(references)
            if isinstance(item.get("document_name"), str)
        }
        references_by_document: dict[str, list[tuple[str, int]]] = {}
        for index, item in enumerate(references):
            name = item.get("document_name")
            if not isinstance(name, str):
                continue
            chunk = str(item.get("chunk_content", ""))
            rank = int(item.get("reference_rank", index + 1))
            references_by_document.setdefault(name, []).append((chunk, rank))
        for bundle in bundles:
            rows = _bundle_rows(session, bundle.id)
            document_references = references_by_document.get(bundle.source_name, [])
            exact_refs = [
                ref
                for ref in references
                if ref.get("document_name") == bundle.source_name and "messages" in ref
            ]
            exact_messages = {
                str(m["message_id"]): m for ref in exact_refs for m in ref["messages"]
            }
            primary_indexes = (
                {
                    index
                    for index, (_, message, _) in enumerate(rows)
                    if message.id in exact_messages
                }
                if exact_refs
                else _matching_indexes(rows, [chunk for chunk, _ in document_references])
            )
            # 某些 LightRAG 存储只返回 file_path，不返回 chunk。此时读取该
            # Bundle 的完整受限窗口，宁可交给栏目提取 Agent 判断，也不做关键词猜测。
            if not primary_indexes and not exact_refs:
                primary_indexes = set(range(len(rows)))
            included = _expand_indexes(rows, primary_indexes, args.context_turns)
            for index in sorted(included):
                mapping, message, participant = rows[index]
                if message.id in seen:
                    continue
                seen.add(message.id)
                output.append(
                    {
                        "message_id": message.id,
                        "document_id": bundle.document_id,
                        "bundle_id": bundle.id,
                        "ordinal": mapping.ordinal,
                        "timestamp": message.timestamp.isoformat(),
                        "participant_id": participant.id,
                        "participant_name": participant.name,
                        "participant_role": participant.role,
                        "kind": message.kind,
                        "content": message.content,
                        **(
                            {"source_period": message_periods.get(message.id, "unknown")}
                            if message_periods is not None
                            else {}
                        ),
                        "is_primary_match": index in primary_indexes and not mapping.is_carry_in,
                        **(
                            {
                                "source_period": exact_messages[message.id]["period"],
                                "mapping_method": "frozen_source_span",
                            }
                            if message.id in exact_messages
                            else {}
                        ),
                        "retrieval_query_ids": [args.retrieval_id],
                        "reference_ranks": [rank for _, rank in document_references],
                        "document_ranks": [document_rank_by_name.get(bundle.source_name, 0)],
                        "context_window_ids": [f"{args.retrieval_id}:{bundle.id}"],
                        "retrieval_provenance": [
                            {
                                "query_id": args.retrieval_id,
                                "reference_rank": rank,
                                "document_rank": document_rank_by_name.get(bundle.source_name, 0),
                                "context_window_id": f"{args.retrieval_id}:{bundle.id}",
                            }
                            for _, rank in document_references
                        ],
                    }
                )
                if len(output) >= args.limit:
                    return _record_located_messages(artifacts, args.retrieval_id, output)
        return _record_located_messages(artifacts, args.retrieval_id, output)

    def get_context(**kwargs: Any) -> dict[str, object]:
        args = GetMessageContextArgs.model_validate(kwargs)
        requested = set(args.message_ids)
        bundle_ids = list(
            dict.fromkeys(
                session.scalars(
                    select(ConversationBundleMessage.bundle_id)
                    .join(
                        ConversationBundle,
                        ConversationBundle.id == ConversationBundleMessage.bundle_id,
                    )
                    .where(
                        ConversationBundle.project_id == project_id,
                        ConversationBundle.graph_version_id == graph_version_id,
                        ConversationBundleMessage.message_id.in_(requested),
                    )
                )
            )
        )
        output: list[dict[str, object]] = []
        seen: set[str] = set()
        for bundle_id in bundle_ids:
            bundle = session.get(ConversationBundle, bundle_id)
            if bundle is None:
                continue
            rows = _bundle_rows(session, bundle_id)
            primary = {
                index for index, (_, message, _) in enumerate(rows) if message.id in requested
            }
            for index in sorted(_expand_indexes(rows, primary, args.context_turns)):
                mapping, message, participant = rows[index]
                if message.id in seen:
                    continue
                seen.add(message.id)
                output.append(
                    {
                        "message_id": message.id,
                        "document_id": bundle.document_id,
                        "bundle_id": bundle.id,
                        "ordinal": mapping.ordinal,
                        "timestamp": message.timestamp.isoformat(),
                        "participant_id": participant.id,
                        "participant_name": participant.name,
                        "participant_role": participant.role,
                        "kind": message.kind,
                        "content": message.content,
                        "is_primary_match": message.id in requested and not mapping.is_carry_in,
                        **(
                            {"source_period": message_periods.get(message.id, "unknown")}
                            if message_periods is not None
                            else {}
                        ),
                    }
                )
        # 相邻消息不是新的 LightRAG 命中。保留该事实，让最终验证能区分 primary
        # hit 与上下文；其正文只留在运行内工件和 Phoenix。
        evidence_set_id = artifacts.add_evidence_set(output, retrieval_id=None)
        return {
            "missing_message_ids": sorted(requested - seen),
            "retrieval_status": "partial" if requested - seen else "ready",
            "evidence_set_id": evidence_set_id,
            "message_count": len(output),
            "primary_message_ids": [
                str(item["message_id"]) for item in output if item.get("is_primary_match") is True
            ],
            "message_ids": [str(item["message_id"]) for item in output],
        }

    return {
        "locate_source_messages": StructuredTool.from_function(
            name="locate_source_messages",
            description=_LOCATE_DESCRIPTION,
            func=locate_messages,
            args_schema=cast(Any, LocateSourceMessagesArgs),
        ),
        "get_message_context": StructuredTool.from_function(
            name="get_message_context",
            description=_CONTEXT_DESCRIPTION,
            func=get_context,
            args_schema=cast(Any, GetMessageContextArgs),
        ),
    }


def _bundle_rows(
    session: Session, bundle_id: str
) -> list[tuple[ConversationBundleMessage, Message, Participant]]:
    return [
        (mapping, message, participant)
        for mapping, message, participant in session.execute(
            select(ConversationBundleMessage, Message, Participant)
            .join(Message, Message.id == ConversationBundleMessage.message_id)
            .join(Participant, Participant.id == Message.participant_id)
            .where(ConversationBundleMessage.bundle_id == bundle_id)
            .order_by(ConversationBundleMessage.ordinal.asc())
        ).all()
    ]


def _record_located_messages(
    artifacts: InvestigationArtifactStore,
    retrieval_id: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """把完整 SQL 行留给工件库，ToolMessage 只返回可继续调度的清单。"""

    evidence_set_id = artifacts.add_evidence_set(rows, retrieval_id=retrieval_id)
    return {
        "retrieval_id": retrieval_id,
        "evidence_set_id": evidence_set_id,
        "message_count": len(rows),
        "primary_message_ids": [
            str(item["message_id"]) for item in rows if item.get("is_primary_match") is True
        ],
        "message_ids": [str(item["message_id"]) for item in rows],
        "negative_evidence": False,
        "retrieval_status": "ready" if rows else "empty",
    }


def _matching_indexes(
    rows: list[tuple[ConversationBundleMessage, Message, Participant]],
    chunk_contents: list[str],
) -> set[int]:
    """使用建图时的完整渲染行定位 chunk，不做正则或词语事实判断。"""

    chunks = [item for item in chunk_contents if item]
    if not chunks:
        return set()
    indexes: set[int] = set()
    for index, (_, message, participant) in enumerate(rows):
        content = " ".join(message.content.split())
        rendered = f"{message.timestamp:%Y-%m-%d %H:%M} {participant.name}：{content}"
        if any(rendered in chunk for chunk in chunks):
            indexes.add(index)
    return indexes


def _expand_indexes(
    rows: list[tuple[ConversationBundleMessage, Message, Participant]],
    primary: set[int],
    context_turns: int,
) -> set[int]:
    if not primary:
        return set()
    included: set[int] = set()
    for index in primary:
        lower = max(0, index - context_turns)
        upper = min(len(rows), index + context_turns + 1)
        included.update(range(lower, upper))
    return included
