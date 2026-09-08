"""PersonaActor 的只读表达风格工具。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.tools import StructuredTool
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.personas.models import IdentityKernel
from moonlightbox.world.bundles import match_bundle_document_reference
from moonlightbox.world.client import LightRAGSidecarClient, LightRAGSidecarError
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    WorldGraphVersion,
)

from .branch_models import Branch
from .config import (
    STYLE_RETRIEVAL_MAX_CONTEXT_CHARS,
    STYLE_RETRIEVAL_MAX_SOURCE_IDS,
    STYLE_RETRIEVAL_QUERY_PROMPT,
    TOOL_DESCRIPTIONS,
    TOOL_SCHEMAS,
)
from .db_models import RuntimeSnapshotRow
from .schemas import GetStyleExamplesArgs


class StyleService:
    """从最新完整 LightRAG 图谱取回可审计的真人表达证据。

    这里不再把按时间相邻的 ``self → target`` 消息拼成问答样本。Actor 提供本轮
    情境，由 LightRAG 从完整图谱检索语义相近的原文；Actor 只从原文观察表达习惯。
    """

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        lightrag_client: LightRAGSidecarClient | None = None,
    ) -> None:
        self.session = session
        self._settings = settings or Settings()
        # 注入客户端只用于测试或由上层复用连接；由本服务创建的客户端会在查询后关闭。
        self._lightrag_client = lightrag_client

    def tool(self, *, branch_id: str, model_version_id: str) -> StructuredTool:
        """把身份绑定在代码闭包中，模型只能提供当前表达情境。"""

        def invoke(**kwargs: Any) -> dict[str, Any]:
            safe_kwargs = {key: value for key, value in kwargs.items() if key != "model_version_id"}
            args = GetStyleExamplesArgs.model_validate(
                {"model_version_id": model_version_id, **safe_kwargs}
            )
            branch = self.session.get(Branch, branch_id)
            if branch is None or branch.model_version_id != model_version_id:
                return _empty_style_result()
            kernel = self.session.scalar(
                select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
            )
            profile = (
                kernel.content.get("style_profile", {})
                if kernel is not None and isinstance(kernel.content, dict)
                else {}
            )
            examples, truncated = self._semantic_examples(branch, args)
            return {
                "tool_name": "get_style_examples",
                "model_version_id": model_version_id,
                "scope": "world",
                "as_of": datetime.now(UTC).isoformat(),
                "source_ids": _example_source_ids(examples),
                "truncated": truncated,
                "data": {
                    "style_profile": profile,
                    "examples": examples,
                },
            }

        return StructuredTool.from_function(
            name="get_style_examples",
            description=TOOL_DESCRIPTIONS["get_style_examples"],
            func=invoke,
            args_schema=cast(Any, TOOL_SCHEMAS["get_style_examples"]),
        )

    def _semantic_examples(
        self, branch: Branch, args: GetStyleExamplesArgs
    ) -> tuple[list[dict[str, object]], bool]:
        """以本轮情境检索图谱，并把 LightRAG 文档来源还原为消息来源。"""

        if not self._settings.lightrag_enabled:
            return [], False
        snapshot = self.session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch.id)
        )
        if snapshot is None or snapshot.graph_version_id is None:
            return [], False
        graph = self.session.get(WorldGraphVersion, snapshot.graph_version_id)
        if graph is None or graph.project_id != branch.project_id or graph.status != "ready":
            return [], False

        client = self._lightrag_client
        owns_client = client is None
        if client is None:
            client = LightRAGSidecarClient(
                self._settings.lightrag_sidecar_url,
                self._settings.lightrag_sidecar_token.get_secret_value(),
                timeout_seconds=self._settings.lightrag_timeout_seconds,
            )
        try:
            retrieval = client.query(
                graph.workspace_key,
                _style_retrieval_query(args),
                mode="mix",
                top_k=min(self._settings.lightrag_query_top_k, max(4, args.limit * 3)),
                chunk_top_k=min(
                    self._settings.lightrag_query_chunk_top_k, max(2, args.limit * 2)
                ),
                max_total_tokens=min(self._settings.lightrag_query_max_total_tokens, 4000),
            )
        except LightRAGSidecarError:
            # 风格证据是可选增强；图谱暂不可用时仍可凭已训练的 style_profile 写作。
            return [], False
        finally:
            if owns_client:
                client.close()

        document_ids = _referenced_document_ids(self.session, graph.id, retrieval.references)
        if not document_ids:
            # 没有可回溯来源的上下文不能交给 Actor，避免把不可审计内容当真人证据。
            return [], False
        all_source_ids = _document_message_ids(self.session, document_ids)
        if not all_source_ids:
            return [], False
        source_ids = all_source_ids[:STYLE_RETRIEVAL_MAX_SOURCE_IDS]
        context = retrieval.context[:STYLE_RETRIEVAL_MAX_CONTEXT_CHARS]
        return (
            [
                {
                    "selection": "lightrag_semantic_context",
                    "query_version": "runtime-style-retrieval-v1",
                    "source_ids": source_ids,
                    "document_ids": document_ids,
                    "context": context,
                }
            ],
            (
                len(retrieval.context) > len(context)
                or len(all_source_ids) > len(source_ids)
            ),
        )


def _style_retrieval_query(args: GetStyleExamplesArgs) -> str:
    """所有风格检索提示词都从 runtime_v1.config 读取，保持模型协议集中。"""

    return STYLE_RETRIEVAL_QUERY_PROMPT.format(
        situation=args.situation,
        intent=args.intent,
        speech_mode=args.speech_mode,
    )


def _referenced_document_ids(
    session: Session, graph_version_id: str, references: list[Any]
) -> list[str]:
    bundles = list(
        session.scalars(
            select(ConversationBundle).where(
                ConversationBundle.graph_version_id == graph_version_id
            )
        )
    )
    source_to_document = {item.source_name: item.document_id for item in bundles}
    allowed_sources = set(source_to_document)
    matched_sources = [
        match_bundle_document_reference(str(reference.file_path), allowed_sources)
        for reference in references
    ]
    return list(
        dict.fromkeys(
            source_to_document[source]
            for source in matched_sources
            if source is not None and source in source_to_document
        )
    )


def _document_message_ids(session: Session, document_ids: list[str]) -> list[str]:
    if not document_ids:
        return []
    return list(
        session.scalars(
            select(ConversationBundleMessage.message_id)
            .join(ConversationBundle, ConversationBundle.id == ConversationBundleMessage.bundle_id)
            .where(ConversationBundle.document_id.in_(document_ids))
            .order_by(ConversationBundle.ordinal, ConversationBundleMessage.ordinal)
        )
    )


def _empty_style_result() -> dict[str, object]:
    return {
        "tool_name": "get_style_examples",
        "scope": "world",
        "as_of": datetime.now(UTC).isoformat(),
        "source_ids": [],
        "truncated": False,
        "data": {"style_profile": {}, "examples": []},
    }


def _example_source_ids(examples: list[dict[str, object]]) -> list[str]:
    source_ids: list[str] = []
    for item in examples:
        raw = item.get("source_ids")
        if isinstance(raw, list):
            source_ids.extend(source_id for source_id in raw if isinstance(source_id, str))
    return list(dict.fromkeys(source_ids))
