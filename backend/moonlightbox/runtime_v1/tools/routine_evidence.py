"""Planner 自主调查的真实材料工具，不另开隐藏模型或在代码中推断生活规律。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.deadlines import bounded_timeout
from moonlightbox.agent_runtime.resilience import check_interruption
from moonlightbox.agent_runtime.tool_errors import ToolServiceError
from moonlightbox.config import Settings
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.wechat_rendering import normalize_wechat_display
from moonlightbox.world.bundles import match_bundle_document_reference
from moonlightbox.world.client import LightRAGSidecarClient, LightRAGSidecarError
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    WorldGraphVersion,
)

from ..schemas import StrictModel
from ..snapshot_sources import frozen_message_ids, profile_entries, temporal_retrieval_scope


class AnalyzeRoutineEvidenceArgs(StrictModel):
    question: str = Field(
        min_length=4,
        max_length=1200,
        description="当前需要理解的具体问题；可包含主体、生活阶段、相反线索。不要只填关键词。",
    )
    limit: int = Field(
        default=6,
        ge=1,
        le=12,
        description="最多召回的原始资料片段数；空结果不证明没有该活动，partial 表示检索未完整成功",
    )


def query_frozen_history(session, snapshot, question, *, settings=None, client=None, limit=6):
    check_interruption()
    settings = settings or Settings()
    if not settings.lightrag_enabled or not snapshot.graph_version_id:
        raise ToolServiceError("冻结图谱检索不可用；不能将服务未启用当成没有资料")
    graph = session.get(WorldGraphVersion, snapshot.graph_version_id)
    if graph is None or graph.status not in {"ready", "superseded"}:
        raise ToolServiceError("分支绑定图谱不可读，禁止改查全局最新图谱")
    temporal = temporal_retrieval_scope(session, snapshot, graph)
    owned = client is None
    client = client or LightRAGSidecarClient(
        settings.lightrag_sidecar_url,
        settings.lightrag_sidecar_token.get_secret_value(),
        timeout_seconds=bounded_timeout(min(settings.lightrag_timeout_seconds, 90.0)),
    )
    try:
        result = client.query(
            graph.workspace_key,
            question,
            mode="mix",
            top_k=settings.lightrag_query_top_k,
            chunk_top_k=limit,
            max_total_tokens=settings.lightrag_query_max_total_tokens,
            **({"temporal": temporal} if temporal else {}),
        )
    except LightRAGSidecarError as error:
        raise ToolServiceError("冻结图谱检索失败", code=error.code) from error
    finally:
        if owned:
            client.close()
    # 网络返回后先检查旧任务是否失效，避免继续扫描 SQL、恢复消息窗口。
    check_interruption()
    if temporal is not None:
        return _temporal_result(session, snapshot, graph, question, result, temporal)
    bundles = list(
        session.scalars(
            select(ConversationBundle).where(ConversationBundle.graph_version_id == graph.id)
        )
    )
    by_source = {item.source_name: item for item in bundles}
    allowed = frozen_message_ids(session, snapshot)
    rows = {}
    unresolved = []
    for reference in result.references:
        check_interruption()
        source = match_bundle_document_reference(str(reference.file_path), set(by_source))
        if source is None:
            unresolved.append(str(reference.file_path))
            continue
        bundle = by_source[source]
        content = reference.content or bundle.content
        items = _messages_for_reference(session, bundle, content, allowed)
        if not items:
            unresolved.append(str(reference.file_path))
        for item in items:
            rows[item.source_id] = asdict(item)
    if result.references and not rows:
        raise ToolServiceError("检索返回引用但无法还原原始消息；这是引用解析失败，不是没有资料")
    return {
        "query": question,
        "context": result.context,
        "messages": list(rows.values()),
        "source_ids": list(rows),
        "unresolved_references": unresolved,
        "retrieval_status": "partial" if unresolved else "ok",
    }


def _temporal_result(session, snapshot, graph, question, result, scope):
    """历史查询只接受精确映射，重新构造原文上下文，不透传跨期图描述。"""
    import json

    diagnostics = result.temporal_diagnostics or {}
    if any(diagnostics.get(key) != value for key, value in scope.items()):
        raise ToolServiceError("检索服务没有确认冻结范围", code="temporal_scope_mismatch")
    allowed = frozen_message_ids(session, snapshot)
    ids = []
    for reference in result.references:
        data = reference.model_dump()
        mapped = data.get("messages", [])
        if data.get("boundary_relation") != "before" or not mapped:
            raise ToolServiceError("历史引用缺少纯前段映射", code="temporal_scope_violation")
        for message in mapped:
            key = message.get("message_id")
            if key not in allowed or message.get("period") != "before":
                raise ToolServiceError("历史引用越过冻结范围", code="temporal_scope_violation")
            if key not in ids:
                ids.append(key)
    rows = session.execute(
        select(Message, Participant)
        .join(Participant, Participant.id == Message.participant_id)
        .where(Message.project_id == graph.project_id, Message.id.in_(ids))
    ).all()
    by_id = {message.id: (message, participant) for message, participant in rows}
    if set(by_id) != set(ids):
        raise ToolServiceError("冻结引用指向缺失原文", code="temporal_mapping_unavailable")
    messages = [
        {
            "source_id": key,
            "sent_at": by_id[key][0].timestamp.isoformat(),
            "speaker": by_id[key][1].name,
            "role": by_id[key][1].role,
            "kind": by_id[key][0].kind,
            "content": by_id[key][0].content,
        }
        for key in ids
    ]
    return {
        "query": question,
        "context": json.dumps(messages, ensure_ascii=False),
        "messages": messages,
        "source_ids": ids,
        "unresolved_references": [],
        "temporal_diagnostics": diagnostics,
        "retrieval_status": "ok"
        if diagnostics.get("historical_raw_coverage_complete")
        else "partial",
    }


def build_routine_evidence_tool(session, *, snapshot, context, settings=None, lightrag_client=None):
    # 此工具只读资料；所有推断由主 Planner 在可观测的原生工具循环中进行。
    def invoke(**kwargs):
        args = AnalyzeRoutineEvidenceArgs.model_validate(kwargs)
        data = query_frozen_history(
            session,
            snapshot,
            args.question,
            settings=settings,
            client=lightrag_client,
            limit=args.limit,
        )
        data["profile_context"] = profile_entries(snapshot)
        return {
            "tool_name": "analyze_routine_evidence",
            "scope": "world",
            "as_of": snapshot.cutoff_at.isoformat(),
            "source_ids": data["source_ids"],
            "data": data,
        }

    return StructuredTool.from_function(
        name="analyze_routine_evidence",
        func=invoke,
        args_schema=AnalyzeRoutineEvidenceArgs,
        description=(
            "只读调查生活规律：排程所需的时间、条件或阶段有疑问时，按具体问题读取冻结图谱线索"
            "和原始 messages，包含发送者、角色与时间。不会自动提取或平均上下班时间；"
            "需自行分析主体、生活阶段和反例，可调整问题继续查询。"
            "服务失败不是空结果；过长正文通过 read_runtime_result 读取。"
        ),
    )


@dataclass(frozen=True, slots=True)
class _EvidenceMessage:
    source_id: str
    sent_at: str
    speaker: str
    role: str
    kind: str
    content: str


def _messages_for_reference(
    session: Session,
    bundle: ConversationBundle,
    reference_content: str,
    allowed_source_ids: set[str],
) -> list[_EvidenceMessage]:
    """把 LightRAG 文本块的位置还原成 Bundle 中的真实消息。

    这里匹配的是 Sidecar 返回的原始块与其来源文档的行位置，不按消息内容关键词筛选。
    """

    span = _reference_line_span(bundle.content, reference_content)
    if span is None:
        return []
    first, last = span
    first = max(0, first - 2)
    last += 2
    rows = list(
        session.execute(
            select(ConversationBundleMessage.ordinal, Message, Participant)
            .join(Message, Message.id == ConversationBundleMessage.message_id)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                ConversationBundleMessage.bundle_id == bundle.id,
                ConversationBundleMessage.ordinal >= first,
                ConversationBundleMessage.ordinal <= last,
                Participant.role.in_(("self", "target")),
            )
            .order_by(ConversationBundleMessage.ordinal)
        )
    )
    result: list[_EvidenceMessage] = []
    for _, message, participant in rows:
        if message.id not in allowed_source_ids:
            continue
        kind, content = normalize_wechat_display(message.kind, message.content)
        normalized = " ".join(content.split()).strip()
        if not normalized:
            continue
        result.append(
            _EvidenceMessage(
                source_id=message.id,
                sent_at=message.timestamp.isoformat(),
                speaker=participant.name,
                role=participant.role,
                kind=kind,
                content=normalized,
            )
        )
    return result


def _reference_line_span(document: str, reference: str) -> tuple[int, int] | None:
    """定位原始检索块覆盖的文档行；允许块在首尾行的中间被切开。"""

    document_lines = document.splitlines()
    reference_lines = [line for line in reference.splitlines() if line]
    if not document_lines or not reference_lines:
        return None
    best: tuple[int, int] | None = None
    for start in range(len(document_lines)):
        if start + len(reference_lines) > len(document_lines):
            break
        first = document_lines[start]
        if first != reference_lines[0] and not first.endswith(reference_lines[0]):
            continue
        score = 0
        for offset, part in enumerate(reference_lines):
            line = document_lines[start + offset]
            first_fragment = offset == 0 and line.endswith(part)
            last_fragment = offset == len(reference_lines) - 1 and line.startswith(part)
            if line == part or first_fragment or last_fragment:
                score += 1
        if best is None or score > best[0]:
            best = score, start
    if best is not None and best[0] / len(reference_lines) >= 0.8:
        # Sidecar/JSON 传输偶尔会清除微信系统载荷里的控制字符；只允许同一
        # 连续行区间内少量差异，不能像旧实现那样把重复行散落匹配到整份文档。
        return best[1], best[1] + len(reference_lines) - 1
    return None
