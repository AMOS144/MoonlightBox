"""按图版本装配栏目只读工具；不依赖 v2/v3 的工作流或事实装配策略。"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Participant
from moonlightbox.world.models import ConversationBundle, WorldGraphVersion

from ..investigation_artifacts import InvestigationArtifactStore
from .current_profile import build_current_profile_tools
from .evidence_page import build_evidence_page_tool
from .graph_query import build_graph_query_tools
from .section_work import build_section_work_tool
from .source_messages import build_source_message_tools
from .temporal_analysis import build_temporal_analysis_tools


def participant_id(session: Session, *, project_id: str, role: str, name: str) -> str:
    participant = session.scalar(
        select(Participant)
        .where(
            Participant.project_id == project_id,
            Participant.role == role,
            Participant.name == name,
        )
        .order_by(Participant.id.asc())
    )
    if participant is None:
        raise ValueError(f"PersonWorld 缺少 {role} 参与者")
    return participant.id


def build_section_tools(
    session: Session,
    *,
    graph: WorldGraphVersion,
    lightrag: Any,
    top_k: int,
    chunk_top_k: int,
    max_total_tokens: int,
    correction_change_set_ids: set[str],
    artifacts: InvestigationArtifactStore,
    temporal_scope: dict[str, object] | None = None,
    message_periods: dict[str, str] | None = None,
    node_scope: dict | None = None,
    profile_snapshot: dict | None = None,
) -> dict[str, Any]:
    """每次调用创建独立检索工件与 Session 绑定，不跨栏目复用可变结果。"""
    sources = set(
        session.scalars(
            select(ConversationBundle.source_name).where(
                ConversationBundle.graph_version_id == graph.id,
            )
        )
    )
    return {
        "save_section_work": build_section_work_tool(artifacts),
        "read_evidence_page": build_evidence_page_tool(artifacts),
        **build_graph_query_tools(
            lightrag,
            workspace=graph.workspace_key,
            allowed_document_names=sources,
            artifacts=artifacts,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_total_tokens=max_total_tokens,
            temporal_scope=temporal_scope,
        ),
        **build_source_message_tools(
            session,
            project_id=graph.project_id,
            graph_version_id=graph.id,
            artifacts=artifacts,
            message_periods=message_periods,
        ),
        **build_current_profile_tools(
            session,
            project_id=graph.project_id,
            graph_version_id=graph.id,
            correction_change_set_ids=correction_change_set_ids,
            node_scope=node_scope,
            profile_snapshot=profile_snapshot,
        ),
        **build_temporal_analysis_tools(
            session, project_id=graph.project_id, graph_version_id=graph.id,
            timezone_name=(node_scope or {}).get("timezone", "Asia/Shanghai"),
        ),
    }
