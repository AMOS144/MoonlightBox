"""冻结分支资料的统一读取边界与画像引用，不读取全局最新候选。"""

import hashlib

from sqlalchemy import select

from moonlightbox.imports.models import Message
from moonlightbox.world.models import ConversationBundle, ConversationBundleMessage


def temporal_retrieval_scope(session, snapshot, graph):
    """历史权限属于冻结快照，不能随虚拟钟推进或由模型的查询参数扩大。

    历史快照必须保存完整的前缀消息清单，而非画像恰好引用的证据 ID。
    校验清单与图来源一致后才能把序号交给 Sidecar，缺少契约时禁止退回全库查询。
    """
    if snapshot.snapshot_mode == "latest_profile":
        return None
    from moonlightbox.agent_runtime.tool_errors import ToolServiceError
    from moonlightbox.world.bundles import chronological_source_order, source_fingerprint
    from moonlightbox.world.jobs import load_world_messages

    binding = (snapshot.profile or {}).get("_runtime_binding", {})
    scope = binding.get("temporal_scope")
    if not isinstance(scope, dict):
        raise ToolServiceError(
            "历史快照缺少冻结检索边界，不能改查全图", code="temporal_scope_missing"
        )
    count = scope.get("included_count")
    if (
        type(count) is not int
        or count < 0
        or scope.get("source_version") != graph.source_fingerprint
    ):
        raise ToolServiceError("历史快照来源或边界不匹配", code="temporal_scope_mismatch")
    messages = load_world_messages(
        session, project_id=graph.project_id, import_ids=graph.source_import_ids
    )
    ids = list(chronological_source_order(messages))
    if (
        count > len(ids)
        or list(snapshot.source_message_ids or []) != ids[:count]
        or source_fingerprint(messages) != graph.source_fingerprint
    ):
        raise ToolServiceError("冻结原文前缀与图谱来源不一致", code="temporal_source_mismatch")
    return {
        "source_version": graph.source_fingerprint,
        "included_count": count,
        "period": "before",
        "access": "runtime",
        "policy": "mixed_to_before_v1",
    }


def profile_entries(snapshot):
    if snapshot.profile.get("profile_schema_version") != "v3":
        return []
    entries = []

    def visit(value, path):
        if isinstance(value, dict):
            if value.get("status") == "unknown":
                return
            for key, item in value.items():
                if (
                    key in {"description", "summary", "overview", "explanation"}
                    and isinstance(item, str)
                    and item.strip()
                ):
                    field = f"{path}.{key}"
                    binding = snapshot.profile.get("_runtime_binding", {})
                    identity = (
                        f"{snapshot.id}:{binding['profile_id']}"
                        if binding.get("profile_id")
                        else snapshot.id
                    )
                    ref = "profile:" + hashlib.sha256(f"{identity}:{field}".encode()).hexdigest()
                    entries.append(
                        {
                            "source_id": ref,
                            "field_path": field,
                            "text": item,
                            "basis": value.get("basis", "inferred"),
                        }
                    )
                else:
                    visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, f"{path}.{i}")

    visit(snapshot.profile, "profile")
    return entries


def frozen_message_ids(session, snapshot):
    ids = set(ordered_frozen_message_ids(session, snapshot))
    # 旧末端快照的记忆来源标签保留读取兼容；原文列表仍只返回真实存在的消息。
    if snapshot.snapshot_mode == "latest_profile":
        ids.update(snapshot.source_message_ids or [])
    return ids


def ordered_frozen_message_ids(session, snapshot):
    """所有原文入口共用顺序与权限；历史范围不再由各工具自行比较时间。"""
    if snapshot.snapshot_mode != "latest_profile":
        from moonlightbox.agent_runtime.tool_errors import ToolServiceError
        from moonlightbox.world.models import WorldGraphVersion

        graph = session.get(WorldGraphVersion, snapshot.graph_version_id)
        if graph is None:
            raise ToolServiceError("冻结图谱不存在", code="temporal_graph_missing")
        temporal_retrieval_scope(session, snapshot, graph)
        return list(snapshot.source_message_ids or [])
    allowed = set(snapshot.source_message_ids or [])
    if snapshot.snapshot_mode == "latest_profile" and snapshot.graph_version_id:
        # 最新导入末端模式下，边界是绑定图谱及 cutoff，不是画像编译恰好引用的少量消息。
        allowed.update(
            session.scalars(
                select(ConversationBundleMessage.message_id)
                .join(
                    ConversationBundle, ConversationBundle.id == ConversationBundleMessage.bundle_id
                )
                .join(Message, Message.id == ConversationBundleMessage.message_id)
                .where(
                    ConversationBundle.graph_version_id == snapshot.graph_version_id,
                    Message.timestamp <= snapshot.cutoff_at,
                )
            )
        )
    from moonlightbox.imports.message_order import message_order_key

    rows = list(session.scalars(select(Message).where(Message.id.in_(allowed))))
    return [
        message.id
        for message in sorted(
            rows, key=lambda m: message_order_key(m.timestamp, m.import_id, m.source_id, m.id)
        )
    ]


def frozen_source_ids(session, snapshot):
    return frozen_message_ids(session, snapshot) | {
        item["source_id"] for item in profile_entries(snapshot)
    }
