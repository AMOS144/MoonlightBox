"""从已批准起点构造栏目时间作用域；模型不能填写边界或资料版本。"""

from dataclasses import dataclass

from moonlightbox.node_investigation.models import NodeInvestigation
from moonlightbox.node_investigation.store import InvestigationStore
from moonlightbox.world.bundles import chronological_source_order


@dataclass(frozen=True)
class NodeCompilationScope:
    graph_id: str
    boundary: dict
    source_version: str
    message_ids: tuple[str, ...]

    @classmethod
    def restore(cls, session, *, graph, envelope):
        """恢复持久化范围，不读取用户后来选择的另一个起点；原文变化则拒绝恢复。"""
        from moonlightbox.world.bundles import source_fingerprint
        from moonlightbox.world.jobs import load_world_messages

        if not isinstance(envelope, dict) or envelope.get("mode") != "retrospective_node":
            raise ValueError("节点范围契约缺失或无效")
        messages = load_world_messages(
            session, project_id=graph.project_id, import_ids=graph.source_import_ids
        )
        ids = tuple(chronological_source_order(messages))
        count = envelope.get("included_count")
        if (
            envelope.get("graph_version_id") != graph.id
            or envelope.get("source_version") != graph.source_fingerprint
            or source_fingerprint(messages) != graph.source_fingerprint
            or type(count) is not int
            or not 0 <= count <= len(ids)
        ):
            raise ValueError("节点范围与当前图谱来源不一致")
        if envelope.get("last_included_ref") != (ids[count - 1] if count else None) or envelope.get(
            "first_excluded_ref"
        ) != (ids[count] if count < len(ids) else None):
            raise ValueError("节点边界前后引用不一致")
        from datetime import datetime
        from zoneinfo import ZoneInfo

        if datetime.fromisoformat(envelope["cutoff_at"]).tzinfo is None:
            raise ValueError("节点边界缺少时区")
        ZoneInfo(envelope["timezone"])
        boundary = {
            k: v
            for k, v in envelope.items()
            if k
            not in {
                "graph_version_id",
                "source_version",
                "before_message_count",
                "after_message_count",
                "mode",
            }
        }
        restored = cls(graph.id, boundary, graph.source_fingerprint, ids)
        if restored.envelope() != envelope:
            raise ValueError("节点范围摘要不一致")
        return restored

    @classmethod
    def from_approved(cls, session, *, graph, investigation_id, preview_hash):
        # 复用现有批准契约，不另写一套“近似起点”的解析规则。
        from moonlightbox.runtime_v1.origin_boundary import approved_boundary
        from moonlightbox.world.jobs import load_world_messages

        boundary = approved_boundary(session, graph.project_id, investigation_id, preview_hash)
        investigation = session.get(NodeInvestigation, investigation_id)
        messages = load_world_messages(
            session, project_id=graph.project_id, import_ids=graph.source_import_ids
        )
        order = chronological_source_order(messages)
        ids = tuple(order)
        if ids != tuple(investigation.state["message_ids"]):
            raise ValueError("批准起点的资料清单与图谱来源不同，不能直接套用边界序号")
        count = boundary.get("included_count")
        if type(count) is not int or not 0 <= count <= len(ids):
            raise ValueError("批准边界的消息数量无效，请重新确认起点")
        if boundary.get("last_included_ref") != (ids[count - 1] if count else None) or boundary.get(
            "first_excluded_ref"
        ) != (ids[count] if count < len(ids) else None):
            raise ValueError("批准边界的前后消息与来源顺序不一致")
        # 批准校验检查过当前原文；再次比对图谱所绑定的原文版本。
        from moonlightbox.world.bundles import source_fingerprint

        if source_fingerprint(messages) != graph.source_fingerprint:
            raise ValueError("图谱原文版本已经变化，请重新构建来源索引")
        return cls(graph.id, boundary, graph.source_fingerprint, ids)

    def retrieval_scope(self):
        return {
            "source_version": self.source_version,
            "included_count": self.boundary["included_count"],
            "access": "compiler",
            "policy": "mixed_to_before_v1",
        }

    def for_candidate_graph(self, session, graph):
        """显式继承已批准图修订的来源边界，不能借换图更换时间范围。"""
        if graph.parent_version_id != self.graph_id:
            raise ValueError("节点范围只能传给绑定图谱的直接修订候选")
        return type(self).restore(
            session,
            graph=graph,
            envelope={
                **self.envelope(),
                "graph_version_id": graph.id,
            },
        )

    def envelope(self):
        return {
            **self.boundary,
            "graph_version_id": self.graph_id,
            "source_version": self.source_version,
            "before_message_count": self.boundary["included_count"],
            "after_message_count": len(self.message_ids) - self.boundary["included_count"],
            "mode": "retrospective_node",
        }

    def message_periods(self):
        included = self.boundary["included_count"]
        return {
            key: "before" if index < included else "after"
            for index, key in enumerate(self.message_ids)
        }

    def model_context(self):
        """模型只需要理解目标时刻与资料范围，不需要填写内部版本/hash。"""
        full = self.envelope()
        return {
            key: full[key]
            for key in (
                "cutoff_at",
                "timezone",
                "before_message_count",
                "after_message_count",
                "mode",
            )
        }

    def initial_messages(self, session):
        """只给前段入口，不把后来的实际回复当成起点已发生的事。"""
        included = self.boundary["included_count"]
        # 最近连续原文 + 前段早期入口；不是全文预读或语义抽样判定。
        indexes = sorted(
            set(range(max(0, included - 30), included))
            | ({0, included // 3, included * 2 // 3} if included else set())
        )
        ids = [self.message_ids[i] for i in indexes]
        # source_rows 需要项目边界，直接从已批准调查绑定，不能从模型输入取得。
        row = session.get(NodeInvestigation, self.boundary["investigation_id"])
        store = InvestigationStore(
            session.get_bind(), row.project_id, self.boundary["investigation_id"]
        )
        rows = store.source_rows(ids, session=session)
        from sqlalchemy import select

        from moonlightbox.world.models import ConversationBundle, ConversationBundleMessage

        mappings = session.execute(
            select(ConversationBundleMessage, ConversationBundle)
            .join(ConversationBundle, ConversationBundle.id == ConversationBundleMessage.bundle_id)
            .where(
                ConversationBundle.graph_version_id == self.graph_id,
                ConversationBundleMessage.message_id.in_(ids),
            )
        ).all()
        by_id = {mapping.message_id: (mapping, bundle) for mapping, bundle in mappings}
        result = []
        for message, participant in rows:
            if message.id not in by_id:
                continue
            mapping, bundle = by_id[message.id]
            result.append(
                {
                    "message_id": message.id,
                    "timestamp": message.timestamp.isoformat(),
                    "participant_id": participant.id,
                    "participant_name": participant.name,
                    "participant_role": participant.role,
                    "kind": message.kind,
                    "content": message.content,
                    "source_period": "before",
                    "mapping_method": "frozen_node_prefix",
                    "document_id": bundle.document_id,
                    "bundle_id": bundle.id,
                    "ordinal": mapping.ordinal,
                    "is_primary_match": False,
                }
            )
        return result


def inherited_node_scope(session, graph, *containers):
    """各入口只从宿主持久化对象继承范围；不允许两个基线混用不同起点。"""
    values = [item.get("node_scope") for item in containers if isinstance(item, dict)]
    scoped = [value for value in values if value is not None]
    if not scoped:
        return None
    if any(value != scoped[0] for value in values):
        raise ValueError("运行、画像或纠正会话的节点范围不一致，禁止回退全量范围")
    return NodeCompilationScope.restore(session, graph=graph, envelope=scoped[0])


def same_node_scope(left, right):
    """纠正随同一节点的图修订延续，但不能跨来源版本或另一个批准起点。"""
    if left is None or right is None:
        return left is right
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    keys = (
        "source_version",
        "preview_hash",
        "included_count",
        "cutoff_at",
        "timezone",
        "last_included_ref",
        "first_excluded_ref",
    )
    return all(key in left and key in right and left[key] == right[key] for key in keys)
