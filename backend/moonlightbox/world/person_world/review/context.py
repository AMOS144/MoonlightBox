"""Revision Agent 的确定性上下文装配器。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.world.models import (
    AtomicWorldClaim,
    PersonWorldProfile,
    PersonWorldRevisionContextSnapshot,
    PersonWorldRevisionSession,
    WorldCorrection,
    WorldEvidence,
    WorldGraphVersion,
)


@dataclass(frozen=True, slots=True)
class AssembledRevisionContext:
    """模型本回合看到的有界内容，以及其可复现的数据库快照。"""

    snapshot: PersonWorldRevisionContextSnapshot
    model_payload: dict[str, object]


class RevisionContextAssembler:
    """冻结 Revision 回合的版本、范围和只读证据来源。

    该类只拼装已有结构与工具结果，不判断用户话语属于哪种纠正，也不从展示文案反查事实。
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def assemble(
        self,
        *,
        revision: PersonWorldRevisionSession,
        graph: WorldGraphVersion,
        profile: PersonWorldProfile | None,
        conversation: list[dict[str, str]],
        selected_statements: list[dict[str, object]],
        source_messages: list[dict[str, object]],
        related_graph_context: str,
        graph_references: list[dict[str, object]],
        current_profile: dict[str, object],
        persist: bool = True,
    ) -> AssembledRevisionContext:
        """创建不可变快照，并返回本轮模型所需的最小上下文。"""

        from ..node_scope import inherited_node_scope
        node_scope = inherited_node_scope(
            self.session, graph, revision.scope,
            profile.generation_summary if profile is not None else {},
        )
        selected_claims = self._selected_claims(revision, graph)
        related_claims = self._related_claims(graph, selected_claims)
        claim_message_windows = self._claim_message_windows(
            revision.project_id,
            [*selected_claims, *related_claims],
        )
        # 工具读到的材料与 Scope 所绑定的证据窗口都要进入同一回合的可重放集合；前者
        # 不会覆盖后者，也不按文本相似度去重，只按真实 message_id 去重。
        all_source_messages = _dedupe_messages([*claim_message_windows, *source_messages])
        if node_scope:
            periods = node_scope.message_periods()
            if any(item.get("message_id") not in periods for item in all_source_messages):
                raise ValueError("纠正原文不属于节点冻结资料")
            all_source_messages = [
                {**item, "source_period": periods[item["message_id"]]}
                for item in all_source_messages
            ]
        evidence_message_ids = [
            str(item["message_id"])
            for item in all_source_messages
            if isinstance(item.get("message_id"), str)
        ]
        active_corrections = self._active_corrections(
            revision.project_id, node_scope.envelope() if node_scope else None
        )
        # Graph 原文摘要会随图版本变化；快照保存其 hash 与引用清单，而非再复制一份长文本。
        graph_manifest = {
            "context_hash": canonical_context_hash(related_graph_context),
            "references": [
                {
                    "reference_id": item.get("reference_id"),
                    "document_name": item.get("document_name"),
                    "reference_rank": item.get("reference_rank"),
                    "document_rank": item.get("document_rank"),
                }
                for item in graph_references
                if isinstance(item, dict)
            ],
        }
        payload: dict[str, object] = {
            # 内部 UUID、Graph ref 与工作 Job 身份只留在 Snapshot；模型只看到可读的
            # Scope 轮廓和稳定展示序号，不能把数据库 ID 复制进下一阶段的草稿。
            "revision_scope": _model_scope(revision.scope),
            "current_profile": current_profile,
            "revision_messages": conversation,
            "selected_profile_statements": _model_selected_statements(selected_statements),
            "selected_claims": [_claim_for_model(item) for item in selected_claims],
            "related_claims": [
                _related_claim_for_model(item, selected_claims, candidate_item=index)
                for index, item in enumerate(related_claims, start=1)
            ],
            "active_corrections": active_corrections,
            "human_assertions": [
                {"content": item["content"], "turn": index}
                for index, item in enumerate(conversation, start=1)
                if item.get("role") == "user" and isinstance(item.get("content"), str)
            ],
            "related_graph_context": related_graph_context,
            "source_messages": all_source_messages,
        }
        snapshot = PersonWorldRevisionContextSnapshot(
            session_id=revision.id,
            session_revision=revision.session_revision,
            base_graph_version_id=graph.id,
            base_profile_id=profile.id if profile is not None else None,
            scope=dict(revision.scope or {}),
            evidence_message_ids=list(dict.fromkeys(evidence_message_ids)),
            related_claim_ids=list(
                dict.fromkeys([item.id for item in [*selected_claims, *related_claims]])
            ),
            graph_manifest=graph_manifest,
            context_hash=canonical_context_hash(payload),
        )
        assembled = AssembledRevisionContext(snapshot=snapshot, model_payload=payload)
        if persist:
            assembled = self.persist_snapshot(assembled, revision)
        return assembled

    def persist_snapshot(self, assembled, revision):
        """只有工具接受本回合交付后冻结最后一次模型实际读取的快照，调查中不反复落库。"""
        existing = self.session.scalar(
            select(PersonWorldRevisionContextSnapshot).where(
                PersonWorldRevisionContextSnapshot.session_id == assembled.snapshot.session_id,
                PersonWorldRevisionContextSnapshot.session_revision
                == assembled.snapshot.session_revision,
            )
        )
        if existing is not None:
            if existing.context_hash != assembled.snapshot.context_hash:
                raise ValueError("同一纠正回合不能覆盖已冻结的不同上下文")
            revision.context_snapshot_id = existing.id
            return AssembledRevisionContext(existing, assembled.model_payload)
        self.session.add(assembled.snapshot)
        self.session.flush()
        revision.context_snapshot_id = assembled.snapshot.id
        return assembled

    def _selected_claims(
        self,
        revision: PersonWorldRevisionSession,
        graph: WorldGraphVersion,
    ) -> list[AtomicWorldClaim]:
        claim_ids = _scope_claim_ids(revision.scope)
        if not claim_ids:
            return []
        rows = list(
            self.session.scalars(
                select(AtomicWorldClaim)
                .where(
                    AtomicWorldClaim.project_id == revision.project_id,
                    AtomicWorldClaim.graph_version_id == graph.id,
                    AtomicWorldClaim.id.in_(claim_ids),
                )
                .order_by(AtomicWorldClaim.created_at.asc(), AtomicWorldClaim.id.asc())
            )
        )
        # 维持用户圈选顺序，而不是数据库碰巧返回的顺序；未找到的 ID 留在 Snapshot Scope，
        # 但不会被伪装成一条可供模型使用的事实。
        by_id = {item.id: item for item in rows}
        return [by_id[item] for item in claim_ids if item in by_id]

    def _related_claims(
        self,
        graph: WorldGraphVersion,
        selected: list[AtomicWorldClaim],
    ) -> list[AtomicWorldClaim]:
        """返回结构上可能相关的候选，不对中文正文或事实真伪作任何判断。"""

        if not selected:
            return []
        # 只按持久化主体、顶层栏目与事实类型列查询。这里既不读取 object_text，也不把
        # 返回项称作冲突；它们只是需要交给用户和 Agent 决定是否纳入的候选。
        selected_ids = {item.id for item in selected}
        conditions = [_related_claim_condition(item) for item in selected]
        return list(
            self.session.scalars(
                select(AtomicWorldClaim)
                .where(
                    AtomicWorldClaim.graph_version_id == graph.id,
                    AtomicWorldClaim.id.not_in(selected_ids),
                    or_(*conditions),
                )
                .order_by(AtomicWorldClaim.created_at.asc(), AtomicWorldClaim.id.asc())
                .limit(60)
            )
        )

    def _claim_message_windows(
        self,
        project_id: str,
        claims: list[AtomicWorldClaim],
    ) -> list[dict[str, object]]:
        evidence_ids = {
            evidence_id
            for claim in claims
            for evidence_id in claim.evidence_ids
            if isinstance(evidence_id, str)
        }
        if not evidence_ids:
            return []
        evidence = list(
            self.session.scalars(select(WorldEvidence).where(WorldEvidence.id.in_(evidence_ids)))
        )
        message_ids = list(
            dict.fromkeys(
                message_id
                for item in evidence
                for message_id in [
                    item.message_id,
                    *list(item.context_before_ids or []),
                    *list(item.context_after_ids or []),
                ]
                if isinstance(message_id, str)
            )
        )
        if not message_ids:
            return []
        rows = list(
            self.session.execute(
                select(Message, Participant)
                .join(Participant, Participant.id == Message.participant_id)
                .where(Message.project_id == project_id, Message.id.in_(message_ids))
                .order_by(Message.timestamp.asc(), Message.id.asc())
            )
        )
        primary = {item.message_id for item in evidence}
        return [
            {
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat(),
                "participant_id": participant.id,
                "participant_name": participant.name,
                "participant_role": participant.role,
                "kind": message.kind,
                "content": message.content,
                "is_primary_match": message.id in primary,
                "source": "claim_evidence" if message.id in primary else "claim_context",
            }
            for message, participant in rows
        ]

    def _active_corrections(self, project_id: str, node_scope=None) -> list[dict[str, object]]:
        """读取已生效的人类纠正；状态是数据库字段，不通过文本猜测其适用范围。"""

        from ..node_scope import same_node_scope
        corrections = self.session.scalars(
            select(WorldCorrection)
            .where(
                WorldCorrection.project_id == project_id,
                WorldCorrection.status == "active",
                WorldCorrection.corrected_interpretation["scope"]["node_scope"]["preview_hash"].as_string()
                == (node_scope or {}).get("preview_hash"),
            )
            .order_by(WorldCorrection.approved_at.asc(), WorldCorrection.created_at.asc())
            .limit(40)
        )
        return [
            {
                "correction_type": item.correction_type,
                "original_interpretation": dict(item.original_interpretation or {}),
                "corrected_interpretation": _without_internal_scope(
                    dict(item.corrected_interpretation or {})
                ),
                "source_message_ids": list(item.source_message_ids or []),
            }
            for item in corrections
            if same_node_scope(
                ((item.corrected_interpretation or {}).get("scope") or {}).get("node_scope"),
                node_scope,
            )
        ]


def _scope_claim_ids(scope: object) -> list[str]:
    if not isinstance(scope, dict):
        return []
    values = scope.get("selected_claim_ids", [])
    return list(dict.fromkeys(item for item in values if isinstance(item, str)))


def _related_claim_condition(selected: AtomicWorldClaim) -> object:
    """构造同主体、同顶层栏目、同事实类型的纯 SQL 候选条件。

    ``unknown`` 分支仅兼容尚未跑 0049 的老数据；它检查的是规范路径列，不是聊天文案。
    """

    subject_name = (
        AtomicWorldClaim.subject_name.is_(None)
        if selected.subject_name is None
        else AtomicWorldClaim.subject_name == selected.subject_name
    )
    domain = _primary_domain(selected)
    fact_type = _fact_type(selected)
    return and_(
        AtomicWorldClaim.subject_kind == selected.subject_kind,
        subject_name,
        or_(
            AtomicWorldClaim.primary_domain == domain,
            and_(
                AtomicWorldClaim.primary_domain == "unknown",
                AtomicWorldClaim.profile_section.like(f"{domain}.%"),
            ),
        ),
        or_(
            AtomicWorldClaim.fact_type == fact_type,
            and_(
                AtomicWorldClaim.fact_type == "unknown",
                AtomicWorldClaim.predicate == fact_type,
            ),
        ),
    )


def _intervals_overlap(left: AtomicWorldClaim, right: AtomicWorldClaim) -> bool:
    """仅在两条事实都有边界时做数值时间区间相交；未知时间不被臆作重叠。"""

    if left.valid_from is None or left.valid_to is None:
        return False
    if right.valid_from is None or right.valid_to is None:
        return False
    return left.valid_from <= right.valid_to and right.valid_from <= left.valid_to


def _claim_for_model(claim: AtomicWorldClaim) -> dict[str, object]:
    return {
        "section": claim.profile_section,
        "primary_domain": _primary_domain(claim),
        "fact_type": _fact_type(claim),
        "statement": claim.object_text,
        "assertion_kind": claim.assertion_kind,
        "derivation": claim.derivation,
        "temporal_status": claim.temporal_status,
        "valid_from": claim.valid_from.isoformat() if claim.valid_from else None,
        "valid_to": claim.valid_to.isoformat() if claim.valid_to else None,
        "evidence_count": len(claim.evidence_ids or []),
    }


def _related_claim_for_model(
    claim: AtomicWorldClaim,
    selected: list[AtomicWorldClaim],
    *,
    candidate_item: int,
) -> dict[str, object]:
    relations: list[str] = []
    for selected_claim in selected:
        if _fact_type(claim) == _fact_type(selected_claim):
            relations.append("same_subject_and_fact_type")
        if (
            claim.profile_section == selected_claim.profile_section
            and claim.normalized_text == selected_claim.normalized_text
        ):
            relations.append("same_fact_key")
        if _intervals_overlap(claim, selected_claim):
            relations.append("temporal_overlap")
    return {
        **_claim_for_model(claim),
        "candidate_item": candidate_item,
        "structural_relations": list(dict.fromkeys(relations)),
    }


def _primary_domain(claim: AtomicWorldClaim) -> str:
    """兼容迁移前 Claim；回填后不再依赖字符串拆分。"""

    if claim.primary_domain and claim.primary_domain != "unknown":
        return claim.primary_domain
    return claim.profile_section.split(".", 1)[0]


def _fact_type(claim: AtomicWorldClaim) -> str:
    if claim.fact_type and claim.fact_type != "unknown":
        return claim.fact_type
    return claim.predicate


def _dedupe_messages(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in rows:
        message_id = item.get("message_id")
        if not isinstance(message_id, str) or message_id in seen:
            continue
        seen.add(message_id)
        result.append(item)
    return result


def _model_scope(scope: object) -> dict[str, object]:
    """生成无 UUID/图对象 ID 的 Scope 描述，保留用户能理解的范围边界。"""

    if not isinstance(scope, dict):
        return {}
    return {
        "selected_item_count": len(_scope_claim_ids(scope)),
        "included_graph_object_count": len(
            scope.get("included_graph_objects", [])
            if isinstance(scope.get("included_graph_objects"), list)
            else []
        ),
        "excluded_item_count": len(
            scope.get("excluded_claim_ids", [])
            if isinstance(scope.get("excluded_claim_ids"), list)
            else []
        ),
        "provisional_target": scope.get("provisional_target"),
        "node_scope": {
            key: scope["node_scope"].get(key)
            for key in (
                "cutoff_at", "timezone", "before_message_count", "after_message_count", "mode"
            )
        } if isinstance(scope.get("node_scope"), dict) else None,
    }


def _model_selected_statements(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scope_item": index,
            "section": item.get("section"),
            "section_label": item.get("section_label"),
            "text": item.get("text"),
            "source_message_ids": item.get("source_message_ids", []),
        }
        for index, item in enumerate(rows, start=1)
        if isinstance(item, dict)
    ]


def _without_internal_scope(value: dict[str, object]) -> dict[str, object]:
    """纠正文本可见，Claim/图对象身份只留在服务端快照。"""

    scope = value.get("scope")
    if not isinstance(scope, dict):
        return value
    return {
        **value,
        "scope": {
            key: item
            for key, item in scope.items()
            if key not in {"claim_ids", "graph_object_refs"}
        },
    }


def canonical_context_hash(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
