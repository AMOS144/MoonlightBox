"""只执行已经批准的候选图变更；模型不能绕过本服务直接写 LightRAG。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import (
    WorldChangeApproval,
    WorldGraphChangeSet,
    WorldGraphOperationLog,
    WorldGraphVersion,
)

from .schemas import GraphOperationDraft


class GraphPatchValidationError(RuntimeError):
    pass


class WorldGraphExecutor:
    """批准校验、幂等日志和候选 workspace CRUD 的唯一入口。"""

    def __init__(self, session: Session, client: LightRAGSidecarClient) -> None:
        self.session = session
        self.client = client

    def apply(
        self,
        *,
        change_set: WorldGraphChangeSet,
        candidate_graph: WorldGraphVersion,
    ) -> dict[str, object]:
        self._validate(change_set, candidate_graph)
        operations = [
            GraphOperationDraft.model_validate(item) for item in change_set.graph_operations
        ]
        ordered = sorted(operations, key=_operation_order)
        results: list[dict[str, object]] = []
        change_set.status = "applying"
        candidate_graph.status = "applying_patch"
        self.session.flush()
        try:
            for operation in ordered:
                result = self._apply_one(change_set, candidate_graph, operation)
                results.append(
                    {
                        "operation_id": operation.operation_id,
                        "operation_type": operation.operation_type,
                        "result": result,
                    }
                )
        except Exception:
            change_set.status = "validation_failed"
            candidate_graph.status = "failed"
            self.session.flush()
            raise
        change_set.status = "applied"
        candidate_graph.status = "validating"
        payload: dict[str, object] = {
            "operations": results,
            "applied_at": datetime.now(UTC).isoformat(),
        }
        change_set.execution_result = payload
        self.session.flush()
        return payload

    def replay(
        self,
        *,
        change_set: WorldGraphChangeSet,
        candidate_graph: WorldGraphVersion,
    ) -> list[dict[str, object]]:
        """在重建的候选图中重放已经发布过的纠正。"""

        if candidate_graph.status not in {"candidate", "applying_patch"}:
            raise GraphPatchValidationError("纠正只能重放到 candidate workspace")
        approval = self.session.scalar(
            select(WorldChangeApproval).where(
                WorldChangeApproval.change_set_id == change_set.id,
                WorldChangeApproval.approval_stage == "graph",
                WorldChangeApproval.payload_hash == change_set.canonical_payload_hash,
            )
        )
        if approval is None:
            raise GraphPatchValidationError("历史纠正缺少有效 Graph Approval")
        return [
            self._apply_one(change_set, candidate_graph, operation)
            for operation in sorted(
                [GraphOperationDraft.model_validate(item) for item in change_set.graph_operations],
                key=_operation_order,
            )
        ]

    def _validate(
        self,
        change_set: WorldGraphChangeSet,
        candidate_graph: WorldGraphVersion,
    ) -> None:
        if change_set.status != "approved":
            raise GraphPatchValidationError("Graph ChangeSet 尚未得到最终批准")
        approvals = list(
            self.session.scalars(
                select(WorldChangeApproval).where(
                    WorldChangeApproval.change_set_id == change_set.id,
                    WorldChangeApproval.approval_stage.in_(["profile", "graph"]),
                    WorldChangeApproval.revision == change_set.revision,
                )
            )
        )
        profile_hash = canonical_profile_patch_hash(
            profile_patch=change_set.profile_patch,
            base_graph_version_id=change_set.base_graph_version_id,
            revision=change_set.revision,
        )
        approval_hashes = {item.approval_stage: item.payload_hash for item in approvals}
        if approval_hashes != {
            "profile": profile_hash,
            "graph": change_set.canonical_payload_hash,
        }:
            raise GraphPatchValidationError("Profile Patch 与 Graph Patch 未分别完成批准")
        expected_hash = canonical_change_set_hash(
            profile_patch=change_set.profile_patch,
            graph_operations=change_set.graph_operations,
            regression_queries=change_set.regression_queries,
            base_graph_version_id=change_set.base_graph_version_id,
            revision=change_set.revision,
        )
        if expected_hash != change_set.canonical_payload_hash:
            raise GraphPatchValidationError("ChangeSet 内容已经发生变化，必须重新批准")
        if candidate_graph.parent_version_id != change_set.base_graph_version_id:
            raise GraphPatchValidationError("候选图不是从用户批准的基础版本建立")
        if candidate_graph.status not in {"candidate", "applying_patch"}:
            raise GraphPatchValidationError("图谱 CRUD 只能作用于 candidate workspace")
        if candidate_graph.change_set_id != change_set.id:
            raise GraphPatchValidationError("候选图没有绑定当前 ChangeSet")
        operations = [
            GraphOperationDraft.model_validate(item) for item in change_set.graph_operations
        ]
        operation_ids = [item.operation_id for item in operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise GraphPatchValidationError("Graph Patch 包含重复 operation_id")
        if any(item.precondition_hash is None for item in operations):
            raise GraphPatchValidationError("Graph Patch 缺少服务端生成的 precondition_hash")
        if any(item.operation_type == "DELETE_ENTITY" and not item.cascade for item in operations):
            raise GraphPatchValidationError("删除实体必须显式批准 cascade=true")
        if any(
            item.operation_type == "MERGE_ENTITIES"
            and (
                not item.source_entities
                or item.target_entity is None
                or item.target_entity in item.source_entities
            )
            for item in operations
        ):
            raise GraphPatchValidationError("实体归并的 source/target 无效")

    def _apply_one(
        self,
        change_set: WorldGraphChangeSet,
        graph: WorldGraphVersion,
        operation: GraphOperationDraft,
    ) -> dict[str, object]:
        existing = self.session.scalar(
            select(WorldGraphOperationLog).where(
                WorldGraphOperationLog.change_set_id == change_set.id,
                WorldGraphOperationLog.graph_version_id == graph.id,
                WorldGraphOperationLog.operation_id == operation.operation_id,
            )
        )
        if existing is not None and existing.status == "succeeded":
            return dict(existing.response_payload or {})
        log = existing or WorldGraphOperationLog(
            change_set_id=change_set.id,
            graph_version_id=graph.id,
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            status="pending",
            request_payload=operation.model_dump(mode="json"),
        )
        if existing is None:
            self.session.add(log)
        self.session.flush()
        try:
            self._validate_precondition(graph.workspace_key, operation)
            result = self._dispatch(graph.workspace_key, operation, change_set.id)
        except Exception as error:
            log.status = "failed"
            log.error_message = f"{type(error).__name__}: {error}"[:2000]
            log.completed_at = datetime.now(UTC)
            self.session.flush()
            raise
        log.status = "succeeded"
        log.response_payload = result
        log.error_message = None
        log.completed_at = datetime.now(UTC)
        self.session.flush()
        return result

    def _validate_precondition(self, workspace: str, operation: GraphOperationDraft) -> None:
        if operation.precondition_hash is None:
            raise GraphPatchValidationError(
                f"操作 {operation.operation_id} 缺少服务端生成的 precondition_hash"
            )
        before = operation_precondition_state(self.client, workspace, operation)
        actual = canonical_graph_state_hash(before)
        if actual != operation.precondition_hash:
            raise GraphPatchValidationError(f"操作 {operation.operation_id} 的图谱前置状态已变化")

    def _dispatch(
        self, workspace: str, operation: GraphOperationDraft, change_set_id: str
    ) -> dict[str, object]:
        key = f"{change_set_id}:{operation.operation_id}"
        if operation.operation_type == "CREATE_ENTITY":
            return self.client.create_entity(
                workspace,
                entity_name=_required(operation.entity_name, "entity_name"),
                description=operation.after_description or "",
                entity_type=operation.entity_type or "UNKNOWN",
                idempotency_key=key,
            )
        if operation.operation_type == "UPDATE_ENTITY":
            current = (
                self.client.get_entity(workspace, _required(operation.entity_name, "entity_name"))
                or {}
            )
            return self.client.update_entity(
                workspace,
                entity_name=_required(operation.entity_name, "entity_name"),
                description=operation.after_description
                if operation.after_description is not None
                else str(current.get("description", "")),
                entity_type=operation.entity_type or str(current.get("entity_type", "UNKNOWN")),
                idempotency_key=key,
            )
        if operation.operation_type == "DELETE_ENTITY":
            return self.client.delete_entity(
                workspace,
                entity_name=_required(operation.entity_name, "entity_name"),
                cascade=operation.cascade,
                idempotency_key=key,
            )
        if operation.operation_type == "CREATE_RELATION":
            return self.client.create_relation(
                workspace,
                source_entity=_required(operation.source_entity, "source_entity"),
                target_entity=_required(operation.target_entity, "target_entity"),
                description=operation.relation_description or "",
                keywords=operation.relation_keywords or "",
                idempotency_key=key,
            )
        if operation.operation_type == "UPDATE_RELATION":
            current = self.client.get_relation(
                workspace,
                _required(operation.source_entity, "source_entity"),
                _required(operation.target_entity, "target_entity"),
            )
            return self.client.update_relation(
                workspace,
                source_entity=_required(operation.source_entity, "source_entity"),
                target_entity=_required(operation.target_entity, "target_entity"),
                description=operation.relation_description
                if operation.relation_description is not None
                else str(current.get("description", "")),
                keywords=operation.relation_keywords
                if operation.relation_keywords is not None
                else str(current.get("keywords", "")),
                idempotency_key=key,
            )
        if operation.operation_type == "DELETE_RELATION":
            return self.client.delete_relation(
                workspace,
                source_entity=_required(operation.source_entity, "source_entity"),
                target_entity=_required(operation.target_entity, "target_entity"),
                idempotency_key=key,
            )
        if operation.operation_type == "MERGE_ENTITIES":
            target = _required(operation.target_entity, "target_entity")
            if not operation.source_entities:
                raise GraphPatchValidationError("MERGE_ENTITIES 缺少 source_entities")
            return self.client.merge_entities(
                workspace,
                source_entities=operation.source_entities,
                target_entity=target,
                idempotency_key=key,
            )
        raise GraphPatchValidationError(f"不支持的图谱操作：{operation.operation_type}")


def canonical_change_set_hash(
    *,
    profile_patch: object,
    graph_operations: object,
    regression_queries: object,
    base_graph_version_id: str,
    revision: int,
) -> str:
    payload = {
        "base_graph_version_id": base_graph_version_id,
        "revision": revision,
        "profile_patch": profile_patch,
        "graph_operations": graph_operations,
        "regression_queries": regression_queries,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_profile_patch_hash(
    *,
    profile_patch: list[dict[str, object]],
    base_graph_version_id: str,
    revision: int,
) -> str:
    """Profile 批准只绑定 Profile 内容，不受后续 Graph Patch 影响。"""

    payload = {
        "profile_patch": profile_patch,
        "base_graph_version_id": base_graph_version_id,
        "revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_graph_state_hash(value: object) -> str:
    """为服务端实际读取的图状态生成稳定摘要，模型不参与计算。"""

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def operation_precondition_state(
    client: LightRAGSidecarClient,
    workspace: str,
    operation: GraphOperationDraft,
) -> object:
    """读取操作将要覆盖的最小图状态，用于批准后的乐观并发检查。"""

    if operation.operation_type in {"CREATE_ENTITY", "UPDATE_ENTITY", "DELETE_ENTITY"}:
        entity_name = _required(operation.entity_name, "entity_name")
        return client.get_entity(workspace, entity_name) or {}
    if operation.operation_type in {
        "CREATE_RELATION",
        "UPDATE_RELATION",
        "DELETE_RELATION",
    }:
        source = _required(operation.source_entity, "source_entity")
        target = _required(operation.target_entity, "target_entity")
        return client.get_relation(workspace, source, target)
    if operation.operation_type == "MERGE_ENTITIES":
        target = _required(operation.target_entity, "target_entity")
        names = list(dict.fromkeys([*operation.source_entities, target]))
        return {name: client.get_entity(workspace, name) or {} for name in names}
    raise GraphPatchValidationError(f"不支持的图谱操作：{operation.operation_type}")


def _operation_order(operation: GraphOperationDraft) -> tuple[int, str]:
    order = {
        "CREATE_ENTITY": 0,
        "UPDATE_ENTITY": 1,
        "MERGE_ENTITIES": 2,
        "CREATE_RELATION": 3,
        "UPDATE_RELATION": 4,
        "DELETE_RELATION": 5,
        "DELETE_ENTITY": 6,
    }
    return order[operation.operation_type], operation.operation_id


def _required(value: str | None, field: str) -> str:
    if value is None or not value.strip():
        raise GraphPatchValidationError(f"图谱操作缺少 {field}")
    return value.strip()
