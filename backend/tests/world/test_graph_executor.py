from moonlightbox.db import Base
from moonlightbox.imports.models import ImportSource  # noqa: F401
from moonlightbox.projects.models import Project  # noqa: F401
from moonlightbox.world.models import (
    WorldChangeApproval,
    WorldGraphChangeSet,
    WorldGraphVersion,
)
from moonlightbox.world.person_world.graph_executor import (
    GraphPatchValidationError,
    WorldGraphExecutor,
    canonical_change_set_hash,
    canonical_graph_state_hash,
    canonical_profile_patch_hash,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class FakeGraphClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_entity(self, workspace: str, entity_name: str) -> dict[str, object]:
        return {"description": "旧描述", "entity_type": "Person"}

    def update_entity(self, workspace: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((workspace, str(kwargs["entity_name"])))
        return {"description": kwargs["description"], "entity_type": kwargs["entity_type"]}


def _graph(graph_id: str, workspace: str, *, parent: str | None = None) -> WorldGraphVersion:
    return WorldGraphVersion(
        id=graph_id,
        project_id="project-1",
        trigger_import_id="import-1",
        workspace_key=workspace,
        status="candidate" if parent else "ready",
        source_fingerprint="s" * 64,
        config_fingerprint=("c" if parent else "b") * 64,
        source_import_ids=["import-1"],
        compiler_version="person-world-agent-v2",
        parent_version_id=parent,
        revision=2 if parent else 1,
        change_set_id="change-1" if parent else None,
    )


def test_executor_rejects_unapproved_patch_and_only_writes_candidate() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        base = _graph("base-1", "world_base")
        candidate = _graph("candidate-1", "world_candidate", parent=base.id)
        before = {"description": "旧描述", "entity_type": "Person"}
        operations = [
            {
                "operation_id": "operation-1",
                "operation_type": "UPDATE_ENTITY",
                "entity_name": "洪欣羽",
                "after_description": "目标人物，而不是用户",
                "entity_type": "Person",
                "reason": "纠正主体归属",
                "source_message_ids": ["message-1"],
                "precondition_hash": canonical_graph_state_hash(before),
                "cascade": False,
            }
        ]
        digest = canonical_change_set_hash(
            profile_patch=[],
            graph_operations=operations,
            regression_queries=[],
            base_graph_version_id=base.id,
            revision=1,
        )
        change_set = WorldGraphChangeSet(
            id="change-1",
            project_id="project-1",
            revision_session_id="revision-1",
            base_graph_version_id=base.id,
            revision=1,
            status="draft",
            profile_patch=[],
            graph_operations=operations,
            affected_entities=["洪欣羽"],
            affected_relations=[],
            regression_queries=[],
            canonical_payload_hash=digest,
            candidate_graph_version_id=candidate.id,
        )
        session.add_all([base, candidate, change_set])
        session.flush()
        client = FakeGraphClient()
        executor = WorldGraphExecutor(session, client)  # type: ignore[arg-type]

        try:
            executor.apply(change_set=change_set, candidate_graph=candidate)
        except GraphPatchValidationError:
            pass
        else:
            raise AssertionError("未批准 ChangeSet 不得执行")
        assert client.calls == []

        change_set.status = "approved"
        profile_hash = canonical_profile_patch_hash(
            profile_patch=change_set.profile_patch,
            base_graph_version_id=base.id,
            revision=change_set.revision,
        )
        session.add_all(
            [
                WorldChangeApproval(
                    change_set_id=change_set.id,
                    approval_stage=stage,
                    revision=change_set.revision,
                    payload_hash=profile_hash if stage == "profile" else digest,
                    approved_by="test",
                )
                for stage in ("profile", "graph")
            ]
        )
        session.flush()
        result = executor.apply(change_set=change_set, candidate_graph=candidate)

        assert result["operations"]
        assert client.calls == [("world_candidate", "洪欣羽")]
        assert base.status == "ready"
        assert candidate.status == "validating"


def test_executor_rejects_payload_changed_after_approval() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        base = _graph("base-2", "world_base_2")
        candidate = _graph("candidate-2", "world_candidate_2", parent=base.id)
        candidate.change_set_id = "change-2"
        operations = [
            {
                "operation_id": "operation-1",
                "operation_type": "UPDATE_ENTITY",
                "entity_name": "洪欣羽",
                "after_description": "原批准描述",
                "entity_type": "Person",
                "reason": "纠正",
                "source_message_ids": [],
                "precondition_hash": canonical_graph_state_hash(
                    {"description": "旧描述", "entity_type": "Person"}
                ),
                "cascade": False,
            }
        ]
        approved_hash = canonical_change_set_hash(
            profile_patch=[],
            graph_operations=operations,
            regression_queries=[],
            base_graph_version_id=base.id,
            revision=1,
        )
        operations[0]["after_description"] = "批准后被篡改"
        profile_hash = canonical_profile_patch_hash(
            profile_patch=[],
            base_graph_version_id=base.id,
            revision=1,
        )
        change_set = WorldGraphChangeSet(
            id="change-2",
            project_id="project-1",
            revision_session_id="revision-2",
            base_graph_version_id=base.id,
            revision=1,
            status="approved",
            profile_patch=[],
            graph_operations=operations,
            affected_entities=["洪欣羽"],
            affected_relations=[],
            regression_queries=[],
            canonical_payload_hash=approved_hash,
            candidate_graph_version_id=candidate.id,
        )
        session.add_all(
            [
                base,
                candidate,
                change_set,
                *[
                    WorldChangeApproval(
                        change_set_id=change_set.id,
                        approval_stage=stage,
                        revision=1,
                        payload_hash=profile_hash if stage == "profile" else approved_hash,
                        approved_by="test",
                    )
                    for stage in ("profile", "graph")
                ],
            ]
        )
        session.flush()
        executor = WorldGraphExecutor(session, FakeGraphClient())  # type: ignore[arg-type]

        try:
            executor.apply(change_set=change_set, candidate_graph=candidate)
        except GraphPatchValidationError as error:
            assert "重新批准" in str(error)
        else:
            raise AssertionError("批准后的 payload 变化必须拒绝")


def test_executor_requires_server_precondition_and_both_approvals() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        base = _graph("base-3", "world_base_3")
        candidate = _graph("candidate-3", "world_candidate_3", parent=base.id)
        candidate.change_set_id = "change-3"
        operations = [
            {
                "operation_id": "operation-1",
                "operation_type": "UPDATE_ENTITY",
                "entity_name": "洪欣羽",
                "after_description": "新描述",
                "reason": "纠正",
                "source_message_ids": [],
                "precondition_hash": None,
            }
        ]
        digest = canonical_change_set_hash(
            profile_patch=[],
            graph_operations=operations,
            regression_queries=[],
            base_graph_version_id=base.id,
            revision=1,
        )
        change_set = WorldGraphChangeSet(
            id="change-3",
            project_id="project-1",
            revision_session_id="revision-3",
            base_graph_version_id=base.id,
            revision=1,
            status="approved",
            profile_patch=[],
            graph_operations=operations,
            affected_entities=["洪欣羽"],
            affected_relations=[],
            regression_queries=[],
            canonical_payload_hash=digest,
            candidate_graph_version_id=candidate.id,
        )
        session.add_all(
            [
                base,
                candidate,
                change_set,
                WorldChangeApproval(
                    change_set_id=change_set.id,
                    approval_stage="graph",
                    revision=1,
                    payload_hash=digest,
                    approved_by="test",
                ),
            ]
        )
        session.flush()
        executor = WorldGraphExecutor(session, FakeGraphClient())  # type: ignore[arg-type]

        try:
            executor.apply(change_set=change_set, candidate_graph=candidate)
        except GraphPatchValidationError as error:
            assert "分别完成批准" in str(error)
        else:
            raise AssertionError("只有 Graph Approval 时不得执行")

        session.add(
            WorldChangeApproval(
                change_set_id=change_set.id,
                approval_stage="profile",
                revision=1,
                payload_hash=canonical_profile_patch_hash(
                    profile_patch=change_set.profile_patch,
                    base_graph_version_id=base.id,
                    revision=1,
                ),
                approved_by="test",
            )
        )
        session.flush()
        try:
            executor.apply(change_set=change_set, candidate_graph=candidate)
        except GraphPatchValidationError as error:
            assert "precondition_hash" in str(error)
        else:
            raise AssertionError("缺少服务端前置状态摘要时不得执行")
