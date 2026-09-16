"""节点任务隔离与冻结契约冒烟，不调用真实模型。"""

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from moonlightbox.world.person_world.node_jobs import (
    NODE_PROFILE_JOB_KIND,
    create_node_compilation_handler,
    enqueue_node_compilation,
)


def test_enqueue_binds_approved_scope_and_does_not_clone_graph():
    service = Mock()
    service.session.get.return_value = NS(id="g", project_id="p", status="ready")
    service.session.scalar.return_value = None
    with patch(
        "moonlightbox.world.person_world.node_jobs.NodeCompilationScope.from_approved"
    ) as build:
        build.return_value.envelope.return_value = {"preview_hash": "h", "included_count": 9}
        enqueue_node_compilation(
            service,
            project_id="p",
            graph_id="g",
            investigation_id="i",
            preview_hash="h",
            idempotency_key="request1",
        )
    args = service.enqueue_unique.call_args
    assert args.args[0] == NODE_PROFILE_JOB_KIND
    assert args.args[1]["node_scope"]["included_count"] == 9
    service.session.add.assert_not_called()


def test_idempotency_key_cannot_change_scope():
    service = Mock()
    service.session.get.return_value = NS(id="g", project_id="p", status="ready")
    service.session.scalar.return_value = NS(payload={"node_scope": "other"})
    with patch("moonlightbox.world.person_world.node_jobs.NodeCompilationScope.from_approved"):
        with pytest.raises(ValueError, match="幂等键"):
            enqueue_node_compilation(
                service,
                project_id="p",
                graph_id="g",
                investigation_id="i",
                preview_hash="h",
                idempotency_key="request1",
            )
    service.enqueue_unique.assert_not_called()


def test_worker_rechecks_scope_before_starting_model():
    service = Mock()
    service.session.get.return_value = NS(id="g", project_id="p", status="ready")
    job = NS(
        payload={
            "graph_version_id": "g",
            "project_id": "p",
            "investigation_id": "i",
            "preview_hash": "old",
            "node_scope": {},
        },
        worker_token="token",
    )
    with patch(
        "moonlightbox.world.person_world.node_jobs.NodeCompilationScope.from_approved",
        side_effect=ValueError("起点已更新"),
    ):
        with pytest.raises(Exception) as error:
            create_node_compilation_handler(NS())(service, job)
    assert error.value.code == "node_scope_stale"


def test_worker_uses_same_graph_and_persists_draft_receipt():
    from moonlightbox.config import Settings

    service = Mock()
    graph = NS(id="g", project_id="p", status="ready")
    service.session.get.return_value = graph
    service.session.scalar.return_value = NS(id="draft")
    job = NS(
        id="job",
        worker_token="token",
        payload={
            "graph_version_id": "g",
            "project_id": "p",
            "investigation_id": "i",
            "preview_hash": "hash",
            "node_scope": {"preview_hash": "hash"},
        },
    )
    scope = Mock(boundary={"timezone": "Asia/Shanghai"})
    scope.envelope.return_value = job.payload["node_scope"]
    compiler, sidecar = Mock(), Mock()
    with (
        patch(
            "moonlightbox.world.person_world.node_jobs.NodeCompilationScope.from_approved",
            return_value=scope,
        ),
        patch(
            "moonlightbox.world.person_world.node_jobs._participant_names",
            return_value=("目标", "t", "用户"),
        ),
        patch("moonlightbox.world.person_world.node_jobs.PersonWorldCoordinatorV3") as coordinator,
    ):
        coordinator.return_value.run.return_value = NS(
            run_id="run", generation_summary={"failed_sections": []}
        )
        create_node_compilation_handler(
            Settings(), compiler_client=compiler, lightrag_client=sidecar
        )(service, job)
        assert coordinator.call_args.kwargs["graph"] is graph
        assert coordinator.call_args.kwargs["node_scope"] is scope
        coordinator.return_value.run.assert_called_once_with(
            mode="node_compile", resume_key="job:job"
        )
    assert service.checkpoint.call_args.args[1]["profile_draft_id"] == "draft"
    assert graph.status == "ready"
    sidecar.clone_workspace.assert_not_called()
    sidecar.close.assert_not_called()
