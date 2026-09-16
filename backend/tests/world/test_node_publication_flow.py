"""节点发布与历史分支闭环冒烟：真实数据库事务，不调用收费模型。"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from moonlightbox.db import Base
from moonlightbox.jobs.models import Job
from moonlightbox.model_registry import register_models
from moonlightbox.projects.models import Project
from moonlightbox.runtime_v1.db_models import RuntimeClockRow, RuntimeSnapshotRow
from moonlightbox.runtime_v1.service import RuntimeService
from moonlightbox.world.models import PersonWorldProfile, PersonWorldProfileDraft, WorldGraphVersion
from moonlightbox.world.person_world.contracts.profile_v3 import PersonWorldProfileV3
from moonlightbox.world.person_world.coordinator_v3 import empty_legacy_projection
from moonlightbox.world.person_world.node_review import (
    approve_node_profile,
    node_review_payload,
    review_node_draft,
)
from moonlightbox.world.person_world.node_scope import NodeCompilationScope
from moonlightbox.world.person_world.publication import (
    WorldPublicationError,
    _publish_pair,
    active_publication,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


def test_independent_nodes_publish_and_freeze_branch(tmp_path):
    register_models()
    engine = create_engine(f"sqlite:///{tmp_path}/flow.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id="p", name="项目"))
        graph = WorldGraphVersion(
            id="g",
            project_id="p",
            trigger_import_id="i",
            workspace_key="w",
            status="ready",
            source_fingerprint="s",
            config_fingerprint="c",
            compiler_version="v3",
            source_import_ids=[],
        )
        value = PersonWorldProfileV3(subject_participant_id="target").model_dump(mode="json")
        global_profile = PersonWorldProfile(
            project_id="p",
            graph_version_id="g",
            subject_person_id="target",
            profile_schema_version="v3",
            profile_v3=value,
            compiler_version="v3",
            source_message_ids=[],
            retrieval_manifest=[],
            **empty_legacy_projection().model_dump(mode="json"),
        )
        session.add_all([graph, global_profile])
        session.flush()
        global_pub = _publish_pair(
            session,
            graph=graph,
            profile=global_profile,
            correction_head_hash="",
            published_by="local_user",
        )
        session.commit()
        publications = []
        for index in (1, 2):
            scope = NodeCompilationScope(
                "g",
                {
                    "preview_hash": str(index) * 64,
                    "investigation_id": "inv",
                    "included_count": index,
                    "cutoff_at": "2026-05-01T10:00:00+08:00",
                    "timezone": "Asia/Shanghai",
                    "last_included_ref": f"m{index}",
                    "first_excluded_ref": f"m{index + 1}",
                },
                "s",
                ("m1", "m2", "m3"),
            )
            draft = PersonWorldProfileDraft(
                project_id="p",
                graph_version_id="g",
                agent_run_id=f"run{index}",
                profile_schema_version="v3",
                profile_v3=value,
                status="awaiting_review",
                claim_ids=[],
                payload={},
                generation_summary={"node_scope": scope.envelope()},
            )
            session.add(draft)
            session.flush()
            # 范围恢复本身由 test_node_scope_inheritance 验证，这里固定素材聚焦发布事务。
            with patch.object(NodeCompilationScope, "restore", return_value=scope):
                profile = review_node_draft(session, project_id="p", draft_id=draft.id)
                assert (
                    review_node_draft(session, project_id="p", draft_id=draft.id).id == profile.id
                )
                receipt = node_review_payload(session, profile)
                with pytest.raises(WorldPublicationError, match="审核内容"):
                    approve_node_profile(
                        session, project_id="p", profile_id=profile.id, approval_hash="wrong"
                    )
                pub = approve_node_profile(
                    session,
                    project_id="p",
                    profile_id=profile.id,
                    approval_hash=receipt["approval_hash"],
                )
                session.commit()
                publications.append(pub.id)
                service = object.__new__(RuntimeService)
                service.session = session
                branch = service.create_branch(
                    project_id="p",
                    investigation_id="inv",
                    preview_hash=scope.boundary["preview_hash"],
                    publication_id=pub.id,
                    title="历史分支",
                )
                snapshot = session.scalar(
                    select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch.id)
                )
                assert snapshot.source_message_ids == list(scope.message_ids[:index])
                assert snapshot.snapshot_mode == "historical_cutoff"
                assert snapshot.profile_id == profile.id
                assert snapshot.profile["_runtime_binding"]["publication_id"] == pub.id
                assert branch.origin_event_id is None and branch.model_version_id is None
                assert branch.lifecycle_status == "preparing"
                clock = session.get(RuntimeClockRow, branch.id)
                assert clock.status == "paused" and clock.timezone == "Asia/Shanghai"
                assert session.scalar(
                    select(Job).where(Job.dedupe_key == f"runtime-v1-cycle:bootstrap:{branch.id}")
                )
            assert active_publication(session, "p").id == global_pub.id
            assert graph.status == "ready"
        assert active_publication(session, "p", "1" * 64).id == publications[0]
        assert active_publication(session, "p", "2" * 64).id == publications[1]
        # 同一 Revision 流程可纠正节点画像；这里只模拟已完成执行的候选，不调用图谱服务。
        from unittest.mock import Mock

        from moonlightbox.world.models import WorldGraphChangeSet
        from moonlightbox.world.person_world.publication import publish_change_set
        from moonlightbox.world.person_world.review.service import PersonWorldReviewService

        revision = PersonWorldReviewService(
            session, compiler=Mock(), lightrag=Mock()
        ).create_session(
            project_id="p",
            user_message="修改这个起点的理解",
            idempotency_key="node-revision",
            base_profile_id=profile.id,
            run_agent=False,
        )
        assert revision.scope["node_scope"] == scope.envelope()
        candidate = WorldGraphVersion(
            id="child",
            project_id="p",
            trigger_import_id="i",
            workspace_key="child",
            status="publish_ready",
            source_fingerprint="s",
            config_fingerprint="correction-c",
            compiler_version="v3",
            source_import_ids=[],
            parent_version_id="g",
        )
        revised = PersonWorldProfile(
            project_id="p",
            graph_version_id="child",
            subject_person_id="target",
            node_boundary_hash="2" * 64,
            profile_schema_version="v3",
            profile_v3=value,
            compiler_version="v3",
            source_message_ids=[],
            retrieval_manifest=[],
            generation_summary={"node_scope": {**scope.envelope(), "graph_version_id": "child"}},
            **empty_legacy_projection().model_dump(mode="json"),
        )
        session.add_all([candidate, revised])
        session.flush()
        revision.status = "publish_ready"
        change = WorldGraphChangeSet(
            project_id="p",
            revision_session_id=revision.id,
            base_graph_version_id="g",
            revision=1,
            status="publish_ready",
            canonical_payload_hash="approved",
            candidate_graph_version_id="child",
            execution_result={"profile_id": revised.id},
        )
        session.add(change)
        session.flush()
        with patch.object(NodeCompilationScope, "restore", return_value=scope):
            corrected = publish_change_set(
                session,
                project_id="p",
                revision_session_id=revision.id,
                change_set_id=change.id,
                expected_session_revision=revision.session_revision,
            )
        session.commit()
        assert active_publication(session, "p", "2" * 64).id == corrected.id
        assert active_publication(session, "p", "1" * 64).id == publications[0]
        assert active_publication(session, "p").id == global_pub.id
        assert node_review_payload(session, profile)["profile_id"] == revised.id
        session.refresh(snapshot)
        assert snapshot.profile_id == profile.id  # 已创建分支不随以后纠正漂移。
        assert graph.status == "ready"
    engine.dispose()


def test_node_migration_preserves_old_rows_and_scopes_uniqueness():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy.exc import IntegrityError

    path = Path(__file__).parents[2] / "alembic/versions/0062_node_profile_publications.py"
    spec = importlib.util.spec_from_file_location("node_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with create_engine("sqlite://").begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE person_world_profiles (id TEXT PRIMARY KEY, graph_version_id TEXT UNIQUE)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE world_publications (id TEXT PRIMARY KEY, project_id TEXT, status TEXT)"
        )
        connection.exec_driver_sql("INSERT INTO person_world_profiles VALUES ('old', 'g')")
        connection.exec_driver_sql(
            "INSERT INTO world_publications VALUES ('oldpub', 'p', 'active')"
        )
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()
        assert connection.exec_driver_sql("SELECT * FROM person_world_profiles").one() == (
            "old",
            "g",
            None,
        )
        connection.exec_driver_sql("INSERT INTO person_world_profiles VALUES ('node', 'g', 'n')")
        with pytest.raises(IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO person_world_profiles VALUES ('global2', 'g', NULL)"
            )
        connection.exec_driver_sql(
            "INSERT INTO world_publications VALUES ('nodepub', 'p', 'active', 'n')"
        )
        with pytest.raises(IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO world_publications VALUES ('nodepub2', 'p', 'active', 'n')"
            )
