from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from moonlightbox.db import Database
from moonlightbox.jobs.service import JobService
from moonlightbox.worker import recover_interrupted_jobs
from sqlalchemy import MetaData, create_engine, func, inspect, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


def test_upgrade_creates_projects_table(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")

    tables = inspect(create_engine(database_url)).get_table_names()
    assert "projects" in tables
    assert "jobs" in tables
    assert {
        "import_sources",
        "participants",
        "messages",
        "event_nodes",
        "analysis_revisions",
        "analysis_runs",
        "event_candidates",
        "model_versions",
        "branches",
        "branch_messages",
        "world_graph_versions",
        "conversation_bundles",
        "conversation_bundle_messages",
        "person_world_profiles",
        "runtime_initializations",
        "node_investigations",
    }.issubset(tables)


def test_job_lease_and_dedupe_migration(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'job-lease-migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    inspector = inspect(create_engine(database_url))

    columns = {column["name"] for column in inspector.get_columns("jobs")}
    assert {"dedupe_key", "worker_token", "lease_expires_at"}.issubset(columns)
    assert any(
        index["unique"] and tuple(index["column_names"]) == ("dedupe_key",)
        for index in inspector.get_indexes("jobs")
    )
    checks = {
        constraint["name"]: constraint["sqltext"]
        for constraint in inspector.get_check_constraints("jobs")
    }
    assert "ck_jobs_lease_fields" in checks


def test_actor_typing_timeout_migration(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'actor-typing-migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    # 0041 已有意删除 ConversationActor 表；这里验收的是 0023 当时的历史契约，
    # 不能把升级到当前 Runtime v1 后仍保留旧表当作通过条件。
    command.upgrade(config, "0023_add_assistant_typing_timeout")

    columns = {
        column["name"]
        for column in inspect(create_engine(database_url)).get_columns(
            "conversation_actor_states"
        )
    }
    assert "assistant_typing_until" in columns


def test_state_field_provenance_migration(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'state-provenance-migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    # branch_state_versions 属于在 0041 退役的旧运行时，验证其历史迁移而非 head。
    command.upgrade(config, "0027_state_field_provenance")

    columns = {
        column["name"]
        for column in inspect(create_engine(database_url)).get_columns(
            "branch_state_versions"
        )
    }
    assert {"field_evidence", "field_confidence"}.issubset(columns)


def test_subject_cognitive_agent_migration_creates_tables_columns_and_indexes(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'subject-cognitive-agent.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    # Subject Cognitive Agent 被 0041 整体退役。本测试只覆盖 0024 的创建合同。
    command.upgrade(config, "0024_subject_cognitive_agent")

    inspector = inspect(create_engine(database_url))
    expected_columns = {
        "perception_events": {
            "project_id",
            "branch_id",
            "event_type",
            "occurred_at",
            "source",
            "confidence",
            "idempotency_key",
            "visible_through",
            "evidence",
            "model_protocol_version",
        },
        "cognitive_cycles": {
            "trigger_event_id",
            "input_cutoff_at",
            "starting_state_version_id",
            "memory_ids",
            "goal_ids",
            "private_note_id",
            "structured_changes",
            "final_decision",
            "status",
            "model_version_id",
            "model_protocol_version",
        },
        "private_cognition_notes": {
            "trigger_event_id",
            "content",
            "subjective_feelings",
            "attention_target",
            "desired_actions",
            "memory_ids",
            "confidence",
        },
        "mental_state_versions": {
            "version",
            "state",
            "previous_version_id",
            "source_cycle_id",
            "evidence",
            "is_current",
        },
        "agent_goals": {
            "goal_type",
            "content",
            "source",
            "priority",
            "supporting_evidence",
            "opposing_evidence",
            "status",
            "review_at",
        },
        "agent_intentions": {
            "intention_type",
            "content",
            "goal_id",
            "trigger_event_id",
            "status",
            "expression_plan",
        },
        "agent_wakeups": {
            "wake_at",
            "reason",
            "goal_id",
            "event_id",
            "idempotency_key",
            "status",
        },
    }
    assert expected_columns.keys() <= set(inspector.get_table_names())
    for table_name, columns in expected_columns.items():
        actual = {column["name"] for column in inspector.get_columns(table_name)}
        assert {"id", "project_id", "branch_id", "created_at"}.issubset(actual)
        assert columns.issubset(actual)

    perception_uniques = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("perception_events")
    }
    assert ("branch_id", "idempotency_key") in perception_uniques
    perception_indexes = {
        tuple(index["column_names"])
        for index in inspector.get_indexes("perception_events")
    }
    assert ("branch_id", "occurred_at") in perception_indexes
    state_indexes = {
        tuple(index["column_names"]): index["unique"]
        for index in inspector.get_indexes("mental_state_versions")
    }
    assert state_indexes[("branch_id",)] == 1
    state_uniques = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("mental_state_versions")
    }
    assert ("branch_id", "version") in state_uniques
    wakeup_uniques = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("agent_wakeups")
    }
    assert ("branch_id", "idempotency_key") in wakeup_uniques
    wakeup_indexes = {
        tuple(index["column_names"])
        for index in inspector.get_indexes("agent_wakeups")
    }
    assert ("status", "wake_at") in wakeup_indexes


def test_subject_cognitive_agent_migration_is_reversible(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'subject-cognitive-agent-downgrade.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "0024_subject_cognitive_agent")
    command.downgrade(config, "0023_add_assistant_typing_timeout")

    inspector = inspect(create_engine(database_url))
    assert not {
        "perception_events",
        "cognitive_cycles",
        "private_cognition_notes",
        "mental_state_versions",
        "agent_goals",
        "agent_intentions",
        "agent_wakeups",
    }.intersection(inspector.get_table_names())
    assert not any(
        tuple(index["column_names"]) == ("id", "project_id")
        for index in inspector.get_indexes("branches")
    )


def test_0010_migrates_legacy_running_jobs_and_remains_reversible(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'legacy-running-job.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0009_add_analysis_run_lease")
    old_engine = create_engine(database_url)
    old_metadata = MetaData()
    old_metadata.reflect(old_engine)
    jobs = old_metadata.tables["jobs"]
    now = datetime.now(UTC)
    statuses = ["running", "queued", "succeeded", "failed", "cancelled"]
    with old_engine.begin() as connection:
        connection.execute(
            jobs.insert(),
            [
                {
                    "id": f"job-{status}",
                    "kind": "legacy",
                    "payload": {"status": status},
                    "status": status,
                    "progress": 0.4,
                    "checkpoint": {"offset": 7},
                    "error_code": "legacy_error" if status == "failed" else None,
                    "error_message": "旧错误" if status == "failed" else None,
                    "created_at": now,
                    "updated_at": now,
                }
                for status in statuses
            ],
        )
    old_engine.dispose()

    # 此处验证 0010 的数据迁移与可逆性。不能从 head 回退，因为 0041 明确不可逆。
    command.upgrade(config, "0010_add_job_lease_and_dedupe")
    upgraded_engine = create_engine(database_url)
    upgraded_metadata = MetaData()
    upgraded_metadata.reflect(upgraded_engine)
    upgraded_jobs = upgraded_metadata.tables["jobs"]
    with upgraded_engine.connect() as connection:
        rows = {row["id"]: row for row in connection.execute(select(upgraded_jobs)).mappings()}
    running = rows["job-running"]
    assert running["status"] == "interrupted"
    assert running["error_code"] == "worker_interrupted"
    assert running["error_message"] == "Worker 升级后恢复遗留运行任务"
    assert running["progress"] == 0.4
    assert running["checkpoint"] == {"offset": 7}
    assert running["worker_token"] is None
    assert running["lease_expires_at"] is None
    assert running["dedupe_key"] is None
    for status in statuses[1:]:
        row = rows[f"job-{status}"]
        assert row["status"] == status
        assert row["payload"] == {"status": status}
        assert row["checkpoint"] == {"offset": 7}
    upgraded_engine.dispose()

    command.downgrade(config, "0009_add_analysis_run_lease")
    command.upgrade(config, "0010_add_job_lease_and_dedupe")

    database = Database(database_url)
    assert recover_interrupted_jobs(database) == 1
    with Session(database.engine) as session:
        recovered = JobService(session).get("job-running")
        assert recovered.status == "queued"
        assert recovered.progress == 0.4
        assert recovered.checkpoint == {"offset": 7}
        assert recovered.error_code is None
    database.close()

    constrained_engine = create_engine(database_url)
    constrained_metadata = MetaData()
    constrained_metadata.reflect(constrained_engine)
    with constrained_engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        with pytest.raises(IntegrityError):
            connection.execute(
                constrained_metadata.tables["jobs"].insert(),
                {
                    "id": "invalid-running",
                    "kind": "legacy",
                    "payload": {},
                    "status": "running",
                    "progress": 0.0,
                    "checkpoint": None,
                    "error_code": None,
                    "error_message": None,
                    "dedupe_key": None,
                    "worker_token": None,
                    "lease_expires_at": None,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        with pytest.raises(IntegrityError):
            connection.execute(
                constrained_metadata.tables["import_sources"].insert(),
                {
                    "id": "invalid-import",
                    "project_id": "missing-project",
                    "preview_id": "preview",
                    "source_path": "/tmp/missing",
                    "message_count": 0,
                    "confirmed_at": now,
                },
            )
    constrained_engine.dispose()


def test_analysis_revision_migration_links_exact_source_candidate(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'revision-candidate-migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    inspector = inspect(create_engine(database_url))

    columns = {column["name"] for column in inspector.get_columns("analysis_revisions")}
    assert "candidate_id" in columns
    assert any(
        tuple(foreign_key["constrained_columns"]) == ("candidate_id",)
        and tuple(foreign_key["referred_columns"]) == ("id",)
        and foreign_key["options"].get("ondelete") == "SET NULL"
        for foreign_key in inspector.get_foreign_keys("analysis_revisions")
    )
    assert any(
        tuple(index["column_names"]) == ("candidate_id",)
        for index in inspector.get_indexes("analysis_revisions")
    )


def test_analysis_run_migration_declares_indexes_constraints_and_cascades(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'analysis-run-migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    inspector = inspect(create_engine(database_url))

    run_columns = {column["name"] for column in inspector.get_columns("analysis_runs")}
    assert {
        "project_id",
        "import_id",
        "analysis_version",
        "prompt_version",
        "model",
        "config",
        "config_fingerprint",
        "window_ids",
        "window_manifest_fingerprint",
        "status",
        "total_windows",
        "completed_windows",
        "checkpoint",
        "error_category",
        "error_message",
        "created_at",
        "updated_at",
        "completed_at",
    }.issubset(run_columns)
    run_unique_columns = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("analysis_runs")
    }
    assert (
        "import_id",
        "analysis_version",
        "prompt_version",
        "model",
        "config_fingerprint",
    ) in run_unique_columns
    run_indexes = {tuple(index["column_names"]) for index in inspector.get_indexes("analysis_runs")}
    assert ("project_id",) in run_indexes
    assert ("import_id",) in run_indexes
    assert ("status",) in run_indexes

    candidate_unique_columns = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("event_candidates")
    }
    assert ("run_id", "window_id", "candidate_key") in candidate_unique_columns
    candidate_indexes = {
        tuple(index["column_names"]) for index in inspector.get_indexes("event_candidates")
    }
    assert ("run_id",) in candidate_indexes
    assert ("run_id", "status") in candidate_indexes

    run_foreign_keys = inspector.get_foreign_keys("analysis_runs")
    assert all(
        foreign_key["options"].get("ondelete") == "CASCADE" for foreign_key in run_foreign_keys
    )
    assert any(
        tuple(foreign_key["constrained_columns"]) == ("import_id", "project_id")
        and tuple(foreign_key["referred_columns"]) == ("id", "project_id")
        for foreign_key in run_foreign_keys
    )
    import_indexes = inspector.get_indexes("import_sources")
    assert any(
        index["unique"] and tuple(index["column_names"]) == ("id", "project_id")
        for index in import_indexes
    )
    [candidate_foreign_key] = inspector.get_foreign_keys("event_candidates")
    assert candidate_foreign_key["options"].get("ondelete") == "CASCADE"


def _upgrade_analysis_database(tmp_path: Path, name: str) -> Engine:
    database_url = f"sqlite:///{tmp_path / name}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    return create_engine(database_url)


def _seed_analysis_parent(connection: Connection, metadata: MetaData) -> None:
    now = datetime.now(UTC)
    connection.execute(
        metadata.tables["projects"].insert(),
        {"id": "project-1", "name": "迁移约束测试"},
    )
    connection.execute(
        metadata.tables["import_sources"].insert(),
        {
            "id": "import-1",
            "project_id": "project-1",
            "preview_id": "preview-1",
            "source_path": "/tmp/chat.json",
            "message_count": 10,
            "confirmed_at": now,
        },
    )


def _valid_run_values(run_id: str) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "id": run_id,
        "project_id": "project-1",
        "import_id": "import-1",
        "analysis_version": "important-event-v2",
        "prompt_version": "prompt-v1",
        "model": "cloud-model-v1",
        "config": {"threshold": 0.72},
        "config_fingerprint": f"fingerprint-{run_id}",
        "window_ids": ["window-0", "window-1"],
        "window_manifest_fingerprint": f"manifest-{run_id}",
        "status": "queued",
        "total_windows": 2,
        "completed_windows": 0,
        "checkpoint": 0,
        "error_category": None,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "unknown"},
        {"completed_windows": 1, "checkpoint": 0},
        {"status": "succeeded", "completed_at": datetime.now(UTC)},
        {"status": "running", "completed_at": datetime.now(UTC)},
        {
            "status": "failed",
            "completed_at": datetime.now(UTC),
            "error_category": None,
            "error_message": None,
        },
    ],
)
def test_alembic_analysis_run_checks_reject_invalid_rows(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    engine = _upgrade_analysis_database(tmp_path, "invalid-run.db")
    metadata = MetaData()
    metadata.reflect(engine)
    values = _valid_run_values("run-invalid")
    values.update(changes)

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        _seed_analysis_parent(connection, metadata)
        with pytest.raises(IntegrityError):
            connection.execute(metadata.tables["analysis_runs"].insert(), values)


def test_alembic_rejects_window_manifest_length_mismatch(tmp_path: Path) -> None:
    engine = _upgrade_analysis_database(tmp_path, "invalid-window-manifest.db")
    metadata = MetaData()
    metadata.reflect(engine)
    values = _valid_run_values("run-invalid-manifest")
    values["window_ids"] = ["window-0"]

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        _seed_analysis_parent(connection, metadata)
        with pytest.raises(IntegrityError):
            connection.execute(metadata.tables["analysis_runs"].insert(), values)


@pytest.mark.parametrize(
    ("status", "rejection_reason"),
    [("unknown", None), ("rejected", None), ("rejected", "  ")],
)
def test_alembic_candidate_checks_reject_invalid_rows(
    tmp_path: Path,
    status: str,
    rejection_reason: str | None,
) -> None:
    engine = _upgrade_analysis_database(tmp_path, "invalid-candidate.db")
    metadata = MetaData()
    metadata.reflect(engine)
    now = datetime.now(UTC)

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        _seed_analysis_parent(connection, metadata)
        connection.execute(
            metadata.tables["analysis_runs"].insert(),
            _valid_run_values("run-1"),
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                metadata.tables["event_candidates"].insert(),
                {
                    "id": "candidate-1",
                    "run_id": "run-1",
                    "window_id": "window-0",
                    "candidate_key": "candidate-0",
                    "raw_payload": {},
                    "review_payload": None,
                    "status": status,
                    "rejection_reason": rejection_reason,
                    "scores": {},
                    "created_at": now,
                    "updated_at": now,
                },
            )


def test_alembic_foreign_key_cascades_analysis_run_and_candidates(
    tmp_path: Path,
) -> None:
    engine = _upgrade_analysis_database(tmp_path, "cascade.db")
    metadata = MetaData()
    metadata.reflect(engine)
    runs = metadata.tables["analysis_runs"]
    candidates = metadata.tables["event_candidates"]
    now = datetime.now(UTC)

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        _seed_analysis_parent(connection, metadata)
        connection.execute(runs.insert(), _valid_run_values("run-1"))
        connection.execute(
            candidates.insert(),
            {
                "id": "candidate-1",
                "run_id": "run-1",
                "window_id": "window-0",
                "candidate_key": "candidate-0",
                "raw_payload": {},
                "review_payload": None,
                "status": "accepted",
                "rejection_reason": None,
                "scores": {},
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            metadata.tables["import_sources"]
            .delete()
            .where(metadata.tables["import_sources"].c.id == "import-1")
        )

        assert connection.scalar(select(func.count()).select_from(runs)) == 0
        assert connection.scalar(select(func.count()).select_from(candidates)) == 0


def test_alembic_rejects_analysis_run_with_mismatched_project_and_import(
    tmp_path: Path,
) -> None:
    engine = _upgrade_analysis_database(tmp_path, "mismatched-parent.db")
    metadata = MetaData()
    metadata.reflect(engine)

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        _seed_analysis_parent(connection, metadata)
        connection.execute(
            metadata.tables["projects"].insert(),
            {"id": "project-2", "name": "其他项目"},
        )
        values = _valid_run_values("run-mismatched")
        values["project_id"] = "project-2"

        with pytest.raises(IntegrityError):
            connection.execute(metadata.tables["analysis_runs"].insert(), values)


def test_analysis_run_downgrade_removes_import_parent_index(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'downgrade.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    # 这是 0007 的历史 schema 回滚测试。0041 明确不可逆，不能从当前 head 跨过它
    # 回退到 0006；只升级到本用例需要验证的 revision 才能准确覆盖该契约。
    command.upgrade(config, "0007_create_analysis_runs")
    upgraded = inspect(create_engine(database_url))
    assert any(
        index["unique"] and tuple(index["column_names"]) == ("id", "project_id")
        for index in upgraded.get_indexes("import_sources")
    )

    command.downgrade(config, "0006_create_branches")

    downgraded = inspect(create_engine(database_url))
    assert "analysis_runs" not in downgraded.get_table_names()
    assert not any(
        tuple(index["column_names"]) == ("id", "project_id")
        for index in downgraded.get_indexes("import_sources")
    )
