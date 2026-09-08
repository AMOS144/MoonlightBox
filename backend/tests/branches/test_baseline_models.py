from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_0018_creates_branch_baseline_schema_and_is_reversible(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'baseline.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "0018_add_branch_baseline_history")
    inspector = inspect(create_engine(database_url))
    tables = set(inspector.get_table_names())
    assert {
        "branch_baseline_manifests",
        "branch_baseline_event_snapshots",
        "branch_baseline_states",
    }.issubset(tables)
    branch_columns = {
        column["name"] for column in inspector.get_columns("branches")
    }
    assert {
        "origin_import_id",
        "origin_boundary_message_id",
        "baseline_manifest_id",
        "baseline_job_id",
        "baseline_status",
        "baseline_error_code",
        "baseline_error_message",
        "baseline_ready_at",
    }.issubset(branch_columns)
    state_columns = {
        column["name"]
        for column in inspector.get_columns("branch_state_versions")
    }
    assert {"baseline_manifest_id", "baseline_state_id"}.issubset(state_columns)

    command.downgrade(config, "0017_add_branch_continual_memory")
    downgraded = inspect(create_engine(database_url))
    assert "branch_baseline_manifests" not in downgraded.get_table_names()
    assert "baseline_status" not in {
        column["name"] for column in downgraded.get_columns("branches")
    }

    command.upgrade(config, "0018_add_branch_baseline_history")
    assert "branch_baseline_manifests" in inspect(
        create_engine(database_url)
    ).get_table_names()
