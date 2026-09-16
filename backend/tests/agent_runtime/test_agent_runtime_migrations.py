"""统一 Agent Harness 移除旧调用账本后的迁移验收。"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Column, MetaData, String, Table, create_engine, inspect

_LEGACY_LEDGER_TABLES = {
    "agent_runs",
    "agent_steps",
    "agent_tool_invocations",
    "agent_context_snapshots",
    "agent_trace_events",
}


def test_unified_agent_runtime_migrations_remove_ledger_and_keep_input_revisions(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'agent-runtime-migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")

    inspector = inspect(create_engine(database_url))
    tables = set(inspector.get_table_names())
    assert not _LEGACY_LEDGER_TABLES.intersection(tables)
    assert "runtime_input_revision" in {
        column["name"] for column in inspector.get_columns("branches")
    }
    assert "input_revision" in {
        column["name"] for column in inspector.get_columns("person_world_revision_sessions")
    }


def test_0053_drops_legacy_agent_ledger_from_existing_installation(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'legacy-agent-ledger.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0052_add_revision_input_revision")

    engine = create_engine(database_url)
    legacy_metadata = MetaData()
    for table_name in sorted(_LEGACY_LEDGER_TABLES):
        Table(table_name, legacy_metadata, Column("id", String(36), primary_key=True))
    legacy_metadata.create_all(engine)

    command.upgrade(config, "head")
    tables = set(inspect(engine).get_table_names())
    assert not _LEGACY_LEDGER_TABLES.intersection(tables)
