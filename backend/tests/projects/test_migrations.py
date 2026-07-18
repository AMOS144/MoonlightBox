from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


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
        "model_versions",
    }.issubset(tables)
