"""起点接口不泄漏训练依赖，也不绕过历史编译门禁。"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from moonlightbox.runtime_v1.profile_projection import _runtime_time_anchor
from moonlightbox.runtime_v1.router import RuntimeBranchCreate
from moonlightbox.runtime_v1.service import RuntimeService
from pydantic import ValidationError


def test_create_contract_rejects_legacy_fields():
    value = RuntimeBranchCreate(investigation_id="i", preview_hash="h", title="新起点")
    assert set(value.model_dump()) == {
        "investigation_id",
        "preview_hash",
        "title",
        "publication_id",
    }
    with pytest.raises(ValidationError):
        RuntimeBranchCreate(**value.model_dump(), model_version_id="old")


def test_new_clock_uses_only_approved_boundary():
    session = Mock()
    branch = SimpleNamespace(
        origin_boundary={
            "cutoff_at": "2026-05-01T10:30:00+08:00",
            "timezone": "Asia/Shanghai",
        }
    )
    origin, zone = _runtime_time_anchor(session, branch)
    assert origin.astimezone(UTC) == datetime(2026, 5, 1, 2, 30, tzinfo=UTC)
    assert zone == "Asia/Shanghai"
    session.get.assert_not_called()
    session.scalar.assert_not_called()


def test_existing_clock_is_preserved():
    session = Mock()
    clock = SimpleNamespace(virtual_anchor=datetime(2026, 5, 1), timezone="UTC+08:00")
    session.get.return_value = clock
    assert _runtime_time_anchor(session, SimpleNamespace(id="b", origin_boundary=None)) == (
        clock.virtual_anchor,
        clock.timezone,
    )
    session.scalar.assert_not_called()


def test_creation_cannot_enqueue_latest_background():
    session = Mock()
    service = object.__new__(RuntimeService)
    service.session = session
    with patch(
        "moonlightbox.world.person_world.publication.active_publication", return_value=None
    ) as resolve:
        with pytest.raises(ValueError, match="尚未审核发布"):
            service.create_branch(
                project_id="p", investigation_id="i", preview_hash="h", title="分支"
            )
        resolve.assert_called_once_with(session, "p", "h")
    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_migration_preserves_legacy_references():
    import importlib.util
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).parents[2] / "alembic/versions/0061_branch_origin_boundary.py"
    spec = importlib.util.spec_from_file_location("boundary_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sa.create_engine("sqlite://").begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE branches (id VARCHAR(36) PRIMARY KEY, "
            "origin_event_id VARCHAR(36) NOT NULL, model_version_id VARCHAR(36) NOT NULL)"
        )
        connection.exec_driver_sql("INSERT INTO branches VALUES ('b', 'event', 'model')")
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()
        assert connection.exec_driver_sql("SELECT * FROM branches").one() == (
            "b",
            "event",
            "model",
            None,
        )
        connection.exec_driver_sql("INSERT INTO branches (id) VALUES ('new')")
