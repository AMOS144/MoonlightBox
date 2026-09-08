from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from moonlightbox.events.schemas import EventNodeRead
from sqlalchemy import MetaData, create_engine, inspect, select

_SHADOW_TABLE = "_moonlightbox_0012_event_node_backup"
_HISTORICAL_BEFORE = "__moonlightbox_v3_state_v2__metadata:普通历史文本"
_HISTORICAL_AFTER = "__moonlightbox_v3_state_v2__payload_in_before"


def _config(database_url: str) -> Config:
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _legacy_event(event_id: str = "legacy-event") -> dict[str, object]:
    return {
        "id": event_id,
        "project_id": "project-1",
        "type": "travel",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "before_state": _HISTORICAL_BEFORE,
        "after_state": _HISTORICAL_AFTER,
        "emotion_labels": ["期待"],
        "topic": "历史旅行文本",
        "conflict_level": 0,
        "importance": 0.8,
        "reason": "旧版可自由使用类型和状态文本",
        "evidence_ids": ["m1", "m2"],
        "status": "active",
        "created_at": datetime(2026, 7, 1, tzinfo=UTC),
    }


def _seed_0011_database(database_url: str) -> None:
    config = _config(database_url)
    command.upgrade(config, "0011_link_revision_source_candidate")
    engine = create_engine(database_url)
    metadata = MetaData()
    metadata.reflect(engine)
    with engine.begin() as connection:
        connection.execute(
            metadata.tables["projects"].insert(),
            {"id": "project-1", "name": "V3 迁移测试"},
        )
        connection.execute(
            metadata.tables["event_nodes"].insert(),
            _legacy_event(),
        )
    engine.dispose()


def test_initial_0011_upgrade_uses_history_defaults_without_shadow(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'initial-upgrade.db'}"
    _seed_0011_database(database_url)
    config = _config(database_url)

    command.upgrade(config, "0012_add_v3_event_contract")

    engine = create_engine(database_url)
    metadata = MetaData()
    metadata.reflect(engine)
    with engine.connect() as connection:
        row = connection.execute(select(metadata.tables["event_nodes"])).mappings().one()
    assert row["lane"] == "relationship"
    assert row["event_status"] == "occurred"
    assert row["title"] == "历史旅行文本"
    assert row["summary"] == "旧版可自由使用类型和状态文本"
    assert row["source_lanes"] == ["relationship"]
    assert row["before_state"] == _HISTORICAL_BEFORE
    assert row["after_state"] == _HISTORICAL_AFTER
    assert _SHADOW_TABLE not in inspect(engine).get_table_names()
    engine.dispose()


def test_shadow_table_round_trip_restores_complete_v3_event(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'shadow-round-trip.db'}"
    _seed_0011_database(database_url)
    config = _config(database_url)
    command.upgrade(config, "0012_add_v3_event_contract")

    engine = create_engine(database_url)
    metadata = MetaData()
    metadata.reflect(engine)
    event_nodes = metadata.tables["event_nodes"]
    shared_event = _legacy_event("shared-event")
    shared_event.update(
        {
            "type": "travel",
            "before_state": None,
            "after_state": None,
            "topic": "兼容显示主题",
            "reason": "兼容显示摘要",
            "lane": "shared_experience",
            "event_status": "confirmed",
            "title": "一起去旅行",
            "summary": "共同完成一次长途旅行",
            "started_at": datetime(2026, 7, 2, tzinfo=UTC),
            "ended_at": datetime(2026, 7, 8, tzinfo=UTC),
            "source_lanes": ["relationship", "shared_experience"],
        }
    )
    with engine.begin() as connection:
        connection.execute(event_nodes.insert(), shared_event)
    engine.dispose()

    command.downgrade(config, "0011_link_revision_source_candidate")

    downgraded_engine = create_engine(database_url)
    downgraded_metadata = MetaData()
    downgraded_metadata.reflect(downgraded_engine)
    assert _SHADOW_TABLE in downgraded_metadata.tables
    shadow_primary_key = inspect(downgraded_engine).get_pk_constraint(_SHADOW_TABLE)
    assert shadow_primary_key["constrained_columns"] == ["event_id"]
    with downgraded_engine.connect() as connection:
        downgraded_shared = (
            connection.execute(
                select(downgraded_metadata.tables["event_nodes"]).where(
                    downgraded_metadata.tables["event_nodes"].c.id == "shared-event"
                )
            )
            .mappings()
            .one()
        )
        assert downgraded_shared["before_state"] == "（V3 兼容：无前置关系状态）"
        assert downgraded_shared["after_state"] == "（V3 兼容：无后置关系状态）"
    downgraded_engine.dispose()

    command.upgrade(config, "0012_add_v3_event_contract")

    restored_engine = create_engine(database_url)
    restored_metadata = MetaData()
    restored_metadata.reflect(restored_engine)
    assert _SHADOW_TABLE not in restored_metadata.tables
    with restored_engine.connect() as connection:
        restored = (
            connection.execute(
                select(restored_metadata.tables["event_nodes"]).where(
                    restored_metadata.tables["event_nodes"].c.id == "shared-event"
                )
            )
            .mappings()
            .one()
        )
        historical = (
            connection.execute(
                select(restored_metadata.tables["event_nodes"]).where(
                    restored_metadata.tables["event_nodes"].c.id == "legacy-event"
                )
            )
            .mappings()
            .one()
        )
    assert restored["lane"] == "shared_experience"
    assert restored["event_status"] == "confirmed"
    assert restored["title"] == "一起去旅行"
    assert restored["summary"] == "共同完成一次长途旅行"
    assert restored["source_lanes"] == ["relationship", "shared_experience"]
    assert restored["before_state"] is None
    assert restored["after_state"] is None
    assert EventNodeRead.model_validate(dict(restored)).lane == "shared_experience"
    assert historical["before_state"] == _HISTORICAL_BEFORE
    assert historical["after_state"] == _HISTORICAL_AFTER
    restored_engine.dispose()


def test_upgrade_rejects_malformed_shadow_table_before_schema_change(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'malformed-shadow.db'}"
    _seed_0011_database(database_url)
    config = _config(database_url)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.exec_driver_sql(f"CREATE TABLE {_SHADOW_TABLE} (event_id TEXT, lane TEXT)")
    engine.dispose()

    with pytest.raises(RuntimeError, match="0012 影子表结构无效"):
        command.upgrade(config, "0012_add_v3_event_contract")

    inspector = inspect(create_engine(database_url))
    columns = {column["name"] for column in inspector.get_columns("event_nodes")}
    assert "lane" not in columns
