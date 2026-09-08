"""为事件节点增加 V3 双通道合同字段。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

revision: str = "0012_add_v3_event_contract"
down_revision: str | Sequence[str] | None = "0011_link_revision_source_candidate"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SHADOW_TABLE_NAME = "_moonlightbox_0012_event_node_backup"
_BEFORE_STATE_FALLBACK = "（V3 兼容：无前置关系状态）"
_AFTER_STATE_FALLBACK = "（V3 兼容：无后置关系状态）"
_SHADOW_COLUMNS = (
    "event_id",
    "lane",
    "event_status",
    "title",
    "summary",
    "started_at",
    "ended_at",
    "source_lanes",
    "before_state",
    "after_state",
)
_SHADOW_COLUMN_NAMES = set(_SHADOW_COLUMNS)


def _event_nodes() -> sa.Table:
    return sa.table(
        "event_nodes",
        sa.column("id", sa.String(36)),
        sa.column("lane", sa.String(32)),
        sa.column("event_status", sa.String(32)),
        sa.column("title", sa.Text()),
        sa.column("summary", sa.Text()),
        sa.column("started_at", sa.DateTime(timezone=True)),
        sa.column("ended_at", sa.DateTime(timezone=True)),
        sa.column("source_lanes", sa.JSON()),
        sa.column("topic", sa.Text()),
        sa.column("reason", sa.Text()),
        sa.column("before_state", sa.Text()),
        sa.column("after_state", sa.Text()),
    )


def _shadow_table() -> sa.Table:
    return sa.table(
        _SHADOW_TABLE_NAME,
        sa.column("event_id", sa.String(36)),
        sa.column("lane", sa.String(32)),
        sa.column("event_status", sa.String(32)),
        sa.column("title", sa.Text()),
        sa.column("summary", sa.Text()),
        sa.column("started_at", sa.DateTime(timezone=True)),
        sa.column("ended_at", sa.DateTime(timezone=True)),
        sa.column("source_lanes", sa.JSON()),
        sa.column("before_state", sa.Text()),
        sa.column("after_state", sa.Text()),
    )


def _shadow_exists(connection: Connection) -> bool:
    return _SHADOW_TABLE_NAME in sa.inspect(connection).get_table_names()


def _invalid_shadow(reason: str) -> RuntimeError:
    return RuntimeError(f"0012 影子表结构无效：{reason}")


def _validate_shadow_table(connection: Connection) -> None:
    inspector = sa.inspect(connection)
    columns = {column["name"]: column for column in inspector.get_columns(_SHADOW_TABLE_NAME)}
    if set(columns) != _SHADOW_COLUMN_NAMES:
        raise _invalid_shadow("列集合不匹配")
    expected_types: dict[str, type[sa.types.TypeEngine[object]]] = {
        "event_id": sa.String,
        "lane": sa.String,
        "event_status": sa.String,
        "title": sa.Text,
        "summary": sa.Text,
        "started_at": sa.DateTime,
        "ended_at": sa.DateTime,
        "source_lanes": sa.JSON,
        "before_state": sa.Text,
        "after_state": sa.Text,
    }
    if any(
        not isinstance(columns[name]["type"], expected_type)
        for name, expected_type in expected_types.items()
    ):
        raise _invalid_shadow("字段类型不匹配")
    required_columns = {
        "event_id",
        "lane",
        "event_status",
        "title",
        "summary",
        "source_lanes",
    }
    if any(columns[name]["nullable"] for name in required_columns):
        raise _invalid_shadow("必填字段允许为空")
    primary_key = inspector.get_pk_constraint(_SHADOW_TABLE_NAME)
    if primary_key["constrained_columns"] != ["event_id"]:
        raise _invalid_shadow("event_id 必须是唯一主键")

    shadow = _shadow_table()
    events = _event_nodes()
    shadow_ids = list(connection.scalars(sa.select(shadow.c.event_id)))
    event_ids = list(connection.scalars(sa.select(events.c.id)))
    if len(shadow_ids) != len(set(shadow_ids)):
        raise _invalid_shadow("event_id 存在重复")
    if set(shadow_ids) != set(event_ids):
        raise _invalid_shadow("event_id 与事件节点不一致")


def _create_shadow_table() -> None:
    op.create_table(
        _SHADOW_TABLE_NAME,
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("lane", sa.String(32), nullable=False),
        sa.Column("event_status", sa.String(32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_lanes", sa.JSON(), nullable=False),
        sa.Column("before_state", sa.Text(), nullable=True),
        sa.Column("after_state", sa.Text(), nullable=True),
    )


def _add_v3_columns() -> None:
    with op.batch_alter_table("event_nodes") as batch:
        batch.add_column(sa.Column("lane", sa.String(32), nullable=True))
        batch.add_column(sa.Column("event_status", sa.String(32), nullable=True))
        batch.add_column(sa.Column("title", sa.Text(), nullable=True))
        batch.add_column(sa.Column("summary", sa.Text(), nullable=True))
        batch.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("source_lanes", sa.JSON(), nullable=True))


def _finalize_v3_nullability() -> None:
    with op.batch_alter_table("event_nodes") as batch:
        batch.alter_column("lane", existing_type=sa.String(32), nullable=False)
        batch.alter_column(
            "event_status",
            existing_type=sa.String(32),
            nullable=False,
        )
        batch.alter_column("title", existing_type=sa.Text(), nullable=False)
        batch.alter_column("summary", existing_type=sa.Text(), nullable=False)
        batch.alter_column(
            "source_lanes",
            existing_type=sa.JSON(),
            nullable=False,
        )
        batch.alter_column(
            "before_state",
            existing_type=sa.Text(),
            nullable=True,
        )
        batch.alter_column(
            "after_state",
            existing_type=sa.Text(),
            nullable=True,
        )


def _fill_relationship_states(events: sa.Table) -> None:
    op.execute(
        events.update()
        .where(
            events.c.lane == "relationship",
            sa.or_(
                events.c.before_state.is_(None),
                sa.func.trim(events.c.before_state) == "",
            ),
        )
        .values(before_state=_BEFORE_STATE_FALLBACK)
    )
    op.execute(
        events.update()
        .where(
            events.c.lane == "relationship",
            sa.or_(
                events.c.after_state.is_(None),
                sa.func.trim(events.c.after_state) == "",
            ),
        )
        .values(after_state=_AFTER_STATE_FALLBACK)
    )


def upgrade() -> None:
    connection = op.get_bind()
    has_shadow = _shadow_exists(connection)
    if has_shadow:
        _validate_shadow_table(connection)

    _add_v3_columns()
    events = _event_nodes()
    if has_shadow:
        shadow = _shadow_table()
        rows = list(connection.execute(sa.select(shadow)).mappings())
        for row in rows:
            connection.execute(
                events.update()
                .where(events.c.id == row["event_id"])
                .values(
                    lane=row["lane"],
                    event_status=row["event_status"],
                    title=row["title"],
                    summary=row["summary"],
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    source_lanes=row["source_lanes"],
                )
            )
    else:
        op.execute(
            events.update().values(
                lane="relationship",
                event_status="occurred",
                title=events.c.topic,
                summary=events.c.reason,
                source_lanes=["relationship"],
            )
        )

    _finalize_v3_nullability()
    if has_shadow:
        shadow = _shadow_table()
        rows = list(connection.execute(sa.select(shadow)).mappings())
        for row in rows:
            connection.execute(
                events.update()
                .where(events.c.id == row["event_id"])
                .values(
                    before_state=row["before_state"],
                    after_state=row["after_state"],
                )
            )
    _fill_relationship_states(events)
    if has_shadow:
        op.drop_table(_SHADOW_TABLE_NAME)


def downgrade() -> None:
    connection = op.get_bind()
    if _shadow_exists(connection):
        raise RuntimeError("0012 影子表已存在，拒绝覆盖兼容数据")

    _create_shadow_table()
    events = _event_nodes()
    shadow = _shadow_table()
    connection.execute(
        shadow.insert().from_select(
            list(_SHADOW_COLUMNS),
            sa.select(
                events.c.id,
                events.c.lane,
                events.c.event_status,
                events.c.title,
                events.c.summary,
                events.c.started_at,
                events.c.ended_at,
                events.c.source_lanes,
                events.c.before_state,
                events.c.after_state,
            ),
        )
    )
    _validate_shadow_table(connection)

    op.execute(
        events.update()
        .where(events.c.before_state.is_(None))
        .values(before_state=_BEFORE_STATE_FALLBACK)
    )
    op.execute(
        events.update()
        .where(events.c.after_state.is_(None))
        .values(after_state=_AFTER_STATE_FALLBACK)
    )

    with op.batch_alter_table("event_nodes") as batch:
        batch.alter_column(
            "before_state",
            existing_type=sa.Text(),
            nullable=False,
        )
        batch.alter_column(
            "after_state",
            existing_type=sa.Text(),
            nullable=False,
        )
        batch.drop_column("source_lanes")
        batch.drop_column("ended_at")
        batch.drop_column("started_at")
        batch.drop_column("summary")
        batch.drop_column("title")
        batch.drop_column("event_status")
        batch.drop_column("lane")
