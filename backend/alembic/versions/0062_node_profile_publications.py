"""共享图谱上的独立节点画像；旧画像与发布记录保持全量作用域。"""

import sqlalchemy as sa
from alembic import op

revision = "0062_node_profile_publications"
down_revision = "0061_branch_origin_boundary"
branch_labels = None
depends_on = None


def upgrade():
    # 早期 SQLite 唯一约束没有名字，批量重建时显式赋名后删除。
    convention = {"uq": "uq_%(table_name)s_%(column_0_name)s"}
    inspector = sa.inspect(op.get_bind())
    constraints = inspector.get_unique_constraints("person_world_profiles")
    # 早期迁移使用当前 ORM metadata 建表：全新安装可能已经包含新增字段与索引。
    profile_columns = {c["name"] for c in inspector.get_columns("person_world_profiles")}
    profile_indexes = {i["name"] for i in inspector.get_indexes("person_world_profiles")}
    publication_columns = {c["name"] for c in inspector.get_columns("world_publications")}
    publication_indexes = {i["name"] for i in inspector.get_indexes("world_publications")}
    with op.batch_alter_table("person_world_profiles", naming_convention=convention) as batch:
        for constraint in constraints:
            if constraint["column_names"] == ["graph_version_id"]:
                batch.drop_constraint(
                    constraint["name"] or "uq_person_world_profiles_graph_version_id",
                    type_="unique",
                )
        if "node_boundary_hash" not in profile_columns:
            batch.add_column(sa.Column("node_boundary_hash", sa.String(64), nullable=True))
        if "ix_person_world_profiles_node_boundary_hash" not in profile_indexes:
            batch.create_index(
                "ix_person_world_profiles_node_boundary_hash", ["node_boundary_hash"]
            )
        if "ux_profile_global_graph" not in profile_indexes:
            batch.create_index(
                "ux_profile_global_graph",
                ["graph_version_id"],
                unique=True,
                sqlite_where=sa.text("node_boundary_hash IS NULL"),
                postgresql_where=sa.text("node_boundary_hash IS NULL"),
            )
    with op.batch_alter_table("world_publications") as batch:
        if "node_boundary_hash" not in publication_columns:
            batch.add_column(sa.Column("node_boundary_hash", sa.String(64), nullable=True))
        if "ix_world_publications_node_boundary_hash" not in publication_indexes:
            batch.create_index("ix_world_publications_node_boundary_hash", ["node_boundary_hash"])
        if "ux_node_publication_active" not in publication_indexes:
            batch.create_index(
                "ux_node_publication_active",
                ["project_id", "node_boundary_hash"],
                unique=True,
                sqlite_where=sa.text("status = 'active'"),
                postgresql_where=sa.text("status = 'active'"),
            )


def downgrade():
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM person_world_profiles WHERE node_boundary_hash IS NOT NULL"
            )
        )
        .scalar()
    ):
        raise RuntimeError("已有节点画像，不能无损恢复单图单画像约束")
    with op.batch_alter_table("person_world_profiles") as batch:
        batch.drop_index("ux_profile_global_graph")
        batch.drop_index("ix_person_world_profiles_node_boundary_hash")
        batch.drop_column("node_boundary_hash")
        batch.create_unique_constraint(
            "uq_person_world_profiles_graph_version_id", ["graph_version_id"]
        )
    with op.batch_alter_table("world_publications") as batch:
        batch.drop_index("ux_node_publication_active")
        batch.drop_index("ix_world_publications_node_boundary_hash")
        batch.drop_column("node_boundary_hash")
