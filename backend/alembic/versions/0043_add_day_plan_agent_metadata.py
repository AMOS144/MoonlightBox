"""为 DayPlanAgent 保存生成边界与可审计规划轨迹。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0043_add_day_plan_agent_metadata"
down_revision: str | None = "0042_add_runtime_cycle_traces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 0039 使用当时导入的 ORM 元数据建表。全新安装在已经包含 DayPlanAgent
    # 模型的代码上执行时，``runtime_day_plans`` 会提前拥有该列；升级旧库时则由
    # 本迁移补齐。两种路径都必须可用。
    bind = op.get_bind()
    plan_columns = {column["name"] for column in sa.inspect(bind).get_columns("runtime_day_plans")}
    if "generation_metadata" not in plan_columns:
        op.add_column(
            "runtime_day_plans",
            sa.Column(
                "generation_metadata",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )
    trace_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("runtime_cycle_traces")
    }
    if "planner_steps" not in trace_columns:
        op.add_column(
            "runtime_cycle_traces",
            sa.Column("planner_steps", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        )
    if "planner_proposal" not in trace_columns:
        op.add_column(
            "runtime_cycle_traces", sa.Column("planner_proposal", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    op.drop_column("runtime_cycle_traces", "planner_proposal")
    op.drop_column("runtime_cycle_traces", "planner_steps")
    op.drop_column("runtime_day_plans", "generation_metadata")
