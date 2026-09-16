"""区分取消请求和执行已退出，取消中的 Worker 仍持有租约。"""

from alembic import op

revision = "0057_job_cancellation_ack"
down_revision = "0056_director_conversation_vectors"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("ck_jobs_lease_fields", type_="check")
        batch.create_check_constraint(
            "ck_jobs_lease_fields",
            """
            (status IN ('running', 'cancelling') AND worker_token IS NOT NULL
             AND length(trim(worker_token)) > 0 AND lease_expires_at IS NOT NULL)
            OR (status NOT IN ('running', 'cancelling') AND worker_token IS NULL
                AND lease_expires_at IS NULL)
        """,
        )


def downgrade():
    # 不把仍在退出的 Worker 冒充为已结束，必须先等待它们完成。
    from sqlalchemy import text

    if op.get_bind().execute(text("SELECT count(*) FROM jobs WHERE status='cancelling'")).scalar():
        raise RuntimeError("仍有取消中的任务，不能回退租约约束")
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("ck_jobs_lease_fields", type_="check")
        batch.create_check_constraint(
            "ck_jobs_lease_fields",
            """
            (status='running' AND worker_token IS NOT NULL
             AND length(trim(worker_token)) > 0 AND lease_expires_at IS NOT NULL)
            OR (status!='running' AND worker_token IS NULL AND lease_expires_at IS NULL)
        """,
        )
