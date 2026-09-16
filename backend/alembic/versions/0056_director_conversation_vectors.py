"""可重建的分支语义向量，不改变原始聊天和世界图谱。"""

from alembic import op

revision = "0056_director_conversation_vectors"
down_revision = "0055_runtime_life_opportunities"
branch_labels = None
depends_on = None


def upgrade():
    from moonlightbox.runtime_v1.conversation_index import ConversationVector

    ConversationVector.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    from moonlightbox.runtime_v1.conversation_index import ConversationVector

    ConversationVector.__table__.drop(op.get_bind(), checkfirst=True)
