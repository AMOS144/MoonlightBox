"""从 Runtime 快照读取真人历史，不依赖旧 Branch Baseline。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant

from .branch_models import Branch
from .db_models import RuntimeSnapshotRow


class RuntimeSourceHistory:
    """读取当前完整导入图谱的原始对话证据。

    LightRAG 尚不支持按时间截断，因此与 ``OriginWorldSnapshot`` 一致：v1
    使用最新导入末端前的完整对话，而非旧 Baseline 的节点边界。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def authentic_example_rows(
        self, branch: Branch, *, message_limit: int = 500
    ) -> list[tuple[Message, str]]:
        if message_limit <= 0:
            return []
        snapshot = self._session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch.id)
        )
        if snapshot is None or not snapshot.source_message_ids:
            return []
        rows = list(
            self._session.execute(
                select(Message, Participant.role)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.project_id == branch.project_id,
                    Message.id.in_(snapshot.source_message_ids),
                    Participant.role.in_(("self", "target")),
                )
                .order_by(Message.timestamp.desc(), Message.source_id.desc(), Message.id.desc())
                .limit(message_limit)
            )
        )
        return list(reversed(rows))
