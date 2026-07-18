from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.generation import BranchGenerator
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.branches.schemas import BranchCreate


class BranchNotFoundError(LookupError):
    pass


class BranchService:
    def __init__(self, session: Session, generator: BranchGenerator) -> None:
        self._session = session
        self._generator = generator

    def create(self, project_id: str, payload: BranchCreate) -> Branch:
        branch = Branch(project_id=project_id, **payload.model_dump())
        self._session.add(branch)
        self._session.commit()
        self._session.refresh(branch)
        return branch

    def list_branches(self, project_id: str) -> list[Branch]:
        return list(
            self._session.scalars(
                select(Branch)
                .where(Branch.project_id == project_id)
                .order_by(Branch.created_at.desc())
            )
        )

    def messages(self, project_id: str, branch_id: str) -> list[BranchMessage]:
        self._get(project_id, branch_id)
        return list(
            self._session.scalars(
                select(BranchMessage)
                .where(BranchMessage.branch_id == branch_id)
                .order_by(BranchMessage.sequence)
            )
        )

    def add_user_message(
        self,
        project_id: str,
        branch_id: str,
        content: str,
    ) -> BranchMessage:
        branch = self._get(project_id, branch_id)
        history = self.messages(project_id, branch_id)
        user = BranchMessage(
            branch_id=branch.id,
            sequence=len(history),
            role="user",
            content=content,
            generation_metadata={},
        )
        self._session.add(user)
        prompt_messages = [
            {"role": message.role, "content": message.content} for message in history
        ] + [{"role": "user", "content": content}]
        system_prompt = (
            f"分支起点：{branch.origin_time.isoformat()}。"
            "只能使用该时点及之前的信息。"
            f"状态：{json.dumps(branch.state_snapshot, ensure_ascii=False)}"
        )
        reply = self._generator.generate(
            branch.model_version_id,
            system_prompt,
            prompt_messages,
        )
        assistant = BranchMessage(
            branch_id=branch.id,
            sequence=len(history) + 1,
            role="assistant",
            content=reply,
            generation_metadata={"model_version_id": branch.model_version_id},
        )
        self._session.add(assistant)
        self._session.commit()
        self._session.refresh(assistant)
        return assistant

    def _get(self, project_id: str, branch_id: str) -> Branch:
        branch = self._session.get(Branch, branch_id)
        if branch is None or branch.project_id != project_id:
            raise BranchNotFoundError(branch_id)
        return branch
