import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import BranchMemoryEpisode
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.jobs.models import Job

CONTINUAL_MEMORY_JOB_KIND = "branch_continual_memory"


class EpisodeService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record_turn(
        self,
        branch: Branch,
        user: BranchMessage | list[BranchMessage],
        assistants: list[BranchMessage],
    ) -> tuple[BranchMemoryEpisode, Job]:
        users = [user] if isinstance(user, BranchMessage) else list(user)
        if not users:
            raise ValueError("成功轮次必须至少包含一条用户贡献")
        if not assistants:
            raise ValueError("成功轮次必须至少包含一个数字人气泡")
        if any(item.role != "user" for item in users) or any(
            item.role != "assistant" for item in assistants
        ):
            raise ValueError("episode 的消息角色不合法")
        users.sort(key=lambda item: item.sequence)
        user = users[-1]
        assistant_turn_ids = {item.turn_id for item in assistants}
        if len(assistant_turn_ids) != 1:
            raise ValueError("同一 episode 的数字人气泡必须属于同一轮次")
        assistant_turn_id = next(iter(assistant_turn_ids))
        existing = self._session.scalar(
            select(BranchMemoryEpisode).where(
                BranchMemoryEpisode.branch_id == branch.id,
                BranchMemoryEpisode.user_turn_id == user.turn_id,
                BranchMemoryEpisode.assistant_turn_id == assistant_turn_id,
            )
        )
        if existing is None:
            user_messages: list[dict[str, object]] = [
                {
                    "message_id": item.id,
                    "turn_id": item.turn_id,
                    "content": item.content,
                    "sequence": item.sequence,
                    "created_at": item.created_at.isoformat()
                    if item.created_at is not None
                    else None,
                }
                for item in users
            ]
            user_content = "\n".join(item.content for item in users)
            bubbles: list[dict[str, object]] = [
                {
                    "type": item.type,
                    "content": item.content,
                    "asset_id": item.media_asset_id,
                    "bubble_index": item.bubble_index,
                    "delay_ms": item.delay_ms,
                }
                for item in sorted(assistants, key=lambda item: item.bubble_index)
            ]
            payload = {
                "branch_id": branch.id,
                "user_turn_id": user.turn_id,
                "user_messages": user_messages,
                "assistant_turn_id": assistant_turn_id,
                "user_content": user_content,
                "assistant_bubbles": bubbles,
                "model_version_id": branch.model_version_id,
            }
            existing = BranchMemoryEpisode(
                branch_id=branch.id,
                user_turn_id=user.turn_id,
                assistant_turn_id=assistant_turn_id,
                user_content=user_content,
                user_messages=user_messages,
                assistant_bubbles=bubbles,
                model_version_id=branch.model_version_id,
                episode_hash=hashlib.sha256(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                importance=_estimate_importance(user_content, bubbles),
                processing_status="pending",
                started_at=users[0].created_at or datetime.now(UTC),
                ended_at=assistants[-1].created_at or datetime.now(UTC),
            )
            self._session.add(existing)
            self._session.flush()
        dedupe_key = f"branch-memory:{existing.id}"
        job = self._session.scalar(select(Job).where(Job.dedupe_key == dedupe_key))
        if job is None:
            job = Job(
                kind=CONTINUAL_MEMORY_JOB_KIND,
                payload={
                    "project_id": branch.project_id,
                    "branch_id": branch.id,
                    "episode_id": existing.id,
                },
                dedupe_key=dedupe_key,
            )
            self._session.add(job)
            self._session.flush()
        return existing, job


def _estimate_importance(
    user_content: str,
    bubbles: list[dict[str, object]],
) -> float:
    text = " ".join([user_content, *(str(item.get("content", "")) for item in bubbles)])
    score = 1.0 + min(4.0, len(text) / 80)
    if any(
        keyword in text
        for keyword in (
            "分手",
            "喜欢",
            "爱",
            "讨厌",
            "生气",
            "对不起",
            "以后",
            "一起",
            "决定",
        )
    ):
        score += 2.0
    return min(10.0, score)
