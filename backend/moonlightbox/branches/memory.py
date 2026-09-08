import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent.models import AgentGoal, MentalStateVersion
from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.context import ContextMemory
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_models import (
    BranchMemoryItem,
    IdentityKernel,
)
from moonlightbox.branches.history import BranchHistoryService
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.memory_policy import retrieval_policy
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.events.models import EventNode
from moonlightbox.media.context import model_message_content
from moonlightbox.media.models import MediaSemanticAnnotation
from moonlightbox.training.models import ModelVersion


@dataclass(frozen=True)
class MemoryPacket:
    background: str
    prompt_messages: list[dict[str, str]]
    memories: tuple[ContextMemory, ...] = ()
    identity_kernel: dict[str, object] | None = None
    branch_state: dict[str, object] | None = None
    continuity_memories: tuple[ContextMemory, ...] = ()
    baseline_manifest: dict[str, object] | None = None
    baseline_prompt_messages: tuple[dict[str, str], ...] = ()
    retrieval_intent: str = "general"
    mental_state: dict[str, object] | None = None
    active_goals: tuple[str, ...] = ()


def build_memory_packet(
    session: Session,
    branch: Branch,
    current_content: str,
    history: list[BranchMessage],
    *,
    branch_turn_limit: int = 12,
    repository: ProjectMemoryRepository | None = None,
    continuity_repository: BranchContinuityRepository | None = None,
) -> MemoryPacket:
    prompt_messages = _group_valid_history(
        history,
        branch_turn_limit,
        _trusted_model_lineage(session, branch.model_version_id),
    )
    previous_assistant = next(
        (item["content"] for item in reversed(prompt_messages) if item["role"] == "assistant"),
        "",
    )
    active_belief_ids = branch.state_snapshot.get("active_belief_ids", [])
    active_beliefs = (
        list(
            session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.id.in_(active_belief_ids),
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                )
            )
        )
        if isinstance(active_belief_ids, list) and active_belief_ids
        else []
    )
    base_query = (
        f"上一句：{previous_assistant}\n当前：{current_content}"
        if previous_assistant
        else current_content
    )
    query_text = "\n".join(
        (
            base_query,
            "当前活跃信念：" + "；".join(item.content for item in active_beliefs),
            "当前关系状态："
            + json.dumps(
                branch.state_snapshot.get("relationship_state", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    )
    policy = retrieval_policy(current_content)
    retrieved = (
        repository.retrieve(session, branch, query_text)
        if repository is not None and policy.retrieve_project_history
        else ()
    )
    continuity_memories = (
        continuity_repository.retrieve(session, branch, query_text)
        if continuity_repository is not None and policy.retrieve_branch_continuity
        else ()
    )
    memories = (_branch_context_memory(session, branch), *retrieved)
    kernel = session.scalar(
        select(IdentityKernel).where(IdentityKernel.model_version_id == branch.model_version_id)
    )
    manifest = (
        session.get(BranchBaselineManifest, branch.baseline_manifest_id)
        if branch.baseline_manifest_id is not None
        else None
    )
    baseline_rows = BranchHistoryService(session).recent_rows(branch)
    baseline_asset_ids = {
        message.media_asset_id
        for message, _role in baseline_rows
        if message.media_asset_id is not None
    }
    annotations_by_asset = {
        annotation.asset_id: annotation
        for annotation in session.scalars(
            select(MediaSemanticAnnotation).where(
                MediaSemanticAnnotation.asset_id.in_(baseline_asset_ids)
            )
        )
    } if baseline_asset_ids else {}
    mental_state = session.scalar(
        select(MentalStateVersion).where(
            MentalStateVersion.branch_id == branch.id,
            MentalStateVersion.is_current.is_(True),
        )
    )
    active_goals = tuple(
        goal.content
        for goal in session.scalars(
            select(AgentGoal)
            .where(
                AgentGoal.branch_id == branch.id,
                AgentGoal.status == "active",
            )
            .order_by(AgentGoal.priority.desc(), AgentGoal.created_at.desc())
            .limit(3)
        )
    )
    baseline_prompt_messages = tuple(
        {
            "role": "user" if role == "self" else "assistant",
            "content": model_message_content(
                message,
                annotations_by_asset.get(message.media_asset_id),
            )[1],
        }
        for message, role in baseline_rows
    )
    return MemoryPacket(
        background="\n".join(memory.content for memory in memories),
        prompt_messages=prompt_messages,
        memories=memories,
        identity_kernel=kernel.content if kernel is not None else {},
        branch_state=branch.state_snapshot,
        continuity_memories=continuity_memories,
        baseline_manifest=(
            {
                "id": manifest.id,
                "import_id": manifest.import_id,
                "boundary_message_id": manifest.boundary_message_id,
                "boundary_timestamp": manifest.boundary_timestamp.isoformat(),
                "message_count": manifest.message_count,
            }
            if manifest is not None
            else None
        ),
        baseline_prompt_messages=baseline_prompt_messages,
        retrieval_intent=policy.intent,
        mental_state=dict(mental_state.state) if mental_state is not None else {},
        active_goals=active_goals,
    )


def _branch_context_memory(session: Session, branch: Branch) -> ContextMemory:
    event = session.get(EventNode, branch.origin_event_id)
    lines = [
        "当前分支设定（回答必须优先遵循，不能与其他分支混用）：",
        f"分支名称：{branch.title}",
        f"分支起点时间：{branch.origin_time.isoformat()}",
    ]
    if event is not None:
        lines.extend(
            (
                f"起点事件：{event.title}",
                f"事件摘要：{event.summary}",
                f"事件主题：{event.topic}",
            )
        )
    if branch.state_snapshot:
        lines.append(
            "分支状态："
            + json.dumps(
                branch.state_snapshot,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return ContextMemory(
        resource_id=f"branch:{branch.id}",
        resource_type="event",
        content="\n".join(lines),
    )


def _group_valid_history(
    history: list[BranchMessage],
    turn_limit: int,
    trusted_model_version_ids: set[str],
) -> list[dict[str, str]]:
    grouped: list[list[BranchMessage]] = []
    for message in history:
        has_visible_payload = bool(message.content.strip()) or bool(
            message.type in {"sticker", "emoji", "image", "audio", "video", "quote"}
            and message.media_asset_id
        )
        if (
            message.generation_status not in (None, "completed")
            or not has_visible_payload
        ):
            continue
        if grouped and grouped[-1][0].turn_id == message.turn_id:
            previous = grouped[-1][-1]
            if (
                previous.content,
                previous.type,
                previous.media_asset_id,
            ) != (
                message.content,
                message.type,
                message.media_asset_id,
            ):
                grouped[-1].append(message)
        else:
            grouped.append([message])
    filtered: list[list[BranchMessage]] = []
    for turn in grouped:
        if turn[0].role == "user":
            filtered.append(turn)
            continue
        metadata = turn[0].generation_metadata or {}
        valid_assistant = (
            metadata.get("model_version_id") in trusted_model_version_ids
            and metadata.get("quarantined") is not True
            and metadata.get("retracted") is not True
            and not any(
                bool((message.generation_metadata or {}).get("degraded"))
                for message in turn
            )
        )
        if valid_assistant:
            filtered.append(turn)
        elif filtered and filtered[-1][0].role == "user":
            filtered.pop()
    if filtered and filtered[-1][0].role == "user":
        filtered.pop()
    prompt_messages: list[dict[str, str]] = []
    for turn in filtered[-turn_limit:]:
        content = "\n".join(
            (
                f"[表情资产:{message.media_asset_id}]"
                if message.type == "sticker"
                else _bounded_history_content(message.content)
            )
            for message in turn
        )
        prompt_messages.append({"role": turn[0].role, "content": content})
    return prompt_messages


def _bounded_history_content(content: str, limit: int = 1_000) -> str:
    normalized = content.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "…"


def _trusted_model_lineage(session: Session, model_version_id: str) -> set[str]:
    """Keep grounded conversation continuity across an accepted model upgrade."""

    trusted = {model_version_id}
    cursor = model_version_id
    for _depth in range(16):
        version = session.get(ModelVersion, cursor)
        if version is None or not isinstance(version.training_config, dict):
            break
        previous = version.training_config.get("upgraded_from_model_version_id")
        if not isinstance(previous, str) or not previous or previous in trusted:
            break
        previous_version = session.get(ModelVersion, previous)
        if (
            previous_version is None
            or previous_version.project_id != version.project_id
            or previous_version.status != "ready"
        ):
            break
        trusted.add(previous)
        cursor = previous
    return trusted
