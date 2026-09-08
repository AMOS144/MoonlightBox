import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.agent.jobs import COGNITIVE_CYCLE_JOB_KIND
from moonlightbox.agent.models import (
    AgentGoal,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
)
from moonlightbox.agent.service import ShadowCognitionService
from moonlightbox.agent.types import FusedAgentTurn
from moonlightbox.branches.actor_models import (
    ConversationActorState,
    ConversationExpressionPlan,
    ConversationPendingBubble,
)
from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.context import (
    ContextBubble,
    ContextBuilder,
    ContextMemory,
    ContextPacket,
    ContextRequest,
    ContextTurn,
)
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_models import BranchMemoryItem, IdentityKernel
from moonlightbox.branches.embeddings import LocalChineseEmbedder, TextEmbedder
from moonlightbox.branches.episodes import EpisodeService
from moonlightbox.branches.generation import BranchGenerator
from moonlightbox.branches.memory import build_memory_packet
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.models import (
    Branch,
    BranchMessage,
    uses_authoritative_cognition,
)
from moonlightbox.branches.proactive import (
    ProactiveDecisionEngine,
    derive_endogenous_impetus,
)
from moonlightbox.branches.replies import (
    GeneratedBubble,
    GeneratedReplyTurn,
    drop_invalid_sticker_bubbles,
    validate_reply_sticker_ids,
)
from moonlightbox.branches.reviewer import ReplyReviewer, ReviewFailedError
from moonlightbox.branches.understanding import (
    grounded_retry_instruction,
    requires_cloud_review,
    understand_contributions,
    validate_grounded_reply,
)
from moonlightbox.config import Settings
from moonlightbox.imports.models import Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.media.models import MediaAsset
from moonlightbox.media.selection import select_reusable_media
from moonlightbox.training.conversation_action_policy import (
    ConversationActionPolicy,
    choose_conversation_action,
    quote_content,
)
from moonlightbox.training.expression_policy import ExpressionPolicy, should_respond
from moonlightbox.training.media_behavior_policy import (
    MediaBehaviorPolicy,
    choose_media_modality,
)
from moonlightbox.training.models import ModelVersion
from moonlightbox.training.sticker_policy import (
    RecentStickerBubble,
    StickerContext,
    StickerPolicy,
    build_sticker_policy_from_database,
    rank_stickers,
    should_send_sticker,
)

BRANCH_CONVERSATION_JOB_KIND = "branch_conversation_actor"


class StickerPolicyRuntimeCache:
    """缓存冻结分支的资产候选策略，避免每轮重复向量化历史。"""

    def __init__(self, *, max_entries: int = 32) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, ...], StickerPolicy] = OrderedDict()

    def get_or_build(
        self,
        key: tuple[str, ...],
        factory: Callable[[], StickerPolicy],
    ) -> StickerPolicy:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        policy = factory()
        self._entries[key] = policy
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return policy


def create_conversation_actor_handler(
    generator: BranchGenerator,
    reviewer: ReplyReviewer | None = None,
    memory_repository: ProjectMemoryRepository | None = None,
    continuity_repository: BranchContinuityRepository | None = None,
) -> JobHandler:
    sticker_policy_cache = StickerPolicyRuntimeCache()

    def handle(job_service: JobService, job: Job) -> None:
        branch_id = job.payload.get("branch_id")
        project_id = job.payload.get("project_id")
        if not isinstance(branch_id, str) or not isinstance(project_id, str):
            raise JobHandlerError("invalid_payload", "会话 Actor 任务参数无效")
        actor = BranchConversationActor(
            job_service.session,
            generator,
            reviewer=reviewer,
            memory_repository=memory_repository,
            continuity_repository=continuity_repository,
            sticker_policy_cache=sticker_policy_cache,
        )
        try:
            actor.process(
                project_id,
                branch_id,
                proactive=bool(job.payload.get("proactive", False)),
                cognitive_cycle_id=(
                    str(job.payload["cognitive_cycle_id"])
                    if isinstance(job.payload.get("cognitive_cycle_id"), str)
                    else None
                ),
                lease_owner=job.worker_token,
            )
        except Exception as error:
            actor.reset_after_failure(branch_id)
            if isinstance(error, ReviewFailedError):
                raise JobHandlerError(
                    "reply_review_failed",
                    "数字人回复复核失败",
                ) from error
            raise

    return handle


class BranchConversationActor:
    def __init__(
        self,
        session: Session,
        generator: BranchGenerator,
        *,
        reviewer: ReplyReviewer | None = None,
        memory_repository: ProjectMemoryRepository | None = None,
        continuity_repository: BranchContinuityRepository | None = None,
        sticker_embedder: TextEmbedder | None = None,
        sticker_policy_cache: StickerPolicyRuntimeCache | None = None,
    ) -> None:
        self._session = session
        self._generator = generator
        self._reviewer = reviewer
        self._memory_repository = memory_repository
        self._continuity_repository = continuity_repository
        self._sticker_policy_cache = (
            sticker_policy_cache
            if sticker_policy_cache is not None
            else StickerPolicyRuntimeCache()
        )
        self._sticker_embedder = (
            sticker_embedder
            if sticker_embedder is not None
            else _default_sticker_embedder(
                Settings().model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
            )
        )

    def process(
        self,
        project_id: str,
        branch_id: str,
        *,
        proactive: bool = False,
        cognitive_cycle_id: str | None = None,
        lease_owner: str | None = None,
    ) -> None:
        branch = self._session.get(Branch, branch_id)
        if branch is None or branch.project_id != project_id:
            raise JobHandlerError("branch_not_found", "时间分支不存在")
        if branch.baseline_status != "ready" or branch.lifecycle_status != "active":
            raise JobHandlerError("branch_not_ready", "时间分支尚不可聊天")
        state = self._state(branch_id)
        now = datetime.now(UTC)
        if (
            state.lease_owner is not None
            and state.lease_owner != lease_owner
            and state.lease_expires_at is not None
            and _aware(state.lease_expires_at) > now
        ):
            return
        state.lease_owner = lease_owner or f"direct-{uuid4()}"
        state.lease_expires_at = now + timedelta(minutes=2)
        self._session.commit()
        for _attempt in range(4):
            contributions = self._unobserved_user_messages(branch_id, state)
            typing_until = state.user_typing_until
            if (
                contributions
                and typing_until is not None
                and _aware(typing_until) > datetime.now(UTC)
            ):
                state.status = "waiting"
                state.assistant_typing_until = None
                state.next_review_at = typing_until
                state.version += 1
                self._release(state)
                self._session.commit()
                return
            if not contributions:
                if not proactive:
                    state.status = "idle"
                    state.assistant_typing_until = None
                    self._release(state)
                    self._session.commit()
                    return
                proactive_cycle = (
                    self._scoped_completed_cycle(branch, cognitive_cycle_id)
                    if cognitive_cycle_id is not None
                    else None
                )
                last_message = self._last_message(branch_id)
                identity_kernel = self._session.scalar(
                    select(IdentityKernel).where(
                        IdentityKernel.model_version_id == branch.model_version_id
                    )
                )
                behavioral_rhythm = self._behavioral_rhythm(branch, identity_kernel)
                recent_reflection_id = self._session.scalar(
                    select(BranchMemoryItem.id)
                    .where(
                        BranchMemoryItem.branch_id == branch.id,
                        BranchMemoryItem.kind == "reflection",
                        BranchMemoryItem.review_status == "approved",
                        BranchMemoryItem.valid_to.is_(None),
                    )
                    .order_by(BranchMemoryItem.created_at.desc())
                )
                active_goals = tuple(
                    (goal.id, goal.priority)
                    for goal in self._session.scalars(
                        select(AgentGoal)
                        .where(
                            AgentGoal.branch_id == branch.id,
                            AgentGoal.status == "active",
                        )
                        .order_by(AgentGoal.priority.desc())
                        .limit(3)
                    )
                )
                situational = branch.state_snapshot.get("situational_state")
                situational_evidence = (
                    situational.get("evidence_ids") if isinstance(situational, dict) else None
                )
                impetus = derive_endogenous_impetus(
                    state={
                        "approach_motivation": state.approach_motivation,
                        "open_sequences": state.open_sequences,
                    },
                    behavioral_rhythm=(
                        behavioral_rhythm if isinstance(behavioral_rhythm, dict) else None
                    ),
                    active_goals=active_goals,
                    recent_reflection_id=recent_reflection_id,
                    situational_evidence_id=(
                        str(situational_evidence[-1])
                        if isinstance(situational_evidence, list) and situational_evidence
                        else None
                    ),
                    now=now,
                )
                if impetus is not None:
                    state.approach_motivation = {
                        "score": impetus.score,
                        "trigger": impetus.trigger,
                        "reason": impetus.reason,
                        "source": "endogenous",
                    }
                decision = ProactiveDecisionEngine().decide(
                    state={
                        "approach_motivation": state.approach_motivation,
                        "inhibition": state.inhibition,
                        "open_sequences": state.open_sequences,
                    },
                    last_message_at=(last_message.created_at if last_message is not None else None),
                    unanswered_assistant_count=self._unanswered_assistant_count(branch_id),
                    behavioral_rhythm=(
                        behavioral_rhythm if isinstance(behavioral_rhythm, dict) else None
                    ),
                )
                state.next_review_at = decision.next_review_at
                if proactive_cycle is None and not decision.should_speak:
                    state.status = "idle"
                    self._release(state)
                    self._session.commit()
                    return
                first_sequence = self._latest_sequence(branch_id) + 1
                basis_sequence = self._latest_sequence(branch_id)
                combined_content = "（此刻没有新的用户消息）"
            else:
                first_sequence = contributions[0].sequence
                basis_sequence = contributions[-1].sequence
                combined_content = "\n".join(message.content for message in contributions)
            state.status = "thinking"
            state.assistant_typing_until = datetime.now(UTC) + timedelta(seconds=15)
            state.attention_focus = {
                "latest_user_sequence": basis_sequence,
                "content": combined_content,
            }
            understanding = understand_contributions(combined_content)
            state.shared_ground = {
                **state.shared_ground,
                "topic": self._current_topic(branch_id, basis_sequence),
                "latest_user_contribution": combined_content,
                "understanding": understanding,
                "open_sequences": list(state.open_sequences),
                "proactive_impetus": (
                    dict(state.approach_motivation) if proactive and not contributions else {}
                ),
            }
            state.version += 1
            self._session.commit()

            packet = self._context_packet(
                branch,
                combined_content,
                first_sequence=first_sequence,
                state=state,
                proactive=proactive and not contributions,
            )
            context = ContextBuilder().to_chat_messages(packet)
            if contributions and packet.reply_protocol == "persona_text":
                self._enqueue_contribution_cognition(
                    branch,
                    contributions,
                    combined_content,
                )
            active_cycle = (
                self._scoped_completed_cycle(branch, cognitive_cycle_id)
                if cognitive_cycle_id is not None
                else self._active_cycle_for_message(branch, contributions[-1])
                if uses_authoritative_cognition(branch.subject_agent_mode)
                and contributions
                and packet.reply_protocol != "persona_text"
                else None
            )
            completed_cycle_decision = (
                active_cycle.final_decision
                if active_cycle is not None
                and active_cycle.status == "succeeded"
                and isinstance(active_cycle.final_decision, dict)
                else {}
            )
            fused_turn: FusedAgentTurn | None = None
            grounding_retry_reason: str | None = None
            if contributions and not self._persona_should_respond(
                branch, combined_content
            ):
                self._persist_silence(
                    state=state,
                    contributions=contributions,
                    basis_sequence=basis_sequence,
                )
                state.shared_ground = {
                    **state.shared_ground,
                    "last_expression_decision": {
                        "express": False,
                        "source": "historical_expression_policy",
                    },
                }
                self._release(state)
                self._session.commit()
                return
            if (
                proactive
                and not contributions
                and completed_cycle_decision.get("express") is False
            ):
                self._persist_silence(
                    state=state,
                    contributions=contributions,
                    basis_sequence=basis_sequence,
                )
                state.shared_ground = {
                    **state.shared_ground,
                    "last_expression_decision": {
                        "express": False,
                        "source": "completed_cognitive_cycle",
                    },
                }
                self._release(state)
                self._session.commit()
                return
            # 公开回复必须由人格 LoRA 在完整会话上下文中直接决定。认知周期只更新
            # 私密状态和表达倾向，不能提供一句脱离上下文的公开内容草稿。
            trusted_content_draft: str | None = None
            if active_cycle is not None and packet.reply_protocol != "persona_text":
                generate_fused = getattr(self._generator, "generate_fused", None)
                if not callable(generate_fused):
                    raise TypeError("active 分支生成器不支持融合推理")
                fused_turn = generate_fused(
                    branch.model_version_id,
                    context[0].content,
                    [{"role": item.role, "content": item.content} for item in context[1:]],
                    allowed_sticker_ids=packet.allowed_sticker_ids,
                )
                if not isinstance(fused_turn, FusedAgentTurn):
                    raise TypeError("融合生成器返回了无效结果")
                draft = fused_turn.reply
            else:
                draft = self._generator.generate(
                    branch.model_version_id,
                    context[0].content,
                    [{"role": item.role, "content": item.content} for item in context[1:]],
                )
            if draft is not None:
                if self._reviewer is not None and requires_cloud_review(packet):
                    draft = self._reviewer.review(packet, draft).reply
                draft = drop_invalid_sticker_bubbles(
                    draft,
                    packet.allowed_sticker_ids,
                )
                validate_reply_sticker_ids(draft, packet.allowed_sticker_ids)
                try:
                    validate_grounded_reply(packet, draft)
                except ValueError as error:
                    grounding_retry_reason = str(error)
                    if trusted_content_draft is not None:
                        # A style adapter may decorate a safe draft with an
                        # unsupported claim. Never answer that failure by asking
                        # the same adapter to invent fresh content. Fall back to
                        # the authoritative cognition/state draft and validate it.
                        draft = GeneratedReplyTurn(
                            bubbles=(
                                GeneratedBubble(
                                    type="text",
                                    content=trusted_content_draft,
                                    delay_ms=0,
                                ),
                            ),
                            raw_output=trusted_content_draft,
                        )
                        try:
                            validate_grounded_reply(packet, draft)
                        except ValueError as trusted_error:
                            raise ReviewFailedError(
                                "可信内容草稿仍未通过事实硬校验"
                            ) from trusted_error
                    else:
                        # Legacy/fused branches do not have a separately trusted
                        # content draft, so retain their bounded public retry.
                        draft = self._generator.generate(
                            branch.model_version_id,
                            context[0].content
                            + "\n"
                            + grounded_retry_instruction(error, packet),
                            [
                                {"role": item.role, "content": item.content}
                                for item in context[1:]
                            ],
                        )
                        if self._reviewer is not None and requires_cloud_review(packet):
                            draft = self._reviewer.review(packet, draft).reply
                        draft = drop_invalid_sticker_bubbles(
                            draft,
                            packet.allowed_sticker_ids,
                        )
                        validate_reply_sticker_ids(draft, packet.allowed_sticker_ids)
                        try:
                            validate_grounded_reply(packet, draft)
                        except ValueError as retry_error:
                            raise ReviewFailedError(
                                "回复二次生成后仍未通过事实与风格硬校验"
                            ) from retry_error
                draft = self._apply_learned_chat_behavior(
                    branch,
                    draft,
                    allowed_sticker_ids=packet.allowed_sticker_ids,
                    seed=f"{branch.id}:{basis_sequence}:{combined_content}",
                )
                draft = self._apply_learned_conversation_action(
                    branch,
                    draft,
                    user_content=combined_content,
                    proactive=proactive and not contributions,
                    seed=f"{branch.id}:{basis_sequence}:{combined_content}",
                )
                draft = self._apply_learned_media_behavior(
                    branch,
                    draft,
                    user_content=combined_content,
                    proactive=proactive and not contributions,
                    seed=f"{branch.id}:{basis_sequence}:{combined_content}",
                )
                validate_reply_sticker_ids(draft, packet.allowed_sticker_ids)
            elif fused_turn is None or fused_turn.expression_decision.express:
                raise TypeError("表达决定为真时必须生成公开回复")

            self._session.expire_all()
            state = self._state(branch_id)
            latest_user_sequence = self._latest_user_sequence(branch_id)
            if latest_user_sequence > basis_sequence:
                if active_cycle is not None:
                    stale_cycle = self._session.get(CognitiveCycle, active_cycle.id)
                    if stale_cycle is not None and stale_cycle.status in {
                        "pending",
                        "running",
                    }:
                        stale_cycle.status = "invalidated"
                        stale_cycle.completed_at = datetime.now(UTC)
                state.status = "observing"
                self._session.commit()
                continue
            if fused_turn is not None and active_cycle is not None:
                ShadowCognitionService(self._session).persist_active_result(
                    active_cycle.id,
                    fused_turn,
                    commit=False,
                )
                if not fused_turn.expression_decision.express:
                    self._persist_silence(
                        state=state,
                        contributions=contributions,
                        basis_sequence=basis_sequence,
                    )
                    self._release(state)
                    self._session.commit()
                    return
                if draft is None:
                    raise TypeError("表达决定为真时必须生成公开回复")
            self._persist_expression(
                branch=branch,
                state=state,
                contributions=contributions,
                first_sequence=first_sequence,
                basis_sequence=basis_sequence,
                reply=draft,
                proactive=proactive and not contributions,
                grounding_retry_reason=grounding_retry_reason,
            )
            self._release(state)
            self._session.commit()
            return
        state.status = "retrying"
        state.assistant_typing_until = None
        state.next_review_at = datetime.now(UTC) + timedelta(seconds=1)
        self._session.commit()
        raise JobHandlerError("conversation_changed", "用户仍在连续输入，稍后重新理解")

    def _release(self, state: ConversationActorState) -> None:
        state.lease_owner = None
        state.lease_expires_at = None

    def reset_after_failure(self, branch_id: str) -> None:
        """失败时清理租约与输入提示，避免前端永久显示正在输入。"""

        self._session.rollback()
        state = self._session.scalar(
            select(ConversationActorState).where(ConversationActorState.branch_id == branch_id)
        )
        if state is None:
            return
        now = datetime.now(UTC)
        inhibition = dict(state.inhibition)
        failure_count = int(_number(inhibition.get("failure_count"), 0)) + 1
        backoff_minutes = min(360, 5 * (2 ** min(failure_count - 1, 6)))
        inhibition["failure_count"] = failure_count
        inhibition["score"] = min(
            1.0,
            _number(inhibition.get("score"), 0.0) + 0.1,
        )
        inhibition["last_failure_at"] = now.isoformat()
        state.status = "idle"
        state.assistant_typing_until = None
        state.next_review_at = now + timedelta(minutes=backoff_minutes)
        state.inhibition = inhibition
        self._release(state)
        state.version += 1
        self._session.commit()

    def _persist_expression(
        self,
        *,
        branch: Branch,
        state: ConversationActorState,
        contributions: list[BranchMessage],
        first_sequence: int,
        basis_sequence: int,
        reply: object,
        proactive: bool,
        grounding_retry_reason: str | None,
    ) -> None:
        from moonlightbox.branches.replies import GeneratedReplyTurn

        if not isinstance(reply, GeneratedReplyTurn):
            raise TypeError("生成器返回了无效回复")
        generation_metadata: dict[str, object] = {
            "model_version_id": branch.model_version_id,
            "degraded": reply.degraded,
            "normalization": reply.normalization,
            "policy_appended": reply.policy_appended,
            "actor_state_version": state.version,
            "grounding_retry_count": int(grounding_retry_reason is not None),
            "grounding_retry_reason": grounding_retry_reason,
        }
        plan = ConversationExpressionPlan(
            branch_id=branch.id,
            trigger_sequence_from=first_sequence,
            trigger_sequence_to=basis_sequence,
            intent="回应最新连续贡献",
            state_version=state.version,
            status="expressing",
            proactive=proactive,
            generation_metadata=generation_metadata,
        )
        self._session.add(plan)
        self._session.flush()
        pending: list[ConversationPendingBubble] = []
        scheduled_at = datetime.now(UTC)
        for index, bubble in enumerate(reply.bubbles):
            asset_id = bubble.asset_id
            if bubble.type == "sticker":
                asset = self._session.get(MediaAsset, asset_id)
                if (
                    asset is None
                    or asset.project_id != branch.project_id
                    or asset.kind != "sticker"
                ):
                    raise ValueError("模型返回了无效表情资产")
            elif bubble.type in {"image", "audio", "video"}:
                asset = self._session.get(MediaAsset, asset_id)
                if (
                    asset is None
                    or asset.project_id != branch.project_id
                    or not asset.mime_type.startswith(f"{bubble.type}/")
                ):
                    raise ValueError("生成行为引用了未经验证的媒体资产")
            elif bubble.type == "quote":
                _validate_quote_action(bubble.content)
            elif bubble.type == "reaction":
                _validate_reaction_action(bubble.content)
            elif bubble.type == "retract" and index == 0:
                raise ValueError("撤回行为之前必须存在已发送内容")
            scheduled_at += timedelta(milliseconds=max(0, bubble.delay_ms))
            pending.append(
                ConversationPendingBubble(
                    plan_id=plan.id,
                    branch_id=branch.id,
                    position=index,
                    content=bubble.content or "",
                    message_type=bubble.type,
                    media_asset_id=asset_id,
                    delay_ms=bubble.delay_ms,
                    send_not_before=scheduled_at,
                    basis_sequence=basis_sequence,
                )
            )
        self._session.add_all(pending)
        self._session.flush()
        now = datetime.now(UTC)
        for contribution in contributions:
            contribution.observed_at = now
        if contributions:
            state.observed_message_sequence = basis_sequence
        state.status = "expressing"
        state.assistant_typing_until = max(
            (item.send_not_before or now for item in pending),
            default=now,
        ) + timedelta(seconds=1)
        state.expression_intentions = [
            {
                "plan_id": plan.id,
                "basis_sequence": basis_sequence,
                "bubble_count": len(pending),
            }
        ]
        state.next_review_at = min(
            (item.send_not_before or now for item in pending),
            default=now,
        )
        state.version += 1
        deliver_due_pending_bubbles(
            self._session,
            branch_id=branch.id,
            now=now,
            commit=False,
        )

    def _persist_silence(
        self,
        *,
        state: ConversationActorState,
        contributions: list[BranchMessage],
        basis_sequence: int,
    ) -> None:
        """把沉默作为已执行决定，不创建任何助手消息。"""

        now = datetime.now(UTC)
        for contribution in contributions:
            contribution.observed_at = now
        if contributions:
            state.observed_message_sequence = basis_sequence
        state.status = "idle"
        state.assistant_typing_until = None
        state.expression_intentions = []
        state.next_review_at = now + timedelta(minutes=30)
        state.version += 1

    def _current_topic(self, branch_id: str, basis_sequence: int) -> str:
        """从最近真实对话中保留仍在延续的话题原文。"""

        messages = list(
            self._session.scalars(
                select(BranchMessage)
                .where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.sequence <= basis_sequence,
                    BranchMessage.type == "text",
                )
                .order_by(BranchMessage.sequence.desc())
                .limit(8)
            )
        )
        low_signal = ("不知道", "不确定", "纠结", "没想好", "嗯", "哦", "啊")
        for message in messages:
            content = message.content.strip().replace("\n", " / ")
            if not content:
                continue
            if any(marker in content for marker in low_signal) and len(content) <= 12:
                continue
            return content[:100]
        return ""

    def _active_cycle_for_message(
        self,
        branch: Branch,
        message: BranchMessage,
    ) -> CognitiveCycle:
        event = self._session.scalar(
            select(PerceptionEvent).where(
                PerceptionEvent.branch_id == branch.id,
                PerceptionEvent.idempotency_key == f"user-message:{message.id}",
            )
        )
        if event is None:
            raise RuntimeError("active 分支缺少用户感知事件")
        cycle = self._session.scalar(
            select(CognitiveCycle).where(
                CognitiveCycle.branch_id == branch.id,
                CognitiveCycle.trigger_event_id == event.id,
            )
        )
        if cycle is None:
            raise RuntimeError("active 分支缺少实时认知周期")
        return cycle

    def _enqueue_contribution_cognition(
        self,
        branch: Branch,
        contributions: list[BranchMessage],
        content: str,
    ) -> CognitiveCycle:
        """为一次连续表达创建唯一的后台认知周期。"""

        first = contributions[0]
        last = contributions[-1]
        cognition = ShadowCognitionService(self._session)
        event = cognition.append_perception_event(
            project_id=branch.project_id,
            branch_id=branch.id,
            event_type="user_contribution_cluster",
            occurred_at=last.created_at,
            source="conversation_actor",
            idempotency_key=f"user-cluster:{first.id}:{last.id}",
            evidence={
                "content": content,
                "branch_message_ids": [item.id for item in contributions],
                "sequence_from": first.sequence,
                "sequence_to": last.sequence,
            },
            commit=False,
        )
        current = self._session.scalar(
            select(MentalStateVersion).where(
                MentalStateVersion.branch_id == branch.id,
                MentalStateVersion.is_current.is_(True),
            )
        )
        if current is None:
            current = cognition.ensure_initial_mental_state(
                project_id=branch.project_id,
                branch_id=branch.id,
                state=_initial_mental_state(branch.state_snapshot),
                evidence={"branch_state_snapshot": True},
                model_version_id=branch.model_version_id,
                commit=False,
            )
        cycle = cognition.create_pending_cycle(
            project_id=branch.project_id,
            branch_id=branch.id,
            trigger_event_id=event.id,
            input_cutoff_at=last.created_at,
            starting_state_version_id=current.id,
            model_version_id=branch.model_version_id,
            evidence={
                "branch_message_ids": [item.id for item in contributions],
                "contribution_cluster": True,
            },
            commit=False,
        )
        JobService(self._session).enqueue_unique(
            COGNITIVE_CYCLE_JOB_KIND,
            {
                "project_id": branch.project_id,
                "branch_id": branch.id,
                "cycle_id": cycle.id,
            },
            dedupe_key=f"{COGNITIVE_CYCLE_JOB_KIND}:{cycle.id}",
            commit=False,
        )
        return cycle

    def _scoped_completed_cycle(
        self,
        branch: Branch,
        cycle_id: str,
    ) -> CognitiveCycle:
        cycle = self._session.get(CognitiveCycle, cycle_id)
        if (
            cycle is None
            or cycle.project_id != branch.project_id
            or cycle.branch_id != branch.id
            or cycle.status != "succeeded"
        ):
            raise RuntimeError("主动表达引用了无效的认知周期")
        return cycle

    def _context_packet(
        self,
        branch: Branch,
        content: str,
        *,
        first_sequence: int,
        state: ConversationActorState,
        proactive: bool,
    ) -> ContextPacket:
        history = list(
            self._session.scalars(
                select(BranchMessage)
                .where(
                    BranchMessage.branch_id == branch.id,
                    BranchMessage.sequence < first_sequence,
                )
                .order_by(BranchMessage.sequence)
            )
        )
        memory = build_memory_packet(
            self._session,
            branch,
            content,
            history,
            repository=self._memory_repository,
            continuity_repository=self._continuity_repository,
        )
        participant_name = (
            self._session.scalar(
                select(Participant.name).where(
                    Participant.project_id == branch.project_id,
                    Participant.role == "target",
                )
            )
            or "对方"
        )
        active, contested = self._belief_context(memory.branch_state or {})
        model_version = self._session.get(ModelVersion, branch.model_version_id)
        reply_protocol = (
            "persona_text"
            if model_version is not None
            and str(model_version.training_config.get("reply_protocol_version", "")).startswith(
                "persona-text"
            )
            else "compact"
        )
        return ContextBuilder().build_packet(
            ContextRequest(
                persona=participant_name,
                cutoff=branch.origin_time.isoformat(),
                memories=memory.memories,
                history=tuple(
                    ContextTurn(
                        role=("assistant" if item["role"] == "assistant" else "user"),
                        bubbles=(ContextBubble(type="text", content=item["content"]),),
                    )
                    for item in memory.prompt_messages
                ),
                current_user_content=content,
                allowed_sticker_ids=(
                    () if proactive else self._sticker_top_k(branch, content, history)
                ),
                identity_kernel=memory.identity_kernel or {},
                branch_state=memory.branch_state or {},
                active_beliefs=active,
                contested_beliefs=contested,
                shared_ground=state.shared_ground,
                continuity_memories=memory.continuity_memories,
                baseline_manifest=memory.baseline_manifest or {},
                baseline_history=tuple(
                    ContextTurn(
                        role=("assistant" if item["role"] == "assistant" else "user"),
                        bubbles=(ContextBubble(type="text", content=item["content"]),),
                    )
                    for item in memory.baseline_prompt_messages
                ),
                conversation_mode="proactive" if proactive else "responsive",
                reply_protocol=reply_protocol,
                mental_state=memory.mental_state or {},
                active_goals=memory.active_goals,
            )
        )

    def _sticker_top_k(
        self,
        branch: Branch,
        content: str,
        history: list[BranchMessage],
    ) -> tuple[str, ...]:
        version = self._session.get(ModelVersion, branch.model_version_id)
        if version is None:
            return ()
        configured = StickerPolicy.from_metadata(version.training_config.get("sticker_policy"))
        if not configured.enabled:
            return ()
        manifest = self._session.scalar(
            select(BranchBaselineManifest).where(BranchBaselineManifest.branch_id == branch.id)
        )
        target_id = self._session.scalar(
            select(Participant.id).where(
                Participant.project_id == branch.project_id,
                Participant.role == "target",
            )
        )
        if manifest is None or target_id is None:
            return ()
        cache_key = (
            branch.id,
            branch.model_version_id,
            manifest.id,
            manifest.boundary_timestamp.isoformat(),
            manifest.boundary_source_id,
            manifest.boundary_message_id,
            configured.version,
            json.dumps(configured.parameters, sort_keys=True),
        )
        policy = self._sticker_policy_cache.get_or_build(
            cache_key,
            lambda: build_sticker_policy_from_database(
                self._session,
                project_id=branch.project_id,
                target_id=target_id,
                cutoff=manifest.boundary_timestamp,
                branch_time=branch.created_at,
                import_id=manifest.import_id,
                boundary_source_id=manifest.boundary_source_id,
                boundary_message_id=manifest.boundary_message_id,
                parameters=configured.parameters,
                embedder=self._sticker_embedder,
            ),
        )
        context = StickerContext(
            current_text=content,
            recent=tuple(
                RecentStickerBubble(
                    kind=message.type,
                    content=message.content,
                    asset_id=message.media_asset_id,
                    timestamp=message.created_at,
                    sender="target" if message.role == "assistant" else "other",
                )
                for message in history[-8:]
            ),
        )
        if not should_send_sticker(policy, context):
            return ()
        return tuple(item.asset_id for item in rank_stickers(policy, context))

    def _apply_learned_chat_behavior(
        self,
        branch: Branch,
        reply: GeneratedReplyTurn,
        *,
        allowed_sticker_ids: tuple[str, ...],
        seed: str,
    ) -> GeneratedReplyTurn:
        """Apply learned modality and rhythm outside the language adapter."""

        kernel = self._session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == branch.model_version_id)
        )
        rhythm = self._behavioral_rhythm(branch, kernel)
        base_delay = _learned_inter_bubble_delay(rhythm if isinstance(rhythm, dict) else {})
        bubbles: list[GeneratedBubble] = []
        for index, bubble in enumerate(reply.bubbles):
            delay = 0
            if index:
                delay = _personalized_delay(base_delay, f"{seed}:{index}")
                if base_delay is None:
                    delay = bubble.delay_ms
            bubbles.append(
                GeneratedBubble(
                    content=bubble.content,
                    delay_ms=delay,
                    type=bubble.type,
                    asset_id=bubble.asset_id,
                )
            )
        has_sticker = any(bubble.type == "sticker" for bubble in bubbles)
        policy_appended = reply.policy_appended
        if allowed_sticker_ids and not has_sticker:
            bubbles.append(
                GeneratedBubble(
                    content=None,
                    delay_ms=_personalized_delay(
                        base_delay if base_delay is not None else 800,
                        f"{seed}:sticker",
                    ),
                    type="sticker",
                    asset_id=allowed_sticker_ids[0],
                )
            )
            policy_appended = True
        return GeneratedReplyTurn(
            bubbles=tuple(bubbles),
            raw_output=reply.raw_output,
            degraded=reply.degraded,
            normalization=reply.normalization,
            policy_appended=policy_appended,
            attempted_invalid_sticker_ids=reply.attempted_invalid_sticker_ids,
        )

    def _behavioral_rhythm(
        self,
        branch: Branch,
        kernel: IdentityKernel | None,
    ) -> dict[str, object] | None:
        content = kernel.content if kernel is not None else {}
        rhythm = content.get("behavioral_rhythm") if isinstance(content, dict) else None
        if isinstance(rhythm, dict) and rhythm:
            return rhythm
        version = self._session.get(ModelVersion, branch.model_version_id)
        configured = (
            version.training_config.get("behavioral_rhythm")
            if version is not None and isinstance(version.training_config, dict)
            else None
        )
        return configured if isinstance(configured, dict) and configured else None

    def _apply_learned_conversation_action(
        self,
        branch: Branch,
        reply: GeneratedReplyTurn,
        *,
        user_content: str,
        proactive: bool,
        seed: str,
    ) -> GeneratedReplyTurn:
        version = self._session.get(ModelVersion, branch.model_version_id)
        policy = ConversationActionPolicy.from_metadata(
            version.training_config.get("conversation_action_policy")
            if version is not None and isinstance(version.training_config, dict)
            else None
        )
        action = choose_conversation_action(
            policy,
            user_content,
            seed=seed,
            proactive=proactive,
        )
        if action is None:
            return reply
        bubbles = list(reply.bubbles)
        if action == "quote":
            for index, bubble in enumerate(bubbles):
                if bubble.type == "text" and bubble.content:
                    bubbles[index] = GeneratedBubble(
                        type="quote",
                        content=quote_content(bubble.content, user_content),
                        delay_ms=bubble.delay_ms,
                    )
                    break
            else:
                return reply
        elif action == "call":
            bubbles.append(
                GeneratedBubble(
                    type="call",
                    content="发起了语音通话",
                    delay_ms=self._action_delay(branch, seed),
                )
            )
        elif action == "reaction":
            bubbles.append(
                GeneratedBubble(
                    type="reaction",
                    content=json.dumps(
                        {"reaction": "拍了拍", "target_content": user_content[:80]},
                        ensure_ascii=False,
                    ),
                    delay_ms=self._action_delay(branch, seed),
                )
            )
        elif action == "retract" and bubbles:
            bubbles.append(
                GeneratedBubble(
                    type="retract",
                    content=None,
                    delay_ms=self._action_delay(branch, seed),
                )
            )
        return GeneratedReplyTurn(
            bubbles=tuple(bubbles),
            raw_output=reply.raw_output,
            degraded=reply.degraded,
            normalization=reply.normalization,
            policy_appended=True,
            attempted_invalid_sticker_ids=reply.attempted_invalid_sticker_ids,
        )

    def _action_delay(self, branch: Branch, seed: str) -> int:
        rhythm = self._behavioral_rhythm(branch, None)
        base_delay = _learned_inter_bubble_delay(rhythm or {})
        return _personalized_delay(base_delay or 800, f"{seed}:action")

    def _apply_learned_media_behavior(
        self,
        branch: Branch,
        reply: GeneratedReplyTurn,
        *,
        user_content: str,
        proactive: bool,
        seed: str,
    ) -> GeneratedReplyTurn:
        """Add only approved, branch-bounded media with a strong semantic match."""

        version = self._session.get(ModelVersion, branch.model_version_id)
        policy = MediaBehaviorPolicy.from_metadata(
            version.training_config.get("media_behavior_policy")
            if version is not None and isinstance(version.training_config, dict)
            else None
        )
        modality = choose_media_modality(
            policy,
            user_content,
            seed=seed,
            proactive=proactive,
        )
        if modality is None:
            return reply
        draft_text = "\n".join(
            bubble.content or "" for bubble in reply.bubbles if bubble.type == "text"
        ).strip()
        query = draft_text if modality == "audio" else "\n".join(
            item for item in (user_content.strip(), draft_text) if item
        )
        selection = select_reusable_media(
            self._session,
            branch=branch,
            modality=modality,
            query=query,
            embedder=self._sticker_embedder,
        )
        if selection is None:
            return reply
        media_bubble = GeneratedBubble(
            type=modality,
            content=None,
            asset_id=selection.asset_id,
            delay_ms=self._action_delay(branch, f"{seed}:media"),
        )
        if modality == "audio":
            # The approved recording becomes the public expression only when
            # its transcript closely matches the intended draft; avoid sending
            # duplicate text plus voice.
            bubbles = [
                bubble for bubble in reply.bubbles if bubble.type != "text"
            ]
            bubbles.insert(0, media_bubble)
        elif len(reply.bubbles) < 10:
            bubbles = [*reply.bubbles, media_bubble]
        else:
            return reply
        return GeneratedReplyTurn(
            bubbles=tuple(bubbles),
            raw_output=reply.raw_output,
            degraded=reply.degraded,
            normalization=reply.normalization,
            policy_appended=True,
            attempted_invalid_sticker_ids=reply.attempted_invalid_sticker_ids,
        )

    def _persona_should_respond(self, branch: Branch, content: str) -> bool:
        version = self._session.get(ModelVersion, branch.model_version_id)
        if version is None:
            return True
        policy = ExpressionPolicy.from_metadata(version.training_config.get("expression_policy"))
        return should_respond(policy, content)

    def _belief_context(
        self, branch_state: dict[str, object]
    ) -> tuple[tuple[ContextMemory, ...], tuple[ContextMemory, ...]]:
        active_ids = _string_list(branch_state.get("active_belief_ids"))
        contested_ids = _string_list(branch_state.get("contested_belief_ids"))
        items = (
            {
                item.id: item
                for item in self._session.scalars(
                    select(BranchMemoryItem).where(
                        BranchMemoryItem.id.in_([*active_ids, *contested_ids]),
                        BranchMemoryItem.valid_to.is_(None),
                    )
                )
            }
            if active_ids or contested_ids
            else {}
        )

        def convert(ids: list[str]) -> tuple[ContextMemory, ...]:
            return tuple(
                ContextMemory(
                    resource_id=f"belief:{item.id}",
                    resource_type="belief",
                    content=item.content,
                    authority="subjective",
                    evidence_ids=tuple(item.source_episode_ids),
                )
                for item_id in ids
                if (item := items.get(item_id)) is not None
            )

        return convert(active_ids), convert(contested_ids)

    def _state(self, branch_id: str) -> ConversationActorState:
        state = self._session.scalar(
            select(ConversationActorState).where(ConversationActorState.branch_id == branch_id)
        )
        if state is None:
            state = ConversationActorState(branch_id=branch_id)
            self._session.add(state)
            self._session.flush()
        return state

    def _unobserved_user_messages(
        self,
        branch_id: str,
        state: ConversationActorState,
    ) -> list[BranchMessage]:
        return list(
            self._session.scalars(
                select(BranchMessage)
                .where(
                    BranchMessage.branch_id == branch_id,
                    BranchMessage.role == "user",
                    BranchMessage.sequence > state.observed_message_sequence,
                )
                .order_by(BranchMessage.sequence)
            )
        )

    def _latest_user_sequence(self, branch_id: str) -> int:
        value = self._session.scalar(
            select(func.max(BranchMessage.sequence)).where(
                BranchMessage.branch_id == branch_id,
                BranchMessage.role == "user",
            )
        )
        return int(value) if value is not None else -1

    def _latest_sequence(self, branch_id: str) -> int:
        value = self._session.scalar(
            select(func.max(BranchMessage.sequence)).where(BranchMessage.branch_id == branch_id)
        )
        return int(value) if value is not None else -1

    def _last_message(self, branch_id: str) -> BranchMessage | None:
        return self._session.scalar(
            select(BranchMessage)
            .where(BranchMessage.branch_id == branch_id)
            .order_by(BranchMessage.sequence.desc())
        )

    def _unanswered_assistant_count(self, branch_id: str) -> int:
        messages = list(
            self._session.scalars(
                select(BranchMessage)
                .where(BranchMessage.branch_id == branch_id)
                .order_by(BranchMessage.sequence.desc())
                .limit(12)
            )
        )
        count = 0
        for message in messages:
            if message.role != "assistant":
                break
            count += 1
        return count


def enqueue_due_actor_reviews(session: Session) -> int:
    """为到达自然复查时间的分支创建幂等主动联系任务。"""

    now = datetime.now(UTC)
    states = list(
        session.scalars(
            select(ConversationActorState).where(
                ConversationActorState.next_review_at.is_not(None),
                ConversationActorState.next_review_at <= now,
            )
        )
    )
    created = 0
    for state in states:
        branch = session.get(Branch, state.branch_id)
        if (
            branch is None
            or branch.lifecycle_status != "active"
            or branch.baseline_status != "ready"
        ):
            continue
        review_timestamp = (
            int(state.next_review_at.timestamp()) if state.next_review_at is not None else 0
        )
        dedupe_key = f"branch-proactive:{branch.id}:{state.version}:{review_timestamp}"
        if session.scalar(select(Job.id).where(Job.dedupe_key == dedupe_key)):
            continue
        session.add(
            Job(
                kind=BRANCH_CONVERSATION_JOB_KIND,
                payload={
                    "project_id": branch.project_id,
                    "branch_id": branch.id,
                    "proactive": True,
                },
                dedupe_key=dedupe_key,
            )
        )
        # 到期时间是一次性触发器；入队时原子消费，避免 Worker 执行期间因
        # Actor 版本变化再次排入相同主动表达任务。
        state.next_review_at = None
        state.version += 1
        created += 1
    if created:
        session.commit()
    return created


def deliver_due_pending_bubbles(
    session: Session,
    *,
    branch_id: str | None = None,
    now: datetime | None = None,
    commit: bool = True,
) -> int:
    """Persist only bubbles whose server-side delivery time has arrived.

    Pending bubbles stay absent from the public message stream, so a new user
    contribution can cancel them before they become visible.
    """

    current = now or datetime.now(UTC)
    query = select(ConversationExpressionPlan).where(
        ConversationExpressionPlan.status == "expressing"
    )
    if branch_id is not None:
        query = query.where(ConversationExpressionPlan.branch_id == branch_id)
    plans = list(session.scalars(query.order_by(ConversationExpressionPlan.created_at)))
    sent_count = 0
    for plan in plans:
        latest_user_sequence = session.scalar(
            select(func.max(BranchMessage.sequence)).where(
                BranchMessage.branch_id == plan.branch_id,
                BranchMessage.role == "user",
            )
        )
        if (
            latest_user_sequence is not None
            and int(latest_user_sequence) > plan.trigger_sequence_to
        ):
            _interrupt_expression_plan(session, plan)
            continue
        pending = list(
            session.scalars(
                select(ConversationPendingBubble)
                .where(
                    ConversationPendingBubble.plan_id == plan.id,
                    ConversationPendingBubble.status == "pending",
                )
                .order_by(ConversationPendingBubble.position)
            )
        )
        for item in pending:
            due_at = item.send_not_before or item.created_at
            if _aware(due_at) > _aware(current):
                break
            if item.message_type == "retract":
                target = session.scalar(
                    select(BranchMessage)
                    .where(
                        BranchMessage.expression_plan_id == plan.id,
                        BranchMessage.role == "assistant",
                        BranchMessage.bubble_index < item.position,
                        BranchMessage.type.not_in(("system", "reaction")),
                    )
                    .order_by(BranchMessage.bubble_index.desc())
                )
                if target is None:
                    item.status = "cancelled"
                    item.last_error = "retract_target_missing"
                    continue
                target.generation_metadata = {
                    **target.generation_metadata,
                    "retracted": True,
                }
                message = BranchMessage(
                    branch_id=plan.branch_id,
                    sequence=_latest_branch_sequence(session, plan.branch_id) + 1,
                    role="assistant",
                    content="撤回了一条消息",
                    type="system",
                    turn_id=plan.id,
                    bubble_index=item.position,
                    delay_ms=item.delay_ms,
                    generation_status="completed",
                    generation_metadata=dict(plan.generation_metadata or {}),
                    expression_plan_id=plan.id,
                    actor_intent="撤回刚发送的内容",
                    is_proactive=plan.proactive,
                    created_at=current,
                )
                session.add(message)
                session.flush()
                _append_agent_expression_event(session, plan, message, current)
                item.sent_message_id = message.id
                item.status = "sent"
                sent_count += 1
                continue
            message = BranchMessage(
                branch_id=plan.branch_id,
                sequence=_latest_branch_sequence(session, plan.branch_id) + 1,
                role="assistant",
                content=item.content,
                type=item.message_type,
                media_asset_id=item.media_asset_id,
                turn_id=plan.id,
                bubble_index=item.position,
                delay_ms=item.delay_ms,
                generation_status="completed",
                generation_metadata=dict(plan.generation_metadata or {}),
                expression_plan_id=plan.id,
                actor_intent=plan.intent,
                is_proactive=plan.proactive,
                created_at=current,
            )
            session.add(message)
            session.flush()
            _append_agent_expression_event(session, plan, message, current)
            item.sent_message_id = message.id
            item.status = "sent"
            sent_count += 1
        still_pending = session.scalar(
            select(func.count())
            .select_from(ConversationPendingBubble)
            .where(
                ConversationPendingBubble.plan_id == plan.id,
                ConversationPendingBubble.status == "pending",
            )
        )
        if int(still_pending or 0) == 0:
            _complete_expression_plan(session, plan, current)
        else:
            state = session.scalar(
                select(ConversationActorState).where(
                    ConversationActorState.branch_id == plan.branch_id
                )
            )
            next_due = session.scalar(
                select(func.min(ConversationPendingBubble.send_not_before)).where(
                    ConversationPendingBubble.plan_id == plan.id,
                    ConversationPendingBubble.status == "pending",
                )
            )
            if state is not None:
                state.status = "expressing"
                state.assistant_typing_until = (
                    _aware(next_due) + timedelta(seconds=1)
                    if isinstance(next_due, datetime)
                    else current + timedelta(seconds=1)
                )
                state.next_review_at = next_due
    if commit:
        session.commit()
    return sent_count


def _append_agent_expression_event(
    session: Session,
    plan: ConversationExpressionPlan,
    message: BranchMessage,
    occurred_at: datetime,
) -> PerceptionEvent:
    """Project a delivered public bubble into the subject's durable perception."""

    branch = session.get(Branch, plan.branch_id)
    if branch is None:
        raise RuntimeError("公开表达所属分支不存在")
    return ShadowCognitionService(session).append_perception_event(
        project_id=branch.project_id,
        branch_id=branch.id,
        event_type="agent_expression",
        occurred_at=occurred_at,
        source="branch_conversation",
        idempotency_key=f"assistant-message:{message.id}",
        evidence={
            "branch_message_id": message.id,
            "content": message.content,
            "role": message.role,
            "message_type": message.type,
            "turn_id": message.turn_id,
            "bubble_index": message.bubble_index,
            "media_asset_id": message.media_asset_id,
            "is_proactive": message.is_proactive,
        },
        commit=False,
    )


def interrupt_pending_expression(
    session: Session,
    branch_id: str,
    *,
    commit: bool = False,
) -> int:
    """Cancel every not-yet-sent bubble when the human interjects."""

    plans = list(
        session.scalars(
            select(ConversationExpressionPlan).where(
                ConversationExpressionPlan.branch_id == branch_id,
                ConversationExpressionPlan.status == "expressing",
            )
        )
    )
    for plan in plans:
        _interrupt_expression_plan(session, plan)
    if plans:
        state = session.scalar(
            select(ConversationActorState).where(ConversationActorState.branch_id == branch_id)
        )
        if state is not None:
            state.status = "observing"
            state.assistant_typing_until = None
            state.expression_intentions = []
            state.next_review_at = None
            state.version += 1
    if commit:
        session.commit()
    return len(plans)


def _interrupt_expression_plan(
    session: Session,
    plan: ConversationExpressionPlan,
) -> None:
    for item in session.scalars(
        select(ConversationPendingBubble).where(
            ConversationPendingBubble.plan_id == plan.id,
            ConversationPendingBubble.status == "pending",
        )
    ):
        item.status = "cancelled"
        item.last_error = "interrupted_by_new_user_message"
    plan.status = "interrupted"
    plan.revision += 1
    _record_expression_episode(session, plan)


def _complete_expression_plan(
    session: Session,
    plan: ConversationExpressionPlan,
    now: datetime,
) -> None:
    plan.status = "completed"
    assistants = list(
        session.scalars(
            select(BranchMessage)
            .where(BranchMessage.expression_plan_id == plan.id)
            .order_by(BranchMessage.bubble_index)
        )
    )
    state = session.scalar(
        select(ConversationActorState).where(ConversationActorState.branch_id == plan.branch_id)
    )
    if state is not None:
        state.status = "idle"
        state.assistant_typing_until = None
        state.expression_intentions = []
        if plan.proactive:
            approach = dict(state.approach_motivation)
            approach["consumed_trigger"] = approach.get("trigger")
            approach["score"] = max(0.0, _number(approach.get("score"), 0.0) - 0.2)
            state.approach_motivation = approach
            inhibition = dict(state.inhibition)
            inhibition["score"] = min(
                1.0,
                _number(inhibition.get("score"), 0.0) + 0.15,
            )
            state.inhibition = inhibition
            state.next_review_at = now + timedelta(hours=2)
        else:
            has_question = any(
                message.content.rstrip().endswith(("吗", "呢", "？", "?"))
                or any(marker in message.content for marker in ("还是", "或者", "要不要"))
                for message in assistants
            )
            state.open_sequences = (
                [
                    {
                        "topic": " / ".join(
                            message.content.strip()
                            for message in assistants
                            if message.content.strip()
                        )[:120],
                        "source_plan_id": plan.id,
                    }
                ]
                if has_question
                else list(state.open_sequences)
            )
            state.shared_ground = {
                **state.shared_ground,
                "open_sequences": list(state.open_sequences),
            }
            state.approach_motivation = {
                "score": 0.6 if has_question else 0.45,
                "trigger": f"user-message-{plan.trigger_sequence_to}",
                "reason": "最近互动和未完成表达",
            }
            state.inhibition = {"score": 0.05}
            state.next_review_at = now + timedelta(minutes=30)
        state.version += 1
    _record_expression_episode(session, plan)


def _record_expression_episode(
    session: Session,
    plan: ConversationExpressionPlan,
) -> None:
    if plan.proactive:
        return
    branch = session.get(Branch, plan.branch_id)
    if branch is None:
        return
    contributions = list(
        session.scalars(
            select(BranchMessage)
            .where(
                BranchMessage.branch_id == plan.branch_id,
                BranchMessage.role == "user",
                BranchMessage.sequence >= plan.trigger_sequence_from,
                BranchMessage.sequence <= plan.trigger_sequence_to,
            )
            .order_by(BranchMessage.sequence)
        )
    )
    assistants = list(
        session.scalars(
            select(BranchMessage)
            .where(BranchMessage.expression_plan_id == plan.id)
            .order_by(BranchMessage.bubble_index)
        )
    )
    assistants = [
        item
        for item in assistants
        if item.generation_metadata.get("retracted") is not True
        and item.type != "system"
    ]
    if contributions and assistants:
        EpisodeService(session).record_turn(branch, contributions, assistants)


def _latest_branch_sequence(session: Session, branch_id: str) -> int:
    value = session.scalar(
        select(func.max(BranchMessage.sequence)).where(BranchMessage.branch_id == branch_id)
    )
    return int(value) if value is not None else -1


def _validate_quote_action(content: str | None) -> None:
    try:
        payload = json.loads(content or "")
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("引用行为缺少结构化内容") from error
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("text"), str)
        or not str(payload["text"]).strip()
        or not isinstance(payload.get("quoted_content"), str)
        or not str(payload["quoted_content"]).strip()
    ):
        raise ValueError("引用行为内容无效")


def _validate_reaction_action(content: str | None) -> None:
    try:
        payload = json.loads(content or "")
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("反应行为缺少结构化内容") from error
    if not isinstance(payload, dict) or payload.get("reaction") != "拍了拍":
        raise ValueError("反应行为内容无效")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _number(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def _initial_mental_state(snapshot: dict[str, object]) -> dict[str, object]:
    """从分支快照提取认知层允许演化的初始状态。"""

    mapping_keys = (
        "persona_state",
        "relationship_state",
        "user_model",
        "emotional_tendency",
        "current_goals",
        "current_concerns",
    )
    list_keys = ("active_belief_ids", "contested_belief_ids")
    state: dict[str, object] = {
        key: dict(value) if isinstance(value := snapshot.get(key), dict) else {}
        for key in mapping_keys
    }
    for key in list_keys:
        value = snapshot.get(key)
        state[key] = (
            [item for item in value if isinstance(item, str)]
            if isinstance(value, list)
            else []
        )
    return state


def _learned_inter_bubble_delay(rhythm: dict[str, object]) -> int | None:
    distribution = rhythm.get("inter_bubble_delay_ms")
    if not isinstance(distribution, dict):
        return None
    count = distribution.get("count")
    median = distribution.get("p50")
    if not isinstance(count, int) or count < 3 or not isinstance(median, int | float):
        return None
    return max(100, min(5_000, int(median)))


def _personalized_delay(base_delay: int | None, seed: str) -> int:
    base = 800 if base_delay is None else base_delay
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    factor = 0.7 + (int.from_bytes(digest[:2], "big") / 65_535) * 0.6
    return max(100, min(5_000, round(base * factor)))


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@lru_cache(maxsize=1)
def _default_sticker_embedder(model_root: Path) -> TextEmbedder | None:
    """本地语义模型不可用时关闭语义通道，不回退到外部服务。"""

    try:
        return LocalChineseEmbedder(model_root)
    except Exception:
        return None
