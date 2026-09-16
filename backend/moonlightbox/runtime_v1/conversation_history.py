"""统一历史视图：来源边界留在后端，模型只用消息引用检索与展开。"""

from datetime import UTC

from sqlalchemy import func, select

from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.wechat_rendering import normalize_wechat_display

from .branch_models import BranchMessage
from .snapshot_sources import ordered_frozen_message_ids


class ConversationHistory:
    def __init__(self, session, branch_id, snapshot, now):
        self.session, self.branch_id, self.snapshot, self.now = session, branch_id, snapshot, now
        self._reference_cache = None

    def references(self):
        """仅装载排序索引；原文按需读取，不全量装入工具上下文。"""
        if self._reference_cache is not None:
            return self._reference_cache
        allowed = ordered_frozen_message_ids(self.session, self.snapshot) if self.snapshot else []
        imported = (
            list(
                self.session.scalars(
                    select(Message.id)
                    .join(Participant, Participant.id == Message.participant_id)
                    .where(
                        Message.id.in_(allowed),
                        Participant.role.in_(["self", "target"]),
                    )
                    .order_by(Message.timestamp, Message.id)
                )
            )
            if allowed
            else []
        )
        included = set(imported)
        imported = [key for key in allowed if key in included]
        branch = list(
            self.session.scalars(
                select(BranchMessage.id)
                .where(
                    BranchMessage.branch_id == self.branch_id,
                    BranchMessage.generation_status != "failed",
                    func.coalesce(BranchMessage.observed_at, BranchMessage.created_at)
                    <= self.now.astimezone(UTC),
                )
                .order_by(BranchMessage.sequence)
            )
        )
        self._reference_cache = imported + branch, set(imported)
        return self._reference_cache

    def read(self, *, before_ref=None, around_ref=None, limit=40):
        if before_ref and around_ref:
            raise ValueError("before_ref 与 around_ref 不能同时使用")
        refs, imported = self.references()
        anchor = around_ref or before_ref
        if anchor and anchor not in refs:
            raise ValueError("conversation_reference_out_of_scope")
        if around_ref:
            start = max(0, refs.index(around_ref) - limit // 2)
            end = min(len(refs), start + limit)
        else:
            end = refs.index(before_ref) if before_ref else len(refs)
            start = max(0, end - limit)
        selected = refs[start:end]
        messages = []
        for ref in selected:
            if ref in imported:
                row = self.session.get(Message, ref)
                participant = self.session.get(Participant, row.participant_id)
                kind, content = normalize_wechat_display(row.kind, row.content)
                role = "user" if participant.role == "self" else "assistant"
                at, origin = row.timestamp, "import"
            else:
                row = self.session.get(BranchMessage, ref)
                role, content, kind = row.role, row.content, row.type
                at, origin = row.observed_at or row.created_at, "branch"
            messages.append(
                {
                    "source_ref": ref,
                    "role": role,
                    "content": content,
                    "type": kind,
                    "asset_ref": row.media_asset_id,
                    "speaker": role,
                    "usage_refs": (row.generation_metadata or {}).get("sticker_usage_refs", [])
                    if ref not in imported
                    else [ref],
                    "occurred_at": at.isoformat(),
                    "origin": origin,
                }
            )
        return {
            "messages": messages,
            "source_ids": selected,
            "next_before_ref": selected[0] if start and selected else None,
            "has_later_messages": end < len(refs),
        }

    def search(self, query, limit=8):
        from langchain_core.tools import ToolException

        from moonlightbox.agent_runtime.tool_errors import ToolServiceError

        from .conversation_index import search_branch
        from .tools.routine_evidence import query_frozen_history

        branch = search_branch(self.session, self.branch_id, query, self.now, limit, kind="message")
        try:
            history = (
                query_frozen_history(self.session, self.snapshot, query, limit=limit)
                if self.snapshot
                else None
            )
            historical_status = history["retrieval_status"] if history else "unavailable"
        except (ToolException, ToolServiceError):
            history, historical_status = None, "unavailable"
        refs, imported = self.references()
        allowed = set(refs)
        # 两个检索器的分数不可直接比较；按名次融合，随后按真实消息引用去重。
        scores = {}
        lists = [
            [item["source_ref"] for item in branch["data"]],
            [item["source_id"] for item in history["messages"]] if history else [],
        ]
        for candidates in lists:
            for rank, ref in enumerate(dict.fromkeys(candidates), 1):
                if ref in allowed:
                    scores[ref] = scores.get(ref, 0) + 1 / (60 + rank)
        selected = sorted(scores, key=lambda ref: (-scores[ref], ref))[:limit]
        hits = [self.read(around_ref=ref, limit=1)["messages"][0] for ref in selected]
        incomplete = historical_status not in {"ok", "empty"} or branch["status"] not in {
            "ready",
            "empty",
        }
        return {
            "status": "partial" if incomplete else "ready" if hits else "empty",
            "data": hits,
            "source_ids": selected,
            "coverage": {"historical": historical_status, "recent": branch["status"]},
            "selection": "reciprocal_rank_fusion",
        }
