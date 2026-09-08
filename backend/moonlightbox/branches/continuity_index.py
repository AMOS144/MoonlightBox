from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.branches.context import ContextMemory
from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
)
from moonlightbox.branches.embeddings import TextEmbedder
from moonlightbox.branches.memory_index import (
    ChromaProjectMemoryIndex,
    MemoryDocument,
)
from moonlightbox.branches.models import Branch


class BranchContinuityRepository:
    """按分支隔离长期记忆索引，并在返回前执行数据库硬校验。"""

    def __init__(self, chroma_dir: str, embedder: TextEmbedder) -> None:
        self._chroma_dir = chroma_dir
        self._embedder = embedder
        self._fingerprints: dict[str, tuple[int, int, int]] = {}

    def rebuild_branch(self, session: Session, branch_id: str) -> int:
        documents = _episode_documents(session, branch_id)
        documents.extend(_item_documents(session, branch_id))
        self._index(branch_id).replace(documents)
        self._fingerprints[branch_id] = _branch_fingerprint(session, branch_id)
        return len(documents)

    def retrieve(
        self,
        session: Session,
        branch: Branch,
        query_text: str,
        *,
        now: datetime | None = None,
    ) -> tuple[ContextMemory, ...]:
        if self._fingerprints.get(branch.id) != _branch_fingerprint(session, branch.id):
            self.rebuild_branch(session, branch.id)
        cutoff = _aware(now or datetime.now(UTC))
        episodes = {
            episode.id: episode
            for episode in session.scalars(
                select(BranchMemoryEpisode).where(
                    BranchMemoryEpisode.branch_id == branch.id,
                    BranchMemoryEpisode.processing_status == "processed",
                )
            )
            if _aware(episode.ended_at) <= cutoff
        }
        items = {
            item.id: item
            for item in session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                )
            )
            if _aware(item.valid_from) <= cutoff
        }
        index = self._index(branch.id)
        candidates: list[tuple[str, str, str, float, float, datetime, float, float]] = []
        if episodes:
            for resource_id, content, similarity in index.query(
                query_text,
                resource_type="episode",
                allowed_ids=set(episodes),
                cutoff=cutoff,
                limit=10,
            ):
                episode = episodes[resource_id]
                candidates.append(
                    (
                        resource_id,
                        "episode",
                        content,
                        similarity,
                        episode.importance,
                        _aware(episode.ended_at),
                        1.0,
                        0.45,
                    )
                )
        # Active beliefs are injected from the versioned state by the actor.
        # Retrieving every historical belief here would leak inactive or
        # contested alternatives into the public LoRA prompt.
        for kind in ("experience", "fact", "self_narrative", "reflection"):
            allowed = {item_id for item_id, item in items.items() if item.kind == kind}
            for resource_id, content, similarity in index.query(
                query_text,
                resource_type=kind,
                allowed_ids=allowed,
                cutoff=cutoff,
                limit=10,
            ):
                item = items[resource_id]
                candidates.append(
                    (
                        resource_id,
                        kind,
                        content,
                        similarity,
                        item.importance,
                        _aware(item.valid_from),
                        item.confidence,
                        _authority_score(item),
                    )
                )
        candidates.sort(
            key=lambda row: _combined_score(
                semantic_similarity=row[3],
                importance=row[4],
                happened_at=row[5],
                confidence=row[6],
                authority=row[7],
                now=cutoff,
            ),
            reverse=True,
        )
        consolidated = _consolidate_semantic_candidates(
            candidates,
            items=items,
            now=cutoff,
        )
        return tuple(
            ContextMemory(
                resource_id=f"continuity:{branch.id}:{resource_id}",
                resource_type=kind,
                content=_context_content(
                    kind=kind,
                    resource_id=resource_id,
                    content=content,
                    items=items,
                ),
                authority=_memory_authority(kind, resource_id, items),
                evidence_ids=evidence_ids,
            )
            for (
                (resource_id, kind, content, *_rest),
                evidence_ids,
            ) in consolidated
        )

    def _index(self, branch_id: str) -> ChromaProjectMemoryIndex:
        return ChromaProjectMemoryIndex(
            chroma_dir=self._chroma_dir,
            project_id=f"branch_{branch_id}",
            embedder=self._embedder,
        )


def _context_content(
    *,
    kind: str,
    resource_id: str,
    content: str,
    items: dict[str, BranchMemoryItem],
) -> str:
    item = items.get(resource_id)
    if item is not None:
        # The vector document also contains structured retrieval hints. Public
        # generation receives only the reviewed natural-language memory.
        content = item.content
    if kind != "fact":
        return content
    if item is None or item.verification_status in {
        "verified_interaction",
        "verified_history",
    }:
        return content
    return f"[未验证陈述，不得作为客观事实] {content}"


def _memory_authority(
    kind: str,
    resource_id: str,
    items: dict[str, BranchMemoryItem],
) -> Literal[
    "observed_interaction",
    "conversation_record",
    "reported_claim",
    "subjective",
]:
    if kind == "episode":
        return "conversation_record"
    if kind in {"belief", "self_narrative", "reflection"}:
        return "subjective"
    item = items.get(resource_id)
    if item is None:
        return "observed_interaction"
    if kind == "fact" and item.verification_status not in {
        "verified_interaction",
        "verified_history",
    }:
        return "reported_claim"
    if kind == "experience" and item.verification_status != "verified_interaction":
        return (
            "conversation_record"
            if item.verification_status == "self_claimed"
            else "reported_claim"
        )
    return "observed_interaction"


def _memory_evidence_ids(
    kind: str,
    resource_id: str,
    items: dict[str, BranchMemoryItem],
) -> tuple[str, ...]:
    if kind == "episode":
        return (resource_id,)
    item = items.get(resource_id)
    return tuple(item.source_episode_ids) if item is not None else ()


CandidateRow = tuple[str, str, str, float, float, datetime, float, float]


def _consolidate_semantic_candidates(
    candidates: list[CandidateRow],
    *,
    items: dict[str, BranchMemoryItem],
    now: datetime,
) -> list[tuple[CandidateRow, tuple[str, ...]]]:
    """Collapse duplicate claims and resolve exact support/opposition.

    Episodes remain an immutable audit log. Semantic items that express the
    same structured claim share one retrieval slot, and each root episode can
    contribute only once. Opposition is used to resolve the claim; it is never
    exposed to the LoRA as if it were a positive fact.
    """

    episodes: list[tuple[CandidateRow, tuple[str, ...]]] = []
    groups: dict[tuple[str, str, str, str, str], list[CandidateRow]] = {}
    for candidate in candidates:
        resource_id, kind, *_rest = candidate
        item = items.get(resource_id)
        if kind == "episode" or item is None:
            episodes.append((candidate, (resource_id,)))
            continue
        key = _semantic_claim_key(item)
        groups.setdefault(key, []).append(candidate)

    semantic: list[tuple[CandidateRow, tuple[str, ...]]] = []
    for entries in groups.values():
        roots: dict[str, tuple[BranchMemoryItem, CandidateRow, float]] = {}
        for candidate in entries:
            item = items[candidate[0]]
            signal = max(
                0.0,
                min(1.0, item.confidence * _authority_score(item)),
            )
            root_keys = item.root_episode_hashes or item.source_episode_ids or [item.id]
            for root in root_keys:
                previous = roots.get(root)
                if previous is None or signal > previous[2]:
                    roots[root] = (item, candidate, signal)
        support = 0.0
        oppose = 0.0
        supporting: list[tuple[BranchMemoryItem, CandidateRow]] = []
        for item, candidate, signal in roots.values():
            if item.stance == "oppose":
                oppose = 1 - (1 - oppose) * (1 - signal)
            else:
                support = 1 - (1 - support) * (1 - signal)
                supporting.append((item, candidate))
        if not supporting or support <= oppose:
            continue
        _representative_item, representative = max(
            supporting,
            key=lambda entry: _candidate_score(entry[1], now=now),
        )
        net_strength = support * (1 - oppose)
        representative = (
            representative[0],
            representative[1],
            representative[2],
            max(candidate[3] for _item, candidate in supporting),
            max(candidate[4] for _item, candidate in supporting),
            max(candidate[5] for _item, candidate in supporting),
            net_strength,
            max(candidate[7] for _item, candidate in supporting),
        )
        evidence_ids = tuple(
            dict.fromkeys(
                episode_id
                for item, _candidate in supporting
                for episode_id in item.source_episode_ids
            )
        )
        semantic.append((representative, evidence_ids))

    combined = [*episodes, *semantic]
    combined.sort(
        key=lambda entry: _candidate_score(entry[0], now=now),
        reverse=True,
    )
    return combined


def _candidate_score(candidate: CandidateRow, *, now: datetime) -> float:
    return _combined_score(
        semantic_similarity=candidate[3],
        importance=candidate[4],
        happened_at=candidate[5],
        confidence=candidate[6],
        authority=candidate[7],
        now=now,
    )


def _claim_part(value: str) -> str:
    return "".join(value.split()).casefold()


def _semantic_claim_key(
    item: BranchMemoryItem,
) -> tuple[str, str, str, str, str]:
    predicate = _claim_part(item.predicate)
    object_value = _claim_part(item.object)
    generic_predicates = {
        "发送",
        "回复",
        "表达",
        "说",
        "提到",
        "交流",
        "发生",
        "行为",
    }
    generic_objects = {"消息", "内容", "回复", "对话", "交流", "事情"}
    # Weak extractor schemas must not collapse semantically different turns.
    # Exact normalized content still deduplicates literal re-extraction.
    content_discriminator = (
        _claim_part(item.content)
        if predicate in generic_predicates or object_value in generic_objects
        else ""
    )
    return (
        item.kind,
        _claim_part(item.subject),
        predicate,
        object_value,
        content_discriminator,
    )


def _branch_fingerprint(
    session: Session,
    branch_id: str,
) -> tuple[int, int, int]:
    episode_count = session.scalar(
        select(func.count(BranchMemoryEpisode.id)).where(
            BranchMemoryEpisode.branch_id == branch_id,
            BranchMemoryEpisode.processing_status == "processed",
        )
    )
    item_count = session.scalar(
        select(func.count(BranchMemoryItem.id)).where(BranchMemoryItem.branch_id == branch_id)
    )
    active_item_count = session.scalar(
        select(func.count(BranchMemoryItem.id)).where(
            BranchMemoryItem.branch_id == branch_id,
            BranchMemoryItem.review_status == "approved",
            BranchMemoryItem.valid_to.is_(None),
        )
    )
    return int(episode_count or 0), int(item_count or 0), int(active_item_count or 0)


def _episode_documents(
    session: Session,
    branch_id: str,
) -> list[MemoryDocument]:
    return [
        MemoryDocument(
            resource_id=episode.id,
            resource_type="episode",
            content=(
                f"用户：{episode.user_content}\n数字人："
                + " / ".join(
                    str(bubble.get("content", ""))
                    for bubble in episode.assistant_bubbles
                    if bubble.get("content")
                )
            ),
            started_at=_aware(episode.started_at),
            ended_at=_aware(episode.ended_at),
        )
        for episode in session.scalars(
            select(BranchMemoryEpisode).where(
                BranchMemoryEpisode.branch_id == branch_id,
                BranchMemoryEpisode.processing_status == "processed",
            )
        )
    ]


def _item_documents(
    session: Session,
    branch_id: str,
) -> list[MemoryDocument]:
    return [
        MemoryDocument(
            resource_id=item.id,
            resource_type=item.kind,
            content=(
                f"记忆：{item.content}\n主体：{item.subject}\n"
                f"关系：{item.predicate}\n内容：{item.object}"
            ),
            started_at=_aware(item.valid_from),
            ended_at=_aware(item.valid_from),
        )
        for item in session.scalars(
            select(BranchMemoryItem).where(
                BranchMemoryItem.branch_id == branch_id,
                BranchMemoryItem.review_status == "approved",
                BranchMemoryItem.valid_to.is_(None),
            )
        )
    ]


def _combined_score(
    *,
    semantic_similarity: float,
    importance: float,
    happened_at: datetime,
    confidence: float,
    authority: float,
    now: datetime,
) -> float:
    age_days = max(0.0, (now - happened_at).total_seconds() / 86400)
    recency_score = 1 / (1 + age_days / 30)
    return (
        semantic_similarity * 0.40
        + max(0.0, min(1.0, importance / 10)) * 0.15
        + recency_score * 0.12
        + max(0.0, min(1.0, confidence)) * 0.10
        + max(0.0, min(1.0, authority)) * 0.23
    )


def _authority_score(item: BranchMemoryItem) -> float:
    if item.kind == "fact":
        return {
            "verified_history": 1.0,
            "verified_interaction": 1.0,
            "asserted_by_user": 0.25,
            "self_claimed": 0.15,
            "disputed": 0.05,
            "inferred": 0.35,
        }.get(item.verification_status, 0.2)
    return {
        "experience": {
            "verified_interaction": 0.9,
            "self_claimed": 0.3,
            "asserted_by_user": 0.25,
            "inferred": 0.5,
        }.get(item.verification_status, 0.35),
        "belief": 0.75,
        "self_narrative": 0.75,
        "reflection": 0.65,
    }.get(item.kind, 0.5)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
