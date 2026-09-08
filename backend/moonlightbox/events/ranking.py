import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from moonlightbox.events.reviewer import EventCandidate, EventCandidateReview

DEFAULT_THRESHOLD = 0.72
DEFAULT_MAXIMUM_NODES = 25
DEFAULT_EVIDENCE_JACCARD_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class RankableCandidate:
    """已通过两阶段校验、可参与全局评分的候选。"""

    candidate: EventCandidate
    review: EventCandidateReview
    started_at: datetime
    ended_at: datetime
    valid_follow_up_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """携带可解释分项及总分的候选。"""

    candidate: EventCandidate
    review: EventCandidateReview
    started_at: datetime
    ended_at: datetime
    valid_follow_up_ids: tuple[str, ...]
    scores: dict[str, float]
    source_candidate_id: str | None = None


@dataclass(slots=True)
class _CandidateCluster:
    members: list[RankedCandidate]
    semantic_key: tuple[str, ...] | None
    evidence_key: frozenset[str] | None
    maximum_start: datetime
    minimum_end: datetime


def score_candidate(candidate: RankableCandidate) -> RankedCandidate:
    """按固定权重计算总分，不对参与阈值判断的值做四舍五入。"""

    evidence_quality = _evidence_quality(candidate)
    scores = {
        "state_change_strength": candidate.candidate.state_change_strength,
        "persistence": candidate.review.persistence,
        "evidence_quality": evidence_quality,
        "model_confidence": candidate.candidate.model_confidence,
    }
    scores["total"] = (
        scores["state_change_strength"] * 0.35
        + scores["persistence"] * 0.30
        + scores["evidence_quality"] * 0.20
        + scores["model_confidence"] * 0.15
    )
    return RankedCandidate(
        candidate=candidate.candidate,
        review=candidate.review,
        started_at=candidate.started_at,
        ended_at=candidate.ended_at,
        valid_follow_up_ids=candidate.valid_follow_up_ids,
        scores=scores,
        source_candidate_id=None,
    )


def rank_candidates(
    candidates: Iterable[RankableCandidate],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    maximum_nodes: int | None = DEFAULT_MAXIMUM_NODES,
) -> list[RankedCandidate]:
    """过滤低分候选并稳定排序；候选不足时不会补齐。"""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 0 到 1 之间")
    if maximum_nodes is not None and maximum_nodes < 0:
        raise ValueError("maximum_nodes 不能为负数")
    accepted = [
        scored
        for candidate in candidates
        if (scored := score_candidate(candidate)).scores["total"] >= threshold
    ]
    accepted.sort(key=_rank_key)
    return accepted if maximum_nodes is None else accepted[:maximum_nodes]


def rank_scored_candidates(
    candidates: Iterable[RankedCandidate],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    maximum_nodes: int | None = DEFAULT_MAXIMUM_NODES,
) -> list[RankedCandidate]:
    """排序持久化分数，不重新计算或修改任何评分分项。"""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 0 到 1 之间")
    if maximum_nodes is not None and maximum_nodes < 0:
        raise ValueError("maximum_nodes 不能为负数")
    accepted = [candidate for candidate in candidates if candidate.scores["total"] >= threshold]
    accepted.sort(key=_rank_key)
    return accepted if maximum_nodes is None else accepted[:maximum_nodes]


def merge_ranked_candidates(
    candidates: Sequence[RankedCandidate],
    *,
    evidence_jaccard_threshold: float = DEFAULT_EVIDENCE_JACCARD_THRESHOLD,
    evidence_order: Mapping[str, int] | None = None,
) -> list[RankedCandidate]:
    """使用 complete-link 归并，避免相似链把不相似端点误合。"""

    if not 0 <= evidence_jaccard_threshold <= 1:
        raise ValueError("证据 Jaccard 阈值必须在 0 到 1 之间")
    ordered = sorted(candidates, key=_rank_key)
    clusters: list[_CandidateCluster] = []
    semantic_buckets: dict[tuple[str, ...], dict[int, None]] = {}
    evidence_buckets: dict[tuple[str, str], dict[int, None]] = {}
    month_buckets: dict[tuple[str, tuple[int, int]], dict[int, None]] = {}
    for candidate in ordered:
        semantic_key = _semantic_key(candidate.candidate)
        partitions = _time_partitions(candidate)
        candidate_cluster_ids = {
            cluster_id
            for cluster_id in semantic_buckets.get(semantic_key, {})
            if _cluster_is_in_months(
                cluster_id,
                event_type=candidate.candidate.type,
                partitions=partitions,
                month_buckets=month_buckets,
            )
        }
        evidence_overlap_counts: dict[int, int] = {}
        for evidence_id in sorted(set(candidate.candidate.evidence_ids)):
            for cluster_id in evidence_buckets.get(
                (candidate.candidate.type, evidence_id),
                {},
            ):
                if not _cluster_is_in_months(
                    cluster_id,
                    event_type=candidate.candidate.type,
                    partitions=partitions,
                    month_buckets=month_buckets,
                ):
                    continue
                evidence_overlap_counts[cluster_id] = evidence_overlap_counts.get(cluster_id, 0) + 1
        candidate_evidence_count = len(set(candidate.candidate.evidence_ids))
        for cluster_id, overlap_count in evidence_overlap_counts.items():
            representative_evidence_count = len(
                set(clusters[cluster_id].members[0].candidate.evidence_ids)
            )
            union_count = candidate_evidence_count + representative_evidence_count - overlap_count
            if union_count > 0 and overlap_count / union_count >= evidence_jaccard_threshold:
                candidate_cluster_ids.add(cluster_id)
        chosen_cluster: int | None = None
        for cluster_id in sorted(candidate_cluster_ids):
            cluster = clusters[cluster_id]
            if _fits_cluster(
                candidate,
                cluster,
                semantic_key=semantic_key,
                evidence_jaccard_threshold=evidence_jaccard_threshold,
            ):
                chosen_cluster = cluster_id
                break
        if chosen_cluster is None:
            chosen_cluster = len(clusters)
            clusters.append(
                _CandidateCluster(
                    members=[candidate],
                    semantic_key=semantic_key,
                    evidence_key=frozenset(candidate.candidate.evidence_ids),
                    maximum_start=candidate.started_at,
                    minimum_end=candidate.ended_at,
                )
            )
        else:
            _append_to_cluster(clusters[chosen_cluster], candidate, semantic_key)
        semantic_buckets.setdefault(semantic_key, {})[chosen_cluster] = None
        for evidence_id in set(candidate.candidate.evidence_ids):
            evidence_buckets.setdefault(
                (candidate.candidate.type, evidence_id),
                {},
            )[chosen_cluster] = None
        for partition in partitions:
            month_buckets.setdefault(
                (candidate.candidate.type, partition),
                {},
            )[chosen_cluster] = None

    # 合并结果必须完全保留最高分代表，尤其不能扩大证据或时间边界。
    _ = evidence_order
    merged = [cluster.members[0] for cluster in clusters]
    merged.sort(key=_rank_key)
    return merged


def _evidence_quality(candidate: RankableCandidate) -> float:
    """由事件边界完整性、证据去重和合法后续证据组成确定性分数。"""

    evidence = candidate.candidate.evidence_ids
    unique_evidence = set(evidence)
    quality = 0.0
    if candidate.candidate.start_message_id in unique_evidence:
        quality += 0.2
    if candidate.candidate.end_message_id in unique_evidence:
        quality += 0.2
    if len(unique_evidence) >= 2 and len(unique_evidence) == len(evidence):
        quality += 0.2
    review_evidence = candidate.review.evidence_ids
    if review_evidence:
        legal = sum(
            message_id in set(candidate.valid_follow_up_ids) for message_id in review_evidence
        )
        quality += 0.4 * legal / len(review_evidence)
    return round(min(1.0, quality), 12)


def _same_event(
    left: RankedCandidate,
    right: RankedCandidate,
    evidence_jaccard_threshold: float,
) -> bool:
    if left.candidate.type != right.candidate.type:
        return False
    if not _ranges_overlap(left, right):
        return False
    return (
        _semantic_key(left.candidate) == _semantic_key(right.candidate)
        or _jaccard(
            set(left.candidate.evidence_ids),
            set(right.candidate.evidence_ids),
        )
        >= evidence_jaccard_threshold
    )


def _ranges_overlap(left: RankedCandidate, right: RankedCandidate) -> bool:
    return left.started_at <= right.ended_at and right.started_at <= left.ended_at


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _time_partitions(candidate: RankedCandidate) -> tuple[tuple[int, int], ...]:
    """返回事件跨度覆盖的全部月份，不截断长事件。"""

    start_index = candidate.started_at.year * 12 + candidate.started_at.month - 1
    end_index = candidate.ended_at.year * 12 + candidate.ended_at.month - 1
    if end_index < start_index:
        start_index, end_index = end_index, start_index
    return tuple((index // 12, index % 12 + 1) for index in range(start_index, end_index + 1))


def _cluster_is_in_months(
    cluster_id: int,
    *,
    event_type: str,
    partitions: Sequence[tuple[int, int]],
    month_buckets: Mapping[
        tuple[str, tuple[int, int]],
        Mapping[int, None],
    ],
) -> bool:
    return any(
        cluster_id in month_buckets.get((event_type, partition), {}) for partition in partitions
    )


def _fits_cluster(
    candidate: RankedCandidate,
    cluster: _CandidateCluster,
    *,
    semantic_key: tuple[str, ...],
    evidence_jaccard_threshold: float,
) -> bool:
    if cluster.semantic_key == semantic_key:
        return (
            candidate.started_at <= cluster.minimum_end
            and candidate.ended_at >= cluster.maximum_start
        )
    if cluster.evidence_key == frozenset(candidate.candidate.evidence_ids):
        return (
            candidate.started_at <= cluster.minimum_end
            and candidate.ended_at >= cluster.maximum_start
        )
    return all(
        _same_event(candidate, member, evidence_jaccard_threshold) for member in cluster.members
    )


def _append_to_cluster(
    cluster: _CandidateCluster,
    candidate: RankedCandidate,
    semantic_key: tuple[str, ...],
) -> None:
    cluster.members.append(candidate)
    if cluster.semantic_key != semantic_key:
        cluster.semantic_key = None
    candidate_evidence_key = frozenset(candidate.candidate.evidence_ids)
    if cluster.evidence_key != candidate_evidence_key:
        cluster.evidence_key = None
    cluster.maximum_start = max(cluster.maximum_start, candidate.started_at)
    cluster.minimum_end = min(cluster.minimum_end, candidate.ended_at)


def _semantic_key(candidate: EventCandidate) -> tuple[str, ...]:
    return (
        candidate.type,
        _canonical_text(candidate.before_state),
        _canonical_text(candidate.after_state),
        _canonical_text(candidate.topic),
    )


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )
    return re.sub(r"^(?:已经|已)", "", compact, count=1)


def _rank_key(candidate: RankedCandidate) -> tuple[float, datetime, str]:
    return (
        -candidate.scores["total"],
        candidate.started_at,
        candidate.candidate.candidate_key,
    )
