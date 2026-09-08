import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import heappop, heappush

from moonlightbox.events.v3_reviewer import V3CandidateReview, V3EventCandidate

# 0.60 proved too brittle for real, sparse conversations: a fully supported
# candidate scoring 0.5925 was silently discarded and an otherwise successful
# analysis published an empty timeline. 0.55 still requires multi-signal
# support while avoiding that cliff at the review boundary.
DEFAULT_THRESHOLD = 0.55
DEFAULT_MAXIMUM_NODES = 25
DEFAULT_EVIDENCE_JACCARD_THRESHOLD = 0.5
_MAXIMUM_EVENT_GAP = timedelta(hours=24)
_TRAVEL_INTIMACY_TYPES = frozenset({"travel", "intimacy_increased"})


@dataclass(frozen=True, slots=True)
class V3Scores:
    """V3 的六个评分维度及其加权总分。"""

    event_significance: float
    relationship_impact: float
    evidence_quality: float
    persistence: float
    type_support: float
    model_confidence: float
    total: float

    def __post_init__(self) -> None:
        for field_name in (
            "event_significance",
            "relationship_impact",
            "evidence_quality",
            "persistence",
            "type_support",
            "model_confidence",
            "total",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{field_name} 必须是有限的 0..1 数值")


@dataclass(frozen=True, slots=True)
class V3ReviewSnapshot:
    """可序列化且深度不可变的来源复核快照。"""

    facts_supported: bool
    occurrence_supported: bool
    bilateral_confirmation: bool
    evidence_alignment: float
    persistence: float
    type_support: float
    relationship_impact: float
    event_significance: float
    model_confidence: float
    evidence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class V3RankableCandidate:
    """通过校验、等待 V3 评分的单一来源候选。"""

    candidate: V3EventCandidate
    review: V3CandidateReview
    started_at: datetime
    ended_at: datetime
    valid_follow_up_ids: tuple[str, ...] = ()
    source_lane: str | None = None
    source_candidate_id: str | None = None

    def __post_init__(self) -> None:
        if self.started_at > self.ended_at:
            raise ValueError("started_at 不能晚于 ended_at")
        if self.source_lane == "":
            raise ValueError("source_lane 不能为空")


@dataclass(frozen=True, slots=True)
class V3RankedCandidate:
    """带评分、来源并集与合并时间范围的 V3 候选。"""

    candidate: V3EventCandidate
    review: V3CandidateReview
    started_at: datetime
    ended_at: datetime
    scores: V3Scores
    source_lanes: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    source_scores: tuple[tuple[str, V3Scores], ...] = ()
    source_reviews: tuple[tuple[str, V3ReviewSnapshot], ...] = ()
    global_importance: float | None = None
    global_reason: str | None = None
    display_summary: str | None = None
    summary_model: str | None = None

    def __post_init__(self) -> None:
        if self.global_importance is not None and not 0 <= self.global_importance <= 1:
            raise ValueError("global_importance 必须是 0..1 数值")
        if self.global_reason is not None and not self.global_reason.strip():
            raise ValueError("global_reason 不能为空")
        if self.display_summary is not None and not self.display_summary.strip():
            raise ValueError("display_summary 不能为空")


@dataclass(slots=True)
class _CandidateCluster:
    members: list[V3RankedCandidate]


@dataclass(frozen=True, slots=True)
class _CompressedCandidateGroup:
    representative: V3RankedCandidate
    members: tuple[V3RankedCandidate, ...]


@dataclass(frozen=True, slots=True)
class _PrefixPosting:
    posting_id: int
    cluster_id: int
    evidence: frozenset[str]
    token_position: int
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True, slots=True)
class _SemanticPosting:
    cluster_id: int
    started_at: datetime
    ended_at: datetime


@dataclass(slots=True)
class _ActiveIntervalIndex[
    IntervalKey: Hashable,
    IntervalPosting: (_PrefixPosting, _SemanticPosting),
]:
    active_by_key: dict[IntervalKey, dict[int, IntervalPosting]]
    expiry_heaps: dict[IntervalKey, list[tuple[datetime, int]]]


def score_candidate(candidate: V3RankableCandidate) -> V3RankedCandidate:
    """使用固定六维公式评分，审查证据质量受完整性上限约束。"""

    event_significance = min(
        _unit_score(
            candidate.candidate.event_significance,
            "candidate.event_significance",
        ),
        _unit_score(
            candidate.review.event_significance,
            "review.event_significance",
        ),
    )
    relationship_impact = min(
        _unit_score(
            candidate.candidate.relationship_impact,
            "candidate.relationship_impact",
        ),
        _unit_score(
            candidate.review.relationship_impact,
            "review.relationship_impact",
        ),
    )
    evidence_quality = _evidence_quality(candidate)
    persistence = _unit_score(candidate.review.persistence, "persistence")
    type_support = _unit_score(candidate.review.type_support, "type_support")
    model_confidence = min(
        _unit_score(
            candidate.candidate.model_confidence,
            "candidate.model_confidence",
        ),
        _unit_score(
            candidate.review.model_confidence,
            "review.model_confidence",
        ),
    )
    total = (
        event_significance * 0.25
        + relationship_impact * 0.20
        + evidence_quality * 0.25
        + persistence * 0.10
        + type_support * 0.10
        + model_confidence * 0.10
    )
    scores = V3Scores(
        event_significance=event_significance,
        relationship_impact=relationship_impact,
        evidence_quality=evidence_quality,
        persistence=persistence,
        type_support=type_support,
        model_confidence=model_confidence,
        total=total,
    )
    source_id = candidate.source_candidate_id or candidate.candidate.candidate_key
    review_snapshot = _review_snapshot(candidate.review)
    return V3RankedCandidate(
        candidate=candidate.candidate,
        review=candidate.review,
        started_at=candidate.started_at,
        ended_at=candidate.ended_at,
        scores=scores,
        source_lanes=(candidate.source_lane or candidate.candidate.lane,),
        source_candidate_ids=(source_id,),
        source_scores=((source_id, scores),),
        source_reviews=((source_id, review_snapshot),),
    )


def rank_candidates(
    candidates: Iterable[V3RankableCandidate],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    maximum_nodes: int | None = DEFAULT_MAXIMUM_NODES,
    max_nodes: int | None = None,
) -> list[V3RankedCandidate]:
    """先过滤阈值，再 complete-link 合并，最后按确定性 MMR 选择。"""

    scored = [score_candidate(candidate) for candidate in candidates]
    return rank_scored_candidates(
        scored,
        threshold=threshold,
        maximum_nodes=maximum_nodes,
        max_nodes=max_nodes,
    )


def rank_scored_candidates(
    candidates: Iterable[V3RankedCandidate],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    maximum_nodes: int | None = DEFAULT_MAXIMUM_NODES,
    max_nodes: int | None = None,
) -> list[V3RankedCandidate]:
    """排序已评分候选，不重算或修改评分。"""

    _validate_threshold(threshold)
    limit = max_nodes if max_nodes is not None else maximum_nodes
    _validate_maximum_nodes(limit)
    accepted = [candidate for candidate in candidates if candidate.scores.total >= threshold]
    merged = merge_ranked_candidates(accepted)
    return select_diverse_candidates(merged, maximum_nodes=limit)


def merge_ranked_candidates(
    candidates: Sequence[V3RankedCandidate],
    *,
    evidence_jaccard_threshold: float = DEFAULT_EVIDENCE_JACCARD_THRESHOLD,
) -> list[V3RankedCandidate]:
    """使用有界候选桶与 complete-link 聚类合并同一事件。"""

    _validate_threshold(evidence_jaccard_threshold)
    _validate_timezones(candidates)
    ordered = sorted(candidates, key=_cluster_order_key)
    compressed_groups = _compress_exact_candidates(ordered)
    evidence_frequencies = Counter(
        evidence_id
        for group in compressed_groups
        for evidence_id in set(group.representative.candidate.evidence_ids)
    )
    clusters: list[_CandidateCluster] = []
    prefix_index: _ActiveIntervalIndex[str, _PrefixPosting] = _ActiveIntervalIndex(
        active_by_key={},
        expiry_heaps={},
    )
    semantic_index: _ActiveIntervalIndex[
        tuple[str, ...],
        _SemanticPosting,
    ] = _ActiveIntervalIndex(active_by_key={}, expiry_heaps={})
    global_index: _ActiveIntervalIndex[
        None,
        _SemanticPosting,
    ] = _ActiveIntervalIndex(active_by_key={}, expiry_heaps={})
    for posting_id, group in enumerate(compressed_groups):
        candidate = group.representative
        evidence = set(candidate.candidate.evidence_ids)
        evidence_signature = frozenset(evidence)
        semantic_key = _semantic_key(candidate.candidate)
        possible_cluster_ids = _overlap_candidate_cluster_ids(
            evidence=evidence_signature,
            started_at=candidate.started_at,
            ended_at=candidate.ended_at,
            evidence_jaccard_threshold=evidence_jaccard_threshold,
            prefix_index=prefix_index,
            evidence_frequencies=evidence_frequencies,
        )
        if evidence_jaccard_threshold == 0:
            possible_cluster_ids.update(
                posting.cluster_id
                for posting in _active_interval_postings(
                    global_index,
                    None,
                    candidate.started_at,
                )
            )
        possible_cluster_ids.update(
            posting.cluster_id
            for posting in _active_interval_postings(
                semantic_index,
                semantic_key,
                candidate.started_at,
            )
        )
        chosen_cluster: int | None = None
        for cluster_id in sorted(possible_cluster_ids):
            cluster = clusters[cluster_id]
            if _fits_cluster(
                candidate,
                cluster,
                evidence_jaccard_threshold=evidence_jaccard_threshold,
            ):
                chosen_cluster = cluster_id
                break

        if chosen_cluster is None:
            chosen_cluster = len(clusters)
            clusters.append(_new_cluster(candidate))
        else:
            _append_to_cluster(clusters[chosen_cluster], candidate)
        for equivalent_member in group.members[1:]:
            _append_to_cluster(
                clusters[chosen_cluster],
                equivalent_member,
            )

        semantic_posting = _SemanticPosting(
            cluster_id=chosen_cluster,
            started_at=candidate.started_at,
            ended_at=candidate.ended_at,
        )
        _register_active_interval(
            semantic_index,
            semantic_key,
            posting_id,
            semantic_posting,
        )
        if evidence_jaccard_threshold == 0:
            _register_active_interval(
                global_index,
                None,
                posting_id,
                semantic_posting,
            )
        _index_prefix_tokens(
            evidence=evidence_signature,
            started_at=candidate.started_at,
            ended_at=candidate.ended_at,
            evidence_jaccard_threshold=evidence_jaccard_threshold,
            posting_id=posting_id,
            cluster_id=chosen_cluster,
            prefix_index=prefix_index,
            evidence_frequencies=evidence_frequencies,
        )

    merged = [_merge_cluster(cluster) for cluster in clusters]
    merged.sort(key=_rank_key)
    return merged


def select_diverse_candidates(
    candidates: Sequence[V3RankedCandidate],
    *,
    maximum_nodes: int | None = DEFAULT_MAXIMUM_NODES,
) -> list[V3RankedCandidate]:
    """按固定惩罚选取候选；惩罚只影响次序，不修改总分。"""

    _validate_maximum_nodes(maximum_nodes)
    grouped: dict[
        tuple[str, tuple[str, ...]],
        deque[V3RankedCandidate],
    ] = {}
    raw_groups: defaultdict[
        tuple[str, tuple[str, ...]],
        list[V3RankedCandidate],
    ] = defaultdict(list)
    for candidate in candidates:
        group_key = (candidate.candidate.type, candidate.source_lanes)
        raw_groups[group_key].append(candidate)
    for group_key, members in raw_groups.items():
        grouped[group_key] = deque(sorted(members, key=_rank_key))

    selected: list[V3RankedCandidate] = []
    while grouped and (maximum_nodes is None or len(selected) < maximum_nodes):
        chosen = min(
            (members[0] for members in grouped.values()),
            key=lambda candidate: _mmr_key(candidate, selected),
        )
        selected.append(chosen)
        chosen_group = (chosen.candidate.type, chosen.source_lanes)
        grouped[chosen_group].popleft()
        if not grouped[chosen_group]:
            del grouped[chosen_group]
    return selected


def _evidence_quality(candidate: V3RankableCandidate) -> float:
    """证据质量等于审查对齐度乘以封顶的结构完整性。"""

    alignment = _unit_score(
        candidate.review.evidence_alignment,
        "evidence_alignment",
    )
    evidence_ids = candidate.candidate.evidence_ids
    unique_evidence = set(evidence_ids)
    completeness = 0.0
    if {
        candidate.candidate.start_message_id,
        candidate.candidate.end_message_id,
    }.issubset(unique_evidence):
        completeness += 0.5
    if len(unique_evidence) >= 2 and len(unique_evidence) == len(evidence_ids):
        completeness += 0.25
    review_evidence = set(candidate.review.evidence_ids)
    if review_evidence:
        legal_follow_up_ids = set(candidate.valid_follow_up_ids)
        legal_count = len(review_evidence & legal_follow_up_ids)
        completeness += 0.25 * legal_count / len(review_evidence)
    return alignment * min(1.0, completeness)


def _review_snapshot(review: V3CandidateReview) -> V3ReviewSnapshot:
    return V3ReviewSnapshot(
        facts_supported=review.facts_supported,
        occurrence_supported=review.occurrence_supported,
        bilateral_confirmation=review.bilateral_confirmation,
        evidence_alignment=review.evidence_alignment,
        persistence=review.persistence,
        type_support=review.type_support,
        relationship_impact=review.relationship_impact,
        event_significance=review.event_significance,
        model_confidence=review.model_confidence,
        evidence_ids=tuple(review.evidence_ids),
        reason=review.reason,
    )


def _required_overlap(
    threshold: float,
    left_size: int,
    right_size: int,
) -> int:
    """返回两个集合达到 Jaccard 阈值所需的最小交集。"""

    required = threshold / (1 + threshold) * (left_size + right_size)
    return math.ceil(required - 1e-12)


def _prefix_length(evidence_count: int, threshold: float) -> int:
    """返回标准 Jaccard prefix-filtering 的前缀长度。"""

    if evidence_count < 0:
        raise ValueError("证据数量不能为负数")
    length = evidence_count - math.ceil(threshold * evidence_count) + 1
    return min(evidence_count, max(0, length))


def _overlap_candidate_cluster_ids(
    *,
    evidence: frozenset[str],
    started_at: datetime,
    ended_at: datetime,
    evidence_jaccard_threshold: float,
    prefix_index: _ActiveIntervalIndex[str, _PrefixPosting],
    evidence_frequencies: Counter[str],
) -> set[int]:
    """使用 AllPairs 长度、位置与精确 Jaccard 过滤候选簇。"""

    candidate_cluster_ids: set[int] = set()
    ordered_evidence = sorted(
        evidence,
        key=lambda evidence_id: (
            evidence_frequencies[evidence_id],
            evidence_id,
        ),
    )
    prefix = ordered_evidence[: _prefix_length(len(evidence), evidence_jaccard_threshold)]
    minimum_length = math.ceil(evidence_jaccard_threshold * len(evidence))
    maximum_length = (
        math.floor(len(evidence) / evidence_jaccard_threshold)
        if evidence_jaccard_threshold > 0
        else math.inf
    )
    overlap_counts: dict[int, int] = {}
    postings_by_id: dict[int, _PrefixPosting] = {}
    pruned_postings: set[int] = set()
    seen_hits: set[tuple[int, str]] = set()
    for candidate_position, token in enumerate(prefix):
        for posting in _active_prefix_postings(
            prefix_index,
            token,
            started_at,
        ):
            hit_key = (posting.posting_id, token)
            if hit_key in seen_hits:
                continue
            seen_hits.add(hit_key)
            if not _time_values_are_close(
                started_at,
                ended_at,
                posting.started_at,
                posting.ended_at,
            ):
                continue
            indexed_length = len(posting.evidence)
            if not minimum_length <= indexed_length <= maximum_length:
                continue
            if posting.posting_id in pruned_postings:
                continue
            postings_by_id[posting.posting_id] = posting
            overlap_count = overlap_counts.get(posting.posting_id, 0) + 1
            overlap_counts[posting.posting_id] = overlap_count
            required_overlap = _required_overlap(
                evidence_jaccard_threshold,
                len(evidence),
                indexed_length,
            )
            maximum_possible_overlap = overlap_count + min(
                len(evidence) - candidate_position - 1,
                indexed_length - posting.token_position - 1,
            )
            if maximum_possible_overlap < required_overlap:
                pruned_postings.add(posting.posting_id)

    for posting_id, posting in postings_by_id.items():
        if posting_id in pruned_postings:
            continue
        intersection_count = len(evidence & posting.evidence)
        required_overlap = _required_overlap(
            evidence_jaccard_threshold,
            len(evidence),
            len(posting.evidence),
        )
        if intersection_count < required_overlap:
            continue
        union_count = len(evidence) + len(posting.evidence) - intersection_count
        if (intersection_count / union_count if union_count else 0.0) >= evidence_jaccard_threshold:
            candidate_cluster_ids.add(posting.cluster_id)
    return candidate_cluster_ids


def _index_prefix_tokens(
    *,
    evidence: frozenset[str],
    started_at: datetime,
    ended_at: datetime,
    evidence_jaccard_threshold: float,
    posting_id: int,
    cluster_id: int,
    prefix_index: _ActiveIntervalIndex[str, _PrefixPosting],
    evidence_frequencies: Counter[str],
) -> None:
    """按全局频率顺序只索引标准 Jaccard 前缀 token。"""

    ordered_evidence = sorted(
        evidence,
        key=lambda evidence_id: (
            evidence_frequencies[evidence_id],
            evidence_id,
        ),
    )
    prefix = ordered_evidence[: _prefix_length(len(evidence), evidence_jaccard_threshold)]
    for token_position, token in enumerate(prefix):
        posting = _PrefixPosting(
            posting_id=posting_id,
            cluster_id=cluster_id,
            evidence=evidence,
            token_position=token_position,
            started_at=started_at,
            ended_at=ended_at,
        )
        _register_active_interval(
            prefix_index,
            token,
            posting_id,
            posting,
        )


def _active_prefix_postings(
    prefix_index: _ActiveIntervalIndex[str, _PrefixPosting],
    token: str,
    current_started_at: datetime,
) -> tuple[_PrefixPosting, ...]:
    """查询 token 对应的时间活动 posting。"""

    return _active_interval_postings(
        prefix_index,
        token,
        current_started_at,
    )


def _register_active_interval[
    IntervalKey: Hashable,
    IntervalPosting: (_PrefixPosting, _SemanticPosting),
](
    index: _ActiveIntervalIndex[IntervalKey, IntervalPosting],
    key: IntervalKey,
    posting_id: int,
    posting: IntervalPosting,
) -> None:
    """按稳定处理序号注册区间 posting。"""

    index.active_by_key.setdefault(key, {})[posting_id] = posting
    heappush(
        index.expiry_heaps.setdefault(key, []),
        (posting.ended_at, posting_id),
    )


def _active_interval_postings[
    IntervalKey: Hashable,
    IntervalPosting: (_PrefixPosting, _SemanticPosting),
](
    index: _ActiveIntervalIndex[IntervalKey, IntervalPosting],
    key: IntervalKey,
    current_started_at: datetime,
) -> tuple[IntervalPosting, ...]:
    """淘汰时间窗口外条目并返回稳定活动视图。"""

    active = index.active_by_key.get(key)
    heap = index.expiry_heaps.get(key)
    if not active or not heap:
        return ()
    while heap and _ended_before_active_window(
        heap[0][0],
        current_started_at,
    ):
        _, posting_id = heappop(heap)
        active.pop(posting_id, None)
    return tuple(active.values())


def _ended_before_active_window(
    ended_at: datetime,
    current_started_at: datetime,
) -> bool:
    try:
        return ended_at < current_started_at and current_started_at - ended_at > _MAXIMUM_EVENT_GAP
    except TypeError as error:
        raise ValueError("候选时间时区必须一致") from error


def _compress_exact_candidates(
    ordered: Sequence[V3RankedCandidate],
) -> list[_CompressedCandidateGroup]:
    """仅压缩对聚类关系完全等价的签名、语义与时间候选。"""

    grouped: dict[
        tuple[
            frozenset[str],
            tuple[str, ...],
            str,
            tuple[str, ...],
            datetime,
            datetime,
        ],
        list[V3RankedCandidate],
    ] = {}
    for candidate in ordered:
        key = (
            frozenset(candidate.candidate.evidence_ids),
            candidate.source_lanes,
            candidate.candidate.type,
            _semantic_key(candidate.candidate),
            candidate.started_at,
            candidate.ended_at,
        )
        grouped.setdefault(key, []).append(candidate)
    return [
        _CompressedCandidateGroup(
            representative=members[0],
            members=tuple(members),
        )
        for members in grouped.values()
    ]


def _new_cluster(candidate: V3RankedCandidate) -> _CandidateCluster:
    return _CandidateCluster(members=[candidate])


def _fits_cluster(
    candidate: V3RankedCandidate,
    cluster: _CandidateCluster,
    *,
    evidence_jaccard_threshold: float,
) -> bool:
    return all(
        _same_event(candidate, member, evidence_jaccard_threshold) for member in cluster.members
    )


def _same_event(
    left: V3RankedCandidate,
    right: V3RankedCandidate,
    evidence_jaccard_threshold: float,
) -> bool:
    if not _time_is_close(left, right):
        return False
    left_type = left.candidate.type
    right_type = right.candidate.type
    evidence_similarity = _jaccard(
        set(left.candidate.evidence_ids),
        set(right.candidate.evidence_ids),
    )
    lanes_overlap = bool(set(left.source_lanes) & set(right.source_lanes))
    pair_threshold = _pair_evidence_threshold(
        left,
        right,
        evidence_jaccard_threshold,
    )
    if left_type != right_type:
        if lanes_overlap:
            return evidence_similarity >= pair_threshold and _time_overlap_ratio(left, right) >= 0.8
        if {left_type, right_type} == _TRAVEL_INTIMACY_TYPES:
            return evidence_similarity >= pair_threshold
        return False
    return evidence_similarity >= pair_threshold or _semantic_key(left.candidate) == _semantic_key(
        right.candidate
    )


def _pair_evidence_threshold(
    left: V3RankedCandidate,
    right: V3RankedCandidate,
    configured_threshold: float,
) -> float:
    same_lane = bool(set(left.source_lanes) & set(right.source_lanes))
    if same_lane and left.candidate.type != right.candidate.type:
        return max(configured_threshold, 0.8)
    return configured_threshold


def _append_to_cluster(
    cluster: _CandidateCluster,
    candidate: V3RankedCandidate,
) -> None:
    cluster.members.append(candidate)


def _merge_cluster(cluster: _CandidateCluster) -> V3RankedCandidate:
    representative = min(cluster.members, key=_rank_key)
    ordered_members = sorted(cluster.members, key=_rank_key)
    evidence_ids = list(representative.candidate.evidence_ids)
    seen_evidence = set(evidence_ids)
    review_evidence_ids: list[str] = []
    seen_review_evidence: set[str] = set()
    source_score_map: dict[str, V3Scores] = {}
    source_review_map: dict[str, V3ReviewSnapshot] = {}
    source_weights: dict[str, int] = {}
    for member in ordered_members:
        for evidence_id in member.candidate.evidence_ids:
            if evidence_id not in seen_evidence:
                evidence_ids.append(evidence_id)
                seen_evidence.add(evidence_id)
        for evidence_id in member.review.evidence_ids:
            if evidence_id not in seen_review_evidence:
                review_evidence_ids.append(evidence_id)
                seen_review_evidence.add(evidence_id)
        weight = max(1, len(set(member.candidate.evidence_ids)))
        member_scores = member.source_scores or tuple(
            (source_id, member.scores) for source_id in member.source_candidate_ids
        )
        member_reviews = member.source_reviews or tuple(
            (source_id, _review_snapshot(member.review))
            for source_id in member.source_candidate_ids
        )
        for source_id, source_score in member_scores:
            source_score_map.setdefault(source_id, source_score)
            source_weights.setdefault(source_id, weight)
        for source_id, source_review in member_reviews:
            source_review_map.setdefault(source_id, source_review)

    source_scores = tuple(sorted(source_score_map.items()))
    source_reviews = tuple(sorted(source_review_map.items()))
    conservative_scores = V3Scores(
        event_significance=min(score.event_significance for _, score in source_scores),
        relationship_impact=min(score.relationship_impact for _, score in source_scores),
        evidence_quality=_weighted_evidence_quality(
            source_scores,
            source_weights,
        ),
        persistence=min(score.persistence for _, score in source_scores),
        type_support=min(score.type_support for _, score in source_scores),
        model_confidence=min(score.model_confidence for _, score in source_scores),
        total=0,
    )
    conservative_scores = V3Scores(
        event_significance=conservative_scores.event_significance,
        relationship_impact=conservative_scores.relationship_impact,
        evidence_quality=conservative_scores.evidence_quality,
        persistence=conservative_scores.persistence,
        type_support=conservative_scores.type_support,
        model_confidence=conservative_scores.model_confidence,
        total=_score_total(conservative_scores),
    )
    merged_review = V3CandidateReview(
        facts_supported=all(review.facts_supported for _, review in source_reviews),
        occurrence_supported=all(review.occurrence_supported for _, review in source_reviews),
        bilateral_confirmation=all(review.bilateral_confirmation for _, review in source_reviews),
        evidence_alignment=_weighted_evidence_alignment(
            source_reviews,
            source_weights,
        ),
        persistence=min(review.persistence for _, review in source_reviews),
        type_support=min(review.type_support for _, review in source_reviews),
        relationship_impact=min(review.relationship_impact for _, review in source_reviews),
        event_significance=min(review.event_significance for _, review in source_reviews),
        model_confidence=min(review.model_confidence for _, review in source_reviews),
        evidence_ids=review_evidence_ids,
        reason=representative.review.reason,
    )
    candidate_payload = representative.candidate.model_dump(mode="python")
    candidate_payload["evidence_ids"] = evidence_ids
    candidate_payload["candidate_key"] = "pending"
    candidate = V3EventCandidate.model_validate(candidate_payload)
    return V3RankedCandidate(
        candidate=candidate,
        review=merged_review,
        started_at=min(member.started_at for member in cluster.members),
        ended_at=max(member.ended_at for member in cluster.members),
        scores=conservative_scores,
        source_lanes=tuple(
            sorted({lane for member in cluster.members for lane in member.source_lanes})
        ),
        source_candidate_ids=tuple(
            sorted(
                {
                    source_id
                    for member in cluster.members
                    for source_id in member.source_candidate_ids
                }
            )
        ),
        source_scores=source_scores,
        source_reviews=source_reviews,
    )


def _weighted_evidence_quality(
    source_scores: Sequence[tuple[str, V3Scores]],
    source_weights: dict[str, int],
) -> float:
    total_weight = sum(source_weights[source_id] for source_id, _ in source_scores)
    return (
        sum(
            score.evidence_quality * source_weights[source_id] for source_id, score in source_scores
        )
        / total_weight
    )


def _weighted_evidence_alignment(
    source_reviews: Sequence[tuple[str, V3ReviewSnapshot]],
    source_weights: dict[str, int],
) -> float:
    total_weight = sum(source_weights[source_id] for source_id, _ in source_reviews)
    return (
        sum(
            review.evidence_alignment * source_weights[source_id]
            for source_id, review in source_reviews
        )
        / total_weight
    )


def _score_total(scores: V3Scores) -> float:
    return (
        scores.event_significance * 0.25
        + scores.relationship_impact * 0.20
        + scores.evidence_quality * 0.25
        + scores.persistence * 0.10
        + scores.type_support * 0.10
        + scores.model_confidence * 0.10
    )


def _mmr_key(
    candidate: V3RankedCandidate,
    selected: Sequence[V3RankedCandidate],
) -> tuple[float, float, datetime, str]:
    same_type_count = sum(item.candidate.type == candidate.candidate.type for item in selected)
    candidate_lanes = set(candidate.source_lanes)
    same_lane_count = sum(bool(candidate_lanes & set(item.source_lanes)) for item in selected)
    adjusted = candidate.scores.total - 0.06 * same_type_count - 0.03 * same_lane_count
    return (
        -adjusted,
        -candidate.scores.total,
        candidate.started_at,
        candidate.candidate.candidate_key,
    )


def _time_is_close(
    left: V3RankedCandidate,
    right: V3RankedCandidate,
) -> bool:
    return _time_values_are_close(
        left.started_at,
        left.ended_at,
        right.started_at,
        right.ended_at,
    )


def _time_values_are_close(
    left_started_at: datetime,
    left_ended_at: datetime,
    right_started_at: datetime,
    right_ended_at: datetime,
) -> bool:
    try:
        if left_ended_at < right_started_at:
            return right_started_at - left_ended_at <= _MAXIMUM_EVENT_GAP
        if right_ended_at < left_started_at:
            return left_started_at - right_ended_at <= _MAXIMUM_EVENT_GAP
        return True
    except TypeError as error:
        raise ValueError("候选时间时区必须一致") from error


def _time_overlap_ratio(
    left: V3RankedCandidate,
    right: V3RankedCandidate,
) -> float:
    overlap = (
        min(left.ended_at, right.ended_at) - max(left.started_at, right.started_at)
    ).total_seconds()
    if overlap <= 0:
        return 0.0
    shortest = min(
        (left.ended_at - left.started_at).total_seconds(),
        (right.ended_at - right.started_at).total_seconds(),
    )
    return overlap / shortest if shortest > 0 else 1.0


def _time_bucket_keys(
    started_at: datetime,
    ended_at: datetime,
    *,
    expand: bool = False,
) -> tuple[tuple[int, int, int], ...]:
    """只生成起止端点及相邻月，跨度再长也保持常量空间。"""

    start_index = started_at.year * 12 + started_at.month - 1
    end_index = ended_at.year * 12 + ended_at.month - 1
    endpoint_indices = {start_index, end_index}
    if expand:
        endpoint_indices.update(
            {
                start_index - 1,
                start_index + 1,
                end_index - 1,
                end_index + 1,
            }
        )
    return tuple(
        (
            index // 12,
            index % 12 + 1,
            index,
        )
        for index in sorted(endpoint_indices)
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _semantic_key(candidate: V3EventCandidate) -> tuple[str, ...]:
    return (
        candidate.type,
        _canonical_text(candidate.before_state or ""),
        _canonical_text(candidate.after_state or ""),
        _canonical_text(candidate.topic),
        _canonical_text(candidate.title),
    )


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )
    return re.sub(r"^(?:已经|已)", "", compact, count=1)


def _unit_score(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} 必须是有限的 0..1 数值")
    return float(value)


def _validate_threshold(value: float) -> None:
    _unit_score(value, "threshold")


def _validate_maximum_nodes(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("最大节点数必须是非负整数")


def _validate_timezones(candidates: Sequence[V3RankedCandidate]) -> None:
    if not candidates:
        return
    reference = candidates[0].started_at
    try:
        for candidate in candidates:
            _ = candidate.started_at - reference
            _ = candidate.ended_at - reference
    except TypeError as error:
        raise ValueError("候选时间时区必须一致") from error


def _rank_key(
    candidate: V3RankedCandidate,
) -> tuple[float, datetime, str]:
    return (
        -candidate.scores.total,
        candidate.started_at,
        candidate.candidate.candidate_key,
    )


def _cluster_order_key(
    candidate: V3RankedCandidate,
) -> tuple[datetime, str, float]:
    return (
        candidate.started_at,
        candidate.candidate.candidate_key,
        -candidate.scores.total,
    )
