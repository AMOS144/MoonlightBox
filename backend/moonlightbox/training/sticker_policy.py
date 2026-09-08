import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, Protocol

from sqlalchemy import and_, or_, select
from sqlalchemy.sql.elements import ColumnElement

from moonlightbox.imports.models import Message, Participant
from moonlightbox.media.models import MediaAsset

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


STICKER_POLICY_VERSION = "context-only-sticker-v2"


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class SemanticContext:
    text_hash: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class StickerHistoryItem:
    message_id: str
    project_id: str
    target_id: str
    timestamp: datetime
    kind: str
    text: str = ""
    asset_id: str | None = None
    asset_project_id: str | None = None
    asset_kind: str | None = None
    is_target: bool = True
    sender: str = ""


@dataclass(frozen=True)
class StickerAssetProfile:
    asset_id: str
    usage_count: int
    latest_used_at: str
    feature_counts: dict[str, int]
    contexts: tuple[dict[str, int], ...] = ()
    semantic_contexts: tuple[SemanticContext, ...] = ()


@dataclass(frozen=True)
class StickerPolicy:
    enabled: bool = False
    version: str = STICKER_POLICY_VERSION
    project_id: str = ""
    target_id: str = ""
    boundary: dict[str, str] = field(default_factory=dict)
    asset_summary: dict[str, int] = field(default_factory=dict)
    parameters: dict[str, float | int | str] = field(default_factory=dict)
    assets: tuple[StickerAssetProfile, ...] = ()
    positive_contexts: tuple[dict[str, int], ...] = ()
    negative_contexts: tuple[dict[str, int], ...] = ()
    positive_semantic_contexts: tuple[SemanticContext, ...] = ()
    negative_semantic_contexts: tuple[SemanticContext, ...] = ()
    _embedder: TextEmbedder | None = field(default=None, repr=False, compare=False)

    @property
    def positive_context_count(self) -> int:
        return len(self.positive_contexts)

    @property
    def negative_context_count(self) -> int:
        return len(self.negative_contexts)

    def to_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "version": self.version,
            "project_id": self.project_id,
            "target_id": self.target_id,
            "boundary": dict(self.boundary),
            "asset_summary": dict(self.asset_summary),
            "parameters": dict(self.parameters),
            "assets": [asdict(asset) for asset in self.assets],
            "positive_contexts": list(self.positive_contexts),
            "negative_contexts": list(self.negative_contexts),
            "positive_semantic_contexts": [
                asdict(item) for item in self.positive_semantic_contexts
            ],
            "negative_semantic_contexts": [
                asdict(item) for item in self.negative_semantic_contexts
            ],
        }

    @classmethod
    def from_metadata(cls, payload: object) -> "StickerPolicy":
        if not isinstance(payload, dict):
            return cls()
        if (
            payload.get("version") != STICKER_POLICY_VERSION
            or payload.get("enabled") is not True
        ):
            return cls()
        raw_assets = payload.get("assets")
        assets: list[StickerAssetProfile] = []
        if isinstance(raw_assets, list):
            for raw in raw_assets:
                if not isinstance(raw, dict):
                    continue
                asset_id = raw.get("asset_id")
                usage_count = raw.get("usage_count")
                latest_used_at = raw.get("latest_used_at")
                feature_counts = raw.get("feature_counts")
                if (
                    isinstance(asset_id, str)
                    and isinstance(usage_count, int)
                    and isinstance(latest_used_at, str)
                    and isinstance(feature_counts, dict)
                ):
                    assets.append(
                        StickerAssetProfile(
                            asset_id=asset_id,
                            usage_count=usage_count,
                            latest_used_at=latest_used_at,
                            feature_counts={
                                str(key): int(value)
                                for key, value in feature_counts.items()
                                if isinstance(value, int)
                            },
                            contexts=_feature_documents(raw.get("contexts")),
                            semantic_contexts=_semantic_documents(
                                raw.get("semantic_contexts")
                            ),
                        )
                    )
        return cls(
            enabled=True,
            project_id=str(payload.get("project_id", "")),
            target_id=str(payload.get("target_id", "")),
            boundary=_string_dict(payload.get("boundary")),
            asset_summary=_int_dict(payload.get("asset_summary")),
            parameters=_number_dict(payload.get("parameters")),
            assets=tuple(assets),
            positive_contexts=_feature_documents(payload.get("positive_contexts")),
            negative_contexts=_feature_documents(payload.get("negative_contexts")),
            positive_semantic_contexts=_semantic_documents(
                payload.get("positive_semantic_contexts")
            ),
            negative_semantic_contexts=_semantic_documents(
                payload.get("negative_semantic_contexts")
            ),
        )


@dataclass(frozen=True)
class RecentStickerBubble:
    kind: str
    content: str = ""
    asset_id: str | None = None
    timestamp: datetime | None = None
    sender: str = ""


@dataclass(frozen=True)
class StickerContext:
    current_text: str
    recent: tuple[RecentStickerBubble, ...]
    now: datetime | None = None
    pragmatic: str | None = None
    relationship: str | None = None
    emotion: str | None = None


@dataclass(frozen=True)
class StickerEvaluationCase:
    split: str
    context: StickerContext
    actual_asset_id: str | None


@dataclass(frozen=True)
class RankedSticker:
    asset_id: str
    score: float
    reasons: tuple[str, ...]


def build_sticker_evaluation_cases(
    items: list[StickerHistoryItem],
    *,
    project_id: str,
    target_id: str,
    start: datetime,
    end: datetime,
    split: Literal["valid", "test"],
) -> tuple[StickerEvaluationCase, ...]:
    """Build held-out modality labels from real messages, not text-only SFT rows."""

    start_at = _aware(start)
    end_at = _aware(end)
    ordered = sorted(
        (
            item
            for item in items
            if item.project_id == project_id
            and item.target_id == target_id
            and _aware(item.timestamp) <= end_at
        ),
        key=lambda item: (_aware(item.timestamp), item.message_id),
    )
    cases: list[StickerEvaluationCase] = []
    for index, item in enumerate(ordered):
        timestamp = _aware(item.timestamp)
        if not item.is_target or timestamp < start_at or timestamp > end_at:
            continue
        actual_asset_id: str | None
        if _is_verified_sticker(item, project_id):
            actual_asset_id = item.asset_id
        elif item.kind == "text":
            actual_asset_id = None
        else:
            continue
        history = ordered[max(0, index - 8) : index]
        current_text = next(
            (
                previous.text
                for previous in reversed(history)
                if not previous.is_target
                and previous.kind == "text"
                and previous.text.strip()
            ),
            "",
        )
        cases.append(
            StickerEvaluationCase(
                split=split,
                context=StickerContext(
                    current_text=current_text,
                    recent=tuple(
                        RecentStickerBubble(
                            kind=previous.kind,
                            content=previous.text,
                            asset_id=previous.asset_id,
                            timestamp=previous.timestamp,
                            sender="target" if previous.is_target else "other",
                        )
                        for previous in history
                    ),
                    now=item.timestamp,
                ),
                actual_asset_id=actual_asset_id,
            )
        )
    return tuple(cases)


def build_sticker_policy(
    items: list[StickerHistoryItem],
    *,
    project_id: str,
    target_id: str,
    cutoff: datetime,
    branch_time: datetime,
    top_k: int = 5,
    minimum_context_hits: int = 1,
    modality_threshold: float = 0.5,
    repetition_penalty: float = 5.0,
    embedder: TextEmbedder | None = None,
) -> StickerPolicy:
    """仅从目标对象的真实历史记录构建策略，不读取或分析资产内容。"""

    effective_cutoff = min(_aware(cutoff), _aware(branch_time))
    scoped = sorted(
        (
            item
            for item in items
            if item.project_id == project_id
            and item.target_id == target_id
            and _aware(item.timestamp) <= effective_cutoff
        ),
        key=lambda item: (_aware(item.timestamp), item.message_id),
    )
    feature_counts_by_asset: dict[str, Counter[str]] = {}
    contexts_by_asset: dict[str, list[dict[str, int]]] = {}
    usage_counts: Counter[str] = Counter()
    latest: dict[str, datetime] = {}
    positive_contexts: list[dict[str, int]] = []
    negative_contexts: list[dict[str, int]] = []
    positive_semantic_rows: list[tuple[str, str]] = []
    negative_semantic_rows: list[str] = []
    for index, item in enumerate(scoped):
        history = scoped[max(0, index - 4) : index]
        features = _history_features(history)
        document = {feature: 1 for feature in features}
        semantic_text = _semantic_history_text(history)
        if item.is_target and item.kind == "text" and document:
            negative_contexts.append(document)
            if semantic_text:
                negative_semantic_rows.append(semantic_text)
        if not item.is_target or not _is_verified_sticker(item, project_id):
            continue
        asset_id = item.asset_id
        if asset_id is None:
            continue
        if document:
            positive_contexts.append(document)
            contexts_by_asset.setdefault(asset_id, []).append(document)
            if semantic_text:
                positive_semantic_rows.append((asset_id, semantic_text))
        usage_counts[asset_id] += 1
        latest[asset_id] = max(latest.get(asset_id, item.timestamp), item.timestamp)
        feature_counts_by_asset.setdefault(asset_id, Counter()).update(features)
    semantic_by_asset: dict[str, list[SemanticContext]] = {}
    positive_semantic: list[SemanticContext] = []
    negative_semantic: list[SemanticContext] = []
    semantic_status = "unavailable"
    semantic_failure = "not_configured"
    runtime_embedder: TextEmbedder | None = None
    semantic_history_top_n = 64
    asset_semantic_top_n = 8
    all_semantic_rows = [
        row
        for asset_id in sorted(usage_counts)
        for row in [
            item
            for item in positive_semantic_rows
            if item[0] == asset_id
        ][-asset_semantic_top_n:]
    ]
    negative_capacity = (
        min(len(negative_semantic_rows), semantic_history_top_n // 4)
        if negative_semantic_rows
        else 0
    )
    retained_semantic_rows = all_semantic_rows[
        -(semantic_history_top_n - negative_capacity) :
    ]
    retained_negative_rows = (
        negative_semantic_rows[-negative_capacity:]
        if negative_capacity
        else []
    )
    semantic_inputs = [
        text for _asset_id, text in retained_semantic_rows
    ] + retained_negative_rows
    if embedder is not None and semantic_inputs:
        try:
            vectors = embedder.embed(semantic_inputs)
            if len(vectors) != len(semantic_inputs):
                raise ValueError("语义向量数量不匹配")
            documents = [
                _semantic_document(text, vector)
                for text, vector in zip(semantic_inputs, vectors, strict=True)
            ]
        except Exception as error:
            semantic_failure = type(error).__name__
        else:
            positive_semantic = documents[: len(retained_semantic_rows)]
            negative_semantic = documents[len(retained_semantic_rows) :]
            for (asset_id, _text), semantic_document in zip(
                retained_semantic_rows,
                positive_semantic,
                strict=True,
            ):
                semantic_by_asset.setdefault(asset_id, []).append(
                    semantic_document
                )
            semantic_status = "available"
            semantic_failure = ""
            runtime_embedder = embedder
    assets = tuple(
        StickerAssetProfile(
            asset_id=asset_id,
            usage_count=usage_counts[asset_id],
            latest_used_at=_aware(latest[asset_id]).isoformat(),
            feature_counts=dict(sorted(feature_counts_by_asset[asset_id].items())),
            contexts=tuple(contexts_by_asset.get(asset_id, [])),
            semantic_contexts=tuple(
                semantic_by_asset.get(asset_id, [])[-asset_semantic_top_n:]
            ),
        )
        for asset_id in sorted(usage_counts)
    )
    return StickerPolicy(
        enabled=bool(assets),
        project_id=project_id,
        target_id=target_id,
        boundary={
            "cutoff": _aware(cutoff).isoformat(),
            "branch_time": _aware(branch_time).isoformat(),
            "effective_cutoff": effective_cutoff.isoformat(),
        },
        asset_summary={
            "linked_sticker_count": len(assets),
            "sticker_event_count": sum(usage_counts.values()),
        },
        parameters={
            "top_k": top_k,
            "minimum_context_hits": minimum_context_hits,
            "modality_threshold": modality_threshold,
            "repetition_penalty": repetition_penalty,
            "ngram_min": 2,
            "ngram_max": 3,
            "similarity": "char-2-3gram-tfidf-cosine",
            "modality_k": 3,
            "asset_context_top_n": 3,
            "semantic_context_top_n": asset_semantic_top_n,
            "semantic_history_top_n": semantic_history_top_n,
            "semantic_weight": 0.65,
            "char_weight": 0.35,
            "semantic_status": semantic_status,
            "semantic_failure": semantic_failure,
        },
        assets=assets,
        positive_contexts=tuple(positive_contexts),
        negative_contexts=tuple(negative_contexts),
        positive_semantic_contexts=tuple(
            positive_semantic[-semantic_history_top_n:]
        ),
        negative_semantic_contexts=tuple(
            negative_semantic[-semantic_history_top_n:]
        ),
        _embedder=runtime_embedder,
    )


def build_sticker_policy_from_database(
    session: "Session",
    *,
    project_id: str,
    target_id: str,
    cutoff: datetime,
    branch_time: datetime,
    import_id: str | None = None,
    boundary_source_id: str | None = None,
    boundary_message_id: str | None = None,
    parameters: dict[str, float | int | str] | None = None,
    embedder: TextEmbedder | None = None,
) -> StickerPolicy:
    """按项目、对象、导入记录和稳定边界读取已关联资产。"""

    effective_cutoff = min(_aware(cutoff), _aware(branch_time))
    conditions: list[ColumnElement[bool]] = [
        Message.project_id == project_id,
        Message.timestamp <= effective_cutoff,
        Participant.role.in_(("self", "target")),
    ]
    if import_id is not None:
        conditions.append(Message.import_id == import_id)
    if boundary_source_id is not None and boundary_message_id is not None:
        conditions.append(
            or_(
                Message.timestamp < _aware(cutoff),
                and_(
                    Message.timestamp == _aware(cutoff),
                    Message.source_id < boundary_source_id,
                ),
                and_(
                    Message.timestamp == _aware(cutoff),
                    Message.source_id == boundary_source_id,
                    Message.id <= boundary_message_id,
                ),
            )
        )
    rows = session.execute(
        select(Message, Participant)
        .join(Participant, Participant.id == Message.participant_id)
        .where(*conditions)
        .order_by(Message.timestamp, Message.source_id, Message.id)
    )
    items: list[StickerHistoryItem] = []
    for message, participant in rows:
        asset = (
            session.get(MediaAsset, message.media_asset_id)
            if message.media_asset_id is not None
            else None
        )
        items.append(
            StickerHistoryItem(
                message_id=message.id,
                project_id=message.project_id,
                target_id=target_id,
                timestamp=message.timestamp,
                kind=message.kind,
                text=message.content,
                asset_id=message.media_asset_id,
                asset_project_id=asset.project_id if asset is not None else None,
                asset_kind=asset.kind if asset is not None else None,
                is_target=participant.id == target_id,
                sender=participant.name,
            )
        )
    resolved = parameters or {}
    policy = build_sticker_policy(
        items,
        project_id=project_id,
        target_id=target_id,
        cutoff=cutoff,
        branch_time=branch_time,
        top_k=int(resolved.get("top_k", 5)),
        minimum_context_hits=int(resolved.get("minimum_context_hits", 1)),
        modality_threshold=float(resolved.get("modality_threshold", 0.5)),
        repetition_penalty=float(resolved.get("repetition_penalty", 5.0)),
        embedder=embedder,
    )
    return replace(
        policy,
        parameters={
            **policy.parameters,
            **resolved,
            "semantic_status": policy.parameters["semantic_status"],
            "semantic_failure": policy.parameters["semantic_failure"],
        },
    )


def build_sticker_evaluation_cases_from_database(
    session: "Session",
    *,
    project_id: str,
    target_id: str,
    start: datetime,
    end: datetime,
    split: Literal["valid", "test"],
    import_id: str | None = None,
) -> tuple[StickerEvaluationCase, ...]:
    """Read real held-out text/sticker outcomes for modality calibration."""

    conditions: list[ColumnElement[bool]] = [
        Message.project_id == project_id,
        Message.timestamp <= _aware(end),
        Participant.role.in_(("self", "target")),
    ]
    if import_id is not None:
        conditions.append(Message.import_id == import_id)
    rows = session.execute(
        select(Message, Participant)
        .join(Participant, Participant.id == Message.participant_id)
        .where(*conditions)
        .order_by(Message.timestamp, Message.source_id, Message.id)
    )
    items: list[StickerHistoryItem] = []
    for message, participant in rows:
        asset = (
            session.get(MediaAsset, message.media_asset_id)
            if message.media_asset_id is not None
            else None
        )
        items.append(
            StickerHistoryItem(
                message_id=message.id,
                project_id=message.project_id,
                target_id=target_id,
                timestamp=message.timestamp,
                kind=message.kind,
                text=message.content,
                asset_id=message.media_asset_id,
                asset_project_id=asset.project_id if asset is not None else None,
                asset_kind=asset.kind if asset is not None else None,
                is_target=participant.id == target_id,
                sender=participant.name,
            )
        )
    return build_sticker_evaluation_cases(
        items,
        project_id=project_id,
        target_id=target_id,
        start=start,
        end=end,
        split=split,
    )


def rank_stickers(
    policy: StickerPolicy,
    context: StickerContext,
    *,
    top_k: int | None = None,
) -> tuple[RankedSticker, ...]:
    """只返回与当前上下文相关的资产候选，不决定是否发送 sticker。"""

    if not policy.enabled or not policy.assets:
        return ()
    features = {feature: 1 for feature in _context_features(context)}
    idf = _history_idf(policy)
    semantic_query = _semantic_query(policy, context)
    minimum_similarity = (
        0.0
        if _explicit_sticker_request(context.current_text)
        else float(policy.parameters.get("minimum_similarity", 0.12))
    )
    total_usage = sum(asset.usage_count for asset in policy.assets)
    repeated_ids = Counter(
        item.asset_id
        for item in context.recent
        if item.kind == "sticker" and item.asset_id is not None
    )
    repetition_penalty = float(policy.parameters.get("repetition_penalty", 5.0))
    cutoff_text = policy.boundary.get("effective_cutoff")
    cutoff = _parse_datetime(cutoff_text)
    scored: list[RankedSticker] = []
    for asset in policy.assets:
        top_n = int(policy.parameters.get("asset_context_top_n", 3))
        contexts = asset.contexts or (asset.feature_counts,)
        char_similarity = _top_n_mean(
            (
                _tfidf_cosine_similarity(features, document, idf)
                for document in contexts
            ),
            top_n,
        )
        semantic_similarity = _top_n_mean(
            (
                _cosine_similarity(semantic_query, document.vector)
                for document in asset.semantic_contexts
            ),
            top_n,
        ) if semantic_query is not None else 0.0
        similarity = _combined_similarity(
            policy,
            char_similarity,
            semantic_similarity,
        )
        if similarity < minimum_similarity:
            continue
        prior = asset.usage_count / max(1, total_usage)
        latest = _parse_datetime(asset.latest_used_at)
        age_days = max(0.0, (cutoff - latest).total_seconds() / 86_400)
        recency = math.exp(-age_days / 180.0)
        repeat_count = repeated_ids[asset.asset_id]
        score = (
            similarity
            + 0.25 * prior
            + 0.05 * recency
            - repetition_penalty * repeat_count
        )
        reasons = [
            (
                f"语义余弦 {semantic_similarity:.4f}，"
                f"字符 TF-IDF {char_similarity:.4f}"
                if semantic_query is not None
                else f"上下文 TF-IDF 相似度 {char_similarity:.4f}"
            ),
            f"个人先验 {asset.usage_count}/{total_usage}",
            f"近期性 {recency:.4f}",
        ]
        if repeat_count:
            reasons.append(f"连续重复惩罚 {repeat_count} 次")
        scored.append(RankedSticker(asset.asset_id, round(score, 8), tuple(reasons)))
    limit = top_k if top_k is not None else int(policy.parameters.get("top_k", 5))
    return tuple(
        sorted(scored, key=lambda item: (-item.score, item.asset_id))[: max(0, limit)]
    )


def sticker_modality_score(policy: StickerPolicy, context: StickerContext) -> float:
    """Estimate whether this person would use a sticker in the current turn."""

    if not policy.enabled or not policy.assets:
        return 0.0
    features = {feature: 1 for feature in _context_features(context)}
    idf = _history_idf(policy)
    neighbor_count = max(1, int(policy.parameters.get("modality_k", 3)))
    positive_char = _top_n_mean(
        (
            _tfidf_cosine_similarity(features, document, idf)
            for document in policy.positive_contexts
        ),
        neighbor_count,
    )
    negative_char = _top_n_mean(
        (
            _tfidf_cosine_similarity(features, document, idf)
            for document in policy.negative_contexts
        ),
        neighbor_count,
    )
    semantic_query = _semantic_query(policy, context)
    positive_semantic = (
        _top_n_mean(
            (
                _cosine_similarity(semantic_query, document.vector)
                for document in policy.positive_semantic_contexts
            ),
            neighbor_count,
        )
        if semantic_query is not None
        else 0.0
    )
    negative_semantic = (
        _top_n_mean(
            (
                _cosine_similarity(semantic_query, document.vector)
                for document in policy.negative_semantic_contexts
            ),
            neighbor_count,
        )
        if semantic_query is not None
        else 0.0
    )
    semantic_weight = (
        float(policy.parameters.get("semantic_weight", 0.65))
        if semantic_query is not None
        else 0.0
    )
    positive_similarity = (
        semantic_weight * positive_semantic
        + (1.0 - semantic_weight) * positive_char
    )
    negative_similarity = (
        semantic_weight * negative_semantic
        + (1.0 - semantic_weight) * negative_char
    )
    similarity_total = positive_similarity + negative_similarity
    similarity_probability = (
        positive_similarity / similarity_total
        if similarity_total > 0
        else 0.0
    )
    positive_count = len(policy.positive_contexts)
    negative_count = len(policy.negative_contexts)
    prior = (positive_count + 1) / (positive_count + negative_count + 2)
    return round(0.8 * similarity_probability + 0.2 * prior, 8)


def should_send_sticker(policy: StickerPolicy, context: StickerContext) -> bool:
    """Make the modality decision separately from asset retrieval."""

    evidence_count = len(policy.positive_contexts) + len(policy.negative_contexts)
    if evidence_count < int(policy.parameters.get("minimum_context_hits", 1)):
        return False
    if _explicit_sticker_request(context.current_text):
        return bool(policy.assets)
    threshold = float(policy.parameters.get("modality_threshold", 0.5))
    return sticker_modality_score(policy, context) >= threshold


def _explicit_sticker_request(content: str) -> bool:
    return bool(
        re.search(
            r"(?:发|来|用|回)(?:一)?(?:个|张)?(?:表情|表情包|sticker)",
            content,
            flags=re.IGNORECASE,
        )
    )


def tune_sticker_policy_on_valid(
    policy: StickerPolicy,
    cases: tuple[StickerEvaluationCase, ...],
    *,
    parameter_grid: tuple[dict[str, float | int], ...],
) -> tuple[StickerPolicy, dict[str, float]]:
    """只用 valid 网格调参，并返回固定口径的离线指标。"""

    if not cases or any(case.split != "valid" for case in cases):
        raise ValueError("StickerPolicy 参数网格只能读取 valid 样本")
    if not parameter_grid:
        raise ValueError("valid 参数网格不能为空")
    score_cache: dict[tuple[tuple[str, object], ...], tuple[float, ...]] = {}
    ranking_cache: dict[
        tuple[tuple[str, object], ...],
        tuple[tuple[str, ...], ...],
    ] = {}
    best: tuple[tuple[float, float, float, float], StickerPolicy, dict[str, float]] | None = None
    for parameters in parameter_grid:
        candidate = replace(
            policy,
            parameters={
                **policy.parameters,
                **parameters,
                "tuned_on_split": "valid",
            },
        )
        score_key = tuple(
            sorted(
                (key, value)
                for key, value in candidate.parameters.items()
                if key
                not in {
                    "modality_threshold",
                    "minimum_similarity",
                    "top_k",
                    "repetition_penalty",
                    "asset_context_top_n",
                    "tuned_on_split",
                }
            )
        )
        scores = score_cache.get(score_key)
        if scores is None:
            scores = tuple(sticker_modality_score(candidate, case.context) for case in cases)
            score_cache[score_key] = scores
        ranking_key = tuple(
            sorted(
                (key, value)
                for key, value in candidate.parameters.items()
                if key not in {"modality_threshold", "tuned_on_split"}
            )
        )
        rankings = ranking_cache.get(ranking_key)
        if rankings is None:
            rankings = tuple(
                tuple(item.asset_id for item in rank_stickers(candidate, case.context, top_k=5))
                for case in cases
            )
            ranking_cache[ranking_key] = rankings
        threshold = float(candidate.parameters.get("modality_threshold", 0.5))
        predictions = tuple(
            ranking if score >= threshold else ()
            for ranking, score in zip(rankings, scores, strict=True)
        )
        metrics = _sticker_case_metrics(candidate, cases, predictions)
        score = (
            metrics["modality_f1"],
            metrics["seen_positive_recall_at_5"],
            metrics["seen_positive_mrr"],
            metrics["coverage"],
        )
        if best is None or score > best[0]:
            best = (score, candidate, metrics)
    if best is None:
        raise RuntimeError("valid 参数调优没有产生候选")
    metrics = best[2]
    valid_count = max(1, int(metrics["case_count"]))
    seen_count = max(1, int(metrics["seen_positive_count"]))
    calibrated = replace(
        best[1],
        parameters={
            **best[1].parameters,
            "threshold_source": "valid-only-confidence-lower-bound-v1",
            "minimum_modality_f1": _confidence_lower_bound(
                metrics["modality_f1"],
                valid_count,
            ),
            "minimum_seen_positive_recall_at_5": _confidence_lower_bound(
                metrics["seen_positive_recall_at_5"],
                seen_count,
            ),
            "minimum_seen_positive_mrr": _confidence_lower_bound(
                metrics["seen_positive_mrr"],
                seen_count,
            ),
        },
    )
    return calibrated, metrics


def evaluate_sticker_policy(
    policy: StickerPolicy,
    cases: tuple[StickerEvaluationCase, ...],
    *,
    expected_split: str,
) -> dict[str, float]:
    """使用冻结参数评估指定切分，不执行任何调参。"""

    if not cases or any(case.split != expected_split for case in cases):
        raise ValueError(f"StickerPolicy 评估只接受 {expected_split} 样本")
    return _evaluate_sticker_cases(policy, cases)


def _evaluate_sticker_cases(
    policy: StickerPolicy,
    cases: tuple[StickerEvaluationCase, ...],
) -> dict[str, float]:
    predictions = tuple(
        (
            tuple(item.asset_id for item in rank_stickers(policy, case.context, top_k=5))
            if should_send_sticker(policy, case.context)
            else ()
        )
        for case in cases
    )
    return _sticker_case_metrics(policy, cases, predictions)


def _sticker_case_metrics(
    policy: StickerPolicy,
    cases: tuple[StickerEvaluationCase, ...],
    predictions: tuple[tuple[str, ...], ...],
) -> dict[str, float]:
    actual = tuple(case.actual_asset_id for case in cases)
    known = tuple(asset.asset_id for asset in policy.assets)
    base = evaluate_sticker_predictions(
        predictions=predictions,
        actual=actual,
        known_asset_ids=known,
    )
    seen_pairs = [
        (ranking, expected)
        for ranking, expected in zip(predictions, actual, strict=True)
        if expected is not None and expected in known
    ]
    recall = (
        sum(expected in ranking for ranking, expected in seen_pairs) / len(seen_pairs)
        if seen_pairs
        else 0.0
    )
    mrr = (
        sum(
            1.0 / (ranking.index(expected) + 1) if expected in ranking else 0.0
            for ranking, expected in seen_pairs
        )
        / len(seen_pairs)
        if seen_pairs
        else 0.0
    )
    return {
        "seen_positive_recall_at_5": recall,
        "seen_positive_mrr": mrr,
        "modality_f1": base["modality_f1"],
        "coverage": base["coverage"],
        "case_count": float(len(cases)),
        "seen_positive_count": float(len(seen_pairs)),
    }


def evaluate_sticker_predictions(
    *,
    predictions: tuple[tuple[str, ...], ...],
    actual: tuple[str | None, ...],
    known_asset_ids: tuple[str, ...],
) -> dict[str, float]:
    if len(predictions) != len(actual):
        raise ValueError("预测与真实标签数量不一致")
    true_positive = sum(
        bool(predicted) and expected is not None
        for predicted, expected in zip(predictions, actual, strict=True)
    )
    predicted_positive = sum(bool(predicted) for predicted in predictions)
    actual_positive = sum(expected is not None for expected in actual)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ranked_cases = [
        (predicted, expected)
        for predicted, expected in zip(predictions, actual, strict=True)
        if expected is not None
    ]
    recall_at_k = (
        sum(expected in predicted for predicted, expected in ranked_cases) / len(ranked_cases)
        if ranked_cases
        else 0.0
    )
    reciprocal_ranks = [
        1.0 / (predicted.index(expected) + 1)
        if expected in predicted
        else 0.0
        for predicted, expected in ranked_cases
    ]
    first_ids = [predicted[0] if predicted else None for predicted in predictions]
    repeat_pairs = sum(
        current is not None and current == previous
        for previous, current in zip(first_ids, first_ids[1:], strict=False)
    )
    unique_predicted = {asset_id for predicted in predictions for asset_id in predicted}
    return {
        "modality_precision": precision,
        "modality_recall": recall,
        "modality_f1": f1,
        "recall_at_k": recall_at_k,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ineffective_rate": (
            sum(
                bool(predicted) and expected is None
                for predicted, expected in zip(predictions, actual, strict=True)
            )
            / predicted_positive
            if predicted_positive
            else 0.0
        ),
        "consecutive_repeat_rate": repeat_pairs / max(1, len(predictions) - 1),
        "coverage": len(unique_predicted) / len(set(known_asset_ids)) if known_asset_ids else 0.0,
    }


def _history_features(items: list[StickerHistoryItem]) -> tuple[str, ...]:
    recent = tuple(
        RecentStickerBubble(
            kind=item.kind,
            content=item.text,
            asset_id=item.asset_id,
            timestamp=item.timestamp,
            sender="target" if item.is_target else "other",
        )
        for item in items
    )
    current_text = next((item.text for item in reversed(items) if item.text.strip()), "")
    return _context_features(StickerContext(current_text=current_text, recent=recent))


def _context_features(context: StickerContext) -> tuple[str, ...]:
    features: set[str] = set()
    normalized = _normalize_text(context.current_text)
    if normalized:
        for size in (2, 3):
            for index in range(max(0, len(normalized) - size + 1)):
                features.add(f"char-{size}:{normalized[index:index + size]}")
    recent_kinds = ",".join(item.kind for item in context.recent[-3:])
    if recent_kinds:
        features.add(f"recent-kinds:{recent_kinds}")
    recent_senders = ",".join(
        item.sender for item in context.recent[-3:] if item.sender
    )
    if recent_senders:
        features.add(f"recent-senders:{recent_senders}")
    recent_assets = ",".join(
        item.asset_id
        for item in context.recent[-4:]
        if item.kind == "sticker" and item.asset_id
    )
    if recent_assets:
        features.add(f"recent-assets:{recent_assets}")
    if context.recent and context.recent[-1].sender:
        features.add(f"previous-sender:{context.recent[-1].sender}")
    sticker_positions = ",".join(
        str(index)
        for index, item in enumerate(context.recent[-4:])
        if item.kind == "sticker"
    )
    if sticker_positions:
        features.add(f"recent-sticker-positions:{sticker_positions}")
    consecutive_stickers = 0
    for item in reversed(context.recent):
        if item.kind != "sticker":
            break
        consecutive_stickers += 1
    if consecutive_stickers:
        features.add(f"consecutive-stickers:{min(consecutive_stickers, 3)}")
    pragmatic = context.pragmatic or _pragmatic_label(context.current_text)
    relationship = context.relationship or _relationship_label(context.current_text)
    emotion = context.emotion or _emotion_label(context.current_text)
    for prefix, value in (
        ("pragmatic", pragmatic),
        ("relationship", relationship),
        ("emotion", emotion),
    ):
        if value:
            features.add(f"{prefix}:{value}")
    if len(context.recent) >= 2:
        left = context.recent[-2].timestamp
        right = context.recent[-1].timestamp
        if left is not None and right is not None:
            features.add(f"gap:{_gap_bucket(_aware(right) - _aware(left))}")
    return tuple(sorted(features))


def _normalize_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())[:80]


def _semantic_history_text(items: list[StickerHistoryItem]) -> str:
    context = StickerContext(
        current_text=next(
            (item.text for item in reversed(items) if item.text.strip()),
            "",
        ),
        recent=tuple(
            RecentStickerBubble(
                kind=item.kind,
                content=item.text,
                asset_id=item.asset_id,
                sender=item.sender or ("target" if item.is_target else "other"),
            )
            for item in items[-8:]
        ),
    )
    return _semantic_context_text(context)


def _semantic_context_text(context: StickerContext) -> str:
    parts = [
        (
            f"[sender={item.sender or 'unknown'}][kind={item.kind}]"
            + (f"[asset={item.asset_id}]" if item.asset_id else "")
            + item.content
        )
        for item in context.recent[-8:]
    ]
    if context.current_text:
        parts.append(f"[sender=current][kind=text]{context.current_text}")
    return "\n".join(parts)


def _semantic_document(text: str, vector: list[float]) -> SemanticContext:
    resolved = tuple(float(number) for number in vector)
    if not resolved or not all(math.isfinite(number) for number in resolved):
        raise ValueError("语义向量无效")
    return SemanticContext(
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        vector=resolved,
    )


def _semantic_query(
    policy: StickerPolicy,
    context: StickerContext,
) -> tuple[float, ...] | None:
    if policy._embedder is None:
        return None
    text = _semantic_context_text(context)
    if not text:
        return None
    try:
        vectors = policy._embedder.embed([text])
        if len(vectors) != 1:
            return None
        document = _semantic_document(text, vectors[0])
    except Exception:
        return None
    return document.vector


def _paired_contexts(
    feature_contexts: tuple[dict[str, int], ...],
    semantic_contexts: tuple[SemanticContext, ...],
) -> tuple[tuple[dict[str, int], SemanticContext], ...]:
    empty = SemanticContext(text_hash="", vector=())
    return tuple(
        (
            document,
            semantic_contexts[index] if index < len(semantic_contexts) else empty,
        )
        for index, document in enumerate(feature_contexts)
    )


def _combined_similarity(
    policy: StickerPolicy,
    char_similarity: float,
    semantic_similarity: float,
) -> float:
    if policy._embedder is None:
        return char_similarity
    char_weight = float(policy.parameters.get("char_weight", 0.35))
    semantic_weight = float(policy.parameters.get("semantic_weight", 0.65))
    return char_weight * char_similarity + semantic_weight * semantic_similarity


def _cosine_similarity(
    left: tuple[float, ...] | None,
    right: tuple[float, ...],
) -> float:
    if left is None or not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not numerator or not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _pragmatic_label(value: str) -> str | None:
    if any(marker in value for marker in ("?", "？", "吗", "呢")):
        return "question"
    if any(marker in value for marker in ("早", "晚安", "睡了")):
        return "greeting"
    if any(marker in value for marker in ("抱抱", "别难过", "辛苦")):
        return "comfort"
    return None


def _relationship_label(value: str) -> str | None:
    if any(marker in value for marker in ("想你", "爱你", "宝贝", "亲亲", "抱抱")):
        return "intimate"
    return None


def _emotion_label(value: str) -> str | None:
    if any(marker in value for marker in ("开心", "哈哈", "好耶")):
        return "positive"
    if any(marker in value for marker in ("难过", "哭", "委屈", "累")):
        return "negative"
    return None


def _gap_bucket(value: timedelta) -> str:
    seconds = value.total_seconds()
    if seconds < 60:
        return "under-1m"
    if seconds < 3600:
        return "1m-1h"
    if seconds < 86_400:
        return "1h-1d"
    return "over-1d"


def _is_verified_sticker(item: StickerHistoryItem, project_id: str) -> bool:
    return (
        item.kind == "sticker"
        and bool(item.asset_id)
        and item.asset_project_id == project_id
        and item.asset_kind == "sticker"
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, UTC)
    return _aware(datetime.fromisoformat(value))


def _string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, str)}


def _int_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, int)}


def _number_dict(value: object) -> dict[str, float | int | str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int | float | str)
    }


def _feature_documents(value: object) -> tuple[dict[str, int], ...]:
    if not isinstance(value, list):
        return ()
    documents: list[dict[str, int]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        documents.append(
            {
                str(key): int(count)
                for key, count in item.items()
                if isinstance(count, int)
            }
        )
    return tuple(documents)


def _semantic_documents(value: object) -> tuple[SemanticContext, ...]:
    if not isinstance(value, list):
        return ()
    result: list[SemanticContext] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text_hash = item.get("text_hash")
        vector = item.get("vector")
        if not isinstance(text_hash, str) or not isinstance(vector, list):
            continue
        if not all(isinstance(number, int | float) for number in vector):
            continue
        result.append(
            SemanticContext(
                text_hash=text_hash,
                vector=tuple(float(number) for number in vector),
            )
        )
    return tuple(result)


def _history_idf(policy: StickerPolicy) -> dict[str, float]:
    documents = (*policy.positive_contexts, *policy.negative_contexts)
    if not documents:
        return {}
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document)
    count = len(documents)
    return {
        feature: math.log((1.0 + count) / (1.0 + frequency)) + 1.0
        for feature, frequency in document_frequency.items()
    }


def _tfidf_cosine_similarity(
    left: dict[str, int],
    right: dict[str, int],
    idf: dict[str, float],
) -> float:
    if not left or not right:
        return 0.0
    left_total = sum(left.values())
    right_total = sum(right.values())
    left_weights = {
        key: value / left_total * idf.get(key, 1.0)
        for key, value in left.items()
    }
    right_weights = {
        key: value / right_total * idf.get(key, 1.0)
        for key, value in right.items()
    }
    numerator = sum(
        value * right_weights.get(key, 0.0)
        for key, value in left_weights.items()
    )
    left_norm = math.sqrt(sum(value * value for value in left_weights.values()))
    right_norm = math.sqrt(sum(value * value for value in right_weights.values()))
    return numerator / (left_norm * right_norm) if numerator and right_norm else 0.0


def _top_n_mean(values: Iterable[float], count: int) -> float:
    ranked = sorted((float(value) for value in values), reverse=True)
    selected = ranked[: max(1, count)]
    return sum(selected) / len(selected) if selected else 0.0


def _confidence_lower_bound(value: float, count: int) -> float:
    """用 valid 样本量预注册近似 95% 置信下界。"""

    resolved_count = max(1, count)
    margin = 1.96 * math.sqrt(max(0.0, value * (1.0 - value)) / resolved_count)
    return max(0.0, value - margin)
