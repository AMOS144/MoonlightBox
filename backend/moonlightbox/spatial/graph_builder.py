import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.spatial.models import (
    PlaceEntity,
    PlaceGraphEdge,
    PlaceGraphNode,
    PlaceGraphSnapshot,
    PlaceMention,
    PlaceResolution,
    VisitEpisode,
)


@dataclass(slots=True)
class _NodeAccumulator:
    place: PlaceEntity
    mentions: list[PlaceMention]
    visits: list[VisitEpisode]
    effective_visit_count: float = 0.0
    effective_dwell_hours: float = 0.0
    dwell_quality: float = 0.0
    evidence_strength: float = 0.0
    visit_share: float = 0.0
    dwell_share: float | None = None
    recency: float = 0.0
    base_activity: float = 0.0
    activity_weight: float = 0.0


def build_place_graph_snapshot(
    session: Session,
    *,
    project_id: str,
    run_id: str,
    person_id: str,
    graph_version: str,
    timezone: str = "Asia/Shanghai",
    cutoff_at: datetime | None = None,
) -> PlaceGraphSnapshot:
    existing = session.scalar(
        select(PlaceGraphSnapshot).where(
            PlaceGraphSnapshot.run_id == run_id,
            PlaceGraphSnapshot.person_id == person_id,
        )
    )
    if existing is not None:
        return existing
    now = cutoff_at or datetime.now(UTC)
    mention_rows = session.execute(
        select(PlaceMention, PlaceResolution)
        .join(PlaceResolution, PlaceResolution.mention_id == PlaceMention.id)
        .where(
            PlaceMention.run_id == run_id,
            PlaceMention.subject_id == person_id,
            PlaceResolution.place_id.is_not(None),
        )
    ).all()
    place_mentions: dict[str, list[PlaceMention]] = defaultdict(list)
    for mention, resolution in mention_rows:
        if resolution.place_id is not None:
            place_mentions[resolution.place_id].append(mention)
    visits = list(
        session.scalars(
            select(VisitEpisode)
            .where(VisitEpisode.run_id == run_id, VisitEpisode.subject_id == person_id)
            .order_by(VisitEpisode.started_at_lower, VisitEpisode.id)
        )
    )
    place_visits: dict[str, list[VisitEpisode]] = defaultdict(list)
    for visit in visits:
        place_visits[visit.place_id].append(visit)
    place_ids = sorted(set(place_mentions) | set(place_visits))
    accumulators: list[_NodeAccumulator] = []
    for place_id in place_ids:
        place = session.get(PlaceEntity, place_id)
        if place is None:
            continue
        accumulator = _NodeAccumulator(
            place=place,
            mentions=place_mentions[place_id],
            visits=place_visits[place_id],
        )
        _calculate_raw_node_metrics(accumulator, now)
        accumulators.append(accumulator)
    _calculate_shares_and_activity(accumulators)

    content = {
        "person_id": person_id,
        "graph_version": graph_version,
        "nodes": [
            {
                "place_id": item.place.id,
                "activity_weight": round(item.activity_weight, 12),
                "visit_count": round(item.effective_visit_count, 12),
            }
            for item in accumulators
        ],
    }
    content_hash = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reliability = (
        sum(item.evidence_strength for item in accumulators) / len(accumulators)
        if accumulators
        else 0.0
    )
    warnings: list[str] = []
    if not accumulators:
        warnings.append("没有可形成地点图的已解析到访")
    if accumulators and not any(item.place.latitude is not None for item in accumulators):
        warnings.append("地点均未完成地理定位，只能在未定位地点列表中展示")
    snapshot = PlaceGraphSnapshot(
        project_id=project_id,
        person_id=person_id,
        run_id=run_id,
        graph_version=graph_version,
        cutoff_at=cutoff_at,
        timezone=timezone,
        parameters={
            "visit_share_alpha": 3.0,
            "dwell_share_beta_hours": 6.0,
            "activity_components": {
                "visit_share": 0.40,
                "dwell_share": 0.30,
                "recency": 0.15,
                "evidence_strength": 0.15,
            },
        },
        statistics={
            "place_count": len(accumulators),
            "visit_episode_count": len(visits),
            "geocoded_place_count": sum(
                item.place.latitude is not None and item.place.longitude is not None
                for item in accumulators
            ),
        },
        content_hash=content_hash,
        reliability=reliability,
        warnings=warnings,
    )
    session.add(snapshot)
    session.flush()
    for item in accumulators:
        session.add(
            PlaceGraphNode(
                snapshot_id=snapshot.id,
                place_id=item.place.id,
                metrics=_node_metrics(item),
                activity_weight=item.activity_weight,
                salience_score=0.0,
                role_distribution=item.place.role_distribution,
            )
        )
    _build_edges(session, snapshot.id, visits)
    session.flush()
    return snapshot


def _calculate_raw_node_metrics(item: _NodeAccumulator, now: datetime) -> None:
    quality = [max(0.0, min(1.0, visit.confidence)) for visit in item.visits]
    item.effective_visit_count = sum(quality)
    item.evidence_strength = 1.0 - _product(1.0 - min(value, 0.95) for value in quality)
    for visit, q_value in zip(item.visits, quality, strict=False):
        dwell = _known_dwell(visit)
        if dwell is not None:
            item.effective_dwell_hours += q_value * dwell
            item.dwell_quality += q_value
    recency_numerator = 0.0
    for visit, q_value in zip(item.visits, quality, strict=False):
        timestamp = visit.started_at_lower or visit.started_at_upper
        if timestamp is None:
            continue
        age_days = max(
            0.0,
            (_as_aware(now) - _as_aware(timestamp)).total_seconds() / 86400,
        )
        recency_numerator += q_value * math.exp(-math.log(2) * age_days / 60.0)
    item.recency = (
        recency_numerator / item.effective_visit_count
        if item.effective_visit_count > 0
        else 0.0
    )


def _calculate_shares_and_activity(items: list[_NodeAccumulator]) -> None:
    if not items:
        return
    total_visits = sum(item.effective_visit_count for item in items)
    uniform_prior = 1.0 / len(items)
    alpha = 3.0
    for item in items:
        item.visit_share = (
            item.effective_visit_count + alpha * uniform_prior
        ) / (total_visits + alpha)
    total_quality = sum(item.effective_visit_count for item in items)
    total_dwell_quality = sum(item.dwell_quality for item in items)
    total_dwell = sum(item.effective_dwell_hours for item in items)
    dwell_coverage = total_dwell_quality / total_quality if total_quality else 0.0
    if dwell_coverage >= 0.35:
        beta = 6.0
        for item in items:
            item.dwell_share = (
                item.effective_dwell_hours + beta * uniform_prior
            ) / (total_dwell + beta)
    for item in items:
        components = [
            (item.visit_share, 0.40),
            (item.recency, 0.15),
            (item.evidence_strength, 0.15),
        ]
        if item.dwell_share is not None:
            components.append((item.dwell_share, 0.30))
        total_component_weight = sum(weight for _, weight in components)
        item.base_activity = sum(value * weight for value, weight in components) / (
            total_component_weight
        )
    total_activity = sum(item.base_activity for item in items)
    for item in items:
        item.activity_weight = item.base_activity / total_activity if total_activity else 0.0


def _node_metrics(item: _NodeAccumulator) -> dict[str, object]:
    starts = [
        value
        for value in (visit.started_at_lower for visit in item.visits)
        if value is not None
    ]
    mention_times = [
        value
        for mention in item.mentions
        for value in (mention.time_start_lower, mention.time_start_upper)
        if value is not None
    ]
    hourly_mass = [0.0] * 168
    for visit in item.visits:
        timestamp = visit.started_at_lower or visit.started_at_upper
        if timestamp is None:
            continue
        slot = timestamp.weekday() * 24 + timestamp.hour
        hourly_mass[slot] += 0.5 * max(0.0, min(1.0, visit.confidence))
    dwell_coverage = (
        item.dwell_quality / item.effective_visit_count
        if item.effective_visit_count
        else 0.0
    )
    return {
        "first_seen_at": min(mention_times).isoformat() if mention_times else None,
        "last_seen_at": max(mention_times).isoformat() if mention_times else None,
        "first_visit_at": min(starts).isoformat() if starts else None,
        "last_visit_at": max(starts).isoformat() if starts else None,
        "mention_count": len({mention.message_id for mention in item.mentions}),
        "effective_visit_count": item.effective_visit_count,
        "raw_visit_episode_count": len(item.visits),
        "effective_dwell_hours": item.effective_dwell_hours,
        "dwell_coverage": dwell_coverage,
        "evidence_strength": item.evidence_strength,
        "visit_share": item.visit_share,
        "dwell_share": item.dwell_share,
        "recency": item.recency,
        "hourly_mass": hourly_mass,
    }


def _build_edges(
    session: Session,
    snapshot_id: str,
    visits: list[VisitEpisode],
) -> None:
    edge_groups: dict[tuple[str, str], list[tuple[VisitEpisode, VisitEpisode, float]]] = (
        defaultdict(list)
    )
    ordered = sorted(
        visits,
        key=lambda item: (item.started_at_lower or datetime.min, item.id),
    )
    for first, second in zip(ordered, ordered[1:], strict=False):
        if first.place_id == second.place_id:
            continue
        first_time = first.ended_at_upper or first.ended_at_lower or first.started_at_upper
        second_time = second.started_at_lower or second.started_at_upper
        if first_time is None or second_time is None:
            continue
        gap = _as_aware(second_time) - _as_aware(first_time)
        if gap < timedelta(0) or gap > timedelta(hours=12):
            continue
        quality = min(first.confidence, second.confidence)
        temporal = max(0.2, 1.0 - gap.total_seconds() / timedelta(hours=12).total_seconds())
        edge_groups[(first.place_id, second.place_id)].append(
            (first, second, quality * temporal)
        )
    for (from_place_id, to_place_id), evidence in sorted(edge_groups.items()):
        qualities = [item[2] for item in evidence]
        durations = [
            (
                _as_aware(second.started_at_lower) - _as_aware(first.ended_at_upper)
            ).total_seconds()
            for first, second, _quality in evidence
            if first.ended_at_upper is not None and second.started_at_lower is not None
        ]
        session.add(
            PlaceGraphEdge(
                snapshot_id=snapshot_id,
                from_place_id=from_place_id,
                to_place_id=to_place_id,
                metrics={
                    "raw_transition_count": len(evidence),
                    "effective_transition_count": sum(qualities),
                    "observed_duration_seconds": durations,
                    "evidence_ids": [
                        [first.id, second.id] for first, second, _quality in evidence
                    ],
                },
                confidence=1.0 - _product(1.0 - value for value in qualities),
            )
        )


def _known_dwell(visit: VisitEpisode) -> float | None:
    if visit.dwell_hours_lower is None or visit.dwell_hours_upper is None:
        return None
    return (visit.dwell_hours_lower + visit.dwell_hours_upper) / 2


def _product(values: Iterable[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _as_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
