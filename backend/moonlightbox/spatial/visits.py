from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.spatial.models import (
    PlaceMention,
    PlaceResolution,
    VisitEpisode,
    VisitObservation,
)

PRESENCE_STRENGTH = {
    "present": 0.95,
    "arrived": 0.90,
    "departed": 0.90,
    "en_route": 0.65,
    "passed_by": 0.55,
    "planned": 0.25,
    "mentioned": 0.05,
    "cancelled": 0.0,
}

STATE_FACTOR = {
    "present": 1.0,
    "arrived": 1.0,
    "departed": 1.0,
    "passed_by": 0.25,
    "en_route": 0.0,
    "planned": 0.0,
    "mentioned": 0.0,
    "cancelled": 0.0,
}

AUTHORITY_FACTOR = {
    "structured_location": 1.0,
    "self_reported": 0.85,
    "jointly_confirmed": 0.80,
    "target_reported_by_self": 0.45,
    "third_party": 0.35,
    "inferred": 0.20,
}


def build_visit_observations(
    session: Session,
    *,
    run_id: str,
    target_person_id: str,
) -> list[VisitObservation]:
    rows = session.execute(
        select(PlaceMention, PlaceResolution)
        .join(PlaceResolution, PlaceResolution.mention_id == PlaceMention.id)
        .where(PlaceMention.run_id == run_id)
        .order_by(PlaceMention.time_start_lower, PlaceMention.message_id, PlaceMention.id)
    ).all()
    observations: list[VisitObservation] = []
    for mention, resolution in rows:
        if mention.subject_id is None:
            continue
        state = visit_state_for_mention(mention)
        authority = source_authority_for_mention(mention, target_person_id)
        entity_confidence = resolution.confidence if resolution.place_id else 0.0
        subject_confidence = 1.0 if mention.subject_resolution == "explicit" else 0.75
        time_confidence = 1.0 if mention.time_precision != "unknown" else 0.5
        raw_confidence = max(
            0.0,
            min(
                1.0,
                0.30 * mention.confidence
                + 0.25 * entity_confidence
                + 0.15 * subject_confidence
                + 0.15 * time_confidence
                + 0.15 * PRESENCE_STRENGTH[state],
            ),
        )
        observation = VisitObservation(
            run_id=run_id,
            subject_id=mention.subject_id,
            place_id=resolution.place_id,
            mention_ids=[mention.id],
            visit_state=state,
            assertion_mode=mention.assertion_mode,
            started_at_lower=mention.time_start_lower,
            started_at_upper=mention.time_start_upper,
            ended_at_lower=mention.time_end_lower,
            ended_at_upper=mention.time_end_upper,
            purpose_distribution={},
            companion_person_ids=[],
            source_authority=authority,
            verification_status=verification_status(authority),
            confidence=raw_confidence,
            evidence_message_ids=list(dict.fromkeys(mention.evidence_message_ids)),
        )
        session.add(observation)
        observations.append(observation)
    session.flush()
    return observations


def build_visit_episodes(
    session: Session,
    *,
    run_id: str,
    subject_id: str,
    merge_gap: timedelta = timedelta(hours=12),
) -> list[VisitEpisode]:
    observations = list(
        session.scalars(
            select(VisitObservation)
            .where(
                VisitObservation.run_id == run_id,
                VisitObservation.subject_id == subject_id,
                VisitObservation.place_id.is_not(None),
            )
            .order_by(VisitObservation.started_at_lower, VisitObservation.id)
        )
    )
    groups: dict[str, list[list[VisitObservation]]] = defaultdict(list)
    for observation in observations:
        if observation.visit_state not in {"arrived", "present", "departed", "passed_by"}:
            continue
        place_id = observation.place_id
        if place_id is None:
            continue
        place_groups = groups[place_id]
        if not place_groups or not _can_merge(place_groups[-1], observation, merge_gap):
            place_groups.append([observation])
        else:
            place_groups[-1].append(observation)

    episodes: list[VisitEpisode] = []
    for place_id, place_groups in sorted(groups.items()):
        for group in place_groups:
            q_values = [observation_quality(item) for item in group]
            started = _minimum(item.started_at_lower for item in group)
            started_upper = _minimum(item.started_at_upper for item in group)
            ended = _maximum(item.ended_at_lower for item in group)
            ended_upper = _maximum(item.ended_at_upper for item in group)
            dwell_lower = _duration_hours(started_upper, ended)
            dwell_upper = _duration_hours(started, ended_upper)
            episode = VisitEpisode(
                run_id=run_id,
                subject_id=subject_id,
                place_id=place_id,
                observation_ids=[item.id for item in group],
                started_at_lower=started,
                started_at_upper=started_upper,
                ended_at_lower=ended,
                ended_at_upper=ended_upper,
                dwell_hours_lower=dwell_lower,
                dwell_hours_upper=dwell_upper,
                left_censored=all(item.visit_state != "arrived" for item in group),
                right_censored=all(item.visit_state != "departed" for item in group),
                confidence=1.0 - _product(1.0 - value for value in q_values),
                evidence_root_ids=sorted(
                    {message_id for item in group for message_id in item.evidence_message_ids}
                ),
            )
            session.add(episode)
            episodes.append(episode)
    session.flush()
    return episodes


def visit_state_for_mention(mention: PlaceMention) -> str:
    if mention.assertion_mode in {"cancelled", "negated"}:
        return "cancelled"
    if mention.assertion_mode in {"planned", "hypothetical"} or mention.movement_phase == "planned":
        return "planned"
    if mention.movement_phase == "arriving":
        return "arrived"
    if mention.movement_phase == "departing" or mention.relation == "from":
        return "departed"
    if mention.movement_phase == "in_transit":
        return "en_route"
    if mention.relation == "through":
        return "passed_by"
    if mention.relation in {"at", "near"}:
        return "present"
    return "mentioned"


def source_authority_for_mention(mention: PlaceMention, target_person_id: str) -> str:
    if mention.extractor_method == "structured_location":
        return "structured_location"
    if mention.speaker_id == mention.subject_id:
        return "self_reported"
    if mention.subject_id == target_person_id:
        return "target_reported_by_self"
    return "third_party"


def verification_status(authority: str) -> str:
    return {
        "structured_location": "observed",
        "self_reported": "self_reported",
        "jointly_confirmed": "confirmed",
        "target_reported_by_self": "third_party",
        "third_party": "third_party",
        "inferred": "inferred",
    }[authority]


def observation_quality(observation: VisitObservation) -> float:
    return (
        observation.confidence
        * AUTHORITY_FACTOR.get(observation.source_authority, 0.20)
        * STATE_FACTOR.get(observation.visit_state, 0.0)
    )


def _can_merge(
    current: list[VisitObservation],
    candidate: VisitObservation,
    merge_gap: timedelta,
) -> bool:
    previous = current[-1]
    if previous.started_at_lower is None or candidate.started_at_lower is None:
        return False
    return candidate.started_at_lower - previous.started_at_lower <= merge_gap


def _minimum(values: Iterable[datetime | None]) -> datetime | None:
    retained = [value for value in values if isinstance(value, datetime)]
    return min(retained) if retained else None


def _maximum(values: Iterable[datetime | None]) -> datetime | None:
    retained = [value for value in values if isinstance(value, datetime)]
    return max(retained) if retained else None


def _duration_hours(started: datetime | None, ended: datetime | None) -> float | None:
    if started is None or ended is None or ended < started:
        return None
    return (ended - started).total_seconds() / 3600


def _product(values: Iterable[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result
