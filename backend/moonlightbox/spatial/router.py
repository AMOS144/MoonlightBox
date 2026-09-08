import math
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Participant
from moonlightbox.spatial.jobs import enqueue_spatial_analysis
from moonlightbox.spatial.models import (
    PlaceEntity,
    PlaceGraphNode,
    PlaceGraphSnapshot,
    VisitEpisode,
)

Metric = Literal["activity", "visits", "dwell"]
PrivacyLevel = Literal["exact", "blurred", "hidden"]


def create_spatial_router(database: Database, settings: Settings) -> APIRouter:
    router = APIRouter(tags=["spatial"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.get("/api/projects/{project_id}/place-heatmap")
    def get_target_heatmap(
        project_id: str,
        session: SessionDependency,
        snapshot_id: str | None = None,
        metric: Metric = "activity",
        min_confidence: float = Query(default=0.0, ge=0, le=1),
        privacy_level: PrivacyLevel | None = None,
        normalization: Literal["relative", "fixed"] = "relative",
        from_at: Annotated[datetime | None, Query(alias="from")] = None,
        to_at: Annotated[datetime | None, Query(alias="to")] = None,
        weekdays: str | None = None,
        hour_start: int | None = Query(default=None, ge=0, le=23),
        hour_end: int | None = Query(default=None, ge=1, le=24),
    ) -> dict[str, object]:
        target = session.scalar(
            select(Participant).where(
                Participant.project_id == project_id,
                Participant.role == "target",
            )
        )
        if target is None:
            raise HTTPException(status_code=404, detail="项目没有目标人物")
        return _heatmap_response(
            session,
            project_id=project_id,
            person=target,
            snapshot_id=snapshot_id,
            metric=metric,
            min_confidence=min_confidence,
            privacy_level=privacy_level or settings.spatial_heatmap_default_privacy,
            normalization=normalization,
            from_at=from_at,
            to_at=to_at,
            weekdays=_parse_weekdays(weekdays),
            hour_start=hour_start,
            hour_end=hour_end,
        )

    @router.post("/api/projects/{project_id}/spatial-analysis", status_code=202)
    def analyze_existing_imports(
        project_id: str,
        session: SessionDependency,
    ) -> dict[str, object]:
        if not settings.spatial_analysis_enabled:
            raise HTTPException(
                status_code=409,
                detail="请先设置 MOONLIGHTBOX_SPATIAL_ANALYSIS_ENABLED=true",
            )
        import_ids = list(
            session.scalars(
                select(ImportSource.id)
                .where(ImportSource.project_id == project_id)
                .order_by(ImportSource.confirmed_at, ImportSource.id)
            )
        )
        if not import_ids:
            raise HTTPException(status_code=404, detail="项目还没有已确认的聊天导入")
        jobs = [
            enqueue_spatial_analysis(
                session,
                settings=settings,
                project_id=project_id,
                import_id=import_id,
            )
            for import_id in import_ids
        ]
        return {
            "project_id": project_id,
            "job_ids": [job.id for job in jobs],
            "import_count": len(import_ids),
        }

    @router.get("/api/projects/{project_id}/people/{person_id}/place-heatmap")
    def get_person_heatmap(
        project_id: str,
        person_id: str,
        session: SessionDependency,
        snapshot_id: str | None = None,
        metric: Metric = "activity",
        min_confidence: float = Query(default=0.0, ge=0, le=1),
        privacy_level: PrivacyLevel | None = None,
        normalization: Literal["relative", "fixed"] = "relative",
        from_at: Annotated[datetime | None, Query(alias="from")] = None,
        to_at: Annotated[datetime | None, Query(alias="to")] = None,
        weekdays: str | None = None,
        hour_start: int | None = Query(default=None, ge=0, le=23),
        hour_end: int | None = Query(default=None, ge=1, le=24),
    ) -> dict[str, object]:
        person = session.scalar(
            select(Participant).where(
                Participant.id == person_id,
                Participant.project_id == project_id,
            )
        )
        if person is None:
            raise HTTPException(status_code=404, detail="人物不存在")
        return _heatmap_response(
            session,
            project_id=project_id,
            person=person,
            snapshot_id=snapshot_id,
            metric=metric,
            min_confidence=min_confidence,
            privacy_level=privacy_level or settings.spatial_heatmap_default_privacy,
            normalization=normalization,
            from_at=from_at,
            to_at=to_at,
            weekdays=_parse_weekdays(weekdays),
            hour_start=hour_start,
            hour_end=hour_end,
        )

    return router


def _heatmap_response(
    session: Session,
    *,
    project_id: str,
    person: Participant,
    snapshot_id: str | None,
    metric: Metric,
    min_confidence: float,
    privacy_level: PrivacyLevel,
    normalization: Literal["relative", "fixed"],
    from_at: datetime | None,
    to_at: datetime | None,
    weekdays: set[int] | None,
    hour_start: int | None,
    hour_end: int | None,
) -> dict[str, object]:
    if from_at is not None and to_at is not None and from_at > to_at:
        raise HTTPException(status_code=422, detail="from 不能晚于 to")
    if hour_start is not None and hour_end is not None and hour_start >= hour_end:
        raise HTTPException(status_code=422, detail="hour_start 必须小于 hour_end")
    query = select(PlaceGraphSnapshot).where(
        PlaceGraphSnapshot.project_id == project_id,
        PlaceGraphSnapshot.person_id == person.id,
    )
    if snapshot_id is not None:
        query = query.where(PlaceGraphSnapshot.id == snapshot_id)
    snapshot = session.scalar(
        query.order_by(PlaceGraphSnapshot.created_at.desc(), PlaceGraphSnapshot.id.desc())
    )
    if snapshot is None:
        return {
            "type": "FeatureCollection",
            "properties": {
                "person_id": person.id,
                "person_name": person.name,
                "snapshot_id": None,
                "metric": metric,
                "located_mass": 0.0,
                "unlocated_mass": 0.0,
                "normalization": normalization,
                "data_warning": "空间分析尚未生成个人地点图",
            },
            "features": [],
            "unlocated_places": [],
        }
    rows = session.execute(
        select(PlaceGraphNode, PlaceEntity)
        .join(PlaceEntity, PlaceEntity.id == PlaceGraphNode.place_id)
        .where(PlaceGraphNode.snapshot_id == snapshot.id)
        .order_by(PlaceGraphNode.activity_weight.desc(), PlaceEntity.canonical_name)
    ).all()
    filtered = _filtered_metrics(
        session,
        snapshot=snapshot,
        place_ids={node.place_id for node, _place in rows},
        from_at=from_at,
        to_at=to_at,
        weekdays=weekdays,
        hour_start=hour_start,
        hour_end=hour_end,
    )
    has_filter = any(
        value is not None
        for value in (from_at, to_at, weekdays, hour_start, hour_end)
    )
    values = [
        _filtered_metric_value(filtered.get(node.place_id), metric)
        if has_filter
        else _metric_value(node, metric)
        for node, _place in rows
    ]
    max_value = max(values, default=0.0)
    features: list[dict[str, object]] = []
    unlocated: list[dict[str, object]] = []
    located_mass = 0.0
    unlocated_mass = 0.0
    for (node, place), raw_value in zip(rows, values, strict=False):
        if has_filter and raw_value <= 0:
            continue
        filtered_item = filtered.get(node.place_id) if has_filter else None
        evidence_strength = (
            filtered_item["evidence_strength"]
            if filtered_item is not None
            else _metric_float(node.metrics, "evidence_strength")
        )
        if evidence_strength < min_confidence:
            continue
        display_value = (
            math.sqrt(raw_value / max_value)
            if normalization == "relative" and max_value > 0
            else math.sqrt(max(0.0, min(1.0, raw_value)))
        )
        properties = {
            "place_id": place.id,
            "display_name": place.canonical_name,
            "primary_role": _primary_role(node.role_distribution),
            "role_distribution": node.role_distribution,
            "heat_raw": raw_value,
            "heat_display": display_value,
            "activity_weight": (
                filtered_item["activity"] if filtered_item is not None else node.activity_weight
            ),
            "effective_visit_count": (
                filtered_item["visits"]
                if filtered_item is not None
                else _metric_float(node.metrics, "effective_visit_count")
            ),
            "effective_dwell_hours": (
                filtered_item["dwell"]
                if filtered_item is not None
                else _metric_float(node.metrics, "effective_dwell_hours")
            ),
            "dwell_coverage": (
                filtered_item["dwell_coverage"]
                if filtered_item is not None
                else _metric_float(node.metrics, "dwell_coverage")
            ),
            "evidence_strength": evidence_strength,
            "first_visit_at": node.metrics.get("first_visit_at"),
            "last_visit_at": node.metrics.get("last_visit_at"),
            "geo_resolution": place.geo_resolution,
            "uncertainty_radius_m": place.uncertainty_radius_m,
            "privacy_level": privacy_level,
            "coordinate_crs": place.coordinate_crs,
        }
        has_coordinate = place.longitude is not None and place.latitude is not None
        if has_coordinate and privacy_level != "hidden":
            longitude, latitude = _privacy_coordinate(
                place.longitude,
                place.latitude,
                privacy_level,
            )
            located_mass += raw_value
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [longitude, latitude],
                    },
                    "properties": properties,
                }
            )
        else:
            unlocated_mass += raw_value
            unlocated.append(properties)
    return {
        "type": "FeatureCollection",
        "properties": {
            "person_id": person.id,
            "person_name": person.name,
            "snapshot_id": snapshot.id,
            "graph_version": snapshot.graph_version,
            "created_at": snapshot.created_at.isoformat(),
            "metric": metric,
            "located_mass": located_mass,
            "unlocated_mass": unlocated_mass,
            "normalization": normalization,
            "reliability": snapshot.reliability,
            "warnings": snapshot.warnings,
            "data_warning": snapshot.warnings[0] if snapshot.warnings else None,
            "filters": {
                "from": from_at.isoformat() if from_at else None,
                "to": to_at.isoformat() if to_at else None,
                "weekdays": sorted(weekdays) if weekdays is not None else None,
                "hour_start": hour_start,
                "hour_end": hour_end,
            },
        },
        "features": features,
        "unlocated_places": unlocated,
    }


def _metric_value(node: PlaceGraphNode, metric: Metric) -> float:
    if metric == "visits":
        return _metric_float(node.metrics, "effective_visit_count")
    if metric == "dwell":
        return _metric_float(node.metrics, "effective_dwell_hours")
    return node.activity_weight


def _metric_float(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _primary_role(distribution: dict[str, float]) -> str | None:
    if not distribution:
        return None
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    top_name, top_value = ordered[0]
    second_value = ordered[1][1] if len(ordered) > 1 else 0.0
    return top_name if top_value >= 0.75 and top_value - second_value >= 0.20 else None


def _privacy_coordinate(
    longitude: float,
    latitude: float,
    privacy_level: PrivacyLevel,
) -> tuple[float, float]:
    if privacy_level == "exact":
        return longitude, latitude
    # 约 1 km 网格；量化在后端完成，精确家庭/医疗坐标不会下发给浏览器。
    return round(longitude, 2), round(latitude, 2)


def _parse_weekdays(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    try:
        weekdays = {int(item.strip()) for item in value.split(",")}
    except ValueError as error:
        raise HTTPException(status_code=422, detail="weekdays 必须是 0 到 6 的逗号列表") from error
    if not weekdays or not weekdays.issubset(set(range(7))):
        raise HTTPException(status_code=422, detail="weekdays 必须是 0 到 6 的逗号列表")
    return weekdays


def _filtered_metrics(
    session: Session,
    *,
    snapshot: PlaceGraphSnapshot,
    place_ids: set[str],
    from_at: datetime | None,
    to_at: datetime | None,
    weekdays: set[int] | None,
    hour_start: int | None,
    hour_end: int | None,
) -> dict[str, dict[str, float]]:
    visits = list(
        session.scalars(
            select(VisitEpisode).where(
                VisitEpisode.run_id == snapshot.run_id,
                VisitEpisode.subject_id == snapshot.person_id,
                VisitEpisode.place_id.in_(place_ids),
            )
        )
    )
    retained: list[VisitEpisode] = []
    for visit in visits:
        timestamp = visit.started_at_lower or visit.started_at_upper
        if timestamp is None:
            continue
        if from_at is not None and timestamp < _matching_awareness(from_at, timestamp):
            continue
        if to_at is not None and timestamp > _matching_awareness(to_at, timestamp):
            continue
        if weekdays is not None and timestamp.weekday() not in weekdays:
            continue
        if hour_start is not None and timestamp.hour < hour_start:
            continue
        if hour_end is not None and timestamp.hour >= hour_end:
            continue
        retained.append(visit)
    by_place: dict[str, list[VisitEpisode]] = {place_id: [] for place_id in place_ids}
    for visit in retained:
        by_place[visit.place_id].append(visit)
    active_ids = [place_id for place_id, values in by_place.items() if values]
    if not active_ids:
        return {}
    alpha = 3.0
    prior = 1.0 / len(active_ids)
    total_visits = sum(
        visit.confidence for place_id in active_ids for visit in by_place[place_id]
    )
    intermediate: dict[str, dict[str, float]] = {}
    for place_id in active_ids:
        place_visits = by_place[place_id]
        visit_mass = sum(visit.confidence for visit in place_visits)
        known_dwell = [
            (visit.confidence, (visit.dwell_hours_lower + visit.dwell_hours_upper) / 2)
            for visit in place_visits
            if visit.dwell_hours_lower is not None and visit.dwell_hours_upper is not None
        ]
        dwell_mass = sum(quality * dwell for quality, dwell in known_dwell)
        dwell_quality = sum(quality for quality, _dwell in known_dwell)
        evidence = 1.0 - math.prod(
            1.0 - min(visit.confidence, 0.95) for visit in place_visits
        )
        visit_share = (visit_mass + alpha * prior) / (total_visits + alpha)
        base_activity = 0.40 * visit_share + 0.15 * evidence + 0.15
        intermediate[place_id] = {
            "visits": visit_mass,
            "dwell": dwell_mass,
            "dwell_coverage": dwell_quality / visit_mass if visit_mass else 0.0,
            "evidence_strength": evidence,
            "base_activity": base_activity / 0.70,
        }
    total_activity = sum(value["base_activity"] for value in intermediate.values())
    for value in intermediate.values():
        value["activity"] = value["base_activity"] / total_activity if total_activity else 0.0
    return intermediate


def _filtered_metric_value(
    value: dict[str, float] | None,
    metric: Metric,
) -> float:
    if value is None:
        return 0.0
    return value[metric]


def _matching_awareness(reference: datetime, value: datetime) -> datetime:
    if value.tzinfo is None:
        return reference.replace(tzinfo=None)
    return reference if reference.tzinfo is not None else reference.replace(tzinfo=value.tzinfo)
