import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message
from moonlightbox.spatial.config import SpatialPipelineConfig
from moonlightbox.spatial.episodes import extract_structured_location
from moonlightbox.spatial.map_provider import Coordinate, MapPlace, MapProvider
from moonlightbox.spatial.models import (
    PlaceAlias,
    PlaceCandidate,
    PlaceEntity,
    PlaceMention,
    PlaceProviderEnrichment,
    PlaceResolution,
)


@dataclass(frozen=True, slots=True)
class ResolutionCandidate:
    key: str
    source: str
    canonical_name: str
    place_type: str
    place_id: str | None = None
    coordinate: Coordinate | None = None
    administrative: dict[str, object] = field(default_factory=dict)
    provider_place: MapPlace | None = None
    features: dict[str, float] = field(default_factory=dict)
    contradictions: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        weights = {
            "exact_alias": 0.30,
            "name_similarity": 0.22,
            "subject_ownership": 0.12,
            "type_compatibility": 0.10,
            "recent_context": 0.10,
            "structured_coordinate": 0.10,
            "geographic_context": 0.06,
        }
        value = sum(weights[name] * self.features.get(name, 0.0) for name in weights)
        return max(0.0, min(1.0, value - 0.18 * len(self.contradictions)))


def normalize_place_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[\s，。！？、,.!?;；:：'\"“”‘’()（）\[\]【】]+", "", normalized)


class PlaceResolutionService:
    """候选生成可并行，但该服务按固定顺序执行唯一的数据库归并。"""

    def __init__(
        self,
        *,
        provider: MapProvider,
        config: SpatialPipelineConfig,
        persist_provider_enrichment: bool,
    ) -> None:
        self._provider = provider
        self._config = config
        self._persist_provider_enrichment = persist_provider_enrichment

    def resolve_run(
        self,
        session: Session,
        *,
        run_id: str,
        project_id: str,
        target_person_id: str,
    ) -> list[PlaceResolution]:
        mentions = list(
            session.scalars(
                select(PlaceMention)
                .where(PlaceMention.run_id == run_id, PlaceMention.status == "active")
                .order_by(PlaceMention.created_at, PlaceMention.message_id, PlaceMention.id)
            )
        )
        mentions.sort(key=lambda item: (_resolution_phase(session, item), item.message_id, item.id))
        resolutions: list[PlaceResolution] = []
        for mention in mentions:
            existing = session.scalar(
                select(PlaceResolution).where(
                    PlaceResolution.run_id == run_id,
                    PlaceResolution.mention_id == mention.id,
                )
            )
            if existing is not None:
                resolutions.append(existing)
                continue
            resolutions.append(
                self._resolve_one(
                    session,
                    mention=mention,
                    project_id=project_id,
                    target_person_id=target_person_id,
                )
            )
            session.flush()
        return resolutions

    def _resolve_one(
        self,
        session: Session,
        *,
        mention: PlaceMention,
        project_id: str,
        target_person_id: str,
    ) -> PlaceResolution:
        candidates = self._collect_candidates(
            session,
            mention=mention,
            project_id=project_id,
            target_person_id=target_person_id,
        )
        ranked = sorted(candidates, key=lambda item: (-item.score, item.key))
        persisted = [
            self._persist_candidate(session, mention, item, rank)
            for rank, item in enumerate(ranked, 1)
        ]
        top = ranked[0] if ranked else None
        second_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = top.score - second_score if top else 0.0
        if top is not None and (
            top.score >= self._config.automatic_link_threshold
            and margin >= self._config.automatic_link_margin
        ):
            decision = "linked"
        elif top is not None and (
            top.score >= self._config.provisional_link_threshold
            and margin >= self._config.provisional_link_margin
        ):
            decision = "provisional"
        elif (
            provider_top := _deterministic_provider_candidate(mention, ranked)
        ) is not None:
            # The language model only extracts the evidence span. AMap candidates
            # are selected here with deterministic rules; no LLM judges POIs.
            top = provider_top
            second_score = max(
                (item.score for item in ranked if item.key != top.key),
                default=0.0,
            )
            margin = top.score - second_score
            decision = "provisional"
        elif mention.mention_type in {"personal_anchor", "named_poi", "administrative"}:
            top = self._semantic_candidate(mention, target_person_id)
            persisted.append(
                self._persist_candidate(session, mention, top, len(persisted) + 1)
            )
            decision = "created_semantic"
            margin = top.score
        else:
            decision = "unresolved"

        place: PlaceEntity | None = None
        selected_candidate: PlaceCandidate | None = None
        if top is not None and decision != "unresolved":
            selected_candidate = next(
                (item for item in persisted if item.candidate_key == top.key),
                None,
            )
            place = session.get(PlaceEntity, top.place_id) if top.place_id else None
            if place is None:
                place = self._create_place(
                    session,
                    project_id=project_id,
                    target_person_id=target_person_id,
                    mention=mention,
                    candidate=top,
                    status="active" if decision == "linked" else "provisional",
                )
            self._upsert_alias(session, mention, place, decision)
            if top.provider_place is not None and self._persist_provider_enrichment:
                self._persist_enrichment(session, place, top.provider_place)

        resolution = PlaceResolution(
            run_id=mention.run_id,
            mention_id=mention.id,
            place_id=place.id if place else None,
            candidate_id=selected_candidate.id if selected_candidate else None,
            decision=decision,
            confidence=top.score if top else 0.0,
            margin=margin,
            reason=_resolution_reason(decision, top, margin),
            resolver_version=self._config.resolver_version,
        )
        session.add(resolution)
        return resolution

    def _collect_candidates(
        self,
        session: Session,
        *,
        mention: PlaceMention,
        project_id: str,
        target_person_id: str,
    ) -> list[ResolutionCandidate]:
        normalized = normalize_place_alias(mention.raw_text)
        candidates: dict[str, ResolutionCandidate] = {}
        aliases = session.execute(
            select(PlaceAlias, PlaceEntity)
            .join(PlaceEntity, PlaceEntity.id == PlaceAlias.place_id)
            .where(
                PlaceAlias.project_id == project_id,
                PlaceAlias.normalized_alias == normalized,
                PlaceEntity.status.in_({"provisional", "active", "ambiguous"}),
            )
        ).all()
        for alias, place in aliases:
            candidate = ResolutionCandidate(
                key=f"place:{place.id}",
                source="personal_graph",
                canonical_name=place.canonical_name,
                place_type=place.place_type,
                place_id=place.id,
                coordinate=_entity_coordinate(place),
                administrative=place.address_components,
                features={
                    "exact_alias": 1.0,
                    "name_similarity": _name_similarity(mention.raw_text, place.canonical_name),
                    "subject_ownership": (
                        1.0 if place.owner_person_id == mention.subject_id else 0.5
                    ),
                    "type_compatibility": _type_compatibility(
                        mention.mention_type,
                        place.place_type,
                    ),
                    "recent_context": min(1.0, alias.confidence),
                },
            )
            candidates[candidate.key] = candidate

        recent = self._recent_context_candidate(session, mention)
        if recent is not None:
            candidates.setdefault(recent.key, recent)

        message = session.get(Message, mention.message_id)
        structured = extract_structured_location(message.raw) if message is not None else None
        if structured is not None:
            coordinate = Coordinate(structured.longitude, structured.latitude, "GCJ-02")
            key = f"coordinate:{structured.longitude:.6f}:{structured.latitude:.6f}"
            candidates[key] = ResolutionCandidate(
                key=key,
                source="structured_location",
                canonical_name=structured.label or structured.address or mention.raw_text,
                place_type="poi" if structured.label else "coordinate",
                coordinate=coordinate,
                features={
                    "exact_alias": 1.0 if structured.label == mention.raw_text else 0.6,
                    "name_similarity": 1.0,
                    "subject_ownership": 1.0,
                    "type_compatibility": 1.0,
                    "structured_coordinate": 1.0,
                },
            )

        # Every extracted physical-place span is sent to the configured map
        # provider. Generic anchors such as “家” may still remain semantic when
        # the provider response lacks enough evidence, but the provider call is
        # never delegated to or selected by the language model.
        city_hint, adcode_hint, near = self._geographic_hints(session, mention)
        try:
            provider_places = self._provider.search_poi(
                mention.raw_text,
                city_hint=city_hint,
                adcode_hint=adcode_hint,
                near=near,
                category=None,
                limit=5,
            )
        except (RuntimeError, OSError):
            provider_places = []
        for provider_place in provider_places:
            candidate = _provider_candidate(
                mention,
                provider_place,
                target_person_id,
                city_hint=city_hint,
                adcode_hint=adcode_hint,
                near=near,
            )
            candidates.setdefault(candidate.key, candidate)
        return list(candidates.values())

    def _geographic_hints(
        self,
        session: Session,
        mention: PlaceMention,
    ) -> tuple[str | None, str | None, Coordinate | None]:
        city_hint: str | None = None
        adcode_hint: str | None = None
        near: Coordinate | None = None
        context_mentions = list(
            session.scalars(
                select(PlaceMention).where(
                    PlaceMention.bundle_id == mention.bundle_id,
                    PlaceMention.mention_type == "administrative",
                    PlaceMention.id != mention.id,
                )
            )
        )
        if context_mentions:
            city_hint = context_mentions[0].raw_text
        recent = session.execute(
            select(PlaceEntity)
            .join(PlaceResolution, PlaceResolution.place_id == PlaceEntity.id)
            .join(PlaceMention, PlaceMention.id == PlaceResolution.mention_id)
            .where(
                PlaceMention.run_id == mention.run_id,
                PlaceMention.subject_id == mention.subject_id,
                PlaceMention.created_at <= mention.created_at,
                PlaceResolution.place_id.is_not(None),
            )
            .order_by(PlaceMention.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if recent is not None:
            near = _entity_coordinate(recent)
            city_value = recent.address_components.get("cityname") or recent.address_components.get(
                "city"
            )
            adcode_value = recent.address_components.get("adcode")
            if isinstance(city_value, str) and city_value.strip():
                city_hint = city_value.strip()
            if isinstance(adcode_value, str) and adcode_value.strip():
                adcode_hint = adcode_value.strip()
        return city_hint, adcode_hint, near

    def _recent_context_candidate(
        self,
        session: Session,
        mention: PlaceMention,
    ) -> ResolutionCandidate | None:
        if mention.mention_type not in {"relative_place", "deictic_place"}:
            return None
        row = session.execute(
            select(PlaceResolution, PlaceEntity)
            .join(PlaceMention, PlaceMention.id == PlaceResolution.mention_id)
            .join(PlaceEntity, PlaceEntity.id == PlaceResolution.place_id)
            .where(
                PlaceMention.run_id == mention.run_id,
                PlaceMention.subject_id == mention.subject_id,
                PlaceMention.created_at <= mention.created_at,
                PlaceResolution.place_id.is_not(None),
            )
            .order_by(PlaceMention.created_at.desc())
            .limit(1)
        ).first()
        if row is None:
            return None
        _resolution, place = row
        return ResolutionCandidate(
            key=f"place:{place.id}",
            source="recent_context",
            canonical_name=place.canonical_name,
            place_type=place.place_type,
            place_id=place.id,
            coordinate=_entity_coordinate(place),
            features={
                "exact_alias": 0.8,
                "subject_ownership": 1.0 if place.owner_person_id == mention.subject_id else 0.4,
                "type_compatibility": 0.8,
                "recent_context": 1.0,
                "name_similarity": 0.3,
            },
        )

    @staticmethod
    def _semantic_candidate(
        mention: PlaceMention,
        target_person_id: str,
    ) -> ResolutionCandidate:
        owner = mention.subject_id or target_person_id
        return ResolutionCandidate(
            key=f"semantic:{owner}:{normalize_place_alias(mention.raw_text)}",
            source="semantic",
            canonical_name=mention.raw_text.strip(),
            place_type=mention.mention_type,
            features={
                "exact_alias": 1.0,
                "name_similarity": 1.0,
                "subject_ownership": 1.0,
                "type_compatibility": 1.0,
            },
        )

    def _persist_candidate(
        self,
        session: Session,
        mention: PlaceMention,
        candidate: ResolutionCandidate,
        rank: int,
    ) -> PlaceCandidate:
        may_store_provider_data = (
            candidate.source != "map_provider" or self._persist_provider_enrichment
        )
        row = PlaceCandidate(
            mention_id=mention.id,
            candidate_key=candidate.key,
            candidate_source=candidate.source,
            canonical_name=(
                candidate.canonical_name if may_store_provider_data else mention.raw_text
            ),
            place_type=(
                candidate.place_type if may_store_provider_data else mention.mention_type
            ),
            latitude=(
                candidate.coordinate.latitude
                if may_store_provider_data and candidate.coordinate
                else None
            ),
            longitude=(
                candidate.coordinate.longitude
                if may_store_provider_data and candidate.coordinate
                else None
            ),
            coordinate_crs=(
                candidate.coordinate.crs
                if may_store_provider_data and candidate.coordinate
                else None
            ),
            address_components=(candidate.administrative if may_store_provider_data else {}),
            provider_payload_hash=(
                candidate.provider_place.raw_hash if candidate.provider_place else None
            ),
            feature_scores=candidate.features,
            contradictions=list(candidate.contradictions),
            raw_score=candidate.score,
            confidence=candidate.score,
            rank=rank,
        )
        session.add(row)
        session.flush()
        return row

    def _create_place(
        self,
        session: Session,
        *,
        project_id: str,
        target_person_id: str,
        mention: PlaceMention,
        candidate: ResolutionCandidate,
        status: str,
    ) -> PlaceEntity:
        role = _role_for_mention(mention)
        # The selected POI's identity and coordinate are core place-graph data.
        # `persist_provider_enrichment` only controls the separate raw provider
        # enrichment record, not whether the heatmap can retain its coordinate.
        place = PlaceEntity(
            project_id=project_id,
            owner_person_id=(mention.subject_id or target_person_id),
            canonical_name=candidate.canonical_name,
            place_type=candidate.place_type,
            role_distribution={role: 1.0} if role else {},
            geo_resolution=_geo_resolution(candidate),
            latitude=(
                candidate.coordinate.latitude
                if candidate.coordinate
                else None
            ),
            longitude=(
                candidate.coordinate.longitude
                if candidate.coordinate
                else None
            ),
            coordinate_crs=(
                candidate.coordinate.crs
                if candidate.coordinate
                else None
            ),
            address_components=candidate.administrative,
            field_provenance={
                "canonical_name": {"source": "chat_evidence", "mention_id": mention.id},
                "coordinate": {"source": candidate.source},
            },
            status=status,
        )
        session.add(place)
        session.flush()
        return place

    @staticmethod
    def _upsert_alias(
        session: Session,
        mention: PlaceMention,
        place: PlaceEntity,
        decision: str,
    ) -> None:
        normalized = normalize_place_alias(mention.raw_text)
        alias = session.scalar(
            select(PlaceAlias).where(
                PlaceAlias.project_id == mention.project_id,
                PlaceAlias.place_id == place.id,
                PlaceAlias.normalized_alias == normalized,
                PlaceAlias.speaker_id == mention.speaker_id,
            )
        )
        if alias is None:
            confidence = 0.85 if decision == "linked" else 0.6
            session.add(
                PlaceAlias(
                    project_id=mention.project_id,
                    place_id=place.id,
                    raw_alias=mention.raw_text,
                    normalized_alias=normalized,
                    alias_kind=(
                        "role_anchor"
                        if mention.mention_type == "personal_anchor"
                        else "deictic"
                        if mention.mention_type == "deictic_place"
                        else "stable_name"
                    ),
                    speaker_id=mention.speaker_id,
                    perspective=mention.subject_resolution,
                    supporting_mention_ids=[mention.id],
                    support_quality=mention.confidence,
                    confidence=confidence,
                    retrieval_eligible=False,
                )
            )
            return
        if mention.id not in alias.supporting_mention_ids:
            alias.supporting_mention_ids = [*alias.supporting_mention_ids, mention.id]
            alias.support_quality += mention.confidence
            alias.confidence = min(0.98, 1 - math.exp(-alias.support_quality))
            alias.retrieval_eligible = (
                len(set(alias.supporting_mention_ids)) >= 2 and alias.confidence >= 0.7
            )

    @staticmethod
    def _persist_enrichment(
        session: Session,
        place: PlaceEntity,
        provider: MapPlace,
    ) -> None:
        request_hash = provider.raw_hash or provider.provider_place_id
        session.add(
            PlaceProviderEnrichment(
                place_id=place.id,
                provider=provider.provider,
                provider_place_id=provider.provider_place_id,
                provider_name=provider.name,
                provider_address=provider.address,
                provider_category=provider.category,
                latitude=provider.coordinate.latitude if provider.coordinate else None,
                longitude=provider.coordinate.longitude if provider.coordinate else None,
                coordinate_crs=provider.coordinate.crs if provider.coordinate else None,
                administrative=provider.administrative,
                request_hash=request_hash,
                response_hash=request_hash,
                persistence_allowed=True,
            )
        )


def _resolution_phase(session: Session, mention: PlaceMention) -> int:
    message = session.get(Message, mention.message_id)
    if message is not None and extract_structured_location(message.raw) is not None:
        return 0
    if mention.mention_type in {"named_poi", "administrative"}:
        return 1
    if mention.mention_type == "personal_anchor":
        return 2
    return 3


def _provider_candidate(
    mention: PlaceMention,
    place: MapPlace,
    target_person_id: str,
    *,
    city_hint: str | None,
    adcode_hint: str | None,
    near: Coordinate | None,
) -> ResolutionCandidate:
    provider_reference_hash = sha256(
        f"{place.provider}:{place.provider_place_id}".encode()
    ).hexdigest()
    return ResolutionCandidate(
        key=f"provider_hash:{provider_reference_hash}",
        source="map_provider",
        canonical_name=place.name,
        place_type=place.category or mention.mention_type,
        coordinate=place.coordinate,
        administrative=place.administrative,
        provider_place=place,
        features={
            "exact_alias": (
                1.0
                if normalize_place_alias(mention.raw_text)
                == normalize_place_alias(place.name)
                else 0.0
            ),
            "name_similarity": _name_similarity(mention.raw_text, place.name),
            "subject_ownership": 1.0 if mention.subject_id == target_person_id else 0.5,
            "type_compatibility": 0.9,
            "geographic_context": _geographic_compatibility(
                place,
                city_hint=city_hint,
                adcode_hint=adcode_hint,
                near=near,
            ),
        },
    )


def _deterministic_provider_candidate(
    mention: PlaceMention,
    candidates: list[ResolutionCandidate],
    *,
    cluster_radius_meters: float = 500.0,
) -> ResolutionCandidate | None:
    """Accept an AMap POI only when its name strongly supports the chat span.

    AMap is still queried for every mention. This gate prevents generic text
    such as “家”“医院”“总部” from silently becoming AMap's default-city result.
    """

    if mention.mention_type not in {"named_poi", "administrative"}:
        return None
    provider_candidates = [
        item
        for item in candidates
        if item.source == "map_provider" and item.coordinate is not None
    ]
    if not provider_candidates:
        return None
    top = max(provider_candidates, key=lambda item: (item.score, item.key))
    exact_alias = top.features.get("exact_alias", 0.0) >= 1.0
    geographic_context = top.features.get("geographic_context", 0.0)
    if exact_alias and geographic_context >= 0.8:
        return top
    if exact_alias and len(provider_candidates) == 1 and len(mention.raw_text.strip()) >= 4:
        return top
    if top.features.get("name_similarity", 0.0) < 0.70 or top.coordinate is None:
        return None
    nearby = [
        item
        for item in provider_candidates
        if item.coordinate is not None
        and item.coordinate.crs == top.coordinate.crs
        and _haversine_meters(top.coordinate, item.coordinate) <= cluster_radius_meters
    ]
    if len(nearby) >= 2:
        return top
    return None


def _name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_place_alias(left), normalize_place_alias(right)).ratio()


def _type_compatibility(mention_type: str, place_type: str) -> float:
    if mention_type == place_type:
        return 1.0
    if mention_type == "named_poi" and place_type not in {"administrative", "coordinate"}:
        return 0.8
    return 0.4


def _entity_coordinate(place: PlaceEntity) -> Coordinate | None:
    if place.longitude is None or place.latitude is None or place.coordinate_crs is None:
        return None
    return Coordinate(place.longitude, place.latitude, place.coordinate_crs)


def _geographic_compatibility(
    place: MapPlace,
    *,
    city_hint: str | None,
    adcode_hint: str | None,
    near: Coordinate | None,
) -> float:
    candidate_adcode = place.administrative.get("adcode")
    if adcode_hint and isinstance(candidate_adcode, str):
        same_area = candidate_adcode.startswith(adcode_hint) or adcode_hint.startswith(
            candidate_adcode
        )
        return 1.0 if same_area else 0.0
    if city_hint:
        names = "".join(
            str(place.administrative.get(key, ""))
            for key in ("pname", "cityname", "adname")
        )
        if normalize_place_alias(city_hint) in normalize_place_alias(names):
            return 1.0
    if near is not None and place.coordinate is not None and near.crs == place.coordinate.crs:
        return max(0.0, 1.0 - _haversine_meters(near, place.coordinate) / 20_000)
    return 0.0


def _haversine_meters(left: Coordinate, right: Coordinate) -> float:
    radius = 6_371_000.0
    left_latitude = math.radians(left.latitude)
    right_latitude = math.radians(right.latitude)
    latitude_delta = math.radians(right.latitude - left.latitude)
    longitude_delta = math.radians(right.longitude - left.longitude)
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(left_latitude)
        * math.cos(right_latitude)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def _role_for_mention(mention: PlaceMention) -> str | None:
    text = normalize_place_alias(mention.raw_text)
    if text in {"家", "我家", "你家", "妈妈家", "爸妈家"}:
        return "home"
    if text in {"公司", "单位", "办公室", "新单位", "老单位"}:
        return "workplace"
    if text in {"学校", "大学", "中学", "小学"}:
        return "school"
    return None


def _geo_resolution(candidate: ResolutionCandidate) -> str:
    if candidate.coordinate is not None:
        return "coordinate" if candidate.source == "structured_location" else "poi"
    if candidate.place_type == "administrative":
        return "district"
    return "semantic_only"


def _resolution_reason(
    decision: str,
    candidate: ResolutionCandidate | None,
    margin: float,
) -> str:
    if candidate is None:
        return "没有满足条件的已有地点、上下文或地图候选"
    return (
        f"decision={decision}; source={candidate.source}; "
        f"score={candidate.score:.3f}; margin={margin:.3f}"
    )
