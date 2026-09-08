import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from moonlightbox.imports.models import Message, Participant


@dataclass(frozen=True, slots=True)
class EpisodeMessage:
    id: str
    timestamp: datetime
    participant_id: str
    participant_name: str
    participant_role: str
    kind: str
    content: str
    raw: dict[str, Any]
    media_asset_id: str | None


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    id: str
    project_id: str
    import_id: str
    message_ids: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    segmentation_version: str
    content_hash: str
    time_partition: str
    index_text: str
    character_count: int
    has_structured_location: bool
    participant_ids: tuple[str, ...] = ()
    media_asset_ids: tuple[str, ...] = ()
    input_hash: str = ""


@dataclass(frozen=True, slots=True)
class StructuredLocation:
    latitude: float
    longitude: float
    label: str | None
    address: str | None


def to_episode_message(message: Message, participant: Participant) -> EpisodeMessage:
    return EpisodeMessage(
        id=message.id,
        timestamp=message.timestamp,
        participant_id=participant.id,
        participant_name=participant.name,
        participant_role=participant.role,
        kind=message.kind,
        content=message.content,
        raw=message.raw,
        media_asset_id=message.media_asset_id,
    )


def build_episode_manifests(
    messages: list[EpisodeMessage],
    *,
    project_id: str,
    import_id: str,
    segmentation_version: str,
    episode_gap: timedelta,
    character_budget: int,
    message_limit: int,
) -> list[EpisodeManifest]:
    if episode_gap <= timedelta(0):
        raise ValueError("episode_gap 必须大于零")
    if character_budget <= 0 or message_limit <= 0:
        raise ValueError("Episode 预算必须大于零")
    ordered = sorted(messages, key=lambda item: (item.timestamp, item.id))
    groups: list[list[EpisodeMessage]] = []
    current: list[EpisodeMessage] = []
    current_characters = 0
    for message in ordered:
        rendered = render_message_for_index(message)
        should_split = bool(
            current
            and (
                message.timestamp - current[-1].timestamp > episode_gap
                or len(current) >= message_limit
                or current_characters + len(rendered) > character_budget
            )
        )
        if should_split:
            groups.append(current)
            current = []
            current_characters = 0
        current.append(message)
        current_characters += len(rendered)
    if current:
        groups.append(current)
    return [
        _manifest(
            group,
            project_id=project_id,
            import_id=import_id,
            segmentation_version=segmentation_version,
        )
        for group in groups
    ]


def render_message_for_index(message: EpisodeMessage) -> str:
    content = normalize_index_text(message.content)
    marker = structured_location_marker(message.raw)
    media = f" [媒体:{message.kind}]" if message.media_asset_id else ""
    marker_text = f" {marker}" if marker else ""
    return (
        f"[{message.id}] {message.timestamp.isoformat()} "
        f"{message.participant_name}({message.participant_role}): "
        f"{content}{marker_text}{media}"
    )


def normalize_index_text(value: str) -> str:
    normalized = value.replace("\u3000", " ")
    return re.sub(r"[ \t]+", " ", normalized).strip()


def structured_location_marker(raw: dict[str, Any]) -> str | None:
    location = extract_structured_location(raw)
    if location is None:
        return None
    latitude = location.latitude
    longitude = location.longitude
    label = location.label
    address = location.address
    parts = [f"经度={longitude:.6f}", f"纬度={latitude:.6f}"]
    if label:
        parts.append(f"名称={label}")
    if address:
        parts.append(f"地址={address}")
    return f"[结构化位置:{';'.join(parts)}]"


def extract_structured_location(raw: dict[str, Any]) -> StructuredLocation | None:
    found = _find_location_object(raw)
    if found is None:
        return None
    latitude, longitude, label, address = found
    return StructuredLocation(
        latitude=latitude,
        longitude=longitude,
        label=label,
        address=address,
    )


def _manifest(
    messages: list[EpisodeMessage],
    *,
    project_id: str,
    import_id: str,
    segmentation_version: str,
) -> EpisodeManifest:
    lines = [render_message_for_index(message) for message in messages]
    index_text = "\n".join(lines)
    message_ids = tuple(message.id for message in messages)
    identity = {
        "project_id": project_id,
        "import_id": import_id,
        "segmentation_version": segmentation_version,
        "message_ids": message_ids,
    }
    episode_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    content_hash = hashlib.sha256(index_text.encode()).hexdigest()
    input_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "id": message.id,
                    "timestamp": message.timestamp.isoformat(),
                    "participant_id": message.participant_id,
                    "kind": message.kind,
                    "content": message.content,
                    "raw": message.raw,
                    "media_asset_id": message.media_asset_id,
                }
                for message in messages
            ],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    return EpisodeManifest(
        id=episode_id,
        project_id=project_id,
        import_id=import_id,
        message_ids=message_ids,
        started_at=messages[0].timestamp,
        ended_at=messages[-1].timestamp,
        segmentation_version=segmentation_version,
        content_hash=content_hash,
        time_partition=messages[0].timestamp.strftime("%Y-%m"),
        index_text=index_text,
        character_count=len(index_text),
        has_structured_location=any(
            structured_location_marker(message.raw) is not None for message in messages
        ),
        participant_ids=tuple(dict.fromkeys(message.participant_id for message in messages)),
        media_asset_ids=tuple(
            dict.fromkeys(
                message.media_asset_id
                for message in messages
                if message.media_asset_id is not None
            )
        ),
        input_hash=input_hash,
    )


def _find_location_object(
    value: object,
    *,
    depth: int = 0,
) -> tuple[float, float, str | None, str | None] | None:
    if depth > 6:
        return None
    if isinstance(value, dict):
        latitude = _first_float(value, ("latitude", "lat", "y"))
        longitude = _first_float(value, ("longitude", "lng", "lon", "x"))
        if (
            latitude is not None
            and longitude is not None
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
            and _looks_like_location_container(value)
        ):
            return (
                latitude,
                longitude,
                _first_string(value, ("name", "title", "label", "poiname")),
                _first_string(value, ("address", "addr", "poiaddress")),
            )
        for key, child in value.items():
            if isinstance(child, dict | list):
                found = _find_location_object(child, depth=depth + 1)
                if found is not None and (
                    _location_key(str(key)) or _looks_like_location_container(child)
                ):
                    return found
    if isinstance(value, list):
        for child in value:
            found = _find_location_object(child, depth=depth + 1)
            if found is not None:
                return found
    return None


def _looks_like_location_container(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return any(
        _location_key(str(key))
        for key in value
    ) or any(key in value for key in ("address", "addr", "poiname"))


def _location_key(value: str) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in ("location", "position", "poi", "map"))


def _first_float(value: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        candidate = value.get(key)
        try:
            if candidate not in (None, ""):
                return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _first_string(value: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None
