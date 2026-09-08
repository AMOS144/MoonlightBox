import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from moonlightbox.spatial.episodes import EpisodeManifest
from moonlightbox.spatial.retrieval import EpisodeHit


@dataclass(frozen=True, slots=True)
class BundleManifest:
    bundle_hash: str
    episode_ids: tuple[str, ...]
    candidate_episode_ids: tuple[str, ...]
    message_ids: tuple[str, ...]
    started_at: datetime
    ended_at: datetime


def build_bundles(
    episodes: list[EpisodeManifest],
    hits: list[EpisodeHit],
    *,
    neighbor_gap: timedelta,
    character_budget: int,
) -> list[BundleManifest]:
    if neighbor_gap <= timedelta(0) or character_budget <= 0:
        raise ValueError("Bundle 配置无效")
    ordered = sorted(episodes, key=lambda item: (item.started_at, item.id))
    positions = {episode.id: index for index, episode in enumerate(ordered)}
    candidate_ids = {hit.episode_id for hit in hits}
    ranges: list[tuple[int, int]] = []
    for hit in hits:
        position = positions.get(hit.episode_id)
        if position is None:
            continue
        start = position
        end = position
        previous_is_near = (
            position > 0
            and ordered[position].started_at - ordered[position - 1].ended_at <= neighbor_gap
        )
        if previous_is_near:
            start -= 1
        if (
            position + 1 < len(ordered)
            and ordered[position + 1].started_at - ordered[position].ended_at <= neighbor_gap
        ):
            end += 1
        ranges.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    bundles: list[BundleManifest] = []
    for start, end in merged:
        chunk: list[EpisodeManifest] = []
        characters = 0
        for episode in ordered[start : end + 1]:
            if chunk and characters + episode.character_count > character_budget:
                bundles.append(_bundle(chunk, candidate_ids))
                chunk = []
                characters = 0
            chunk.append(episode)
            characters += episode.character_count
        if chunk:
            bundles.append(_bundle(chunk, candidate_ids))
    return bundles


def _bundle(
    episodes: list[EpisodeManifest],
    candidate_ids: set[str],
) -> BundleManifest:
    episode_ids = tuple(item.id for item in episodes)
    message_ids = tuple(message_id for item in episodes for message_id in item.message_ids)
    bundle_hash = hashlib.sha256(
        json.dumps(
            {"episode_ids": episode_ids, "message_ids": message_ids},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return BundleManifest(
        bundle_hash=bundle_hash,
        episode_ids=episode_ids,
        candidate_episode_ids=tuple(item for item in episode_ids if item in candidate_ids),
        message_ids=message_ids,
        started_at=episodes[0].started_at,
        ended_at=episodes[-1].ended_at,
    )
