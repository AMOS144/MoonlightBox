from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import chromadb
import yaml
from chromadb.api.types import Embeddings

from moonlightbox.branches.embeddings import TextEmbedder
from moonlightbox.spatial.episodes import EpisodeManifest


@dataclass(frozen=True, slots=True)
class SpatialQuery:
    query_id: str
    family: str
    text: str


@dataclass(slots=True)
class EpisodeHit:
    episode_id: str
    time_partition: str
    max_similarity: float = 0.0
    query_matches: list[dict[str, object]] = field(default_factory=list)
    forced_reason: str | None = None


def load_query_bank(path: Path | None = None) -> tuple[str, list[SpatialQuery]]:
    source = path or Path(__file__).with_name("query_bank.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("空间 query bank 根节点必须是对象")
    version = payload.get("version")
    families = payload.get("families")
    if not isinstance(version, str) or not isinstance(families, dict):
        raise ValueError("空间 query bank 缺少 version 或 families")
    queries: list[SpatialQuery] = []
    for family, values in families.items():
        if not isinstance(family, str) or not isinstance(values, list):
            raise ValueError("空间 query family 格式无效")
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("空间 query 不能为空")
            queries.append(
                SpatialQuery(
                    query_id=f"{family}:{index + 1}",
                    family=family,
                    text=value.strip(),
                )
            )
    return version, queries


class SpatialEpisodeIndex:
    def __init__(
        self,
        *,
        chroma_dir: str,
        project_id: str,
        import_id: str,
        embedder: TextEmbedder,
    ) -> None:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection_name = (
            f"spatial_{project_id.replace('-', '_')}_{import_id.replace('-', '_')}"
        )[:63]
        self._embedder = embedder
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace(self, episodes: list[EpisodeManifest]) -> None:
        current_ids = set(self._collection.get(include=[])["ids"])
        desired_ids = {episode.id for episode in episodes}
        obsolete = sorted(current_ids - desired_ids)
        if obsolete:
            self._collection.delete(ids=obsolete)
        if not episodes:
            return
        self._collection.upsert(
            ids=[episode.id for episode in episodes],
            documents=[episode.index_text for episode in episodes],
            embeddings=cast(
                Embeddings,
                self._embedder.embed([episode.index_text for episode in episodes]),
            ),
            metadatas=[
                {
                    "time_partition": episode.time_partition,
                    "started_at": episode.started_at.isoformat(),
                    "ended_at": episode.ended_at.isoformat(),
                    "content_hash": episode.content_hash,
                }
                for episode in episodes
            ],
        )

    def retrieve(
        self,
        episodes: list[EpisodeManifest],
        queries: list[SpatialQuery],
        *,
        top_k_per_partition: int,
        minimum_similarity: float,
    ) -> list[EpisodeHit]:
        by_id = {episode.id: episode for episode in episodes}
        hits: dict[str, EpisodeHit] = {}
        collection_count = self._collection.count()
        if queries and collection_count:
            embeddings = cast(
                Embeddings,
                self._embedder.embed([query.text for query in queries]),
            )
            for partition in sorted({episode.time_partition for episode in episodes}):
                partition_count = sum(
                    episode.time_partition == partition for episode in episodes
                )
                if partition_count == 0:
                    continue
                result = self._collection.query(
                    query_embeddings=embeddings,
                    n_results=min(top_k_per_partition, partition_count),
                    where={"time_partition": partition},
                    include=["distances"],
                )
                result_ids = _nested(result.get("ids"))
                distances = _nested(result.get("distances"))
                for query, ids, query_distances in zip(
                    queries,
                    result_ids,
                    distances,
                    strict=False,
                ):
                    for episode_id, distance in zip(ids, query_distances, strict=False):
                        similarity = 1.0 - float(distance)
                        if similarity < minimum_similarity or episode_id not in by_id:
                            continue
                        hit = hits.setdefault(
                            str(episode_id),
                            EpisodeHit(
                                episode_id=str(episode_id),
                                time_partition=partition,
                            ),
                        )
                        hit.max_similarity = max(hit.max_similarity, similarity)
                        hit.query_matches.append(
                            {
                                "query_id": query.query_id,
                                "family": query.family,
                                "similarity": similarity,
                            }
                        )
        for episode in episodes:
            if not episode.has_structured_location:
                continue
            hit = hits.setdefault(
                episode.id,
                EpisodeHit(
                    episode_id=episode.id,
                    time_partition=episode.time_partition,
                ),
            )
            hit.forced_reason = "structured_location"
        return sorted(
            hits.values(),
            key=lambda item: (by_id[item.episode_id].started_at, item.episode_id),
        )


def _nested(value: object) -> list[list[Any]]:
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, list) else [] for item in value]

