from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast

import chromadb
from chromadb.api.types import Embeddings

from moonlightbox.branches.embeddings import TextEmbedder


class SemanticCollection(Protocol):
    def query(self, **kwargs: object) -> dict[str, Any]: ...


class TimeSafeMemoryIndex:
    def __init__(self, collection: SemanticCollection, *, max_distance: float = 0.8):
        self._collection = collection
        self._max_distance = max_distance

    def search(
        self,
        query: str,
        *,
        allowed_ids: set[str],
        cutoff: datetime,
        limit: int,
    ) -> list[str]:
        if not query.strip() or not allowed_ids:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=max(limit * 4, limit),
            include=["distances", "metadatas"],
        )
        ids = _first_list(result.get("ids"))
        distances = _first_list(result.get("distances"))
        metadatas = _first_list(result.get("metadatas"))
        selected: list[str] = []
        for item_id, distance, metadata in zip(ids, distances, metadatas, strict=False):
            if item_id not in allowed_ids or float(distance) > self._max_distance:
                continue
            timestamp = datetime.fromisoformat(str(metadata["timestamp"]))
            if timestamp > cutoff:
                continue
            selected.append(str(item_id))
            if len(selected) == limit:
                break
        return selected


def _first_list(value: object) -> list[Any]:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        return []
    return value[0]


@dataclass(frozen=True)
class MemoryDocument:
    resource_id: str
    resource_type: str
    content: str
    started_at: datetime
    ended_at: datetime


class ChromaProjectMemoryIndex:
    def __init__(
        self,
        *,
        chroma_dir: str,
        project_id: str,
        embedder: TextEmbedder,
    ) -> None:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection_name = f"memory_{project_id.replace('-', '_')}"
        self._embedder = embedder
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, documents: list[MemoryDocument]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[document.resource_id for document in documents],
            documents=[document.content for document in documents],
            embeddings=cast(
                Embeddings,
                self._embedder.embed([document.content for document in documents]),
            ),
            metadatas=[
                {
                    "resource_type": document.resource_type,
                    "started_at": document.started_at.isoformat(),
                    "ended_at": document.ended_at.isoformat(),
                }
                for document in documents
            ],
        )

    def replace(self, documents: list[MemoryDocument]) -> None:
        current_ids = set(self._collection.get(include=[])["ids"])
        desired_ids = {document.resource_id for document in documents}
        obsolete = sorted(current_ids - desired_ids)
        if obsolete:
            self._collection.delete(ids=obsolete)
        self.upsert(documents)

    def query(
        self,
        text: str,
        *,
        resource_type: str,
        allowed_ids: set[str],
        cutoff: datetime,
        limit: int = 2,
        minimum_similarity: float = 0.45,
    ) -> list[tuple[str, str, float]]:
        if not text.strip() or not allowed_ids:
            return []
        collection_count = self._collection.count()
        if collection_count == 0:
            return []
        result = self._collection.query(
            query_embeddings=cast(Embeddings, self._embedder.embed([text])),
            n_results=min(max(limit * 10, limit), collection_count),
            where={"resource_type": resource_type},
            include=["documents", "distances", "metadatas"],
        )
        ids = _first_list(result.get("ids"))
        documents = _first_list(result.get("documents"))
        distances = _first_list(result.get("distances"))
        metadatas = _first_list(result.get("metadatas"))
        selected: list[tuple[str, str, float]] = []
        for item_id, content, distance, metadata in zip(
            ids,
            documents,
            distances,
            metadatas,
            strict=False,
        ):
            if item_id not in allowed_ids:
                continue
            ended_at = datetime.fromisoformat(str(metadata["ended_at"]))
            if ended_at > cutoff:
                continue
            similarity = 1.0 - float(distance)
            if similarity < minimum_similarity:
                continue
            selected.append((str(item_id), str(content), similarity))
            if len(selected) == limit:
                break
        return selected
