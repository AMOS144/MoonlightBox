from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from moonlightbox.graph.models import EventEdge, TemporalFact


class VectorIndex(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class RetrievedFact:
    fact: TemporalFact
    score: float
    sources: list[str]


class GraphRagRetriever:
    def __init__(
        self,
        facts: list[TemporalFact],
        edges: list[EventEdge],
        vector_index: VectorIndex,
    ) -> None:
        self._facts = {fact.id: fact for fact in facts}
        self._edges = edges
        self._vector_index = vector_index

    def retrieve(
        self,
        query: str,
        cutoff: datetime,
        seed_ids: set[str] | None = None,
        limit: int = 10,
    ) -> list[RetrievedFact]:
        scores = dict(self._vector_index.search(query, limit * 3))
        sources = {fact_id: ["vector"] for fact_id in scores}

        for fact_id in self._expand(seed_ids or set()):
            scores[fact_id] = max(scores.get(fact_id, 0.0), 0.25)
            sources.setdefault(fact_id, []).append("graph")

        results = [
            RetrievedFact(fact=fact, score=scores[fact_id], sources=sources[fact_id])
            for fact_id, fact in self._facts.items()
            if fact_id in scores and _is_valid_at(fact, cutoff)
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _expand(self, seed_ids: set[str]) -> set[str]:
        neighbors: set[str] = set()
        for edge in self._edges:
            if edge.source_id in seed_ids:
                neighbors.add(edge.target_id)
            if edge.target_id in seed_ids:
                neighbors.add(edge.source_id)
        return neighbors


def _is_valid_at(fact: TemporalFact, cutoff: datetime) -> bool:
    return fact.valid_from <= cutoff and (
        fact.valid_to is None or cutoff < fact.valid_to
    )
