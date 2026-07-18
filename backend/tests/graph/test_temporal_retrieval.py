from datetime import datetime


class FakeVectorIndex:
    def search(self, _query: str, _limit: int) -> list[tuple[str, float]]:
        return [("past", 0.8), ("future", 0.99)]


def test_retrieval_never_returns_future_facts() -> None:
    from moonlightbox.graph.models import TemporalFact
    from moonlightbox.graph.retrieval import GraphRagRetriever

    facts = [
        TemporalFact(
            id="past",
            subject="甲",
            predicate="关系",
            object="朋友",
            valid_from=datetime(2025, 1, 1),
            valid_to=None,
            evidence_ids=["m1"],
        ),
        TemporalFact(
            id="future",
            subject="甲",
            predicate="关系",
            object="恋人",
            valid_from=datetime(2026, 7, 1),
            valid_to=None,
            evidence_ids=["m99"],
        ),
    ]
    retriever = GraphRagRetriever(facts, [], FakeVectorIndex())

    results = retriever.retrieve("他们是什么关系", datetime(2026, 1, 1))

    assert [result.fact.id for result in results] == ["past"]
    assert results[0].fact.evidence_ids == ["m1"]


def test_graph_neighbors_are_combined_with_vector_results() -> None:
    from moonlightbox.graph.models import EventEdge, TemporalFact
    from moonlightbox.graph.retrieval import GraphRagRetriever

    class EmptyVectorIndex:
        def search(self, _query: str, _limit: int) -> list[tuple[str, float]]:
            return []

    fact = TemporalFact(
        id="relationship",
        subject="甲",
        predicate="关系",
        object="朋友",
        valid_from=datetime(2025, 1, 1),
        valid_to=None,
        evidence_ids=["m1"],
    )
    edge = EventEdge("event-1", "relationship", "updates", ["m1"])
    retriever = GraphRagRetriever([fact], [edge], EmptyVectorIndex())

    results = retriever.retrieve(
        "关系",
        datetime(2026, 1, 1),
        seed_ids={"event-1"},
    )

    assert results[0].fact.id == "relationship"
    assert results[0].sources == ["graph"]
