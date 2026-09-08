from datetime import datetime
from pathlib import Path


def test_memory_index_applies_id_distance_and_time_gates() -> None:
    from moonlightbox.branches.memory_index import TimeSafeMemoryIndex

    class Collection:
        def query(self, **_kwargs: object) -> dict[str, object]:
            return {
                "ids": [["valid", "future", "far"]],
                "distances": [[0.2, 0.1, 0.9]],
                "metadatas": [
                    [
                        {"timestamp": "2026-01-01T00:00:00"},
                        {"timestamp": "2027-01-01T00:00:00"},
                        {"timestamp": "2026-01-01T00:00:00"},
                    ]
                ],
            }

    result = TimeSafeMemoryIndex(Collection()).search(
        "旅行",
        allowed_ids={"valid", "future", "far"},
        cutoff=datetime(2026, 6, 1),
        limit=2,
    )

    assert result == ["valid"]


def test_chroma_project_index_filters_type_time_and_allowed_ids(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.memory_index import (
        ChromaProjectMemoryIndex,
        MemoryDocument,
    )

    class Embedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "旅行" in text else [0.0, 1.0] for text in texts]

    index = ChromaProjectMemoryIndex(
        chroma_dir=str(tmp_path / "chroma"),
        project_id="project-1",
        embedder=Embedder(),
    )
    index.upsert(
        [
            MemoryDocument(
                resource_id="exchange-1",
                resource_type="exchange",
                content="一起去杭州旅行",
                started_at=datetime(2026, 1, 1),
                ended_at=datetime(2026, 1, 2),
            ),
            MemoryDocument(
                resource_id="exchange-future",
                resource_type="exchange",
                content="以后一起旅行",
                started_at=datetime(2027, 1, 1),
                ended_at=datetime(2027, 1, 2),
            ),
        ]
    )

    result = index.query(
        "旅行",
        resource_type="exchange",
        allowed_ids={"exchange-1", "exchange-future"},
        cutoff=datetime(2026, 6, 1),
    )

    assert [item[0] for item in result] == ["exchange-1"]
