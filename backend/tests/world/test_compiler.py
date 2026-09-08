from typing import Any

from moonlightbox.world.client import (
    LightRAGMetadata,
    LightRAGReference,
    LightRAGRetrieval,
)
from moonlightbox.world.compiler import PROFILE_QUESTIONS, BackgroundCompiler
from moonlightbox.world.schemas import WorldProfileDraft


class FakeLightRAG:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    def query(self, workspace: str, query: str, **_: Any) -> LightRAGRetrieval:
        self.queries.append((workspace, query))
        return LightRAGRetrieval(
            context="小月说自己刚去了新公司。",
            references=[LightRAGReference(file_path="bundle_1.txt")],
            metadata=LightRAGMetadata(
                lightrag_version="1.5.6",
                embedding_model="embedding",
                embedding_dimension=3,
                extraction_model="extractor",
                chunking_strategy="fixed_token",
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
                entity_prompt_version="prompt-v1",
            ),
        )


class FakeStructuredCompiler:
    def create_structured_completion(self, **_: Any) -> WorldProfileDraft:
        statement = {
            "text": "在新公司工作",
            "source_status": "direct",
            "source_document_ids": ["bundle_1.txt", "invented.txt"],
        }
        return WorldProfileDraft.model_validate(
            {
                "identity": {
                    "names": [],
                    "aliases": [],
                    "self_descriptions": [],
                    "roles": [statement],
                },
                "work_and_education": [statement],
                "places": [],
                "social_relationships": [],
                "preferences": [],
                "recurring_activities": [],
                "routine_summary": {
                    "workdays": [],
                    "weekends": [],
                    "other_patterns": [],
                },
                "life_phases": [],
                "relationship_with_user": {"overview": [], "changes_over_time": []},
                "important_events": [],
                "unresolved_candidates": [],
            }
        )


def test_compiler_uses_fixed_mix_queries_and_maps_sources_to_messages() -> None:
    lightrag = FakeLightRAG()
    compiler = BackgroundCompiler(lightrag, FakeStructuredCompiler())  # type: ignore[arg-type]
    progress: list[tuple[int, int]] = []

    result = compiler.compile(
        workspace="world_1",
        project_id="project-1",
        subject_name="小月",
        user_name="我",
        source_messages_by_document={"bundle_1.txt": ["m1", "m2"]},
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(lightrag.queries) == len(PROFILE_QUESTIONS)
    assert all(workspace == "world_1" for workspace, _ in lightrag.queries)
    assert result.source_message_ids == ("m1", "m2")
    assert result.draft.work_and_education[0].source_document_ids == ["bundle_1.txt"]
    assert progress[-1] == (len(PROFILE_QUESTIONS), len(PROFILE_QUESTIONS))
