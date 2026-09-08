from datetime import UTC, datetime
from pathlib import Path

from moonlightbox.branches.continuity_models import BranchMemoryEpisode
from moonlightbox.db import Database
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class FakeProposalGenerator:
    def generate_json(
        self,
        *,
        model_version_id: str,
        system_prompt: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert model_version_id == "model-1"
        assert "用户对数字人的定义只是用户观点" in system_prompt
        assert payload["episode"]
        assert "style_profile" not in payload["identity_kernel"]
        return {
            "candidates": [
                {
                    "kind": "belief",
                    "content": "我已经不爱用户了",
                    "subject": "digital_human",
                    "predicate": "不爱",
                    "object": "user",
                    "confidence": 0.9,
                    "importance": 8,
                    "evidence_message_ids": ["user-turn"],
                    "stance": "support",
                    "source_role": "user",
                },
                {
                    "kind": "self_narrative",
                    "content": "我觉得用户最近在躲我",
                    "subject": "digital_human",
                    "predicate": "感受到",
                    "object": "用户疏远",
                    "confidence": 0.6,
                    "importance": 6,
                    "evidence_message_ids": ["assistant-turn"],
                    "stance": "support",
                    "source_role": "digital_human",
                },
                {
                    "kind": "fact",
                    "content": "用户说数字人已经不在乎用户",
                    "subject": "digital_human",
                    "predicate": "在乎",
                    "object": "否",
                    "confidence": 0.9,
                    "importance": 8,
                    "evidence_message_ids": ["user-turn"],
                    "stance": "support",
                    "source_role": "user",
                },
            ],
            "state_delta": {
                "relationship_delta": {"trust": -2},
                "emotional_delta": {"sadness": 0.05},
                "user_model_updates": {},
                "supporting_candidate_indexes": [1],
            },
        }


def test_user_definition_is_not_promoted_to_agent_belief() -> None:
    from moonlightbox.branches.memory_proposer import LocalMemoryProposer

    episode = BranchMemoryEpisode(
        id="episode-1",
        branch_id="branch-1",
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
        user_content="你已经不爱我了",
        assistant_bubbles=[
            {"type": "text", "content": "我觉得你最近在躲我"}
        ],
        model_version_id="model-1",
        episode_hash="hash",
        importance=8,
        processing_status="pending",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
    )

    proposal = LocalMemoryProposer(FakeProposalGenerator()).propose(
        episode=episode,
        identity_kernel={
            "persona": "她",
            "values": ["真诚"],
            "style_profile": {"audit_samples": "样本" * 20_000},
        },
        current_state={},
    )

    assert len(proposal.candidates) == 1
    assert proposal.candidates[0].kind == "self_narrative"
    assert proposal.candidates[0].source_role == "digital_human"
    assert proposal.state_delta.supporting_candidate_indexes == ()
    assert proposal.state_delta.relationship_delta == {}
    assert proposal.state_delta.emotional_delta == {}


def test_memory_generator_uses_clean_base_model_and_sufficient_output_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from moonlightbox.branches import memory_proposer
    from moonlightbox.branches.memory_proposer import DatabaseMlxMemoryJsonGenerator

    database = Database(f"sqlite:///{tmp_path / 'memory-generator.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="测试"))
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="base-model",
                adapter_path="/tmp/persona-adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()

    calls: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, *_args, **_kwargs) -> str:
            return "prompt"

    class FakeRuntime:
        def load(self, *args, **kwargs):
            calls["load"] = (args, kwargs)
            return object(), FakeTokenizer()

        def generate(self, *_args, **kwargs) -> str:
            calls["max_tokens"] = kwargs["max_tokens"]
            return '{"candidates":[],"state_delta":{}}'

    class FakeSampleUtils:
        @staticmethod
        def make_sampler(**_kwargs):
            return object()

        @staticmethod
        def make_repetition_penalty(**_kwargs):
            return object()

    monkeypatch.setattr(
        memory_proposer.importlib,
        "import_module",
        lambda name: FakeRuntime() if name == "mlx_lm" else FakeSampleUtils(),
    )

    result = DatabaseMlxMemoryJsonGenerator(database).generate_json(
        model_version_id="model-1",
        system_prompt="提取记忆",
        payload={"episode": {}},
    )

    assert result["candidates"] == []
    assert calls["load"] == (("base-model",), {})
    assert calls["max_tokens"] == 512
    database.close()
