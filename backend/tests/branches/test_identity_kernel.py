from pathlib import Path

import pytest
from moonlightbox.db import Database
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


def test_identity_kernel_is_evidence_backed_and_immutable(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_models import IdentityKernel
    from moonlightbox.branches.identity import (
        IdentityKernelLockedError,
        IdentityKernelProposal,
        IdentityKernelService,
    )

    database = Database(f"sqlite:///{tmp_path / 'identity.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="人格内核"))
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="qwen",
                adapter_path="/tmp/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()
        service = IdentityKernelService(session)
        proposal = IdentityKernelProposal(
            persona="她",
            values=["真诚"],
            stable_preferences=["喜欢直接表达"],
            relationship_boundaries=["不接受被替自己定义感受"],
            language_patterns=["简短自然"],
            typical_reactions=["不确定时会直接澄清"],
        )

        kernel = service.create_and_lock(
            project_id="project-1",
            model_version_id="model-1",
            proposal=proposal,
            evidence_message_ids=["m1", "m2"],
        )

        assert kernel.locked_at is not None
        assert kernel.content["persona"] == "她"
        assert kernel.evidence_message_ids == ["m1", "m2"]
        assert kernel.field_evidence["values"] == ["m1", "m2"]
        assert kernel.field_confidence["values"] > 0
        assert session.query(IdentityKernel).count() == 1
        with pytest.raises(IdentityKernelLockedError):
            service.replace(kernel.id, proposal.model_copy(update={"persona": "另一人"}))
    database.close()


def test_identity_kernel_builder_derives_patterns_from_real_targets() -> None:
    from datetime import UTC, datetime

    from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample

    examples = [
        TrainingExample(
            messages=[
                ChatTurn(role="user", content="在吗"),
                ChatTurn(
                    role="assistant",
                    content='{"bubbles":[{"type":"text","content":"在呀，怎么啦"}]}',
                ),
            ],
            source_ids=["u1", "a1"],
            target_at=datetime.now(UTC),
        ),
        TrainingExample(
            messages=[
                ChatTurn(role="user", content="你必须答应我"),
                ChatTurn(
                    role="assistant",
                    content='{"bubbles":[{"type":"text","content":"不要，我不想这样"}]}',
                ),
            ],
            source_ids=["u2", "a2"],
            target_at=datetime.now(UTC),
        ),
    ]

    proposal = EvidenceBackedIdentityKernelBuilder().build(
        persona="她",
        examples=examples,
        event_contexts=[],
    )

    assert any("平均" in pattern for pattern in proposal.language_patterns)
    assert any("拒绝" in reaction for reaction in proposal.typical_reactions)
    assert "哒" in proposal.style_profile["forbidden_unobserved_markers"]
    assert proposal.field_evidence["language_patterns"] == ["u1", "a1", "u2", "a2"]
    assert all(
        not source_id.startswith("policy:")
        for source_ids in proposal.field_evidence.values()
        for source_id in source_ids
    )
    assert proposal.behavioral_rhythm["sample_count"] == 2
    assert proposal.behavioral_rhythm["derivation"] == (
        "authentic_target_chat_examples_only"
    )


def test_identity_kernel_derives_active_hours_and_proactive_propensity() -> None:
    from datetime import UTC, datetime

    from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample

    examples = [
        TrainingExample(
            messages=[
                ChatTurn(role="user", content="上下文"),
                ChatTurn(role="assistant", content="真人回复"),
            ],
            source_ids=[f"u{index}", f"a{index}"],
            target_at=datetime(2026, 8, 1 + index % 3, 21 + index % 2, tzinfo=UTC),
            conversation_mode=("proactive" if index < 3 else "responsive"),
        )
        for index in range(12)
    ]

    proposal = EvidenceBackedIdentityKernelBuilder().build(
        persona="她",
        examples=examples,
        event_contexts=[],
    )
    rhythm = proposal.behavioral_rhythm

    assert rhythm["sample_count"] == 12
    assert rhythm["active_hours"] == [21, 22]
    assert rhythm["proactive_turn_count"] == 3
    assert rhythm["proactive_turn_rate"] == 0.285714
    assert rhythm["timezone_offset_minutes"] == 0


def test_identity_kernel_extracts_only_explicit_authentic_preferences() -> None:
    from datetime import UTC, datetime

    from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample

    examples = [
        TrainingExample(
            messages=[
                ChatTurn(role="user", content="喜欢吃什么"),
                ChatTurn(role="assistant", content="我喜欢吃火锅，也不喜欢香菜"),
            ],
            source_ids=["u1", "a1"],
            target_at=datetime.now(UTC),
        ),
        TrainingExample(
            messages=[
                ChatTurn(role="user", content="系统测试"),
                ChatTurn(role="assistant", content="我喜欢凭空添加的东西"),
            ],
            source_ids=["policy:synthetic", "a2"],
            target_at=datetime.now(UTC),
            kind="policy_augmented",
        ),
    ]

    proposal = EvidenceBackedIdentityKernelBuilder().build(
        persona="她",
        examples=examples,
        event_contexts=[],
    )

    assert proposal.expressed_preferences == ["我喜欢吃火锅", "不喜欢香菜"]
    assert proposal.field_evidence["expressed_preferences"] == ["u1", "a1"]
    assert "凭空添加" not in "".join(proposal.expressed_preferences)


def test_naive_historical_timestamps_use_explicit_project_timezone_default() -> None:
    from datetime import datetime

    from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample

    example = TrainingExample(
        messages=[
            ChatTurn(role="user", content="在吗"),
            ChatTurn(role="assistant", content="在"),
        ],
        source_ids=["u1", "a1"],
        target_at=datetime(2026, 8, 5, 21),
    )

    rhythm = EvidenceBackedIdentityKernelBuilder(
        default_timezone_offset_minutes=480
    ).build(
        persona="她",
        examples=[example],
        event_contexts=[],
    ).behavioral_rhythm

    assert rhythm["timezone_offset_minutes"] == 480
    assert rhythm["timezone_derivation"] == (
        "configured_default_for_naive_source_timestamps"
    )


def test_identity_kernel_extracts_only_text_from_compact_protocol() -> None:
    from datetime import UTC, datetime

    from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample

    example = TrainingExample(
        messages=[
            ChatTurn(role="user", content="在吗"),
            ChatTurn(
                role="assistant",
                content=(
                    "<bubble>在呀</bubble><delay>60000</delay>"
                    "<sticker>asset-1</sticker><delay>0</delay>"
                ),
            ),
        ],
        source_ids=["u1", "a1"],
        target_at=datetime.now(UTC),
    )

    proposal = EvidenceBackedIdentityKernelBuilder().build(
        persona="她",
        examples=[example],
        event_contexts=[],
    )

    assert proposal.style_profile["sample_count"] == 1
    assert proposal.style_profile["average_length"] == 2.0
    assert "真实回复平均约 2 个字符" in proposal.language_patterns


def test_identity_kernel_creation_is_idempotent(tmp_path: Path) -> None:
    from moonlightbox.branches.identity import (
        IdentityKernelProposal,
        IdentityKernelService,
    )

    database = Database(f"sqlite:///{tmp_path / 'identity-idempotent.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="人格内核"))
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="qwen",
                adapter_path="/tmp/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()
        service = IdentityKernelService(session)
        proposal = IdentityKernelProposal(
            persona="她",
            values=["真诚"],
            stable_preferences=[],
            relationship_boundaries=[],
            language_patterns=[],
            typical_reactions=[],
        )

        first = service.create_and_lock(
            project_id="project-1",
            model_version_id="model-1",
            proposal=proposal,
            evidence_message_ids=["m1"],
        )
        second = service.create_and_lock(
            project_id="project-1",
            model_version_id="model-1",
            proposal=proposal,
            evidence_message_ids=["m1"],
        )

        assert first.id == second.id
    database.close()
