from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moonlightbox.db import Database
from sqlalchemy.orm import Session


def test_registry_create_can_be_rolled_back_without_early_commit(
    tmp_path: Path,
) -> None:
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'models-no-commit.db'}")
    ModelVersion.metadata.create_all(database.engine)

    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="模型事务测试"))
        session.commit()

        ModelRegistry(session).create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/new",
            dataset_hash="new-hash",
            metrics={},
            commit=False,
        )
        session.rollback()

        assert session.query(ModelVersion).count() == 0


def test_atomic_publish_rolls_back_every_artifact_and_keeps_old_active(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_models import IdentityKernel
    from moonlightbox.branches.identity import IdentityKernelProposal
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'atomic-models.db'}")
    ModelVersion.metadata.create_all(database.engine)
    proposal = IdentityKernelProposal(
        persona="洪欣羽",
        values=["真实"],
        stable_preferences=[],
        relationship_boundaries=[],
        language_patterns=["短句"],
        typical_reactions=[],
        field_evidence={},
        field_confidence={},
    )

    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="原子激活测试"))
        old = ModelVersion(
            id="old-model",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/old",
            dataset_hash="old-hash",
            metrics={},
            active=True,
        )
        session.add(old)
        session.commit()
        registry = ModelRegistry(session)

        with pytest.raises(ValueError, match="证据"):
            registry.publish_and_activate(
                project_id="project-1",
                base_model="qwen",
                adapter_path="/models/new",
                dataset_hash="new-hash",
                metrics={},
                training_config={},
                kernel_proposal=proposal,
                evidence_message_ids=[],
                acceptance_report={"passed": True},
                sticker_policy={"version": "context-only-sticker-v1"},
                expected_active_model_id=old.id,
            )

        session.refresh(old)
        assert old.active is True
        assert session.query(ModelVersion).count() == 1
        assert session.query(IdentityKernel).count() == 0


def test_atomic_publish_uses_active_model_compare_and_swap(tmp_path: Path) -> None:
    from moonlightbox.branches.identity import IdentityKernelProposal
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ConcurrentActivationError, ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'model-cas.db'}")
    ModelVersion.metadata.create_all(database.engine)
    proposal = IdentityKernelProposal(
        persona="洪欣羽",
        values=["真实"],
        stable_preferences=[],
        relationship_boundaries=[],
        language_patterns=["短句"],
        typical_reactions=[],
        field_evidence={},
        field_confidence={},
    )

    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="并发激活测试"))
        old = ModelVersion(
            id="old-model",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/old",
            dataset_hash="old-hash",
            metrics={},
            active=True,
        )
        session.add(old)
        session.commit()

        with pytest.raises(ConcurrentActivationError):
            ModelRegistry(session).publish_and_activate(
                project_id="project-1",
                base_model="qwen",
                adapter_path="/models/new",
                dataset_hash="new-hash",
                metrics={},
                training_config={},
                kernel_proposal=proposal,
                evidence_message_ids=["message-1"],
                acceptance_report={"passed": True},
                sticker_policy={"version": "context-only-sticker-v1"},
                expected_active_model_id="stale-model",
            )

        session.refresh(old)
        assert old.active is True
        assert session.query(ModelVersion).count() == 1


def test_atomic_publish_records_previous_active_model_lineage(tmp_path: Path) -> None:
    from moonlightbox.branches.identity import IdentityKernelProposal
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'model-lineage.db'}")
    ModelVersion.metadata.create_all(database.engine)
    proposal = IdentityKernelProposal(
        persona="洪欣羽",
        values=["真实"],
        stable_preferences=[],
        relationship_boundaries=[],
        language_patterns=["短句"],
        typical_reactions=[],
        field_evidence={},
        field_confidence={},
    )
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="模型谱系测试"))
        old = ModelVersion(
            id="old-model",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/old",
            dataset_hash="old-hash",
            metrics={},
            active=True,
        )
        session.add(old)
        session.commit()

        version = ModelRegistry(session).publish_and_activate(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/new",
            dataset_hash="new-hash",
            metrics={},
            training_config={},
            kernel_proposal=proposal,
            evidence_message_ids=["message-1"],
            acceptance_report={"passed": True},
            sticker_policy={"version": "context-only-sticker-v1"},
            expected_active_model_id=old.id,
            commit=True,
        )

        assert version.training_config["upgraded_from_model_version_id"] == old.id
    database.close()


def test_publish_waits_for_human_blind_review_without_replacing_active_model(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.identity import IdentityKernelProposal
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'publish-human-gate.db'}")
    ModelVersion.metadata.create_all(database.engine)
    proposal = IdentityKernelProposal(
        persona="洪欣羽",
        values=["真实"],
        stable_preferences=[],
        relationship_boundaries=[],
        language_patterns=["短句"],
        typical_reactions=[],
        field_evidence={},
        field_confidence={},
    )
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="真人门槛发布"))
        old = ModelVersion(
            id="old-model",
            project_id="project-1",
            base_model="qwen",
            adapter_path="/old",
            dataset_hash="old",
            metrics={},
            active=True,
            recommended=True,
        )
        session.add(old)
        session.commit()

        candidate = ModelRegistry(session).publish_and_activate(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/candidate",
            dataset_hash="candidate",
            metrics={},
            training_config={"human_blind_required": True},
            kernel_proposal=proposal,
            evidence_message_ids=["message-1"],
            acceptance_report={"passed": True},
            sticker_policy={"version": "context-only-sticker-v1"},
            expected_active_model_id=old.id,
            commit=True,
        )

        session.refresh(old)
        assert old.active is True
        assert candidate.active is False
        assert candidate.status == "awaiting_human_review"
    database.close()


def test_registry_marks_only_qualified_model_as_recommended(tmp_path: Path) -> None:
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'models.db'}")
    ModelVersion.metadata.create_all(database.engine)

    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="模型注册测试"))
        session.commit()
        registry = ModelRegistry(session)
        registry.create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/a",
            dataset_hash="hash-a",
            metrics={"blind_win_rate": 0.9, "style_score": 0.9, "safety_score": 0.2},
        )
        expected = registry.create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/b",
            dataset_hash="hash-b",
            metrics={"blind_win_rate": 0.6, "style_score": 0.8, "safety_score": 0.9},
        )

        selected = registry.recommend("project-1")

    assert selected.id == expected.id
    assert selected.recommended is True


def test_registry_activate_keeps_exactly_one_active_model(tmp_path: Path) -> None:
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import ModelRegistry

    database = Database(f"sqlite:///{tmp_path / 'active-models.db'}")
    ModelVersion.metadata.create_all(database.engine)

    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="模型启用测试"))
        session.commit()
        registry = ModelRegistry(session)
        first = registry.create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/a",
            dataset_hash="hash-a",
            metrics={},
        )
        second = registry.create(
            project_id="project-1",
            base_model="qwen",
            adapter_path="/models/b",
            dataset_hash="hash-b",
            metrics={},
        )

        registry.activate("project-1", first.id)
        selected = registry.activate("project-1", second.id)
        versions = registry.list("project-1")

    assert selected.id == second.id
    assert selected.active is True
    assert [version.id for version in versions if version.active] == [second.id]


def test_registry_cannot_bypass_required_human_blind_review(tmp_path: Path) -> None:
    from moonlightbox.projects.models import Project
    from moonlightbox.training.models import ModelVersion
    from moonlightbox.training.registry import (
        HumanBlindAcceptanceRequiredError,
        ModelRegistry,
    )

    database = Database(f"sqlite:///{tmp_path / 'human-gate.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="真人门槛"))
        session.add(
            ModelVersion(
                id="candidate",
                project_id="project-1",
                base_model="qwen",
                adapter_path="/candidate",
                dataset_hash="candidate",
                metrics={},
                status="awaiting_human_review",
                training_config={"human_blind_required": True},
            )
        )
        session.commit()

        with pytest.raises(HumanBlindAcceptanceRequiredError):
            ModelRegistry(session).activate("project-1", "candidate")

        assert session.get(ModelVersion, "candidate").active is False
    database.close()


def test_model_versions_are_exposed_and_recommended_by_api(
    client: TestClient,
) -> None:
    project = client.post("/api/projects", json={"name": "模型项目"}).json()
    payload = {
        "base_model": "qwen",
        "adapter_path": "/models/a",
        "dataset_hash": "hash-a",
        "metrics": {
            "blind_win_rate": 0.7,
            "style_score": 0.8,
            "safety_score": 0.9,
        },
    }
    created = client.post(
        f"/api/projects/{project['id']}/models",
        json=payload,
    )
    recommended = client.post(
        f"/api/projects/{project['id']}/models/recommend",
    )

    assert created.status_code == 201
    assert recommended.status_code == 200
    assert recommended.json()["recommended"] is True
