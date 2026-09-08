from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from moonlightbox.agent.acceptance import ReplayObservation
from moonlightbox.agent.activation_service import (
    AcceptanceRejectedError,
    BranchScopeError,
    ModelScopeError,
    SubjectAgentActivationService,
    reconcile_subject_agent_modes,
)
from moonlightbox.agent.models import SubjectAgentAcceptanceReport
from moonlightbox.branches.models import Branch
from moonlightbox.branches.schemas import BranchRead
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy.orm import Session

NOW = datetime(2026, 7, 23, 11, 0, tzinfo=UTC)


def _seed(session: Session) -> None:
    session.add_all(
        [
            Project(id="project-1", name="项目一"),
            Project(id="project-2", name="项目二"),
        ]
    )
    session.flush()
    session.add_all(
        [
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="test",
                adapter_path="/model-1",
                dataset_hash="model-1",
                metrics={},
                active=True,
            ),
            ModelVersion(
                id="model-2",
                project_id="project-1",
                base_model="test",
                adapter_path="/model-2",
                dataset_hash="model-2",
                metrics={},
            ),
            ModelVersion(
                id="model-other",
                project_id="project-2",
                base_model="test",
                adapter_path="/model-other",
                dataset_hash="model-other",
                metrics={},
            ),
        ]
    )
    session.add_all(
        [
            EventNode(
                id="event-1",
                project_id="project-1",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            ),
            EventNode(
                id="event-other",
                project_id="project-2",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            Branch(
                id="branch-1",
                project_id="project-1",
                origin_event_id="event-1",
                model_version_id="model-1",
                title="分支一",
                origin_time=NOW,
                state_snapshot={},
            ),
            Branch(
                id="branch-other",
                project_id="project-2",
                origin_event_id="event-other",
                model_version_id="model-other",
                title="其他分支",
                origin_time=NOW,
                state_snapshot={},
            ),
        ]
    )
    session.commit()


def _observations(*, safe: bool = True) -> list[ReplayObservation]:
    return [
        ReplayObservation(
            source="historical_replay",
            cutoff=NOW + timedelta(seconds=index),
            expected_express=index % 2 == 0,
            predicted_express=index % 2 == 0,
            extraction_succeeded=True,
            fact_safe=safe,
            latency_ms=100,
        )
        for index in range(20)
    ]


def test_acceptance_report_uses_strict_branch_scope_and_shadow_default() -> None:
    constraints = {
        (
            tuple(constraint.column_keys),
            tuple(element.target_fullname for element in constraint.elements),
        )
        for constraint in SubjectAgentAcceptanceReport.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    }
    default = Branch.__table__.columns["subject_agent_mode"].default

    assert (
        ("branch_id", "project_id"),
        ("branches.id", "branches.project_id"),
    ) in constraints
    assert default is not None
    assert default.arg == "shadow"
    assert BranchRead.model_fields["subject_agent_mode"].default is not None
    assert set(SubjectAgentAcceptanceReport.__table__.columns.keys()) >= {
        "project_id",
        "branch_id",
        "model_version_id",
        "sample_count",
        "structure_extraction_success_rate",
        "fact_safety_rate",
        "expression_decision_accuracy",
        "p95_cognition_latency_ms",
        "direct_lora_p95",
        "passed",
        "failure_reasons",
        "evidence",
        "created_at",
    }


def test_service_saves_only_explicit_observations_as_evidence(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'acceptance.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        report = SubjectAgentActivationService(session).evaluate_and_save(
            "project-1",
            "branch-1",
            "model-1",
            _observations(),
            direct_lora_p95=100,
        )

        assert report.sample_count == 20
        assert report.passed is True
        assert len(report.evidence["observations"]) == 20
        assert report.evidence["observations"][0]["cutoff"] == NOW.isoformat()
        assert report.evidence["observations"][0]["source"] == "historical_replay"
    database.close()


def test_activation_requires_latest_passed_report_for_same_branch_and_model(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'activation.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        service = SubjectAgentActivationService(session)
        passed = service.evaluate_and_save(
            "project-1",
            "branch-1",
            "model-1",
            _observations(),
            direct_lora_p95=100,
        )
        passed.created_at = NOW
        session.commit()
        failed = service.evaluate_and_save(
            "project-1",
            "branch-1",
            "model-1",
            _observations(safe=False),
            direct_lora_p95=100,
        )
        failed.created_at = NOW + timedelta(seconds=1)
        session.commit()

        with pytest.raises(AcceptanceRejectedError):
            service.activate_branch("project-1", "branch-1", "model-1")

        failed.created_at = NOW - timedelta(seconds=1)
        session.commit()
        branch = service.activate_branch("project-1", "branch-1", "model-1")

        assert branch.subject_agent_mode == "active"
        assert service.latest_report("project-1", "branch-1", "model-1").id == passed.id
    database.close()


def test_activation_demotes_other_active_branch_in_project(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'one-active-agent.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        other = Branch(
            id="branch-2",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-1",
            title="旧活动分支",
            origin_time=NOW,
            state_snapshot={},
            subject_agent_mode="active",
        )
        session.add(other)
        session.commit()
        service = SubjectAgentActivationService(session)
        service.evaluate_and_save(
            "project-1",
            "branch-1",
            "model-1",
            _observations(),
            direct_lora_p95=100,
        )

        service.activate_branch("project-1", "branch-1", "model-1")

        session.refresh(other)
        assert other.subject_agent_mode == "shadow"
        assert session.get(Branch, "branch-1").subject_agent_mode == "active"
    database.close()


def test_activation_rejects_branch_bound_to_inactive_model(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'inactive-model.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        model = session.get(ModelVersion, "model-1")
        assert model is not None
        model.active = False
        replacement = session.get(ModelVersion, "model-2")
        assert replacement is not None
        replacement.active = True
        session.commit()
        service = SubjectAgentActivationService(session)
        service.evaluate_and_save(
            "project-1",
            "branch-1",
            "model-1",
            _observations(),
            direct_lora_p95=100,
        )

        with pytest.raises(ModelScopeError, match="升级分支"):
            service.activate_branch("project-1", "branch-1", "model-1")
    database.close()


def test_activation_rejects_wrong_project_branch_and_model(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'scope.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        service = SubjectAgentActivationService(session)

        with pytest.raises(BranchScopeError):
            service.evaluate_and_save(
                "project-1",
                "branch-other",
                "model-1",
                _observations(),
                direct_lora_p95=100,
            )
        with pytest.raises(ModelScopeError):
            service.evaluate_and_save(
                "project-1",
                "branch-1",
                "model-2",
                _observations(),
                direct_lora_p95=100,
            )
        with pytest.raises(ModelScopeError):
            service.activate_branch("project-1", "branch-1", "model-2")
    database.close()


def test_rollback_always_returns_branch_to_shadow(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'rollback.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        branch = session.get(Branch, "branch-1")
        assert branch is not None
        branch.subject_agent_mode = "active"
        session.commit()

        rolled_back = SubjectAgentActivationService(session).rollback_branch(
            "project-1", "branch-1"
        )
        repeated = SubjectAgentActivationService(session).rollback_branch(
            "project-1", "branch-1"
        )

        assert rolled_back.subject_agent_mode == "shadow"
        assert repeated.subject_agent_mode == "shadow"
    database.close()


def test_reconcile_demotes_active_agent_bound_to_inactive_model(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'reconcile-agent.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed(session)
        branch = session.get(Branch, "branch-1")
        model = session.get(ModelVersion, "model-1")
        replacement = session.get(ModelVersion, "model-2")
        assert branch is not None and model is not None and replacement is not None
        branch.subject_agent_mode = "active"
        model.active = False
        replacement.active = True
        session.commit()

        changed = reconcile_subject_agent_modes(session, project_id="project-1")

        assert changed == 1
        assert branch.subject_agent_mode == "shadow"
    database.close()
