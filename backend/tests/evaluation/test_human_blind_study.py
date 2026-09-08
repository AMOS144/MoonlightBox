from datetime import UTC, datetime
from pathlib import Path

import pytest
from moonlightbox.agent.models import SubjectAgentAcceptanceReport
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database
from moonlightbox.evaluation.blind_service import (
    BlindStudyStateError,
    HumanBlindStudyService,
)
from moonlightbox.evaluation.models import HumanBlindCase
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import select
from sqlalchemy.orm import Session


def _cases(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "context": [f"用户：第 {index} 个问题"],
            "human_reply": f"真人回答 {index}",
            "candidate_reply": f"候选回答 {index}",
        }
        for index in range(count)
    ]


def test_human_blind_study_hides_identity_and_requires_replay_before_activation(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'human-blind.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="真人盲测"))
        session.add_all(
            [
                ModelVersion(
                    id="old-model",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="/old",
                    dataset_hash="old",
                    metrics={},
                    active=True,
                    recommended=True,
                ),
                ModelVersion(
                    id="candidate",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="/candidate",
                    dataset_hash="candidate",
                    metrics={},
                    status="awaiting_human_review",
                    training_config={
                        "human_blind_required": True,
                        "upgraded_from_model_version_id": "old-model",
                    },
                ),
            ]
        )
        session.commit()
        service = HumanBlindStudyService(session)
        study = service.create(
            project_id="project-1",
            model_version_id="candidate",
            cases=_cases(),
            seed=17,
        )

        public = service.public_cases(study.id)
        assert len(public) == 20
        assert set(public[0]) == {"id", "context", "options"}
        assert "human_reply" not in public[0]
        assert "candidate_reply" not in public[0]

        cases = list(
            session.scalars(
                select(HumanBlindCase)
                .where(HumanBlindCase.study_id == study.id)
                .order_by(HumanBlindCase.position)
            )
        )
        for case in cases:
            service.rate(
                study_id=study.id,
                case_id=case.id,
                rater_key="human-rater",
                choice=case.candidate_option,
            )

        with pytest.raises(BlindStudyStateError, match="尚未通过历史回放"):
            service.finalize(
                project_id="project-1",
                model_version_id="candidate",
                study_id=study.id,
            )

        old = session.get(ModelVersion, "old-model")
        candidate = session.get(ModelVersion, "candidate")
        session.refresh(study)
        assert study.status == "open"
        assert old is not None and old.active is True
        assert candidate is not None and candidate.active is False
        assert "human_blind_report" not in candidate.training_config
    database.close()


def test_human_blind_study_fails_closed_with_too_few_ratings(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'human-blind-fail.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="真人盲测"))
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
        service = HumanBlindStudyService(session)
        study = service.create(
            project_id="project-1",
            model_version_id="candidate",
            cases=_cases(),
        )

        completed = service.finalize(
            project_id="project-1",
            model_version_id="candidate",
            study_id=study.id,
        )

        candidate = session.get(ModelVersion, "candidate")
        assert completed.status == "failed"
        assert candidate is not None and candidate.active is False
        assert candidate.status == "human_review_failed"
    database.close()


def test_new_blind_study_supersedes_stale_open_study(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'human-blind-superseded.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="真人盲测换代"))
        session.add_all(
            [
                ModelVersion(
                    id="candidate-1",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="/candidate-1",
                    dataset_hash="candidate-1",
                    metrics={},
                ),
                ModelVersion(
                    id="candidate-2",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="/candidate-2",
                    dataset_hash="candidate-2",
                    metrics={},
                ),
            ]
        )
        session.commit()
        service = HumanBlindStudyService(session)
        stale = service.create(
            project_id="project-1",
            model_version_id="candidate-1",
            cases=_cases(),
        )
        current = service.create(
            project_id="project-1",
            model_version_id="candidate-2",
            cases=_cases(),
        )

        session.refresh(stale)
        assert stale.status == "superseded"
        assert stale.report["replacement_model_version_id"] == "candidate-2"
        assert current.status == "open"
        with pytest.raises(BlindStudyStateError, match="已经结束"):
            service.public_cases(stale.id)
    database.close()


def test_human_blind_default_cannot_pass_when_candidate_loses_majority(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'human-blind-majority.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="真人盲测多数门槛"))
        session.add(
            ModelVersion(
                id="candidate",
                project_id="project-1",
                base_model="qwen",
                adapter_path="/candidate",
                dataset_hash="candidate",
                metrics={},
                training_config={"human_blind_required": True},
            )
        )
        session.commit()
        service = HumanBlindStudyService(session)
        study = service.create(
            project_id="project-1",
            model_version_id="candidate",
            cases=_cases(),
        )
        cases = list(
            session.scalars(
                select(HumanBlindCase)
                .where(HumanBlindCase.study_id == study.id)
                .order_by(HumanBlindCase.position)
            )
        )
        for index, case in enumerate(cases):
            service.rate(
                study_id=study.id,
                case_id=case.id,
                rater_key="human-rater",
                choice=(
                    case.candidate_option
                    if index < 9
                    else ("b" if case.candidate_option == "a" else "a")
                ),
            )

        completed = service.finalize(
            project_id="project-1",
            model_version_id="candidate",
            study_id=study.id,
        )

        assert completed.candidate_preference_rate == 0.45
        assert completed.status == "failed"
    database.close()


def test_stale_human_study_cannot_replace_changed_active_model(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'stale-human-blind.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="并发真人盲测"))
        session.add_all(
            [
                ModelVersion(
                    id="new-active",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="/new-active",
                    dataset_hash="new-active",
                    metrics={},
                    active=True,
                ),
                ModelVersion(
                    id="stale-candidate",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="/stale",
                    dataset_hash="stale",
                    metrics={},
                    status="awaiting_human_review",
                    training_config={
                        "human_blind_required": True,
                        "upgraded_from_model_version_id": "old-active",
                    },
                ),
            ]
        )
        session.commit()
        service = HumanBlindStudyService(session)
        study = service.create(
            project_id="project-1",
            model_version_id="stale-candidate",
            cases=_cases(),
        )

        with pytest.raises(BlindStudyStateError, match="活动模型已变化"):
            service.finalize(
                project_id="project-1",
                model_version_id="stale-candidate",
                study_id=study.id,
            )

        assert session.get(ModelVersion, "new-active").active is True
        assert session.get(ModelVersion, "stale-candidate").active is False
    database.close()


def test_passing_blind_study_activates_newest_branch_with_passed_replay(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'blind-activates-branch.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="完整验收门"))
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
        session.flush()
        session.add(
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
            )
        )
        session.flush()
        session.add(
            Branch(
                id="branch-1",
                project_id="project-1",
                origin_event_id="event-1",
                model_version_id="candidate",
                title="已通过回放的分支",
                origin_time=datetime.now(UTC),
                state_snapshot={},
            )
        )
        session.flush()
        session.add(
            SubjectAgentAcceptanceReport(
                project_id="project-1",
                branch_id="branch-1",
                model_version_id="candidate",
                sample_count=20,
                structure_extraction_success_rate=1,
                fact_safety_rate=1,
                expression_decision_accuracy=1,
                p95_cognition_latency_ms=100,
                direct_lora_p95=100,
                passed=True,
                failure_reasons=[],
                evidence={},
            )
        )
        session.commit()
        service = HumanBlindStudyService(session)
        study = service.create(
            project_id="project-1",
            model_version_id="candidate",
            cases=_cases(),
        )
        cases = list(
            session.scalars(
                select(HumanBlindCase).where(HumanBlindCase.study_id == study.id)
            )
        )
        for case in cases:
            service.rate(
                study_id=study.id,
                case_id=case.id,
                rater_key="human-rater",
                choice=case.candidate_option,
            )

        completed = service.finalize(
            project_id="project-1",
            model_version_id="candidate",
            study_id=study.id,
        )

        branch = session.get(Branch, "branch-1")
        assert completed.status == "passed"
        assert completed.report["activated_branch_id"] == "branch-1"
        assert branch is not None and branch.subject_agent_mode == "active"
    database.close()
