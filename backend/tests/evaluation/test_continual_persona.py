from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchStateVersion,
)
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class FakeScenarioExecutor:
    def __init__(self, failed_ids: set[str] | None = None) -> None:
        self.failed_ids = failed_ids or set()

    def run_case(
        self,
        *,
        project_id: str,
        model_version_id: str,
        case: object,
    ) -> object:
        from moonlightbox.evaluation.continual_persona import (
            ContinualPersonaCase,
            ContinualPersonaCaseResult,
        )

        del project_id, model_version_id
        assert isinstance(case, ContinualPersonaCase)
        return ContinualPersonaCaseResult(
            case_id=case.id,
            passed=case.id not in self.failed_ids,
            violations=([] if case.id not in self.failed_ids else ["模拟失败"]),
        )


def _setup(session: Session) -> Branch:
    now = datetime.now(UTC)
    session.add(Project(id="project-1", name="持续人格验收"))
    session.flush()
    session.add(
        EventNode(
            id="event-1",
            project_id="project-1",
            type="relationship",
            title="起点",
            summary="开始",
            start_message_id="m1",
            end_message_id="m2",
            emotion_labels=[],
            topic="关系",
            conflict_level=0,
            importance=0.8,
            reason="起点",
            evidence_ids=["m1"],
        )
    )
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
    session.flush()
    branch = Branch(
        id="branch-1",
        project_id="project-1",
        origin_event_id="event-1",
        model_version_id="model-1",
        title="分支",
        origin_time=now,
        state_snapshot={"protocol_version": "continual-persona-v1-pending"},
    )
    session.add(branch)
    session.commit()
    return branch


def test_acceptance_report_activates_protocol_only_after_all_cases_pass(
    tmp_path: Path,
) -> None:
    from moonlightbox.evaluation.continual_persona import (
        ContinualPersonaAcceptanceRunner,
    )

    database = Database(f"sqlite:///{tmp_path / 'acceptance.db'}")
    Project.metadata.create_all(database.engine)
    fixture = Path("backend/tests/fixtures/continual_persona_acceptance_v1.json")
    with Session(database.engine) as session:
        branch = _setup(session)

        report = ContinualPersonaAcceptanceRunner(
            session,
            FakeScenarioExecutor(),
        ).run(
            project_id="project-1",
            model_version_id="model-1",
            fixture_path=fixture,
        )

        assert report.passed is True
        assert report.case_count == 9
        assert report.passed_count == 9
        assert branch.state_snapshot["protocol_version"] == "continual-persona-v1"
    database.close()


def test_acceptance_report_classifies_failures(tmp_path: Path) -> None:
    from moonlightbox.evaluation.continual_persona import (
        ContinualPersonaAcceptanceRunner,
    )

    database = Database(f"sqlite:///{tmp_path / 'failed.db'}")
    Project.metadata.create_all(database.engine)
    fixture = Path("backend/tests/fixtures/continual_persona_acceptance_v1.json")
    with Session(database.engine) as session:
        branch = _setup(session)

        report = ContinualPersonaAcceptanceRunner(
            session,
            FakeScenarioExecutor(
                {"strict-branch-isolation", "lineage-no-self-reinforcement"}
            ),
        ).run(
            project_id="project-1",
            model_version_id="model-1",
            fixture_path=fixture,
        )

        assert report.passed is False
        assert report.branch_leak_failures == 1
        assert report.lineage_failures == 1
        assert branch.state_snapshot["protocol_version"] == (
            "continual-persona-v1-pending"
        )
    database.close()


def test_database_acceptance_does_not_pass_without_longitudinal_evidence(
    tmp_path: Path,
) -> None:
    from moonlightbox.evaluation.continual_persona import (
        ContinualPersonaCase,
        DatabaseInvariantScenarioExecutor,
    )

    database = Database(f"sqlite:///{tmp_path / 'non-vacuous.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _setup(session)
        executor = DatabaseInvariantScenarioExecutor(session)

        growth = executor.run_case(
            project_id="project-1",
            model_version_id="model-1",
            case=ContinualPersonaCase(
                id="gradual-trust-growth",
                category="identity_drift",
                description="渐进成长",
            ),
        )
        narrative = executor.run_case(
            project_id="project-1",
            model_version_id="model-1",
            case=ContinualPersonaCase(
                id="autonomous-self-narrative",
                category="identity_drift",
                description="自主叙事",
            ),
        )

        assert growth.passed is False
        assert any("状态版本不足" in item for item in growth.violations)
        assert narrative.passed is False
        assert any("互动不足" in item for item in narrative.violations)
        assert any("阶段反思" in item for item in narrative.violations)
    database.close()


def test_database_acceptance_recognizes_real_longitudinal_evidence(
    tmp_path: Path,
) -> None:
    from moonlightbox.evaluation.continual_persona import (
        ContinualPersonaCase,
        DatabaseInvariantScenarioExecutor,
    )

    database = Database(f"sqlite:///{tmp_path / 'longitudinal.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch = _setup(session)
        start = datetime(2026, 7, 1, tzinfo=UTC)
        episodes: list[BranchMemoryEpisode] = []
        for index in range(20):
            at = start + timedelta(days=index)
            episodes.append(
                BranchMemoryEpisode(
                    id=f"episode-{index}",
                    branch_id=branch.id,
                    user_turn_id=f"user-{index}",
                    assistant_turn_id=f"assistant-{index}",
                    user_content="继续聊",
                    assistant_bubbles=[{"type": "text", "content": "好"}],
                    model_version_id="model-1",
                    episode_hash=f"episode-hash-{index}",
                    importance=3,
                    processing_status="processed",
                    started_at=at,
                    ended_at=at + timedelta(minutes=1),
                )
            )
        session.add_all(episodes)
        previous_id: str | None = None
        states: list[BranchStateVersion] = []
        for version in range(1, 6):
            state = BranchStateVersion(
                id=f"state-{version}",
                branch_id=branch.id,
                version=version,
                previous_version_id=previous_id,
                relationship_state={"trust": 50 + version},
                reason="真实互动成长",
                source_episode_ids=[f"episode-{version - 1}"],
                is_current=version == 5,
            )
            states.append(state)
            previous_id = state.id
        session.add_all(states)
        items = [
            BranchMemoryItem(
                id=f"item-{index}",
                branch_id=branch.id,
                kind=("self_narrative" if index == 0 else "experience"),
                content=f"成长记忆 {index}",
                subject="digital_human",
                predicate="成长",
                object=str(index),
                confidence=0.8,
                importance=5,
                valid_from=start,
                source_episode_ids=[f"episode-{index}"],
                lineage_hash=f"lineage-{index}",
                review_status="approved",
            )
            for index in range(5)
        ]
        items.append(
            BranchMemoryItem(
                id="reflection-1",
                branch_id=branch.id,
                kind="reflection",
                content="多次互动后逐渐建立信任",
                subject="digital_human",
                predicate="关系反思",
                object="逐渐信任",
                confidence=0.8,
                importance=7,
                valid_from=start + timedelta(days=19),
                source_episode_ids=["episode-0", "episode-19"],
                lineage_hash="reflection-lineage",
                review_status="approved",
            )
        )
        session.add_all(items)
        session.commit()
        executor = DatabaseInvariantScenarioExecutor(session)

        growth = executor.run_case(
            project_id="project-1",
            model_version_id="model-1",
            case=ContinualPersonaCase(
                id="gradual-trust-growth",
                category="identity_drift",
                description="渐进成长",
            ),
        )
        narrative = executor.run_case(
            project_id="project-1",
            model_version_id="model-1",
            case=ContinualPersonaCase(
                id="autonomous-self-narrative",
                category="identity_drift",
                description="自主叙事",
            ),
        )

        assert growth.passed is True
        assert narrative.passed is True
    database.close()
