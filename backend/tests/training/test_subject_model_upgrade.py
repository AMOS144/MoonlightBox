from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.training.model_acceptance import AcceptanceReport
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class PassingRunner:
    def run(self, **_kwargs: object) -> AcceptanceReport:
        return AcceptanceReport(
            case_count=5,
            passed_count=5,
            structure_failures=0,
            forbidden_fact_failures=0,
            failed_case_ids=(),
            passed=True,
        )


def test_legacy_active_model_is_cloned_only_after_v2_acceptance(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_models import IdentityKernel
    from moonlightbox.training.subject_upgrade import SubjectPersonaModelUpgrader

    database = Database(f"sqlite:///{tmp_path / 'model-upgrade.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="模型升级"))
        session.flush()
        source = ImportSource(
            id="import-1",
            project_id="project-1",
            preview_id="preview-1",
            source_path="/tmp/chat.txt",
            message_count=2,
            confirmed_at=now,
        )
        user = Participant(
            id="user-1",
            project_id="project-1",
            name="我",
            role="self",
        )
        target = Participant(
            id="target-1",
            project_id="project-1",
            name="她",
            role="target",
        )
        session.add_all([source, user, target])
        session.flush()
        session.add_all(
            [
                Message(
                    project_id="project-1",
                    import_id=source.id,
                    participant_id=user.id,
                    source_id="m1",
                    timestamp=now,
                    kind="text",
                    content="还好吗",
                    raw={},
                ),
                Message(
                    project_id="project-1",
                    import_id=source.id,
                    participant_id=target.id,
                    source_id="m2",
                    timestamp=now + timedelta(seconds=3),
                    kind="text",
                    content="还好呀",
                    raw={},
                ),
            ]
        )
        old = ModelVersion(
            id="model-old",
            project_id="project-1",
            base_model="/models/qwen",
            adapter_path="/models/adapter",
            dataset_hash="dataset-hash",
            metrics={},
            active=True,
        )
        session.add(old)
        session.commit()

        upgraded = SubjectPersonaModelUpgrader(
            session,
            PassingRunner(),  # type: ignore[arg-type]
        ).upgrade("project-1")
        kernel = session.query(IdentityKernel).filter(
            IdentityKernel.model_version_id == upgraded.id
        ).one()

        assert upgraded.id != old.id
        assert upgraded.active is True
        assert old.active is False
        assert kernel.schema_version == "subject-persona-v2"
        assert kernel.acceptance_report_id is not None
        assert kernel.field_evidence["language_patterns"] == ["m1", "m2"]
    database.close()
