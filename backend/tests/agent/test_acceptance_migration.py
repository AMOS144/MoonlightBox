from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[3]


def _config(database_path: Path) -> Config:
    config = Config(str(ROOT / "backend" / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def test_migration_adds_shadow_default_and_acceptance_report_scope(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "migration.db"
    config = _config(database_path)
    command.upgrade(config, "0025_unique_cycle_trigger")
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO branches (
                    id, project_id, origin_event_id, model_version_id, title,
                    origin_time, state_snapshot, created_at
                ) VALUES (
                    'legacy-branch', 'project-1', 'event-1', 'model-1', '旧分支',
                    '2026-07-23 00:00:00', '{}', '2026-07-23 00:00:00'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO branches (
                    id, project_id, origin_event_id, model_version_id, title,
                    origin_time, state_snapshot, generation_policy_version, created_at
                ) VALUES (
                    'candidate-branch', 'project-1', 'event-1', 'model-1', '候选分支',
                    '2026-07-23 00:00:00', '{}', 'subject-v2-candidate-v1',
                    '2026-07-23 00:00:00'
                )
                """
            )
        )

    command.upgrade(config, "head")

    inspector = inspect(engine)
    branch_columns = {column["name"]: column for column in inspector.get_columns("branches")}
    report_columns = {
        column["name"]
        for column in inspector.get_columns("subject_agent_acceptance_reports")
    }
    report_foreign_keys = inspector.get_foreign_keys(
        "subject_agent_acceptance_reports"
    )
    with engine.connect() as connection:
        mode = connection.scalar(
            text(
                "SELECT subject_agent_mode FROM branches WHERE id = 'legacy-branch'"
            )
        )
        candidate_mode = connection.scalar(
            text(
                "SELECT subject_agent_mode FROM branches "
                "WHERE id = 'candidate-branch'"
            )
        )

    assert branch_columns["subject_agent_mode"]["default"] in ("'shadow'", "shadow")
    assert mode == "shadow"
    assert candidate_mode == "preview"
    assert {
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
    }.issubset(report_columns)
    assert any(
        foreign_key["constrained_columns"] == ["branch_id", "project_id"]
        and foreign_key["referred_table"] == "branches"
        and foreign_key["referred_columns"] == ["id", "project_id"]
        for foreign_key in report_foreign_keys
    )
    engine.dispose()
