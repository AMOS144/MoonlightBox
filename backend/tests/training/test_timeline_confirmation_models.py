from sqlalchemy import inspect


def test_timeline_confirmation_stores_immutable_snapshot_contract() -> None:
    from moonlightbox.training.models import TimelineConfirmation

    columns = {column.key for column in inspect(TimelineConfirmation).columns}

    assert {
        "project_id",
        "import_id",
        "analysis_run_id",
        "confirmation_fingerprint",
        "active_event_ids",
        "rejected_event_ids",
        "event_revision_snapshots",
        "config_snapshot",
        "status",
        "training_job_id",
    } <= columns
    assert TimelineConfirmation.__table__.c.confirmation_fingerprint.unique is True


def test_model_version_records_training_origin_and_active_state() -> None:
    from moonlightbox.training.models import ModelVersion

    columns = {column.key for column in inspect(ModelVersion).columns}

    assert {
        "active",
        "timeline_confirmation_id",
        "training_job_id",
        "training_config",
    } <= columns
