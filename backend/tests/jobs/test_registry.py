import pytest


def test_unknown_job_kind_is_rejected() -> None:
    from moonlightbox.jobs.registry import JobRegistry, UnknownJobKindError

    registry = JobRegistry()

    with pytest.raises(UnknownJobKindError):
        registry.get("missing")
