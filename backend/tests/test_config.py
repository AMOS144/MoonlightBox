from moonlightbox.config import Settings
from moonlightbox.world.jobs import build_world_job_snapshot


def test_lightrag_timeout_allows_two_hour_indexing_requests() -> None:
    settings = Settings(lightrag_timeout_seconds=7200)

    assert settings.lightrag_timeout_seconds == 7200
    assert build_world_job_snapshot(settings)["timeout_seconds"] == 7200


def test_world_compiler_timeout_allows_two_hour_requests() -> None:
    settings = Settings(world_compiler_timeout_seconds=7200)

    assert settings.world_compiler_timeout_seconds == 7200
