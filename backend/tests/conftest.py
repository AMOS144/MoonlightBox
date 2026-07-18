from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
