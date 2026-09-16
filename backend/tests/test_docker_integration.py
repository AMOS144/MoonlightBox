import os
import stat
import subprocess
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
COMPOSE_PATH = ROOT_DIR / "docker-compose.yml"
ENTRYPOINT_PATH = ROOT_DIR / "backend" / "docker-entrypoint.sh"


def _load_compose() -> dict[str, object]:
    loaded = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_compose_coordinates_migration_backend_isolated_workers_and_frontend() -> None:
    compose = _load_compose()
    services = compose["services"]
    assert isinstance(services, dict)

    migration = services["migration"]
    backend = services["backend"]
    realtime_worker = services["realtime-worker"]
    cognition_worker = services["cognition-worker"]
    background_worker = services["background-worker"]
    training_worker = services["training-worker"]
    frontend = services["frontend"]

    assert migration["image"] == backend["image"]
    assert realtime_worker["image"] == cognition_worker["image"]
    assert training_worker["image"] == background_worker["image"]
    assert cognition_worker["image"] == background_worker["image"]
    assert backend["depends_on"]["migration"]["condition"] == "service_completed_successfully"
    assert (
        background_worker["depends_on"]["migration"]["condition"]
        == "service_completed_successfully"
    )
    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"
    assert backend["healthcheck"]["test"][0] == "CMD"
    assert realtime_worker["restart"] == "unless-stopped"
    assert cognition_worker["restart"] == "unless-stopped"
    assert background_worker["restart"] == "unless-stopped"
    assert training_worker["restart"] == "unless-stopped"


def test_compose_backend_processes_share_runtime_configuration_and_volumes() -> None:
    compose = _load_compose()
    services = compose["services"]
    migration = services["migration"]
    backend = services["backend"]
    realtime_worker = services["realtime-worker"]
    cognition_worker = services["cognition-worker"]
    background_worker = services["background-worker"]
    training_worker = services["training-worker"]

    assert migration["entrypoint"] == ["/app/backend/docker-entrypoint.sh"]
    assert migration["command"] == ["/bin/true"]
    assert backend["command"][:2] == ["/app/.venv/bin/uvicorn", "moonlightbox.api:app"]
    assert realtime_worker["command"] == [
        "/app/.venv/bin/python",
        "-m",
        "moonlightbox.worker_main",
    ]
    assert cognition_worker["command"] == realtime_worker["command"]
    assert background_worker["command"] == realtime_worker["command"]

    for service in (
        migration,
        backend,
        realtime_worker,
        cognition_worker,
        background_worker,
        training_worker,
    ):
        assert service["working_dir"] == "/app"
        assert service["env_file"] == backend["env_file"]
        assert service["environment"]["PYTHONPATH"] == "/app/backend"
        assert service["environment"]["MOONLIGHTBOX_DATABASE_URL"] == (
            "sqlite:////app/data/moonlightbox.db"
        )
        assert service["environment"]["MOONLIGHTBOX_AUTO_CREATE_SCHEMA"] == "false"
        assert (
            "${MOONLIGHTBOX_HOST_DATA_DIR:-../.runtime-data/data}:/app/data"
            in service["volumes"]
        )
        assert "./models:/app/models" in service["volumes"]

    assert "MOONLIGHTBOX_NODE_ANALYSIS_API_KEY" not in backend["environment"]
    assert realtime_worker["environment"]["MOONLIGHTBOX_WORKER_ROLE"] == "realtime"
    assert cognition_worker["environment"]["MOONLIGHTBOX_WORKER_ROLE"] == "cognition"
    assert background_worker["environment"]["MOONLIGHTBOX_WORKER_ROLE"] == "background"
    assert training_worker["environment"]["MOONLIGHTBOX_WORKER_ROLE"] == "training"
    assert training_worker["gpus"] == "all"
    assert "MOONLIGHTBOX_PERSONA_INFERENCE_URL" in backend["environment"]
    assert "MOONLIGHTBOX_PERSONA_INFERENCE_TOKEN" in backend["environment"]
    assert backend["env_file"] == realtime_worker["env_file"]
    assert backend["volumes"] == realtime_worker["volumes"]


def test_backend_image_contains_migration_entrypoint_without_mlx_dependency() -> None:
    dockerfile = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY backend ./backend" in dockerfile
    assert "chmod +x /app/backend/docker-entrypoint.sh" in dockerfile
    assert 'ENV PATH="/app/.venv/bin:$PATH"' in dockerfile
    assert "--extra mlx" not in dockerfile


def test_entrypoint_stops_when_migration_fails(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "service-started"
    _write_executable(bin_dir / "alembic", "#!/bin/sh\nexit 23\n")
    _write_executable(
        bin_dir / "fake-service",
        f"#!/bin/sh\ntouch {marker}\n",
    )

    result = subprocess.run(
        [str(ENTRYPOINT_PATH), "fake-service"],
        cwd=ROOT_DIR,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert not marker.exists()
    assert "数据库迁移失败，服务未启动" in result.stderr


def test_entrypoint_executes_target_after_successful_migration(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    alembic_log = tmp_path / "alembic.log"
    service_log = tmp_path / "service.log"
    _write_executable(
        bin_dir / "alembic",
        f'#!/bin/sh\nprintf "%s\\n" "$*" > {alembic_log}\n',
    )
    _write_executable(
        bin_dir / "fake-service",
        f'#!/bin/sh\nprintf "%s\\n" "$$:$*" > {service_log}\n',
    )

    process = subprocess.Popen(
        [str(ENTRYPOINT_PATH), "fake-service", "--flag", "value"],
        cwd=ROOT_DIR,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = process.communicate(timeout=5)

    assert process.returncode == 0, stderr
    assert alembic_log.read_text(encoding="utf-8").strip() == (
        "-c backend/alembic.ini upgrade head"
    )
    service_pid, service_args = service_log.read_text(encoding="utf-8").strip().split(":", 1)
    assert int(service_pid) == process.pid
    assert service_args == "--flag value"
