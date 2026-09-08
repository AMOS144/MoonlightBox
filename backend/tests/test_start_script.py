import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _prepare_fake_venv(project_dir: Path) -> None:
    python_path = project_dir / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(sys.executable)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("等待启动脚本状态超时")


def test_frontend_uses_dedicated_non_conflicting_port() -> None:
    project_root = Path(__file__).parents[2]
    vite_config = (project_root / "frontend" / "vite.config.ts").read_text()
    start_script = (project_root / "scripts" / "start.sh").read_text()

    assert "MOONLIGHTBOX_FRONTEND_PORT ?? '5175'" in vite_config
    assert "MOONLIGHTBOX_BACKEND_PORT ?? '8001'" in vite_config
    assert 'BACKEND_PORT="${MOONLIGHTBOX_BACKEND_PORT:-8001}"' in start_script
    assert 'FRONTEND_PORT="${MOONLIGHTBOX_FRONTEND_PORT:-5175}"' in start_script
    assert 'PERSONA_RUNTIME_PORT="${PERSONA_RUNTIME_PORT:-8765}"' in start_script
    assert 'ensure_port_available "$BACKEND_PORT" "后端"' in start_script
    assert 'ensure_port_available "$FRONTEND_PORT" "前端"' in start_script
    assert 'MANAGE_PERSONA_RUNTIME="${MOONLIGHTBOX_MANAGE_PERSONA_RUNTIME:-auto}"' in start_script
    health_block = start_script[
        start_script.index("for ((attempt = 1;") : start_script.index(
            'echo "正在启动实时会话 Worker"'
        )
    ]
    assert 'service_is_running "$PERSONA_RUNTIME_PID"' in health_block
    assert 'group_is_running "$PERSONA_RUNTIME_PID"' not in health_block
    assert 'uvicorn moonlightbox.api:app --app-dir backend --port "$BACKEND_PORT"' in (start_script)
    assert 'npm run dev -- --strictPort --port "$FRONTEND_PORT"' in start_script
    assert "5173" not in start_script


def test_start_script_prepares_starts_and_cleans_up_isolated_workers(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "frontend").mkdir(parents=True)
    _prepare_fake_venv(project_dir)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "commands.log"
    _write_fake_command(
        bin_dir / "uv",
        'echo "uv $*" >> "$START_TEST_LOG"\n'
        'case "$*" in\n'
        '  *worker_main*) trap \'echo "worker stopped" >> "$START_TEST_LOG"; exit 0\' TERM; '
        "while :; do sleep 0.01; done ;;\n"
        "  *uvicorn*) trap 'exit 0' TERM; while :; do sleep 0.01; done ;;\n"
        "esac",
    )
    _write_fake_command(
        bin_dir / "npm",
        'echo "npm $*" >> "$START_TEST_LOG"\n'
        'case "$*" in "run dev"*) trap \'exit 0\' TERM; '
        "while :; do sleep 0.01; done ;; esac",
    )
    _write_fake_command(bin_dir / "node", "exit 0")
    _write_fake_command(
        bin_dir / "curl",
        'echo "curl $*" >> "$START_TEST_LOG"\nexit 0',
    )
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:/bin:/usr/bin",
        "START_TEST_LOG": str(log_path),
        "MOONLIGHTBOX_ROOT_DIR": str(project_dir),
        "MOONLIGHTBOX_MONITOR_POLL_SECONDS": "0.01",
    }

    process = subprocess.Popen(
        ["bash", "scripts/start.sh"],
        cwd=Path(__file__).parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_until(
        lambda: (
            log_path.exists()
            and log_path.read_text(encoding="utf-8").count(
                "uv run python -m moonlightbox.worker_main"
            )
            == 4
        )
    )
    process.terminate()
    _, stderr = process.communicate(timeout=10)

    assert log_path.exists(), stderr
    commands = log_path.read_text(encoding="utf-8")
    assert process.returncode == 130
    assert "uv sync --extra linux-ml" in commands
    assert "npm install" in commands
    assert "uv run alembic -c backend/alembic.ini upgrade head" in commands
    assert "uv run uvicorn moonlightbox.api:app --app-dir backend" in commands
    assert "--reload" not in commands
    assert "uv run uvicorn moonlightbox.persona_runtime:app --app-dir backend" in commands
    assert "curl --fail --silent --show-error http://127.0.0.1:8765/health" in commands
    assert commands.count("uv run python -m moonlightbox.worker_main") == 4
    assert (
        "MOONLIGHTBOX_WORKER_ROLE=realtime"
        in (Path(__file__).parents[2] / "scripts" / "start.sh").read_text()
    )
    assert (
        "MOONLIGHTBOX_WORKER_ROLE=background"
        in (Path(__file__).parents[2] / "scripts" / "start.sh").read_text()
    )
    assert (
        "MOONLIGHTBOX_WORKER_ROLE=cognition"
        in (Path(__file__).parents[2] / "scripts" / "start.sh").read_text()
    )
    assert (
        "MOONLIGHTBOX_WORKER_ROLE=training"
        in (Path(__file__).parents[2] / "scripts" / "start.sh").read_text()
    )
    assert "npm run dev" in commands
    assert "worker stopped" in commands
    assert (project_dir / "data").is_dir()
    assert (project_dir / "models").is_dir()


def test_start_script_force_kills_stubborn_child_with_bounded_wait(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "frontend").mkdir(parents=True)
    _prepare_fake_venv(project_dir)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "commands.log"
    process_path = tmp_path / "worker-processes.txt"
    _write_fake_command(
        bin_dir / "uv",
        'echo "uv $*" >> "$START_TEST_LOG"\n'
        'case "$*" in\n'
        "  *worker_main*) sh -c 'trap \"\" TERM; while :; do sleep 0.05; done' "
        "</dev/null >/dev/null 2>&1 & "
        'echo "$$ $!" >> "$START_TEST_PROCESSES"; trap \'\' TERM; wait ;;\n'
        "  *uvicorn*) trap 'exit 0' TERM; while :; do sleep 0.05; done ;;\n"
        "esac",
    )
    _write_fake_command(
        bin_dir / "npm",
        'echo "npm $*" >> "$START_TEST_LOG"\n'
        'case "$*" in "run dev"*) trap \'exit 0\' TERM; '
        "while :; do sleep 0.05; done ;; esac",
    )
    _write_fake_command(bin_dir / "node", "exit 0")
    _write_fake_command(bin_dir / "curl", "exit 0")
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:/bin:/usr/bin",
        "START_TEST_LOG": str(log_path),
        "START_TEST_PROCESSES": str(process_path),
        "MOONLIGHTBOX_ROOT_DIR": str(project_dir),
        "MOONLIGHTBOX_SHUTDOWN_GRACE_STEPS": "1",
        "MOONLIGHTBOX_SHUTDOWN_POLL_SECONDS": "0.01",
        "MOONLIGHTBOX_MONITOR_POLL_SECONDS": "0.01",
    }

    process = subprocess.Popen(
        ["bash", "scripts/start.sh"],
        cwd=Path(__file__).parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_until(
        lambda: process_path.exists() and len(process_path.read_text(encoding="utf-8").split()) == 8
    )
    process.terminate()
    process.communicate(timeout=8)

    assert process.returncode == 130
    process_ids = [int(value) for value in process_path.read_text(encoding="utf-8").split()]
    assert len(process_ids) == 8
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and (any(_process_exists(pid) for pid in process_ids)):
        time.sleep(0.01)
    try:
        assert all(not _process_exists(pid) for pid in process_ids)
    finally:
        for pid in process_ids:
            if _process_exists(pid):
                os.kill(pid, 9)
