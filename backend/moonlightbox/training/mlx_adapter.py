import importlib.util
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO, cast

ProgressCallback = Callable[[dict[str, object]], None]


class CommandRunner(Protocol):
    def run(self, command: list[str], on_line: Callable[[str], None]) -> int: ...


class MlxEnvironmentError(RuntimeError):
    pass


class MlxTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MlxLoraConfig:
    model: str
    data_dir: Path
    adapter_dir: Path
    iterations: int = 600
    batch_size: int = 1
    learning_rate: float = 1e-5


@dataclass(frozen=True)
class TrainingResult:
    adapter_dir: Path
    iterations: int


class MlxLmAdapter:
    def __init__(
        self,
        runner: CommandRunner | None = None,
        check_environment: bool = True,
        python_executable: str = "python",
    ) -> None:
        self._runner = runner or SubprocessRunner()
        self._check_environment = check_environment
        self._python_executable = python_executable

    def train(
        self,
        config: MlxLoraConfig,
        on_progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        if self._check_environment:
            _validate_environment()
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        callback = on_progress or (lambda _: None)
        command = [
            self._python_executable,
            "-m",
            "mlx_lm.lora",
            "--model",
            config.model,
            "--train",
            "--data",
            str(config.data_dir),
            "--adapter-path",
            str(config.adapter_dir),
            "--iters",
            str(config.iterations),
            "--batch-size",
            str(config.batch_size),
            "--learning-rate",
            str(config.learning_rate),
        ]

        def handle_line(line: str) -> None:
            match = re.search(r"Iter\s+(\d+):\s+Train loss\s+([0-9.]+)", line)
            if match:
                callback(
                    {
                        "stage": "training",
                        "iteration": int(match.group(1)),
                        "loss": float(match.group(2)),
                    }
                )
            elif "Saved adapter" in line:
                callback({"stage": "saved", "iteration": config.iterations})

        exit_code = self._runner.run(command, handle_line)
        if exit_code != 0:
            raise MlxTrainingError(f"MLX-LM 训练失败，退出码：{exit_code}")
        return TrainingResult(config.adapter_dir, config.iterations)


class SubprocessRunner:
    def run(self, command: list[str], on_line: Callable[[str], None]) -> int:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _consume_output(cast(TextIO | None, process.stdout), on_line)
        return process.wait()


def _consume_output(stream: TextIO | None, on_line: Callable[[str], None]) -> None:
    if stream is None:
        return
    for line in stream:
        on_line(line.rstrip())


def _validate_environment() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise MlxEnvironmentError("MLX-LM 训练仅支持 Apple Silicon")
    if importlib.util.find_spec("mlx_lm") is None:
        raise MlxEnvironmentError("未安装官方 mlx-lm 依赖")
