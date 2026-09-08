import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import yaml

# MLX adapter 只保留给 macOS 历史兼容性；Linux 训练机使用 PyTorch + PEFT。
pytest.importorskip("mlx_lm", reason="MLX adapter 测试仅适用于 macOS 环境")


def test_checkpoint_pruning_keeps_only_current_best(tmp_path: Path) -> None:
    from moonlightbox.training.mlx_adapter import _prune_adapter_checkpoints

    first = tmp_path / "0000001_adapters.safetensors"
    best = tmp_path / "0000002_adapters.safetensors"
    latest = tmp_path / "0000003_adapters.safetensors"
    for path in (first, best, latest):
        path.write_bytes(path.name.encode())

    _prune_adapter_checkpoints(tmp_path, keep=best)

    assert not first.exists()
    assert best.exists()
    assert not latest.exists()


class FakeRunner:
    def __init__(self) -> None:
        self.command: list[str] = []

    def run(
        self,
        command: list[str],
        on_line: object,
        should_stop: object | None = None,
    ) -> int:
        self.command = command
        on_line("Iter 10: Train loss 1.23")
        on_line("Iter 10: Val loss 1.11")
        on_line("Saved adapter weights")
        return 0


class StreamingEarlyStopRunner:
    def __init__(self) -> None:
        self.command: list[str] = []
        self.emitted_iterations: list[int] = []
        self.stopped = False

    def run(
        self,
        command: list[str],
        on_line: object,
        should_stop: object | None = None,
    ) -> int:
        self.command = command
        config_path = Path(command[-1])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        adapter_dir = Path(payload["adapter_path"])
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        points = ((2, 0.8), (4, 0.9), (6, 1.0), (8, 1.1))
        for iteration, loss in points:
            checkpoint_iteration = iteration - 1
            checkpoint = adapter_dir / (
                f"{checkpoint_iteration:07d}_adapters.safetensors"
            )
            checkpoint.write_bytes(f"checkpoint-{checkpoint_iteration}".encode())
            on_line(f"Iter {iteration}: Val loss {loss}")
            self.emitted_iterations.append(iteration)
            if callable(should_stop) and should_stop():
                self.stopped = True
                return 130
        (adapter_dir / "adapters.safetensors").write_bytes(b"last")
        return 0


class BaselineDominatesRunner:
    def run(
        self,
        command: list[str],
        on_line: object,
        should_stop: object | None = None,
    ) -> int:
        config_path = Path(command[-1])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        adapter_dir = Path(payload["adapter_path"])
        (adapter_dir / "0000001_adapters.safetensors").write_bytes(b"publishable")
        on_line("Iter 1: Val loss 0.01")
        on_line("Iter 2: Val loss 0.80")
        (adapter_dir / "adapters.safetensors").write_bytes(b"last")
        return 0


class SparseCheckpointRunner:
    def run(
        self,
        command: list[str],
        on_line: object,
        should_stop: object | None = None,
    ) -> int:
        config_path = Path(command[-1])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        adapter_dir = Path(payload["adapter_path"])
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        on_line("Iter 50: Val loss 0.70")
        (adapter_dir / "0000050_adapters.safetensors").write_bytes(b"checkpoint-50")
        on_line("Saved adapter weights to 0000050_adapters.safetensors")
        on_line("Iter 100: Val loss 0.80")
        (adapter_dir / "0000100_adapters.safetensors").write_bytes(b"checkpoint-100")
        on_line("Saved adapter weights to 0000100_adapters.safetensors")
        (adapter_dir / "adapters.safetensors").write_bytes(b"latest")
        return 0


def test_mlx_adapter_builds_official_lora_command_and_reports_progress(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig

    runner = FakeRunner()
    progress: list[dict[str, object]] = []
    adapter = MlxLmAdapter(runner, check_environment=False)
    config = MlxLoraConfig(
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        data_dir=tmp_path / "dataset",
        adapter_dir=tmp_path / "adapter",
        iterations=100,
    )

    result = adapter.train(config, progress.append)

    assert runner.command[:4] == ["python", "-m", "mlx_lm", "lora"]
    assert runner.command[4] == "--config"
    payload = yaml.safe_load(Path(runner.command[5]).read_text(encoding="utf-8"))
    assert payload["model"] == config.model
    assert payload["adapter_path"] == str(config.adapter_dir)
    assert payload["mask_prompt"] is True
    assert payload["grad_accumulation_steps"] == 4
    assert payload["steps_per_eval"] == 50
    assert payload["save_every"] == 100
    assert progress[0]["iteration"] == 10
    assert progress[1]["validation_loss"] == 1.11
    assert result.adapter_dir == config.adapter_dir


def test_sparse_checkpoint_selection_is_deterministic_and_bounded(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig

    progress: list[dict[str, object]] = []
    adapter_dir = tmp_path / "adapter"
    result = MlxLmAdapter(
        SparseCheckpointRunner(),
        check_environment=False,
    ).train(
        MlxLoraConfig(
            model="models/Qwen3-8B-4bit",
            data_dir=tmp_path / "dataset",
            adapter_dir=adapter_dir,
            iterations=100,
            steps_per_evaluation=50,
            save_every=50,
        ),
        progress.append,
    )

    selection = result.best_checkpoint_selection
    assert selection is not None
    assert selection.validation_loss == 0.70
    assert selection.validation_iteration == 50
    assert selection.checkpoint_iteration == 50
    assert selection.approximation_steps == 1
    assert selection.mapping_rule == (
        "mlx-lm-0.31.3-validation-before-step:"
        "checkpoint=nearest-saved-checkpoint-to-pre-step-weights"
    )
    assert progress[-1]["iteration"] == 100
    assert len(list(adapter_dir.glob("[0-9]" * 7 + "_adapters.safetensors"))) <= 2


def test_mlx_adapter_writes_explicit_version_checked_yaml(tmp_path: Path) -> None:
    from moonlightbox.training.mlx_adapter import (
        MlxLmAdapter,
        MlxLoraCapabilities,
        MlxLoraConfig,
    )

    runner = FakeRunner()
    capabilities = MlxLoraCapabilities(
        version="0.31.3",
        config_fields=frozenset(
            {
                "model",
                "data",
                "adapter_path",
                "train",
                "fine_tune_type",
                "optimizer",
                "optimizer_config",
                "batch_size",
                "iters",
                "val_batches",
                "steps_per_eval",
                "save_every",
                "max_seq_length",
                "seed",
                "learning_rate",
                "lr_schedule",
                "grad_checkpoint",
                "grad_accumulation_steps",
                "num_layers",
                "lora_parameters",
                "mask_prompt",
                "steps_per_report",
            }
        ),
    )
    adapter = MlxLmAdapter(
        runner,
        check_environment=False,
        capabilities=capabilities,
    )
    config = MlxLoraConfig(
        model="models/Qwen3-8B-4bit",
        data_dir=tmp_path / "dataset",
        adapter_dir=tmp_path / "adapter",
        train_example_count=24,
        epochs=3,
        batch_size=2,
        learning_rate=5e-6,
        seed=17,
        num_layers=16,
        rank=32,
        alpha=64,
        dropout=0.1,
        target_modules=("self_attn.q_proj", "self_attn.v_proj"),
        warmup_steps=3,
        weight_decay=0.01,
    )

    result = adapter.train(config)

    assert runner.command[:4] == ["python", "-m", "mlx_lm", "lora"]
    assert runner.command[4] == "--config"
    yaml_path = Path(runner.command[5])
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["model"] == "models/Qwen3-8B-4bit"
    assert payload["data"] == str(config.data_dir)
    assert payload["adapter_path"] == str(config.adapter_dir)
    assert payload["train"] is True
    assert payload["batch_size"] == 2
    assert payload["iters"] == 9
    assert payload["optimizer"] == "adamw"
    assert payload["optimizer_config"]["adamw"]["weight_decay"] == 0.01
    assert payload["lr_schedule"]["name"] == "cosine_decay"
    assert payload["lr_schedule"]["warmup"] == 3
    assert payload["grad_checkpoint"] is True
    assert payload["lora_parameters"] == {
        "rank": 32,
        "scale": 2.0,
        "dropout": 0.1,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    assert result.config_path == yaml_path
    assert result.mlx_version == "0.31.3"
    assert result.command == tuple(runner.command)


def test_effective_steps_per_epoch_include_gradient_accumulation() -> None:
    from moonlightbox.training.mlx_adapter import MlxLoraConfig

    config = MlxLoraConfig(
        model="models/Qwen3-8B-4bit",
        data_dir=Path("dataset"),
        adapter_dir=Path("adapter"),
        train_example_count=1344,
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
    )

    assert config.effective_batch_size == 4
    assert config.steps_per_epoch == 336
    assert config.resolved_iterations == 336


def test_mlx_adapter_omits_optional_unsupported_gradient_checkpoint(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import (
        MlxLmAdapter,
        MlxLoraCapabilities,
        MlxLoraConfig,
    )

    runner = FakeRunner()
    supported = {
        "model",
        "data",
        "adapter_path",
        "train",
        "fine_tune_type",
        "optimizer",
        "optimizer_config",
        "batch_size",
        "iters",
        "val_batches",
        "steps_per_eval",
        "save_every",
        "max_seq_length",
        "seed",
        "learning_rate",
        "lr_schedule",
        "grad_accumulation_steps",
        "num_layers",
        "lora_parameters",
        "mask_prompt",
        "steps_per_report",
    }
    adapter = MlxLmAdapter(
        runner,
        check_environment=False,
        capabilities=MlxLoraCapabilities("0.31.3", frozenset(supported)),
    )

    result = adapter.train(
        MlxLoraConfig(
            model="models/Qwen3-8B-4bit",
            data_dir=tmp_path / "dataset",
            adapter_dir=tmp_path / "adapter",
            train_example_count=4,
            epochs=1,
        )
    )

    payload = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert "grad_checkpoint" not in payload


def test_default_search_space_is_versioned_and_starts_from_clean_base() -> None:
    from moonlightbox.training.mlx_adapter import default_lora_candidates

    candidates = default_lora_candidates()

    assert len(candidates) >= 3
    assert {candidate.rank for candidate in candidates} == {16, 32}
    assert {candidate.dropout for candidate in candidates} == {0.05, 0.1}
    assert {candidate.learning_rate for candidate in candidates} == {5e-6, 1e-5}
    assert len({candidate.seed for candidate in candidates}) >= 2
    assert all(candidate.resume_adapter_file is None for candidate in candidates)
    assert all(candidate.search_space_version for candidate in candidates)


def test_high_capacity_persona_search_covers_attention_and_mid_depth_mlp() -> None:
    from moonlightbox.training.mlx_adapter import high_capacity_persona_candidates

    candidates = high_capacity_persona_candidates()

    assert [candidate.candidate_id for candidate in candidates] == [
        "rank32-qkvo-all",
        "rank32-all-linear-12",
        "rank32-all-linear-18",
    ]
    assert {candidate.search_space_version for candidate in candidates} == {
        "moonlightbox-persona-capacity-search-v4"
    }
    all_linear = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == "rank32-all-linear-18"
    )
    assert {
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= set(all_linear.target_modules)
    assert all_linear.learning_rate == 4e-6
    assert all_linear.dropout == 0.1


def test_installed_mlx_lora_help_matches_required_yaml_schema() -> None:
    if importlib.util.find_spec("mlx_lm") is None:
        pytest.skip("未安装可选 mlx-lm 依赖")
    from moonlightbox.training.mlx_adapter import validate_mlx_lora_help

    capabilities = validate_mlx_lora_help(python_executable=sys.executable)

    assert capabilities.version
    assert {
        "model",
        "data",
        "adapter_path",
        "optimizer_config",
        "lr_schedule",
        "lora_parameters",
    } <= capabilities.config_fields


def test_streaming_runner_stops_before_max_iterations_and_accepts_active_exit(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import (
        MlxLmAdapter,
        MlxLoraConfig,
    )

    runner = StreamingEarlyStopRunner()
    adapter = MlxLmAdapter(runner, check_environment=False)
    result = adapter.train(
        MlxLoraConfig(
            model="models/Qwen3-8B-4bit",
            data_dir=tmp_path / "dataset",
            adapter_dir=tmp_path / "adapter",
            iterations=100,
            steps_per_evaluation=2,
            save_every=1,
            early_stopping_patience=2,
        )
    )

    assert runner.stopped is True
    assert runner.emitted_iterations == [2, 4, 6]
    assert result.early_stopped is True
    assert result.stopped_iteration == 6
    assert result.best_validation_iteration == 2
    assert result.best_checkpoint_iteration == 1
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.read_bytes() == b"checkpoint-1"


def test_patience_zero_disables_streaming_early_stop(tmp_path: Path) -> None:
    from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig

    runner = StreamingEarlyStopRunner()
    result = MlxLmAdapter(runner, check_environment=False).train(
        MlxLoraConfig(
            model="models/Qwen3-8B-4bit",
            data_dir=tmp_path / "dataset",
            adapter_dir=tmp_path / "adapter",
            iterations=100,
            steps_per_evaluation=2,
            save_every=1,
            early_stopping_patience=0,
        )
    )

    assert runner.emitted_iterations == [2, 4, 6, 8]
    assert result.early_stopped is False
    assert result.stopped_iteration is None


def test_best_selection_excludes_baseline_without_publishable_checkpoint(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig

    result = MlxLmAdapter(
        BaselineDominatesRunner(),
        check_environment=False,
    ).train(
        MlxLoraConfig(
            model="models/Qwen3-8B-4bit",
            data_dir=tmp_path / "dataset",
            adapter_dir=tmp_path / "adapter",
            iterations=10,
            save_every=1,
        )
    )

    selection = result.best_checkpoint_selection
    assert selection is not None
    assert selection.validation_loss == 0.8
    assert selection.validation_iteration == 2
    assert selection.checkpoint_iteration == 1
    assert selection.checkpoint.read_bytes() == b"publishable"


def test_missing_streamed_validation_falls_back_to_independent_valid_evaluation(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig

    class BufferedValidationRunner:
        def run(self, command, on_line, should_stop=None):  # type: ignore[no-untyped-def]
            config = yaml.safe_load(Path(command[-1]).read_text(encoding="utf-8"))
            adapter_dir = Path(config["adapter_path"])
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapters.safetensors").write_bytes(b"final")
            on_line("Iter 10: Train loss 1.234")
            return 0

    class RecoveringAdapter(MlxLmAdapter):
        def evaluate_validation_loss(self, config: MlxLoraConfig) -> float:
            assert (config.data_dir / "valid.jsonl").is_file()
            return 0.73

    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "valid.jsonl").write_text("{}\n", encoding="utf-8")
    result = RecoveringAdapter(
        BufferedValidationRunner(),
        check_environment=False,
    ).train(
        MlxLoraConfig(
            model="models/Qwen3-8B-4bit",
            data_dir=data_dir,
            adapter_dir=tmp_path / "adapter",
            iterations=10,
        )
    )

    selection = result.best_checkpoint_selection
    assert selection is not None
    assert selection.validation_loss == 0.73
    assert selection.checkpoint.name == "adapters.safetensors"
    assert "independent-valid-evaluation" in selection.mapping_rule


def test_masked_target_preflight_rejects_truncated_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.mlx_adapter import (
        MlxLoraConfig,
        MlxTrainingError,
        _validate_masked_target_windows,
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **_kwargs):  # type: ignore[no-untyped-def]
            length = 12 if len(messages) == 2 else 10
            return {"input_ids": list(range(length))}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    row = {
        "messages": [
            {"role": "user", "content": "context"},
            {"role": "assistant", "content": "target"},
        ]
    }
    for split in ("train", "valid"):
        (data_dir / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(MlxTrainingError, match="拒绝产生 NaN adapter"):
        _validate_masked_target_windows(
            MlxLoraConfig(
                model="models/fake",
                data_dir=data_dir,
                adapter_dir=tmp_path / "adapter",
                iterations=1,
                max_seq_length=10,
            )
        )


def test_subprocess_runner_sends_sigint_and_child_exits_normally() -> None:
    from moonlightbox.training.mlx_adapter import SubprocessRunner

    lines: list[str] = []
    stop = False

    def on_line(line: str) -> None:
        nonlocal stop
        lines.append(line)
        stop = line == "ready"

    exit_code = SubprocessRunner(stop_timeout_seconds=0.1).run(
        [
            sys.executable,
            "-c",
            (
                "import signal,sys,time;"
                "signal.signal(signal.SIGINT,lambda *_: sys.exit(0));"
                "print('ready',flush=True);"
                "time.sleep(10)"
            ),
        ],
        on_line,
        lambda: stop,
    )

    assert lines == ["ready"]
    assert exit_code == 0


def test_subprocess_runner_terminates_child_that_ignores_sigint() -> None:
    from moonlightbox.training.mlx_adapter import SubprocessRunner

    stop = False

    def on_line(line: str) -> None:
        nonlocal stop
        stop = line == "ready"

    started = time.monotonic()
    exit_code = SubprocessRunner(stop_timeout_seconds=0.05).run(
        [
            sys.executable,
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGINT,signal.SIG_IGN);"
                "print('ready',flush=True);"
                "time.sleep(10)"
            ),
        ],
        on_line,
        lambda: stop,
    )

    assert exit_code != 0
    assert time.monotonic() - started < 1


def test_subprocess_runner_preserves_ordinary_failure_exit_code() -> None:
    from moonlightbox.training.mlx_adapter import SubprocessRunner

    exit_code = SubprocessRunner(stop_timeout_seconds=0.05).run(
        [sys.executable, "-c", "raise SystemExit(7)"],
        lambda _line: None,
        lambda: False,
    )

    assert exit_code == 7


def test_real_subprocess_failure_is_not_classified_as_early_stop(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import (
        MlxLmAdapter,
        MlxLoraConfig,
        MlxTrainingError,
        SubprocessRunner,
    )

    class ExitSevenProcessRunner:
        def run(
            self,
            _command: list[str],
            on_line: object,
            should_stop: object | None = None,
        ) -> int:
            return SubprocessRunner(stop_timeout_seconds=0.05).run(
                [sys.executable, "-c", "raise SystemExit(7)"],
                on_line,
                should_stop,
            )

    adapter = MlxLmAdapter(ExitSevenProcessRunner(), check_environment=False)
    with pytest.raises(MlxTrainingError, match="退出码：7"):
        adapter.train(
            MlxLoraConfig(
                model="models/Qwen3-8B-4bit",
                data_dir=tmp_path / "dataset",
                adapter_dir=tmp_path / "adapter",
                iterations=10,
            )
        )


def test_unsupported_mlx_version_fails_closed(tmp_path: Path) -> None:
    from moonlightbox.training.mlx_adapter import (
        MlxEnvironmentError,
        MlxLmAdapter,
        MlxLoraCapabilities,
        MlxLoraConfig,
    )

    adapter = MlxLmAdapter(
        FakeRunner(),
        check_environment=False,
        capabilities=MlxLoraCapabilities(
            version="0.99.0",
            config_fields=_fake_supported_fields(),
        ),
    )

    with pytest.raises(MlxEnvironmentError, match="重新验证"):
        adapter.train(
            MlxLoraConfig(
                model="models/Qwen3-8B-4bit",
                data_dir=tmp_path / "dataset",
                adapter_dir=tmp_path / "adapter",
            )
        )


def test_subprocess_runner_kills_ignoring_process_group() -> None:
    import os

    from moonlightbox.training.mlx_adapter import SubprocessRunner

    child_pid: int | None = None
    stop = False

    def on_line(line: str) -> None:
        nonlocal child_pid, stop
        child_pid = int(line)
        stop = True

    exit_code = SubprocessRunner(
        stop_timeout_seconds=0.05,
        terminate_timeout_seconds=0.05,
    ).run(
        [
            sys.executable,
            "-c",
            (
                "import signal,subprocess,sys,time;"
                "signal.signal(signal.SIGINT,signal.SIG_IGN);"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time;"
                "signal.signal(signal.SIGINT,signal.SIG_IGN);"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "time.sleep(10)']);"
                "print(p.pid,flush=True);"
                "time.sleep(10)"
            ),
        ],
        on_line,
        lambda: stop,
    )

    assert exit_code != 0
    assert child_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_subprocess_runner_cleans_child_after_parent_exits_zero_on_sigint() -> None:
    import os

    from moonlightbox.training.mlx_adapter import SubprocessRunner

    child_pid: int | None = None
    stop = False

    def on_line(line: str) -> None:
        nonlocal child_pid, stop
        child_pid = int(line)
        stop = True

    exit_code = SubprocessRunner(
        stop_timeout_seconds=0.05,
        terminate_timeout_seconds=0.05,
    ).run(
        [
            sys.executable,
            "-c",
            (
                "import signal,subprocess,sys,time;"
                "signal.signal(signal.SIGINT,lambda *_: sys.exit(0));"
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time;"
                "signal.signal(signal.SIGINT,signal.SIG_IGN);"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "time.sleep(10)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                "print(p.pid,flush=True);"
                "time.sleep(10)"
            ),
        ],
        on_line,
        lambda: stop,
    )

    assert exit_code == 0
    assert child_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def _fake_supported_fields() -> frozenset[str]:
    return frozenset(
        {
            "model",
            "data",
            "adapter_path",
            "train",
            "fine_tune_type",
            "optimizer",
            "optimizer_config",
            "batch_size",
            "iters",
            "val_batches",
            "steps_per_eval",
            "steps_per_report",
            "save_every",
            "max_seq_length",
            "seed",
            "learning_rate",
            "lr_schedule",
            "grad_checkpoint",
            "grad_accumulation_steps",
            "num_layers",
            "lora_parameters",
            "mask_prompt",
        }
    )
