import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Thread
from typing import Protocol, TextIO, cast

import yaml

ProgressCallback = Callable[[dict[str, object]], None]
CHECKPOINT_STRATEGIES = {
    "0.31.3": (
        "mlx-lm-0.31.3-validation-before-step:"
        "checkpoint=nearest-saved-checkpoint-to-pre-step-weights"
    ),
}
GENERATED_CHECKPOINT_PATTERN = re.compile(
    r"^(?P<iteration>[0-9]{7})_adapters\.safetensors$"
)


class CommandRunner(Protocol):
    def run(
        self,
        command: list[str],
        on_line: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> int: ...


class MlxEnvironmentError(RuntimeError):
    pass


class MlxTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MlxLoraConfig:
    model: str
    data_dir: Path
    adapter_dir: Path
    iterations: int | None = 600
    train_example_count: int = 0
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    validation_batches: int = 10
    steps_per_evaluation: int = 50
    save_every: int = 100
    max_seq_length: int = 2048
    seed: int = 42
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_checkpoint: bool = True
    num_layers: int = 16
    target_modules: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    early_stopping_patience: int = 0
    # Full persona runs need every saved checkpoint for held-out style
    # selection. This is an orchestration control and is intentionally not
    # serialized into the mlx-lm YAML.
    preserve_checkpoints: bool = False
    resume_adapter_file: Path | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def steps_per_epoch(self) -> int:
        if self.train_example_count < 1:
            raise ValueError("计算有效 epoch 需要正数训练样本量")
        return max(1, math.ceil(self.train_example_count / self.effective_batch_size))

    @property
    def resolved_iterations(self) -> int:
        if self.train_example_count > 0:
            return self.steps_per_epoch * self.epochs
        if self.iterations is None or self.iterations < 1:
            raise ValueError("训练步数必须由正数 iterations 或训练集大小推导")
        return self.iterations


@dataclass(frozen=True)
class MlxLoraCapabilities:
    version: str
    config_fields: frozenset[str]


@dataclass(frozen=True)
class LoraCandidate:
    candidate_id: str
    search_space_version: str
    rank: int
    alpha: int
    num_layers: int
    target_modules: tuple[str, ...]
    dropout: float
    learning_rate: float
    seed: int
    resume_adapter_file: Path | None = None


def default_lora_candidates() -> tuple[LoraCandidate, ...]:
    """返回经过版本标记的高保真 LoRA 搜索空间。"""

    attention = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    query_value = ("self_attn.q_proj", "self_attn.v_proj")
    version_name = "moonlightbox-lora-search-v1"
    return (
        LoraCandidate(
            "rank16-attn16",
            version_name,
            16,
            32,
            16,
            attention,
            0.05,
            1e-5,
            17,
        ),
        LoraCandidate(
            "rank32-attn24",
            version_name,
            32,
            64,
            24,
            attention,
            0.1,
            5e-6,
            29,
        ),
        LoraCandidate(
            "rank32-qv-all",
            version_name,
            32,
            64,
            -1,
            query_value,
            0.05,
            1e-5,
            43,
        ),
    )


def high_capacity_persona_candidates() -> tuple[LoraCandidate, ...]:
    """搜索 Q/V 容量不足与全层 MLP 失稳之间的中间适配强度。"""

    attention = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    all_linear = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    version_name = "moonlightbox-persona-capacity-search-v4"
    return (
        LoraCandidate(
            "rank32-qkvo-all",
            version_name,
            32,
            64,
            -1,
            attention,
            0.05,
            7.5e-6,
            47,
        ),
        LoraCandidate(
            "rank32-all-linear-12",
            version_name,
            32,
            64,
            12,
            all_linear,
            0.1,
            5e-6,
            53,
        ),
        LoraCandidate(
            "rank32-all-linear-18",
            version_name,
            32,
            64,
            18,
            all_linear,
            0.1,
            4e-6,
            61,
        ),
    )


@dataclass(frozen=True)
class BestCheckpointSelection:
    validation_loss: float
    validation_iteration: int
    checkpoint_iteration: int
    checkpoint: Path
    approximation_steps: int = 0
    mapping_rule: str = (
        "mlx-lm-0.31.3-validation-before-step:"
        "checkpoint=nearest-saved-checkpoint-to-pre-step-weights"
    )


@dataclass(frozen=True)
class TrainingResult:
    adapter_dir: Path
    iterations: int
    config_path: Path | None = None
    mlx_version: str = "unknown"
    validation_curve: tuple[dict[str, float | int], ...] = ()
    best_checkpoint_selection: BestCheckpointSelection | None = None
    command: tuple[str, ...] = ()
    early_stopped: bool = False
    stopped_iteration: int | None = None

    @property
    def best_checkpoint(self) -> Path | None:
        selection = self.best_checkpoint_selection
        return selection.checkpoint if selection is not None else None

    @property
    def best_validation_iteration(self) -> int | None:
        selection = self.best_checkpoint_selection
        return selection.validation_iteration if selection is not None else None

    @property
    def best_checkpoint_iteration(self) -> int | None:
        selection = self.best_checkpoint_selection
        return selection.checkpoint_iteration if selection is not None else None

    @property
    def checkpoint_mapping_rule(self) -> str:
        selection = self.best_checkpoint_selection
        return (
            selection.mapping_rule
            if selection is not None
            else (
                "mlx-lm-0.31.3-validation-before-step:"
                "checkpoint=nearest-saved-checkpoint-to-pre-step-weights"
            )
        )


class MlxLmAdapter:
    def __init__(
        self,
        runner: CommandRunner | None = None,
        check_environment: bool = True,
        python_executable: str = "python",
        capabilities: MlxLoraCapabilities | None = None,
    ) -> None:
        self._runner = runner or SubprocessRunner()
        self._check_environment = check_environment
        self._python_executable = python_executable
        self._capabilities = capabilities

    @property
    def capabilities(self) -> MlxLoraCapabilities:
        return self._capabilities or detect_mlx_lora_capabilities()

    def train(
        self,
        config: MlxLoraConfig,
        on_progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        if self._check_environment:
            _validate_environment()
            _validate_masked_target_windows(config)
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        callback = on_progress or (lambda _: None)
        capabilities = self.capabilities
        mapping_rule = CHECKPOINT_STRATEGIES.get(capabilities.version)
        if mapping_rule is None:
            raise MlxEnvironmentError(
                f"mlx-lm {capabilities.version} 尚未验证 checkpoint 映射，"
                "请重新验证后再训练"
            )
        config_path = config.adapter_dir / "training.yaml"
        _atomic_write_yaml(config_path, normalized_mlx_yaml(config, capabilities))
        command = [
            self._python_executable,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(config_path),
        ]
        validation_curve: list[dict[str, float | int]] = []
        best_selection: BestCheckpointSelection | None = None
        stale_validations = 0
        stop_requested = False
        stopped_iteration: int | None = None
        latest_iteration = 0
        processed_validations: set[int] = set()

        def refresh_selection() -> None:
            nonlocal best_selection
            nonlocal stale_validations
            nonlocal stop_requested
            nonlocal stopped_iteration
            best_selection = _best_checkpoint_selection(
                config.adapter_dir,
                validation_curve,
                mapping_rule,
            )
            mapped = _mapped_validation_selections(
                config.adapter_dir,
                validation_curve,
                mapping_rule,
            )
            for selection in sorted(
                mapped,
                key=lambda item: item.validation_iteration,
            ):
                if (
                    selection.validation_iteration in processed_validations
                    or selection.approximation_steps > 1
                ):
                    continue
                processed_validations.add(selection.validation_iteration)
                previous = [
                    item
                    for item in mapped
                    if item.validation_iteration < selection.validation_iteration
                    and item.validation_iteration in processed_validations
                ]
                if not previous or selection.validation_loss < min(
                    item.validation_loss for item in previous
                ):
                    stale_validations = 0
                else:
                    stale_validations += 1
                if (
                    config.early_stopping_patience > 0
                    and stale_validations >= config.early_stopping_patience
                    and best_selection is not None
                ):
                    stop_requested = True
                    stopped_iteration = selection.validation_iteration
            if best_selection is not None and not config.preserve_checkpoints:
                latest = _latest_generated_checkpoint(config.adapter_dir)
                _prune_adapter_checkpoints(
                    config.adapter_dir,
                    keep=best_selection.checkpoint,
                    latest=latest,
                )

        def handle_line(line: str) -> None:
            nonlocal best_selection
            nonlocal latest_iteration
            match = re.search(
                r"Iter\s+(\d+):\s+Train loss\s+([-+0-9.eE]+)",
                line,
            )
            if match:
                latest_iteration = int(match.group(1))
                callback(
                    {
                        "stage": "training",
                        "iteration": latest_iteration,
                        "loss": float(match.group(2)),
                    }
                )
                return
            validation_match = re.search(
                r"Iter\s+(\d+):\s+Val loss\s+([-+0-9.eE]+)",
                line,
            )
            if validation_match:
                iteration = int(validation_match.group(1))
                latest_iteration = max(latest_iteration, iteration)
                validation_loss = float(validation_match.group(2))
                point: dict[str, float | int] = {
                    "iteration": iteration,
                    "validation_loss": validation_loss,
                }
                validation_curve.append(point)
                callback(
                    {
                        "stage": "training",
                        **point,
                    }
                )
                refresh_selection()
            elif "Saved adapter" in line:
                latest = _latest_generated_checkpoint(config.adapter_dir)
                saved_iteration = (
                    _checkpoint_iteration(latest)
                    if latest is not None
                    else latest_iteration
                )
                callback(
                    {
                        "stage": "saved",
                        "iteration": saved_iteration,
                    }
                )
                refresh_selection()

        exit_code = self._runner.run(
            command,
            handle_line,
            lambda: stop_requested,
        )
        if exit_code != 0 and not stop_requested:
            raise MlxTrainingError(f"MLX-LM 训练失败，退出码：{exit_code}")
        if best_selection is None:
            refresh_selection()
        # mlx-lm writes its final adapter even when output buffering or an early
        # stop prevents a save line from being paired with the preceding
        # validation report.  Do not discard a successfully trained candidate
        # solely because the streaming transcript was incomplete: evaluate the
        # final adapter against a temporary, validation-only dataset instead.
        final_adapter = config.adapter_dir / "adapters.safetensors"
        if best_selection is None and final_adapter.is_file():
            validation_loss = self.evaluate_validation_loss(config)
            validation_iteration = max(2, latest_iteration or config.resolved_iterations)
            validation_curve.append(
                {
                    "iteration": validation_iteration,
                    "validation_loss": validation_loss,
                }
            )
            best_selection = BestCheckpointSelection(
                validation_loss=validation_loss,
                validation_iteration=validation_iteration,
                checkpoint_iteration=config.resolved_iterations,
                checkpoint=final_adapter,
                approximation_steps=0,
                mapping_rule=(
                    f"{mapping_rule}:fallback=independent-valid-evaluation-of-final-adapter"
                ),
            )
        return TrainingResult(
            config.adapter_dir,
            config.resolved_iterations,
            config_path,
            capabilities.version,
            tuple(validation_curve),
            best_selection,
            tuple(command),
            stop_requested,
            stopped_iteration,
        )

    def evaluate_validation_loss(self, config: MlxLoraConfig) -> float:
        """Evaluate the final adapter on valid.jsonl without touching test data."""

        valid_path = config.data_dir / "valid.jsonl"
        if not valid_path.is_file():
            raise MlxTrainingError("训练数据缺少 valid.jsonl，无法复验候选")
        with tempfile.TemporaryDirectory(prefix="moonlightbox-valid-") as directory:
            evaluation_data = Path(directory)
            shutil.copy2(valid_path, evaluation_data / "test.jsonl")
            evaluation_config = MlxLoraConfig(
                **{
                    **config.__dict__,
                    "data_dir": evaluation_data,
                }
            )
            return self.evaluate_loss(evaluation_config, config.adapter_dir)

    def evaluate_loss(self, config: MlxLoraConfig, checkpoint: Path) -> float:
        """对已恢复的 adapter 运行独立 test split loss 复验。"""

        if self._check_environment:
            _validate_environment()
        capabilities = self.capabilities
        required_fields = {
            "model",
            "data",
            "adapter_path",
            "train",
            "test",
            "test_batches",
            "batch_size",
            "max_seq_length",
            "seed",
        }
        unsupported = required_fields - capabilities.config_fields
        if unsupported:
            fields = "、".join(sorted(unsupported))
            raise MlxEnvironmentError(
                f"mlx-lm {capabilities.version} 不支持独立 test loss 字段：{fields}"
            )
        payload: dict[str, object] = {
            "model": config.model,
            "data": str(config.data_dir),
            "adapter_path": str(checkpoint),
            "train": False,
            "test": True,
            "test_batches": config.validation_batches,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "seed": config.seed,
        }
        config_path = checkpoint / "evaluation.yaml"
        _atomic_write_yaml(
            config_path,
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        )
        command = [
            self._python_executable,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(config_path),
        ]
        test_loss: float | None = None
        observed_lines: list[str] = []

        def handle_line(line: str) -> None:
            nonlocal test_loss
            observed_lines.append(line)
            match = re.search(r"Test loss\s+([-+0-9.eE]+)", line)
            if match:
                test_loss = float(match.group(1))

        exit_code = self._runner.run(command, handle_line)
        if exit_code != 0:
            raise MlxTrainingError(f"MLX-LM 独立测试失败，退出码：{exit_code}")
        if test_loss is None or not math.isfinite(test_loss):
            tail = " | ".join(observed_lines[-8:])
            raise MlxTrainingError(
                "MLX-LM 独立测试未返回有效 test loss"
                + (f"；输出：{tail}" if tail else "；没有捕获到输出")
            )
        return test_loss


def detect_mlx_lora_capabilities() -> MlxLoraCapabilities:
    """从当前安装包读取真实配置 schema，避免向旧版本传入未知字段。"""

    try:
        mlx_version = version("mlx-lm")
    except PackageNotFoundError as error:
        raise MlxEnvironmentError("未安装官方 mlx-lm 依赖") from error
    try:
        module = import_module("mlx_lm.lora")
        defaults = module.__dict__.get("CONFIG_DEFAULTS")
    except ImportError as error:
        raise MlxEnvironmentError("无法读取 mlx-lm LoRA 配置 schema") from error
    if not isinstance(defaults, dict):
        raise MlxEnvironmentError("mlx-lm LoRA 配置 schema 格式无效")
    return MlxLoraCapabilities(mlx_version, frozenset(str(key) for key in defaults))


def validate_mlx_lora_help(
    *,
    python_executable: str = "python",
) -> MlxLoraCapabilities:
    """只读执行 --help，并与安装包 schema 交叉验证关键入口。"""

    capabilities = detect_mlx_lora_capabilities()
    completed = subprocess.run(
        [python_executable, "-m", "mlx_lm", "lora", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or "--config" not in completed.stdout:
        raise MlxEnvironmentError("当前 mlx-lm 不支持 LoRA YAML 配置")
    return capabilities


def normalized_mlx_yaml(
    config: MlxLoraConfig,
    capabilities: MlxLoraCapabilities | None = None,
) -> str:
    """生成与实际训练文件完全一致的规范化 YAML。"""

    resolved = capabilities or detect_mlx_lora_capabilities()
    payload = _build_yaml_payload(config, resolved)
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)


def _build_yaml_payload(
    config: MlxLoraConfig,
    capabilities: MlxLoraCapabilities,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": config.model,
        "data": str(config.data_dir),
        "adapter_path": str(config.adapter_dir),
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {
            "adamw": {
                "weight_decay": config.weight_decay,
            }
        },
        "batch_size": config.batch_size,
        "iters": config.resolved_iterations,
        "val_batches": config.validation_batches,
        "steps_per_eval": config.steps_per_evaluation,
        "steps_per_report": min(10, config.steps_per_evaluation),
        "save_every": config.save_every,
        "max_seq_length": config.max_seq_length,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [
                config.learning_rate,
                max(1, config.resolved_iterations - config.warmup_steps),
            ],
            "warmup": min(config.warmup_steps, config.resolved_iterations - 1),
            "warmup_init": 0.0,
        },
        "grad_accumulation_steps": config.gradient_accumulation_steps,
        "num_layers": config.num_layers,
        "lora_parameters": {
            "rank": config.rank,
            "scale": config.alpha / config.rank,
            "dropout": config.dropout,
            "keys": list(config.target_modules),
        },
        "mask_prompt": True,
    }
    if config.resume_adapter_file is not None:
        payload["resume_adapter_file"] = str(config.resume_adapter_file)
    if "grad_checkpoint" in capabilities.config_fields:
        payload["grad_checkpoint"] = config.grad_checkpoint
    unsupported = set(payload) - capabilities.config_fields
    if unsupported:
        fields = "、".join(sorted(unsupported))
        raise MlxEnvironmentError(
            f"mlx-lm {capabilities.version} 不支持配置字段：{fields}"
        )
    return payload


def _atomic_write_yaml(path: Path, content: str) -> None:
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _validate_masked_target_windows(config: MlxLoraConfig) -> None:
    """Fail before training if MLX truncation would remove supervision tokens."""

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            local_files_only=Path(config.model).exists(),
            trust_remote_code=True,
        )
    except Exception as error:
        raise MlxEnvironmentError("无法加载 tokenizer 进行目标窗口预检") from error
    failures: list[str] = []
    for split in ("train", "valid"):
        path = config.data_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                failures.append(f"{split}:{line_number}:invalid_messages")
                continue
            full = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
            )["input_ids"]
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=(messages[-1].get("role") == "assistant"),
                tokenize=True,
                return_dict=True,
            )["input_ids"]
            if len(prompt) >= config.max_seq_length or len(full) > config.max_seq_length:
                failures.append(
                    f"{split}:{line_number}:tokens={len(full)}:target_offset={len(prompt)}"
                )
                if len(failures) >= 20:
                    break
        if len(failures) >= 20:
            break
    if failures:
        raise MlxTrainingError(
            "mask_prompt 目标会被 max_seq_length 截断，拒绝产生 NaN adapter："
            + ", ".join(failures)
        )


def _best_checkpoint_selection(
    adapter_dir: Path,
    validation_curve: list[dict[str, float | int]],
    mapping_rule: str,
) -> BestCheckpointSelection | None:
    selections = _mapped_validation_selections(
        adapter_dir,
        validation_curve,
        mapping_rule,
    )
    if not selections:
        return None
    return min(
        selections,
        key=lambda item: (
            item.validation_loss,
            item.approximation_steps,
            item.validation_iteration,
            item.checkpoint_iteration,
        ),
    )


def _mapped_validation_selections(
    adapter_dir: Path,
    validation_curve: list[dict[str, float | int]],
    mapping_rule: str,
) -> list[BestCheckpointSelection]:
    checkpoints = _generated_adapter_checkpoints(adapter_dir)
    selections: list[BestCheckpointSelection] = []
    for point in validation_curve:
        validation_iteration = int(point["iteration"])
        if validation_iteration <= 1 or not checkpoints:
            continue
        target_iteration = validation_iteration - 1
        checkpoint = min(
            checkpoints,
            key=lambda path: (
                abs(_checkpoint_iteration(path) - target_iteration),
                _checkpoint_iteration(path),
            ),
        )
        checkpoint_iteration = _checkpoint_iteration(checkpoint)
        selections.append(
            BestCheckpointSelection(
                validation_loss=float(point["validation_loss"]),
                validation_iteration=validation_iteration,
                checkpoint_iteration=checkpoint_iteration,
                checkpoint=checkpoint,
                approximation_steps=abs(checkpoint_iteration - target_iteration),
                mapping_rule=mapping_rule,
            )
        )
    return selections


def _generated_adapter_checkpoints(adapter_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in adapter_dir.iterdir()
        if path.is_file() and GENERATED_CHECKPOINT_PATTERN.fullmatch(path.name)
    )


def _checkpoint_iteration(path: Path) -> int:
    match = GENERATED_CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"不是受管 checkpoint：{path.name}")
    return int(match.group("iteration"))


def _latest_generated_checkpoint(adapter_dir: Path) -> Path | None:
    checkpoints = _generated_adapter_checkpoints(adapter_dir)
    return max(checkpoints, key=_checkpoint_iteration) if checkpoints else None


def _prune_adapter_checkpoints(
    adapter_dir: Path,
    *,
    keep: Path,
    latest: Path | None = None,
) -> None:
    """仅保留当前最佳和最新 checkpoint，防止训练耗尽本地磁盘。"""

    retained = {keep.resolve()}
    if latest is not None:
        retained.add(latest.resolve())
    for checkpoint in _generated_adapter_checkpoints(adapter_dir):
        if checkpoint.resolve() not in retained:
            checkpoint.unlink()


def cleanup_generated_checkpoints(
    output_dir: Path,
    *,
    include_latest: bool = False,
    keep: tuple[Path, ...] = (),
) -> None:
    """只清理任务输出目录内严格命名的训练权重文件。"""

    root = output_dir.resolve()
    retained = {path.resolve() for path in keep}
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.resolve() in retained:
            continue
        if not path.resolve().is_relative_to(root):
            continue
        if GENERATED_CHECKPOINT_PATTERN.fullmatch(path.name) or (
            include_latest and path.name == "adapters.safetensors"
        ):
            path.unlink()


class SubprocessRunner:
    def __init__(
        self,
        *,
        stop_timeout_seconds: float = 5.0,
        terminate_timeout_seconds: float = 5.0,
    ) -> None:
        if stop_timeout_seconds <= 0:
            raise ValueError("停止超时必须大于零")
        if terminate_timeout_seconds <= 0:
            raise ValueError("终止超时必须大于零")
        self._stop_timeout_seconds = stop_timeout_seconds
        self._terminate_timeout_seconds = terminate_timeout_seconds

    def run(
        self,
        command: list[str],
        on_line: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stream = cast(TextIO | None, process.stdout)
            stop_sent = False
            watcher: Thread | None = None
            if stream is not None:
                for line in stream:
                    on_line(line.rstrip())
                    if (
                        not stop_sent
                        and should_stop is not None
                        and should_stop()
                        and process.poll() is None
                    ):
                        os.killpg(process.pid, signal.SIGINT)
                        stop_sent = True
                        watcher = Thread(
                            target=_terminate_if_running,
                            args=(
                                process,
                                self._stop_timeout_seconds,
                                self._terminate_timeout_seconds,
                            ),
                            daemon=True,
                        )
                        watcher.start()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=self._terminate_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        exit_code = process.wait()
        if watcher is not None:
            watcher.join(
                timeout=self._stop_timeout_seconds
                + self._terminate_timeout_seconds
                + 1
            )
        return exit_code


def _terminate_if_running(
    process: subprocess.Popen[str],
    sigint_timeout_seconds: float,
    terminate_timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + sigint_timeout_seconds
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    deadline = time.monotonic() + terminate_timeout_seconds
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A managed macOS worker can reap the child while denying a later
        # process-group probe. Treat cleanup as complete instead of crashing
        # the background terminator after a successful early stop.
        return False
    return True


def _validate_environment() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise MlxEnvironmentError("MLX-LM 训练仅支持 Apple Silicon")
    if importlib.util.find_spec("mlx_lm") is None:
        raise MlxEnvironmentError("未安装官方 mlx-lm 依赖")
