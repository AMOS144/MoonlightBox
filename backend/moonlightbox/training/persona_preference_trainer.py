import json
import math
import shutil
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from moonlightbox.training.persona_preference import (
    iterate_preference_batches,
    load_tokenized_preference_dataset,
)
from moonlightbox.training.persona_preference_loss import persona_simpo_loss


@dataclass(frozen=True)
class PreferenceTrainingConfig:
    base_model: str
    initial_adapter_file: str
    data_dir: str
    output_dir: str
    iterations: int = 50
    learning_rate: float = 5e-7
    num_layers: int = 12
    rank: int = 32
    scale: float = 2.0
    dropout: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    beta: float = 2.0
    target_margin: float = 0.5
    sft_weight: float = 0.2
    seed: int = 71
    validation_batches: int = 50
    evaluation_interval: int = 20
    checkpoint_interval: int = 20

    def validate(self) -> None:
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("偏好训练迭代数和 batch size 必须为正数")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("偏好训练梯度累积必须为正数")
        if self.learning_rate <= 0 or self.beta <= 0:
            raise ValueError("偏好训练 learning rate 和 beta 必须为正数")
        if not 0 <= self.sft_weight <= 1:
            raise ValueError("偏好训练 SFT 权重必须位于 0 到 1")
        if min(
            self.validation_batches,
            self.evaluation_interval,
            self.checkpoint_interval,
        ) < 1:
            raise ValueError("偏好训练验证与 checkpoint 间隔必须为正数")
        if not Path(self.initial_adapter_file).is_file():
            raise ValueError("偏好训练初始 adapter 不存在")
        if not (Path(self.data_dir) / "train.jsonl").is_file():
            raise ValueError("偏好训练数据不存在")


@dataclass(frozen=True)
class PreferenceTrainingResult:
    adapter_file: str
    iterations: int
    train_pair_count: int
    valid_pair_count: int
    skipped_oversized_train_pairs: int
    skipped_oversized_valid_pairs: int
    config_path: str
    metrics_path: str
    train_metrics: tuple[dict[str, float | int], ...]
    validation_metrics: tuple[dict[str, float | int], ...]
    selected_iteration: int
    selected_validation_loss: float


class PreferenceMetricsCallback:
    """Keep the preference curve as an auditable checkpoint-selection input."""

    def __init__(self) -> None:
        self.train_metrics: list[dict[str, float | int]] = []
        self.validation_metrics: list[dict[str, float | int]] = []

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        self.train_metrics.append(_numeric_metrics(train_info))

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:
        self.validation_metrics.append(_numeric_metrics(val_info))


def run_persona_preference_training(
    config: PreferenceTrainingConfig,
) -> PreferenceTrainingResult:
    config.validate()
    mx.random.seed(config.seed)
    model, tokenizer = load(
        config.base_model,
        tokenizer_config={"trust_remote_code": True},
    )
    model.freeze()
    structure = _initial_lora_structure(config)
    linear_to_lora_layers(
        model,
        structure["num_layers"],
        {
            "rank": structure["rank"],
            "scale": structure["scale"],
            "dropout": structure["dropout"],
            "keys": structure["keys"],
        },
    )
    model.load_weights(config.initial_adapter_file, strict=False)
    print_trainable_parameters(model)
    data_dir = Path(config.data_dir)
    train_dataset = load_tokenized_preference_dataset(
        data_dir / "train.jsonl",
        tokenizer,
        max_seq_length=config.max_seq_length,
    )
    valid_dataset = load_tokenized_preference_dataset(
        data_dir / "valid.jsonl",
        tokenizer,
        max_seq_length=config.max_seq_length,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    write_preference_adapter_config(config, output_dir)
    config_path = output_dir / "preference_config.json"
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int):
        pad_token_id = 0
    iterator = partial(
        iterate_preference_batches,
        pad_token_id=pad_token_id,
        seed=config.seed,
    )
    loss = partial(
        persona_simpo_loss,
        beta=config.beta,
        target_margin=config.target_margin,
        sft_weight=config.sft_weight,
    )
    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=config.iterations,
        val_batches=config.validation_batches,
        steps_per_report=5,
        steps_per_eval=config.evaluation_interval,
        steps_per_save=config.checkpoint_interval,
        adapter_file=str(adapter_file),
        max_seq_length=config.max_seq_length,
        grad_checkpoint=True,
        grad_accumulation_steps=config.gradient_accumulation_steps,
    )
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=0.01,
    )
    metrics_callback = PreferenceMetricsCallback()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        args=training_args,
        loss=loss,
        iterate_batches=iterator,
        training_callback=metrics_callback,
    )
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)
    selected_iteration, selected_validation_loss = _select_best_checkpoint(
        output_dir,
        metrics_callback.validation_metrics,
        final_iteration=config.iterations,
    )
    metrics_path = output_dir / "preference_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "moonlightbox-persona-preference-metrics-v1",
                "train": metrics_callback.train_metrics,
                "validation": metrics_callback.validation_metrics,
                "selected_iteration": selected_iteration,
                "selected_validation_loss": selected_validation_loss,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return PreferenceTrainingResult(
        adapter_file=str(adapter_file),
        iterations=config.iterations,
        train_pair_count=len(train_dataset),
        valid_pair_count=len(valid_dataset),
        skipped_oversized_train_pairs=train_dataset.skipped_oversized,
        skipped_oversized_valid_pairs=valid_dataset.skipped_oversized,
        config_path=str(config_path),
        metrics_path=str(metrics_path),
        train_metrics=tuple(metrics_callback.train_metrics),
        validation_metrics=tuple(metrics_callback.validation_metrics),
        selected_iteration=selected_iteration,
        selected_validation_loss=selected_validation_loss,
    )


def _select_best_checkpoint(
    output_dir: Path,
    validation_metrics: list[dict[str, float | int]],
    *,
    final_iteration: int,
) -> tuple[int, float]:
    """Publish the lowest validated checkpoint instead of blindly using the final step."""

    candidates: list[tuple[float, int, Path]] = []
    for metric in validation_metrics:
        iteration_value = metric.get("iteration")
        loss_value = metric.get("val_loss")
        if not isinstance(iteration_value, int | float) or not isinstance(
            loss_value, int | float
        ):
            continue
        iteration = int(iteration_value)
        loss = float(loss_value)
        checkpoint = output_dir / f"{iteration:07d}_adapters.safetensors"
        if math.isfinite(loss) and checkpoint.is_file():
            candidates.append((loss, iteration, checkpoint))
    final_loss = next(
        (
            float(metric["val_loss"])
            for metric in reversed(validation_metrics)
            if metric.get("iteration") == final_iteration
            and isinstance(metric.get("val_loss"), int | float)
        ),
        math.inf,
    )
    adapter_file = output_dir / "adapters.safetensors"
    if adapter_file.is_file() and math.isfinite(final_loss):
        candidates.append((final_loss, final_iteration, adapter_file))
    if not candidates and adapter_file.is_file():
        fallback_loss = next(
            (
                float(metric["val_loss"])
                for metric in reversed(validation_metrics)
                if isinstance(metric.get("val_loss"), int | float)
                and math.isfinite(float(metric["val_loss"]))
            ),
            math.inf,
        )
        if math.isfinite(fallback_loss):
            candidates.append((fallback_loss, final_iteration, adapter_file))
    if not candidates:
        raise ValueError("偏好训练没有可发布的验证 checkpoint")
    loss, iteration, checkpoint = min(candidates, key=lambda item: (item[0], item[1]))
    if checkpoint != adapter_file:
        temporary = output_dir / ".selected-adapters.safetensors.tmp"
        shutil.copy2(checkpoint, temporary)
        temporary.replace(adapter_file)
    return iteration, loss


def _initial_lora_structure(config: PreferenceTrainingConfig) -> dict[str, object]:
    source = Path(config.initial_adapter_file).parent / "adapter_config.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("初始 adapter 结构配置无效") from error
    lora = payload.get("lora_parameters") if isinstance(payload, dict) else None
    num_layers = payload.get("num_layers") if isinstance(payload, dict) else None
    rank = lora.get("rank") if isinstance(lora, dict) else None
    keys = lora.get("keys") if isinstance(lora, dict) else None
    scale = lora.get("scale", config.scale) if isinstance(lora, dict) else None
    dropout = lora.get("dropout", config.dropout) if isinstance(lora, dict) else None
    if (
        not isinstance(num_layers, int)
        or num_layers < 1
        or not isinstance(rank, int)
        or rank < 1
        or not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or not isinstance(scale, int | float)
        or not isinstance(dropout, int | float)
    ):
        raise ValueError("初始 adapter 缺少完整 LoRA 结构")
    return {
        "num_layers": num_layers,
        "rank": rank,
        "keys": tuple(keys),
        "scale": float(scale),
        "dropout": float(dropout),
    }


def materialize_preference_checkpoint(
    config: PreferenceTrainingConfig,
    checkpoint_file: Path,
    output_dir: Path,
) -> Path:
    """Turn a numbered weight snapshot into a directory loadable by mlx-lm."""

    if not checkpoint_file.is_file():
        raise ValueError("偏好训练 checkpoint 不存在")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "adapters.safetensors"
    shutil.copy2(checkpoint_file, destination)
    write_preference_adapter_config(config, output_dir)
    return destination


def write_preference_adapter_config(
    config: PreferenceTrainingConfig,
    output_dir: Path,
) -> Path:
    """Preserve the exact LoRA structure contract required by mlx-lm.load."""

    source = Path(config.initial_adapter_file).parent / "adapter_config.json"
    if not source.is_file():
        raise ValueError("初始 adapter 缺少 MLX 结构配置")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("初始 adapter 结构配置无效")
    payload.update(
        {
            "adapter_path": str(output_dir),
            "resume_adapter_file": config.initial_adapter_file,
            "preference_training": {
                "schema_version": "moonlightbox-persona-simpo-v1",
                "beta": config.beta,
                "target_margin": config.target_margin,
                "sft_weight": config.sft_weight,
            },
        }
    )
    path = output_dir / "adapter_config.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _numeric_metrics(values: dict[str, object]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            result[key] = value
            continue
        if hasattr(value, "item"):
            scalar = value.item()
            if isinstance(scalar, int | float) and not isinstance(scalar, bool):
                result[key] = scalar
    return result
