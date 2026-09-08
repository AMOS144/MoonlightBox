# ruff: noqa: E501

"""Linux 的偏好阶段训练。

偏好样本中的 chosen 回复被作为监督目标继续微调当前 PEFT adapter。拒绝样本保留在
审计数据中，不会被写入训练目标；这一阶段不依赖 MLX 的自定义训练器。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from moonlightbox.training.peft_adapter import PeftLmAdapter, PeftLoraConfig


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
    # 偏好阶段复用同一张 4GB GPU，不能把基础阶段的显存预算放大。
    max_seq_length: int = 512
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
        if not Path(self.initial_adapter_file).is_file():
            raise ValueError("偏好训练初始 PEFT adapter 不存在")
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


def run_persona_preference_training(config: PreferenceTrainingConfig) -> PreferenceTrainingResult:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / "chosen-supervision"
    train_count = _write_chosen_dataset(
        Path(config.data_dir) / "train.jsonl", prepared_dir / "train.jsonl"
    )
    valid_count = _write_chosen_dataset(
        Path(config.data_dir) / "valid.jsonl", prepared_dir / "valid.jsonl"
    )
    (prepared_dir / "test.jsonl").write_text(
        (prepared_dir / "valid.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    if train_count < 1 or valid_count < 1:
        raise ValueError("偏好数据没有可用于 chosen 监督的样本")
    config_path = output_dir / "preference_config.json"
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    initial = Path(config.initial_adapter_file)
    result = PeftLmAdapter().train(
        PeftLoraConfig(
            model=config.base_model,
            data_dir=prepared_dir,
            adapter_dir=output_dir,
            iterations=config.iterations,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_seq_length=config.max_seq_length,
            validation_batches=config.validation_batches,
            steps_per_evaluation=config.evaluation_interval,
            save_every=config.checkpoint_interval,
            seed=config.seed,
            rank=config.rank,
            alpha=max(1, round(config.rank * config.scale)),
            dropout=config.dropout,
            resume_adapter_file=initial,
        )
    )
    validation_loss = (
        float(result.validation_curve[-1]["validation_loss"])
        if result.validation_curve
        else float("inf")
    )
    metrics_path = output_dir / "preference_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "backend": "pytorch-transformers-peft",
                "objective": "chosen_supervision_after_memory_authority_filter",
                "train_pair_count": train_count,
                "valid_pair_count": valid_count,
                "validation_curve": list(result.validation_curve),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return PreferenceTrainingResult(
        adapter_file=str(output_dir / "adapter_model.safetensors"),
        iterations=result.iterations,
        train_pair_count=train_count,
        valid_pair_count=valid_count,
        skipped_oversized_train_pairs=0,
        skipped_oversized_valid_pairs=0,
        config_path=str(config_path),
        metrics_path=str(metrics_path),
        train_metrics=(),
        validation_metrics=result.validation_curve,
        selected_iteration=result.best_checkpoint_iteration or result.iterations,
        selected_validation_loss=validation_loss,
    )


def _write_chosen_dataset(source: Path, target: Path) -> int:
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        prompt = pair.get("prompt")
        chosen = pair.get("chosen")
        if not isinstance(prompt, list) or not isinstance(chosen, str) or not chosen.strip():
            continue
        messages = [item for item in prompt if isinstance(item, dict)]
        if not all(
            isinstance(item.get("role"), str) and isinstance(item.get("content"), str)
            for item in messages
        ):
            continue
        messages.append({"role": "assistant", "content": chosen})
        rows.append(json.dumps({"messages": messages}, ensure_ascii=False))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
