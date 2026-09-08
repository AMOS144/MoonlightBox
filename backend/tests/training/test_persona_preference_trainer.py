from pathlib import Path

import pytest

# 当前 Linux 运行路径为 peft_preference_trainer；本文件覆盖历史 MLX trainer。
pytest.importorskip("mlx", reason="历史 MLX 偏好训练测试仅适用于 macOS 环境")


def test_preference_training_config_rejects_missing_inputs(tmp_path: Path) -> None:
    from moonlightbox.training.persona_preference_trainer import (
        PreferenceTrainingConfig,
    )

    config = PreferenceTrainingConfig(
        base_model="base",
        initial_adapter_file=str(tmp_path / "missing.safetensors"),
        data_dir=str(tmp_path / "missing-data"),
        output_dir=str(tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="初始 adapter 不存在"):
        config.validate()


def test_preference_training_config_rejects_unanchored_sft_weight(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference_trainer import (
        PreferenceTrainingConfig,
    )

    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    config = PreferenceTrainingConfig(
        base_model="base",
        initial_adapter_file=str(adapter),
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "output"),
        sft_weight=1.1,
    )

    with pytest.raises(ValueError, match="SFT 权重"):
        config.validate()


def test_preference_adapter_config_preserves_lora_structure(tmp_path: Path) -> None:
    import json

    from moonlightbox.training.persona_preference_trainer import (
        PreferenceTrainingConfig,
        write_preference_adapter_config,
    )

    source = tmp_path / "source"
    source.mkdir()
    adapter = source / "adapters.safetensors"
    adapter.write_bytes(b"adapter")
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "num_layers": 12,
                "lora_parameters": {"rank": 32, "keys": ["mlp.up_proj"]},
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    config = PreferenceTrainingConfig(
        base_model="base",
        initial_adapter_file=str(adapter),
        data_dir=str(data_dir),
        output_dir=str(output),
    )

    path = write_preference_adapter_config(config, output)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["num_layers"] == 12
    assert payload["lora_parameters"]["rank"] == 32
    assert payload["resume_adapter_file"] == str(adapter)
    assert payload["preference_training"]["schema_version"].endswith("v1")


def test_preference_training_resumes_exact_initial_lora_structure(tmp_path: Path) -> None:
    import json

    from moonlightbox.training.persona_preference_trainer import (
        PreferenceTrainingConfig,
        _initial_lora_structure,
    )

    source = tmp_path / "source"
    source.mkdir()
    adapter = source / "adapters.safetensors"
    adapter.write_bytes(b"adapter")
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "num_layers": 16,
                "lora_parameters": {
                    "rank": 16,
                    "scale": 2.0,
                    "dropout": 0.05,
                    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
                },
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    config = PreferenceTrainingConfig(
        base_model="base",
        initial_adapter_file=str(adapter),
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "output"),
        num_layers=12,
        rank=32,
    )

    assert _initial_lora_structure(config) == {
        "num_layers": 16,
        "rank": 16,
        "scale": 2.0,
        "dropout": 0.05,
        "keys": ("self_attn.q_proj", "self_attn.v_proj"),
    }


def test_metrics_callback_records_only_json_numeric_values() -> None:
    from moonlightbox.training.persona_preference_trainer import (
        PreferenceMetricsCallback,
    )

    callback = PreferenceMetricsCallback()
    callback.on_train_loss_report(
        {"iteration": 5, "train_loss": 1.25, "ignored": "value"}
    )
    callback.on_val_loss_report({"iteration": 0, "val_loss": 2.5})

    assert callback.train_metrics == [{"iteration": 5, "train_loss": 1.25}]
    assert callback.validation_metrics == [{"iteration": 0, "val_loss": 2.5}]


def test_materialize_preference_checkpoint_is_mlx_loadable_shape(
    tmp_path: Path,
) -> None:
    import json

    from moonlightbox.training.persona_preference_trainer import (
        PreferenceTrainingConfig,
        materialize_preference_checkpoint,
    )

    source = tmp_path / "source"
    source.mkdir()
    initial = source / "adapters.safetensors"
    initial.write_bytes(b"initial")
    (source / "adapter_config.json").write_text(
        json.dumps({"num_layers": 12, "lora_parameters": {"rank": 32}}),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "0000010_adapters.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "checkpoint-10"
    config = PreferenceTrainingConfig(
        base_model="base",
        initial_adapter_file=str(initial),
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "training"),
    )

    materialized = materialize_preference_checkpoint(config, checkpoint, output)

    assert materialized.read_bytes() == b"checkpoint"
    assert (output / "adapter_config.json").is_file()


def test_preference_training_publishes_lowest_validated_checkpoint(tmp_path: Path) -> None:
    from moonlightbox.training.persona_preference_trainer import (
        _select_best_checkpoint,
    )

    (tmp_path / "adapters.safetensors").write_bytes(b"final")
    (tmp_path / "0000020_adapters.safetensors").write_bytes(b"step-20")
    (tmp_path / "0000040_adapters.safetensors").write_bytes(b"step-40")

    iteration, loss = _select_best_checkpoint(
        tmp_path,
        [
            {"iteration": 20, "val_loss": 0.8},
            {"iteration": 40, "val_loss": 0.7},
            {"iteration": 60, "val_loss": 0.9},
        ],
        final_iteration=60,
    )

    assert (iteration, loss) == (40, 0.7)
    assert (tmp_path / "adapters.safetensors").read_bytes() == b"step-40"
