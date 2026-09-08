import json
from pathlib import Path

import pytest


def test_linux_preference_config_rejects_missing_peft_adapter(tmp_path: Path) -> None:
    from moonlightbox.training.peft_preference_trainer import PreferenceTrainingConfig

    config = PreferenceTrainingConfig(
        base_model="base",
        initial_adapter_file=str(tmp_path / "missing.safetensors"),
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="PEFT adapter"):
        config.validate()


def test_linux_preference_dataset_uses_chosen_reply_only(tmp_path: Path) -> None:
    from moonlightbox.training.peft_preference_trainer import _write_chosen_dataset

    source = tmp_path / "pairs.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "在吗"}],
                        "chosen": "在呀",
                        "rejected": "您好，有什么可以帮助您？",
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"prompt": [], "chosen": ""}, ensure_ascii=False),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "chosen" / "train.jsonl"

    assert _write_chosen_dataset(source, target) == 1
    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["messages"] == [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "在呀"},
    ]
