import json
from datetime import datetime, timedelta
from pathlib import Path

from moonlightbox.imports.types import ImportedMessage, MessageKind


def make_message(
    source_id: str,
    sender: str,
    content: str,
    minute: int,
) -> ImportedMessage:
    return ImportedMessage(
        source_id=source_id,
        timestamp=datetime(2026, 1, 1, 12) + timedelta(minutes=minute),
        sender=sender,
        kind=MessageKind.TEXT,
        content=content,
        raw={},
    )


def test_dataset_excludes_messages_after_historical_cutoff() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "今天见面吗", 0),
        make_message("m2", "她", "好呀", 1),
        make_message("m3", "我", "后来我们分手了", 60),
        make_message("m4", "她", "再见", 61),
    ]

    examples = DatasetBuilder(context_turns=4).build(
        messages,
        target_sender="她",
        cutoff=datetime(2026, 1, 1, 12, 30),
    )

    assert len(examples) == 1
    assert examples[0].source_ids == ["m1", "m2"]
    assert all("分手" not in turn.content for turn in examples[0].messages)


def test_dataset_export_uses_temporal_split_and_reproducible_manifest(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "第一问", 0),
        make_message("m2", "她", "第一答", 1),
        make_message("m3", "我", "第二问", 60),
        make_message("m4", "她", "第二答", 61),
    ]
    cutoff = datetime(2026, 1, 2)
    builder = DatasetBuilder(context_turns=1)
    examples = builder.build(messages, target_sender="她", cutoff=cutoff)

    first = builder.write_mlx_dataset(examples, tmp_path, "她", cutoff)
    second = builder.write_mlx_dataset(examples, tmp_path, "她", cutoff)
    train = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8"))
    valid = json.loads((tmp_path / "valid.jsonl").read_text(encoding="utf-8"))

    assert train["metadata"]["source_ids"] == ["m1", "m2"]
    assert valid["metadata"]["source_ids"] == ["m3", "m4"]
    assert first.dataset_hash == second.dataset_hash
