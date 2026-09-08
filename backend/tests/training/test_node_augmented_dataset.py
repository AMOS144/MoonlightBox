import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

from moonlightbox.imports.types import ImportedMessage, MessageKind


def _message(source_id: str, sender: str, content: str, minute: int) -> ImportedMessage:
    return ImportedMessage(
        source_id=source_id,
        timestamp=datetime(2026, 1, 1, 12) + timedelta(minutes=minute),
        sender=sender,
        kind=MessageKind.TEXT,
        content=content,
        raw={},
    )


def test_event_augmented_example_keeps_real_target_reply() -> None:
    from moonlightbox.training.dataset_builder import (
        ConfirmedEventContext,
        DatasetBuilder,
    )

    messages = [
        _message("m1", "我", "五一去杭州吗", 0),
        _message("m2", "她", "好呀，我们一起去", 1),
        _message("m3", "我", "订票吧", 2),
        _message("m4", "她", "我来看看时间", 3),
    ]
    builder = DatasetBuilder(context_turns=2, maximum_event_ratio=1.0)
    regular = builder.build(messages, target_sender="她", cutoff=datetime(2026, 1, 2))
    augmented = builder.augment_with_events(
        regular,
        [
            ConfirmedEventContext(
                event_id="event-1",
                title="杭州共同旅行",
                summary="双方确认五一共同旅行",
                lane="shared_experience",
                event_status="confirmed",
                evidence_ids=("m1", "m2"),
            )
        ],
    )

    enhanced = next(example for example in augmented if example.kind == "event_augmented")
    assert enhanced.messages[0].role == "system"
    assert "杭州共同旅行" in enhanced.messages[0].content
    assert enhanced.messages[-1].role == "assistant"
    target = ET.fromstring(f"<turn>{enhanced.messages[-1].content}</turn>")
    assert target[0].tag == "bubble"
    assert target[0].text == "好呀，我们一起去"
    assert enhanced.source_ids[-1] == "m2"


def test_event_augmentation_is_deduplicated_and_ratio_capped() -> None:
    from moonlightbox.training.dataset_builder import (
        ConfirmedEventContext,
        DatasetBuilder,
    )

    messages = [
        _message("m1", "我", "第一问", 0),
        _message("m2", "她", "第一答", 1),
        _message("m3", "我", "第二问", 2),
        _message("m4", "她", "第二答", 3),
        _message("m5", "我", "第三问", 4),
        _message("m6", "她", "第三答", 5),
        _message("m7", "我", "第四问", 6),
        _message("m8", "她", "第四答", 7),
    ]
    builder = DatasetBuilder(context_turns=1, maximum_event_ratio=0.25)
    regular = builder.build(messages, target_sender="她", cutoff=datetime(2026, 1, 2))
    event = ConfirmedEventContext(
        event_id="event-1",
        title="重要经历",
        summary="共同经历摘要",
        lane="shared_experience",
        event_status="occurred",
        evidence_ids=("m1", "m2", "m3", "m4", "m5", "m6"),
    )

    augmented = builder.augment_with_events(regular, [event, event])
    enhanced = [item for item in augmented if item.kind == "event_augmented"]

    assert len(enhanced) == 1
    assert len({tuple(item.source_ids) for item in enhanced}) == len(enhanced)


def test_extended_manifest_records_confirmation_and_sample_counts(tmp_path: Path) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        _message("m1", "我", "问题", 0),
        _message("m2", "她", "回答", 1),
    ]
    builder = DatasetBuilder(context_turns=1)
    examples = builder.build(messages, target_sender="她", cutoff=datetime(2026, 1, 2))

    manifest = builder.write_mlx_dataset(
        examples,
        tmp_path,
        "她",
        datetime(2026, 1, 2),
        confirmation_id="confirmation-1",
        analysis_run_id="run-1",
        node_snapshot_hash="node-hash",
        base_model="mlx-community/test",
        training_config={"iterations": 10},
    )
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest.regular_example_count == 1
    assert manifest.augmented_example_count == 0
    assert payload["confirmation_id"] == "confirmation-1"
    assert payload["analysis_run_id"] == "run-1"
    assert payload["node_snapshot_hash"] == "node-hash"
