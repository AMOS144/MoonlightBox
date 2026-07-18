import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from moonlightbox.imports.types import ImportedMessage, MessageKind


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class TrainingExample:
    messages: list[ChatTurn]
    source_ids: list[str]
    target_at: datetime


@dataclass(frozen=True)
class DatasetManifest:
    dataset_hash: str
    example_count: int
    cutoff: str
    target_sender: str
    context_turns: int


class DatasetBuilder:
    def __init__(self, context_turns: int = 12) -> None:
        if context_turns < 1:
            raise ValueError("上下文轮数必须大于零")
        self._context_turns = context_turns

    def build(
        self,
        messages: list[ImportedMessage],
        target_sender: str,
        cutoff: datetime,
    ) -> list[TrainingExample]:
        eligible = sorted(
            (
                message
                for message in messages
                if message.timestamp <= cutoff and message.kind is MessageKind.TEXT
            ),
            key=lambda message: message.timestamp,
        )
        examples: list[TrainingExample] = []
        for index, target in enumerate(eligible):
            if target.sender != target_sender or index == 0:
                continue
            context = eligible[max(0, index - self._context_turns) : index]
            turns = [
                ChatTurn(
                    role="assistant" if message.sender == target_sender else "user",
                    content=message.content,
                )
                for message in context
            ]
            turns.append(ChatTurn(role="assistant", content=target.content))
            examples.append(
                TrainingExample(
                    messages=turns,
                    source_ids=[message.source_id for message in context] + [target.source_id],
                    target_at=target.timestamp,
                )
            )
        return examples

    def write_mlx_dataset(
        self,
        examples: list[TrainingExample],
        directory: Path,
        target_sender: str,
        cutoff: datetime,
    ) -> DatasetManifest:
        directory.mkdir(parents=True, exist_ok=True)
        ordered = sorted(examples, key=lambda example: example.target_at)
        split_at = max(1, int(len(ordered) * 0.9)) if ordered else 0
        _write_jsonl(directory / "train.jsonl", ordered[:split_at])
        _write_jsonl(directory / "valid.jsonl", ordered[split_at:])

        serialized = json.dumps(
            [_example_payload(example) for example in ordered],
            ensure_ascii=False,
            sort_keys=True,
        )
        manifest = DatasetManifest(
            dataset_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            example_count=len(ordered),
            cutoff=cutoff.isoformat(),
            target_sender=target_sender,
            context_turns=self._context_turns,
        )
        (directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest


def _write_jsonl(path: Path, examples: list[TrainingExample]) -> None:
    path.write_text(
        "".join(
            json.dumps(_example_payload(example), ensure_ascii=False) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )


def _example_payload(example: TrainingExample) -> dict[str, object]:
    return {
        "messages": [asdict(turn) for turn in example.messages],
        "metadata": {
            "source_ids": example.source_ids,
            "target_at": example.target_at.isoformat(),
        },
    }
