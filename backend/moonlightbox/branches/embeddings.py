from pathlib import Path
from typing import Protocol

from fastembed import TextEmbedding


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalChineseEmbedder:
    def __init__(self, model_root: Path) -> None:
        self._model = TextEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            cache_dir=str(model_root),
            local_files_only=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]
