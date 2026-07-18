from pathlib import Path
from typing import Protocol

from moonlightbox.imports.types import ImportResult


class ChatImporter(Protocol):
    def sniff(self, path: Path) -> float: ...

    def parse(self, path: Path) -> ImportResult: ...
