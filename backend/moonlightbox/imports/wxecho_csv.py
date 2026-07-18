import csv
from datetime import datetime
from pathlib import Path

from moonlightbox.imports.types import (
    ImportedMessage,
    ImportError,
    ImportResult,
    MessageKind,
)

KIND_MAPPING = {
    "文本": MessageKind.TEXT,
    "图片": MessageKind.IMAGE,
    "视频": MessageKind.VIDEO,
    "语音": MessageKind.AUDIO,
    "文件": MessageKind.FILE,
    "系统": MessageKind.SYSTEM,
}


class WxechoCsvImporter:
    def sniff(self, path: Path) -> float:
        if path.suffix.lower() != ".csv":
            return 0.0
        try:
            header = path.open(encoding="utf-8-sig").readline().strip()
        except (OSError, UnicodeError):
            return 0.0
        return 1.0 if header == "时间,发送者,类型,内容" else 0.2

    def parse(self, path: Path) -> ImportResult:
        result = ImportResult()
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = csv.DictReader(source)
            for line, row in enumerate(rows, start=2):
                try:
                    result.messages.append(self._parse_row(line, row))
                except (KeyError, TypeError, ValueError) as error:
                    result.errors.append(
                        ImportError(
                            line=line,
                            code="invalid_row",
                            message=str(error),
                        )
                    )
        return result

    def _parse_row(self, line: int, row: dict[str, str | None]) -> ImportedMessage:
        timestamp_text = self._required(row, "时间")
        sender = self._required(row, "发送者")
        kind_text = self._required(row, "类型")
        content = row.get("内容") or ""

        return ImportedMessage(
            source_id=f"csv:{line}",
            timestamp=datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S"),
            sender=sender,
            kind=KIND_MAPPING.get(kind_text, MessageKind.UNKNOWN),
            content=content,
            raw=dict(row),
        )

    @staticmethod
    def _required(row: dict[str, str | None], field: str) -> str:
        value = row[field]
        if value is None or not value.strip():
            raise ValueError(f"字段“{field}”不能为空")
        return value.strip()
