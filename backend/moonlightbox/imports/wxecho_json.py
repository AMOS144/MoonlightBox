import json
from datetime import datetime
from pathlib import Path
from typing import Any

from moonlightbox.imports.types import (
    ImportedMessage,
    ImportError,
    ImportResult,
    MessageKind,
)
from moonlightbox.imports.wxecho_csv import KIND_MAPPING


class WxechoJsonImporter:
    def sniff(self, path: Path) -> float:
        return 0.9 if path.suffix.lower() == ".json" else 0.0

    def parse(self, path: Path) -> ImportResult:
        result = ImportResult()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("wxecho JSON 顶层必须是数组")

        for index, item in enumerate(payload, start=1):
            try:
                result.messages.append(self._parse_item(index, item))
            except (KeyError, TypeError, ValueError) as error:
                result.errors.append(ImportError(index, "invalid_item", str(error)))
        return result

    def _parse_item(self, index: int, item: Any) -> ImportedMessage:
        if not isinstance(item, dict):
            raise TypeError("消息必须是对象")

        time_text = self._required(item, "time")
        sender = self._required(item, "sender")
        type_name = self._required(item, "type_name")
        explicit_kind = item.get("kind")
        try:
            kind = (
                MessageKind(explicit_kind)
                if isinstance(explicit_kind, str)
                else KIND_MAPPING.get(type_name, MessageKind.UNKNOWN)
            )
        except ValueError:
            kind = MessageKind.UNKNOWN
        content = str(item.get("content") or "")
        if explicit_kind is None and kind is not MessageKind.TEXT:
            content = f"[{type_name}]"

        return ImportedMessage(
            source_id=str(item.get("server_id") or f"json:{index}"),
            timestamp=datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S"),
            sender=sender,
            kind=kind,
            content=content,
            raw=dict(item),
        )

    @staticmethod
    def _required(item: dict[str, Any], field: str) -> str:
        value = item[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"字段“{field}”不能为空")
        return value.strip()
