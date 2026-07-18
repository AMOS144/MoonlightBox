import re
from datetime import datetime
from pathlib import Path

from moonlightbox.imports.types import ImportedMessage, ImportError, ImportResult, MessageKind

MESSAGE_PATTERN = re.compile(
    r"^\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"(?P<sender>[^:]+): (?P<content>.*)$"
)
PLACEHOLDER_KINDS = {
    "[图片]": MessageKind.IMAGE,
    "[视频]": MessageKind.VIDEO,
    "[语音]": MessageKind.AUDIO,
    "[文件]": MessageKind.FILE,
}


class WxechoTxtImporter:
    def sniff(self, path: Path) -> float:
        if path.suffix.lower() != ".txt":
            return 0.0
        with path.open(encoding="utf-8") as source:
            return 1.0 if source.readline().startswith("微信聊天记录:") else 0.2

    def parse(self, path: Path) -> ImportResult:
        result = ImportResult()
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.startswith("["):
                    continue
                match = MESSAGE_PATTERN.match(line.rstrip("\n"))
                if match is None:
                    result.errors.append(
                        ImportError(line_number, "invalid_line", "无法解析消息行")
                    )
                    continue
                content = match.group("content")
                result.messages.append(
                    ImportedMessage(
                        source_id=f"txt:{line_number}",
                        timestamp=datetime.strptime(
                            match.group("time"),
                            "%Y-%m-%d %H:%M:%S",
                        ),
                        sender=match.group("sender").strip(),
                        kind=PLACEHOLDER_KINDS.get(content, MessageKind.TEXT),
                        content=content,
                        raw={"line": line.rstrip("\n")},
                    )
                )
        return result
