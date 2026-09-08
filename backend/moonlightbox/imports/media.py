from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from moonlightbox.imports.types import ImportedMessage


@dataclass(frozen=True)
class MediaFile:
    relative_path: str
    stored_path: Path


def normalize_media_path(value: str) -> str:
    normalized = value.replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or ".." in path.parts:
        raise ValueError("媒体相对路径不安全")
    return path.as_posix()


def match_message_media(
    message: ImportedMessage,
    files: list[MediaFile],
) -> MediaFile | None:
    candidates: set[str] = set()
    for value in message.raw.values():
        if not isinstance(value, str):
            continue
        normalized = value.replace("\\", "/")
        candidates.add(normalized.lower())
        candidates.add(PurePosixPath(normalized).name.lower())
    for media in files:
        relative = normalize_media_path(media.relative_path).lower()
        if relative in candidates or PurePosixPath(relative).name in candidates:
            return media
    return None


def match_participant_avatar(
    participant_name: str,
    files: list[MediaFile],
) -> MediaFile | None:
    name = participant_name.casefold()
    for media in files:
        relative = normalize_media_path(media.relative_path).casefold()
        filename = PurePosixPath(relative).name
        if ("avatar" in relative or "head" in relative or "头像" in relative) and (
            name in filename
        ):
            return media
    return None
