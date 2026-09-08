import hashlib
import re
import sqlite3
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import zstandard
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from moonlightbox.imports.wechat_schema import (
    WechatMediaSchema,
    WechatSchemaAdapter,
)


@dataclass(frozen=True)
class WechatMediaRecord:
    message_id: str
    md5: str | None
    relative_path: str | None


@dataclass(frozen=True)
class ResolvedWechatMedia:
    record: WechatMediaRecord
    file_path: Path


@dataclass(frozen=True)
class WechatStickerMessage:
    local_id: int
    timestamp: int
    sender_username: str
    md5: str
    cdn_url: str
    alternate_urls: tuple[str, ...] = ()


class WechatMediaExtractor:
    def records(self, database_path: Path) -> list[WechatMediaRecord]:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
            schema = WechatSchemaAdapter.detect(connection)
            if schema is None:
                return []
            return _read_records(connection, schema)

    def resolve(
        self,
        records: list[WechatMediaRecord],
        media_roots: list[Path],
    ) -> list[ResolvedWechatMedia]:
        files = _index_files(media_roots)
        resolved: list[ResolvedWechatMedia] = []
        for record in records:
            file_path = _resolve_record(record, media_roots, files)
            if file_path is not None:
                resolved.append(ResolvedWechatMedia(record=record, file_path=file_path))
        return resolved

    def contact_stickers(
        self,
        message_directory: Path,
        contact_username: str,
    ) -> list[WechatStickerMessage]:
        table = "Msg_" + hashlib.md5(contact_username.encode()).hexdigest()
        stickers: list[WechatStickerMessage] = []
        for database_path in sorted(message_directory.glob("message_[0-9]*.db")):
            with sqlite3.connect(database_path) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if table not in tables:
                    continue
                senders = {
                    int(row_id): str(username)
                    for row_id, username in connection.execute(
                        "SELECT rowid, user_name FROM Name2Id"
                    )
                }
                rows = connection.execute(
                    f"SELECT local_id, create_time, real_sender_id, "
                    f"message_content FROM {_quote_identifier(table)} "
                    "WHERE local_type = 47"
                )
                for local_id, timestamp, sender_id, content in rows:
                    emoji = _decode_emoji(content)
                    if emoji is None:
                        continue
                    md5 = emoji.get("md5", "").casefold()
                    cdn_url = emoji.get("cdnurl", "")
                    if not re.fullmatch(r"[0-9a-f]{32}", md5):
                        continue
                    alternate_urls = tuple(
                        url
                        for key in ("externurl", "thumburl")
                        if (url := emoji.get(key, "")) and url != cdn_url
                    )
                    stickers.append(
                        WechatStickerMessage(
                            local_id=int(local_id),
                            timestamp=int(timestamp),
                            sender_username=senders.get(int(sender_id), ""),
                            md5=md5,
                            cdn_url=cdn_url,
                            alternate_urls=alternate_urls,
                        )
                    )
        stickers.sort(key=lambda item: (item.timestamp, item.local_id))
        return stickers

    def avatar_bytes(
        self,
        head_image_database: Path,
        username: str,
    ) -> bytes | None:
        with sqlite3.connect(head_image_database) as connection:
            row = connection.execute(
                "SELECT image_buffer FROM head_image WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None or not row[0]:
            return None
        content = bytes(row[0])
        if not content.startswith(b"\xff\xd8\xff"):
            return None
        return content

    def store_sticker_bytes(
        self,
        emoticon_database: Path,
        account_root: Path,
        md5: str,
    ) -> bytes | None:
        with sqlite3.connect(emoticon_database) as connection:
            row = connection.execute(
                "SELECT package_id_, emoticon_offset_, emoticon_size_ "
                "FROM kStoreEmoticonFilesTable WHERE md5_ = ?",
                (md5,),
            ).fetchone()
        if row is None:
            return None
        package_id, offset, size = row
        package_hash = hashlib.md5(str(package_id).encode()).hexdigest()
        package_path = (
            account_root
            / "business"
            / "emoticon"
            / "PersistStore"
            / package_hash[:2]
            / package_hash
        )
        if not package_path.is_file():
            return None
        with package_path.open("rb") as package:
            package.seek(int(offset))
            content = package.read(int(size))
        if hashlib.md5(content).hexdigest() != md5:
            return None
        return content

    def nonstore_sticker_urls(
        self,
        emoticon_database: Path,
        md5: str,
    ) -> tuple[str, ...]:
        with sqlite3.connect(emoticon_database) as connection:
            row = connection.execute(
                "SELECT cdn_url, extern_url, thumb_url FROM kNonStoreEmoticonTable WHERE md5 = ?",
                (md5,),
            ).fetchone()
        if row is None:
            return ()
        return tuple(dict.fromkeys(str(url) for url in row if url))


def _read_records(
    connection: sqlite3.Connection,
    schema: WechatMediaSchema,
) -> list[WechatMediaRecord]:
    columns = [schema.message_id_column]
    columns.append(schema.md5_column or "NULL")
    columns.append(schema.path_column or "NULL")
    table = _quote_identifier(schema.table)
    selected = ", ".join(
        "NULL" if column == "NULL" else _quote_identifier(column) for column in columns
    )
    rows = connection.execute(f"SELECT {selected} FROM {table}")
    return [
        WechatMediaRecord(
            message_id=str(message_id),
            md5=str(md5).casefold() if md5 else None,
            relative_path=_safe_relative_path(str(path)) if path else None,
        )
        for message_id, md5, path in rows
    ]


def _index_files(media_roots: list[Path]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for root in media_roots:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            continue
        for candidate in resolved_root.rglob("*"):
            if not candidate.is_file():
                continue
            indexed.setdefault(candidate.name.casefold(), candidate)
            indexed.setdefault(candidate.stem.casefold(), candidate)
    return indexed


def _resolve_record(
    record: WechatMediaRecord,
    roots: list[Path],
    indexed: dict[str, Path],
) -> Path | None:
    if record.relative_path:
        for root in roots:
            root_path = root.resolve()
            candidate = (root_path / record.relative_path).resolve()
            if candidate.is_relative_to(root_path) and candidate.is_file():
                return candidate
    if record.md5:
        return indexed.get(record.md5.casefold())
    return None


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or ".." in path.parts:
        raise ValueError("微信媒体路径不安全")
    return path.as_posix()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _decode_emoji(value: object) -> dict[str, str] | None:
    if not isinstance(value, (bytes, bytearray, memoryview, str)):
        return None
    if isinstance(value, str):
        decoded = value
    else:
        raw = bytes(value)
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                decoded = zstandard.ZstdDecompressor().decompress(raw).decode("utf-8")
            except (zstandard.ZstdError, UnicodeDecodeError):
                return None
    try:
        root = ET.fromstring(decoded)
    except ET.ParseError:
        return None
    emoji = root.find("emoji")
    return dict(emoji.attrib) if emoji is not None else None


def decode_wechat_image_dat(
    data: bytes,
    *,
    image_key: bytes,
    xor_key: int,
) -> tuple[bytes, str]:
    """解码微信 4.x 本地图片缓存。"""
    if len(data) < 15 or data[:6] not in (b"\x07\x08V1\x08\x07", b"\x07\x08V2\x08\x07"):
        raise ValueError("不支持的微信图片缓存格式")
    aes_size, xor_size = struct.unpack("<II", data[6:14])
    file_data = data[15:]
    aligned_aes_size = aes_size + (16 - aes_size % 16) if aes_size else 0
    if len(file_data) < aligned_aes_size + xor_size:
        raise ValueError("微信图片缓存内容不完整")
    encrypted = file_data[:aligned_aes_size]
    remaining = file_data[aligned_aes_size:]
    aes_key = b"cfcd208495d565ef" if data[:6] == b"\x07\x08V1\x08\x07" else image_key[:16]
    decryptor = Cipher(algorithms.AES(aes_key), modes.ECB()).decryptor()
    aes_plain = decryptor.update(encrypted) + decryptor.finalize()
    if aes_plain:
        padding = aes_plain[-1]
        if not 0 < padding <= 16 or not aes_plain.endswith(bytes([padding]) * padding):
            raise ValueError("微信图片缓存密钥无效")
        aes_plain = aes_plain[:-padding]
    raw_size = len(remaining) - xor_size
    decoded = aes_plain + remaining[:raw_size] + bytes(
        value ^ xor_key for value in remaining[raw_size:]
    )
    suffix = _decoded_image_suffix(decoded)
    if suffix is None:
        raise ValueError("微信图片解码后格式不可识别")
    return decoded, suffix


def _decoded_image_suffix(content: bytes) -> str | None:
    if content.startswith(b"GIF8"):
        return ".gif"
    if content.startswith(b"\x89PNG"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return None
