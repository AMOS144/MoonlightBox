import hashlib
import io
import re
import sqlite3
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from moonlightbox.imports.wechat_media import decode_wechat_image_dat


@dataclass(frozen=True)
class WechatLocalMedia:
    timestamp: int
    role: str
    kind: str
    filename: str
    content: bytes
    source_key: str


class WechatLocalMediaExtractor:
    def extract(
        self,
        *,
        decrypted_message_root: Path,
        account_root: Path,
        target_username: str,
        self_username: str,
        image_key: bytes,
        image_xor_key: int,
        silk_decoder: Path,
        wanted: set[tuple[int, str, str]] | None = None,
    ) -> list[WechatLocalMedia]:
        table = "Msg_" + hashlib.md5(target_username.encode()).hexdigest()
        rows = _message_rows(decrypted_message_root, table)
        voices = _voice_data(decrypted_message_root, target_username)
        media: list[WechatLocalMedia] = []
        for row in rows:
            role = "target" if row.sender == target_username else "self"
            if wanted is not None and (row.timestamp, role, row.kind) not in wanted:
                continue
            if row.kind == "image":
                item = _image_media(
                    row,
                    account_root,
                    table.removeprefix("Msg_"),
                    image_key,
                    image_xor_key,
                )
            elif row.kind == "audio":
                item = _audio_media(row, voices.get(row.local_id), silk_decoder)
            else:
                item = _video_media(row, account_root)
            if item is not None:
                media.append(
                    WechatLocalMedia(
                        timestamp=row.timestamp,
                        role=role,
                        kind=row.kind,
                        filename=item[0],
                        content=item[1],
                        source_key=f"wechat-{row.kind}:{row.server_id}",
                    )
                )
        return media


@dataclass(frozen=True)
class _MessageRow:
    local_id: int
    server_id: int
    timestamp: int
    sender: str
    kind: str
    packed: bytes


def _message_rows(message_root: Path, table: str) -> list[_MessageRow]:
    rows: list[_MessageRow] = []
    kind_by_type = {3: "image", 34: "audio", 43: "video"}
    for database in sorted(message_root.glob("message_[0-9]*.db")):
        with sqlite3.connect(database) as connection:
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
            query = (
                f"SELECT local_id, server_id, create_time, real_sender_id, "
                f"local_type, packed_info_data FROM {_quote(table)} "
                "WHERE local_type IN (3, 34, 43) ORDER BY create_time, local_id"
            )
            for local_id, server_id, timestamp, sender_id, local_type, packed in (
                connection.execute(query)
            ):
                rows.append(
                    _MessageRow(
                        local_id=int(local_id),
                        server_id=int(server_id),
                        timestamp=int(timestamp),
                        sender=senders.get(int(sender_id), ""),
                        kind=kind_by_type[int(local_type)],
                        packed=bytes(packed or b""),
                    )
                )
    rows.sort(key=lambda item: (item.timestamp, item.local_id))
    return rows


def _voice_data(message_root: Path, target_username: str) -> dict[int, bytes]:
    voices: dict[int, bytes] = {}
    for database in sorted(message_root.glob("media_[0-9]*.db")):
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT rowid FROM Name2Id WHERE user_name = ?",
                (target_username,),
            ).fetchone()
            if row is None:
                continue
            for local_id, content in connection.execute(
                "SELECT local_id, voice_data FROM VoiceInfo WHERE chat_name_id = ?",
                (int(row[0]),),
            ):
                if content:
                    voices[int(local_id)] = bytes(content)
    return voices


def _packed_hash(packed: bytes) -> str | None:
    match = re.search(rb"[0-9a-f]{32}", packed.lower())
    return match.group().decode() if match else None


def _image_media(
    row: _MessageRow,
    account_root: Path,
    chat_hash: str,
    image_key: bytes,
    xor_key: int,
) -> tuple[str, bytes] | None:
    cache_id = _packed_hash(row.packed)
    if cache_id is None:
        return None
    month = datetime.fromtimestamp(row.timestamp).strftime("%Y-%m")
    directory = account_root / "msg" / "attach" / chat_hash / month / "Img"
    candidates = (
        directory / f"{cache_id}.dat",
        directory / f"{cache_id}_h.dat",
        directory / f"{cache_id}_t.dat",
    )
    for source in candidates:
        if not source.is_file():
            continue
        try:
            content, suffix = decode_wechat_image_dat(
                source.read_bytes(),
                image_key=image_key,
                xor_key=xor_key,
            )
        except ValueError:
            continue
        return cache_id + suffix, content
    return None


def _audio_media(
    row: _MessageRow,
    silk: bytes | None,
    decoder: Path,
) -> tuple[str, bytes] | None:
    if silk is None or not decoder.is_file():
        return None
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "voice.silk"
        output = Path(directory) / "voice.pcm"
        source.write_bytes(silk)
        completed = subprocess.run(
            [str(decoder), str(source), str(output)],
            capture_output=True,
            check=False,
            timeout=20,
        )
        if completed.returncode != 0 or not output.is_file():
            return None
        pcm = output.read_bytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return f"{row.server_id}.wav", buffer.getvalue()


def _video_media(
    row: _MessageRow,
    account_root: Path,
) -> tuple[str, bytes] | None:
    cache_id = _packed_hash(row.packed)
    if cache_id is None:
        return None
    month = datetime.fromtimestamp(row.timestamp).strftime("%Y-%m")
    directory = account_root / "msg" / "video" / month
    video = directory / f"{cache_id}.mp4"
    if video.is_file():
        return video.name, video.read_bytes()
    thumbnail = directory / f"{cache_id}_thumb.jpg"
    if thumbnail.is_file():
        return thumbnail.name, thumbnail.read_bytes()
    return None


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
