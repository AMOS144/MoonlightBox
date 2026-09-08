import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.wechat_local_media import (
    WechatLocalMedia,
    WechatLocalMediaExtractor,
)
from moonlightbox.imports.wechat_media import (
    WechatMediaExtractor,
    WechatStickerMessage,
)
from moonlightbox.media.models import MediaAsset
from moonlightbox.media.service import MediaStore


@dataclass(frozen=True)
class WechatMediaImportReport:
    sticker_message_count: int
    linked_sticker_count: int
    missing_sticker_count: int
    unique_sticker_asset_count: int
    avatar_count: int
    linked_image_count: int
    linked_audio_count: int
    linked_video_count: int


class WechatProjectMediaImporter:
    def __init__(
        self,
        data_dir: Path,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._store = MediaStore(data_dir)
        self._extractor = WechatMediaExtractor()
        self._client = client or httpx.Client(follow_redirects=True, timeout=30)

    def import_media(
        self,
        session: Session,
        *,
        project_id: str,
        decrypted_root: Path,
        wechat_account_root: Path,
        target_username: str,
        self_username: str,
        image_key: bytes | None = None,
        image_xor_key: int = 0,
        silk_decoder: Path | None = None,
    ) -> WechatMediaImportReport:
        participants = {
            participant.role: participant
            for participant in session.scalars(
                select(Participant).where(
                    Participant.project_id == project_id,
                    Participant.role.in_(("self", "target")),
                )
            )
        }
        stickers = self._extractor.contact_stickers(
            decrypted_root / "message",
            target_username,
        )
        by_turn: dict[
            tuple[int, str],
            deque[WechatStickerMessage],
        ] = defaultdict(deque)
        for sticker in stickers:
            role = "target" if sticker.sender_username == target_username else "self"
            by_turn[(sticker.timestamp, role)].append(sticker)

        imported_messages = list(
            session.execute(
                select(Message, Participant.role)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.project_id == project_id,
                    Message.content == "[表情]",
                    Participant.role.in_(("self", "target")),
                )
                .order_by(Message.timestamp, Message.source_id)
            )
        )
        assets_by_md5: dict[str, MediaAsset | None] = {
            cast(str, asset.source_key).removeprefix("wechat-sticker:"): asset
            for asset in session.scalars(
                select(MediaAsset).where(
                    MediaAsset.project_id == project_id,
                    MediaAsset.source_key.like("wechat-sticker:%"),
                )
            )
        }
        linked = 0
        for message, role in imported_messages:
            matches = by_turn.get((int(message.timestamp.timestamp()), role))
            if not matches:
                continue
            sticker = matches.popleft()
            if sticker.md5 not in assets_by_md5:
                assets_by_md5[sticker.md5] = self._save_sticker(
                    session,
                    project_id,
                    sticker,
                    decrypted_root,
                    wechat_account_root,
                )
            asset = assets_by_md5[sticker.md5]
            if asset is None:
                continue
            message.kind = "sticker"
            message.media_asset_id = asset.id
            linked += 1

        avatar_count = 0
        head_image_database = decrypted_root / "head_image" / "head_image.db"
        for role, username in (
            ("self", self_username),
            ("target", target_username),
        ):
            participant = participants.get(role)
            content = self._extractor.avatar_bytes(
                head_image_database,
                username,
            )
            if participant is None or content is None:
                continue
            asset = self._store.save(
                session,
                project_id,
                kind="avatar",
                filename=f"{username}.jpg",
                content=content,
                source_key=f"wechat-avatar:{username}",
            )
            participant.avatar_asset_id = asset.id
            avatar_count += 1

        linked_media = {"image": 0, "audio": 0, "video": 0}
        if image_key is not None and silk_decoder is not None:
            imported_media = list(
                session.execute(
                    select(Message, Participant.role)
                    .join(Participant, Participant.id == Message.participant_id)
                    .where(
                        Message.project_id == project_id,
                        Message.kind.in_(("image", "audio", "video")),
                        Message.media_asset_id.is_(None),
                        Participant.role.in_(("self", "target")),
                    )
                    .order_by(Message.timestamp, Message.source_id)
                )
            )
            local_media = WechatLocalMediaExtractor().extract(
                decrypted_message_root=decrypted_root / "message",
                account_root=wechat_account_root,
                target_username=target_username,
                self_username=self_username,
                image_key=image_key,
                image_xor_key=image_xor_key,
                silk_decoder=silk_decoder,
                wanted={
                    (int(message.timestamp.timestamp()), role, message.kind)
                    for message, role in imported_media
                },
            )
            by_media_turn: dict[
                tuple[int, str, str],
                deque[WechatLocalMedia],
            ] = defaultdict(deque)
            for item in local_media:
                by_media_turn[(item.timestamp, item.role, item.kind)].append(item)
            for message, role in imported_media:
                media_matches = by_media_turn.get(
                    (int(message.timestamp.timestamp()), role, message.kind)
                )
                if not media_matches:
                    continue
                media_item = media_matches.popleft()
                is_video_thumbnail = (
                    media_item.kind == "video"
                    and media_item.filename.endswith("_thumb.jpg")
                )
                asset_kind = "video_thumbnail" if is_video_thumbnail else media_item.kind
                asset = self._store.save(
                    session,
                    project_id,
                    kind=asset_kind,
                    filename=media_item.filename,
                    content=media_item.content,
                    source_key=media_item.source_key,
                )
                if is_video_thumbnail:
                    message.kind = "video_thumbnail"
                message.media_asset_id = asset.id
                linked_media[media_item.kind] += 1

        session.commit()
        return WechatMediaImportReport(
            sticker_message_count=len(imported_messages),
            linked_sticker_count=linked,
            missing_sticker_count=len(imported_messages) - linked,
            unique_sticker_asset_count=sum(asset is not None for asset in assets_by_md5.values()),
            avatar_count=avatar_count,
            linked_image_count=linked_media["image"],
            linked_audio_count=linked_media["audio"],
            linked_video_count=linked_media["video"],
        )

    def _save_sticker(
        self,
        session: Session,
        project_id: str,
        sticker: WechatStickerMessage,
        decrypted_root: Path,
        wechat_account_root: Path,
    ) -> MediaAsset | None:
        fallback: bytes | None = None
        content: bytes | None = None
        emoticon_database = decrypted_root / "emoticon" / "emoticon.db"
        database_urls = self._extractor.nonstore_sticker_urls(
            emoticon_database,
            sticker.md5,
        )
        for url in dict.fromkeys((sticker.cdn_url, *sticker.alternate_urls, *database_urls)):
            if not _allowed_sticker_url(url):
                continue
            try:
                response = self._client.get(url)
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            candidate = response.content
            if _image_suffix(candidate) == ".bin":
                continue
            if hashlib.md5(candidate).hexdigest() == sticker.md5:
                content = candidate
                break
            fallback = fallback or candidate
        content = content or fallback
        if content is None:
            content = self._extractor.store_sticker_bytes(
                emoticon_database,
                wechat_account_root,
                sticker.md5,
            )
        if content is None:
            return None
        return self._store.save(
            session,
            project_id,
            kind="sticker",
            filename=sticker.md5 + _image_suffix(content),
            content=content,
            source_key=f"wechat-sticker:{sticker.md5}",
        )


def _allowed_sticker_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    return parsed.scheme in {"http", "https"} and (
        host == "emoji.qpic.cn"
        or host.endswith(".qpic.cn")
        or host.endswith(".weixin.qq.com")
        or host.endswith(".tc.qq.com")
        or host.endswith(".qq.com")
    )


def _image_suffix(content: bytes) -> str:
    if content.startswith(b"GIF8"):
        return ".gif"
    if content.startswith(b"\x89PNG"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return ".bin"
