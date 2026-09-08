import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.media.models import MediaAsset
from moonlightbox.media.service import MediaStore
from moonlightbox.projects.models import Project


@dataclass(frozen=True)
class CompleteExportReport:
    message_count: int
    media_count: int
    avatar_count: int


class CompleteWechatExporter:
    """将数据库中的聊天和已关联媒体整理为可重新导入的完整目录。"""

    def __init__(self, data_dir: Path) -> None:
        self._store = MediaStore(data_dir)

    def export(
        self,
        session: Session,
        *,
        project_id: str,
        destination: Path,
    ) -> CompleteExportReport:
        project = session.get(Project, project_id)
        if project is None:
            raise LookupError("项目不存在")
        source = session.scalar(
            select(ImportSource)
            .where(ImportSource.project_id == project_id)
            .order_by(ImportSource.confirmed_at.desc())
        )
        if source is None:
            raise LookupError("项目没有已确认的聊天导入")
        rows = list(
            session.execute(
                select(Message, Participant, MediaAsset)
                .join(Participant, Participant.id == Message.participant_id)
                .outerjoin(MediaAsset, MediaAsset.id == Message.media_asset_id)
                .where(
                    Message.project_id == project_id,
                    Message.import_id == source.id,
                )
                .order_by(Message.timestamp, Message.source_id)
            )
        )
        if not rows:
            raise ValueError("最新聊天导入中没有消息")
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            media_paths: dict[str, str] = {}
            payload: list[dict[str, object]] = []
            for message, participant, asset in rows:
                media_path = (
                    self._copy_asset(asset, temporary / "media", media_paths)
                    if asset is not None
                    else None
                )
                item: dict[str, object] = {
                    "server_id": message.source_id,
                    "time": message.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender": _participant_name(participant, project.name),
                    "type_name": _type_name(message.kind),
                    "kind": message.kind,
                    "content": message.content,
                }
                if media_path is not None:
                    item["media_path"] = media_path
                payload.append(item)
            (temporary / "chat.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            avatar_count = self._copy_avatars(
                session,
                project_id,
                project.name,
                temporary,
            )
            self._archive_assets(temporary)
            report = CompleteExportReport(
                message_count=len(payload),
                media_count=len(media_paths),
                avatar_count=avatar_count,
            )
            (temporary / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "moonlightbox-complete-wechat-v1",
                        "project_id": project_id,
                        **asdict(report),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if destination.exists():
                marker = destination / "manifest.json"
                if not marker.is_file():
                    raise FileExistsError("目标目录已存在且不是完整导出目录")
                shutil.rmtree(destination)
            temporary.replace(destination)
            return report
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _copy_asset(
        self,
        asset: MediaAsset,
        media_dir: Path,
        copied: dict[str, str],
    ) -> str:
        existing = copied.get(asset.id)
        if existing is not None:
            return existing
        source = self._store.path_for(asset)
        if not source.is_file():
            raise FileNotFoundError(f"媒体文件缺失：{asset.id}")
        suffix = source.suffix.lower()
        relative = Path("media") / f"{asset.sha256}{suffix}"
        target = media_dir / relative.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[asset.id] = relative.as_posix()
        return relative.as_posix()

    def _copy_avatars(
        self,
        session: Session,
        project_id: str,
        target_name: str,
        destination: Path,
    ) -> int:
        rows = list(
            session.execute(
                select(Participant, MediaAsset)
                .join(MediaAsset, MediaAsset.id == Participant.avatar_asset_id)
                .where(Participant.project_id == project_id)
            )
        )
        avatars_dir = destination / "avatars"
        count = 0
        for participant, asset in rows:
            source = self._store.path_for(asset)
            if not source.is_file():
                continue
            avatars_dir.mkdir(parents=True, exist_ok=True)
            display_name = _participant_name(participant, target_name)
            safe_name = display_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            shutil.copy2(
                source,
                avatars_dir / f"{safe_name}-avatar{source.suffix.lower()}",
            )
            count += 1
        return count

    @staticmethod
    def _archive_assets(destination: Path) -> None:
        archive_path = destination / "assets.zip"
        roots = [destination / "media", destination / "avatars"]
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for root in roots:
                if not root.is_dir():
                    continue
                for source in sorted(root.rglob("*")):
                    if source.is_file():
                        archive.write(source, source.relative_to(destination).as_posix())
        for root in roots:
            if root.exists():
                shutil.rmtree(root)


def _type_name(kind: str) -> str:
    return {
        "text": "文本",
        "image": "图片",
        "video_thumbnail": "图片",
        "sticker": "表情",
        "video": "视频",
        "audio": "语音",
        "file": "文件",
        "system": "系统",
    }.get(kind, "其他")


def _participant_name(participant: Participant, target_name: str) -> str:
    return target_name if participant.role == "target" else participant.name
