import argparse
import json

from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.media.annotation_pipeline import (
    LinuxVisualAnnotator,
    LinuxWhisperTranscriber,
    annotate_project_media,
)
from moonlightbox.media.service import MediaStore
from moonlightbox.projects.models import Project


def main() -> None:
    parser = argparse.ArgumentParser(description="本地图片理解与音频转写")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--modality", choices=("all", "image", "audio"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-below-confidence", type=float)
    parser.add_argument(
        "--vision-model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--whisper-model",
        default="openai/whisper-small",
    )
    args = parser.parse_args()
    if args.retry_below_confidence is not None and not (0 <= args.retry_below_confidence <= 1):
        parser.error("--retry-below-confidence 必须位于 0 到 1")
    settings = Settings()
    database = Database(settings.database_url)
    audio = (
        LinuxWhisperTranscriber(args.whisper_model) if args.modality in {"all", "audio"} else None
    )
    visual = LinuxVisualAnnotator(args.vision_model) if args.modality in {"all", "image"} else None
    with Session(database.engine) as session:
        if session.get(Project, args.project_id) is None:
            parser.error("项目不存在")
        result = annotate_project_media(
            session,
            project_id=args.project_id,
            store=MediaStore(settings.data_dir),
            audio_transcriber=audio,
            visual_annotator=visual,
            retry_failed=args.retry_failed,
            retry_below_confidence=args.retry_below_confidence,
            limit=args.limit,
            progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
        )
    database.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
