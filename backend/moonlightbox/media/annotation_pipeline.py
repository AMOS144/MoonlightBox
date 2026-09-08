# ruff: noqa: E501

import ast
import importlib.metadata
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.media.models import MediaAsset, MediaSemanticAnnotation
from moonlightbox.media.service import MediaAnnotationService, MediaStore


@dataclass(frozen=True)
class SemanticResult:
    summary: str = ""
    transcript: str = ""
    ocr_text: str = ""
    safety_tags: tuple[str, ...] = ()
    confidence: float = 0.0


class AudioTranscriber(Protocol):
    model_id: str
    source_version: str

    def transcribe(self, path: Path) -> SemanticResult: ...


class VisualAnnotator(Protocol):
    model_id: str
    source_version: str

    def annotate(self, path: Path) -> SemanticResult: ...


class LinuxWhisperTranscriber:
    """用 Transformers Whisper 在 Linux CUDA/CPU 上转写音频。"""

    def __init__(self, model_id: str = "openai/whisper-small") -> None:
        self.model_id = model_id
        self.source_version = _package_version("transformers")
        if self.source_version == "not-installed":
            raise RuntimeError("请执行 uv sync --extra linux-ml 后再执行音频转写")
        self._pipeline: object | None = None

    def transcribe(self, path: Path) -> SemanticResult:
        pipeline = self._load()
        raw = pipeline(str(path), generate_kwargs={"language": "chinese", "task": "transcribe"})
        if not isinstance(raw, dict):
            raise ValueError("Whisper 返回格式无效")
        transcript = str(raw.get("text", "")).strip()
        return SemanticResult(
            transcript=transcript,
            safety_tags=tuple(_text_safety_tags(transcript)),
            confidence=0.7 if transcript else 0.0,
        )

    def _load(self) -> object:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError("请执行 uv sync --extra linux-ml 后再执行音频转写") from error
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
        )
        return self._pipeline


class LinuxVisualAnnotator:
    """用 Transformers 视觉语言模型在 Linux 上生成受限图片语义。"""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.model_id = model_id
        self.source_version = _package_version("transformers")
        if self.source_version == "not-installed":
            raise RuntimeError("请执行 uv sync --extra linux-ml 后再执行图片理解")
        self._runtime: tuple[object, object, object] | None = None

    def annotate(self, path: Path) -> SemanticResult:
        torch, model, processor = self._load()
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Linux 图片理解需要 Pillow") from error
        prompt = (
            "只依据图片内容输出一个 JSON 对象，不要 Markdown。字段："
            "summary（简洁中文客观描述）、ocr_text（可见文字，没有则空字符串）、"
            "safety_tags（仅 sensitive,private_document,identity_document,financial,medical,explicit,child）、"
            "confidence（0到1）。人物身份不确定时不要猜测姓名。"
        )
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=240, do_sample=False)
        input_length = inputs["input_ids"].shape[-1]
        payload = _json_object(
            processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0]
        )
        summary = _string(payload.get("summary"))
        ocr_text = _string(payload.get("ocr_text"))
        tags = _safety_tags(payload.get("safety_tags"))
        tags.update(_text_safety_tags(ocr_text))
        if not summary:
            raise ValueError("视觉模型没有返回有效 summary")
        return SemanticResult(
            summary=summary,
            ocr_text=ocr_text,
            safety_tags=tuple(sorted(tags)),
            confidence=_bounded_confidence(payload.get("confidence"), maximum=0.9),
        )

    def _load(self) -> tuple[object, object, object]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("请执行 uv sync --extra linux-ml 后再执行图片理解") from error
        kwargs: dict[str, object] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            kwargs.update(
                {
                    "torch_dtype": torch.float16,
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    "device_map": "auto",
                }
            )
        else:
            kwargs["torch_dtype"] = torch.float32
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        if not torch.cuda.is_available():
            model.to("cpu")
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._runtime = torch, model, processor
        return self._runtime


class MlxWhisperTranscriber:
    def __init__(
        self,
        model_id: str = "mlx-community/whisper-large-v3-turbo",
    ) -> None:
        self.model_id = model_id
        self.source_version = _package_version("mlx-whisper")
        if self.source_version == "not-installed":
            raise RuntimeError("请安装 mlx 可选依赖后再执行音频转写")

    def transcribe(self, path: Path) -> SemanticResult:
        try:
            import mlx_whisper  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("请安装 mlx 可选依赖后再执行音频转写") from error
        raw = mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=self.model_id,
            language="zh",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            verbose=False,
        )
        transcript = str(raw.get("text", "")).strip()
        segments = raw.get("segments")
        confidence = _whisper_confidence(segments if isinstance(segments, list) else [])
        return SemanticResult(
            transcript=transcript,
            safety_tags=tuple(_text_safety_tags(transcript)),
            confidence=confidence,
        )


class MlxVisualAnnotator:
    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    ) -> None:
        self.model_id = model_id
        self.source_version = _package_version("mlx-vlm")
        if self.source_version == "not-installed":
            raise RuntimeError("请安装 mlx 可选依赖后再执行图片理解")
        self._runtime: tuple[object, object, object] | None = None

    def annotate(self, path: Path) -> SemanticResult:
        model, processor, config = self._load()
        try:
            from mlx_vlm import generate  # type: ignore[import-not-found]
            from mlx_vlm.prompt_utils import (  # type: ignore[import-not-found]
                apply_chat_template,
            )
        except ImportError as error:
            raise RuntimeError("请安装 mlx 可选依赖后再执行图片理解") from error
        prompt = (
            "只依据图片内容输出一个 JSON 对象，不要 Markdown。字段："
            "summary（简洁中文客观描述）、ocr_text（可见文字，没有则空字符串）、"
            "safety_tags（只可从 sensitive,private_document,identity_document,financial,"
            "medical,explicit,child 中选择）、confidence（0到1）。"
            "人物身份不确定时不要猜测姓名；涉及证件、账单、病历、儿童或裸露必须标注。"
        )
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        output = generate(
            model,
            processor,
            formatted,
            [str(path)],
            verbose=False,
            max_tokens=240,
            temperature=0.0,
        )
        text = getattr(output, "text", output)
        payload = _json_object(str(text))
        summary = _string(payload.get("summary"))
        ocr_text = _string(payload.get("ocr_text"))
        tags = _safety_tags(payload.get("safety_tags"))
        tags.update(_text_safety_tags(ocr_text))
        confidence = _bounded_confidence(payload.get("confidence"), maximum=0.9)
        if not summary:
            raise ValueError("视觉模型没有返回有效 summary")
        return SemanticResult(
            summary=summary,
            ocr_text=ocr_text,
            safety_tags=tuple(sorted(tags)),
            confidence=confidence,
        )

    def _load(self) -> tuple[object, object, object]:
        if self._runtime is None:
            try:
                from mlx_vlm import load
                from mlx_vlm.utils import load_config  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("请安装 mlx 可选依赖后再执行图片理解") from error
            model, processor = load(self.model_id)
            self._runtime = (model, processor, load_config(self.model_id))
        return self._runtime


def annotate_project_media(
    session: Session,
    *,
    project_id: str,
    store: MediaStore,
    audio_transcriber: AudioTranscriber | None = None,
    visual_annotator: VisualAnnotator | None = None,
    retry_failed: bool = False,
    retry_below_confidence: float | None = None,
    limit: int | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    existing = {
        item.asset_id: item
        for item in session.scalars(
            select(MediaSemanticAnnotation).where(MediaSemanticAnnotation.project_id == project_id)
        )
    }
    assets = list(
        session.scalars(
            select(MediaAsset)
            .join(Message, Message.media_asset_id == MediaAsset.id)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                MediaAsset.project_id == project_id,
                Message.project_id == project_id,
                Participant.role.in_(("self", "target")),
            )
            .distinct()
            .order_by(MediaAsset.created_at, MediaAsset.id)
        )
    )
    counts = {
        "eligible": 0,
        "succeeded": 0,
        "failed": 0,
        "needs_review": 0,
        "skipped": 0,
        "audio": 0,
        "image": 0,
    }
    service = MediaAnnotationService(session)
    processed = 0
    for asset in assets:
        modality = _supported_modality(asset)
        if modality is None:
            counts["skipped"] += 1
            continue
        previous = existing.get(asset.id)
        if previous is not None:
            retry_for_failure = previous.status == "failed" and retry_failed
            retry_for_confidence = (
                retry_below_confidence is not None and previous.confidence < retry_below_confidence
            )
            if not retry_for_failure and not retry_for_confidence:
                counts["skipped"] += 1
                continue
        provider: AudioTranscriber | VisualAnnotator | None = (
            audio_transcriber if modality == "audio" else visual_annotator
        )
        if provider is None:
            counts["skipped"] += 1
            continue
        if limit is not None and processed >= limit:
            break
        counts["eligible"] += 1
        processed += 1
        try:
            path = store.path_for(asset)
            if not path.is_file():
                raise FileNotFoundError("媒体文件不存在")
            result = (
                cast(AudioTranscriber, provider).transcribe(path)
                if modality == "audio"
                else cast(VisualAnnotator, provider).annotate(path)
            )
            minimum_confidence = 0.35 if modality == "audio" else 0.55
            empty_transcript = modality == "audio" and not result.transcript.strip()
            low_confidence = result.confidence < minimum_confidence
            annotation_status = (
                "needs_review" if empty_transcript or low_confidence else "succeeded"
            )
            failure_code = (
                "empty_transcript"
                if empty_transcript
                else "low_confidence"
                if low_confidence
                else None
            )
            service.annotate(
                project_id,
                asset.id,
                summary=result.summary,
                transcript=result.transcript,
                ocr_text=result.ocr_text,
                safety_tags=list(result.safety_tags),
                source_model=provider.model_id,
                source_version=provider.source_version,
                confidence=result.confidence,
                status=annotation_status,
                failure_code=failure_code,
                reuse_decision="pending",
            )
            counts[annotation_status] += 1
            if annotation_status == "succeeded":
                counts[modality] += 1
        except Exception as error:
            session.rollback()
            service.annotate(
                project_id,
                asset.id,
                source_model=provider.model_id,
                source_version=provider.source_version,
                confidence=0.0,
                status="failed",
                failure_code=type(error).__name__[:64],
                reuse_decision="pending",
            )
            counts["failed"] += 1
        if progress is not None:
            progress({**counts, "asset_id": asset.id, "modality": modality})
    return {"project_id": project_id, **counts}


def _supported_modality(asset: MediaAsset) -> str | None:
    mime = asset.mime_type.casefold()
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("image/") and asset.kind != "sticker":
        return "image"
    return None


def _whisper_confidence(segments: list[object]) -> float:
    values: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        average_log_probability = segment.get("avg_logprob")
        no_speech_probability = segment.get("no_speech_prob")
        if not isinstance(average_log_probability, int | float):
            continue
        speech = (
            1 - float(no_speech_probability)
            if isinstance(no_speech_probability, int | float)
            else 1.0
        )
        values.append(math.exp(min(0.0, float(average_log_probability))) * speech)
    return max(0.0, min(0.98, sum(values) / len(values))) if values else 0.0


def _json_object(value: str) -> dict[str, object]:
    cleaned = value.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("视觉模型输出不是 JSON 对象")
    candidate = cleaned[start:]
    try:
        payload, _end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        end = candidate.rfind("}")
        if end <= 0:
            raise ValueError("视觉模型输出不是 JSON 对象") from None
        try:
            payload = ast.literal_eval(candidate[: end + 1])
        except (ValueError, SyntaxError):
            raise ValueError("视觉模型输出不是 JSON 对象") from None
    if not isinstance(payload, dict):
        raise ValueError("视觉模型输出不是 JSON 对象")
    return payload


def _bounded_confidence(value: object, *, maximum: float) -> float:
    if not isinstance(value, int | float | str) or isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(maximum, parsed))


def _text_safety_tags(value: str) -> set[str]:
    tags: set[str] = set()
    compact = re.sub(r"\s+", "", value)
    if re.search(r"(?<!\d)1\d{10}(?!\d)", compact):
        tags.add("sensitive")
    if re.search(r"(?<!\d)\d{17}[\dXx](?!\d)", compact):
        tags.add("identity_document")
    if re.search(r"(?<!\d)\d{16,19}(?!\d)", compact):
        tags.add("financial")
    return tags


def _safety_tags(value: object) -> set[str]:
    allowed = {
        "sensitive",
        "private_document",
        "identity_document",
        "financial",
        "medical",
        "explicit",
        "child",
    }
    return (
        {
            item.strip().casefold()
            for item in value
            if isinstance(item, str) and item.strip().casefold() in allowed
        }
        if isinstance(value, list)
        else set()
    )


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
