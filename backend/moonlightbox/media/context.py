"""把已审计的媒体语义转换为模型可读、不可误解的聊天上下文。"""

import json

from moonlightbox.imports.models import Message
from moonlightbox.imports.wechat_rendering import normalize_wechat_display
from moonlightbox.media.models import MediaSemanticAnnotation


def model_message_content(
    message: Message,
    annotation: MediaSemanticAnnotation | None = None,
) -> tuple[str, str]:
    """返回（可见类型，模型上下文）。

    媒体“能理解”不等于“能复用”：这里只读 succeeded 标注，
    不参考 reuse_decision。未成功标注时显式告诉模型内容未知。
    """

    visible_kind, visible_content = normalize_wechat_display(
        message.kind,
        message.content,
    )
    if visible_kind == "sticker":
        return visible_kind, "[表情]"
    if visible_kind == "call":
        return visible_kind, f"[通话事件：{visible_content}]"
    if visible_kind == "system":
        return visible_kind, f"[系统事件：{visible_content}]"
    if visible_kind == "reaction":
        return visible_kind, f"[聊天动作：{visible_content}]"
    if visible_kind == "quote":
        try:
            quoted = json.loads(visible_content)
        except (json.JSONDecodeError, TypeError):
            quoted = {}
        if isinstance(quoted, dict):
            text = quoted.get("text")
            quoted_content = quoted.get("quoted_content")
            if isinstance(text, str) and isinstance(quoted_content, str):
                return "quote", f"[引用消息：{quoted_content}]\n{text}"

    if message.kind == "audio":
        transcript = _successful_text(annotation, "transcript")
        confidence = _confidence(annotation)
        return "audio", (
            f"[语音转写：{transcript}]"
            if transcript and confidence >= 0.75
            else f"[自动语音转写（可能有误）：{transcript}]"
            if transcript and confidence >= 0.35
            else "[语音，内容未转写]"
        )
    if message.kind in {"image", "video_thumbnail"}:
        summary = _successful_text(annotation, "summary")
        ocr_text = _successful_text(annotation, "ocr_text")
        details = []
        if summary:
            details.append(summary)
        if ocr_text:
            details.append("可见文字：" + ocr_text)
        confidence = _confidence(annotation)
        return "image", (
            "[图片内容：" + "；".join(details) + "]"
            if details and confidence >= 0.85
            else "[自动图片描述（可能有误）：" + "；".join(details) + "]"
            if details and confidence >= 0.55
            else "[图片，内容未标注]"
        )
    if message.kind == "video":
        summary = _successful_text(annotation, "summary")
        return "video", (
            f"[视频内容：{summary}]" if summary else "[视频，内容未标注]"
        )
    return visible_kind, visible_content


def is_reply_label_message(kind: str) -> bool:
    """系统/通话事件可作时序背景，但不能独自制造“应回复”标签。"""

    return kind not in {"call", "system"}


def _successful_text(
    annotation: MediaSemanticAnnotation | None,
    field: str,
) -> str:
    if annotation is None or annotation.status != "succeeded":
        return ""
    value = getattr(annotation, field, "")
    return value.strip() if isinstance(value, str) else ""


def _confidence(annotation: MediaSemanticAnnotation | None) -> float:
    return float(annotation.confidence) if annotation is not None else 0.0
