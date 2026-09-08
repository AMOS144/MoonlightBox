import json
import re
import xml.etree.ElementTree as ET


def normalize_wechat_display(kind: str, content: str) -> tuple[str, str]:
    """把微信协议内容转换成用户可读的消息。"""
    stripped = content.strip()
    if "<voipmsg" in stripped:
        return "call", _call_summary(stripped)
    if '<sysmsg type="revokemsg">' in stripped:
        return "system", "撤回了一条消息"
    if "拍了拍" in stripped or "拍一拍" in stripped:
        return "reaction", "拍了拍"
    quote = re.fullmatch(
        r"(?P<text>.*?)\n>\s*[^:：\n]+[:：]\s*(?P<quoted>.*)",
        stripped,
        flags=re.DOTALL,
    )
    if quote is not None:
        return "quote", json.dumps(
            {
                "text": quote.group("text").strip(),
                "quoted_content": _quoted_summary(quote.group("quoted").strip()),
            },
            ensure_ascii=False,
        )
    if "<voicemsg" in stripped:
        return "text", "[语音]"
    return kind, content


def _call_summary(content: str) -> str:
    start = content.find("<voipmsg")
    try:
        root = ET.fromstring(content[start:])
    except ET.ParseError:
        return "通话"
    message = (root.findtext(".//msg") or "通话已结束").strip()
    room_type = (root.findtext(".//room_type") or "0").strip()
    call_type = "视频通话" if room_type == "1" else "语音通话"
    return f"{call_type} · {message}"


def _quoted_summary(content: str) -> str:
    if "<voicemsg" in content:
        return "[语音]"
    if "<img " in content:
        return "[图片]"
    if "<videomsg" in content:
        return "[视频]"
    if content.startswith("<"):
        return "[消息]"
    return content
