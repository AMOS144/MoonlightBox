import re
from collections.abc import Sequence

PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IDENTITY_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def redact_text(text: str, custom_terms: Sequence[str] = ()) -> str:
    redacted = PHONE_PATTERN.sub("[手机号]", text)
    redacted = EMAIL_PATTERN.sub("[邮箱]", redacted)
    redacted = IDENTITY_PATTERN.sub("[身份证]", redacted)
    for term in sorted(set(custom_terms), key=len, reverse=True):
        if term:
            redacted = redacted.replace(term, "[自定义敏感词]")
    return redacted
