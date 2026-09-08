import re
from dataclasses import dataclass
from typing import Literal

MemoryQueryIntent = Literal[
    "current_state",
    "commitment",
    "prior_utterance",
    "relationship",
    "preference",
    "past_event",
    "general",
]


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    intent: MemoryQueryIntent
    retrieve_project_history: bool
    retrieve_branch_continuity: bool


def retrieval_policy(content: str) -> MemoryRetrievalPolicy:
    """Route a turn to memory layers before semantic retrieval.

    Current state is deliberately not a vector-memory question. Its only
    authority is the expiring situational projection carried by the branch.
    """

    compact = re.sub(r"\s+", "", _direct_user_content(content))
    if is_current_state_question(compact):
        return MemoryRetrievalPolicy("current_state", False, False)
    if any(marker in compact for marker in ("答应", "保证", "说好", "承诺")):
        return MemoryRetrievalPolicy("commitment", True, True)
    if any(marker in compact for marker in ("说过", "讲过", "提过", "聊过")):
        return MemoryRetrievalPolicy("prior_utterance", True, True)
    if any(marker in compact for marker in ("记得", "以前", "之前", "那次", "当时", "去过")):
        return MemoryRetrievalPolicy("past_event", True, True)
    if any(marker in compact for marker in ("我们", "关系", "爱我", "在乎", "分手")):
        return MemoryRetrievalPolicy("relationship", True, True)
    if any(marker in compact for marker in ("喜欢", "讨厌", "习惯", "偏好")):
        return MemoryRetrievalPolicy("preference", True, True)
    return MemoryRetrievalPolicy("general", True, True)


def _direct_user_content(content: str) -> str:
    """Ignore quoted chat history when deciding what the user is asking now."""

    direct_lines = [
        line
        for line in content.splitlines()
        if not line.lstrip().startswith((">", "＞", "[引用消息："))
    ]
    return "\n".join(direct_lines).strip()


def is_current_state_question(content: str) -> bool:
    compact = re.sub(r"\s+", "", content)
    if re.search(
        r"(?:在干嘛|干嘛呢|做什么呢?|在哪里|在哪呢?|忙什么|忙不忙|忙吗|"
        r"睡了吗|吃了吗|下班了吗|天气怎么样|天气好吗|下雨了吗|下雪了吗|"
        r"可以电话吗|可以打电话吗|能打电话吗|方便电话吗|方便接电话吗|方便吗|"
        r"几盒|几份|多少|几个|"
        r"[一二三四五六七八九十两\d]+(?:盒|份|个|件|张|瓶|袋)吗)",
        compact,
    ):
        return True
    if not compact.rstrip().endswith(("吗", "呢", "？", "?")):
        return False
    return bool(
        re.search(
            r"(?:现在|目前|刚才|今天|是否|已经|还在(?!乎)).{0,10}"
            r"(?:做|干|吃|睡|忙|上班|下班|工作|开会|炒股|停止|在哪|哪里|"
            r"开心|难过|生气|舒服|累|心情|电话|方便)",
            compact,
        )
    )
