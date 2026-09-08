from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256

from moonlightbox.events.normalization import NormalizedMessage


@dataclass(frozen=True, slots=True)
class MessageSession:
    """由时间间隔划分的一段连续会话。"""

    session_id: str
    messages: tuple[NormalizedMessage, ...]


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """受字符预算约束、可直接提交分析的连续消息窗口。"""

    window_id: str
    session_id: str
    window_ordinal: int
    start_message_ordinal: int
    end_message_ordinal: int
    messages: tuple[NormalizedMessage, ...]
    start_message_id: str
    end_message_id: str
    character_count: int


@dataclass(frozen=True, slots=True)
class PersistenceContext:
    """候选窗口之后按会话顺序排列的持续性观察上下文。"""

    remaining_session_messages: tuple[NormalizedMessage, ...]
    following_sessions: tuple[MessageSession, ...]

    @property
    def messages(self) -> tuple[NormalizedMessage, ...]:
        following_messages = tuple(
            message for session in self.following_sessions for message in session.messages
        )
        return self.remaining_session_messages + following_messages


def split_sessions(
    messages: Sequence[NormalizedMessage],
    *,
    gap: timedelta,
) -> list[MessageSession]:
    """按时间间隔切分会话；间隔本身不产生任何消息或节点。"""

    if gap <= timedelta(0):
        raise ValueError("gap 必须大于零")
    if not messages:
        return []

    ordered = sorted(messages, key=lambda message: (message.timestamp, message.source_id))
    groups: list[list[NormalizedMessage]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.timestamp - previous.timestamp > gap:
            groups.append([])
        groups[-1].append(current)

    return [_make_session(group) for group in groups]


def build_analysis_windows(
    sessions: Sequence[MessageSession],
    *,
    character_budget: int,
    overlap_messages: int = 0,
) -> list[AnalysisWindow]:
    """按内容字符数构建窗口。

    ``overlap_messages`` 是每个窗口期望保留的消息数上限，不是强制值。
    实际重叠最多为上一窗口消息数的一半；仅单条超长消息可以超过字符预算。
    """

    if character_budget <= 0:
        raise ValueError("character_budget 必须大于零")
    if overlap_messages < 0:
        raise ValueError("overlap_messages 不能为负数")

    windows: list[AnalysisWindow] = []
    for session in sessions:
        windows.extend(
            _build_session_windows(
                session,
                character_budget=character_budget,
                overlap_messages=overlap_messages,
            )
        )
    return windows


def get_persistence_context(
    candidate: AnalysisWindow,
    sessions: Sequence[MessageSession],
    *,
    max_sessions: int = 3,
) -> PersistenceContext:
    """取得同会话剩余消息及其后最多三个完整会话。"""

    if not 0 <= max_sessions <= 3:
        raise ValueError("max_sessions 必须在 0 到 3 之间")

    for index, session in enumerate(sessions):
        if session.session_id == candidate.session_id:
            remaining = session.messages[candidate.end_message_ordinal + 1 :]
            following = tuple(sessions[index + 1 : index + 1 + max_sessions])
            return PersistenceContext(
                remaining_session_messages=remaining,
                following_sessions=following,
            )
    raise ValueError("候选窗口不属于给定会话")


def _make_session(messages: list[NormalizedMessage]) -> MessageSession:
    identity = "\x1f".join(
        f"{message.timestamp.isoformat()}\x1e{message.source_id}" for message in messages
    )
    session_id = f"session-{sha256(identity.encode()).hexdigest()[:16]}"
    return MessageSession(session_id=session_id, messages=tuple(messages))


def _build_session_windows(
    session: MessageSession,
    *,
    character_budget: int,
    overlap_messages: int,
) -> list[AnalysisWindow]:
    messages = session.messages
    windows: list[AnalysisWindow] = []
    start = 0

    while start < len(messages):
        end = start
        character_count = 0
        while end < len(messages):
            next_count = len(messages[end].content)
            if end > start and character_count + next_count > character_budget:
                break
            character_count += next_count
            end += 1

        window_messages = messages[start:end]
        windows.append(
            _make_window(
                session.session_id,
                window_ordinal=len(windows),
                start_message_ordinal=start,
                messages=window_messages,
                character_count=character_count,
            )
        )
        if end == len(messages):
            break

        actual_overlap = min(overlap_messages, len(window_messages) // 2)
        start = end - actual_overlap

    return windows


def _make_window(
    session_id: str,
    window_ordinal: int,
    start_message_ordinal: int,
    messages: tuple[NormalizedMessage, ...],
    character_count: int,
) -> AnalysisWindow:
    start_message_id = messages[0].source_id
    end_message_id = messages[-1].source_id
    end_message_ordinal = start_message_ordinal + len(messages) - 1
    identity = (
        f"{session_id}\x1f{window_ordinal}\x1f{start_message_ordinal}\x1f{end_message_ordinal}"
    )
    window_id = f"window-{sha256(identity.encode()).hexdigest()[:16]}"
    return AnalysisWindow(
        window_id=window_id,
        session_id=session_id,
        window_ordinal=window_ordinal,
        start_message_ordinal=start_message_ordinal,
        end_message_ordinal=end_message_ordinal,
        messages=messages,
        start_message_id=start_message_id,
        end_message_id=end_message_id,
        character_count=character_count,
    )
