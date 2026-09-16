"""云端输入被风控拒绝时的最小遮罩:逐条消息替换正文,注册表跨流程复用。

不做事先内容检测——供应商的 422 就是免费且准确的检测器。被拒绝后在新增
上下文里二分定位真正触发的消息,只遮这几条,其余原文照常发送;遮住的消息
进入项目级注册表,之后的回合和其他流程发送前自动避开,定位成本只付一次。
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import ForeignKey, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from moonlightbox.db import Base

MASK_PLACEHOLDER = "[该消息因内容策略未发送云端,仅保留时间与发送人]"
CHUNK_PLACEHOLDER = "[该检索片段因内容策略未发送云端]"

# ("ref", message_ref) 持久化到注册表;("raw", 消息下标) 只遮本次发送的自由文本。
Unit = tuple[str, str | int]


class SensitiveContentMask(Base):
    __tablename__ = "sensitive_content_masks"
    __table_args__ = (UniqueConstraint("project_id", "message_ref"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    message_ref: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class MaskRegistry:
    """项目级遮罩注册表;所有云端流程共享同一份。"""

    def __init__(self, engine, project_id):
        self._engine = engine
        self._project_id = project_id

    def load(self) -> frozenset[str]:
        with Session(self._engine) as session:
            return frozenset(
                session.scalars(
                    select(SensitiveContentMask.message_ref).where(
                        SensitiveContentMask.project_id == self._project_id
                    )
                )
            )

    def add(self, refs, *, source: str) -> list[str]:
        refs = [str(ref) for ref in refs]
        if not refs:
            return []
        with Session(self._engine) as session:
            existing = set(
                session.scalars(
                    select(SensitiveContentMask.message_ref).where(
                        SensitiveContentMask.project_id == self._project_id,
                        SensitiveContentMask.message_ref.in_(refs),
                    )
                )
            )
            added = [ref for ref in dict.fromkeys(refs) if ref not in existing]
            session.add_all(
                SensitiveContentMask(
                    project_id=self._project_id, message_ref=ref, source=source
                )
                for ref in added
            )
            session.commit()
        return added


def _mask_entry(entry, refs: set[str]) -> int:
    if (
        isinstance(entry, dict)
        and entry.get("message_ref") in refs
        and isinstance(entry.get("content"), str)
        and entry["content"] != MASK_PLACEHOLDER
    ):
        entry["content"] = MASK_PLACEHOLDER
        return 1
    return 0


def _mask_json_obj(obj: dict, refs: set[str]) -> int:
    count = 0
    messages = obj.get("messages")
    if isinstance(messages, list):
        count += sum(_mask_entry(entry, refs) for entry in messages)
    fragments = obj.get("fragments")
    if isinstance(fragments, list):
        touched = False
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            fragment_refs = {str(ref) for ref in fragment.get("message_refs") or []}
            hit = bool(fragment_refs & refs)
            entries = fragment.get("messages")
            if isinstance(entries, list):
                count += sum(_mask_entry(entry, refs) for entry in entries)
            if hit and isinstance(fragment.get("chunk"), str):
                if fragment["chunk"] != CHUNK_PLACEHOLDER:
                    fragment["chunk"] = CHUNK_PLACEHOLDER
                    count += 1
            touched = touched or hit
        # 检索结果顶层的 context 是原文拼接,没有引用可定位;命中片段时整体遮罩。
        if touched and isinstance(obj.get("context"), str) and obj["context"] != CHUNK_PLACEHOLDER:
            obj["context"] = CHUNK_PLACEHOLDER
            count += 1
    return count


def _parse_json_obj(content):
    if not isinstance(content, str):
        return None
    try:
        obj = json.loads(content)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def mask_messages(messages: Sequence, refs: set[str], raw: set[int] = frozenset()) -> list:
    """返回遮罩后的新消息列表;不改动原消息,不触碰无法解析的内容。"""

    masked = []
    for index, message in enumerate(messages):
        content = getattr(message, "content", None)
        if index in raw and isinstance(content, str) and content != MASK_PLACEHOLDER:
            masked.append(message.model_copy(update={"content": MASK_PLACEHOLDER}))
            continue
        obj = _parse_json_obj(content)
        if obj is None or not refs:
            masked.append(message)
            continue
        if _mask_json_obj(obj, refs):
            masked.append(
                message.model_copy(update={"content": json.dumps(obj, ensure_ascii=False)})
            )
        else:
            masked.append(message)
    return masked


def collect_candidates(messages: Sequence) -> list[Unit]:
    """收集可遮罩单元:JSON 里带 message_ref 的消息条目,以及无法解析的自由文本。"""

    refs: list[str] = []
    seen: set[str] = set()
    raw: list[int] = []
    for index, message in enumerate(messages):
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content or content == MASK_PLACEHOLDER:
            continue
        obj = _parse_json_obj(content)
        if obj is None:
            raw.append(index)
            continue
        buckets = [obj.get("messages") or []]
        for fragment in obj.get("fragments") or []:
            if isinstance(fragment, dict) and isinstance(fragment.get("messages"), list):
                buckets.append(fragment["messages"])
        for bucket in buckets:
            for entry in bucket:
                if not isinstance(entry, dict):
                    continue
                ref = entry.get("message_ref")
                if (
                    isinstance(ref, str)
                    and ref not in seen
                    and isinstance(entry.get("content"), str)
                    and entry["content"] != MASK_PLACEHOLDER
                ):
                    seen.add(ref)
                    refs.append(ref)
    return [("ref", ref) for ref in refs] + [("raw", index) for index in raw]


class StillRejected(Exception):
    """try_call 把供应商风控拒绝翻译成这个内部信号;其他异常原样上抛。"""


class SensitiveContentHandler:
    """挂在 ChatModel 上的发送策略:已知遮罩先行,拒绝后二分定位最小遮罩集。"""

    def __init__(
        self,
        registry: MaskRegistry,
        *,
        source: str,
        max_trials: int = 16,
        on_mask: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._registry = registry
        self._source = source
        self._max_trials = max(2, max_trials)
        self._on_mask = on_mask
        self.known: set[str] = set(registry.load())
        # raw 遮罩只记住绝对下标,无法持久化;历史被追加时下标保持有效。
        self._session_raw: set[int] = set()
        self._accepted_len = 0

    def masked_view(self, messages: Sequence) -> list:
        if not self.known and not self._session_raw:
            return list(messages)
        return mask_messages(messages, set(self.known), set(self._session_raw))

    def note_accepted(self, messages: Sequence) -> None:
        # 压缩/恢复会改写历史;长度回退时下标假设失效,session raw 遮罩作废。
        if len(messages) < self._accepted_len:
            self._session_raw.clear()
        self._accepted_len = len(messages)

    def recover(self, messages: Sequence, try_call):
        """try_call(masked_messages) 返回结果或抛 provider_input_rejected。

        成功时返回模型结果并持久化新增遮罩;定位失败返回 None,由调用方抛原始错误。
        """

        # 候选在整个消息列表上收集:供应商审核按整次请求判定,历史里已被接受过的
        # 消息也可能在新一轮触发,raw 下标因此保持绝对下标。
        units = collect_candidates(messages)
        if not units:
            return None
        trials = 0

        def trial(masked_refs: set[str], masked_raw: set[int]) -> bool:
            nonlocal trials
            trials += 1
            view = mask_messages(
                messages, set(self.known) | masked_refs, self._session_raw | masked_raw
            )
            try:
                try_call(view)
            except StillRejected:
                return False
            return True

        all_refs = {unit[1] for unit in units if unit[0] == "ref"}
        all_raw = {unit[1] for unit in units if unit[0] == "raw"}
        # 嫌疑集全遮仍被拒,说明触发内容不在可定位单元里,放弃而不是无限重试。
        if not trial(set(all_refs), set(all_raw)):
            return None
        masked_refs = set(all_refs)
        masked_raw = set(all_raw)
        pending = [sorted(all_refs)]
        while pending and trials < self._max_trials:
            # 深度优先:先把含触发项的分支收窄到底,预算才不会浪费在无关分支上。
            group = pending.pop()
            if not group or not set(group) <= masked_refs:
                continue
            if len(group) == 1:
                if trial(masked_refs - set(group), masked_raw):
                    masked_refs -= set(group)
                continue
            half = set(group[: len(group) // 2])
            if trial(masked_refs - half, masked_raw):
                masked_refs -= half
                pending.append(group[len(group) // 2 :])
            else:
                pending.append(group[len(group) // 2 :])
                pending.append(group[: len(group) // 2])
        # raw 单元通常很少,逐条试摘,避免把无辜的自由文本一起遮掉。
        for index in sorted(masked_raw):
            if trials >= self._max_trials:
                break
            if trial(masked_refs, masked_raw - {index}):
                masked_raw -= {index}
        # 以最终遮罩集正式调用一次,使用其结果作为本次 invoke 的产出。供应商判定
        # 存在非确定性,这一步仍可能被拒;此时放弃定位,交回原始错误而不是泄漏内部信号。
        view = mask_messages(
            messages, set(self.known) | masked_refs, self._session_raw | masked_raw
        )
        try:
            result = try_call(view)
        except StillRejected:
            return None
        added = self._registry.add(sorted(masked_refs), source=self._source)
        self.known |= set(added)
        self._session_raw |= masked_raw
        if added and self._on_mask is not None:
            self._on_mask(added)
        return result
