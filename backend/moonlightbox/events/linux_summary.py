"""Linux 上的节点展示摘要生成器。"""

from __future__ import annotations

from collections.abc import Sequence

from moonlightbox.agent.linux_inference import SharedLinuxModelRuntime
from moonlightbox.events.local_summary import LocalSummaryError, clean_summary
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.v3_ranking import V3RankedCandidate


class LinuxEventNarrativeSummarizer:
    """使用干净 Hugging Face 基座生成可选的节点展示回忆。"""

    def __init__(self, model_path: str, *, device: str = "auto", load_in_4bit: bool = True) -> None:
        self.model_path = model_path
        self._runtime = SharedLinuxModelRuntime(device=device, load_in_4bit=load_in_4bit)

    def summarize(
        self,
        candidate: V3RankedCandidate,
        messages: Sequence[NormalizedMessage],
    ) -> str:
        transcript = "\n".join(
            f"{message.timestamp.isoformat()} {message.sender}：{message.content}"
            for message in messages
        )
        prompt = (
            "你负责把一段真实聊天整理成自然的中文回忆。只依据原文，用两到三句话讲清发生了什么、"
            "双方当时明确表达的状态，以及聊天中确实发生的关系变化。使用‘两个人’或原文姓名叙述，"
            "不要替任何一方使用第一人称。不要猜测内心、延伸之后的结果或提到模型。"
            "直接输出80到180字的正文。"
        )
        for attempt in range(2):
            raw = self._runtime.generate_raw(
                base_model=self.model_path,
                adapter_path=None,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"标题：{candidate.candidate.title}\n聊天原文：\n{transcript}",
                    },
                ],
                max_tokens=240,
            )
            summary = clean_summary(raw)
            if len(summary) >= 20:
                return summary
            if attempt == 0:
                continue
        raise LocalSummaryError("Linux 本地模型生成的回忆摘要过短")
