import importlib
import re
from collections.abc import Sequence
from typing import Protocol, cast

from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.v3_ranking import V3RankedCandidate


class LocalSummaryError(RuntimeError):
    pass


class MlxSummaryTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


class MlxSummaryRuntime(Protocol):
    def load(self, model_path: str) -> tuple[object, MlxSummaryTokenizer]: ...

    def generate(
        self,
        model: object,
        tokenizer: object,
        *,
        prompt: str,
        max_tokens: int,
        sampler: object,
        logits_processors: list[object],
        verbose: bool,
    ) -> str: ...


class MlxSummarySampleUtils(Protocol):
    def make_sampler(
        self,
        *,
        temp: float,
        top_p: float,
        min_p: float,
    ) -> object: ...

    def make_repetition_penalty(
        self,
        *,
        penalty: float,
        context_size: int,
    ) -> object: ...


class MlxEventNarrativeSummarizer:
    """使用同一个本地模型实例逐节点生成展示用回忆摘要。"""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._runtime: MlxSummaryRuntime | None = None
        self._sample_utils: MlxSummarySampleUtils | None = None
        self._model: object | None = None
        self._tokenizer: MlxSummaryTokenizer | None = None

    def summarize(
        self,
        candidate: V3RankedCandidate,
        messages: Sequence[NormalizedMessage],
    ) -> str:
        self._ensure_loaded()
        if (
            self._runtime is None
            or self._sample_utils is None
            or self._model is None
            or self._tokenizer is None
        ):
            raise LocalSummaryError("本地摘要模型未正确加载")
        transcript = "\n".join(
            f"{message.timestamp.isoformat()} {message.sender}：{message.content}"
            for message in messages
        )
        prompt = self._tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "你负责把一段真实聊天整理成自然的中文回忆。"
                        "只依据原文，用两到三句话讲清发生了什么、双方当时明确表达的状态，"
                        "以及聊天中确实发生的关系变化。使用“两个人”或原文姓名叙述，"
                        "不要替任何一方使用第一人称。语气像在回忆往事，不要写分析报告，"
                        "不要列分数，不要提模型或节点。不得猜测内心，不得把玩笑判断为承诺，"
                        "不得写原文时间范围之后的结果、感悟或成长；除非原文明确提到，"
                        "不要使用“后来”“从那以后”“意识到”“学会了”等延伸结论。"
                        "直接输出80到180字的正文。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"标题：{candidate.candidate.title}\n"
                        f"聊天原文：\n{transcript}"
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for attempt in range(2):
            raw = self._runtime.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=240,
                sampler=self._sample_utils.make_sampler(
                    temp=0.2 + attempt * 0.1,
                    top_p=0.9,
                    min_p=0.0,
                ),
                logits_processors=[
                    self._sample_utils.make_repetition_penalty(
                        penalty=1.08,
                        context_size=96,
                    )
                ],
                verbose=False,
            )
            summary = _clean_summary(raw)
            if len(summary) >= 20:
                return summary
        raise LocalSummaryError("本地模型生成的回忆摘要过短")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            self._runtime = cast(
                MlxSummaryRuntime,
                importlib.import_module("mlx_lm"),
            )
            self._sample_utils = cast(
                MlxSummarySampleUtils,
                importlib.import_module("mlx_lm.sample_utils"),
            )
            self._model, self._tokenizer = self._runtime.load(self.model_path)
        except (ImportError, OSError, ValueError) as error:
            raise LocalSummaryError("无法加载本地节点摘要模型") from error


def _clean_summary(raw: str) -> str:
    without_thinking = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    normalized = without_thinking.strip().strip('"').strip("“”")
    normalized = normalized.replace("她们", "两个人").replace("“此入”", "“此人”")
    normalized = re.sub(r"我(?=回应|则|表示|提到|已经|也在)", "另一方", normalized)
    return re.sub(r"\s+", " ", normalized)[:300].strip()
