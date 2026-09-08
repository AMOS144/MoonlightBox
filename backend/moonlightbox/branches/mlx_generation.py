import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import IdentityKernel
from moonlightbox.branches.generation import (
    GenerationFailedError,
    GeneratorUnavailableError,
)
from moonlightbox.branches.replies import (
    GeneratedBubble,
    GeneratedReplyTurn,
    ReplyStructureError,
    parse_persona_text_turn,
    parse_reply_turn,
)
from moonlightbox.db import Database
from moonlightbox.training.bubble_protocol import (
    allowed_sticker_ids_from_prompt,
    compact_retry_instruction,
    evidence_grounded_style_fallback_lines,
    persona_style_transfer_instruction,
    prompt_is_persona_style_transfer,
    prompt_uses_compact_protocol,
    prompt_uses_persona_text_protocol,
)
from moonlightbox.training.models import ModelVersion
from moonlightbox.training.style_profile import style_violations


class MlxTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


class MlxRuntime(Protocol):
    def load(
        self,
        model_path: str,
        *,
        adapter_path: str,
    ) -> tuple[object, MlxTokenizer]: ...

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


class MlxSampleUtils(Protocol):
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


class RawMlxRuntime(Protocol):
    def generate_raw(
        self,
        *,
        base_model: str,
        adapter_path: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        should_cancel: Callable[[], bool] | None = None,
        deadline: datetime | None = None,
    ) -> str: ...


class DatabaseReplyGenerator:
    """数据库绑定的回复业务层；可注入任意原始文本模型 runtime。"""

    def __init__(
        self,
        database: Database,
        runtime: RawMlxRuntime | None = None,
    ) -> None:
        self._database = database
        if runtime is None:
            # 业务层默认明确选择 Linux runtime；MLX runtime 仅能作为历史调用方
            # 显式注入的兼容实现，不能再因遗漏参数被选中。
            from moonlightbox.agent.linux_inference import SharedLinuxModelRuntime

            runtime = SharedLinuxModelRuntime()
        self._runtime = runtime

    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        should_cancel: Callable[[], bool] | None = None,
        deadline: datetime | None = None,
    ) -> GeneratedReplyTurn:
        with Session(self._database.engine) as session:
            version = session.get(ModelVersion, model_version_id)
            if version is None:
                raise GeneratorUnavailableError("模型版本不存在")
            base_model = version.base_model
            adapter_path = version.adapter_path
            style_transfer_enabled = bool(
                (version.training_config or {}).get(
                    "runtime_style_transfer_enabled",
                    False,
                )
            )
            kernel = session.scalar(
                select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
            )
            style_profile = (
                kernel.content.get("style_profile")
                if kernel is not None and isinstance(kernel.content, dict)
                else None
            )

        contextual_messages = [
            {"role": "system", "content": system_prompt},
            *[dict(message) for message in messages],
        ]
        compact = prompt_uses_compact_protocol(system_prompt)
        persona_text = prompt_uses_persona_text_protocol(system_prompt)
        style_transfer_request = prompt_is_persona_style_transfer(system_prompt)
        allowed_sticker_ids = allowed_sticker_ids_from_prompt(system_prompt)
        last_style_violations: list[str] = []
        for attempt in range(3):
            raw_output = self._runtime.generate_raw(
                base_model=base_model,
                adapter_path=adapter_path,
                messages=contextual_messages,
                max_tokens=160,
                should_cancel=should_cancel,
                deadline=deadline,
            )
            try:
                reply = (
                    parse_persona_text_turn(raw_output)
                    if persona_text
                    else parse_reply_turn(
                        raw_output,
                        allowed_sticker_ids=allowed_sticker_ids,
                        normalize_compact=compact,
                    )
                )
                natural_text = "\n".join(
                    bubble.content or "" for bubble in reply.bubbles if bubble.type == "text"
                )
                last_style_violations = style_violations(
                    natural_text,
                    style_profile if isinstance(style_profile, dict) else None,
                )
                if not last_style_violations:
                    if (
                        persona_text
                        and style_transfer_enabled
                        and not style_transfer_request
                        and natural_text.strip()
                    ):
                        rewritten = self._rewrite_in_persona_style(
                            base_model=base_model,
                            adapter_path=adapter_path,
                            draft=natural_text,
                            style_profile=(
                                style_profile if isinstance(style_profile, dict) else None
                            ),
                            should_cancel=should_cancel,
                            deadline=deadline,
                        )
                        if rewritten is not None:
                            return rewritten
                    return reply
                contextual_messages[0] = {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n上一条回复使用了本人真实语料中未出现或极少出现的表达："
                        + "、".join(last_style_violations)
                        + "。重新回复，保持原意，但只用本人真实的短句、口头禅和语气。"
                    ),
                }
                continue
            except ReplyStructureError:
                if attempt < 2:
                    retry_instruction = (
                        "上一条包含机器格式。只输出本人会发送的自然聊天文字，多条消息换行。"
                        if persona_text
                        else compact_retry_instruction(allowed_sticker_ids or ())
                        if compact
                        else "上一个输出格式无效，只输出合法气泡 JSON。"
                    )
                    contextual_messages[0] = {
                        "role": "system",
                        "content": system_prompt + "\n" + retry_instruction,
                    }
                continue
        if last_style_violations:
            raise GenerationFailedError(
                "模型连续生成不符合本人语言画像的回复：" + "、".join(last_style_violations)
            )
        raise GenerationFailedError("模型未能生成有效回复结构")

    def rewrite_content_draft(
        self,
        model_version_id: str,
        draft: str,
    ) -> GeneratedReplyTurn | None:
        """Let LoRA choose phrasing while trusted state owns factual content."""

        return self.rewrite_content_draft_with_examples(
            model_version_id,
            draft,
            (),
        )

    def rewrite_content_draft_with_examples(
        self,
        model_version_id: str,
        draft: str,
        style_examples: tuple[str, ...],
    ) -> GeneratedReplyTurn | None:
        """Let LoRA use authentic examples without inheriting their facts."""

        with Session(self._database.engine) as session:
            version = session.get(ModelVersion, model_version_id)
            if version is None:
                raise GeneratorUnavailableError("模型版本不存在")
            kernel = session.scalar(
                select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
            )
            style_profile = (
                kernel.content.get("style_profile")
                if kernel is not None and isinstance(kernel.content, dict)
                else None
            )
            rewritten = self._rewrite_in_persona_style(
                base_model=version.base_model,
                adapter_path=version.adapter_path,
                draft=draft,
                style_examples=style_examples,
                style_profile=(style_profile if isinstance(style_profile, dict) else None),
                should_cancel=None,
                deadline=None,
            )
            if rewritten is not None:
                rewritten_text = "\n".join(
                    bubble.content or "" for bubble in rewritten.bubbles if bubble.type == "text"
                )
                fallback_lines = evidence_grounded_style_fallback_lines(
                    draft,
                    rewritten_text,
                    style_examples,
                )
                if fallback_lines != tuple(
                    bubble.content
                    for bubble in rewritten.bubbles
                    if bubble.type == "text" and bubble.content is not None
                ):
                    rewritten = GeneratedReplyTurn(
                        bubbles=tuple(
                            GeneratedBubble(type="text", content=line, delay_ms=0)
                            for line in fallback_lines
                        ),
                        raw_output="\n".join(fallback_lines),
                    )
                    rewritten_text = "\n".join(fallback_lines)
                if rewrite_preserves_hard_semantics(draft, rewritten_text):
                    return rewritten
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(type="text", content=draft, delay_ms=0),),
                raw_output=draft,
            )

    def _rewrite_in_persona_style(
        self,
        *,
        base_model: str,
        adapter_path: str,
        draft: str,
        style_examples: tuple[str, ...] = (),
        style_profile: dict[str, object] | None,
        should_cancel: Callable[[], bool] | None,
        deadline: datetime | None,
    ) -> GeneratedReplyTurn | None:
        messages = [
            {
                "role": "system",
                "content": persona_style_transfer_instruction(style_examples),
            },
            {"role": "user", "content": "内容草稿：\n" + draft},
        ]
        for _attempt in range(2):
            raw = self._runtime.generate_raw(
                base_model=base_model,
                adapter_path=adapter_path,
                messages=messages,
                max_tokens=160,
                should_cancel=should_cancel,
                deadline=deadline,
            )
            try:
                rewritten = parse_persona_text_turn(raw)
            except ReplyStructureError:
                messages[0]["content"] += "\n上一条格式无效，只输出自然聊天文字。"
                continue
            text = "\n".join(
                bubble.content or "" for bubble in rewritten.bubbles if bubble.type == "text"
            )
            violations = style_violations(text, style_profile)
            if not violations and rewrite_preserves_hard_semantics(draft, text):
                return rewritten
            messages[0]["content"] += (
                "\n上一条改变了原意或仍有机器表达。严格保留草稿中的否定、数字和专名，"
                "只调整这个人的措辞与分段。"
            )
        return None


# 历史导入兼容：生产代码使用 DatabaseReplyGenerator / DatabaseLinuxGenerator。
DatabaseMlxGenerator = DatabaseReplyGenerator


_NEGATION_MARKERS = ("不", "没", "无", "别", "未", "不能", "不会", "不是")

_RELATIONSHIP_STANCE_GROUPS = (
    ("保持距离", "距离", "少联系", "别联系", "不联系", "冷静一段", "缓一缓", "需要空间"),
    ("修复关系", "修复", "和好", "重新开始", "继续这段关系"),
)

_PERSON_REFERENTS = re.compile(r"我们|咱们|你们|您们|他们|她们|它们|我|咱|你|您|他|她|它")


def rewrite_preserves_hard_semantics(draft: str, rewritten: str) -> bool:
    """拒绝已知高风险的否定翻转、数字/人物漂移和内容坍缩。"""

    draft_numbers = re.findall(r"\d+(?:\.\d+)?", draft)
    rewritten_numbers = re.findall(r"\d+(?:\.\d+)?", rewritten)
    if draft_numbers != rewritten_numbers:
        return False
    # 风格改写可以改变措辞，但不能把“你做的事”改成“我做的事”。宁可回退到
    # 可信内容草稿，也不要让模型凭风格补写改变说话者、听者或第三方指代。
    if Counter(_PERSON_REFERENTS.findall(draft)) != Counter(_PERSON_REFERENTS.findall(rewritten)):
        return False
    for marker in _NEGATION_MARKERS:
        if draft.count(marker) != rewritten.count(marker):
            return False
    for stance_group in _RELATIONSHIP_STANCE_GROUPS:
        if any(marker in draft for marker in stance_group) and not any(
            marker in rewritten for marker in stance_group
        ):
            return False
    compact_draft = re.sub(r"\s|[，。！？!?；;、]", "", draft)
    compact_rewritten = re.sub(r"\s|[，。！？!?；;、]", "", rewritten)
    if not compact_draft or not compact_rewritten:
        return False
    ratio = len(compact_rewritten) / len(compact_draft)
    return 0.5 <= ratio <= 1.8
