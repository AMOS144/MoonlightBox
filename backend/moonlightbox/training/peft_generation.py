# ruff: noqa: E501

"""模型验收阶段的 Linux PEFT 回复生成器。"""

from __future__ import annotations

from moonlightbox.agent.linux_inference import PersonaSamplingConfig, SharedLinuxModelRuntime
from moonlightbox.branches.replies import (
    GeneratedReplyTurn,
    ReplyStructureError,
    parse_persona_text_turn,
    parse_reply_turn,
)
from moonlightbox.training.bubble_protocol import (
    allowed_sticker_ids_from_prompt,
    prompt_uses_compact_protocol,
    prompt_uses_persona_text_protocol,
)
from moonlightbox.training.model_acceptance import (
    ModelOutputStructureError,
    retry_format_instruction,
)
from moonlightbox.training.style_profile import style_violations


class PeftPathReplyGenerator:
    """验收与生产使用同一 Transformers + PEFT 解码路径。"""

    def __init__(
        self,
        sampling: PersonaSamplingConfig | None = None,
        *,
        device: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        self._runtime = SharedLinuxModelRuntime(
            device=device,
            load_in_4bit=load_in_4bit,
            sampling=sampling,
        )
        self._style_profile: dict[str, object] | None = None

    def set_style_profile(self, profile: dict[str, object] | None) -> None:
        self._style_profile = profile

    def set_seed(self, seed: int) -> None:
        self._runtime.set_seed(seed)

    def generate(
        self,
        base_model: str,
        adapter_path: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        reply_protocol = (
            "persona_text"
            if prompt_uses_persona_text_protocol(system_prompt)
            else "compact"
            if prompt_uses_compact_protocol(system_prompt)
            else "legacy_json"
        )
        context_messages = [{"role": "system", "content": system_prompt}, *messages]
        allowed_sticker_ids = allowed_sticker_ids_from_prompt(system_prompt)
        raw_attempts: list[str] = []
        failure_reasons: list[str] = []
        invalid_sticker_attempts: list[tuple[str, ...]] = []
        for attempt in range(3):
            raw = self._runtime.generate_raw(
                base_model=base_model,
                adapter_path=adapter_path,
                messages=context_messages,
                max_tokens=192,
            )
            raw_attempts.append(raw)
            try:
                reply = (
                    parse_persona_text_turn(raw)
                    if reply_protocol == "persona_text"
                    else parse_reply_turn(
                        raw,
                        allowed_sticker_ids=allowed_sticker_ids,
                        normalize_compact=reply_protocol == "compact",
                    )
                )
                natural_text = "\n".join(
                    bubble.content or "" for bubble in reply.bubbles if bubble.type == "text"
                )
                violations = style_violations(natural_text, self._style_profile)
                if not violations:
                    return reply
                failure_reasons.append(
                    "回复使用了本人真实语料中未出现的表达：" + "、".join(violations)
                )
                invalid_sticker_attempts.append(())
                if attempt < 2:
                    context_messages[0] = {
                        "role": "system",
                        "content": system_prompt
                        + "\n上一条回复使用了本人真实语料中未出现或极少出现的表达："
                        + "、".join(violations)
                        + "。重新回复，保持原意，只用本人真实短句和语气。",
                    }
            except ReplyStructureError as error:
                failure_reasons.append(str(error))
                invalid_sticker_attempts.append(error.attempted_invalid_sticker_ids)
                if attempt < 2:
                    context_messages[0] = {
                        "role": "system",
                        "content": system_prompt
                        + "\n"
                        + retry_format_instruction(reply_protocol, allowed_sticker_ids or ()),
                    }
        raise ModelOutputStructureError(
            tuple(raw_attempts),
            reasons=tuple(failure_reasons),
            attempted_invalid_sticker_ids=tuple(invalid_sticker_attempts),
        )

    def close(self) -> None:
        self._runtime.release()
