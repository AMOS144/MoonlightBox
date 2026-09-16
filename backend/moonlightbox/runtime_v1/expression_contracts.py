"""表达任务与图文交付；不包含文件路径、发送权限或独立心理状态。"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .director_contracts import Contract


class TextPart(Contract):
    kind: Literal["text"] = Field(description="文字消息类型，固定 text")
    text: str = Field(
        min_length=1,
        max_length=4000,
        description="实际发送的一条聊天气泡；不得全空白，不含分析、舞台说明或工具调用",
    )

    @model_validator(mode="after")
    def nonblank(self):
        if not self.text.strip():
            raise ValueError("文字不能为空白")
        return self


class StickerPart(Contract):
    kind: Literal["sticker"] = Field(description="表情消息类型，固定 sticker；不包含文字字段")
    asset_ref: str = Field(
        min_length=1,
        description="当前素材候选返回且 available=true 的 asset_ref；不能编造 ID、URL 或文件路径",
    )


class ExpressionClarification(Contract):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="确实阻碍表达的具体澄清问题；这是内部协作，不直接发给用户",
    )
    related_refs: list[str] = Field(
        default_factory=list, description="问题涉及的已有消息或上下文引用；无则 []"
    )


class ExpressionResult(Contract):
    status: Literal["ready", "needs_clarification"] = Field(
        description="ready 时 messages 非空、clarification=null；needs_clarification 时 messages=[] 且 clarification 非空。Director 的 speak 只能提交 ready。"
    )
    messages: list[Annotated[TextPart | StickerPart, Field(discriminator="kind")]] = Field(
        default_factory=list,
        max_length=16,
        description="按实际发送顺序排列的文字/表情气泡；不是候选台词列表；ready 时至少一条",
    )
    clarification: ExpressionClarification | None = Field(
        default=None, description="仅 needs_clarification 填写；ready 必须 null"
    )

    @model_validator(mode="after")
    def mutually_exclusive(self):
        if self.status == "ready":
            if not self.messages or self.clarification is not None:
                raise ValueError("ready 必须只有非空 messages")
        elif self.messages or self.clarification is None:
            raise ValueError("needs_clarification 必须只有 clarification")
        return self
