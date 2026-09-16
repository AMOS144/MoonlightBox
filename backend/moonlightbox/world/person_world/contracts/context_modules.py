"""情境实例与按 kind 判别的 details。未知字段不阻止其余生活内容生成。"""

from datetime import datetime
from functools import reduce
from operator import or_
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .context_module_details import MODULE_FIELDS, MODULE_FIELD_MEANINGS, MODULE_LABELS


class ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModuleField(ProfileModel):
    status: Literal["described", "unknown", "not_applicable"] = Field(
        default="unknown", description="已有理解、尚未知或不适用；不适用不等于没有资料"
    )
    basis: Literal["stated", "inferred", "user_corrected"] = Field(
        default="inferred", description="明确表达、综合推断或用户已确认纠正；按实际来源选择"
    )
    value: str | None = Field(
        default=None, description="此情境字段的简明值；未知或不适用可 null，不编造组织、收入或时间"
    )
    description: str = Field(
        default="",
        description="用中文解释具体生活情境、限制或感受；约20–80字为软目标，不是长度校验",
    )
    reference_message_ids: list[str] = Field(
        default_factory=list, description="可回查的原始消息 UUID；无可定位引用填 []，不能猜测"
    )


class ContextModuleRef(ProfileModel):
    module_id: str = Field(description="本次模块工具返回的实例 ID；不是 kind 或栏目名")
    revision: int = Field(ge=1, description="实际读取的模块 revision，原样使用工具返回值")
    field_paths: list[str] = Field(
        default_factory=list,
        description="实际使用的字段路径；业务字段从 details. 开始，使用 get_context_module_spec 的 read_paths",
    )
    usage: Literal["summary", "interpretation_context"] = Field(
        default="interpretation_context",
        description="summary 复述模块内容；interpretation_context 用作综合理解的情境，不直接推导人格",
    )


class ModulePeriod(ProfileModel):
    start_at: datetime | None = Field(
        default=None, description="有资料支持的 ISO 开始时间；不明填 null，不强制精确化模糊时间"
    )
    end_at: datetime | None = Field(
        default=None,
        description="有资料支持的 ISO 结束时间，不得早于 start_at；持续中或不明填 null",
    )
    description: str = Field(default="", description="时期的自然语言说明，可保留大致时间和不确定性")

    @model_validator(mode="after")
    def ordered(self):
        if self.start_at and self.end_at and self.start_at > self.end_at:
            raise ValueError("情境结束时间不能早于开始时间")
        return self


class ContextModuleBase(ProfileModel):
    # 模型仅能沿用工具返回的实例 ID；新实例提交 null，由装配器生成。
    id: str | None = Field(
        default=None, description="新实例填 null；更新时沿用工具返回的实例 ID，不自行生成"
    )
    revision: int = Field(
        default=1, ge=1, description="新实例从1开始；既有实例沿用已读取版本，由后端管理提交版本"
    )
    title: str = Field(description="此段具体生活情境的简短名称，同一 kind 可有多个不同实例")
    status: Literal["current", "planned", "past", "paused", "unknown"] = Field(
        default="unknown",
        description="当前进行、未来计划、已结束、暂停或不明；与 period 的材料一致",
    )
    basis: Literal["stated", "inferred", "user_corrected"] = Field(
        default="inferred", description="模块适用性的来源；可依据整体材料推断，不需额外激活证据"
    )
    period: ModulePeriod = Field(
        default_factory=ModulePeriod, description="此情境适用的时间范围，允许未知"
    )
    summary: str = Field(description="人物正在面对怎样的现实生活情境，不是通用职业介绍")
    related_module_ids: list[str] = Field(
        default_factory=list, description="有关联的已有情境实例 ID；无或尚未分配 ID 时 []"
    )


# create_model 只消除固定字段声明的机械重复；没有开放 dict，更没有正则分析正文。
MODULE_MODELS = {}
for _kind, _groups in MODULE_FIELDS.items():
    _group_models = {
        group: create_model(
            f"{_kind}_{group}",
            __base__=ProfileModel,
            **{
                name: (
                    ModuleField,
                    Field(default_factory=ModuleField, description=MODULE_FIELD_MEANINGS[name]),
                )
                for name in names.split()
            },
        )
        for group, names in _groups.items()
    }
    _details = create_model(
        f"{_kind}_details",
        __base__=ProfileModel,
        **{
            group: (
                model,
                Field(
                    default_factory=model,
                    description=f"{MODULE_LABELS[_kind]}情境的 {group} 分组，逐字段填写；未知保留 unknown",
                ),
            )
            for group, model in _group_models.items()
        },
    )
    MODULE_MODELS[_kind] = create_model(
        f"{_kind}_module",
        __base__=ContextModuleBase,
        kind=(
            Literal[_kind],
            Field(
                default=_kind,
                description=f"固定模块类型：{MODULE_LABELS[_kind]}；可与其他生活模块并存",
            ),
        ),
        schema_version=(
            Literal[f"{_kind}_v1"],
            Field(default=f"{_kind}_v1", description="该模块固定结构版本，不自行更改"),
        ),
        details=(
            _details,
            Field(default_factory=_details, description="此情境的完整细分字段；不要只交概括性摘要"),
        ),
    )

ContextModule = Annotated[reduce(or_, MODULE_MODELS.values()), Field(discriminator="kind")]
