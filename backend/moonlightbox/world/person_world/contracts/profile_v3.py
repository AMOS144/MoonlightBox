"""七栏完整人物画像。旧 v2 保留历史读取，新输出不再经过原子事实证据门槛。"""

from typing import Literal

from pydantic import Field, create_model, model_validator

from .context_modules import ContextModule, ContextModuleRef, ProfileModel
from .psychological_models import (
    MODEL_DIMENSIONS,
    BigFiveModelProfile,
    IPCModelProfile,
    MBTIModelProfile,
    ModelProfile,
    RelationStyle,
    SDTModelProfile,
)
from .world_dimensions import SECTION_DIMENSIONS


class DimensionEntry(ProfileModel):
    id: str | None = Field(default=None, description="沿用已有条目 ID；新条目填 null，由后端分配")
    dimension_id: str = Field(description="固定维度键，必须与所属字段相同，不自行增加维度")
    status: Literal["described", "unknown", "not_applicable"] = Field(
        default="unknown",
        description="described 已形成理解；unknown 仍不清楚；not_applicable 不适用于此人，说明理由",
    )
    basis: Literal["stated", "inferred", "user_corrected"] = Field(
        default="inferred",
        description="stated 明确表达；inferred 综合材料推断；user_corrected 用户已确认的纠正，不把模型猜测写成纠正",
    )
    value: str | None = Field(default=None, description="该维度的简明结论；不清楚填 null")
    description: str = Field(
        default="", description="用中文说明具体理解及适用情境，避免堆砌逐条聊天摘要"
    )
    reference_message_ids: list[str] = Field(
        default_factory=list,
        description="可定位的真实消息 UUID；辅助回查而非数量门槛，没有可定位引用填 []，不编造",
    )
    context_module_refs: list[ContextModuleRef] = Field(
        default_factory=list, description="实际读取并用于理解的情境模块引用；未使用模块填 []"
    )


class ModuleAssessment(ProfileModel):
    """仅用于调查审计；不进入七栏事实树。由 Agent 明确决定是否识别了情境。"""

    status: Literal["identified", "not_identified"] = Field(
        description="是否识别到适用的生活情境；这是审计状态，不是人物事实"
    )
    explanation: str = Field(
        min_length=1, description="简要解释模块识别结果；not_identified 时说明不确定处"
    )


EXPRESSION_FIELDS = {
    "forms_of_address": "常用称呼",
    "particles_and_catchphrases": "语气词与口癖",
    "message_rhythm": "分句与消息节奏",
    "punctuation": "标点与文字形式",
    "humor": "玩笑方式",
    "situational_variation": "不同场景下的表达变化",
}


class ExpressionPattern(ProfileModel):
    """一种表达方式和它的用法；不把具体词形从适用场景中拆开。"""

    form: str = Field(min_length=1, description="实际词形或具体表达方式，不填泛化性格标签")
    use_when: str = Field(description="在什么互动、对象和情绪下适合，怎样使用")
    avoid_when: str = Field(description="不适用或会显得刻意的场景；不清楚可留空")
    reference_message_ids: list[str] = Field(
        default_factory=list, description="支持该表达用法的真实消息 UUID；不编造，无法定位填 []"
    )


class ExpressionDimension(DimensionEntry):
    patterns: list[ExpressionPattern] = Field(
        default_factory=list,
        description="该表达维度下的具体词形或习惯及使用条件，不把人物标签当口癖",
    )


ExpressionProfile = create_model(
    "ExpressionProfile",
    __base__=ProfileModel,
    schema_version=(
        Literal["expression-v1"],
        Field(default="expression-v1", description="表达资料固定版本，不自行改写"),
    ),
    # 历史画像缺字段时保持 unknown，不冒充已经完成编译。
    **{
        name: (
            create_model(
                f"expression_{name}",
                __base__=ExpressionDimension,
                dimension_id=(
                    Literal[name],
                    Field(default=name, description=f"固定表达维度键：{EXPRESSION_FIELDS[name]}"),
                ),
            ),
            Field(description=EXPRESSION_FIELDS[name]),
        )
        for name in EXPRESSION_FIELDS
    },
)


def empty_expression_profile():
    return ExpressionProfile.model_validate({name: {} for name in EXPRESSION_FIELDS})


def _model_profile(model, mbti=False):
    def build():
        cls = {
            "big_five_bfi2": BigFiveModelProfile,
            "sdt": SDTModelProfile,
            "interpersonal_circumplex": IPCModelProfile,
            "mbti": MBTIModelProfile,
        }[model]
        return cls.model_validate(
            dict(
                dimensions=[
                    {"model": model, "dimension_id": key} for key in sorted(MODEL_DIMENSIONS[model])
                ]
            )
        )

    return build


class SectionBase(ProfileModel):
    summary: str = Field(default="", description="本栏目综合理解，用清晰中文概括，不串列消息原话")

    @model_validator(mode="after")
    def check_field_identity(self):
        for name in type(self).model_fields:
            entry = getattr(self, name)
            if isinstance(entry, DimensionEntry) and entry.dimension_id != name:
                raise ValueError(f"{name} 的 dimension_id 与字段不一致")
            model_name = {
                "big_five_bfi2": "big_five_bfi2",
                "mbti": "mbti",
                "motivation": "sdt",
                "interpersonal_style": "interpersonal_circumplex",
            }.get(name)
            profiles = (
                [entry]
                if isinstance(entry, ModelProfile)
                else [style.profile for style in entry]
                if name == "interpersonal_styles"
                else []
            )
            for profile in profiles:
                expected = model_name or "interpersonal_circumplex"
                keys = [dim.dimension_id for dim in profile.dimensions]
                if (
                    len(keys) != len(set(keys))
                    or set(keys) != MODEL_DIMENSIONS[expected]
                    or any(dim.model != expected for dim in profile.dimensions)
                ):
                    missing = sorted(MODEL_DIMENSIONS[expected] - set(keys))
                    extra = sorted(set(keys) - MODEL_DIMENSIONS[expected])
                    raise ValueError(
                        f"{name} 的固定维度缺失={missing}，多余={extra}；"
                        "每项恰好一次，不明填 unknown"
                    )
        return self


SECTION_MODELS = {}
SECTION_RESULT_MODELS = {}
for _section, _fields in SECTION_DIMENSIONS.items():
    # 固定字段的 dimension_id 是结构身份，不需要模型重复猜测或抄写。
    # 保留在持久化/API 中方便定位，漏写时由字段对应的常量默认值补足。
    _field_types = {
        name: create_model(
            f"{_section}_{name}_entry",
            __base__=DimensionEntry,
            dimension_id=(
                Literal[name],
                Field(default=name, description=f"固定维度键：{_fields[name]}"),
            ),
        )
        for name in _fields
    }
    _extra = {}
    if _section == "identity":
        _extra = {
            "big_five_bfi2": (
                BigFiveModelProfile,
                Field(
                    default_factory=_model_profile("big_five_bfi2"),
                    description="大五 BFI-2 的五领域与十五侧面，共20项，用于具体人物理解，不是测验分数",
                ),
            ),
            "mbti": (
                MBTIModelProfile,
                Field(
                    default_factory=_model_profile("mbti", True),
                    description="MBTI 四轴倾向与可选类型，允许未知或混合，不强行定型",
                ),
            ),
        }
    elif _section == "life_context":
        _extra = {
            "context_modules": (
                list[ContextModule],
                Field(
                    default_factory=list,
                    description="适用于此人的现实情境实例，可并存；不适用模块无需创建",
                ),
            )
        }
    elif _section == "social_world":
        _extra = {
            "interpersonal_styles": (
                list[RelationStyle],
                Field(
                    default_factory=list,
                    description="不同非用户关系中的 IPC 相处方式，按关系范围分别描述",
                ),
            )
        }
    elif _section == "agency":
        _extra = {
            "motivation": (
                SDTModelProfile,
                Field(
                    default_factory=_model_profile("sdt"),
                    description="SDT 自主、胜任、联结三种需要在当前生活中怎样被支持或受挫",
                ),
            )
        }
    elif _section == "relationship_with_user":
        _extra = {
            "expression_profile": (
                ExpressionProfile,
                Field(
                    default_factory=empty_expression_profile,
                    description="六类具体表达习惯及适用场景，后续用于说话技能，不是人格标签",
                ),
            ),
            "interpersonal_style": (
                IPCModelProfile,
                Field(
                    default_factory=_model_profile("interpersonal_circumplex"),
                    description="与用户互动中的 IPC 主导性和亲和性，仅描述这段二元关系",
                ),
            ),
        }
    SECTION_MODELS[_section] = create_model(
        f"{_section}_profile_v3",
        __base__=SectionBase,
        **{
            name: (entry, Field(default_factory=entry, description=_fields[name]))
            for name, entry in _field_types.items()
        },
        **_extra,
    )
    SECTION_RESULT_MODELS[_section] = create_model(
        f"{_section}_result_v3",
        __base__=SECTION_MODELS[_section],
        # 存储模型可用 unknown 初始化空表单；Agent 交付模型不能靠默认值掩盖漏填整项。
        **{
            name: (field.annotation, Field(description=field.description))
            for name, field in SECTION_MODELS[_section].model_fields.items()
            if name != "context_modules"
        },
        section=(
            Literal[_section],
            Field(default=_section, description="本任务固定栏目键，不跨栏提交"),
        ),
        overview=(str, Field(default="", description="栏目整体理解的简明中文概括，与细分字段一致")),
        unresolved_questions=(
            list[str],
            Field(
                default_factory=list,
                description="仍影响理解的具体问题，不要求为每个字段制造问题；无则 []",
            ),
        ),
        **(
            {
                "context_modules": (
                    list[ContextModule],
                    Field(description="本次调查识别出的适用情境模块；没有则 [] 并解释识别状态"),
                ),
                "module_assessment": (
                    ModuleAssessment,
                    Field(description="模块识别审计，不属于人物事实，不用空数组掩盖尚未完成调查"),
                ),
            }
            if _section == "life_context"
            else {}
        ),
    )

PersonWorldProfileV3 = create_model(
    "PersonWorldProfileV3",
    __base__=ProfileModel,
    schema_version=(Literal["v3"], "v3"),
    subject_participant_id=(str, ...),
    overview=(str, ""),
    **{name: (model, Field(default_factory=model)) for name, model in SECTION_MODELS.items()},
)


def described_count(value: object) -> int:
    """仅计数呈现条目，不把资料数量换算成人格置信度。"""
    if isinstance(value, dict):
        return int(value.get("status") == "described") + sum(
            described_count(item) for item in value.values()
        )
    if isinstance(value, list):
        return sum(described_count(item) for item in value)
    return 0
