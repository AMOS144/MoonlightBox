"""心理模型值域只规范结构，不从聊天指标计算人格，也不要求测验或证据计数。"""

from functools import reduce
from operator import or_
from typing import Annotated, Literal

from pydantic import Field, create_model, field_validator, model_validator

from .context_modules import ContextModuleRef, ProfileModel

BIG_FIVE_DOMAINS = {
    "extraversion": ["sociability", "assertiveness", "energy_level"],
    "agreeableness": ["compassion", "respectfulness", "trust"],
    "conscientiousness": ["organization", "productiveness", "responsibility"],
    "negative_emotionality": ["anxiety", "depression", "emotional_volatility"],
    "open_mindedness": ["intellectual_curiosity", "aesthetic_sensitivity", "creative_imagination"],
}
MODEL_DIMENSIONS = {
    "big_five_bfi2": {
        key for domain, facets in BIG_FIVE_DOMAINS.items() for key in [domain, *facets]
    },
    "mbti": {"EI", "SN", "TF", "JP"},
    "interpersonal_circumplex": {"agency", "communion"},
    "sdt": {"autonomy", "competence", "relatedness"},
}


class ModelDimensionEntry(ProfileModel):
    id: str | None = Field(default=None, description="新理解条目填 null，已有条目沿用返回的 ID")
    model: Literal["big_five_bfi2", "mbti", "interpersonal_circumplex", "sdt"] = Field(
        description="所属心理模型；必须与该栏目要求的模型一致"
    )
    dimension_id: str = Field(
        description="模型规定的维度键，每个固定维度恰好填写一次；不明确也保留 unknown 项"
    )
    status: Literal["described", "unknown"] = Field(
        default="unknown", description="已形成可说明的理解或仍未知；推断不要求心理测验"
    )
    basis: Literal["inferred", "self_reported", "user_corrected"] = Field(
        default="inferred", description="综合推断、自我报告或用户确认纠正；不能把推断标成自述"
    )
    value: str = Field(
        default="unknown",
        description="使用当前模型允许的枚举；大五/IPC 为 low/moderate/high/mixed/unknown；SDT 为 supported/frustrated/mixed/unknown；MBTI 为该轴字母/mixed/unknown",
    )
    context: str = Field(
        default="", description="这个倾向适用的情境、关系或时期，避免把工作要求直接当成人格"
    )
    description: str = Field(default="", description="用中文说明对人物的具体理解，不堆抽象模型术语")
    reasoning_summary: str = Field(
        default="", description="简要说明哪些整体材料与情境支持该判断，保留其他可能解释"
    )
    roleplay_guidance: str = Field(
        default="", description="后续扮演时如何体现这种倾向，不是固定台词或必须每轮执行的动作"
    )
    uncertainties: list[str] = Field(
        default_factory=list, description="仍不确定、可能相反或仅限特定情境的部分；无则 []"
    )
    reference_message_ids: list[str] = Field(
        default_factory=list,
        description="可回查的真实消息 UUID，无则 []；不要求用数量证明人格，不编造证据",
    )
    context_module_refs: list[ContextModuleRef] = Field(
        default_factory=list, description="实际读取并用于理解的现实情境模块与版本；未使用则 []"
    )

    @model_validator(mode="after")
    def check_model_value(self):
        if self.dimension_id not in MODEL_DIMENSIONS[self.model]:
            raise ValueError("心理维度不属于指定模型")
        choices = {"low", "moderate", "high", "mixed", "unknown"}
        if self.model == "sdt":
            choices = {"supported", "frustrated", "mixed", "unknown"}
        elif self.model == "mbti":
            choices = {*self.dimension_id, "mixed", "unknown"}
        if self.value not in choices:
            raise ValueError("心理维度的值超出模型值域")
        return self


class ModelProfile(ProfileModel):
    summary: str = Field(default="", description="用中文综合说明此心理模型下的人物理解及适用范围")
    dimensions: list[ModelDimensionEntry] = Field(
        default_factory=list,
        description="完整填写模型固定维度，每项恰好一次，无法判断的维度填 unknown",
    )


def _typed_dimension(model, dimensions, values, name):
    # 值域必须进入发给模型的 JSON Schema，而不能只藏在 Python validator 中。
    return create_model(
        name,
        __base__=ModelDimensionEntry,
        model=(Literal[model], Field(default=model, description="此维度所属的固定心理模型")),
        dimension_id=(
            Literal[tuple(sorted(dimensions))],
            Field(description="该模型允许的固定维度键；每项恰好一次"),
        ),
        value=(
            Literal[tuple(values)],
            Field(
                default="unknown",
                description="选择该维度的倾向；不明用 unknown，混合用 mixed，不从消息数量计算分数",
            ),
        ),
    )


BigFiveDimension = _typed_dimension(
    "big_five_bfi2",
    MODEL_DIMENSIONS["big_five_bfi2"],
    ("low", "moderate", "high", "mixed", "unknown"),
    "BigFiveDimension",
)
SDTDimension = _typed_dimension(
    "sdt", MODEL_DIMENSIONS["sdt"], ("supported", "frustrated", "mixed", "unknown"), "SDTDimension"
)
IPCDimension = _typed_dimension(
    "interpersonal_circumplex",
    MODEL_DIMENSIONS["interpersonal_circumplex"],
    ("low", "moderate", "high", "mixed", "unknown"),
    "IPCDimension",
)
MBTIDimension = Annotated[
    reduce(
        or_,
        (
            _typed_dimension("mbti", (axis,), (*axis, "mixed", "unknown"), f"MBTI{axis}Dimension")
            for axis in sorted(MODEL_DIMENSIONS["mbti"])
        ),
    ),
    Field(discriminator="dimension_id"),
]


class BigFiveModelProfile(ModelProfile):
    dimensions: list[BigFiveDimension] = Field(
        default_factory=list,
        min_length=20,
        max_length=20,
        description="共二十项：五个领域本身各一项，加十五个侧面。不能只交十五侧面或只交五领域。",
    )

    @field_validator("dimensions", mode="before")
    @classmethod
    def explain_missing_dimensions(cls, value):
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            ids = {item.get("dimension_id") for item in value}
            missing = sorted(MODEL_DIMENSIONS["big_five_bfi2"] - ids)
            if missing:
                # 只检查固定表单是否完整；不限制每项应推断什么值或需要多少证据。
                raise ValueError(f"大五固定表单缺少这些维度：{missing}；领域本身也需独立填写")
        return value


class SDTModelProfile(ModelProfile):
    dimensions: list[SDTDimension] = Field(
        default_factory=list,
        min_length=3,
        max_length=3,
        description="autonomy、competence、relatedness 各一项，未知也保留对应维度",
    )


class IPCModelProfile(ModelProfile):
    dimensions: list[IPCDimension] = Field(
        default_factory=list,
        min_length=2,
        max_length=2,
        description="agency 与 communion 各一项，结合具体关系范围理解",
    )


class MBTIModelProfile(ModelProfile):
    dimensions: list[MBTIDimension] = Field(
        default_factory=list,
        min_length=4,
        max_length=4,
        description="EI、SN、TF、JP 四轴各一项，value 为该轴字母、mixed 或 unknown",
    )
    candidates: list[str] = Field(
        default_factory=list,
        description="可选四字母类型，每位分别为 E/I/X、S/N/X、T/F/X、J/P/X；X 表示未定，无法概括可 []",
    )

    @model_validator(mode="after")
    def check_candidates(self):
        for candidate in self.candidates:
            if len(candidate) != 4 or any(
                letter.upper() not in choices
                for letter, choices in zip(candidate, ("EIX", "SNX", "TFX", "JPX"), strict=True)
            ):
                raise ValueError("MBTI 候选应为四个字母，可用 x 表示未定轴")
        return self


class RelationStyle(ProfileModel):
    id: str | None = Field(default=None, description="已有关系方式条目 ID；新条目 null")
    relationship_scope: str = Field(
        description="具体针对哪类人或哪段关系，不能把用户关系与其他社会关系混为一谈"
    )
    profile: IPCModelProfile = Field(
        default_factory=IPCModelProfile, description="该关系范围内的主导性与亲和性，两项完整填写"
    )
