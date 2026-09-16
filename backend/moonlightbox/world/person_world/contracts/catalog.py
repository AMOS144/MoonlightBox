"""栏目职责的确定性目录；不在这里对聊天文本作语义判断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PrimaryDomain = Literal[
    "identity",
    "life_context",
    "social_world",
    "agency",
    "practices",
    "life_course",
    "relationship_with_user",
]


@dataclass(frozen=True, slots=True)
class SectionContract:
    """一个顶层栏目唯一的持久化边界和最小可用工具集。"""

    domain: PrimaryDomain
    purpose: str
    allowed_fact_types: tuple[str, ...]
    temporal_policy: str
    sensitivity_policy: str
    ownership_rules: tuple[str, ...]


SECTION_CONTRACTS: tuple[SectionContract, ...] = (
    SectionContract(
        domain="identity",
        purpose="记录可消歧的身份、直接自我描述与本人赋义的自我叙事。",
        allowed_fact_types=("identifier", "self_description", "self_narrative"),
        temporal_policy="区分过去、当前与意向身份；不把意向当既成身份。",
        sensitivity_policy="禁止从语气或行为推断人格、诊断和心理标签。",
        ownership_rules=(
            "工作、学校、地点和现实角色正文归 life_context。",
            "事件本身归 life_course；本栏仅保留本人明确的身份意义。",
        ),
    ),
    SectionContract(
        domain="life_context",
        purpose="记录工作学习、家庭照料、地点环境和明确的现实行动条件。",
        allowed_fact_types=(
            "work_learning",
            "home_care",
            "place_environment",
            "functional_context",
        ),
        temporal_policy="当前、过去、计划和假设必须明确区分。",
        sensitivity_policy="默认不抽取诊断、治疗、性健康、政治或宗教等敏感信息。",
        ownership_rules=(
            "具体一日作息与重复节律归 practices。",
            "入职、搬家、离校等变化的历史维度归 life_course。",
        ),
    ),
    SectionContract(
        domain="social_world",
        purpose="记录目标人物与当前用户之外主体的有方向社会关系。",
        allowed_fact_types=("social_tie",),
        temporal_policy="关系状态须带有效时间或保留未知，不能以当前状态覆盖历史。",
        sensitivity_policy="不得从同场、昵称或频率推断亲密、依恋或人格。",
        ownership_rules=(
            "目标人物与当前用户的二元关系只归 relationship_with_user。",
            "工作职位、组织位置和住处正文归 life_context。",
        ),
    ),
    SectionContract(
        domain="agency",
        purpose="记录明确表达的偏好、价值解释、目标和承诺。",
        allowed_fact_types=(
            "preference",
            "value_interpretation",
            "goal",
            "commitment",
        ),
        temporal_policy="目标与承诺必须带未来/完成/取消等状态和可知时间边界。",
        sensitivity_policy="不得从行为或表达风格反推动机、价值排序或政治宗教立场。",
        ownership_rules=(
            "一次行为和现实岗位归其发生的 life_context 或 life_course。",
            "重复做法归 practices，不能因反复出现就自动成为偏好。",
        ),
    ),
    SectionContract(
        domain="practices",
        purpose="记录在明确条件下重复发生的活动与时间节律。",
        allowed_fact_types=("recurring_activity", "temporal_rhythm"),
        temporal_policy="规律须是明确自述或经多日期观察支持，并保留条件和例外。",
        sensitivity_policy="不以消息发送时间、单次事件或他人作息生成规律。",
        ownership_rules=(
            "工作和地点等背景归 life_context。",
            "一次活动与计划分别归 life_course 和 agency。",
        ),
    ),
    SectionContract(
        domain="life_course",
        purpose="记录有时间边界的经历、可核对转变和跨期轨迹。",
        allowed_fact_types=("episode", "transition", "trajectory"),
        temporal_policy="转变必须有 before/after；轨迹须有多个时间点或明确持续叙述。",
        sensitivity_policy="不得戏剧化敏感经历或替目标人物赋予创伤、成长等意义。",
        ownership_rules=(
            "当前生活位置归 life_context。",
            "未来目标归 agency，重复活动归 practices。",
        ),
    ),
    SectionContract(
        domain="relationship_with_user",
        purpose="记录目标人物与当前用户之间方向明确的关系状态、互动模式与变化。",
        allowed_fact_types=(
            "relationship_standing",
            "interaction_observation",
            "interaction_pattern",
            "relationship_history",
        ),
        temporal_policy="每项需具有 target_to_user、user_to_target 或 mutual 方向和时间边界。",
        sensitivity_policy="不得从单次亲近、聊天频率或沉默推断感情、依恋和满意度。",
        ownership_rules=(
            "用户自身的工作、作息、健康、偏好和经历绝不写入目标人物 Profile。",
            "与其他人的关系归 social_world。",
        ),
    ),
)

_BY_DOMAIN = {contract.domain: contract for contract in SECTION_CONTRACTS}


def get_section_contract(domain: str) -> SectionContract:
    try:
        return _BY_DOMAIN[domain]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"未知 PersonWorld 顶层栏目: {domain}") from error
