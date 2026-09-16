"""编辑目录的中文标签，不包含任何生成默认画像或行为推断。"""

from .context_module_details import MODULE_FIELDS, MODULE_LABELS
from .profile_v3 import EXPRESSION_FIELDS
from .psychological_models import BIG_FIVE_DOMAINS
from .world_dimensions import SECTION_DIMENSIONS, SECTION_LABELS

LABELS = dict(
    pair.split(":", 1)
    for pair in """
organization:组织
name:名称
sector:组织性质
industry:行业
role:工作角色
title:名称与岗位
work_content:工作内容
responsibilities:职责
working_arrangement:工作安排
employment_form:工作形式
workplace:工作地点
schedule:时间安排
flexibility:灵活性
experience:主观体验
workload:工作负荷
autonomy:自主性
satisfaction_and_frustrations:满意与困扰
outlook:后续方向
current_priorities:当前重点
intended_changes:希望改变的事
institution:学习机构
education_form:学习形式
study:学习任务
stage:阶段
field:领域
courses_and_tasks:课程与任务
exams_and_deadlines:考试与期限
arrangement:安排
location:地点
learning_style:学习方式
interests:兴趣
pressure_and_difficulties:压力与困难
peer_environment:同伴环境
next_steps:下一步
current_state:当前状态
employment_background:工作背景
search_stage:求职阶段
direction:方向
target_roles:目标岗位
sector_preferences:行业与组织偏好
location_preferences:地点偏好
progress:进展
applications:投递
interviews:面试
offers_and_decisions:机会与决定
conditions:现实条件
constraints:限制
concerns:顾虑
next_actions:下一步行动
alternatives:其他选择
venture:创业事项
business_or_product:业务或产品
participation:参与方式
own_role:自身角色
team_and_partners:团队与伙伴
operation:日常运作
daily_work:日常工作
time_commitment:时间投入
resources:资源
difficulties:困难
near_term_goals:近期目标
recipient:照料对象
relationship:关系
care_needs:照料需要
tasks:具体任务
coordination:协作安排
division_of_work:分工
support_network:支持网络
life_impact:生活影响
feelings:感受
expected_changes:预期变化
transition:转变
previous_engagement:过去投入
retirement_arrangement:退休安排
daily_life:日常生活
main_activities:主要活动
time_structure:时间结构
social_connections:社会连接
resources_and_constraints:资源与限制
plans_and_interests:计划与兴趣
background:背景
reason_or_context:原因与情境
voluntary_or_constrained:主动或受限
activities:活动
recovery_and_feelings:休整体验
return_or_next_steps:回归或下一步
move:迁移
from_place:迁出地点
to_place:迁入地点
housing:住房
logistics:事务安排
timing:时间
completed_and_pending:已完成与待完成
adaptation:适应
change:变化
before:此前
after:之后
participants:参与者
responsibility_changes:责任变化
unresolved_matters:待解决事项
topic:主题
scope:范围
situation:处境
current_activity:当前活动
people_and_environment:相关人与环境
big_five_bfi2:大五人格（BFI-2）
mbti:MBTI 倾向
motivation:动机体验（SDT）
interpersonal_style:相处方式（IPC）
interpersonal_styles:不同关系中的相处方式
extraversion:外向性
sociability:社交倾向
assertiveness:表达主张
energy_level:活力
agreeableness:宜人性
compassion:同情关怀
respectfulness:尊重
trust:信任
conscientiousness:尽责性
organization:组织性
productiveness:事务推进
responsibility:责任感
negative_emotionality:负性情绪倾向
anxiety:担忧倾向
depression:低落倾向（非诊断）
emotional_volatility:情绪波动
open_mindedness:开放性
intellectual_curiosity:求知好奇
aesthetic_sensitivity:审美敏感
creative_imagination:创造想象
EI:精力取向
SN:信息偏好
TF:判断偏好
JP:组织偏好
agency:主张性
communion:亲近合作
competence:胜任感
relatedness:连接与归属
summary:综述
context_modules:生活情境
overview:整体画像
period:时期
""".strip().splitlines()
)


def profile_editor_catalog():
    return {
        "sections": [
            {"key": key, "label": SECTION_LABELS[key], "fields": fields}
            for key, fields in SECTION_DIMENSIONS.items()
        ],
        "modules": {
            kind: {
                "label": MODULE_LABELS[kind],
                "groups": {group: names.split() for group, names in groups.items()},
                "group_labels": {
                    group: "组织信息" if group == "organization" else LABELS.get(group, group)
                    for group in groups
                },
            }
            for kind, groups in MODULE_FIELDS.items()
        },
        "labels": LABELS,
        "expression_fields": EXPRESSION_FIELDS,
        "model_groups": {"big_five_bfi2": BIG_FIVE_DOMAINS},
    }
