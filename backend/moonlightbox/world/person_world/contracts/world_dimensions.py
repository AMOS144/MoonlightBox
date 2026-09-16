"""v3 栏目的唯一字段目录。只描述编辑位置，不用规则推断人的性格。"""

SECTION_DIMENSIONS = {
    "identity": {
        "names_and_self_reference": "姓名与自称",
        "claimed_roles": "自我角色",
        "self_evaluations": "自我评价",
        "identity_tensions": "身份张力",
    },
    "life_context": {
        "primary_engagements": "主要生活投入",
        "institutional_attachment": "组织环境",
        "living_and_care": "居住与照料",
        "places_and_mobility": "地点与流动",
        "resources_and_constraints": "资源与限制",
        "active_projects": "正在推进的事务",
    },
    "social_world": {
        "household_and_family": "家庭关系",
        "peer_and_collaborative_world": "同伴与协作",
        "care_and_support": "照料与支持",
        "responsibility_and_power": "责任与权力",
        "group_and_institutional_ties": "群体与组织关系",
    },
    "agency": {
        "objects_of_concern": "在意的事",
        "decision_criteria": "取舍方式",
        "goals_and_commitments": "目标与承诺",
        "avoidance_and_boundaries": "回避与边界",
    },
    "practices": {
        "ongoing_practices": "日常实践",
        "project_organization": "事务组织方式",
        "time_rhythms": "实际时间规律",
        "place_based_practices": "地点相关活动",
        "interruptions_and_exceptions": "中断与例外",
    },
    "life_course": {
        "past_anchors": "重要过往",
        "current_phase": "当前阶段",
        "active_transitions": "正在经历的转变",
        "future_horizons": "未来方向",
        "unfinished_matters": "未完成的事",
        "self_authored_meaning": "经历的个人意义",
    },
    "relationship_with_user": {
        "shared_referents": "共同背景",
        "interaction_language": "相处语言",
        "coordination_and_care": "协调与关照",
        "boundaries_and_commitments": "边界与承诺",
        "shared_projects": "共同事务",
        "relationship_change": "关系变化",
    },
}

SECTION_LABELS = dict(
    zip(
        SECTION_DIMENSIONS,
        (
            "身份与性格",
            "现实生活",
            "社会关系",
            "偏好与目标",
            "实践与规律",
            "经历与轨迹",
            "与用户的关系",
        ),
        strict=True,
    )
)
