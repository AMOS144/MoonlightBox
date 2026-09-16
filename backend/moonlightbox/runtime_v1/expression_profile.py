"""从分支绑定的画像取得表达资料，不在运行时生成或偷偷切换活动发布版本。"""

import hashlib
import json
from copy import deepcopy

from moonlightbox.world.person_world.contracts.profile_v3 import EXPRESSION_FIELDS


def expression_material(profile):
    relation = profile.get("relationship_with_user", {}) if isinstance(profile, dict) else {}
    material = relation.get("expression_profile") if isinstance(relation, dict) else None
    if not isinstance(material, dict):
        return {
            "status": "not_compiled",
            "explanation": "此版本画像尚未编译人物表达资料，可按需查询真实互动。",
        }
    material = deepcopy(material)
    populated = [
        key
        for key in EXPRESSION_FIELDS
        if isinstance(material.get(key), dict) and material[key].get("status") == "described"
    ]
    return {
        "status": "available"
        if len(populated) == len(EXPRESSION_FIELDS)
        else "partial"
        if populated
        else "unknown",
        "version": hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "profile": material,
    }
