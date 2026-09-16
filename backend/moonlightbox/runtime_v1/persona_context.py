"""为 PersonaActor 装配隔离的表达材料，不复制 Director 的工具过程与协商全文。"""

from copy import deepcopy
from datetime import datetime


class PersonaContextAssembler:
    def assemble(self, director_view, *, task=None, history=None, related_material=None):
        current = director_view["current"]
        task = task or current.get("expression_task") or {}
        profile = director_view["origin"].get("person_world_profile", {})
        window = director_view["branch"].get("working_window", {})
        recent = deepcopy(window.get("imported_prefix", []) + window.get("messages", []))
        selected = {
            key: deepcopy(profile[key])
            for key in ("profile_schema_version", "overview", "identity", "relationship_with_user")
            if key in profile
        }
        refs = set(task.get("respond_to_refs", []) + task.get("context_refs", []))
        responded, related = [], list(related_material or [])
        if history is not None:
            for ref in task.get("respond_to_refs", []):
                responded.extend(history.read(around_ref=ref, limit=1)["messages"])
            for ref in task.get("context_refs", []):
                # 记忆引用由领域层另外展开；消息引用在这里统一回读。
                if ref in history.references()[0]:
                    related.append(history.read(around_ref=ref, limit=9))
        state = director_view.get("subjective_state", {})
        subjective = {
            "mood": state.get("mood"),
            "concerns": [
                deepcopy(item)
                for item in state.get("concerns", [])
                if refs.intersection(item.get("source_refs", []))
            ],
        }
        life = {
            key: deepcopy(value)
            for key, value in current.get("life_state", {}).items()
            if key in {"activity", "location", "availability", "energy", "fatigue", "valid_until"}
        }
        active = []
        try:
            now = datetime.fromisoformat(director_view["virtual_now"])
            active = [
                deepcopy(block)
                for block in current.get("day_plans", {})
                .get(now.date().isoformat(), {})
                .get("blocks", [])
                if block.get("start", "") <= now.strftime("%H:%M") < block.get("end", "")
            ]
        except (TypeError, ValueError):
            pass
        return {
            "expression_task": deepcopy(task),
            "respond_to_messages": responded,
            "recent_messages": recent,
            "related_material": related,
            "conversation_summaries": deepcopy(director_view["branch"].get("overlay_summary")),
            "subjective_state": subjective,
            "life_state": life,
            "confirmed_current_blocks": active,
            "virtual_now": director_view["virtual_now"],
            "expression_style_profile": deepcopy(current.get("expression_style_profile", {})),
            "person_world_profile": selected,
            "relationship_with_user": deepcopy(profile.get("relationship_with_user", {})),
        }
