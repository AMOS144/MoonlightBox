import json

from moonlightbox.timeline.state import StateSnapshot


def build_persona_prompt(
    snapshot: StateSnapshot,
    style_instruction: str,
) -> str:
    state = {
        "时间": snapshot.at.isoformat(),
        "关系状态": snapshot.relationship_status,
        "当前情绪": snapshot.emotions,
        "未完成话题": snapshot.open_loops,
        "证据消息": snapshot.evidence_ids,
    }
    return (
        "你必须只根据以下历史时点状态回应，不得使用此时点之后发生的信息。\n"
        f"语言风格：{style_instruction}\n"
        f"历史状态：{json.dumps(state, ensure_ascii=False)}"
    )
