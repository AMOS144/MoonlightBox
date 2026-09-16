"""确定性工作投影：保留候选历史与回执，不把自然语言里的“完成”当成提交。"""

import hashlib
import json


def project_turn(state, result, *, actor):
    tasks = dict(state.get("tasks", {}))
    proposals = dict(state.get("proposals", {}))
    commits = dict(state.get("commits", {}))
    delivery = dict(state.get("delivery", {}))
    for message in state.get("messages", []):
        envelope = json.loads(message.content)
        if envelope["recipient"] == actor and delivery.get(message.id) == "pending":
            delivery[message.id] = "processed"
    for message in result.get("messages", []):
        envelope = json.loads(message.content)
        delivery[message.id] = "visible" if envelope["recipient"] == "broadcast" else "pending"
    candidate = result.get("proposal")
    if candidate is not None:
        key = hashlib.sha256(
            (state["cycle_id"] + json.dumps(candidate, sort_keys=True)).encode()
        ).hexdigest()
        proposals.setdefault(
            key,
            {
                "content": candidate,
                "base_plan_version": result["base_plan_version"],
                "input_revision": state["input_revision"],
                "status": "proposed",
            },
        )
    receipt = result.get("receipt")
    if actor == "executor" and receipt and receipt.get("status") == "committed":
        key = receipt["key"]
        if key in commits and commits[key] != receipt:
            raise ValueError("已确认提交回执不可覆盖")
        commits[key] = receipt
        if key in proposals:
            proposals[key] = {**proposals[key], "status": "committed"}
    route = result.get("route", state.get("route"))
    tasks[state.get("task_id", state["cycle_id"])] = {
        "status": result.get("status", "running"),
        "next": route,
        "request": result.get("request", state.get("request")),
    }
    return {"tasks": tasks, "proposals": proposals, "commits": commits, "delivery": delivery}
