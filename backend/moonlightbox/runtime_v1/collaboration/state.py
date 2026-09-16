"""协作图持久化状态，不保存 Session、服务对象或模型客户端。"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage

from .messages import merge_shared_messages


class RuntimeState(TypedDict, total=False):
    schema_version: str
    branch_id: str
    cycle_id: str
    task_id: str
    input_revision: int
    input_event_ids: list[str]
    input_packet: dict
    planner_dispatch: dict | None
    messages: Annotated[list[AnyMessage], merge_shared_messages]
    day_plans: dict
    planner_work: dict | None
    decision: dict | None
    request: dict | None
    proposal: dict | None
    base_plan_version: int | None
    plan_mode: str
    date_features: dict
    expected_state_version: int
    receipt: dict | None
    actor_message: dict | None
    expression_dependencies: str
    expression_assets: dict
    expression_error: str | None
    route: str
    status: str
    handoffs: int
    started_at: float
    seen_requests: list[str]
    result: dict[str, Any]
    usage: dict[str, int]
    tasks: dict[str, dict]
    proposals: dict[str, dict]
    commits: dict[str, dict]
    delivery: dict[str, str]
    life: dict
