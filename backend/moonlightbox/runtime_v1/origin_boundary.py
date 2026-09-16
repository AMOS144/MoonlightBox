"""只消费用户已批准的精确边界；客户端不能另传时间或训练模型。"""

from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

from moonlightbox.node_investigation.models import NodeInvestigation
from moonlightbox.node_investigation.store import InvestigationStore


def approved_boundary(session, project_id, investigation_id, preview_hash):
    row = session.get(NodeInvestigation, investigation_id)
    if row is None or row.project_id != project_id:
        raise LookupError("起点调查不存在")
    confirmed = row.state.get("confirmed")
    if not confirmed or confirmed.get("preview_hash") != preview_hash:
        raise ValueError("起点尚未批准或批准版本已变化，请重新确认")
    store = InvestigationStore(session.get_bind(), project_id, investigation_id)
    rows = store.source_rows(row.state["message_ids"], session=session)
    if store.fingerprint(rows) != row.dataset_version:
        raise ValueError("原始资料已改变，请重新调查并确认起点")
    candidate = next(
        (c for c in row.state["candidates"] if c["id"] == confirmed["candidate_id"]), None
    )
    if candidate is None or candidate["revision"] != confirmed["candidate_revision"]:
        raise ValueError("起点候选已更新，请重新预览并确认")
    origin = datetime.fromisoformat(confirmed["cutoff_at"])
    if origin.tzinfo is None:
        raise ValueError("已确认起点缺少时区，请重新确认")
    ZoneInfo(confirmed["timezone"])
    return {**deepcopy(confirmed), "investigation_id": investigation_id}
