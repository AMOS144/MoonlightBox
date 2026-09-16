"""调查工作台的短事务边界。用户输入不会取消模型，也不会被旧草稿覆盖。"""

import hashlib
import json
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.tool_errors import ToolInputError
from moonlightbox.imports.models import Message, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.projects.models import Project
from moonlightbox.world.models import WorldGraphVersion

from .models import NodeInvestigation

JOB_KIND = "node_investigation_turn"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def activity(state, kind, text):
    state["activity_seq"] = state.get("activity_seq", 0) + 1
    state["activities"] = [
        *state.get("activities", []),
        {
            "id": state["activity_seq"],
            "kind": kind,
            "text": text,
            "at": datetime.now(UTC).isoformat(),
        },
    ][-150:]


def queue_turn(session, row_id, project_id, state):
    """Job 和业务排队状态在同一事务写入，重复 HTTP 不会产生第二个 Agent。"""
    state["turn"] = state.get("turn", 0) + 1
    job = Job(
        id=str(uuid4()),
        kind=JOB_KIND,
        payload={"investigation_id": row_id, "project_id": project_id},
        dedupe_key=f"{JOB_KIND}:{row_id}:{state['turn']}",
    )
    session.add(job)
    state.update(job_id=job.id, status="queued", error=None)


class InvestigationStore:
    def __init__(self, engine, project_id, investigation_id=None):
        self.engine, self.project_id, self.id = engine, project_id, investigation_id

    def source_rows(self, ids=None, *, session=None):
        with nullcontext(session) if session is not None else Session(self.engine) as session:
            rows = list(
                session.execute(
                    select(Message, Participant)
                    .join(Participant, Message.participant_id == Participant.id)
                    .where(Message.project_id == self.project_id)
                )
            )
            # 同秒按导入源及源消息编号稳定排序；冻结后的序号才是边界依据。
            from moonlightbox.imports.message_order import message_order_key

            rows.sort(
                key=lambda r: message_order_key(
                    r[0].timestamp, r[0].import_id, r[0].source_id, r[0].id
                )
            )
            if ids is not None:
                by_id = {m.id: (m, p) for m, p in rows}
                if any(i not in by_id for i in ids):
                    raise ToolInputError(
                        "冻结的原始消息已不存在，请重新建立调查，不能继续确认旧边界"
                    )
                rows = [by_id[i] for i in ids]
            return rows

    @staticmethod
    def fingerprint(rows):
        return digest(
            [(m.id, m.timestamp, m.content, m.kind, p.id, p.name, p.role) for m, p in rows]
        )

    def start(self, timezone):
        ZoneInfo(timezone)
        rows = self.source_rows()
        if not rows or not {"self", "target"}.issubset({p.role for _, p in rows}):
            raise ToolInputError("请先导入聊天并确认 self/target 参与者，再开始节点调查")
        version = self.fingerprint(rows)
        with Session(self.engine) as session:
            if not session.get(Project, self.project_id):
                raise ToolInputError("项目不存在")
            old = session.scalar(
                select(NodeInvestigation).where(
                    NodeInvestigation.project_id == self.project_id,
                    NodeInvestigation.dataset_version == version,
                )
            )
            if old:
                if old.state["timezone"] != timezone:
                    raise ToolInputError("这份记录已绑定其他时区，不能在已有边界上静默更换")
                self.id = old.id
                return self.view()
            previous = session.scalar(
                select(NodeInvestigation)
                .where(NodeInvestigation.project_id == self.project_id)
                .order_by(NodeInvestigation.created_at.desc())
            )
            if previous and previous.state["status"] in {"running", "queued"}:
                raise ToolInputError("旧版本调查仍在运行，请先暂停，再为新导入记录建立调查")
            graph = session.scalar(
                select(WorldGraphVersion)
                .where(
                    WorldGraphVersion.project_id == self.project_id,
                    WorldGraphVersion.status == "ready",
                )
                .order_by(WorldGraphVersion.created_at.desc())
            )
            self.id = str(uuid4())
            state = {
                "message_ids": [m.id for m, _ in rows],
                "timezone": timezone,
                "graph_id": graph.id if graph else None,
                "workspace": graph.workspace_key if graph else None,
                "cursor": 0,
                "read_intervals": [],
                "skipped_intervals": [],
                "candidates": [],
                "inputs": [],
                "pending_question": None,
                "work": {},
                "activities": [],
                "turn": 0,
                "seen_input": 0,
                "status": "queued",
                "confirmed": None,
            }
            queue_turn(session, self.id, self.project_id, state)
            activity(state, "queued", "开始从最早的聊天记录调查；图谱仅作线索，不代替原文")
            session.add(
                NodeInvestigation(
                    id=self.id, project_id=self.project_id, dataset_version=version, state=state
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                self.id = session.scalar(
                    select(NodeInvestigation.id).where(
                        NodeInvestigation.project_id == self.project_id,
                        NodeInvestigation.dataset_version == version,
                    )
                )
                if self.id is None:
                    raise
        return self.view()

    def get(self):
        with Session(self.engine) as session:
            row = session.get(NodeInvestigation, self.id)
            if row is None or row.project_id != self.project_id:
                raise ToolInputError("节点调查不存在")
            session.expunge(row)
            return row

    def mutate(self, operation, *, job_id=None):
        for _ in range(5):
            with Session(self.engine) as session:
                # SQLite 先取得短写事务，避免两个回答都先插入相同下一回合 Job 才竞争 CAS。
                # 此处不包含任何模型/网络调用；其他数据库仍由 revision CAS 保护。
                if self.engine.dialect.name == "sqlite":
                    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                row = session.get(NodeInvestigation, self.id)
                if row is None or row.project_id != self.project_id:
                    raise ToolInputError("节点调查不存在")
                state = deepcopy(row.state)
                if job_id and (
                    state.get("job_id") != job_id or state["status"] not in {"queued", "running"}
                ):
                    raise ToolInputError("任务已暂停、取消或被替代；不能再写入旧执行")
                if job_id and getattr(self, "worker_token", None):
                    job = session.get(Job, job_id)
                    if not job or job.status != "running" or job.worker_token != self.worker_token:
                        raise ToolInputError("Worker 租约已失效，不能提交旧执行")
                result = operation(state, session)
                changed = session.execute(
                    update(NodeInvestigation)
                    .where(
                        NodeInvestigation.id == row.id, NodeInvestigation.revision == row.revision
                    )
                    .values(state=state, revision=row.revision + 1, updated_at=datetime.now(UTC))
                )
                if changed.rowcount == 1:
                    session.commit()
                    return result
                session.rollback()
        raise ToolInputError("工作台刚被其他操作更新，请刷新后重试")

    def view(self):
        row = self.get()
        state = deepcopy(row.state)
        count = len(state.pop("message_ids"))
        state.pop("workspace", None)
        scope_start = state.get("range_start", 0)
        scope_end = state.get("range_end", count)
        state["range_count"] = scope_end - scope_start
        state["read_count"] = sum(
            max(0, min(b, scope_end) - max(a, scope_start)) for a, b in state["read_intervals"]
        )
        with Session(self.engine) as session:
            job = session.get(Job, state.get("job_id"))
            if job and job.status in {"failed", "interrupted", "cancelled", "cancelling"}:
                state.update(status=job.status, error=job.error_code)
        return {
            "id": row.id,
            "dataset_version": row.dataset_version,
            "revision": row.revision,
            "total_messages": count,
            **state,
        }

    def add_input(self, args):
        def apply(state, session):
            if any(i["request_id"] == args.request_id for i in state["inputs"]):
                return
            if any(ref not in state["message_ids"] for ref in args.message_refs):
                raise ToolInputError("圈选的消息不在本次冻结记录中")
            question = state.get("pending_question")
            if args.question_id and (not question or args.question_id != question["id"]):
                raise ToolInputError("这个问题已经被回答或替换，请刷新当前问题")
            if args.kind in {"answer", "skip", "unknown"} and not args.question_id:
                raise ToolInputError("回答、跳过或不知道需要绑定当前问题")
            if not args.text.strip() and args.kind not in {"skip", "unknown"}:
                raise ToolInputError("请填写补充内容")
            item = {**args.model_dump(), "id": str(uuid4()), "seq": len(state["inputs"]) + 1}
            state["inputs"].append(item)
            if args.question_id:
                state["pending_question"] = None
            activity(
                state,
                "user_input",
                args.text or ("用户暂时跳过" if args.kind == "skip" else "用户不确定"),
            )
            active_job = session.get(Job, state.get("job_id"))
            if active_job and active_job.status == "cancelling":
                raise ToolInputError("旧执行正在退出，请稍后发送；草稿仍在输入框中")
            if active_job and active_job.status in {"failed", "interrupted", "cancelled"}:
                state["status"] = active_job.status
            if state["status"] not in {"running", "queued", "failed", "interrupted"}:
                queue_turn(session, self.id, self.project_id, state)

        self.mutate(apply)
        return self.view()

    def set_scope(self, args):
        """只有用户能主动跳过区间；范围变化单独记录，不伪称完整覆盖。"""
        row = self.get()
        rows = self.source_rows(row.state["message_ids"])
        zone = ZoneInfo(row.state["timezone"])
        included = [
            i
            for i, (m, _) in enumerate(rows)
            if (
                not args.start_date
                or (
                    m.timestamp.astimezone(zone).date()
                    if m.timestamp.tzinfo
                    else m.timestamp.date()
                )
                >= args.start_date
            )
            and (
                not args.end_date
                or (
                    m.timestamp.astimezone(zone).date()
                    if m.timestamp.tzinfo
                    else m.timestamp.date()
                )
                <= args.end_date
            )
        ]
        if not included:
            raise ToolInputError("所选日期范围没有记录，请调整日期")
        start, end = included[0], included[-1] + 1

        def apply(state, session):
            job = session.get(Job, state.get("job_id"))
            if job and job.status in {"running", "queued", "cancelling"}:
                raise ToolInputError("请先暂停并等当前执行退出，再改变调查范围")
            state.update(
                range_start=start,
                range_end=end,
                cursor=start,
                requested_range=args.model_dump(mode="json"),
                pending_question=None,
            )
            state["skipped_intervals"] = [
                pair for pair in [[0, start], [end, len(rows)]] if pair[0] < pair[1]
            ]
            activity(
                state,
                "scope",
                f"用户指定范围：{args.start_date or '最早'} 至 {args.end_date or '最后'}；"
                "其他区间未计入覆盖",
            )
            queue_turn(session, self.id, self.project_id, state)

        self.mutate(apply)
        return self.view()

    def preview(self, candidate_id, args):
        row = self.get()
        rows = self.source_rows(row.state["message_ids"])
        if self.fingerprint(rows) != row.dataset_version:
            raise ToolInputError("原始材料或参与者已改变，请新建调查后重新确认")
        candidate = next((c for c in row.state["candidates"] if c["id"] == candidate_id), None)
        if not candidate or candidate["revision"] != args.candidate_revision:
            raise ToolInputError("候选已更新，请重新阅读并预览")
        if candidate["status"] != "ready" or args.option_index >= len(candidate["boundaries"]):
            raise ToolInputError("请选择已准备好的有效起点")
        option = candidate["boundaries"][args.option_index]
        if option["kind"] == "message":
            pos = row.state["message_ids"].index(option["message_ref"])
            moment = rows[pos][0].timestamp
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=ZoneInfo(row.state["timezone"]))
            cutoff = moment.isoformat()
            included = pos + (option["side"] == "after")
        else:
            try:
                when = datetime.fromisoformat(args.exact_time or "")
                if when.tzinfo is None:
                    raise ValueError()
            except ValueError as error:
                raise ToolInputError(
                    "空档起点需要用户明确带时区的时间，例如 2026-05-01T10:00:00+08:00"
                ) from error
            zone = ZoneInfo(row.state["timezone"])
            times = [
                (
                    m.timestamp.replace(tzinfo=zone) if m.timestamp.tzinfo is None else m.timestamp
                ).astimezone(UTC)
                for m, _ in rows
            ]
            when = when.astimezone(UTC)
            if not times[0] < when < times[-1] or when in times:
                raise ToolInputError(
                    "空档起点必须在记录范围内且不与消息时刻重合；消息处请使用前/后边界"
                )
            included = sum(t <= when for t in times)
            cutoff = when.isoformat()
        result = {
            "dataset_version": row.dataset_version,
            "candidate_id": candidate_id,
            "candidate_revision": candidate["revision"],
            "option_index": args.option_index,
            "exact_time": args.exact_time,
            "cutoff_at": cutoff,
            "included_count": included,
            "last_included_ref": rows[included - 1][0].id if included else None,
            "first_excluded_ref": rows[included][0].id if included < len(rows) else None,
            "timezone": row.state["timezone"],
            "option": option,
        }
        result["preview_hash"] = digest(result)
        result["messages"] = [
            self.message_json(m, p, i)
            for i, (m, p) in enumerate(rows)
            if max(0, included - 3) <= i < included + 3
        ]
        self.mutate(lambda state, _: state.update(preview=result))
        return result

    @staticmethod
    def message_json(message, participant, ordinal):
        return {
            "message_ref": message.id,
            "ordinal": ordinal,
            "sent_at": message.timestamp.isoformat(),
            "speaker": participant.name,
            "role": participant.role,
            "kind": message.kind,
            "content": message.content,
        }

    def confirm(self, candidate_id, args):
        from .schemas import PreviewArgs

        preview = self.get().state.get("preview")
        if (
            not preview
            or preview["preview_hash"] != args.preview_hash
            or preview["candidate_id"] != candidate_id
        ):
            raise ToolInputError("请先预览这个起点，再确认相同版本")
        checked = self.preview(
            candidate_id,
            PreviewArgs(
                candidate_revision=preview["candidate_revision"],
                option_index=preview["option_index"],
                exact_time=preview["exact_time"],
            ),
        )
        if checked["preview_hash"] != args.preview_hash:
            raise ToolInputError("边界已经变化，请重新预览")

        def apply(state, session):
            candidate = next(c for c in state["candidates"] if c["id"] == candidate_id)
            if candidate["revision"] != checked["candidate_revision"]:
                raise ToolInputError("候选刚刚改变，请重新确认")
            if (
                self.fingerprint(self.source_rows(state["message_ids"], session=session))
                != checked["dataset_version"]
            ):
                raise ToolInputError("原始材料已改变，请重新建立调查并预览")
            state["confirmed"] = {
                **checked,
                "approved_at": datetime.now(UTC).isoformat(),
                "continue_investigating": args.continue_investigating,
            }
            if not args.continue_investigating:
                state["status"] = "paused"
                job = session.get(Job, state.get("job_id"))
                if job and job.status == "queued":
                    job.status = "cancelled"
                elif job and job.status == "running":
                    job.status = "cancelling"
            elif state["status"] not in {"queued", "running", "waiting_for_user"}:
                queue_turn(session, self.id, self.project_id, state)
            activity(
                state, "confirmed", "用户确认分支起点；历史人物画像尚未编译，不自动创建聊天分支"
            )

        self.mutate(apply)
        return self.view()
