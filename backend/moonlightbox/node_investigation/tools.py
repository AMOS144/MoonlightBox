"""只读检索与可编辑草稿工具；从不修改原始消息、已发布图谱或批准结果。"""

from collections import Counter

from langchain_core.tools import StructuredTool
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.contracts import RegisteredTool, ToolContract
from moonlightbox.agent_runtime.tool_errors import ToolInputError, ToolServiceError
from moonlightbox.agent_runtime.tool_execution import serial
from moonlightbox.world.models import ConversationBundle, WorldGraphVersion
from moonlightbox.world.person_world.tools.graph_query import match_bundle_document_reference
from moonlightbox.world.person_world.tools.source_messages import _bundle_rows, _matching_indexes

from .schemas import CandidateArgs, ContextArgs, EmptyArgs, ReadArgs, SearchArgs, WorkArgs
from .store import activity, digest


class InvestigationTools:
    def __init__(self, store, job_id, client):
        self.store, self.job_id, self.client = store, job_id, client
        row = store.get()
        self.rows = store.source_rows(row.state["message_ids"])
        if store.fingerprint(self.rows) != row.dataset_version:
            raise ToolInputError("原始数据已发生变化，请以新数据开始调查")
        if not row.state["graph_id"] and client is not None:
            # 原文调查可以先开始；索引稍后完成时，在下一回合绑定匹配导入集合的图版本。
            # 已绑定的图不随“最新版本”悄悄切换，保持本轮检索的可复现性。
            with Session(store.engine) as session:
                graphs = session.scalars(
                    select(WorldGraphVersion)
                    .where(
                        WorldGraphVersion.project_id == store.project_id,
                        WorldGraphVersion.status == "ready",
                    )
                    .order_by(WorldGraphVersion.created_at.desc())
                )
                imports = {m.import_id for m, _ in self.rows}
                graph = next((g for g in graphs if set(g.source_import_ids) == imports), None)
                if graph:
                    store.mutate(
                        lambda state, _: state.update(
                            graph_id=graph.id, workspace=graph.workspace_key
                        ),
                        job_id=job_id,
                    )
        self.positions = {m.id: i for i, (m, _) in enumerate(self.rows)}
        self.observed = {c["id"]: c["revision"] for c in row.state["candidates"]}
        self.seen_input = len(row.state["inputs"])

    def snapshot(self):
        return {"observed_candidate_revisions": self.observed}

    def restore(self, value):
        self.observed = dict(value.get("observed_candidate_revisions", {}))

    def overview(self):
        view = self.store.view()
        self.observed = {c["id"]: c["revision"] for c in view["candidates"]}
        gaps = sorted(
            ((self.rows[i][0].timestamp - self.rows[i - 1][0].timestamp).total_seconds(), i)
            for i in range(1, len(self.rows))
        )
        view["record_range"] = [
            self.rows[0][0].timestamp.isoformat(),
            self.rows[-1][0].timestamp.isoformat(),
        ]
        view["monthly_message_counts"] = dict(
            Counter(m.timestamp.strftime("%Y-%m") for m, _ in self.rows)
        )
        view["participants"] = list(
            {p.id: {"name": p.name, "role": p.role} for _, p in self.rows}.values()
        )
        view["largest_gaps"] = [
            {
                "seconds": seconds,
                "before_ref": self.rows[i - 1][0].id,
                "after_ref": self.rows[i][0].id,
            }
            for seconds, i in gaps[-12:][::-1]
        ]
        view["retrieval_readiness"] = {
            "original_messages": "ready",
            "rag_and_graph": "configured" if view["graph_id"] and self.client else "unavailable",
            "note": "configured 表示有已完成图版本，实际故障由工具返回；空档不证明经历",
        }
        for key in (
            "activities",
            "activity_seq",
            "preview",
            "id",
            "job_id",
            "revision",
            "turn",
            "dataset_version",
            "confirmed",
        ):
            view.pop(key, None)
        for candidate in view["candidates"]:
            candidate.pop("creation_hash", None)
        for item in view["inputs"]:
            item.pop("request_id", None)
        return view

    def messages(self, indexes):
        return [self.store.message_json(*self.rows[i], i) for i in sorted(set(indexes))]

    def read(self, **kwargs):
        args = ReadArgs(**kwargs)
        state = self.store.get().state
        # 显式游标使进程在工具落库后、检查点前中断时重放同一页，不会跳过正文。
        cursor = args.cursor
        range_end = state.get("range_end", len(self.rows))
        if cursor > state["cursor"] or not state.get("range_start", 0) <= cursor <= range_end:
            raise ToolInputError(
                "不能跳过未读记录；使用概览 cursor 或上一页 next_cursor", field="cursor"
            )
        end = min(range_end, cursor + args.limit)
        result = self.messages(range(cursor, end))

        def apply(state, _):
            state["cursor"] = max(state["cursor"], end)
            merged = []
            for a, b in sorted([*state["read_intervals"], [cursor, end]]):
                if a == b:
                    continue
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            state["read_intervals"] = merged
            if result:
                activity(
                    state,
                    "read",
                    f"阅读至 {result[-1]['sent_at']}，游标 {state['cursor']}；覆盖按已读区间计算",
                )

        self.store.mutate(apply, job_id=self.job_id)
        return {
            "messages": result,
            "next_cursor": end,
            "end_of_record": end == range_end,
            "scope_end": range_end,
            "total": len(self.rows),
        }

    def context(self, **kwargs):
        args = ContextArgs(**kwargs)
        if any(ref not in self.positions for ref in args.message_refs):
            raise ToolInputError(
                "引用不属于当前冻结记录，请使用原文工具返回的 message_ref", field="message_refs"
            )
        indexes = set()
        for ref in args.message_refs:
            pos = self.positions[ref]
            indexes.update(
                range(max(0, pos - args.radius), min(len(self.rows), pos + args.radius + 1))
            )
        return {"messages": self.messages(indexes), "advances_coverage": False}

    def search(self, mode, **kwargs):
        args = SearchArgs(**kwargs)
        state = self.store.get().state
        if not state["workspace"] or self.client is None:
            raise ToolServiceError(
                "当前没有可查询的已完成索引；可继续顺序读原文，不代表没有相关经历",
                code="retrieval_not_ready",
            )
        retrieval = self.client.query(
            state["workspace"],
            args.question,
            mode=mode,
            top_k=20,
            chunk_top_k=args.limit,
            max_total_tokens=8000,
        )
        with Session(self.store.engine) as session:
            bundles = list(
                session.scalars(
                    select(ConversationBundle).where(
                        ConversationBundle.project_id == self.store.project_id,
                        ConversationBundle.graph_version_id == state["graph_id"],
                    )
                )
            )
            by_name = {b.source_name: b for b in bundles}
            fragments = []
            for ref in retrieval.references[: args.limit]:
                name = match_bundle_document_reference(ref.file_path, set(by_name))
                if name is None:
                    continue
                bundle_rows = _bundle_rows(session, by_name[name].id)
                hits = _matching_indexes(bundle_rows, [ref.content or ""])
                # 不知道 chunk 对应哪条时明示返回 Bundle 开头，不伪称精确命中。
                ids = (
                    [bundle_rows[i][1].id for i in sorted(hits)]
                    if hits
                    else [r[1].id for r in bundle_rows[:12]]
                )
                ids = [i for i in ids if i in self.positions]
                fragments.append(
                    {
                        "chunk": ref.content,
                        "message_refs": ids,
                        "mapping": "chunk_matched" if hits else "bundle_context_only",
                        "messages": self.messages([self.positions[i] for i in ids[:12]]),
                    }
                )
        self.store.mutate(
            lambda s, _: activity(
                s,
                "search",
                f"{'原文语义检索' if mode == 'naive' else '图谱检索'}：{args.question}；"
                f"返回 {len(fragments)} 个可定位片段",
            ),
            job_id=self.job_id,
        )
        return {
            "mode": mode,
            "context": retrieval.context,
            "fragments": fragments,
            "raw_reference_count": len(retrieval.references),
            "advances_coverage": False,
            "status": "found" if fragments else "no_locatable_references",
            "note": "检索结果仅是线索；无法定位不等于没有经历。发生时间可能早于提及时间。",
        }

    def save_work(self, **kwargs):
        args = WorkArgs(**kwargs)
        self.store.mutate(lambda s, _: s.update(work=args.model_dump()), job_id=self.job_id)
        return {"status": "saved", "work": args.model_dump()}

    def upsert(self, **kwargs):
        args = CandidateArgs(**kwargs)
        refs = [*args.source_refs, *(b.message_ref for b in args.boundaries if b.message_ref)]
        if any(ref not in self.positions for ref in refs):
            raise ToolInputError("候选包含无效原文引用，请先读取实际消息", field="source_refs")
        if args.status == "ready" and not args.boundaries:
            raise ToolInputError("ready 候选至少需要一个可预览的边界", field="boundaries")
        if args.status == "ready" and (
            len(args.boundaries) != 1 or args.boundaries[0].kind != "message"
        ):
            raise ToolInputError("ready 候选必须有且只有一个消息边界", field="boundaries")
        content = args.model_dump(exclude={"candidate_ref"})

        def apply(state, _):
            if len(state["inputs"]) > self.seen_input:
                raise ToolInputError("有尚未读到的用户补充。下一次思考会收到新输入，再据此更新候选")
            if any(i not in {x["id"] for x in state["inputs"]} for i in args.user_input_refs):
                raise ToolInputError(
                    "用户补充引用无效，请查看当前调查输入", field="user_input_refs"
                )
            candidate = next(
                (c for c in state["candidates"] if c["id"] == args.candidate_ref), None
            )
            if args.candidate_ref and candidate is None:
                raise ToolInputError("候选不存在，请读取当前工作台")
            if candidate:
                if (state.get("confirmed") or {}).get("candidate_id") == candidate["id"]:
                    raise ToolInputError("用户已确认此候选，不能直接改写；请另列补充或向用户说明")
                if all(candidate.get(key) == value for key, value in content.items()):
                    return candidate
                if candidate["revision"] != self.observed.get(candidate["id"]):
                    raise ToolInputError(
                        "候选已被更新，请先调用 get_record_overview 阅读最新版本再修改"
                    )
                candidate.update(content, revision=candidate["revision"] + 1)
            else:
                # Worker 在工具回执存档前重启时，相同新增提案不会重复生成卡片。
                duplicate = next(
                    (c for c in state["candidates"] if c.get("creation_hash") == digest(content)),
                    None,
                )
                if duplicate:
                    return duplicate
                from uuid import uuid4

                candidate = {
                    **content,
                    "id": str(uuid4()),
                    "revision": 1,
                    "creation_hash": digest(content),
                }
                state["candidates"].append(candidate)
            activity(state, "candidate", f"{candidate['title']}：{args.change_reason}")
            return candidate

        result = self.store.mutate(apply, job_id=self.job_id)
        self.observed[result["id"]] = result["revision"]
        return result

    def registered(self):
        definitions = [
            (
                "get_record_overview",
                "读取冻结记录概况、真实覆盖率、当前候选和用户补充；用于恢复调查及修复版本冲突。",
                EmptyArgs,
                self.overview,
                False,
            ),
            (
                "read_next_messages",
                "从最早记录顺序分页阅读。保存真实阅读进度，可重读但不跳页。",
                ReadArgs,
                self.read,
                True,
            ),
            (
                "read_message_context",
                "按真实引用补齐前后连续对话，不推进顺序阅读进度。",
                ContextArgs,
                self.context,
                False,
            ),
            (
                "search_source_messages",
                "原文向量 RAG（naive）召回语义片段并还原消息引用；不做关键词或正则事件分类。",
                SearchArgs,
                lambda **kw: self.search("naive", **kw),
                False,
            ),
            (
                "search_world",
                "查询 LightRAG 关系线索（mix）并还原原文，可寻找事后回忆；"
                "不代表历史起点可使用未来信息。",
                SearchArgs,
                lambda **kw: self.search("mix", **kw),
                False,
            ),
            (
                "save_investigation_work",
                "保存跨页理解、未决问题和下一步；不替代保存候选，也不结束回合。",
                WorkArgs,
                self.save_work,
                True,
            ),
            (
                "upsert_node_candidate",
                "增量创建或更新候选经历及可选起点；只改草稿，不执行用户批准或创建分支。",
                CandidateArgs,
                self.upsert,
                True,
            ),
        ]
        return tuple(
            RegisteredTool(
                StructuredTool.from_function(
                    name=name, description=description, args_schema=schema, func=func
                ),
                ToolContract(
                    name=name,
                    side_effect="proposal" if writes else "read_only",
                    timeout_seconds=300 if name.startswith("search_") else 30,
                    max_result_chars=48000,
                    execution=serial(
                        "node_investigation",
                        reason="读后进度、版本及检索工件共享工作台，串行；同步数据库只支持边界取消",
                        cooperative_io=name.startswith("search_"),
                    ),
                ),
            )
            for name, description, schema, func, writes in definitions
        )
