"""统一返回文字/表情包真实使用片段，不调用视觉模型。"""

from typing import Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import Field, model_validator
from sqlalchemy import select

from moonlightbox.agent_runtime.tool_errors import ToolInputError, ToolServiceError
from moonlightbox.config import Settings

from ..branch_models import Branch
from ..conversation_history import ConversationHistory
from ..db_models import RuntimeSnapshotRow
from ..schemas import StrictModel
from ..style_history import search_usage, source_rows, validate_asset


class GetStyleExamplesArgs(StrictModel):
    situation: str = Field(
        default="",
        max_length=1200,
        description="情境搜索时必填：当前对话在谈什么、关系距离与情绪；素材用法查询时可省略",
    )
    intent: str = Field(
        default="",
        max_length=160,
        description="情境搜索时必填：本轮想怎样接话或表达；素材用法查询时可省略",
    )
    speech_mode: Literal["reply", "proactive", "delayed_reply"] = Field(
        default="reply",
        description=(
            "reply 为接话，proactive 为主动发起，delayed_reply 为隔一段时间再接话；"
            "用于匹配互动方式，不会发送消息"
        ),
    )
    limit: int = Field(
        default=4, ge=1, le=6, description="最多召回的互动窗口数；素材历史模式下为本页用例数"
    )
    asset_ref: str | None = Field(
        default=None,
        description="可选：当前材料或工具已提供的素材引用；查看该素材的其他历史用法，不凭空构造",
    )
    usage_cursor: str | None = Field(
        default=None, description="查询同一素材下一页时原样传入上一页 more_usage_cursor；首次省略"
    )

    @model_validator(mode="after")
    def check_query_mode(self):
        if self.asset_ref is not None:
            if not self.asset_ref.strip():
                raise ValueError("asset_ref 不能为空字符串；情境搜索时省略此字段")
        elif not self.situation.strip() or not self.intent.strip():
            raise ValueError("情境搜索需要非空 situation 和 intent；已知素材查询则提供 asset_ref")
        if self.usage_cursor is not None and (not self.asset_ref or not self.usage_cursor.strip()):
            raise ValueError(
                "usage_cursor 必须配合同一 asset_ref，原样使用 more_usage_cursor；首次省略"
            )
        return self


_RETRIEVAL_QUERY = """查找目标人物在相似互动中怎样表达的真实聊天片段。
当前情境：{situation}
表达目的：{intent}；模式：{speech_mode}
匹配情绪、关系距离和互动节奏，包括文字及表情使用位置。
不要把时间相邻的消息强行配成问答，不把生成的解释当成原话。"""


def _retrieval_query(args):
    return _RETRIEVAL_QUERY.format(**args.model_dump())


class StyleService:
    def __init__(self, session, *, settings=None, lightrag_client=None):
        self.session = session
        self.settings = settings or Settings()
        self.client = lightrag_client

    def tool(self, *, branch_id, model_version_id, known_assets=None):
        # 同任务恢复时从持久化工具结果恢复候选集合，不因入口是 Actor 或技能而放宽权限。
        known_assets = known_assets if known_assets is not None else set()

        def invoke(**kwargs):
            args = GetStyleExamplesArgs.model_validate(kwargs)
            branch = self.session.get(Branch, branch_id)
            if branch is None or branch.model_version_id != model_version_id:
                raise ToolServiceError("风格工具绑定已失效")
            snapshot = self.session.scalar(
                select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
            )
            if snapshot is None:
                raise ToolServiceError("分支没有可读历史快照")
            rows = source_rows(self.session, snapshot)
            imported = {m.id for m, _ in rows}
            history = ConversationHistory(self.session, branch_id, snapshot, snapshot.cutoff_at)
            coverage, next_cursor = {}, None
            if args.asset_ref:
                if args.asset_ref not in known_assets:
                    raise ToolInputError(
                        "asset_ref 不在已提供候选中；先按 situation、intent 查询，"
                        "使用返回的 sticker_candidates.asset_ref，不能猜 ID",
                        field="asset_ref",
                    )
                try:
                    validate_asset(self.session, branch, snapshot, args.asset_ref, self.settings)
                except ValueError as error:
                    raise ToolInputError(
                        f"asset_ref 素材不可用（{error}）；重新选择可用候选或改用文字",
                        field="asset_ref",
                    ) from error
                usages = [
                    m.id
                    for m, role in rows
                    if role == "target"
                    and m.kind == "sticker"
                    and m.media_asset_id == args.asset_ref
                ]
                if args.usage_cursor and args.usage_cursor not in usages:
                    raise ToolInputError(
                        "usage_cursor 不属于此素材；保持原 asset_ref 并原样使用 more_usage_cursor，"
                        "或省略游标重新查询",
                        field="usage_cursor",
                    )
                offset = usages.index(args.usage_cursor) + 1 if args.usage_cursor else 0
                hits = usages[offset : offset + args.limit]
                next_cursor = hits[-1] if hits and offset + len(hits) < len(usages) else None
                status = "ready" if hits else "empty"
            else:
                query, rag_ids, rag_status = _retrieval_query(args), [], "unavailable"
                if self.settings.lightrag_enabled:
                    from .routine_evidence import query_frozen_history

                    try:
                        result = query_frozen_history(
                            self.session,
                            snapshot,
                            query,
                            settings=self.settings,
                            client=self.client,
                            limit=args.limit,
                        )
                        rag_ids, rag_status = result["source_ids"], result["retrieval_status"]
                    except (ToolException, ToolServiceError):
                        pass  # 另一条检索仍可用，返回 partial 而不是伪装没有历史。
                usage = search_usage(self.session, branch_id, snapshot, query, args.limit)
                coverage = {"lightrag": rag_status, "usage_index": usage}
                hits = []
                # 交替两种召回，表情用法不会总被普通文字命中挤掉。
                for i in range(max(len(rag_ids), len(usage["source_ids"]))):
                    for source in (rag_ids, usage["source_ids"]):
                        if i < len(source) and source[i] in imported and source[i] not in hits:
                            hits.append(source[i])
                hits = hits[: args.limit]
                incomplete = rag_status not in {"ok", "empty"} or usage["status"] not in {
                    "ready",
                    "empty",
                }
                status = "partial" if incomplete else "ready" if hits else "empty"
            examples, candidates, used = [], {}, set()
            positions = {m.id: i for i, (m, _) in enumerate(rows)}
            for hit in hits:
                if hit not in positions or (hit in used and not args.asset_ref):
                    continue
                i = positions[hit]
                refs = [m.id for m, _ in rows[max(0, i - 4) : i + 5]]
                # 窗口不越过冻结边界，不把模型生成对话作为真人风格来源。
                messages = [history.read(around_ref=ref, limit=1)["messages"][0] for ref in refs]
                used.update(refs)
                examples.append(
                    {
                        "messages": messages,
                        "source_ids": refs,
                        "context_complete": i >= 4 and i + 5 <= len(rows),
                        "before_ref": refs[0],
                        "around_ref": hit,
                    }
                )
                for ref in refs:
                    m, role = rows[positions[ref]]
                    if role != "target" or m.kind != "sticker" or not m.media_asset_id:
                        continue
                    available = True
                    try:
                        validate_asset(
                            self.session, branch, snapshot, m.media_asset_id, self.settings
                        )
                    except ValueError:
                        available = False
                    item = candidates.setdefault(
                        m.media_asset_id,
                        {
                            "asset_ref": m.media_asset_id,
                            "available": available,
                            "used_by": "target",
                            "usage_refs": [],
                            "uses": [],
                        },
                    )
                    if ref not in item["usage_refs"]:
                        item["usage_refs"].append(ref)
                        item["uses"].append({"source_ref": ref, "around_ref": ref})
            known_assets.update(candidates)
            return {
                "tool_name": "get_style_examples",
                "retrieval_status": status,
                "source_ids": sorted(used),
                "coverage": coverage,
                "examples": examples,
                "sticker_candidates": list(candidates.values()),
                "more_usage_cursor": next_cursor,
            }

        def restore(result):
            # 仅恢复同检查点的工具回执；不能从模型声明的素材 ID 生成授权。
            from .submit_expression import supplied_assets

            known_assets.update(supplied_assets(result))

        return StructuredTool.from_function(
            invoke,
            name="get_style_examples",
            args_schema=GetStyleExamplesArgs,
            metadata={
                "required_permissions": ["runtime.read_style"],
                "restore_tool_result": restore,
            },
            description=(
                "只读查询相似互动的真实文字与表情用法。拿不准措辞或素材含义时，"
                "按当前互动、关系距离、情绪和表达目的查询；返回发送者、时间、前后原文及可用素材引用，"
                "不分析图片，不能提供图中文字或画面含义。可查询候选素材在不同上下文中的用法。"
                "more_usage_cursor 非空表示有下一页；索引未就绪或 partial 不等同没有历史。"
            ),
        )
