"""PersonaActor 的只读表达风格工具。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.tools import StructuredTool
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import IdentityKernel
from moonlightbox.branches.history import BranchHistoryService
from moonlightbox.branches.models import Branch
from moonlightbox.imports.wechat_rendering import normalize_wechat_display

from .config import TOOL_DESCRIPTIONS, TOOL_SCHEMAS
from .schemas import GetStyleExamplesArgs


class StyleService:
    """只暴露冻结历史中的真实表达示例，绝不从本轮模型输出反向学习口吻。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def tool(self, *, branch_id: str, model_version_id: str) -> StructuredTool:
        """把身份绑定在代码闭包中，模型只能选择意图和数量。"""

        def invoke(**kwargs: Any) -> dict[str, Any]:
            safe_kwargs = {key: value for key, value in kwargs.items() if key != "model_version_id"}
            args = GetStyleExamplesArgs.model_validate(
                {"model_version_id": model_version_id, **safe_kwargs}
            )
            branch = self.session.get(Branch, branch_id)
            if branch is None or branch.model_version_id != model_version_id:
                return {
                    "tool_name": "get_style_examples",
                    "scope": "world",
                    "as_of": datetime.now(UTC).isoformat(),
                    "source_ids": [],
                    "truncated": False,
                    "data": {"style_profile": {}, "examples": []},
                }
            kernel = self.session.scalar(
                select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
            )
            profile = (
                kernel.content.get("style_profile", {})
                if kernel is not None and isinstance(kernel.content, dict)
                else {}
            )
            examples = _authentic_style_examples(self.session, branch, args.intent, args.limit)
            return {
                "tool_name": "get_style_examples",
                "model_version_id": model_version_id,
                "scope": "world",
                "as_of": datetime.now(UTC).isoformat(),
                "truncated": False,
                "source_ids": _example_source_ids(examples),
                "data": {
                    "style_profile": profile,
                    "examples": examples,
                },
            }

        return StructuredTool.from_function(
            name="get_style_examples",
            description=TOOL_DESCRIPTIONS["get_style_examples"],
            func=invoke,
            args_schema=cast(Any, TOOL_SCHEMAS["get_style_examples"]),
        )


def _authentic_style_examples(
    session: Session,
    branch: Branch,
    intent: str,
    limit: int,
) -> list[dict[str, object]]:
    """从分支冻结边界之前的真人对话选取少量 target 回复示例。"""

    rows = BranchHistoryService(session).authentic_example_rows(branch, message_limit=500)
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    current_role: str | None = None
    current: list[tuple[str, str]] = []
    for message, role in rows:
        kind, content = normalize_wechat_display(message.kind, message.content)
        text = " ".join(content.split()).strip()
        if role not in {"self", "target"} or kind != "text" or not text:
            continue
        if role != current_role:
            if current_role is not None and current:
                groups.append((current_role, current))
            current_role, current = role, []
        current.append((message.id, text))
    if current_role is not None and current:
        groups.append((current_role, current))

    query_chars = set(intent.lower())
    candidates: list[tuple[int, dict[str, object]]] = []
    for index, (role, target_turn) in enumerate(groups):
        if role != "target" or index == 0 or groups[index - 1][0] != "self":
            continue
        prompt_turn = groups[index - 1][1]
        target_text = " / ".join(text for _, text in target_turn[:5])
        prompt_text = " / ".join(text for _, text in prompt_turn[-4:])
        score = len(query_chars & set((prompt_text + target_text).lower()))
        candidates.append(
            (
                score,
                {
                    "source_ids": [
                        *[message_id for message_id, _ in prompt_turn[-4:]],
                        *[message_id for message_id, _ in target_turn[:5]],
                    ],
                    "prompt": prompt_text,
                    "text": target_text,
                },
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in candidates[:limit]]


def _example_source_ids(examples: list[dict[str, object]]) -> list[str]:
    source_ids: list[str] = []
    for item in examples:
        raw = item.get("source_ids")
        if isinstance(raw, list):
            source_ids.extend(source_id for source_id in raw if isinstance(source_id, str))
    return source_ids
