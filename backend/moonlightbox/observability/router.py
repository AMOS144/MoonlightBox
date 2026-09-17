"""从 Phoenix 读取完整 Agent Trace 的只读 API。

这里不建立镜像表或缓存副本。执行 ID 是 Controller 写入所有手工 Span 的关联键，查询时先
找到根 Span，再按 trace_id 取回 LangChain/LangGraph 自动 Span，因此页面和排障脚本都能
看到同一棵原始调用树。
"""

from __future__ import annotations

from time import monotonic, sleep

from fastapi import APIRouter, HTTPException
from httpx import HTTPStatusError

from moonlightbox.config import Settings


def create_observability_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/observability", tags=["observability"])

    @router.get("/agent-executions/{execution_id}")
    def get_agent_execution_trace(execution_id: str) -> dict[str, object]:
        """返回 Phoenix 原始 Span 树及根摘要；只读，不回写业务数据库。"""

        if not settings.phoenix_enabled:
            raise HTTPException(status_code=409, detail="Phoenix 未启用")
        base_url = settings.phoenix_collector_endpoint.removesuffix("/v1/traces")
        # Phoenix BatchSpanProcessor 会在根 Span 结束后异步 export。短暂轮询只处理这段
        # 可见性窗口；绝不把 Trace 镜像回业务数据库，也不会长时间占用 API Worker。
        deadline = monotonic() + 3.0
        root_spans: list[dict[str, object]] = []
        roots: list[dict[str, object]] = []
        while True:
            root_spans = _get_spans(
                base_url=base_url,
                project_name=settings.phoenix_project_name,
                attributes={"moonlightbox.agent.execution_id": execution_id},
            )
            roots = [item for item in root_spans if item.get("name") == "moonlightbox.agent.run"]
            if roots or monotonic() >= deadline:
                break
            sleep(0.1)
        if not roots:
            raise HTTPException(status_code=404, detail="没有找到对应的 Agent Phoenix Trace")
        root = roots[0]
        context = root.get("context")
        trace_id = context.get("trace_id") if isinstance(context, dict) else None
        all_spans = (
            _get_spans(
                base_url=base_url,
                project_name=settings.phoenix_project_name,
                trace_id=str(trace_id),
            )
            if isinstance(trace_id, str) and trace_id
            else root_spans
        )
        return {
            "execution_id": execution_id,
            "phoenix_base_url": base_url,
            "trace_id": trace_id,
            "root": root,
            "spans": all_spans,
        }

    @router.get("/agent-executions/{execution_id}/summary")
    def get_agent_execution_summary(execution_id: str) -> dict[str, object]:
        """从 Phoenix 根 Span 读取审核页所需的真实工具统计。

        不回写数据库、不读取或复制完整 Trace；``execution_id`` 只是一把关联键。
        这使“总调用数/失败数”与 Phoenix 调试页面看到的数字始终来自同一来源。
        """

        if not settings.phoenix_enabled:
            raise HTTPException(status_code=409, detail="Phoenix 未启用")
        base_url = settings.phoenix_collector_endpoint.removesuffix("/v1/traces")
        root = _wait_for_execution_root(
            base_url=base_url,
            project_name=settings.phoenix_project_name,
            execution_id=execution_id,
        )
        if root is None:
            raise HTTPException(status_code=404, detail="没有找到对应的 Agent Phoenix Trace")
        return _agent_execution_summary(root, execution_id=execution_id, phoenix_base_url=base_url)

    @router.get("/agent-owners/{owner_id}/latest-summary")
    def get_latest_owner_execution_summary(owner_id: str) -> dict[str, object]:
        """为旧栏目任务从 Phoenix 找回最后一次执行的真实统计。

        新任务会持久化 ``execution_id``；早于该改造的候选档案没有关联键时，稳定的
        PersonWorld owner id（``{agent_run_id}:{section}``）可作为只读回退查询条件。
        这仍然以 Phoenix 为唯一计数来源，不回填或伪造旧数据库的调用次数。
        """

        if not settings.phoenix_enabled:
            raise HTTPException(status_code=409, detail="Phoenix 未启用")
        base_url = settings.phoenix_collector_endpoint.removesuffix("/v1/traces")
        root = _wait_for_agent_root(
            base_url=base_url,
            project_name=settings.phoenix_project_name,
            attributes={"moonlightbox.owner.id": owner_id},
        )
        if root is None:
            raise HTTPException(status_code=404, detail="没有找到对应的 Agent Phoenix Trace")
        attributes = root.get("attributes")
        values = attributes if isinstance(attributes, dict) else {}
        execution_id = values.get("moonlightbox.agent.execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise HTTPException(status_code=502, detail="Phoenix 根 Span 缺少 execution_id")
        return _agent_execution_summary(root, execution_id=execution_id, phoenix_base_url=base_url)

    @router.get("/runtime-cycles/{cycle_id}")
    def get_runtime_cycle(cycle_id: str) -> dict[str, object]:
        """查询 Cycle 及独立 Agent 树，附带原始输入、输出、工具与压缩快照。"""
        from .investigation import build_investigation

        if not settings.phoenix_enabled:
            raise HTTPException(status_code=409, detail="Phoenix 未启用")
        base_url = settings.phoenix_collector_endpoint.removesuffix("/v1/traces")
        found = _get_spans(
            base_url=base_url,
            project_name=settings.phoenix_project_name,
            attributes={"moonlightbox.cycle.id": cycle_id},
        )
        # 兼容改造前的 Agent owner 关联；不伪造历史 Cycle 的父子关系。
        found += _get_spans(
            base_url=base_url,
            project_name=settings.phoenix_project_name,
            attributes={"moonlightbox.owner.id": cycle_id},
        )
        if not found:
            raise HTTPException(status_code=404, detail="Phoenix 中没有对应 Cycle")
        trace_ids = {s.get("context", {}).get("trace_id") for s in found}
        for trace_id in trace_ids - {None}:
            found += _get_spans(
                base_url=base_url, project_name=settings.phoenix_project_name, trace_id=trace_id
            )
        return {"cycle_id": cycle_id, **build_investigation(found)}

    @router.get("/agent-executions/{execution_id}/investigation")
    def investigate_agent(execution_id: str) -> dict[str, object]:
        from .investigation import build_investigation

        data = get_agent_execution_trace(execution_id)
        return {"execution_id": execution_id, **build_investigation(data["spans"])}

    return router


def _get_spans(
    *,
    base_url: str,
    project_name: str,
    attributes: dict[str, str] | None = None,
    trace_id: str | None = None,
) -> list[dict[str, object]]:
    """使用 Phoenix 官方 Client 查询，避免复制服务端筛选参数的编码规则。"""

    try:
        from phoenix.client import Client as PhoenixClient
    except ModuleNotFoundError as error:
        # arize-phoenix 服务端是 observability extra 的可选依赖；未安装时（如
        # All-in-One 镜像）所有端点已在 phoenix_enabled 检查处拦截，不会走到这里。
        raise HTTPException(status_code=409, detail="Phoenix 组件未安装") from error

    try:
        client = PhoenixClient(base_url=base_url)
        spans = client.spans.get_spans(
            project_identifier=project_name,
            attributes=attributes,
            trace_ids=[trace_id] if trace_id else None,
            limit=20001,
        )
        if len(spans) > 20000:
            raise HTTPException(
                status_code=413, detail="Trace 超过 20000 个 Span，请缩小查询范围；未返回截断结果"
            )
    except HTTPException:
        raise
    except HTTPStatusError as error:
        if error.response.status_code == 404:
            # 项目尚未由第一条 Span 创建，或者 Batch exporter 仍在发送；调用方会在
            # 有界轮询后决定是否返回 404。
            return []
        raise HTTPException(
            status_code=502,
            detail=f"Phoenix 查询失败：HTTP {error.response.status_code}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Phoenix 查询失败：{type(error).__name__}",
        ) from error
    return [dict(item) for item in spans if isinstance(item, dict)]


def _wait_for_execution_root(
    *,
    base_url: str,
    project_name: str,
    execution_id: str,
) -> dict[str, object] | None:
    """等待 BatchSpanProcessor 的短暂导出窗口，并只返回唯一的根 Agent Span。"""

    return _wait_for_agent_root(
        base_url=base_url,
        project_name=project_name,
        attributes={"moonlightbox.agent.execution_id": execution_id},
    )


def _wait_for_agent_root(
    *,
    base_url: str,
    project_name: str,
    attributes: dict[str, str],
) -> dict[str, object] | None:
    """以稳定 Span 属性查询根 Agent Span；同一 owner 取最新的一次。"""

    deadline = monotonic() + 3.0
    while True:
        spans = _get_spans(
            base_url=base_url,
            project_name=project_name,
            attributes=attributes,
        )
        roots = [item for item in spans if item.get("name") == "moonlightbox.agent.run"]
        if roots:
            return max(roots, key=lambda item: str(item.get("start_time") or ""))
        if monotonic() >= deadline:
            return None
        sleep(0.1)


def _agent_execution_summary(
    root: dict[str, object],
    *,
    execution_id: str,
    phoenix_base_url: str,
) -> dict[str, object]:
    """投影根 Span 的确定性计数，缺失字段按零而不是猜测处理。"""

    attributes = root.get("attributes")
    values = attributes if isinstance(attributes, dict) else {}
    context = root.get("context")
    trace_context = context if isinstance(context, dict) else {}
    return {
        "execution_id": execution_id,
        "phoenix_base_url": phoenix_base_url,
        "trace_id": trace_context.get("trace_id"),
        "terminal_reason": values.get("moonlightbox.agent.terminal_reason"),
        "tool_call_count": _non_negative_int(values.get("moonlightbox.agent.tool_calls")),
        "tool_success_count": _non_negative_int(
            values.get("moonlightbox.agent.tool_success_calls")
        ),
        "tool_error_count": _non_negative_int(values.get("moonlightbox.agent.tool_error_calls")),
        "tool_empty_count": _non_negative_int(values.get("moonlightbox.agent.tool_empty_calls")),
        "tool_cancelled_count": _non_negative_int(
            values.get("moonlightbox.agent.tool_cancelled_calls")
        ),
    }


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(value, 0)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0
