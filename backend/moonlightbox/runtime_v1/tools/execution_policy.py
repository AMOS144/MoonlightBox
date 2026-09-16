"""Runtime 工具资源审查清单，包含结果投影的分页缓存写入。"""

from moonlightbox.agent_runtime.tool_execution import ToolExecutionPolicy, serial

_CACHE = "runtime_result_cache"
_SQL = "sqlalchemy_session"

POLICIES = {
    "send_agent_message": serial(
        "branch_outbox", _CACHE, reason="独立短事务写入幂等消息；不并行、不强制中断事务"
    ),
    "read_skill": serial(_CACHE, reason="读取冻结技能资源；统一结果投影写入分页缓存"),
    "search_memory": serial(
        _SQL, _CACHE, "lightrag_client", reason="合并 SQL 记忆与图谱读取；共享会话和分页缓存"
    ),
    "get_style_examples": serial(
        _SQL,
        _CACHE,
        "lightrag_client",
        "asset_authorizations",
        reason="检索表达、读取素材并更新候选授权集合",
    ),
    "search_conversation": serial(_SQL, _CACHE, reason="历史搜索读取同一 SQL Session，并缓存结果"),
    "read_conversation": serial(_SQL, _CACHE, reason="连续消息与媒体读取共享 Session"),
    "get_recent_life_events": serial(_SQL, _CACHE, reason="生活事件账本读取共享 Session"),
    "search_plan_memory": serial(_SQL, _CACHE, reason="规划记忆读取共享 Session"),
    "analyze_routine_evidence": serial(
        _SQL,
        _CACHE,
        "lightrag_client",
        reason="SQL 和图谱调查共享状态；图谱请求前传递 deadline 并检查取消",
        cooperative_io=True,
    ),
    "get_subjective_state": serial(_CACHE, reason="工具体读取冻结上下文，但统一投影会写分页缓存"),
    "get_profile_section": serial(_CACHE, reason="工具体读取分支画像投影，但统一投影会写分页缓存"),
    "get_plan_constraints": serial(_CACHE, reason="工具体读取已装配约束，但统一投影会写分页缓存"),
    "read_runtime_result": serial(_CACHE, reason="读取执行内缓存，不能与缓存写入并发"),
    "update_plan_work": serial(
        "planner_work", _CACHE, reason="完整替换规划草稿，投影阶段也更新草稿；不能并发覆盖"
    ),
}


def execution_policy(name):
    # 扩展工具没有审查结论时必须保守串行，不能靠 read_only 标记推导并行资格。
    return POLICIES.get(name, ToolExecutionPolicy())
