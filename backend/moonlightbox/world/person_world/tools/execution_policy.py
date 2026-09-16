"""PersonWorld/Revision 工具执行清单；同名工具的冻结投影和 SQL 实现分开声明。"""

from moonlightbox.agent_runtime.tool_execution import ToolExecutionPolicy, serial

_SQL = "sqlalchemy_session"
_ART = "investigation_artifacts"
_RAG = "lightrag_client"

POLICIES = {
    "save_section_work": serial(_ART, reason="保存本栏目工作笔记，与工具回执工件共享检查点"),
    "search_world": serial(_RAG, _ART, reason="图谱网络请求返回后写入检索工件，不能与证据恢复并发"),
    "list_graph_entities": serial(
        _RAG, reason="同步共享 Sidecar 客户端，未验证独立客户端及取消隔离"
    ),
    "get_graph_entity": serial(_RAG, reason="同步共享 Sidecar 客户端，未验证并行隔离"),
    "get_entity_neighborhood": serial(_RAG, reason="同步共享 Sidecar 客户端，未验证并行隔离"),
    "get_relation": serial(_RAG, reason="同步共享 Sidecar 客户端，未验证并行隔离"),
    "locate_source_messages": serial(_SQL, _ART, reason="SQL 恢复原文，并写入共享证据工件"),
    "get_message_context": serial(_SQL, _ART, reason="SQL 扩展消息窗口，并写入共享证据工件"),
    "read_evidence_page": serial(_ART, reason="读取可变工件存储，不与新增证据并发"),
    "analyze_evidence_dates": serial(_SQL, reason="SQL 读取指定消息时间并统计分布，共享 Session"),
    "list_context_modules": serial(
        "module_read_tracker", reason="列目录同时记录模块依赖，不是纯只读"
    ),
    "read_context_module": serial(
        "module_read_tracker", reason="读取字段同时记录模块依赖，不是纯只读"
    ),
    "get_context_module_spec": ToolExecutionPolicy(
        parallelism="parallel_safe",
        exclusive_resources=(),
        reason="只读取模块字段定义并生成返回值；无 Session、依赖追踪或工件写入",
    ),
    "get_current_profile_section": serial(
        _SQL, reason="标准实现读取 Publication/Profile，共享 SQL Session"
    ),
    "get_active_corrections": serial(_SQL, reason="读取当前有效纠正，共享 SQL Session"),
}


def execution_policy(name, *, frozen_section=False):
    if name == "get_current_profile_section" and frozen_section:
        return ToolExecutionPolicy(
            parallelism="parallel_safe",
            exclusive_resources=(),
            reason="栏目专用实现只读取本回合完整冻结快照，不访问 SQL，不记录依赖",
        )
    return POLICIES.get(name, ToolExecutionPolicy())
