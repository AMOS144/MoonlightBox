# 工具取消、并行资格与资源审查

## 实际行为与契约

ToolContract.execution 声明 cancellation、parallelism、timeout_enforcement、
exclusive_resources 和中文 reason。声明只给后端调度与 Phoenix，不作为模型的工具参数。

**当前 Controller 仍串行执行全部工具。** parallel_safe 只是经审查的资格，不是线程池开关。
这次没有引入异步 HTTP、强制终止线程、中断 SQL 或跨进程锁。

- boundaries：仅在调用前后检查取消/输入版本；阻塞中的同步调用不能立刻停下。
- cooperative：工具内部也有取消检查点；仍不能强杀任意同步代码。
- boundary_only：外层超时后丢弃迟到结果，不保证阻塞工具准时返回。
- cooperative_io：配合的网络路径传递剩余超时，不是任意网络或数据库操作的硬截止。
- serial：共享资源未隔离，保持串行。
- parallel_safe：工具体读取固定定义或冻结数据；未来并行还须保证记录与结果合并安全。

未知扩展工具默认 boundaries、serial，独占 execution_state，并标明“未审查”。
不能从 read_only 自动推断线程安全。枚举非法、空理由、parallel_safe 同时占用独占资源都会报错。

## 独占资源如何理解

资源名表示调用方绑定的**实际对象类别**，不是整个项目的全局锁名。

| 资源 | 实际保护对象 |
| --- | --- |
| sqlalchemy_session | 闭包共享的 Session、事务及 identity map |
| runtime_result_cache | RuntimeToolbox.results，结果投影也会写入 |
| investigation_artifacts | 检索材料与原文证据工件存储 |
| module_read_tracker | 模块读取时更新的依赖记录 |
| planner_work | PlanningToolbox.work 规划草稿 |
| asset_authorizations | known_assets、素材候选与提交授权状态 |
| lightrag_client | 绑定的 Sidecar 客户端与同步请求路径 |
| submission_receipt | 当前提交作用域中的 typed 接受回执 |
| execution_state | 执行期共享状态、未知校验回调等保守预留 |

串行循环保证同一批工具不会并发操作这些对象。不同 Agent/进程的隔离仍由调用方负责；
这些字符串不会自动给共享 Session 加锁。未来并行调度必须绑定实际资源实例或稳定资源键，
为数据库查询配置独立 Session，并在主执行线程合并工件、依赖、草稿与工具结果。

## Runtime 工具清单

| 工具 | 取消检查 | 并行资格 | 独占资源 | 依据 |
| --- | --- | --- | --- | --- |
| `search_memory` | 仅调用边界 | 串行 | sqlalchemy_session、runtime_result_cache、lightrag_client | 合并 SQL 记忆与图谱读取；共享会话和分页缓存 |
| `get_style_examples` | 仅调用边界 | 串行 | sqlalchemy_session、runtime_result_cache、lightrag_client、asset_authorizations | 检索表达、读取素材并更新候选授权集合 |
| `search_conversation` | 仅调用边界 | 串行 | sqlalchemy_session、runtime_result_cache | 历史搜索读取同一 SQL Session，并缓存结果 |
| `read_conversation` | 仅调用边界 | 串行 | sqlalchemy_session、runtime_result_cache | 连续消息与媒体读取共享 Session |
| `get_recent_life_events` | 仅调用边界 | 串行 | sqlalchemy_session、runtime_result_cache | 生活事件账本读取共享 Session |
| `search_plan_memory` | 仅调用边界 | 串行 | sqlalchemy_session、runtime_result_cache | 规划记忆读取共享 Session |
| `analyze_routine_evidence` | 内部检查点＋边界 | 串行 | sqlalchemy_session、runtime_result_cache、lightrag_client | SQL 和图谱调查共享状态；图谱请求前传递 deadline 并检查取消 |
| `get_subjective_state` | 仅调用边界 | 串行 | runtime_result_cache | 工具体读取冻结上下文，但统一投影会写分页缓存 |
| `get_profile_section` | 仅调用边界 | 串行 | runtime_result_cache | 工具体读取分支画像投影，但统一投影会写分页缓存 |
| `get_plan_constraints` | 仅调用边界 | 串行 | runtime_result_cache | 工具体读取已装配约束，但统一投影会写分页缓存 |
| `read_runtime_result` | 仅调用边界 | 串行 | runtime_result_cache | 读取执行内缓存，不能与缓存写入并发 |
| `update_plan_work` | 仅调用边界 | 串行 | planner_work、runtime_result_cache | 完整替换规划草稿，投影阶段也更新草稿；不能并发覆盖 |

必须审查工具结果投影，不只审查函数体。即使约束/画像工具只读取内存，
RuntimeToolbox.project 也会写分页缓存，故当前仍声明串行。

analyze_routine_evidence 已在图谱请求前、返回后和逐条恢复引用前检查取消；
它自建的客户端使用 bounded_timeout。注入的外部客户端仍必须自行遵守超时，
不能据此声称所有 SQL 或外部请求都可立即取消。

update_plan_work 改为 side_effect=proposal：它会替换工作草稿，不是纯查询。
统一重试层不再自动重放它；正式日程依然由 Executor 保存。

## PersonWorld 与 Revision 工具清单

| 工具 | 取消检查 | 并行资格 | 独占资源 | 依据 |
| --- | --- | --- | --- | --- |
| `search_world` | 仅调用边界 | 串行 | lightrag_client、investigation_artifacts | 图谱网络请求返回后写入检索工件，不能与证据恢复并发 |
| `list_graph_entities` | 仅调用边界 | 串行 | lightrag_client | 同步共享 Sidecar 客户端，未验证独立客户端及取消隔离 |
| `get_graph_entity` | 仅调用边界 | 串行 | lightrag_client | 同步共享 Sidecar 客户端，未验证并行隔离 |
| `get_entity_neighborhood` | 仅调用边界 | 串行 | lightrag_client | 同步共享 Sidecar 客户端，未验证并行隔离 |
| `get_relation` | 仅调用边界 | 串行 | lightrag_client | 同步共享 Sidecar 客户端，未验证并行隔离 |
| `locate_source_messages` | 仅调用边界 | 串行 | sqlalchemy_session、investigation_artifacts | SQL 恢复原文，并写入共享证据工件 |
| `get_message_context` | 仅调用边界 | 串行 | sqlalchemy_session、investigation_artifacts | SQL 扩展消息窗口，并写入共享证据工件 |
| `read_evidence_page` | 仅调用边界 | 串行 | investigation_artifacts | 读取可变工件存储，不与新增证据并发 |
| `analyze_evidence_dates` | 仅调用边界 | 串行 | sqlalchemy_session | SQL 读取指定消息时间并统计分布，共享 Session |
| `list_context_modules` | 仅调用边界 | 串行 | module_read_tracker | 列目录同时记录模块依赖，不是纯只读 |
| `read_context_module` | 仅调用边界 | 串行 | module_read_tracker | 读取字段同时记录模块依赖，不是纯只读 |
| `get_context_module_spec` | 仅调用边界 | 具备资格，尚未启用 | 无 | 只读取模块字段定义并生成返回值；无 Session、依赖追踪或工件写入 |
| `get_current_profile_section` | 仅调用边界 | 串行 | sqlalchemy_session | 标准实现读取 Publication/Profile，共享 SQL Session |
| `get_active_corrections` | 仅调用边界 | 串行 | sqlalchemy_session | 读取当前有效纠正，共享 SQL Session |

### 栏目专用冻结实现

| 工具 | 取消检查 | 并行资格 | 独占资源 | 依据 |
| --- | --- | --- | --- | --- |
| `get_current_profile_section（栏目专用）` | 仅调用边界 | 具备资格，尚未启用 | 无 | 栏目专用实现只读取本回合冻结栏目与摘要，不访问 SQL，不记录依赖 |

section_agent_v3 显式选择 frozen_section=True，Revision 使用标准 SQL 声明；
相同名字不代表相同的资源使用方式。以上两类“具备资格”工具目前同样串行执行。

## 所有提交工具

全部取消能力为 boundaries，全部必须单独串行执行。共同独占
submission_receipt、execution_state、sqlalchemy_session。
通用策略保守考虑注入的校验回调，不能因为某个回调暂不访问 SQL 就自动放开并行。

| 提交工具 | 额外独占资源/用途 |
| --- | --- |
| submit_decision | 校验消息引用与本轮状态 |
| submit_day_plan | 额外 planner_work；验收规划提案 |
| submit_expression | 额外 asset_authorizations、runtime_result_cache |
| submit_life_result | 验收生活事件模式的提案 |
| submit_section | 额外 investigation_artifacts、module_read_tracker |
| submit_revision_turn | 交付纠正回合或等待用户，不批准图谱写入 |
| submit_summary | 交付摘要，保存由服务负责 |
| submit_recovered_plan | 恢复脚本中调用 Executor 提案检查 |
| 扩展提交工具 | 未知 validator 保守使用共同独占资源 |

accepted 不等于 committed；取消不能自动回滚工具内部已经发生的内存状态修改，
也不能替代 Executor 的版本和事务保护。

## 维护位置与观察

- 通用类型：backend/moonlightbox/agent_runtime/tool_execution.py。
- Runtime 审查表：backend/moonlightbox/runtime_v1/tools/execution_policy.py。
- 人物世界审查表：backend/moonlightbox/world/person_world/tools/execution_policy.py。
- 提交工具：agent_runtime/submission.py 自动绑定 submission_policy(name)。

模型工具仍由 LangChain bind_tools 注册，没有新增另一个工具分发注册器。
新增工具要更新其领域审查表；回归测试检查当前 Prompt 声明的工具是否全部覆盖。

Phoenix 在每次 Tool Span 的 moonlightbox.tool 下记录 cancellation、
parallelism、timeout_enforcement、exclusive_resources、execution_policy_reason，
另以 dispatch=serial 标明本次实际调度方式，避免把“可并行”误读成“已并行”。

## 验证

260 项相关回归通过；1 项此前已知的 Persona 上下文取舍测试未纳入。
新增测试覆盖工具清单、同名冻结/SQL 实现、独占资源与并行声明冲突、
状态提案禁止并行、规划草稿不作为只读请求自动重放，以及网络边界取消检查。
本次未重启体验服务，未向分支发送测试消息。
