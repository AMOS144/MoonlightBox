# 统一 Agent 普通重启恢复

## 范围

本次补齐的是执行存档，不是训练 checkpoint，也不另建 Trace 系统。
Director、DayPlan、PersonaActor 和生活事件仍沿用 Runtime 同级协作作用域；
PersonWorld 七栏目、Revision 和独立对话摘要通过同一个 AgentLoopController
接入文件检查点。Phoenix 继续负责调用观测，不承担恢复数据源的职责。

## 统一入口

- `agent_runtime/persistence.py` 管理检查点位置和 SqliteSaver。
- 文件仍叫 `runtime-collaboration.sqlite`，位于业务数据库旁，避免换名丢失现有 Runtime 存档。
- 独立调用通过 `AgentExecutionRequest.checkpoint_path` 接入；已有 Runtime 作用域优先。
- 独立调用身份包含 owner 类型、owner ID、输入版本及数据读取范围；Prompt 版本继续参与原有命名。
- 同一任务重启复用身份，新任务或新用户输入版本不会复用旧问答。
- 当前部署使用 SQLite/Linux；不支持的数据库明确报错，不悄悄退回内存。
- 每个栏目独立连接，仅初始化 WAL 和表结构时使用 Linux 文件锁，模型调用仍可并行。

## 节点保存和恢复

持久化循环使用 LangGraph `durability="sync"`：保存上一个节点后才执行下一节点。
否则进程突然退出时，已经完成的工具批次可能尚未写入异步存档。
恢复初始输入节点时也保留完整状态，不能只更新消息而丢掉预算等字段。

`AgentSpec.snapshot_work_state` / `restore_work_state` 提供领域工作状态桥接：
只接收可序列化数据，不保存数据库 Session、模型客户端或回调。
模型节点和工具批次的结果与工作材料一起保存。

PersonWorld 保存并恢复检索引用、原始消息集合及其编号、模块读取依赖；
重新构造工具闭包后，旧 `retrieval_id`、`evidence_set_id` 仍指向原材料。
Revision 还保存最后一次模型实际看到的装配上下文及快照字段。
恢复已接受结果时不必再次调用模型来重建快照；业务快照持久化校验 hash 并幂等复用。

`waiting_for_user` 是已经接受的一种交付，不是失败：普通恢复不会重复提问或消耗重试次数。
原有失败尝试预算和累计资源计数保持不变，本次没有增加新的调用限制。

## 业务流程接线

PersonWorld 编译 Job 以自身 ID 作为 `resume_key`，复用同一调查实例。
Coordinator 仍负责确定性调度，不成为新的认知 Agent。
原有业务状态保留固定基线、每批输入、模块快照、已接受栏目和阶段完成标记：

- 已接收完成的栏目跳过，不重新执行；
- Agent 已提交但 Coordinator 尚未接收时，从该 Agent 的成功存档接收结果；
- 尚未完成的栏目通过统一循环继续工具调查；
- 草稿阶段和补全阶段隔离，不能将草稿结果误当作补全结果；
- 手动栏目重试按新 Job 隔离；同一重试 Job 重启仍接着原调查；
- 来源正文在 LangGraph 工作存档，业务表仅额外维护必要的来源 ID 与阶段信息。

Revision 的 owner 为纠正会话，输入版本变化后进入新作用域。原有确认、Patch 审核、
Graph Executor 授权和事务保护不变；恢复本身不会批准或发布任何图谱修改。
对话摘要任务也使用统一检查点入口，已提交但未写业务摘要时可复用模型产物。

## 边界与验证

恢复的是最后落盘的节点，不是供应商正在生成的半个请求。未返回的请求需重新发出；
中途退出的只读工具批次可能重做，正式业务写入仍依赖 Executor 的事务与幂等保护。
Worker 沿用已有中断任务回收：异常退出后须等原租约过期，不能抢占活跃 Worker。
未传持久化位置的纯离线调用仍不承诺跨进程恢复。

验证包含真正子进程 `os._exit` 后重启、材料与编号恢复、成功/等待用户结果重放、
七栏草稿完成后中断再继续补全、Revision 快照幂等，以及原 Runtime 恢复回归。
未对体验环境执行停机或重启，未发起真实供应商调用。

本次不引入自动工具契约指纹/迁移框架；不兼容代码升级仍需显式升级执行版本。

### 本次验证记录

- 统一循环、跨进程恢复、PersonWorld v3/Revision/退役边界、Runtime 同级协作及生活事件：
  123 项通过；排除 1 项已有 Persona 画像上下文断言用例。
- Ruff 检查、格式检查及 `git diff --check` 通过（检查本次修改的运行代码）。
- 扩展世界构建回归中，`test_import_builds_world_before_enqueuing_node_analysis` 仍失败：
  其 FakeCompiler 返回普通文本而非 `submit_section`，已不符合当前提交契约。
  没有为兼容该旧假模型重新开放文本交付。
- 一次扩展回归出现时间边界用例失败，单独复跑和最后完整恢复回归均通过；未修改虚拟钟逻辑。
