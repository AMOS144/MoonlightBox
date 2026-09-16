# Director / DayPlan 共用消息工具与可靠交付

工作目录：`.worktrees/runtime-prune`。本记录描述 2026-09-15 的实现，覆盖旧说明中
“只靠当前协作图内 messages 沟通”的限制。不增加 A2A 服务、数据表或第二套 Agent 循环。

## 模型接口

Director、DayPlan 日常规划和 DayPlan 生活事件模式都注册 `send_agent_message`。
它由 LangChain 原生工具绑定，只收 `recipient`（director / day_planner）和 `content`。
分支、发送者、任务身份、虚拟时间、幂等键由宿主绑定。不能给自己发消息、跨分支发消息，
也不能通过消息工具修改日程或给用户发聊天。同一任务内相同正文和收件人复用同一回执。

参数错误进入统一 Controller 的可修复工具错误回执；取消沿用 Controller/Job 信号。
工具使用独立短事务，不提交调用者尚未校验的状态。它属于 communication 写操作，
明确串行执行，不能伪装为只读工具套用只读传输重试；事务不做强制线程中断。

`queued` 表示消息持久化，不表示对方已读、赞同、已执行，也不是计划发布回执。

## 收件与调度

复用 RuntimeEvent：给 Director 的消息类型是 `agent_message`，给 DayPlan 的是
`planner_inbox`。Director 的初次领取与运行中补入都排除 planner_inbox，避免拿错收件箱。

现有 3 秒扫描按收件人调度：

- Director 沿用 input tick，和用户消息一起理解；新输入不取消在途推理。
- DayPlan 将全部尚未分配的到期消息合并为一个现有 planner_task Job。
- 已有运行/排队 Planner 时继续积累，不为每条消息启动模型；在途任务不被新消息打断。
- 已分配消息的失败恢复仍由原 Job 管理，不靠扫描创建新任务绕过失败预算。
- Planner 成功处理后，输入消费标记和对外交付在同一事务中完成。
- 只有一个待答调查时，来信续接该调查检查点；多个待答调查不猜测对应关系。
  需要精确对应时仍可使用已有带 request_ref 的协作答复契约。

没有嵌套调用对方模型，也没有让消息发送工具等待对方完成。

## 上下文与“无需修改”

双方公开发言保存在 RuntimeEvent 的 discussion 中。后续任务自动装配已处理发言和
自己已发出的消息，再合并本次已领取输入。尚未领取的对方来信不越过输入边界提前暴露。
私有工具调用、思考不跨 Agent 传播；这些沟通记录不是人物事实，不自动写成 MemoryRecord。
既有 Runtime 工具分页、Controller 容量保护和规划工作笔记继续生效。

DayPlan 提交结果现在有三个互斥出口：

- proposal：完整计划提案，交 Executor 校验并发布。
- reply：需要协作的具体问题或答复。answer 不再被误标为还在等回答。
- completion：仅收件任务允许，解释已处理且无需改计划；不自动回信，不改计划版本。

初始计划准备不能用 completion 冒充成功。纯通知处理失败也不能把原来有效计划标成失败。
需要等待意见时保留现有 reply / 生活协商结果契约；不要同时用工具和结果重复发送同一问题。

## 维护通知不依赖模型记性

例行计划维护、分支准备的计划提交成功时，Executor 在**同一个计划事务**中写入 Director
通知，包含日期、实际计划版本和提交键。回滚计划也回滚通知；重放提交键不会重复通知。
通知随之后的 tick 被读取，不在维护工作中直接调用 Director。

普通协作计划任务继续使用已有可恢复交付：图检查点保存提交回执，然后持久化通知。
中间重启复用已完成图状态，不重新生成或重复提交计划。纯 completion 不创建空通知，
避免“收到—确认收到—再确认”互相唤醒。

## 文件与验证

- `runtime_v1/tools/send_agent_message.py`：共用 Schema、身份绑定、发送事务和回执。
- `runtime_v1/collaboration/transport.py`：消息入队、批处理调度和公开沟通装配。
- `collaboration/nodes.py`、`planner_tasks.py`：接入现有图、无修改完成、维护可靠通知。
- `jobs.py`：现有扫描按收件人分发；无需数据库迁移。
- 三份 Director / DayPlan Prompt：仅说明协作边界，不重复工具 JSON Schema。

验证使用真实 SQLite、真实 LangGraph 和离线原生工具模型替身；覆盖发送幂等、禁止自发、
批量收件、收件隔离、无修改完成、无回信循环、检查点恢复和维护通知。
没有调用真实供应商，没有修改体验分支，也没有开启概率生活事件开关。

范围说明：本次完成消息交付，不调整生活机会的概率政策，不新增事件后续生命周期。
长历史的语义摘要策略仍沿用现有 Harness，不能将本次接入描述成新增了无限上下文记忆。
