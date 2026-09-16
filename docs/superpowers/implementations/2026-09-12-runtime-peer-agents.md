# Runtime 同级协作实现说明

本次工作目录：`.worktrees/runtime-prune`。对应设计为
[同级 Agent 与共享 State](../specs/2026-09-12-runtime-peer-agents-shared-state-design.md)。

## 入口和职责

- `runtime_v1/service.py`：HTTP/Job 门面，领取事件、准备输入和记录业务完成状态。
- `runtime_v1/collaboration/graph.py`：LangGraph 平级注册 Director、DayPlan、PersonaActor、Executor。
- `collaboration/nodes.py`：把领域 Agent 与 Executor 接到图中；不新增模型/工具循环。
- `collaboration/state.py`、`messages.py`、`routing.py`：共享状态、只追加消息、候选/回执/任务投影。
- `collaboration/plans.py`：按分支本地日期读取昨天、今天、明天的完整已提交计划。
- `collaboration/persistence.py`：官方 SqliteSaver 和 Linux 本机分支单写者锁。
- `agent_runtime/persistence.py`：可选内部 Harness 检查点作用域；不新增 AgentRun/Step/ToolInvocation 账本。

Director 仍使用 `prompts/director.py`。DayPlan 和 PersonaActor 的定义分别移动到
`agents/day_planner.md`、`agents/persona_actor.md`，通过 `agent_catalog.py` 加载。
`subagent_catalog.py` 仅保留历史 Python 导入别名，不包含独立注册逻辑或旧 Prompt。
`tools/delegate_day_plan.py` 已删除；Director 不再注册 `request_day_plan` 工具，也不再输出
`plan_proposal_id` 去批准 Planner 候选。

## 实际执行路径

普通聊天先进入 Director。三天计划预加载，不需要向 DayPlan 查询是否有空或有无冲突。
Director 的 `plan_request` 才会触发同级交接；只返回表达决策时直接进入 PersonaActor。

DayPlan 可以交付完整 proposal，也可以返回 clarification/objection；Director 用
`peer_reply` 回答，或先向用户提问。共享记录及未处理请求随分支检查点保留。
候选进入 Executor 才可能生效；提交后刷新共享计划，Director 再决定如何表达。
只追加的候选投影保留历史版本，不能用后来的提案覆盖已经确认的回执。

初始化直接进入 DayPlan，暂停虚拟钟，完成当天计划并初始化状态后才恢复时钟并 ready。
需要外部信息、模型失败或背景装载失败时保留准备失败状态，不误报 ready。
前端沿用准备页和微信式聊天，不展示内部协商消息。准备失败可从原页面显式重试。

次日维护由 Scheduler 直接发起，不经过 Director 审批、不主动发送聊天消息。
昨天只读；明天的计划不得取消今天的 Wakeup 或推进今天的 LifeState。
暂停虚拟钟时不扫描新的规划或唤醒任务。

## 提交与恢复

外层检查点 thread 绑定分支和协议版本；内部检查点还绑定协作 Cycle、输入版本、角色轮次
及 Prompt 版本。数据库客户端和工具实例不进入 State。工具正文随内部图状态保存，
恢复时重建分页引用，不能只保留 ToolMessage 而丢失 result_ref 的正文。

同一分支使用 Linux 内核文件锁串行执行；Worker 提交时还校验现有 Job 的 worker_token，
并在写事务中检查输入版本。文件锁适用于当前单机、多个 worktree 共用本地数据的环境；
不宣称支持多个主机或网络文件系统上的分布式调度。

Executor 以基线计划版本和幂等键提交，回执与计划同事务保存；已提交计划不受后续回复失败影响。
回复通过业务事件幂等键与 turn_id 回读，恢复不能重复生成用户可见消息。
提交前重新读取虚拟时间，不用调查开始时间绕过已经开始的块。

检查点文件 `runtime-collaboration.sqlite`、`runtime-locks/` 放在实际业务数据库旁。
本项目的业务数据库已外置，因此各 worktree 共用同一状态位置；测试使用临时目录。
这不是第二份人物或聊天数据库，也不是 Phoenix 的替代品。

## 调度与观测

Runtime Scheduler 使用独立线程和数据库会话，不被耗时模型调用阻塞。
realtime/cognition/background 角色可运行它，不需要启用训练或 all Worker。
计划准备在 `generation_metadata.preparation` 中单独保存
`pending → running → ready / retryable_failed / blocked`，与 Job 成功状态及已提交 blocks 分离。
修订失败保留旧日程。自动准备最多四次尝试（初次加三次重试），退避为 30、60、120 秒；
预算耗尽、模型未配置或需要用户澄清时阻塞，不能靠重排 Job 绕过预算。
同一输入版本沿用原 Job 和检查点；准备失败由用户显式重试，暂停分支不自动推进。
进入 Planner 前的写锁冲突也受退避约束；中断任务在租约恢复后可继续，不永久停在 running。

单 Agent 的调用预算保存在内部图状态中，外层累计模型步骤、工具调用、供应商已报告 token、
协作耗时和交接数。重复请求、重复追问和预算耗尽会终止并保留工作，不把无限来回当成正常协商。
预算用于熔断，不是面向用户的调用统计；真实调用和失败仍以 Phoenix 的供应商 span 为准。
Phoenix 以协作 Cycle 为父 span，关联角色节点、内部 Harness、交接与提交。

## 验证和范围

新增离线冒烟使用真实 LangGraph、SQLite 和可控模型，覆盖普通聊天、初始化、追问与回答、
三天时区窗口、次日维护、计划版本冲突、提交幂等、工具调用完成后的模型失败恢复。
原回复写入失败测试继续验证计划不回滚、事件可重试和消息不重复。

Runtime 与统一 Harness 回归共 59 项；另外运行同级协作定向冒烟，覆盖准备退避、
保留旧计划、已发布 v3 显式绑定及留档、原生检查点恢复工作草稿与工具结果。
Ruff、Python 编译检查及前端 `npm run build` 均作为交付检查，不用离线通过代替云端效果。

## 本轮补齐：工作笔记、背景绑定与体验环境

`tools/plan_work.py` 的 LangChain 工具 `update_plan_work` 保存四个明确字段：
已确定安排、未解决问题、日程草稿、下一步。它不写正式 DayPlan，也不升级为人物事实。
工具结果进入原生 LangGraph 检查点；恢复和上下文压缩保留最新笔记。
协作 State 的 `planner_work` 还会把笔记传到下一次 Planner 交接，避免提交被拒后从头调查。
最终提案的 blocks 自动成为最新草稿，即使该轮没有调用笔记工具也不会丢掉候选。
恢复时刷新任务锚点中的本地时间和约束，不把旧 ToolMessage 当成新的约束。

提交前重新读数据库中的 VirtualClock；暂停/恢复会递增输入版本，旧任务不能按旧钟面提交。
Executor 拒绝提案后把具体冲突交还 DayPlan 修正，跨日失效的旧提案不继续写入历史。
失败调查的累计预算继续保留；达到熔断上限后标记 blocked，不能声称会无限自动恢复。

`RuntimeService.refresh_published_background` 只允许显式更新 latest_profile 分支，
读取 active Publication 绑定的 ready 图和 v3 画像，不使用待审核草稿，不重做历史切片。
原背景完整写入 background_rebound 审计事件；旧 world 记忆停用但不删除，
新的来源引用按 profile_id 区分，避免相同字段路径指向旧内容。
原聊天、历史日程、虚拟钟锚点不变；当前及未来已提交日程保留，准备状态置为待修订。
运维入口为 `scripts/refresh_runtime_background.py`，三个目标参数必须显式提供。

本次体验分支 `f9b90dab-cb11-4736-beb2-7633611f6e23` 已更新为 Publication
`86f86b19-d16f-47fa-b4e7-386e2a379371` 的 Profile
`01f1a308-fc49-465d-9c52-592eff5ee81e`（v3）和 Graph
`1400178b-8e4a-468d-afd1-8e7c12d7aeec`。
实际规划记录确认使用此绑定，虚拟日期 2026-05-11 的日程已修订提交为 version 3、ready。
2026-05-12 的计划也已由后台提前生成，version 1、ready；任务队列清空后安全重启最终代码。
模型配置、LoRA、训练进程均未修改；画像版本同时写入 Phoenix 协作根 span。

聊天接口返回当天准备状态及是否有可用旧日程。没有计划时展示准备、自动重试或受阻说明，
不把缺失解释成空闲。单机分支写锁仍会串行执行同一分支任务，所以提示明确说明发送后
可能需要等待准备完成；没有承诺在一个长规划调用中同时执行同分支对话。
