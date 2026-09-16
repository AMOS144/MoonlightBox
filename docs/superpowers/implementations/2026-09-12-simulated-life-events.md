# 模拟生活事件第一版实现

日期：2026-09-12。依据[专项设计](../specs/2026-09-12-dayplan-simulated-life-events-design.md)。

## 实际链路

独立扫描线程累计虚拟时间机会强度 → 持久化机会 → 队列投递一次节点任务 →
Director 提供当前意见 → DayPlan 生成候选（必要时继续协商）→ Executor 原子提交事件、
可选计划修订及 RuntimeEventQueue 通知 → 普通 Director Cycle 消费通知并自主决定表达。

Director / DayPlan 使用原有实例、认知模型、AgentLoopController 与 Phoenix。
没有第三个创作 Agent，没有 HTTP A2A，没有启用 LoRA，没有删除 PersonaActor。

## 文件入口

- `backend/moonlightbox/runtime_v1/life_events/contracts.py`：心理机会、影响边界、当前意见、候选输出类型。
- `life_events/policy.py`：集中维护频率、冷却、累计扰动预算及产品初值。
- `life_events/scheduling.py`：按虚拟时间积分、去重、有限重试与暂停处理。
- `life_events/service.py`：平级节点适配，意见版本、候选协商、输入失效与恢复。
- `life_events/agent_mode.py`：将现有 Agent 的新任务模式装配为统一 Harness 请求；不是额外循环。
- `life_events/commit.py`：仅从 RuntimeExecutor 调用的领域验收与原子写入。
- `life_events/state.py`：机会取消/过期/阻塞时关闭共享任务，候选不成为事实。
- `life_events/models.py`：机会业务状态与不可变日程版本，不是自建调用 trace。
- `agents/day_planner_life_events.md`、`agents/director_life_events.md`：各自独立的行为文件。
- `tools/recent_life_events.py`：LangChain 只读经历工具，保留 simulation 来源并支持分页。
- `collaboration/graph.py`：复用现有图，新增可选的单节点执行和失败预算观察。

普通 Director 和基础 DayPlan 也可以查询已提交模拟经历。画像和原始 LightRAG 不被反向写入。

## 调度与恢复边界

每个后台 Job 只执行一个协作节点，在节点边界保存检查点并释放分支锁；所有 Agent 共享
原来的分支消息列表。单个供应商调用不承诺立即抢占，但新用户输入会使旧结果失效。
已有用户输入、通知或未完成的用户排程请求优先，后台机会可以等待或过期。

首次状态 request 由 DayPlan 节点适配层发出，不为发送一句已知请求额外调用模型。
因此新模式最终输出只有 no_event / discuss / submit_event；设计中的 request_context
在代码中是确定性的共享协议消息，不是另一个模型输出分支。

意见关联输入、计划和生活状态版本。旧输入失效后沿用原机会、重新询问 Director；内循环
检查点使用 context_epoch 区分变化后的上下文。累计墙钟、角色交接及调用预算不因新 Job 重置。
按内循环线程累计用量取最大值，既保留失败成本，也避免恢复后重复计数；真实调用细节仍看 Phoenix。

只有成功提交后才写 RuntimeLifeEventRow 与通知。通知复用现有 RuntimeEventQueue，由普通
Runtime Cycle 完成确认；失败恢复不会重新生成生活事件。no_event 不扣已发生事件数，
但该机会不会重抽。过期和失败候选会在共享记录中明确结束。

## 计划与概率的保守边界

第一版机会率按已有结构化 default_availability 配置：asleep/unknown 不激活，busy、
available、resting 使用不同虚拟小时率。不从活动名称正则推断工作或社交；还没有增加
更细的工作类型/可打断程度字段。具体率是可调整产品初值，不宣称心理学实测概率。

后端只抽心理作用领域、开放方向和现实扰动额度；没有具体剧情题库。
第一版心理重要性上限 low，现实扰动仅 none/minor，范围只限当前块，不开启中大型事件。

实际排程差异按分钟投影验收，不能自报零分钟后删除整段工作。只能在下一合法的 15 分钟
刻度之后修改当前块；保留已经过去的内容、其他块和 branch_commitment 块。
历史证据元数据也不能通过重拆块改写。意图不等于完成，预计结束不自动产生成功事实。

现有计划在首次相关提交前后归档，之后版本不可覆盖。无法恢复升级前未保存的初始历史，
不伪造它。次日唤醒从已提交次日计划首块读取，未准备好就交独立准备扫描恢复，不再默认 07:00。

## 历史迁移记录（已由 2026-09-15 方案替代）

新增迁移 `0055_runtime_life_opportunities`。先迁移，再启动新代码。
2026-09-15 已删除旧开关、hazard、冷却及影响额度；正式 Runtime 始终推进概率节拍。
当前规则以[概率生活推进设计](../specs/2026-09-15-dayplan-probabilistic-life-progression-design.md)为准；
本文其余内容只记录旧版实现，不作为现行启用或约束说明。

本次已核对体验 API 实际使用的库为 `.worktrees/.runtime-data/data/moonlightbox.db`，
并从 0054 升级到 0055。升级前 SQLite 一致性备份：
`.worktrees/.runtime-data/life-events-backup-LryRRZ/moonlightbox.db`。
默认 `.env` 指向的 `runtime-prune/data/moonlightbox.db` 是不同库，本次未迁移它。

没有修改运行中服务的模型设置、没有自动开启开关、没有往体验分支写模拟事件。
不将离线协议测试称为真实供应商生成质量验收；开启真实体验后仍需通过 Phoenix 审核自然程度。

## 验证

新增 `backend/tests/runtime_v1/test_life_events.py` 使用真实 SQLite 和 LangGraph、离线模型，
覆盖协商、明确候选接受、no_event、新输入失效、暂停/过期、实际计划差异、版本归档、
幂等事件与通知、普通 Director 消费通知及失败预算持久化。
同时运行原有同级协作、持久化、Runtime 冒烟和新库迁移冒烟。没有扩展成长时间模型训练或全面测试工程。

最终验证结果：45 项通过（45.10 秒）；修改范围的 Ruff 检查和 `git diff --check` 通过。
体验库确认迁移版本为 0055，生活机会表仍为 0 行；尚未重启体验进程或运行真实供应商生活生成。
