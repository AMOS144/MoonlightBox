# 概率生活推进实施记录

日期：2026-09-15。

对应[设计文档](../specs/2026-09-15-dayplan-probabilistic-life-progression-design.md)。

## 正式路径

生活推进是正常 Runtime 的必需能力，没有 enable 开关。Scheduler 总是扫描活跃分支，
虚拟钟每经过 20 分钟执行一次概率检查；未命中不启动模型。

默认策略集中在 `backend/moonlightbox/runtime_v1/life_events/policies/default.yaml`：
时段命中率使用周期 smoothstep，方向使用加权抽样，强度使用 Beta(2,5)。
可用 `MOONLIGHTBOX_RUNTIME_LIFE_POLICY_PATH` 指定另一份策略文件；这只是路径覆盖。
Worker 启动校验配置，重复键、无效概率或权重明确报错，不静默停止生活推进。

## 调度及持久化

- `LifeScheduleCursor` 每分支一条，保存检查序号、下一检查时间与扫描心跳。
  未命中同样推进；唯一检查身份不包含策略版本。
- `LifeOpportunityRow` 保存实际已经到达的机会、冻结的抽样参数与业务进展。
  新增 `check_key` 和 `kind`；命中后不重新抽签。
- 心跳间隔超过 60 秒且已错过检查时，生成一次恢复区间，不逐点补抽。
  60 秒仅用于辨认调度中断，不是事件冷却或有效期。
- 忙碌期间继续记录实际命中的机会，下一批统一交付。共用 Planner 写锁，
  不取消正在运行的模型，也不为每次命中并发启动一个 Planner。
- 暂停虚拟钟不抽样；恢复从未来节拍继续。正常加速与服务离线分开处理。
- `life_followup` 使用已有 Wakeup，进入 DayPlan 而不是普通 Director 定时唤醒；
  到期不自动宣告成功，也不改变随机节拍。

## Agent 及通信

DayPlan 继续使用统一 AgentLoopController 和平级 LangGraph，不增加 Agent。
模型机会输入只有 `direction`、`intensity`；宿主上下文提供到达时间、
当前时间、已备计划、人物资料、近期经历及协商记录。

模型通过原生 `submit_life_result` 交付：

- `no_event`：消费本批机会，不制造经历或对用户发消息。
- `discuss`：保存候选、让出执行，通过公共 `send_agent_message` 与 Director 商量。
- `submit_event`：提交当前经历及确需修改的各日期计划。

Director 的文字回复通过相同输入队列续接生活任务，不再填写另一份批准结构。
已删除 `life_responses`、`life_advice.py`、旧状态包装路径，以及 Prompt 中的对应要求。
提交成功后在同一事务中写入 Director 通知；收到通知不等于必须告诉用户。

## 提交与恢复

删除每日次数、扰动分钟、冷却、TTL、当前块范围、固定低影响和强制批准限制。
计划可以跨块、跨日；工具先反馈参数问题，Executor 再按最新时刻与版本检查。
仍保护已发生前缀、已确认约定、取消、租约、幂等与事务一致性。

批次消费与事件写入同事务，避免提交后进程退出留下永久 batched 的机会。
网络故障错误码保留到 Job 层，整任务恢复只使用现有统一重试策略，
不再额外读取 opportunity 自己的重试额度。

恢复期间补充的模拟内容带 `generated_during_recovery`、实际记录时间；
指定的发生时刻必须位于恢复区间内。后续事件关联原事件，
不把预计耗时或未来结果自动变成已发生事实。

## 数据迁移

新增 `0063_probabilistic_life_cursor`：

- 已提交经历及计划不变。
- 未激活旧 hazard 机会标为 superseded。
- 无法直接兼容的旧活跃工作标为 blocked，保留草稿并给出
  `legacy_life_work_requires_review`，不伪装完成。
- 旧 hazard / expires_at / attempts 数据列暂留读取兼容，不控制新路径。

共享体验库已从 0062 升级为 0063：
`.worktrees/.runtime-data/data/moonlightbox.db`。
迁移前后均为 17 个分支、71 条聊天消息，生活机会记录为 0；
SQLite quick_check 为 ok。

迁移前一致性备份：
`.worktrees/.runtime-data/data/moonlightbox.before-life-v2-20260915-081318.db`。
先在临时副本验证，再应用共享库；没有修改 worktree 内那份旧数据库。

## 验证边界

真实 SQLite、LangGraph 与离线原生工具模型替身覆盖概率、去重、忙碌积累、
暂停/离线恢复、无需批准的提交、公共消息续接、跨日计划、批次消费及独立后续唤醒。
Runtime 与 Job 回归通过 152 项；新增的三个针对性用例后，生活推进冒烟共 14 项通过。

没有重建图谱、重新编译画像、改动节点范围或删除聊天。
本次没有启动真实模型的持续生活体验；上述测试不是供应商生成质量验收。
标准 Worker 启动后直接运行新机制，不需要另行启用。
