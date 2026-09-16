# Muse Phoenix 观测方式的项目适配

## 范围与不变项

参考本地 `muse-api/server/observability`、`server/utils/opentelemetry_sls.py` 和
`server/router/debug_trace.py`。观测不接管 Agent 决策、预算、任务恢复或领域事务；
不建立 AgentRun / Step / ToolInvocation 镜像表，不改 Prompt，不重跑人物编译或发送聊天。

## Trace 的组织

一次已领取事件的 Runtime Cycle 建立 `moonlightbox.runtime.cycle`（CHAIN）。
协作图、节点及 Executor 校验/提交属于该流程。删除异常退出后另建
`moonlightbox.runtime.cycle_failure` 的行为；异常在原 Span 上记录。

每次 AgentLoopController 执行仍建立独立 AGENT 根，保留 Span Link 和父级 ID，
这与 Muse 的 `muse.agent.run` 一致。通过 ContextVar 传播同一 `cycle.id`、
`cycle.key`、project、branch；运行时 Phoenix Session 使用 `branch:{branch_id}`。
独立 Agent 根不是重复故障；同一次异常又建失败根才是重复记录。

`cycle.id` 是本次领取任务的领域 Cycle ID；`cycle.key` 是幂等关联键，不能混用。
非 Runtime 的 PersonWorld 等 Agent 保持原 owner Session 组织。

## 统一边界

- `observability/phoenix.py`：类型、父级链接、关联字段、异常、快照、annotation。
- `observability/runtime.py`：Cycle 关联上下文与 Executor 阶段装饰器。
- `observability/agent_observer.py`：模型回合、工具、压缩和根摘要。
- `observability/provider.py`：每笔真实 HTTP 尝试、耗时、状态、请求 ID、响应及请求差异。
- `observability/investigation.py`：原始 Span 去重、快照校验重组及调查材料。

LLM Span 与 HTTP attempt 的 CHAIN Span 不重复计费：前者承载模型语义和 usage，
后者解释内部重试及传输失败。HTTP attempt 数与 Agent 轮数分别统计。
LangChain/LangGraph 自动 Span 保留用于定位图节点，不通过隐藏失败节点美化指标。
当前服务没有 Muse 的 SLS 双路导出需求，因此不引入它的 SLS 导出器和命名前缀过滤规则。

## 可定位的问题

1. 模型耗时：请求尝试序号、完整请求耗时、返回状态、request/completion ID、finish_reason。
2. 输入变动：规范化请求 hash/bytes、System + tools 前缀 hash、参数 hash、首个变化字节。
   只对同次 Agent 执行、同模型比较，不猜测云端缓存机制。
3. 上下文：输入与工具 Schema、组成、压缩前后快照、压缩量、模型输出。
4. 工具：参数、结果、耗时、重试、结构化引用、重复调用及消费线索（已有实现保留）。
5. 验收：`moonlightbox.agent.final_validation` 明确 accepted/reason/parsed；
   `runtime.executor.validate*` 与 `runtime.executor.commit*` 分开记录。
6. 根摘要：实际模型/工具统计、usage、缓存、压缩、规则诊断和 Phoenix annotations。
   规则诊断只是调查线索，不意味着已经证明模型理解了哪些材料。

## 成功与失败

供应商请求成功不代表输出通过验收，更不代表消息提交成功。
即使历史检查点携带 `status=succeeded`，`reason=invalid_final_output` 也标 ERROR，
根因摘要明确是最终输出未验收，不再显示“没有故障”。保留原始 status 方便排查历史矛盾。

输入过期或用户取消在统一手工 Span 上标为预期终止并保留原因，异常仍原样向业务层传播；
不改任务重试或数据库状态。LangGraph 自动 Span 可能仍将传播的异常标红，调查时应结合
根的领域终态，而不是把所有红色 Span 当作独立失败请求。

观测初始化、属性写入及收尾失败不能阻塞业务；真实业务异常记录后原样抛出。

## 完整内容及调查入口

继续使用 Phoenix 分块事件保存完整材料，普通 input/output 字段只是 UI 预览。
SDK 配置 512 个属性、2048 个事件、65536 字符属性上限；快照按 60000 字符分块。
读取时检查块号、块数和摘要；缺块或被截断就标记 complete=false，不伪造完整材料。
事件上限仍是有限的，完整性检查负责暴露超限，不承诺无限存储。

- `GET /api/observability/runtime-cycles/{cycle_id}`：汇集 Cycle 和独立 Agent 树。
- `GET /api/observability/agent-executions/{execution_id}/investigation`：执行调查 JSON。
- 原有 Agent Trace 和 summary API 保留。

返回 Phoenix 原始 spans、roots、errors、重组 snapshots 和去重后的 HTTP 尝试计数。
官方 Client 自动翻页；超过 20000 个 Span 明确拒绝，不能悄悄返回前 500 条。
旧 Trace 通过 owner ID 尽量关联，但不会改写它原本缺失的父级、类型或快照。

Muse 用 MongoDB 的消息引用补足原文。本项目已有完整运行快照，优先重组它们以还原
当时实际输入，不读取当前资料冒充历史输入，也不额外建 Trace 数据库。

## 明确的适配限制

- 当前模型是非流式 HTTP，不记录虚假的首 token 延迟；若未来启用流式再添加该指标。
- 不引入 Muse 的专用模型 ID、MongoDB、SLS、计费积分或文件工具特例。
- 不自动发起 Muse 的二次 AI 诊断请求。调查 JSON 可直接人工或交给 Agent 分析，
  不改变当前模型配置或额外产生云端调用。
- 保留正文快照，但仍强制去除 API key 等凭据。
- 不清理或改写已有 Phoenix 历史数据；旧 unknown 和旧绿色无效输出仍保持原貌。

## 验证

离线测试覆盖父子链接、统一 kind、预期终止、旧检查点无效输出、观测失败放行、
大段中文快照重组与缺块、错误计数去重，以及现有 Agent/Runtime 冒烟。
另向独立的 `moonlightbox-observability-smoke` Phoenix 项目真实导出两棵关联树，
验证完整中文快照能经 Collector 取回；不调用模型、不写业务数据库。
