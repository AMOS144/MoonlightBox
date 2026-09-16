# Runtime 原生工具提交与校验职责调整

> 后续迁移已覆盖 PersonWorld、Revision 和摘要维护，并删除旧文本完成协议。
> 当前实现以 [统一交付与旧协议退役](2026-09-13-retire-text-completion-protocol.md) 为准。
> 下文范围与测试数字记录的是本次工作的较早阶段。

## 1. 范围

此次覆盖 Director、DayPlan、PersonaActor 以及 Director/DayPlan 的生活事件模式。
PersonWorld 的七栏目生成、人工审核和图谱多步确认不在此次迁移范围内。

目标不是取消校验，而是把可修复的提案问题放回 Agent 的工具交互中，避免模型已经结束后才得到失败。
底层事务、版本与权限保护继续保留。没有改动模型配置、启用 LoRA 或修改历史对话。

## 2. 如何交付结果

| Agent / 模式 | 提交工具 | 参数中的领域模型 |
| --- | --- | --- |
| Director | `submit_decision` | `LifeDecision` |
| DayPlan | `submit_day_plan` | `DayPlanTurn`，包含计划提案或协作回复 |
| PersonaActor | `submit_expression` | `ExpressionResult` |
| 生活事件模式 | `submit_life_result` | 各模式自身的结果模型 |

工具实现位于 `backend/moonlightbox/agent_runtime/submission.py`，使用 LangChain `StructuredTool`。
该文件提供通用提交机制；各领域工具工厂与校验分别位于 Runtime 的
`tools/submit_decision.py`、`tools/submit_day_plan.py`、`tools/submit_expression.py`、
`tools/submit_life_result.py`。分页参数与读取检查位于 `tools/read_runtime_result.py`。
Agent 适配文件不再定义这些校验或素材授权逻辑。
参数结构来自已有 Pydantic 模型，通过统一循环的 `bind_tools` 交给模型。
System Prompt 不再重复输出 JSON Schema。普通 assistant 文本即便包含合法 JSON，也不视为完成提交。

每次提交必须单独调用；同批混入查询或其他提交时，整批拒绝，避免先后顺序不明确。

## 3. 失败如何修正

统一循环执行工具时，提供本轮校验上下文，包括已完成的工具结果。

1. 类型、必填字段、枚举和日期格式错误：返回带字段路径的 `validation_errors`。
2. 领域条件不满足：返回 `status=rejected`、具体原因、`retryable=true`、`committed=false`。
3. Agent 阅读工具结果后修正，再次调用提交工具。
4. 验证通过：返回 `status=accepted`、`committed=false`，结束本次 Agent 执行，交给工作流。

`retryable=true` 表示这类提案错误可修复，不代表无限重试。既有取消、超时、预算和无进展保护仍然生效。
不再把没有新增来源 ID 当作停滞，也不限制为只允许提交。连续多批重复相同调用参数和结果时，
先提示调整，全部工具仍然可用；提示后仍持续重复才返回 `repeated_tool_cycle`。
修改参数、读取新内容、更新草稿不会仅因缺少来源 ID 被强制收尾。

## 4. 为什么 accepted 还不是 committed

提交工具是无领域写入副作用的提案接口。它不会发送聊天消息，也不会修改计划、人物状态或图谱。
工具接受后，原有工作流仍负责协作路由、必要的表达生成与 Executor 提交。

Executor 在实际写入时仍检查版本、当前时间边界、合法引用和事务一致性。
因此提案接受后仍可能因并发更新等原因提交失败；这必须作为实际执行结果处理，不能把工具回执当成数据库成功。

## 5. 保留与移除的规则

保留：结构合法性、计划日期与时间网格、覆盖与重叠检查、锁定安排保护、允许引用的消息与资源、
待回复消息和表达对象的一致性、版本保护及事务原子性。

移除 DayPlan 的强制调查过程门槛：不再要求调用特定语义检索工具或逐个覆盖固定主题后才准提交。
上下文里的 `required_topics` 改为 `suggested_topics`。Agent 根据已有材料决定是否继续检索；
已有充分上下文时可直接提交，工具失败本身也不必然否定合法计划。

领域校验函数仍复用现有实现，但上述 Agent 的循环最终校验回调不再承担提案验收；
可修复的错误由提交工具返回。没有简单删除所有校验器。

## 6. 恢复与观察

LangGraph 状态保存提交工具回执和 `accepted_submission` 标记。
仅有匹配当前提交工具的完成标记才可复用已完成结果；旧式普通文本完成不能冒充新协议的提交。
恢复时刷新系统消息，保留既有对话与工具历史。已接受的提案无需为了返回同一结果再次请求模型。
失败或中断恢复时重新开始本次墙钟窗口，保留累计模型/工具调用消耗。
同一检查点默认最多执行三次（包括初次），上限由 `max_execution_attempts` 管理。
外层同一协作 Cycle 同样重开时间窗口、保留工作和累计预算，并有三次尝试上限。
这不是无限重试，也不会重发已完成检查点的消息。

供应商返回有合法且唯一调用 ID 的错误参数时，使用 LangChain `invalid_tool_calls` 保存，
由工具执行阶段生成错误回执。未知工具名也交给分发层拒绝，模型可以修正。
回传历史保留原始工具参数、调用 ID 与推理字段；缺失或重复调用 ID 等不可配对的响应仍明确失败。

提交工具使用既有工具执行和 Phoenix 记录链路，不新增自建 Trace 数据库。
查看时需区分模型调用成功、提案被接受、Executor 实际提交成功这三个结果。

## 7. 验证

2026-09-13：统一循环与 Runtime 相关回归共 **83 项通过**。
覆盖拒绝后修正、字段错误反馈、禁止普通文本绕过提交、完成结果恢复、混合批次拒绝，
以及 Director、DayPlan、PersonaActor、生活事件与协作流程。

这些是本地回归验证，不代表已用云端模型跑完一轮真实聊天；未向体验分支新增测试消息。

### 工具错误、重复误判与失败恢复修复后的验证

统一运行时与 Runtime 合并回归：**97 项通过**。新增覆盖：

- 真实模型适配器收到未知工具、损坏参数 JSON、非对象参数后，把错误回传并完成下一轮修正；
  原始调用及推理字段仍可回放。缺失或重复调用 ID 保持协议失败。
- 没有来源 ID 的多栏目读取、相同参数但结果变化的草稿更新不会触发重复保护。
- 持续相同调用和结果先提示调整，工具保持可用；继续重复则有界停止。
- 失败超过旧墙钟后重试、运行中断后的 pending 节点恢复、完成结果复用。
- 恢复时保留工作材料与累计调用数，执行次数和累计预算仍能阻止无限重试。
- 外层协作恢复保留工作与累计交接，只重开单次时间窗口。

额外 PersonWorld 检查：16 项通过、1 项失败。失败项为
`test_persona_reads_frozen_profile_without_mutating_origin`，测试期待 `life_context`，
当前 `PersonaContextAssembler` 只选取版本、概览、身份和与用户关系。此次未改动该装配器，
也没有为了让旧断言通过而扩大表达上下文。本次三项修复不包含该上下文契约差异。
