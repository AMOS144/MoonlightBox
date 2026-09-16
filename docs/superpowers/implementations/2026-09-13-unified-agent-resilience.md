# 统一 Agent 请求韧性实现

## 统一边界

网络故障、请求超时、传输重试、取消和错误分类统一由
`backend/moonlightbox/agent_runtime/resilience.py` 维护。
`AgentLoopController` 在模型和工具执行时建立作用域，Agent 只声明
`AgentSpec.resilience`，不另写传输重试循环。

Runtime 原生工具请求、PersonWorld 栏目与 Revision 的结构化请求均已接入。
作用域内关闭客户端自带重试，避免 SDK、工具和 Agent 多层重试相乘。
非 Agent 的导入分析等调用仍保留既有客户端策略，本次不改变其业务行为。

## 请求策略

- 默认单次请求最多 300 秒，暂时性故障最多重试 2 次（含首次共 3 次）。
- 网络失败、超时、限流、服务端暂时故障和空响应可以重试。
- 认证、配置、协议、参数与结构错误不进行传输重试；模型修正参数或结构是另一回事。
- 指数退避加随机抖动，初始 1 秒、上限 10 秒；等待中检查取消。
- 请求上限还会受工具自身超时、当前操作剩余时间和执行预算约束，取最小值。
- 不重试整个 Agent。复合模型适配器声明真实请求边界，只重试失败请求，
  已完成的上下文组装或前序模型请求不会因此重放。
- 只读工具可以重放；proposal 等提交类操作不自动重放，避免超时后重复提交。
  工具内部嵌套模型调用不会再启动一套重试循环。

请求超时与整个任务预算不是同一件事。本次没有取消既有总预算、修改 token 额度，
也没有改变后台 Job 的自动恢复和退避规则。

## 取消与统一错误

`AgentExecutionRequest.cancellation_requested` 接收应用层取消回调；
既有输入版本检查继续负责让过时执行失效。两者在请求前后、工具批次边界及退避中检查。
Director、DayPlan、PersonaActor、生活事件模式、PersonWorld 栏目和 Revision
入口均可传入取消及状态回调。

这是协作式取消，不是强制杀线程。已经进入同步 HTTP 或同步工具的执行必须等其返回或
底层 timeout；返回后再次检查取消，迟到结果不进入后续提交。
没有创建可能在取消后继续持有数据库会话并写入的后台线程。
自定义同步工具仍必须配合传递 deadline；包装器不能强行中断任意 Python 函数。

统一错误字段为：

```json
{
  "code": "timeout",
  "category": "transport",
  "retryable": true,
  "message": "模型或工具请求超时",
  "attempts": 3
}
```

`retryable` 表示故障性质，不表示已经排入新一次 Job。
取消为 cancelled，新输入失效为 stale，预算阻止继续为 blocked，其他请求失败为 failed。
工具操作错误可回传模型修正；可修复工具错误不等于整个 Agent 已失败。
最终结果通过 `AgentExecutionResult.error` 暴露错误，工具错误携带同一 failure 结构。

## 状态和 Phoenix

`AgentExecutionRequest.on_status` 接收 requesting、retrying、request_succeeded、
request_failed 及执行终态。回调失败不使真实操作重试。
请求事件同时写到当前 Phoenix/OTel Span 的 `agent.request_status`，不新增自建 trace 库。
错误信息不透传供应商原始响应或凭据。

本次提供后端接口，没有新增前端取消按钮、SSE 状态接线或持久化取消 API。
页面显示“正在重试”仍需要应用层转发回调，跨进程取消仍需要调用方提供共享取消信号。
未重启体验服务，也未触发真实付费模型调用。

## 验证

离线 MockTransport 覆盖两类真实客户端的重试次数、认证失败、超时与取消；
循环回归覆盖统一终态、嵌套请求不叠加重试、复合适配器不重放成功前置工作。
相关测试位于 `backend/tests/agent_runtime/test_resilience.py`。
