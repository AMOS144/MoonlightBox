# Runtime 整任务恢复统一归 Job 管理

## 职责

- AgentLoopController 处理单次模型/工具请求的临时故障、退避和取消，继续执行累计 token、工具和模型调用预算保护。
- LangGraph 协作层保留节点位置、讨论、草稿和累计用量。删除 execution_attempt 和协作层恢复次数上限，恢复只刷新本次计时窗口。
- Runtime Job 是整任务是否恢复、何时恢复的唯一决定者。Worker 在调用 Agent 时声明这一职责，Controller 不再用独立恢复次数阻挡 Job 已批准的尝试。未交给 Job 托管的独立 Agent 保留原有保护。

## 状态与预算

沿用现有 Job，不增加数据库实体。failed/interrupted 的 checkpoint.runtime_recovery 明确记录：

- waiting：允许恢复，retry_at 是唯一退避截止时间。
- terminal：不能自动恢复，reason 区分具体不可重试错误和 retry_exhausted。

默认首次执行后最多恢复三次，间隔 30、60、120 秒。生活机会可提供更小的累计额度，仍由 Job 决定恢复，不再通过递增生活 step 新建失败任务。计划准备状态只投影结果和 retry_at，不再自己管理恢复次数和退避。

网络、超时、限流、暂时服务故障以及 Worker 中断可以恢复；认证、协议、未分类程序错误和预算耗尽停止恢复。未知错误不当作网络故障反复执行。Worker 保存异常的明确 code，避免预算错误退化成通用 RuntimeError。

原 Job ID、业务幂等键、runtime_cycle、LangGraph 检查点、已读材料、草稿和累计消耗不清零。手工 resume 可跳过等待，但不能清零次数或绕过不可恢复终态。首次准备仍保留显式恢复入口。

## 释放占位，不绕过预算

等待恢复的 Job 保留分支占位；terminal 不再占位。

本轮实际读过的输入 ID 同步到原 Job 的 runtime_cycle.input_event_ids，不依赖 Phoenix 或自建 Trace 恢复。终态将这些尚未完成的事件/唤醒标记 failed，保留原文和检查点；消息元数据记录 input_status=failed、error_code、failed_job_id，不冒充已回复。

后到且未被本轮读取的输入仍为 queued，可以启动新的任务。已失败输入不会被扫描器不断重新建任务，从而绕过累计额度。新对话仍可从历史看到旧消息，但不把它们偷偷重新投递为同一任务。

失败状态与输入关闭在 Job 终态事务内完成，保留 Worker token CAS；自动扫描、人工恢复同样校验旧状态和版本。生活协作存档由持有分支锁的生活调度器关闭，Job 事务不跨库写检查点。

## 重启与部署

普通 Worker 中断、租约过期、显式 resume 和定时扫描均接同一策略；不再直接改成 queued 绕过恢复预算。旧 failed 行在扫描时按错误码及已有 runtime_retry_attempt 收敛，不清除旧消耗。

本次代码修改不自动重启体验服务，也不自动强制重放现有聊天；上线后的扫描会处理旧失败占位，并允许未读新输入继续。

## 验证

离线回归 77 项通过，覆盖 Job 租约与 API、失败恢复、协作、生活事件、时间片和 Runtime 冒烟。
新增用例确认：同一 Job 恢复保留工作与用量；终态关闭已读输入但保留后到消息；过期租约不能绕过预算；
Job 托管的 Controller 不再受第二套尝试次数拦截，但累计模型调用上限仍然有效。
本次涉及文件的 Ruff 静态检查及 git diff --check 通过。
