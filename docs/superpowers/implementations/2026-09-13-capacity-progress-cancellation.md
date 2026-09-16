# 统一 Agent 循环：容量恢复、重复检测与取消

## 1. 完整请求超限后先压缩

实际模型客户端在发送 HTTP 前，对完整请求 JSON 做估算，包括工具 Schema、推理回传字段、输出预留、模型窗口和 usage 校正。容量不足抛出携带本次估算的 `RequestCapacityExceeded`，不是供应商失败。

`AgentLoopController` 接住该信号，调用当前 Agent 的上下文压缩器，再重新组装完整请求、重新估算。尚未发送的请求不消耗模型调用轮数，也不做网络重试。原字符阈值仅保留为提前整理上下文的软触发点；显式配置的字符硬上限仍是独立部署限制。

同一轮最多进行两次容量恢复。如果压缩未改变消息，或重新组装后的输入估算没有下降，停止压缩；只有无法容纳时才返回 `input_context_limit`。工具 Schema 本身过大等不可压缩情况不会无限循环。纠正 Agent 压缩按完整工具调用批次保留消息，不产生孤立回执。

## 2. 工具定义稳定的结果比较视图

`ToolContract.comparison_projection` 专门用于重复检测，不改变发给模型的结果、Phoenix 原始记录和真实引用身份。默认保留完整结果。

PersonWorld 检索工具显式剔除检索执行编号、证据集合编号及其执行来源元数据。只返回证据集合目录的工具还会纳入集合的实际消息内容，避免把不同正文误判成相同结果。原文消息 ID、正文、真实来源、页码和查询参数仍参与比较。

重复检测仍判断相同调用及结果是否反复出现，而不是要求每轮必须找到新证据。推断、修改草稿、修正参数和得到变化的内容都允许继续。

## 3. 取消请求不等于已经停止

Job 生命周期增加 `cancelling`：

- 排队任务取消：直接进入 `cancelled`，没有执行需要等待。
- 运行任务取消：进入 `cancelling`，保留 Worker 身份和租约。
- Worker 执行栈真正退出：以 Worker token 确认 `cancelled`，清除租约。
- Worker 消失：租约到期后结束取消，不将其重新排队。

Worker 向统一循环传播 Job 取消信号；PersonWorld 并行栏目通过复制上下文继承信号。Director、DayPlan、PersonaActor 也显式接入 Job 检查。Revision 同时检查纠正会话是否已取消或失效。取消后 PersonWorld 不再接受迟到栏目或合并审核候选。

自有模型 HTTP 使用可取消异步传输，同步入口每 100ms 检查取消，随后取消任务、等待请求退出并关闭连接。关闭本地连接不保证供应商停止已经接收的计算或计费。

注入的同步客户端、共享 SQLAlchemy Session 工具和同步 LightRAG 工具不强制在线程外中断：保留原执行策略，等待请求超时或正常退出，在边界丢弃迟到结果。此期间任务保持 `cancelling`，心跳继续续租，不能向用户报告已停止。

前端相关任务轮询识别 `cancelling`；导入和训练页显示等待退出，不提前结束轮询。

## 4. 文件与部署

- 统一控制：`backend/moonlightbox/agent_runtime/controller.py`。
- 请求容量：`agent_runtime/capacity.py`。
- 取消信号与 HTTP：`agent_runtime/cancellation.py`、`http_transport.py`。
- 结果比较：`world/person_world/tools/comparison.py`。
- Job 状态：`jobs/service.py`、`worker.py`。
- 数据库迁移：`0057_job_cancellation_ack`，扩展运行租约约束以容纳 `cancelling`。

现有数据库必须先升级到该迁移再运行新代码；本次代码修改不会自动迁移体验数据库或重启服务。

## 5. 离线验证

冒烟覆盖：小窗口触发压缩、无效压缩有界退出、编号变化不绕过重复保护、真实内容变化允许继续、异步请求取消后资源退出、运行任务等待 Worker 回执、错误 Worker token 不得确认取消。迁移测试在临时数据库升级至 head，不接触体验数据。另运行统一循环、业务 Agent 回归和前端构建。

本次结果：统一循环与 Job 测试 86 项通过；PersonWorld、纠正、协作、生活事件和云客户端测试 157 项通过。业务组排除了已有的 `test_persona_reads_frozen_profile_without_mutating_origin`（旧测试要求 Persona 暴露完整 life_context，与现行表达上下文契约不一致），未借本次修改调整该契约。额外新增真实 RuntimeCloudChatModel + MockTransport 容量恢复测试后，该冒烟文件 8 项全部通过。前端 `npm run build`、Python 编译检查与改动范围 Ruff 检查通过。
