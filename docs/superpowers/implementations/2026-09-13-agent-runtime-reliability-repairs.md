# Agent Runtime 可靠性修复

## 一、异步单栏重试

retry_v3 是同步 Job 入口，内部等待 aretry_v3；栏目批次和 life_context 后续补全均
显式 await。不能只创建协程后继续提交候选。

目标栏目失败时保留失败状态与检查点，不递增 Profile 的 candidate_revision。
同一个重试 Job 在栏目接受后、候选提交前中断，恢复已接受结果而不是重新加载旧 Profile。
候选提交与该重试作用域的 committed 标记使用同一事务：提交后 Job 确认前重启，
直接确认已有结果，不再次递增版本。候选版本已被其他操作更新则拒绝旧重试。

初次编译和单栏重试都通过 section_policy(settings) 注入相同策略。

## 二、LightRAG 请求策略

客户端保留原来的业务错误 code，同时携带 HTTP status_code 供统一运行时分类：

- 429、5xx 是可重试的网络服务失败。
- 400、401、403 等不自动重试。
- 客户端不另设重试循环；只有 Controller 中声明为可重放的只读工具可以重试。
  提案和写入不能因服务器报错自动重放。

GET 与 POST 的实际 httpx timeout 同时受工具截止时间、当前操作剩余时间、
任务剩余时间及客户端超时上限约束。请求前后检查取消。

模型请求超时和工具超时分离；不能拿模型的 30 秒设置无意间把图谱工具的 300 秒裁成 30 秒。
同步 Sidecar I/O 仍只能等请求返回或网络超时，不声称支持任意时刻强杀同步调用。

## 三、检查点契约

Controller 检查点 thread_id 增加 checkpoint_contract_fingerprint，包含：

- 原生提交协议版本；
- AgentSpec.contract_version 与 state_version；
- 工具 Schema、描述、提交身份、权限和副作用类别；
- ToolContract.contract_version；
- 工具函数及闭包 validator 的可执行结构、结果规范化及投影函数。

普通重启、同样的工具重新构建会得到同一指纹；修改结构、直接提交校验代码或显式版本，
使用新的检查点身份，不复用旧成功值，也不跳过新校验器。

**作者约定：**外部规则表、全局配置或间接依赖函数改变提交语义时，仍须递增
ToolContract.contract_version（单工具）或 AgentSpec.contract_version（整个 Agent）；
工作存档格式改变时递增 state_version。指纹不是完整程序依赖分析器，不能自动判断
任意代码变化是否兼容。闭包业务数据、数据库会话、密钥、实例地址和绝对路径不参与 dump。

旧的 submission-v2 检查点保留，但不自动跨契约导入；升级后第一次执行对应 Agent
可能重新调查。同一新契约下继续正常恢复。不删除历史 Profile、人工审批或原始资料。

## 四、工具输入错误与超时配置

工具作者对可由模型修复的问题使用 ToolInputError（LangChain ToolException 的子类），
说明字段、有效范围和修正办法。分页、回归查询范围、未知时区、未知情境类型已经接入。
Controller 返回 invalid_tool_arguments 及安全说明；内部未知 ValueError 仍只在 Trace
保留诊断，不向模型公开任意异常文本。Pydantic 字段错误仍返回结构化校验信息。

新增 MOONLIGHTBOX_PERSON_WORLD_SECTION_TOOL_TIMEOUT_SECONDS，默认 300 秒。
PersonWorld 的图谱工具读取此设置；轻量本地工具保留 30 秒上限，允许此设置进一步收紧。
模型超时继续使用 MOONLIGHTBOX_PERSON_WORLD_SECTION_MODEL_TIMEOUT_SECONDS。
配置示例已更新，没有修改实际环境密钥或模型。

## 验证与上线边界

新增离线回归验证真实 Controller 提交与恢复、契约变更失效、HTTP 503/429/4xx、
实际请求 timeout、非只读操作不重放、安全错误回传，以及真实单栏 Agent 重试与两处
中断恢复边界。没有请求真实云模型、重建人物资产或重启体验服务。

最终回归：agent_runtime、world、runtime_v1 与事件云客户端共 345 项全部通过。
相关文件 Ruff、后端 compileall、git diff --check 通过。新增及相关定向测试另行复跑，
26 项全部通过。
