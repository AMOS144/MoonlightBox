# 所有 Agent 的共享运行时精简

## 范围

此次不是只整理 Director 初始化。检查并覆盖 Director、DayPlan、PersonaActor、
生活事件协商、对话摘要、Director 初始化、PersonWorld 七栏目、Revision，以及通过
run_submission_task 接入的 Graph Patch、回归评估和 Alias。

没有重新设计行为 Prompt、工具权限、预算数值或领域事务；没有删除历史数据。

## 共同机制收敛

| 职责 | 唯一实现位置 | 领域仍需负责 |
| --- | --- | --- |
| 模型调用接口 | agent_runtime/contracts.py::ChatModel | 供应商客户端与任务材料 |
| 预算转换 | agent_runtime/policy.py | 选择任务策略；不同任务预算不强行相同 |
| 首轮任务与完整近期回合保留 | agent_runtime/context.py::retain_recent_turns | 压缩摘要、引用目录、调查草稿与保留回合数 |
| 提交约定说明 | agent_runtime/submission.py::submission_instruction | 声明具体提交工具和产物类型 |
| 模型/工具循环、停止与恢复 | agent_runtime/controller.py | 将 AgentSpec 与 AgentExecutionRequest 接入 |

RuntimeToolbox、PersonWorld 栏目、Revision 和短任务原先分别扫描 AIMessage、
寻找回合边界并保留首轮任务，现在共用一个实现。Runtime 的规划与初始化仍继承
RuntimeToolbox 的结果缓存机制，再加入各自的工作草稿，不额外复制裁剪算法。
没有模型回合时仅保留任务锚点，不重复保留旧摘要。

不同任务的摘要内容和分页方式仍保留：Runtime 使用结果引用；PersonWorld 使用调查附件；
Revision 保留纠正范围；固定资料任务使用材料分页。它们不是冗余的另一套 Agent 循环。

## 删除与修正

- 删除无人引用的 DayPlanLoopState 旧循环兼容壳。
- 删除无人引用的 _revision_tool_messages。
- 删除 DirectorModel、DayPlanModel 和 Controller 内部重复模型协议，直接使用 ChatModel，
  不添加旧名转发壳。
- 删除旧 TypeVar 声明，保留当前 Python 泛型语法。
- 删除领域中的预算转换函数原位置，全部直接引用统一 policy。
- 删除四处重复的消息边界扫描实现。
- Runtime Prompt 组装不再维护第二份提交说明；预览入口补入显式提交工具声明。
- 执行请求 owner_type 补齐已经使用的 runtime_expression，保持现有检查点身份字符串不变。

## 不应为了删代码而删除的内容

- Runtime 的协作总预算与单 Agent 预算约束不同，保留分别计账。
- 工具参数错误在工具层反馈，提交阶段的状态版本与事务保护继续保留。
- PersonWorld Send/async 编排和同步资源隔离不等于第二套模型循环，继续保留。
- Phoenix 观测、取消信号、传输重试、检查点、完整请求容量估算仍由现有统一机制负责。
- 历史画像读取兼容是用户明确要求保留的数据能力，不能因为包含 legacy 就删除。

## 验证边界

新增共享上下文测试覆盖并行工具回执完整性、不同保留回合数、无模型回合、
普通文本回合和参数检查。运行统一 Agent Runtime 测试及各领域接入回归。
仅离线验证，不请求真实云模型、不重启服务、不迁移数据库、不重置体验分支。

实际结果：统一运行时、PersonWorld/Revision、初始化及同级协作组合 156 项通过；
表达、生活事件、提交工具与 Prompt 接入组合 30 项通过；新增预览测试后单独重跑
Prompt 文件 6 项通过（其中 3 项与前述组合重叠）。Ruff 与 git diff --check 通过。
