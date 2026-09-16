# Director、DayPlan、PersonaActor 运行时修复

日期：2026-09-11。范围：将 PersonWorld 调查中暴露的同类问题修复到下游 Runtime，
不重新编译或替换已发布画像，不自动重新绑定历史分支的冻结快照，不启用 LoRA。

## 1. 统一执行，而不是再写三套循环

三个 Agent 继续由 LangGraph 的 `AgentLoopController` 执行，工具通过 LangChain
原生工具绑定；新增 `runtime_v1/agent_support.py` 只负责公共装配、输出解析和结果分页。
行为规则仍分别维护在 Director Prompt、`subagents/day_planner.md`、
`subagents/persona_actor.md`，不把工具 JSON Schema 复制进 Prompt。

最终输出的 Pydantic Schema 在启动时装配，缺字段、错误类型与业务校验结果会反馈给模型修正。
Phoenix 记录实际装配后 Prompt 的 hash，用于区分运行版本。调用和失败情况仍以 Phoenix 为准，
没有重新引入自建 AgentRun/Step/ToolInvocation 表。

Director 和 PersonaActor 的云端认知模型路径保持不变；这次没有更换模型或接回本地语气模型。

## 2. 工具结果、窗口与截止时间

- 删除未使用的旧 6,000 字符风格截断和本地 Trace 截断常量。
- 工具完整结果保存在单次执行内存中；超过 20,000 字符时，返回 12,000 字符的原始 JSON
  文本页、`result_ref`、`next_offset`，模型用 `read_runtime_result` 继续读取。
  页可能不是独立合法 JSON，Prompt 明确说明这一点。引用不能跨运行使用。
- 窗口达到 100,000 字符时，保留原始系统/任务上下文、最近两批完整工具协议及结果索引。
  较早工具正文可按引用恢复；这不是用固定长度字符串静默截掉原文。
- 300,000 字符是当前硬窗口保险丝，和累计工具传输预算分开。
  它是保守字符预算，不是假称掌握供应商精确 tokenizer；实际 token usage 来自供应商响应。
- 工具合同默认 120 秒，LightRAG 单次 HTTP 请求最多 90 秒，并受剩余 Agent 截止时间约束。
  模型请求及其重试共享剩余时间，不能每次重试重新获得一整份超时预算。
- 同步工具通过协作式截止时间约束 HTTP 操作；不声称可以强制终止任意第三方同步函数。

当前应急预算集中在 `runtime_v1/config.py`：Director 300 秒 / 64 次工具调用，
DayPlan 900 秒 / 96 次，PersonaActor 180 秒 / 32 次。这些是失败保险丝，
不是正常工作需要达到的轮数，也不是成功判据。

## 3. 冻结资料不能因新图发布而消失

`tools/routine_evidence.py::query_frozen_history` 按快照绑定的图版本查询，允许读取
`ready` 和 `superseded` 的冻结图；不切换到别的最新图。无法读取仍会报错。
返回 LightRAG 上下文、映射到的原始消息、发送者、角色、时间及来源 ID。

`snapshot_sources.py` 将“画像引用过的少量消息”和“快照有权读取的导入历史”分开：
latest_profile 快照可读取绑定图 Bundle 内、不晚于截止时间的消息。
不会因为画像只引用少数消息就让全部其他历史变成空检索。

网络、绑定和来源映射失败作为工具错误报告，不再返回伪装成功的空列表。
正常查询确实没有命中仍可返回空结果，两者有明确区别。

## 4. DayPlan 由模型理解材料，不用代码猜上下班

`analyze_routine_evidence` 接受 Agent 自己提出的问题和主题，一次调用执行一次资料查询。
移除固定问题表、工具内部嵌套的分析模型和上下班时间加权平均。
工具负责给出资料，Planner 负责追问、主体判断、时间变化与规律推断。

Planner 可直接读取 v3 的生活情境模块、实践理解和动机背景。
法定工作日不等于目标人物当天必然上班，消息发送时刻不等于活动发生时刻，
一次下班也不通过代码自动升级为长期规律。

已审核 v3 画像允许用 `profile_inference` 形成安排；画像字段得到绑定快照的稳定
`profile:` 引用，不强迫所有合理安排重新取得聊天原句。
不知道的具体事项仍用 fallback 表达，不假造来源。

结构和执行校验继续要求：日期正确、全天连续覆盖、15 分钟网格、锁定块不被改变、
引用属于当前冻结上下文。历史输入中仍要求调查的主题，只能由对应成功工具结果满足，
不能因为调用过一个失败工具就视为完成调查。

## 5. 下游记忆与 PersonaActor

v3 画像按字段路径生成稳定引用并幂等初始化 WorldMemory，重复启动不会重复插入。
历史 v2 数据读取兼容保留，不启用旧版生成或重试。

移除 SQL 字符/正则打分冒充语义匹配：数据库候选明确标为最近上下文，
world/both 查询另外接入冻结 LightRAG 的语义历史检索。branch-only 查询目前仍是
明确标识的近期候选读取，不声称它已经是向量检索。

PersonaActor 的风格工具沿用上述冻结资料读取，返回上下文和原始消息，
不按相邻 self/target 消息硬凑问答示例；允许自主追问和分页。
失败不等于“此人没有风格材料”。

## 6. 失败不能变成无声成功

- Director 的 action 必填，空对象不再自动成为 wait。
- speak 必须提供意图和内容点；schedule 必须提供唤醒时间。
- 唤醒时间、计划日期、活动变化依据等错误进入字段/业务反馈修复。
- Runtime 仅接受明确成功的终止状态；模型、上下文、超时等失败保留事件供重试。
- Executor 拒绝决策也不再转换成普通等待而消费事件。
- PersonaActor 失败返回无 message 的错误结果，不构造不合法的空消息；纯空白输出也被拒绝。
- 工具执行失败不计为成功完成对应工具任务。

## 7. 验证及范围边界

离线回归命令：

```bash
PYTHONPATH=backend .venv/bin/pytest -q \
  backend/tests/runtime_v1 backend/tests/agent_runtime backend/tests/world --tb=short
```

114 项通过。新增/更新的覆盖包括：ready/superseded 图读取、检索失败不能变空成功、
完整结果分页恢复、协议完整的上下文压缩、画像引用与幂等记忆初始化、具体字段修复反馈、
超大输入阻断、空决策及空白回复处理。相关 Runtime 与共享 Harness 文件 Ruff 检查通过。

这些是离线回归，不等于已经验收一次真实云端对话或新 DayPlan 的内容质量。
没有为测试发送用户消息、生成新画像、改写已有图谱或替换历史分支快照。
真实运行的耗时、工具命中和模型理解质量，应在后续体验中查看 Phoenix。
