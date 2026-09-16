# PersonWorld v3 重构落地记录

对应设计：`../specs/2026-09-10-person-world-profile-lived-world-redesign.md`。
实现位于 `runtime-prune` worktree；旧的 `moonlight-box-implementation` 不是本次修改目录。

## 结构与入口

`world/person_world/__init__.py` 的默认 `PersonWorldAgent`、`PersonWorldCoordinator` 均指向
`coordinator_v3.py`。新编译输出 `profile_v3`；旧 v1/v2 数据、Prompt 和旧分支 Snapshot 保留，
不会把旧的原子事实改个版本号冒充新画像。
v3 Coordinator 独立实现。后续已整体退役 v2 生成和重试实现；历史数据只保留读取兼容，
需要更新时明确重新生成独立的 v3 候选。共用运行策略与资料转换移至
`section_support.py`，不再从旧 Agent 文件导入。

| 内容 | 实现位置（相对于 backend/moonlightbox） |
| --- | --- |
| 七栏及固定编辑字段 | `world/person_world/contracts/profile_v3.py`、`world_dimensions.py` |
| Big Five、MBTI、SDT、IPC | `world/person_world/contracts/psychological_models.py` |
| 十种生活情境及分组字段 | `world/person_world/contracts/context_modules.py`、`context_module_details/` |
| 七个独立作者 Prompt | `world/person_world/subagents/`（旧版 Prompt 已退役删除） |
| 原生工具与统一 Harness 的连接 | `world/person_world/section_agent_v3.py` |
| spec/list/read 模块工具 | `world/person_world/tools/context_module_spec.py`、`context_modules.py` |
| 真实消息分页与工具正文投影 | `world/person_world/tools/evidence_page.py` |
| 冻结快照、稳定身份与字段依赖 | `world/person_world/context_module_snapshots.py` |
| 纠正范围与合并候选 Diff | `world/person_world/review/profile_v3.py` |
| 确认理解后的后台补全任务 | `world/person_world/review/profile_preview_jobs.py` |
| 候选栏目重试 | `world/person_world/section_retry_v3.py` |

心理模型留在其所属栏目，不新增 psychology 根节点。画像允许无引用的 inferred 判断；
后端不计算人格分数、不要求两个日期或若干条直接证据，也不根据正则命中生成生活规律。
枚举值、字段类型、模块归属与版本仍必须合法。中文长度要求是作者编辑目标，不截断正文。

## 编译与跨栏补全

1. 七栏使用各自工具、资料工件和固定快照并行调查，使用同一个 `AgentLoopController`。
2. `life_context` 完整提交模块；后端接纳后分配 ID/revision，创建新的不可变快照。
3. `practices`、`agency`、`identity` 和读取过目录/模块的消费者集中补全，不相互递归触发。
4. `identity` 最后结合已有摘要填写 overview。
5. 草稿与审计状态落库，等待审核；调用失败不能写成“资料不存在”，也不能发布。

工具 schema 由 LangChain 注册，不在行为 Prompt 中重复 dump。工具返回实际 LightRAG 内容和
可分页读取的原文，不再把原文投影成只有 ID 的空壳。正文累计传输预算与当前上下文大小分开。
模型工具参数错误返回具体字段诊断；Phoenix 保留真实调用与失败，不新增自建调用账本。
`get_context_module_spec.read_paths` 提供从模块根开始的可读路径；业务字段使用
`details.<分组>.<字段>`。作者明确抛出的 LangChain `ToolException` 会作为可读错误返回模型，
普通异常仍不直接泄漏内部异常正文。最终引用 Schema 同样限定这些路径。

存储模型允许用 unknown 初始化空表单，Agent 交付模型则要求完整提交固定字段。
集中补全的“有材料”判断包括固定模块和其他已完成栏目摘要，不以重新读取原文作为必经门槛。
共享 Harness 的最终收尾也不施加跨领域的证据门槛，推断范围由本领域 Prompt 与输出协议负责。
生活情境额外提交仅用于审计的 `module_assessment`：说明识别了哪些模块，或为什么暂未识别。
它不写入人物事实，也不是关键词识别器。只读模块工具不意味着作者不能提交新模块草稿。

## 纠正与发布

前端 `V3ProfileGrid.tsx` 从后端目录读取字段定义，展示七栏、心理子项、模块分组与未知字段。
选择范围使用 `section + module_id/entry_id + field_path`，不使用列表下标或文案定位。
`PersonWorldRevisionPanel.tsx` 展示理解确认、直接修改范围、合并 Diff 和独立图谱批准步骤。

用户确认理解后，先修订直接选择的生活模块，再按读字段依赖让相关栏目重新判断，形成一份合并 Diff。
标题变更不自动重跑人格；字段级修订不能顺带改掉其他字段。新增/删除模块会通知目录消费者。
用户直接确认范围内变动的字段标记为 user_corrected，相关 Agent 的推断仍是 inferred。
仅选中 schedule 时，整个工作模块的 basis 不随之升级为 user_corrected。
identity 还消费其他栏目摘要：摘要发生变化时参与最后一次汇总；仅标题、排序或引用版本变动
不会触发人格重算。不会通过“工作要求严格”等文本规则推断人格。

默认只改 Profile。图谱变更必须由理解中的明确请求与后续独立批准授权。
Executor 使用已批准的完整合并结果，不在批准后再调用模型改写人格。
候选重试有数据库版本比较更新；最终发布再次检查失败栏目和引用版本。

v3 确认理解的 HTTP 请求只冻结输入并在同一事务入队，返回 `profile_compiling`。
后台 Worker 推进模块修订与相关栏目补全，前端通过现有 SSE/轮询显示进度；失败回到可重试的
理解确认态，保留旧画像。每次调用检查输入版本与任务归属，最终数据库比较更新阻止旧结果覆盖
用户新输入。不是点击确认后让浏览器等待十几分钟，也不会先发布半份修改再补人格。

## Runtime 与迁移

`runtime_v1/executor.py` 从分支绑定的 Profile 构建 Snapshot；`context.py`、`context_views.py`
向 Director/PersonaActor 提供七栏画像；`plan_context.py` 提供现实生活、实践与动机背景。
`persona_context.py` 单独组织 PersonaActor 所需材料，并保留心理模型解释与角色建议。
这些读取不查询正在生成的候选，所以不会把未批准修改泄漏到已有分支。
本次不启用 LoRA，不修改 Director/PersonaActor 的既定云端模型选择。

迁移 `0054_add_person_world_v3` 增加独立 JSON 列，兼容新库和历史库，不覆盖 v1/v2。
共享开发数据库已在迁移前备份至 worktree 外的 `.runtime-data/data/backups/`。

## 验证与真实运行状态

验证包括：固定 Schema、实际模型传输对标签联合的兼容、无引用心理推断、模块版本/依赖、
字段级纠正与关联 Diff、导入到审核发布的主链路，以及前端生产构建。

真实调用使用既有 MiniMax-M3 与真实 LightRAG 图谱；烟测不覆盖旧 Profile，不自动批准或修改图谱。
脚本：`scripts/person_world_v3_smoke.py`，支持 `--resume-run-id` 仅恢复失败栏目。
显式 `--refresh-sections life_context` 可重新生成模块并触发集中补全，不能默默跳过已成功却需更新的模块。
JSON 保存于 worktree 外 `.runtime-data/results/`，调用记录在 Phoenix。

纠正烟测脚本为 `scripts/person_world_v3_revision_smoke.py`，只在临时数据库副本中模拟确认的
工作安排更改并输出候选 Diff；临时副本退出即移除，不成为另一个开发数据库。模拟情境不代表
用户确认了真实工作事实，脚本不批准、不发布、不写图。

2026-09-11 已完成本次运行验收：

- 回归套件 111 项通过；最后的范围标记与摘要依赖修复后，相关 29 项再次通过。
- 前端 TypeScript 检查与生产构建通过，新增核心文件 Ruff 和 `git diff --check` 通过。
- 真实编译运行 `fe70cd9a-8a50-44f1-8d3d-1a8bb9154752` 经失败恢复后七栏全部 completed，
  `failed_sections=[]`；生成四个生活情境模块、20 项大五维度、MBTI 候选和非空 overview。
- 工作安排的真实模拟纠正依次完成 life_context、social_world、agency、practices、life_course、
  relationship_with_user、identity，输出七栏与 overview 的合并候选。
- 最后修正了“仅改 schedule 却把整个模块 basis 标为 user_corrected”的组装问题，使用已经成功的
  真实 Agent 输出重放确定性组装，没有追加模型调用或伪造 Phoenix 轨迹。复验确认模块 ID 不变、
  revision 从 1 变为 2、仅 schedule 为 user_corrected，其他模块字段完全不变。
- 两次烟测均未批准、未发布、未写图。共享库原 Profile 仍为 v2；审核页面不会自动换成烟测结果。

结果文件位于 `/home/yuyi/project/moonlightbox_clean/MoonlightBox/.worktrees/.runtime-data/results/`：

| 文件 | 含义 |
| --- | --- |
| `person-world-v3-smoke-20260911-r11.json` | 七栏完成的真实编译画像 |
| `person-world-v3-revision-smoke-20260911.json` | 真实纠正调用的原始合并结果，保留联调记录 |
| `person-world-v3-revision-validated-20260911.json` | 修正范围标记后重放组装得到的最终候选 Diff |

纠正烟测的初始模块来自 r8 编译候选；与完整编译 r11 使用同一批模块 ID 和原始图谱。
重放文件显式标注 `assembly_replay_from` 与 `llm_called_in_this_invocation=false`，
沿用真实成功调用的 Phoenix 关联，不把组装重放当成新的一轮模型生成。

这里的“通过”指流程、结构、版本与修改范围通过验收；心理判断与现实生活归纳仍是需要用户
审核的 AI 理解，不意味着已证明所有判断准确。本次没有启动或自动更新前端体验服务。

已定位并修正的联调问题包括：原文不可见、旧累计工具正文预算、模型值域未进入 Schema、
标签联合在供应商适配层被拒绝、恢复草稿的唯一键冲突，以及错误详情不足。
另已修正最终 Schema 的常量丢失、空数组示例误导、模块字段路径与实例不匹配、
身份综合已有模块却被误判成“无材料”。引用校验进入同次结构修复，不能等收尾结束才首次报错。
另有一次 MiniMax HTTP 422 `input new_sensitive (1026)`，已验证是供应商输入拒绝；
不能将其解释成没有资料，也不能通过静默删减聊天材料掩盖。
