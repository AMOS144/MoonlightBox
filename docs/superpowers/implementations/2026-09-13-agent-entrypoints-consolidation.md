# Agent 入口统一与旧协议清理

## 范围

本次改动覆盖 Director 前置容量拦截、策略注入、Graph Patch、图谱回归评估与 Alias。保留业务 LangGraph 调度、人工逐步批准、Graph/Publication 版本检查和 Executor 事务。未将 LightRAG 内部索引、事件提取、空间分析等批处理重写成调查 Agent。

## Director 与统一策略

ContextAssembler 只负责组装资料，不再调用 tokenizer，也不按旧的 16K/12K/24K 阈值删减资料。Director 不再根据 packet.budgets 提前返回 wait。真实请求窗口仍由 Controller 的完整请求容量估算与有界压缩保护，不能把移除旧拦截理解为无限窗口。

单 Agent 策略定义统一迁至 agent_runtime/policy.py；runtime_v1/config.py 仅保留跨角色协作的总预算。Controller 在入口统一注入网络策略：显式 Agent 配置优先，否则继承实际模型连接的超时和重试配置，测试／自定义模型才使用默认策略。NodeAnalysis 创建原生工具模型时保留其配置的重试次数，不再先清零再落回另一个默认值。

栏目模型超时与图谱工具超时分开；Revision 总任务时限不再通过“请求超时乘三”隐式计算。初次编译和 v3 纠正预览复用栏目策略的同一设置转换器。模型请求依然受剩余任务预算约束。

## 新接入的任务

agent_runtime/tasks.py 是短任务的领域无关适配器：原生模型、工具、明确提交工具、错误回执、检查点、取消与终态都进入 AgentLoopController。大固定输入通过 read_task_material 分页，不复制结构化输出 API 或 JSON 修复轮次。压缩保留完整模型／工具回执批次。

- Alias：模型自主选择人物节点、原文定位和图谱查询工具，通过 submit_alias_resolution 交付。目标和别称必须来自实际节点。移除旧的手写图循环、关键词候选补全与异常兜底；外围导入任务也不再把模型失败当作“无候选”继续发布。
- Graph Patch：Profile Patch 获批后，模型通过只读 search_graph 查询，再提交 submit_graph_patch。引用范围错误回传工具修正；已批准范围、最终批准和写入权限不交给模型。接受提案后仍签发图状态 precondition hash，用户继续审核具体操作。
- 回归评估：read_regression_query 只能执行已批准的查询。模型通过 submit_graph_regression 逐项交付判定；缺项、重复项和未读取的实际结果返回可修复错误。工作检查点保存实际查询结果，恢复时不会丢失供审核页展示的上下文和来源。

行为 Prompt 分别在 world/prompts/alias_resolution.md、world/person_world/prompts/graph_patch.md、graph_regression.md。工具 Schema 随 LangChain 工具传递。

## 退役与兼容边界

删除旧 model_repair.py 及其一次 JSON 修复测试，删除历史非 v3 Profile Patch 生成实现。历史画像仍可读取、定位和解释，但新修改必须先重新生成 v3。

删除 director_agent.py、persona_actor.py、subagent_catalog.py、空 subagents 壳、DirectorLoopState、旧 Prompt 常量导出，以及不支持原生工具的 remote_models.py / inference_client.py。删除旧 tokenizer 阈值及无人使用的工作窗口常量，测试导入改为正式入口。

训练与 LoRA 服务、既有公开人格推理服务、历史数据结构和数据库迁移未删除。事件提取等仍使用的结构化客户端也未删除；只移除了人物世界侧失实的 StructuredCompilerClient 接口，改为声明 create_agent_chat_model 的 AgentCompilerClient。

退役文件与旧 Alias 测试的临时备份：/tmp/moonlightbox-retired-agents-pOH4bc。该目录不是长期版本存档，重启或系统清理可能移除。代码尚未提交，不应通过 git reset 清理当前用户工作区。

## 验证与部署

离线冒烟覆盖 Alias 参数修正及失败不兜底、回归查询覆盖检查与成功恢复、网络策略继承和显式覆盖。纠正测试继续运行真实的 Profile/Graph 分阶段审批，并测试旧版禁止生成。

Alias 重新生成成功后，旧待审候选才与新候选一起替换提交；失败保留原审核材料。Profile 批准记录也在只读图提案完成后写入，仍绑定 Profile 专属哈希，不使用生成图提案后的总哈希。同步入口会把 Agent 终态错误转为可读的错误码与恢复提示。

验证结果：统一循环、Runtime、人物世界及云客户端合计 320 项测试中，首次 319 项通过，发现 1 处上述批准哈希问题。修复后重跑全部人物世界及新增统一任务测试，65 项全部通过。相关生产文件 Ruff 检查、后端 compileall 和 git diff --check 通过；未进行线上模型效果验收。

本次没有请求真实云模型、重新编译体验资产或重启体验服务，没有新增数据库迁移。此前取消生命周期的 0057_job_cancellation_ack 仍须在部署新后端前确认已应用。
