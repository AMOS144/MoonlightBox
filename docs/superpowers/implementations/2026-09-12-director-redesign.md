# Director 重设计：第一版实现与验收记录

日期：2026-09-12

对应设计：[Director 处境理解、情绪与应对、连续对话及分层记忆](../specs/2026-09-12-director-context-memory-and-agency-redesign.md)。

## 已接通的主链路

- `director_contracts.py`：独立的评价、事项、处境更新、表达任务、记忆提案及输入处理回执 Schema。
- `subjective_state.py`：处境更新、来源范围、记忆替代及输入处理的领域规则；没有事件到情绪的正则或规则表。
- `executor.py`：处境状态保存到现有不可变 LifeState JSON 中，沿用版本 CAS 和决策幂等性，不新建第二套心理状态表。Actor 失败不提交依赖表达的更新；行动意向可关联实际决策回执。
- `prompts/director.py`：Lazarus/EMA 启发的处境评价—应对—反馈规则；不每轮强制生成情绪、关切或记忆。
- `context.py`：取消两小时遗忘边界，保留待处理输入和未被成功摘要覆盖的原文，装载已有摘要和快照允许的导入历史前缀。
- `tools/director_context.py`：注册 LangChain 原文分页、分支语义搜索、处境读取和绑定画像展开工具。
- `conversation_index.py`：复用本地 BGE 中文 embedding，在业务数据库保存可重建向量；覆盖分支聊天、有效记忆与已提交模拟事件，不修改 LightRAG。检索标明 ready/partial/index_pending/embedding_unavailable/empty。
- `conversation_maintenance.py` 与 `prompts/conversation_summary.py`：通过统一 Controller 运行连续区间摘要；来源范围由代码绑定，摘要成功持久化才允许替代旧原文。已移除原来的固定字符摘录生成入口。
- 消息/决策事务同时写维护 Job outbox；摘要与索引由后台处理，使用独立维护锁，不持有聊天分支锁。后台任务沿用已有有限退避重试，恢复时跳过已摘要和已索引来源。
- `collaboration/nodes.py`：正常 Director 预取相关分支记忆，绑定新增工具；将表达任务交给原 PersonaActor。
- 普通聊天和生活协商使用同一处境状态；DayPlan 上下文也能读取该状态。Director 与 DayPlan 保持同级，不为普通可用性判断增加交接。

## 兼容与第一版取舍

- 处境状态模块先保持单文件，未为目录形式拆空包。
- `expression_task` 暂通过适配填充原 communication_intent/content_points，保留旧读取契约；新旧心理字段禁止同时提交。
- 历史消息没有 input_status 时标记为 legacy，不伪造“已回复”，也不让所有旧聊天重新触发回复。
- 近期候选数、摘要批次大小是初始容量配置，不用于判断话题结束。未被摘要覆盖的旧消息仍受保护，过大时显式预算失败而非静默截断。
- 语义索引是现有 SQLite 内的派生表；第一版对已过滤的向量做相似度排序，不引入新向量服务。索引失效不代表历史不存在，可回读原文。
- 心理变化由模型解释，代码只保证作用域、持久化、版本与提交边界；不宣称实现了心理学验证过的 EMA。

## 验证

执行 Director 新冒烟、同级协作、模拟生活、Runtime smoke、持久化异常和迁移冒烟，共 **49 个测试通过**。覆盖 Actor 失败、幂等提交、隔夜连续原文、语义来源隔离、摘要恢复及旧计划协作。

另外实际加载现有本地中文 embedding 模型，对两段中文生成向量，均为 **512 维**；未下载新模型、未启动 LoRA。

新改 Python 文件 Ruff 检查及 git diff --check 通过。未运行云端真实连续聊天质量验收，不能用这些测试宣称角色已经自然。

## 环境启用

后续统一历史入口修订：`conversation_history.py` 负责导入历史与分支聊天的融合查询和连续原文视图。`search_conversation` 不再只查分支；`read_conversation` 只接收消息引用与数量，不再接收来源选择；`search_memory` 的公开 Schema 移除 scope，内部固定查询完整可见范围。新增冒烟覆盖跨起点翻页、前后文展开、结果去重、越界引用拒绝与单路失败保留结果。

新增迁移：`0056_director_conversation_vectors`。此次未升级体验数据库、未重启 API/Worker、未往体验分支发送测试聊天。

启用前应备份并核对进程实际使用的数据库，再对同一数据库执行 Alembic upgrade head，随后重启对应本地进程。不要在另一个 worktree 默认数据库执行迁移后误以为体验环境已升级。

维护需要现有 `models/embeddings/fastembed-bge-small-zh-v1.5` 以及原有云端模型配置。模型目录缺失时任务明确失败并按已有策略退避，不自动下载、不悄悄改成关键词检索。
