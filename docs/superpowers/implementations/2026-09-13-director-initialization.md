# Director 分支起点初始化

## 准备顺序和隔离契约

新分支绑定画像和历史、暂停虚拟钟，完成当天 DayPlan，再运行独立 Director 起点任务。
提交成功后才开放聊天并恢复虚拟钟。初始化失败显示失败状态，使用现有准备重试入口；
已有正式计划不重复生成。历史活跃分支不自动补跑，不覆盖其已经发展的状态。

初始化复用 Director 的认知模型和 AgentLoopController，但使用独立的
`owner_type=director_initialization`、输入指纹、检查点线程、工具集合、消息列表和结果缓存。
初始化不进入同级协作消息列表，不调用 PersonaActor，不发消息、不安排唤醒、不修改图谱。
普通 Runtime 只读取已提交的 `subjective_state`、`open_conversation_threads`、`active_commitments`。
调查笔记、预读原文和工具调用保存在任务/检查点/Phoenix，不复制到普通聊天执行上下文。

## 输入和调查

人物画像、虚拟时间、历史截止边界和共享计划构成起点背景。历史以连续分页方式向前预读，
不是固定 30 条；初始原文占扣除固定输入、工具 Schema、输出预留后的窗口余额的 45%。
这是本地启发式预算，使用统一估算函数和 Controller 当前预算窗口；不是精确 tokenizer。
真正发送时仍走完整请求估算、usage 校正与压缩准入。预读按完整消息保留；截到话题中间时
由 Director 根据继续读取位置展开，不用正则判定话题边界。

可用工具复用连续原文读取、统一聊天搜索、记忆搜索和画像栏目读取。
`update_initialization_work` 保存已读范围、暂定理解和待查问题，是单任务草稿，不是人物事实。
它使用显式串行 proposal 契约；不是绕过 Executor 修改正式心理状态。

压缩保留起点背景和当前调查笔记，移出大量预读原文；正文已缓存为稳定结果引用，可分页恢复。
工具调用保持完整批次。进程重启继续独立检查点，工作笔记也在业务表中保留。

## 提交与事务

`submit_initial_state` 的结构复用正常 Director 的 SubjectiveStateUpdate；另有固定字段的开放
话题和有效承诺。心理状态允许整体推断、不强迫填满；承诺与提议由模型根据上下文区分。
提交工具反馈参数、来源引用和不允许的行动依赖等可修复错误。不增加独立心理核验 Agent。

Executor 在分支仍准备中、尚无新增聊天、状态版本未变化时创建新 LifeStateVersion。
原有生活状态字段保留，增加心理状态、话题和承诺；与初始化 ready 状态同事务提交。
计划、画像、历史边界、虚拟起点或状态版本变化会使旧调查无法提交。普通重启不重复初始化。

## 代码位置与部署

- `runtime_v1/initialization.py`：任务编排、输入版本守卫、统一循环适配和唯一的初始化失败处理边界。
- `runtime_v1/initialization_context.py`：历史预读和压缩；不管理分支生命周期或提交正式状态。
- `runtime_v1/tools/initialization.py`：工具装配、调查笔记保存和提交参数检查。
- `runtime_v1/director_contracts.py`：初始心理状态、开放话题、承诺和调查笔记的结构定义。
- `runtime_v1/prompts/director/initialization.md`：Director 起点模式行为。
- `runtime_v1/executor.py::commit_initial_state`：正式状态的事务保护。
- `runtime_v1/service.py::_finish_branch_preparation`：准备完成前的接线。
- `runtime_initializations`：仅存准备状态、输入指纹、调查笔记和错误；不是另一套 Agent trace。
- `0058_director_initialization`：新增业务表，保留所有已有分支数据。
- 分支准备页新增“理解近期经历、当前心境与未结束的交流”阶段。

部署需要先升级数据库到 0058，再重启后端/Worker；不能在旧表结构上直接加载新代码。
本次不触发真实模型编译、不发布画像、不重置当前体验分支。新建分支将执行此准备任务。

## 去重与职责收敛

初始化错误统一由 `initialize_director` 回滚并记录一次失败，不再在内部执行流程重复处理。
已成功提交的 ready 状态不被后续异常降级。分支生命周期的失败状态统一由
`RuntimeService.process_next` 外层处理；内层仍负责本轮执行记录与事件恢复，两者不混用。

每次输入版本检查只读取时钟、共享计划、画像、状态版本及是否出现分支消息，
不再重复装配聊天窗口、历史摘要和整个 ContextPacket。开始调查时仍完整装配一次材料。
提交前的新鲜度检查、取消检查及 Executor 的并发事务保护全部保留。

此次整理没有修改模型行为 Prompt、增加兼容转发层或改变公开准备接口，也没有删除历史数据。
整理后初始化与同级计划协作的 17 项冒烟测试全部通过，包含失败只记录一次、ready 不被降级的
新增用例；相关文件 Ruff 和 `git diff --check` 通过。未重启服务或操作体验分支数据。

## 本轮验证

5 项独立初始化冒烟通过，覆盖完整 DayPlan→初始化→开放聊天、长历史预读、失败恢复、
状态/调查记录隔离、幂等、压缩后回读、取消和输入过期。准备链路回归修正后通过；迁移测试
确认新增表可由 Alembic head 建立。前端构建及 Ruff 检查通过。尚未对体验数据库迁移或请求真实模型。
