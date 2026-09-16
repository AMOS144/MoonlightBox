# DayPlan 验收、事务修复与真实聊天恢复

## 事故结论

2026-09-11 的体验聊天暴露两个独立缺陷。DayPlan 虽然输出了覆盖全天的 16 个块，
但有 4 个非 fallback 块缺少来源，Phoenix 的真正终态是 `blocked / invalid_final_output`，
不能把 JSON 生成等同于合格计划产出。后续 Director 与 PersonaActor 成功生成回复，
但 Worker 没有注册 `media_assets` ORM 表，写入 BranchMessage 时出现
`NoReferencedTableError`。数据库里实际有该表，缺的是进程内 SQLAlchemy 元数据。

原异常处理没有先回滚失效 Session 就尝试再次提交，掩盖了原始错误；外层 Worker
最终回滚，计划不可用状态也丢失，前端看到的仍是 pending 和没有回复。

## 本次修复

1. `model_registry.py` 显式注册全部 ORM 模块，Database 初始化统一调用。
   API、Worker、恢复脚本使用同一个入口。不创建新表，不加载推理模型。
2. DayPlan 保留真实安排来源、画像推断来源，并新增 `simulation_assumption`：
   必须写该块 `assumption`，使用 inferred confidence、空 evidence_ids。
   模拟当天的时间分配无需被伪装成历史事实。行为 Prompt 和真实输出 Schema 同步。
3. 共享 Harness 的最终收束支持可配置定向修正，DayPlan 允许两次额外最终修正，
   仍受整体墙钟与模型调用上限控制，不再一次最终校验失败立即丢弃全部草稿。
   DayPlanRun 携带最后草稿及字段错误，初始计划失败时保存到 generation_metadata。
4. 初始计划通过验收后独立提交；不可用状态也独立保存。提交前检查输入版本。
   Director 发起的 revision 仍与对应决策处于同一事务，不拆散需要原子提交的变化。
5. 回复 flush 和最终 commit 都处于错误恢复边界内。数据库异常先 rollback，
   再加载已提交记录标记失败、恢复事件和唤醒。服务日志保留原始堆栈，Phoenix
   增加 cycle_failure span，Worker 保留具体异常类型。

## 验证

- Runtime、共享 Harness、PersonWorld：118 项通过。
- Jobs：12 项通过；合计 130 项。
- 新测试在纯进程中验证不导入 API 也能解析全部外键；用数据库触发器真实制造回复
  INSERT 失败，验证计划保留、错误记录、事件重新排队，撤销故障后仅产生一条回复。
- 模拟安排必须写解释；最终收束阶段的字段错误能得到修正。相关文件静态检查通过。

## 本次真实恢复

脚本：`scripts/recover_runtime_cycle.py`。默认只读检查；`--apply` 才允许定向模型修正
和提交。系统行为文件位于 `runtime_v1/prompts/day_plan_recovery.py`。

脚本从 Phoenix 读取指定 Cycle 的草稿、合格 Director 决策和 PersonaActor 回复，
校验输入版本、活跃任务及幂等键。仅修正 4 个不合格块，不改变活动、时间边界或
其他合格块，不借用或捏造消息引用。保留原草稿和恢复来源。

此次只新增一次约 58 秒云端修正请求；随后保存了 2026-05-11 的 16 个计划块，并通过
Executor 恢复原回复。回复复用不是新增云端生成，使用独立 recovery span 记录。
原失败 Job 保持失败，Cycle 标记 recovered，避免抹掉事故。已提交的幂等键阻止重复回复。

没有更改 OriginSnapshot、已发布人物画像、LightRAG 图谱、模型配置或 LoRA 开关。
