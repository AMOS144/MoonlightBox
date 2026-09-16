# PersonWorld v2 生成能力退役

## 保留与删除

当前唯一生成实现为 `coordinator_v3.py`、`section_agent_v3.py`；
七份当前作者 Prompt 在 `subagents/`，审核对话仍使用 `revision.md`。
旧 `coordinator.py`、`section_agent.py`、`assembly/`、`subagents/legacy/v2/` 已删除；
`section_retry.py` 中 v2 调查、证据落库、画像重组分支已移除。

v3 所需的运行策略、预算和资料转换函数移到 `section_support.py`，保持实现不变。
数据库历史 v1/v2 JSON、读取 Schema、前端历史渲染、分支已冻结 Snapshot 均保留，
不做批量迁移，不修改画像内容，也不更换模型或 Prompt 正文。

## 入口行为

- 栏目重试只接受 v3 草稿；旧版请求返回 409，提示重新生成 v3。
- 已入队的旧版任务在创建模型客户端前以 `legacy_profile_retired` 失败，
  不能因恢复队列而重新启动旧 Agent。
- 前端旧版候选不显示栏目重试按钮，展示退役说明和“重新生成 v3 画像”。
- 完整重新生成固定调用 v3。对于旧的待审核候选，保留其 Profile、Draft 和图谱，
  在入队事务内将其图版本标为 superseded，并为 v3 建立独立 workspace 候选。
  新候选成功后仍需用户审核，不会自动批准，也不会修改原 LightRAG workspace。
- 已有新版候选在编译或审核时，拒绝重复创建候选。

## 验证

退役测试覆盖旧请求无副作用拒绝、旧队列任务模型调用前拦截、历史数据保留与独立
候选生成、v3 重试幂等性。原导入到审核发布集成测试已移除旧版模型桩，
保留并验证 v3 主链路。独属于旧生成器的测试随实现退役。

本次验证：world 测试通过；前端 TypeScript 与生产构建通过。
没有在真实项目上触发重新生成或发布。

删除前归档的旧 Prompt 和三份旧测试位于
`/tmp/person-world-v2-retired-prompts-tests-20260911.tar.gz`；
该文件是临时恢复备份，不是运行时依赖。
