# Director 输出校验与恢复修复

## 问题

08:00 唤醒已触发，Director 已选择延迟回复，但旧心理字段与新状态同时输出被拒绝。
旧 Schema 仍暴露这些字段，校验提示未指出具体路径。另一个潜在失败是合法 Wakeup
来源未被 Executor 接纳。最终输出失败的兜底值又被标为成功，检查点重试反复回放失败。

## 修改

- 新模型 Schema 隐藏 StatePatch 的 mood、attention、current_goal、open_conversation_threads；
  历史 LifeDecision 仍可读取。新 Director 若继续输出旧字段，反馈具体路径要求移除。
- Executor 的来源校验支持本分支、已到时且未取消的 Wakeup；主观状态来源采用同样边界。
- Director 最终校验增加数据库引用检查，错误可在表达前交回模型修正；提交时再次复核。
- respond_to_refs 只检查可读取的真实消息，不要求与 completed 的消息集合相等。
- input_resolutions 拒绝重复引用，避免同一消息出现互斥状态。
- 失败兜底不再标记成功；兼容识别旧检查点中的伪成功。重试保留材料和累计预算，重新开放
  有限的格式修复机会，不取消整体预算和版本守卫。

## 验证

离线冒烟覆盖 Schema 隔离、重复处理状态、合法及未来 Wakeup、非法消息引用、合并回复、
失败检查点再次调用模型。原有 Director、PersonaActor、协作和事务测试一并运行。
不改变时钟、不清空人物资产，不硬编码起床后的回复内容。
