# Runtime Prompt 维护入口

- `director/system.md`：Director 行为规则。
- `day_planner/system.md`：DayPlan 行为规则。
- `persona_actor/system.md`：PersonaActor 行为规则。
- `node_investigator/system.md`：节点调查 Agent 行为规则（node_investigation 模块复用同一目录）。
- `director/life_events.md`、`day_planner/life_events.md`：同一 Agent 的生活事件任务模式。
- 摘要与日程恢复辅助任务继续使用其独立提示文件，不是新的 Agent。

Markdown frontmatter 只声明描述和业务工具白名单；正文维护行为和字段语义。
工具参数跟随 LangChain StructuredTool，输出契约跟随 Pydantic 模型，不在此手写第二份 Schema。

`agent_catalog.py` 读取唯一作者文件；`prompting.py` 组装行为和输出契约，记录实际工具清单。
工具用途、参数和分页说明只随 LangChain 工具定义传递，不重复拼入行为 Prompt。
Director、DayPlan、PersonaActor 及两种生活事件模式通过原生提交工具交付结果：
`submit_decision`、`submit_day_plan`、`submit_expression`、`submit_life_result`；
节点调查 Agent 通过 `submit_investigation_turn` 交付。
结果模型作为工具参数 Schema 由 `bind_tools` 传给模型，不再复制进 SystemMessage，
普通文本中的 JSON 不视为提交。参数或领域校验失败会作为工具回执返回，供 Agent 修正。
`accepted` 仅表示工作流收到提案，不代表已落库；实际写入仍由 Executor 完成。
摘要维护和手动计划恢复也分别通过 `submit_summary`、`submit_recovered_plan` 交付；不再保留文本 Schema 兼容路径。
`submission_tool_name` 显式指定完成工具，组装器不依据工具名的前缀猜测。
原始聊天、协作和人物资料仍由上下文装配器提供，不改变资料范围或人物决策。

无数据库/云端调用的预览：

```sh
PYTHONPATH=backend .venv/bin/python -m moonlightbox.runtime_v1.prompt_preview director
```

支持 director、day_planner、persona_actor，JSON 分列正文、工具名、输出 Schema 和最终文本。
CLI 预览声明能力；实际运行可能只注册其中一部分工具。实际 SystemMessage 附带
`additional_kwargs.prompt_manifest`，含来源、内容 hash、实际工具名和提交工具配置，跟随已有
Phoenix 输入记录，不另建 trace 存储。完整动态上下文查看同次 Phoenix 输入消息。

通过 `agent_catalog.load_agent_definition` 显式加载，不保留旧模块名或 Prompt 常量导出。
运行策略统一在 `agent_runtime/policy.py`；`runtime_v1/config.py` 只包含跨角色总预算。
PersonWorld 仍保留自己的领域 Prompt 和加载器，本次不修改已验收的七栏目行为。
