# Agent 工具契约与错误回执整理

## 1. 范围与维护位置

本次检查覆盖 Director、初始化、DayPlan、生活事件、表达提交、PersonWorld 七栏目、Revision、Graph Patch、Alias、回归判定和对话摘要，以及技能读取与通用工具分发。

- 工具仍由 LangChain `StructuredTool` 定义，由模型客户端 `bind_tools` 传递；没有另外实现工具注册框架。
- 只读工具参数写在各自工具文件；提交参数写在所属领域的 Pydantic 契约中。
- 参数用途、枚举、空值、列表上限及字段间约束进入 Schema；行为 Prompt 不重复整份工具手册。
- 十类情境字段的语义与固定目录一起维护在 `world/person_world/contracts/context_module_details/`，动态生成的类型不丢掉 description。
- `read_skill` 返回技能所需业务工具的实际 Schema；Director 的静态工具集合不因此改变。`execute_tool.arguments` 是有意开放的分发对象，内部工具仍严格检查。

## 2. 工具逐项索引

| 工具 | 参数契约重点 | 正常结果与错误边界 |
| --- | --- | --- |
| `read_skill` | 已列出的 skill；首次 SKILL.md，后续已列出 resource | 正文、版本和资源目录；不存在的资源是参数错误，不任意读文件 |
| `execute_tool` | 已提供的 tool_name；arguments 为对象 | 返回内部工具结果；错误保留 target_tool_name，字段路径从 arguments 开始 |
| `read_task_material` | 字符 offset、limit | 固定任务文本分页；读完 next_offset=null，越界不是空结果 |
| `read_runtime_result` | 已返回 result_ref；字符 offset | 原始结果文本分页；无效引用/越界反馈修正，不伪造正文 |
| `read_conversation` | before_ref 与 around_ref 二选一；limit 为消息数 | 连续原文与翻页引用；错误引用不能被当成没有历史 |
| `search_conversation` | 具体互动查询、命中上限 | 同时查询历史与分支消息，返回可展开引用 |
| `get_subjective_state` | 可选已有 concern_ref | 当前关切；未知引用反馈修正，省略可读全部 |
| `get_profile_section` | 七栏目枚举 | 当前分支绑定栏目，不偷偷换成其他版本 |
| `search_memory` | query、limit、include_original | 记忆与来源；身份、快照、scope、时间由后端绑定 |
| `get_style_examples` | 情境、意图、模式、limit；已提供 asset_ref 及配套游标 | 真实用法与可用素材；首次缺情境、越权素材、错误游标分别报参数错误；partial 不等于无历史 |
| `get_plan_constraints` | 当前任务绑定，无模型身份参数 | 已确认承诺和锁定块；只读，不修改日程 |
| `search_plan_memory` | 明确排程问题、候选上限 | 分支与冻结人物资料；空结果不是事实否定 |
| `analyze_routine_evidence` | 完整问题、生活维度、资料上限 | 原始材料由 Planner 理解；依赖不可用不归类为参数错误 |
| `get_recent_life_events` | limit、上一页 ISO before | 已提交的模拟经历；非法日期返回可修复错误，候选/未来结果不混入 |
| `update_plan_work` | 完整工作笔记、待查问题、部分草稿、下一步 | 保存可恢复草稿，不是提交日程；草稿允许不完整，最终提案严格校验 |
| `update_initialization_work` | 已读范围、暂定理解、待查问题 | 替换独立初始化笔记，不污染正常 Director 上下文 |
| `search_world` | 完整问题；local/global/hybrid/naive/mix | 图谱线索与检索引用，不能单凭空召回断言不存在事实 |
| `list_graph_entities` | 字面名称筛选、limit | 当前图实体；筛选不是自然语言分析或正则 |
| `get_graph_entity` | 图中完整实体名 | 实体详情，只读 |
| `get_entity_neighborhood` | 实体名、1或2跳 | 局部图邻域，不代表时间范围 |
| `get_relation` | 两个已有实体名称 | 图中关系内容，不自动判真或改图 |
| `locate_source_messages` | 当前 retrieval_id、上下文范围、limit | 还原 SQL 原文；未知 retrieval_id 报错，不返回“没有证据” |
| `get_message_context` | 真实消息 UUID、上下文范围 | 前后原文；缺失引用明确列在 missing_message_ids |
| `read_evidence_page` | evidence_set_id、消息 offset、limit | 证据集合分页；未知集合/越界报错，不与 retrieval_id 混用 |
| `analyze_evidence_dates` | 消息 UUID、IANA 时区 | 消息发送日期分布及缺失引用；不自动推导活动时间与规律 |
| `get_current_profile_section` | 七栏目键 | 当前绑定版本的已有结论；不是新证据 |
| `get_active_corrections` | 精确结构筛选、limit | 当前有效纠正；删除未实现过滤的 valid_at，避免假装支持历史查询 |
| `list_context_modules` | 可选 kinds/statuses | 冻结模块目录、实例 ID 与版本 |
| `read_context_module` | 实例 ID、可选合法字段路径 | 读取模块；无效实例/路径给出修正方向 |
| `get_context_module_spec` | kind 枚举 | 固定分组、字段和 read_paths，不读消息或创建模块 |
| `list_person_nodes` | 无参数 | 已标为人物的图节点，不执行合并 |
| `locate_original_mentions` | names、limit | 名称字面命中原文；不以命中自动判断同一个人 |
| `query_lightrag` | 候选人物完整名称 | 别名调查的局部图线索 |
| `search_graph` | 已确认纠正范围内的具体 question | 待纠正图的只读线索，不提供直接写图权限 |
| `read_regression_query` | 原样使用已批准 query | 候选图实际检索结果；未批准查询不可执行 |

具体工具注册名以源码为准；输入字段及完整嵌套提交 Schema 由 `test_tool_schema_inventory.py` 遍历检查，新增参数缺少说明会使测试失败。

## 3. 提交契约

| 提交类型 | 核心条件 |
| --- | --- |
| DayPlan | proposal/reply 二选一；全天连续覆盖、15分钟对齐、保留锁定块；模拟块 evidence_ids=[]、confidence=inferred 且说明假设；fallback 使用空引用与 fallback；其他依据引用已提供资料 |
| Director | speak 提交 ready 的实际 reply；schedule 给虚拟唤醒时间；心理更新、输入处理状态、记忆来源、DayPlan 请求各自有明确含义 |
| 初始化 | 只提交起点心理状态、开放话题和有效承诺；新关切不依赖旧动作或本次表达 |
| 表达 | ready 对应非空 messages、clarification=null；needs_clarification 相反；素材只取当前可用候选 |
| 生活事件 | no_event 不带事件/计划；submit_event 必须有候选；计划调整关联事件；预期影响与实际结果分开，actual_outcome=null |
| 七栏目 | 所属栏目固定细分字段；心理模型固定维度；模块、表达模式和来源引用均有说明，未知不强填 |
| Revision | kind 与唯一 payload 一致；问题、范围提案和理解确认分开；不因提交而获得用户批准 |
| Graph Patch | 操作类型及对应实体/关系字段明确；只提案，不写已发布图；hash 不由模型猜测 |
| Alias | 候选名称、理由和原文材料分开；空候选允许，提交不合并 |
| 回归判定 | 使用已批准且实际查询过的 query；判断实际内容是否符合预期，不把 HTTP 成功当作通过 |
| 对话摘要 | 保留具体进展及未结束事项，不生成新的聊天事实 |

## 4. tool call 错误协议

调用 ID 由原生 `ToolMessage.tool_call_id` 关联，不需要模型在参数里再次填写。

统一错误回执字段：status、error、tool_name、target_tool_name、message、recoverable、retryable、next_action、validation_errors、failure；提交拒绝保留原 code 字段兼容调用方。

| 情况 | 处理 |
| --- | --- |
| 未声明工具、JSON格式错误、提交与其他调用混在一批 | 返回可修复拒绝，保留调用关联；不冒充模型服务异常 |
| 字段缺失/类型错误/领域字段组合不合法 | invalid_tool_arguments；列出 loc、实际类型和值；模型修正后重新调用 |
| 提案不满足绑定范围、引用或状态条件 | invalid_proposal，给具体原因；不写库，可修改后重交 |
| 未授权、预算耗尽 | 不提示原样重试，更不能通过参数绕过保护 |
| 网络、超时、限流、服务端临时错误 | 保留传输代码，由统一运行时按现有策略重试 |
| 工具绑定资料未就绪 | tool_unavailable，不要求模型修改参数，也不当作无结果 |
| 取消 | 保留 cancellation 分类，不能自动改成可修复参数错误 |
| 未知内部异常 | tool_execution_error；不把异常堆栈或内部资料全部发给模型 |
| 合法查询无命中 | 正常空结果或显式 empty；不是工具执行失败，也不是“人物没有此事实” |

`recoverable=true` 表示模型可修改调用；`retryable=true` 只表示传输层可原样重试。二者不混用。

示例：收到 evidence_ids="" 时回显 actual_type=str、actual_value=""，而不是只说“请查看 Trace”；收到 ["placeholder"] 时错误定位到 evidence_ids，不重复整块提案。[]、null、缺失和空字符串不会混成同一个值。

## 5. 验证与边界

- Schema 检查覆盖直接参数及提交模型的全部嵌套 properties，包含动态生成的七栏目与十类情境。
- 离线重现 DayPlan 空数组故障，验证错误值回显、正确 [] 通过、内部工具字段路径和传输分类。
- 检查未知引用、分页边界和提交后的 Controller 结束行为。
- 不自动把模拟安排改标为画像推断，不把任意字符串、null、占位符强制转成 []。
- 类型与引用校验不能证明语义判断正确；是否有证据支持“通勤一小时”，仍须由 Agent 正确理解材料。不能因结构校验通过就宣布内容质量通过。
- 本次不增加逐字段 LLM 审核，也不把错误回执变成另一套 Agent 循环。

### 本次验证记录

- 统一运行时、提交工具、表达组件及 PersonWorld 契约/审核扩大回归：220 项通过。
- 最后补充实际尝试次数回执后，错误契约与表达组件专项：18 项通过（与上述测试有重合，不累加）。
- Ruff F/E9、Python 编译检查及 `git diff --check` 通过。
- 均为离线测试；没有重新请求云端生成 DayPlan，没有重启体验服务或修改体验分支数据。结构及错误协议通过，不等于新的日程内容已经完成真人效果验收。
