# 全部 Agent 工具参数责任与收窄接口

## 1. 原则和范围

本清单对应当前代码中的 Director、起点初始化、DayPlan（日程与生活事件模式）、
PersonaActor 保留入口、PersonWorld 七栏目、Revision、Graph Patch、Graph Regression、
Alias、对话摘要及其共用工具。普通聊天仍由 Director + speaking 技能表达，不恢复 Actor 中转。
CloudEvaluator 等没有原生工具的独立评分 API 不冒充本清单中的 Agent 工具，也不在本次迁移。

参数分为三种责任：

1. **模型决定**：问题、解释、内容、选择、意图、拟修改的对象和需要确认的范围。
2. **后端绑定或推导**：项目/分支/人物身份、快照、日期锚点、版本、授权、状态 hash、
   编号、完成标记，以及可从所属字段唯一确定的常量。它们不进入可填写 Schema。
3. **模型选择已有引用**：消息、关切、实体、素材、模块、候选条目和工具游标。
   后端返回，模型选择，不自行生成；随后由后端核对范围。

分页条数、上下文展开范围可由模型按需调整，但都有后端默认值；不是每次必填。
不机械删除这些有明确用途的读取控制项，也不把心理、主体、关系、来源性质等判断交给正则。

## 2. 共用工具

| 工具 | 模型决定/可选控制 | 选择已有引用 | 后端负责 |
| --- | --- | --- | --- |
| read_skill | skill、resource（默认 SKILL.md） | 已列出的资源路径 | 技能目录、人物表达资料注入、内容版本、资源权限 |
| execute_tool | tool_name、该工具 arguments | 技能公布的名称 | 固定只读注册表、权限、取消、超时、串行资源、内层校验；不动态注册新 Schema |
| read_runtime_result | offset（默认0） | result_ref、next_offset | 当前运行/检查点附件、页长和正文边界 |
| read_task_material | offset、limit（均可省略） | 返回的 next_offset | 固定任务资料、身份、总长度；不允许换任务 |

## 3. Runtime 查询与工作笔记

| 工具 | 模型负责的参数 | 后端负责的参数/行为 |
| --- | --- | --- |
| search_memory | query；可选 limit、include_original | 分支、快照、排除来源、有效时间、固定 both 范围 |
| search_conversation | query；可选 limit | 合并可见历史和分支，时间与项目边界 |
| read_conversation | before_ref 或 around_ref；可选 limit | 原文读取、分页顺序、可见性；两个引用不能同时传 |
| get_subjective_state | 可选 concern_ref | 当前绑定状态，省略读取全部，不要求模型传状态版本 |
| get_profile_section | 七栏目之一 section | 本分支绑定的画像与版本 |
| get_recent_life_events | 可选 before、limit | 仅已提交且已发生事件；当前虚拟时间和 UTC 存储转换 |
| get_style_examples | 情境模式 situation + intent；素材模式 asset_ref；可选 speech_mode、limit、usage_cursor | 人物与历史边界、真实用法召回、素材可用性、候选授权；不接视觉模型 |
| get_plan_constraints | 无 | 目标日期、已确认承诺、锁定块、旧计划版本 |
| search_plan_memory | query；可选 limit | 目标分支/快照、固定 both 范围和当前时间 |
| analyze_routine_evidence | question；可选 limit | 固定图谱策略与来源恢复；删除只被回显而不参与检索的 topics |
| update_plan_work | established_arrangements、open_questions、draft_blocks、next_steps | 覆盖当前调查笔记、检查点恢复；不是提交正式日程 |
| update_initialization_work | read_ranges、tentative_understanding、open_questions（可省略为默认） | 当前初始化任务、保存时间；不是正式人物状态 |

笔记是模型实际掌握的工作状态，不属于可由后端凭空生成的元数据。保留其内容字段，
不要求写数据库任务 ID、更新时间、完成序号或重复附带输入资料。

## 4. Runtime 提交

### Director：submit_decision

- 模型决定 action，以及实际需要的 reply.messages、state_patch、subjective_state_updates、
  plan_request、peer_reply、life_responses、memory_proposals、next_wakeup_at、private_reason。
- input_resolutions 中“哪些消息已经回应/暂缓/无需回应”仍是模型判断。后端不能把所有已读消息
  一律标成完成；状态与消息引用随后由 Executor 校验。
- speech_mode 区分主动、延迟与普通回复，属于表达意图，可省略，不是系统路由 ID。
- 删除 reply.status 和 reply.clarification：Director 提交的是最终消息，后端绑定 ready；
  不再要求重复填写 communication_intent、content_points、expression_task。
- 旧 state_patch 心理字段不在当前工具输入中。旧数据读取和 Executor 兼容仍在领域模型内。
- 状态版本、执行回执、实际发送时间、幂等与事务由后端负责。

这里没有为了字段数量好看而把整个提案变成自然语言字符串。心理、记忆、计划请求的
含义不同，仍有各自结构；不需要的部分省略即可。是否将不同操作进一步拆为多个提交工具，
需要与原子提交及并发输入契约一起设计，不在本次参数清理中悄然改变调度协议。

### 起点：submit_initial_state

模型提交 subjective_state、open_conversation_threads、active_commitments 的具体理解。
移除新关切的 concern_ref、action_ref、depends_on_expression：起点不存在旧行动，
后端补 null/null/false。真实消息来源仍由模型选择，不能后端猜测证据。

### DayPlan：submit_day_plan

模型选择 proposal 或协作 reply；proposal 仅包括 blocks、assumptions、private_reason。
目标 plan_date 从 DayPlanContext.target_date 绑定，模型不能重复抄写或改日期。
块中的活动、时段、可用性、来源性质和假设仍是计划内容，不是系统元数据。
块 ID、正式版本、提交时间和事务由后端生成；连续覆盖、锁定块和来源检查不删除。

### 生活事件：submit_life_result

模型决定 outcome、reason、事件内容、心理作用、预计耗时、应对意向和是否修改日程。
within_response_scope 是模型对语义边界的判断，**不是批准凭证**；工作流仍检查 Director 的
实际答复、时间边界和扰动约束，不能靠一个 true 获得执行权限。
移除固定的 impact.scope、受保护承诺/长期后果空列表以及 handling.actual_outcome=null；
这些受当前机会契约约束，后端补齐。plan_proposal.plan_date 绑定本次分支本地日期。

### PersonaActor 保留入口：submit_expression

普通聊天不用它。独立表达入口仍支持 ready/needs_clarification 两种真实结果，
所以 status、messages、clarification 在此有语义，不能套用 Director 的精简规则删除。
素材引用由模型选择，素材权限和资源检查由后端完成。

### 对话摘要：submit_summary

只保留 summary、unresolved_items。消息范围、摘要版本和写库归宿全由后端负责；无需改动。

## 5. PersonWorld 七栏目共用查询

| 工具 | 模型负责 | 后端负责 |
| --- | --- | --- |
| search_world | question | workspace、文档白名单、top_k/token预算、固定 mix；删除 mode 技术参数 |
| list_graph_entities | 可选 name_contains、limit | 当前图版本与实体列表；name_contains 只是字面筛选 |
| get_graph_entity | entity_name | 当前图与权限 |
| get_entity_neighborhood | entity_name；可选 depth | 图范围与响应边界 |
| get_relation | source_entity、target_entity | 当前图；不创建关系 |
| locate_source_messages | retrieval_id；可选 context_turns、limit | 检索工件、Bundle 和 SQL 还原、消息权限 |
| get_message_context | message_ids；可选 context_turns | 同 Bundle 连续上下文与身份标注 |
| read_evidence_page | evidence_set_id；可选 offset、limit | 证据工件与消息分页 |
| analyze_evidence_dates | message_ids | 时区从宿主 timezone_name 绑定，默认沿用 Asia/Shanghai；删除模型 timezone 参数 |
| get_current_profile_section | section | 项目、图与 Publication 绑定；允许按需读取其他栏目 |
| get_active_corrections | 可选 primary_domain、fact_type、subject_kind、related_fact_id、limit | 项目、有效性、允许的 ChangeSet；过滤选择有语义，保留但不要求填齐 |
| get_context_module_spec | kind | 固定字段目录和合法 read_paths |
| list_context_modules | 可选 kinds、statuses | 本次冻结模块快照；默认当前/计划/暂停 |
| read_context_module | module_id；可选 field_paths | 模块版本、值和读依赖追踪 |

### 七个 submit_section

identity、life_context、social_world、agency、practices、life_course、relationship_with_user
分别保持已认可的细分内容模型。心理模型的倾向、场景、解释、不确定性和扮演建议是内容，
不是后端可替 Agent 填写的字段。

本次从工具中移除：

- 按字段位置唯一确定的 DimensionEntry.dimension_id；心理维度数组仍需 dimension_id 选择维度。
- 心理维度里重复的固定 model 标签，由对应类型默认值补齐。
- 固定 schema_version。
- 模块实例和引用的 revision：后端按本次固定快照绑定，未知 module_id 明确报错。
- module_assessment.status：根据是否提交模块推导；保留 explanation 由 Agent 解释。

现有 entry/module ID 仍可用于选择要更新的已知条目，新实例 ID 由后端分配。
reference_message_ids、context_module_refs 的使用位置仍由模型决定；不因为后端看过某条材料就
自动把它附到所有结论上。绑定后继续执行原栏目、心理值域、模块关系和引用校验。

## 6. 审核、图谱治理与 Alias

| Agent / 工具 | 模型负责 | 后端负责 / 本次收窄 |
| --- | --- | --- |
| Revision / 共用查询 | 上表的问题与已知引用 | 只读；不能通过参数选择另一个项目或批准版本 |
| submit_revision_turn | question、scope_proposal 或 understanding；问题、选项文字、影响和推荐 | 删除问题选项 id，由后端按本题顺序生成；用户选择仍绑定具体问题/版本 |
| Graph Patch / search_graph | question | 当前待纠正 workspace、查询策略与预算 |
| submit_graph_patch | operation_type、相应实体/关系/合并对象、拟写内容、reason、source_message_ids、cascade 与 regression_queries | 删除 operation_id、before_description、precondition_hash；编号后端生成，旧状态与 hash 从真实图读取；cascade 仅是待审核范围，不是执行授权 |
| Graph Regression / read_regression_query | 选择批准清单中的 query | 候选图、查询策略、实际结果缓存；不允许模型改写回归范围 |
| submit_graph_regression | 对每个 query 的 passed、explanation | 实际检索摘录、来源引用和预期变化由宿主补齐，模型不能自报这些材料 |
| Alias / list_person_nodes | 无 | 当前已筛选人物集合 |
| locate_original_mentions | names；可选 limit | 文档材料范围、真实命中原文；不因字面命中自动判定同一人 |
| query_lightrag | name | 当前图和固定查询策略 |
| submit_alias_resolution | source_entities、target_entity、reason、evidence | 项目、候选 ID、审核状态和真正合并；提交不会 amerge_entities |

回归 query、别名/实体名、用户圈选的 candidate_item 是已有对象的选择引用，不是要求模型
创造后台 ID。保留它们是为了避免后端猜测模型到底在评估或修改哪一项。

## 7. 全部工具统一错误契约

统一出口为 agent_runtime/tool_errors.py 与 submission.py，不各自发明错误格式。

| 场景 | 回执与下一步 |
| --- | --- |
| 类型、缺字段、互斥参数、未知枚举 | rejected / invalid_tool_arguments；validation_errors 包含 loc、msg、实际类型与有界值；correct_arguments |
| 传入已移除的内部字段 | extra_forbidden，删除该键；不把旧字段改名再试，不在服务端静默吞掉 |
| 引用或分页越界 | ToolInputError 指定字段，说明应从哪个工具返回值选择；不是 empty |
| 业务提案冲突 | rejected / invalid_proposal；message 说明约束和修正方向；committed=false；仍在当前工具回合修正 |
| 缺少宿主绑定、资源或服务配置 | error / tool_unavailable；report_failure，不提示模型补 branch_id、版本或凭据 |
| 网络超时/传输故障 | 统一 Controller 判断 retryable 和 attempts；参数错误不原样网络重试 |
| 未知工具、可回执的坏 JSON | 统一调用前拒绝；保留调用 ID，返回可用工具/格式说明供修正 |
| 权限不足或预算耗尽 | 不通过改参数绕过；明确不可恢复的原因 |
| 正常无匹配 | 正常 empty，不算工具错误；partial/不可用不能伪装成无事实 |
| 取消 | 统一取消信号和终态；不伪装为参数错误或主动等待 |

共同字段为 status、code/error、tool_name、target_tool_name、message、recoverable、retryable、
next_action、failure、validation_errors。业务提交回执另外标明 committed=false。
内层 execute_tool 错误保留实际工具名，参数路径加 arguments 前缀。
null、[]、{}、空字符串含义不同；对象内容省略时明确标记，不再回显为假 null。
模块绑定、日期注入和领域校验仍在工具参数解析阶段完成，错误路径保留 result 前缀。

Phoenix 记录真实执行与拒绝；一次错误在父级传播不算额外供应商调用。HTTP 成功但缺工具内容
仍是模型响应异常，不能虚构 tool_call 或编造成功结果。

## 8. 实现与验证边界

input_contracts.input_model 只按工具作者的显式责任规则构造窄 Pydantic 类型；
不按中文含义、正则、字段名后缀或模型猜测自动判定责任。各工具 adapter 注入宿主值，
随后校验原领域模型。LangChain 仍是唯一 Schema 转换/工具注册路径，Controller 仍是唯一循环。

Schema 变化进入已有检查点工具契约指纹；不沿用旧协议的成功提交。历史数据读取不删除。
本次不改数据库结构，不触发图谱写入，不重放测试聊天，不部署后声称云端异常已经解决。

### 本次验证

- 统一 Agent Runtime、Director 工具与提交、初始化、生活事件、Runtime 冒烟、
  PersonWorld v3 和审核流程的选定回归：276 项通过。
- 参数责任专项：6 项通过（与上述集合部分重合），包括宿主日期不可覆盖、
  原领域校验仍执行、图操作编号与目标校验、问题选项编号、七栏目实际工具 Schema、
  已移除字段的错误路径和修正提示。
- 本次核心改动的 Ruff 导入/静态检查与 git diff --check 通过。
- 以上为本地离线验证，未重启体验服务，也未进行真实云端 Agent 重跑。
