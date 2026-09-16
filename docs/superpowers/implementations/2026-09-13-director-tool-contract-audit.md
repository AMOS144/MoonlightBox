# Director 工具契约与错误回执审查

## 范围

检查普通 Director 实际装配的 10 个原生工具，以及 speaking 技能通过
execute_tool 调用的 get_style_examples。不新增 Agent、注册机制或错误重试循环；
仍由 LangChain StructuredTool / bind_tools 注册，AgentLoopController 统一执行、
回传 ToolMessage、重试和记录 Phoenix。此审查不等于云端聊天已重新验收。

## 逐项契约

| 工具 | 输入与边界 | 参数错误与恢复 |
| --- | --- | --- |
| search_memory | 非空 query，2–300 字符；limit 1–12；include_original；身份、快照和范围闭包绑定 | 空白查询在 Schema 校验；依赖失败不能解释为无记忆；一侧历史不可用保留另一侧并标 partial |
| search_conversation | 非空 query，最多 1200 字符；limit 1–20；统一搜索历史和分支 | 参数错误先于查询；历史依赖不可用不吞掉分支结果 |
| read_conversation | before_ref / around_ref 二选一，首次都省略；limit 1–100 | 空引用、互斥参数在 Schema 校验；越界引用指出具体字段；快照丢失是服务错误，不是无历史 |
| get_subjective_state | concern_ref 可省略或 null，指定时必须是已知关切 | 空字符串与未知引用明确报错；可省略引用重新读取全量关切 |
| get_profile_section | 七栏目 Literal，不允许任意路径 | 非法栏目参数错误；整份绑定缺失为服务错误；单栏目未编译标 not_compiled |
| get_recent_life_events | limit 1–30；before 为返回的 ISO occurred_at | 日期解析在参数层；带时区游标转 UTC 后查询；只读已提交且已发生的模拟事件 |
| read_runtime_result | 本次运行/恢复检查点中的 result_ref；offset >= 0 | 引用与越界 offset 单独定位；next_offset=null 是结束，不是错误；page 是 JSON 文本片段 |
| read_skill | skill 枚举；resource 只能选该技能的已列出资源 | 未知资源指出 resource 和可用资源；不是任意文件读取 |
| execute_tool | tool_name + JSON 对象 arguments；只分发已绑定的只读工具 | 未知名称定位 tool_name；内部错误保留 target_tool_name，字段路径加 arguments 前缀；不动态注册 Schema |
| submit_decision | 当前 DirectorSubmission；speak 必须 result.reply；提交工具单独调用 | 真实窄模型排除旧表达/心理字段；缺回复明确指向 result.reply；缺待处理消息列出具体引用；验收不等于落库 |
| get_style_examples（内部） | 情境模式要求 situation + intent；素材模式要求已知 asset_ref；usage_cursor 仅用于同一素材分页 | 两种模式在模型参数层校验；素材授权/资源/游标错误定位具体字段；不引入视觉模型，不扩大素材权限 |

所有工具的嵌套参数均通过实际 convert_to_openai_tool 的 Schema 检查字段说明；
不是只检查领域模型直接调用 model_json_schema 的结果。

## 统一错误语义

- `recoverable=true`：模型可修改参数；`retryable=false`，不原样网络重试参数错误。
- `validation_errors` 的 `loc`、`msg` 指明问题；Pydantic 错误同时回显有界的实际类型和值。
- 显式引用/分页错误由 ToolInputError 携带字段路径。业务验收拒绝仍可使用解释性的 message；
  不能假装每条跨字段或事务冲突都具有单字段定位。
- 可空字段的 null 是合法值，不能统一提示改成 []；空字符串、空对象和 null 不能互相替代。
- 对象因体积或信息边界而省略时明确标记 `[value omitted: object]`，不再回显为 null，
  避免模型误判实际传参类型。
- 服务不可用不建议“修改参数”；网络错误沿用 Controller 的传输重试；未知内部异常不回显内部细节。
- 未知工具、参数 JSON 损坏且调用 ID 可用时，仍由统一运行时构造工具回执；不冒充工具执行成功。
- 真实模型调用、工具拒绝和服务失败仍以 Phoenix 为准，父级异常传播不是额外供应商调用。

## 回归边界

新增 test_director_tool_audit.py 覆盖全部 11 个工具入口的失败回执、内层工具路径、
Schema 字段说明、分页结束与越界、服务错误与输入错误区分。
test_director_submission_contract.py 保留真实嵌套 Schema、最小回复、旧字段拒绝、
可空参数、顶层错位字段等回归。

没有重放用户消息、修改分支资料或放宽素材/引用权限。此次改动不能保证 MiniMax 不再返回
finish_reason=tool_calls 却缺少 tool_calls 的响应；该异常仍应被明确记录而不是伪造调用补齐。
