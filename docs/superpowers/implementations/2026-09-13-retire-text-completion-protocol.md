# 统一提交工具交付，退役文本完成协议

## 唯一完成路径

Agent 调用指定的提交工具 → 工具参数与领域校验 → 接受回执 → Controller 返回 typed 产物。
参数错误返回带路径的工具错误；领域错误返回 rejected，Agent 可继续修改。
普通文本即使是合法 JSON 也不能交付。运行时提醒使用指定工具；持续不提交仍受无进展和预算保护。

所有 AgentSpec 必须显式声明 submission_tool_name，并注册真正的提交工具。
不是看名字的 submit_ 前缀，也不能让普通查询返回一段 accepted JSON 来伪造成功。

## 删除内容

- AgentSpec.output_parser、final_validator、fallback_factory、最终修复配置。
- Controller 的 compile_final 图节点、文本产物解析、最终校验和最终结构修复分支。
- 已接受工具结果包装成 AIMessage 再解析的绕路。
- PersonWorld 栏目调查后额外调用结构化编译器的完成路径。
- Revision 的独立结构化调查计划、固定工具序列与文本回合编译路径。
- 旧的 parse_output、_parse_decision、_parse_proposal 和无工具最终收束辅助函数。
- Prompt 根据工具命名前缀猜交付方式的逻辑。

业务层独立使用的结构化 API（例如用户确认后生成 Patch 的服务）不是 AgentLoopController
的第二完成协议，本次没有删除这些服务或放松确认、授权、事务规则。

## 各入口

| 入口 | 指定提交工具 |
| --- | --- |
| Director | submit_decision |
| DayPlan | submit_day_plan |
| PersonaActor | submit_expression |
| 生活事件协作 | submit_life_result |
| 聊天摘要维护 | submit_summary |
| PersonWorld 七栏目 | submit_section，绑定各自栏目模型 |
| PersonWorld 纠正会话 | submit_revision_turn |
| 手动计划恢复脚本 | submit_recovered_plan |

提交工具内的 SubmissionReceipt 携带已经通过 Pydantic 和领域检查的原对象。
Controller 直接返回对象，不进行第二次 JSON 解析，不为交付额外请求模型。
模型可见回执仍是 JSON，供协议配对和 Phoenix 观察；它不承担内部对象运输。

## PersonWorld 与纠正会话

栏目直接使用原生工具模型，继续拥有原来的查询工具、独立 Prompt、栏目类型及引用范围检查。
校验位于 tools/submit_section.py；工具按调用时已读取的材料检查引用。
不新增“必须检索多少轮”的规则，也不以此次架构整理为由更改栏目内容目标。

Revision 也使用原生 bind_tools，由模型决定是否查询、读取哪条消息以及何时交付一个回合。
question 产物通过提交工具正常接受，回执标为 waiting_for_user，交还用户控制权；
不是将提问当成校验失败。其余合法回合标为 succeeded。
提交回合不会批准 Profile/Graph Patch；用户的多步确认和最终 hash 批准仍由业务服务负责。

纠正上下文在调查中只装配到内存。工具接受后才持久化最后一次模型实际读取的上下文快照，
避免一个用户回合的多次工具调查重复插入同一版本的快照。

## 恢复与副作用

accepted 仍不等于 committed。消息发送、计划写入、Profile 和图谱修改继续交给业务 Executor。
取消、输入版本检查、网络退避、请求容量校正和累计预算没有被移除。

新检查点命名空间加入 submission-v2，避免恢复到已删除的 compile_final 节点，
或者复用旧文本完成结果。历史业务数据保持可读；旧执行检查点不自动跨协议续跑。
新协议的已接受结果可以原样复用，失败恢复继续保留新协议内已完成的工作。

## 验证范围

离线测试覆盖错误回执后修正、typed 对象原样交付、等待用户、不允许文本绕过、
独立提交批次、显式 Prompt 配置、检查点重试与复用，以及七栏目候选生成和纠正会话。
没有启动真实云端生成，没有向体验分支发送测试消息，也没有重启运行服务。

验证结果：254 项相关回归通过。排除 1 项此前已知失败的
`test_persona_reads_frozen_profile_without_mutating_origin`；
该测试要求 Persona 上下文包含 life_context，但现有装配器不提供此字段，
本次没有改变 Persona 的上下文取舍。
