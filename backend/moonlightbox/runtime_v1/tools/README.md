# Runtime 工具维护入口

工具的参数定义、模型可见说明、执行逻辑和可修复错误放在对应文件，不能回填到 Agent 或 Prompt。

| 文件 | 职责 |
| --- | --- |
| submit_decision.py | Director 决策参数、待回复处理、唤醒和协作互斥检查；接入来源校验服务 |
| submit_day_plan.py | DayPlan 提交、日期与时间网格、锁定块及可引用来源检查 |
| submit_expression.py | 表达提交、素材候选授权、资源检查和检查点素材恢复 |
| submit_life_result.py | 生活事件模式的结果提交 |
| read_runtime_result.py | 分页工具参数、结果引用权限与分页边界 |
| 其余查询工具文件 | 各自查询参数、说明和数据服务调用 |

Agent 文件只装配上下文、工具和统一循环，解释运行结果；不再内联提交校验回调。
通用提交回执与动态 args_schema 的构造由 agent_runtime/submission.py 提供。

## 不移动的共享职责

- LifeDecision、DayPlanTurn、ExpressionResult 等是工具、工作流、数据库边界共用的领域契约，
  保留在 schemas.py / 对应 contracts 文件。工具直接引用这些模型作为参数结构，
  其中的字段和 model_validator 已在工具调用时执行，不复制第二套参数模型。
- Executor 保留发送和写入时的最终权限、资源、版本、时间与事务检查。
- 底层数据服务保留自身的数据边界检查，工具可以调用服务，不必把 SQL 或存储权限判断复制过来。
- Harness 保留通用的参数解析异常回执、调用批次限制、取消与预算保护。

本次是 Runtime 工具代码组织整理，不修改 PersonWorld 的领域审核流程，不改变 Agent 行为规则。
