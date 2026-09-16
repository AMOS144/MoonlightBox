"""Runtime 的 LangChain 查询工具与无写入副作用的提案提交工具。

每个模块独立维护自己的参数 schema、模型可见说明、内部查询模板和执行闭包。
跨工具与 Executor 共用的领域数据模型继续位于 contracts/schemas，避免两份定义漂移。
"""
