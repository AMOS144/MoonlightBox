# PersonWorld：Send 并发与阶段快照

## 实现范围

PersonWorldCoordinatorV3 不再自行创建 ThreadPoolExecutor。外层 LangGraph 通过
ainvoke 顺序执行草稿、模块补全、审核汇总；每个调查批次由子图通过 Send 扇出栏目，
async 节点执行后，用 completed_sections 的累加 reducer 汇合。

同步 Worker 继续使用 run()；该入口调用 asyncio.run(arun())。已有事件循环的调用方
必须 await arun()，同步入口会明确报错，避免嵌套事件循环。

这是**异步图编排 + 同步工作单元隔离**，不是全链路无工作线程：

- 单栏仍使用同一个 AgentLoopController.run，未另写模型循环或复制重试策略。
- 现有同步工具、SQLAlchemy Session、SQLite 检查点通过统一的
  agent_runtime.async_execution.run_sync_owned 在工作线程执行。
- 每栏 Session、调查材料和模块读取跟踪器在工作单元内创建；Session 也在该处关闭。
- 短业务事务由协调协程同步提交，不跨线程共享 Coordinator 的 Session。
- 同步调用不阻塞生产环境的编排事件循环；内存 sqlite:// 测试库仍在原线程串行，
  因为其数据库绑定连接／线程。此例外不用于文件数据库。

暂未把所有工具改成原生异步数据库／HTTP 接口，不能将 to_thread 描述为纯协程网络 I/O。

## 栏目间如何读取

1. 草稿批次：七栏读取批次开始时的已发布背景／已接受结果快照。
2. 等草稿批次汇合后，从 life_context 建立固定模块快照。
3. practices、agency 及其他依赖模块的栏目并发补全，读取上一阶段完整 Profile。
4. identity 最后补全，读取其他栏目的已接受修订结果，同时更新 overview。
5. 所有阶段完成后组装候选审核，不直接覆盖已发布 Profile 或图谱。

每个 Send 获得深拷贝的独立输入。完成结果可以立即落库，便于展示进度和恢复；
但不能改变同批次其他任务的冻结输入。因此某栏完成得快，不会改变兄弟栏目的调查条件。

get_current_profile_section 按需提供完整的跨栏冻结内容。profile_snapshot 不重复
注入初始 HumanMessage，避免七个 Agent 每轮都携带一整份 Profile。旧检查点没有
完整快照时仍使用当时保存的摘要，不偷偷替换恢复任务的背景。

## 并发、失败与恢复

- max_concurrency 使用 person_world_section_concurrency，保持原配置，默认 3。
- 单栏普通异常保存为该栏失败，不丢弃其他已完成栏目的结果。
- 保留既有 run ID、batch accepted 清单和单栏检查点 owner ID；普通重启不重新调查
  已接受的栏目。批次完成标记只在全部 Send 汇合后写入。
- Python 任务取消会设置局部协作取消信号；与父 Job 取消回调共同传入原 Controller。
- 对不可中止的同步 I/O，等待工作单元真实退出并关闭资源后再传播 CancelledError；
  不留下仍在执行的后台线程，也不接受取消后的迟到结果。
- 本改动没有增加新的硬性调查轮数或修改模型、Prompt、发布审批权限。

## 验证

烟测覆盖：并发上限为 2 时实际同时执行两栏、事件循环仍可调度、同批次输入不被
提前完成的栏目污染、下一阶段可读新结果、业务写入位于协调线程、单栏失败隔离、
取消等待资源关闭、父 Job 取消传递，以及原有 v3 全流程和检查点恢复。

人物世界与统一 Agent Runtime 回归共 144 项通过；取消回调调整后，新增的 5 项
异步烟测再次全部通过。相关文件 Ruff、后端 compileall 与 git diff --check 通过。

未请求真实云模型、未重新编译体验资产，也未重启正在运行的服务。

## 依据

采用 LangGraph 官方的 Send map-reduce、异步 ainvoke 与 max_concurrency 配置：
[Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)。
不把 Send 的并发能力误认为跨并行任务即时共享可变状态。
