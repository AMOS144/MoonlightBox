# 技能渐进读取与固定工具执行入口

## 本次边界

Director 模型侧固定提供 `read_skill` 和 `execute_tool`。表达业务工具不再作为独立 Schema
绑定到 Director；读取 speaking 后，后续请求仍使用相同工具定义。已有基础工具和提交工具
保持原入口，PersonaActor 及其最终图文提交链路不在本次迁移范围内。

固定 Schema 减少前缀变化，但不承诺模型供应商缓存命中。技能读取结果和工具结果仍进入
当前 Agent 上下文；这不是上下文隔离方案。

## 作者维护与模型调用

- `runtime_v1/skills/speaking/SKILL.md` 维护行为指导，frontmatter 的 tools 声明依赖。
- 读取时从已绑定的 LangChain 工具生成参数格式，附加到技能正文；不手工维护第二份 Schema。
- 工具不可用时，读取结果明确告知不可调用，不凭技能声明获得权限。
- `agent_runtime/tool_dispatch.py` 只在本次调用方提供的白名单中查找工具；不 eval、不动态 import，
  不访问任意数据库函数，也不分发结果提交工具。

```json
{"skill":"speaking"}
```

读完以后，按需调用：

```json
{
  "tool_name": "get_style_examples",
  "arguments": {
    "situation": "当前互动的具体处境",
    "intent": "想参考哪一种表达方式",
    "speech_mode": "reply",
    "limit": 4
  }
}
```

执行返回 `tool_name` 和 `result`；result 是原业务工具结果。参数错误由内层 LangChain／Pydantic
校验后进入 Controller 的标准错误回执，模型可以修正并继续。通用 arguments 必须保持开放对象，
因此仅 execute_tool 的供应商侧 strict 为 false，其他工具保留原严格约束；这不关闭业务参数校验。

## 统一运行时保障

- 一次执行入口调用只分发一次业务调用；没有第二个 Agent 循环、独立重试预算或动态工具加载。
- 内部只读契约的资源和权限取并集，超时与结果预算保守取最小值；仍由 Controller 串行运行。
  当前仅装配风格检索，后续加入不同预算的能力时需重新审查，不宣称支持任意工具。
- 当前分支、模型版本、冻结历史边界和素材范围由 StyleService 的绑定闭包控制，不让模型传这些标识。
- Phoenix 在同一调用 Span 中记录 dispatcher、target_name、target_arguments，不多造一个工具 Span
  将一次调用计成两次。运行时调用计数使用 execute_tool，业务目标通过属性区分。
- 已归一化的完整业务结果保存在检查点；恢复时同时重建外层回执与内层原始分页引用，并恢复素材候选。
- 内部工具参数及提交外契约指纹纳入 execute_tool 的检查点契约；技能正文变化也会改变读取工具契约。
  这些版本信息不放进模型侧 Schema，因此代码更新可使旧检查点失效，却不因正文版本动态改变工具前缀。
- 用户取消与超时沿用原机制；不能强制中断的同步 I/O 仍只在支持的边界退出。

## 验证范围

新增离线冒烟验证：跨轮绑定 Schema 相同、隐藏业务工具不在模型工具表中、Skill 回执有真实参数格式、
错误参数反馈后能修正提交、内层契约更新、分页恢复、实际工具目标的 Phoenix 属性。没有发起真实聊天，
没有重启开发服务，也没有修改业务数据库。

本轮完整 `agent_runtime + runtime_v1` 回归为 179 通过、1 失败；失败是聊天冒烟中的
`input_resolutions 不能处理未来消息`，不是工具参数或分发错误。将该用例与新增入口、
执行策略测试一起重跑后 12 项全部通过，说明尚有时序相关的不稳定表现，不能据此宣称
完整回归稳定全绿。本次没有顺带修改虚拟钟或放宽提交时的时间检查。
