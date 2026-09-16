# 完整请求容量检查

## 改动位置

- `agent_runtime/controller.py`：压缩前的字符统计纳入工具 Schema 和消息
  additional_kwargs；每次模型执行建立统一容量作用域。
- `agent_runtime/capacity.py`：完整请求估算、usage 校正、输出预留与窗口守卫。
- `runtime_v1/cloud_models.py`、`events/cloud_client.py`：在实际 HTTP 请求前调用
  统一守卫。结构化重试和修复后的新请求也重新检查，不依赖 Prompt 文件推测载荷。

## 计算

序列化实际发送的整个 JSON：system、messages、工具调用、工具结果、工具 Schema、
回传的 reasoning_content/reasoning_details，以及 response_format 中的 Schema。
只统计实际请求体，不统计 Authorization 等 HTTP 凭据。
英文采用约四字符一 token，非 ASCII 保守按一字符一 token，另加协议开销估值。
这不是精确 tokenizer，不能保证供应商永不拒绝。

每次发出请求前要求：

```text
完整请求输入估值 × usage 校正系数
  + 实际请求输出额度
  + 安全余量
  <= 模型窗口配置
```

输出额度取实际 max_completion_tokens 或 max_tokens，因此思考与正文共享的
completion 额度不会漏留。JSON 中的参数文字也纳入输入估算，略微保守。
超限返回 input_context_limit（blocked），不进行网络重试、不悄悄降低输出额度，
也不删除供应商要求回传的推理字段。

## 窗口与校正

`AgentBudgetPolicy.model_context_windows` 可按实际 model 名配置已核实窗口。
未配置模型使用 `context_window_tokens=128000` 的保守运行上限，
这不是对 MiniMax 或任何模型真实规格的声明。安全余量默认 2048 tokens。
部署应依据实际供应商型号覆盖配置，不能由模型名称字符串猜窗口。

校正采用真实 prompt_tokens/input_tokens 与本次完整请求原始估值之比。
只向上修正，不因一次低 usage 激进扩大窗口；缓存 token 已在 prompt_tokens 中时不重复相加。
缺失 usage 则保留原估计。校正按端点和模型隔离，保存于 BudgetLedger，
随原有检查点恢复，不跨用户共享校正数据，也不把累计输入消费当作当前窗口占用。

Phoenix 状态事件提供 context_capacity 和 context_usage，可对照估值、实际输入、
输出预留、安全余量与窗口配置。Controller 的字符压缩阈值仍是提前维护上下文的机制；
真实请求守卫是最终安全检查，不会在此自动重跑整个复合 Agent 或改写它的请求。

## 参考与验证

借鉴 [Codex 历史容量估算](https://github.com/openai/codex/blob/a592c38c16cdd7623dacc9168926ebccedfb67d3/codex-rs/core/src/context_manager/history.rs#L440)
使用本地近似量与供应商 usage 结合的方向，不要求拥有供应商 tokenizer。

`backend/tests/agent_runtime/test_capacity.py` 覆盖工具/输出 Schema、推理字段、
输出预留、模型窗口覆盖、usage 缺失和隔离、恢复后校正，以及真实客户端在 HTTP 前拒绝超限请求。
