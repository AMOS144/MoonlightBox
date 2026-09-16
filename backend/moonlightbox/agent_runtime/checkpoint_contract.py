"""检查点身份包含工具及提交契约；不读取闭包中的业务数据、Session 或密钥。"""

import hashlib
import json
from functools import partial
from types import CodeType


def _code(value):
    if isinstance(value, CodeType):
        return {
            "bytecode": value.co_code.hex(),
            "constants": [_code(item) for item in value.co_consts],
            "names": value.co_names,
            "arguments": (value.co_argcount, value.co_kwonlyargcount, value.co_posonlyargcount),
            "flags": value.co_flags,
        }
    if isinstance(value, (tuple, frozenset)):
        items = [_code(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, frozenset) else items
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _implementation(callback, seen=None):
    """只取可执行结构；行号、绝对路径、实例地址不应使普通重启失效。

    闭包中的可调用对象纳入指纹，覆盖提交工具封装的 validator；任意闭包数据不做
    dump。外部配置／全局规则表变化由显式 contract_version 表达。
    """
    seen = set() if seen is None else seen
    if callback is None:
        return None
    if isinstance(callback, partial):
        return _implementation(callback.func, seen)
    function = getattr(callback, "__func__", callback)
    if not hasattr(function, "__code__"):
        function = getattr(type(function), "__call__", None)  # noqa: B004 — 检查实现，不判断可调用性
    code = getattr(function, "__code__", None)
    if code is None:
        return None
    if id(function) in seen:
        return {"recursive": getattr(function, "__qualname__", "")}
    seen = {*seen, id(function)}
    nested = []
    for cell in getattr(function, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and not isinstance(value, type):
            nested.append(_implementation(value, seen))
    return {"code": _code(code), "callbacks": nested}


def checkpoint_contract_fingerprint(spec):
    tools = []
    for registered in sorted(spec.tools, key=lambda item: item.tool.name):
        tool, contract = registered.tool, registered.contract
        schema = tool.args_schema
        if not isinstance(schema, dict):
            schema = tool.get_input_schema().model_json_schema()
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "schema": schema,
                "version": contract.contract_version,
                "submission": registered.is_submission,
                "side_effect": contract.side_effect,
                "permissions": sorted(contract.required_permissions),
                "implementation": _implementation(getattr(tool, "func", None)),
                "async_implementation": _implementation(getattr(tool, "coroutine", None)),
                "normalizer": _implementation(contract.result_normalizer),
                "projector": _implementation(contract.model_result_projector),
            }
        )
    payload = {
        "protocol": "native-submission-v3",
        "state_format": spec.state_version,
        "contract": spec.contract_version,
        "submission_tool": spec.submission_tool_name,
        "tools": tools,
        "input_refresh": _implementation(getattr(spec, "refresh_inputs", None)),
        "work_snapshot": _implementation(getattr(spec, "snapshot_work_state", None)),
        "work_restore": _implementation(getattr(spec, "restore_work_state", None)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
