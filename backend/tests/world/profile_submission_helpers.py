"""旧领域夹具转当前模型输入；仅测试使用，生产不会容忍旧参数混入工具。"""


def section_input(value):
    def walk(node, key=""):
        if isinstance(node, list):
            return [walk(item, key) for item in node]
        if not isinstance(node, dict):
            return node
        result = {name: walk(item, name) for name, item in node.items()}
        result.pop("schema_version", None)
        result.pop("revision", None)
        result.pop("model", None)
        if "dimension_id" in result and key != "dimensions":
            result.pop("dimension_id")
        if key == "module_assessment":
            result.pop("status", None)
        return result

    return walk(value)
