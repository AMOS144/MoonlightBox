"""表达提交工具：管理素材候选、恢复授权与可修复的素材参数错误。"""

from moonlightbox.agent_runtime.submission import result_submission_tool

from ..expression_contracts import ExpressionResult


def supplied_assets(value):
    """只提取工具/装配材料中的可用候选；不会从模型最终输出构造授权。"""
    result = {}
    if isinstance(value, dict):
        # 原文回读也是已提供材料；真正的发送资格仍由领域校验检查。
        if value.get("type") == "sticker" and value.get("asset_ref"):
            ref = value.get("source_ref") or value.get("source_id")
            if ref:
                result[value["asset_ref"]] = value.get("usage_refs") or [ref]
        for item in value.get("sticker_candidates", []):
            if item.get("available"):
                result[item["asset_ref"]] = item.get("usage_refs", [])
        for child in value.values():
            result.update(supplied_assets(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(supplied_assets(child))
    return result


class ExpressionSubmission:
    """每轮独立持有授权材料；不能从模型提交的 asset_ref 反向生成授权。"""

    def __init__(self, context, *, asset_validator=None, exposed_assets=None):
        self.context = context
        self.asset_validator = asset_validator
        self.exposed_assets = exposed_assets if exposed_assets is not None else set()
        self.authorized = supplied_assets(context)

    def restore(self, results):
        """检查点恢复时同步素材来源，保证发送阶段仍可核对授权。"""
        recovered = supplied_assets(results)
        self.authorized.update(recovered)
        self.exposed_assets.update(recovered)

    def validate(self, value, validation):
        if not isinstance(value, ExpressionResult):
            return "invalid_expression_result"
        self.authorized.clear()
        self.authorized.update(supplied_assets(self.context))
        self.authorized.update(supplied_assets(validation.tool_results))
        for part in value.messages:
            if part.kind == "sticker":
                if part.asset_ref not in self.authorized:
                    return "表情包必须来自当前材料或成功检索的可用候选，请查询或改用文字"
                if self.asset_validator is not None:
                    try:
                        self.asset_validator(part.asset_ref)
                    except ValueError as error:
                        return f"素材不可用，请重新选择或改用文字：{error}"
        return None

    def build_tool(self):
        return result_submission_tool("submit_expression", ExpressionResult, self.validate)
