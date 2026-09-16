"""PersonWorld 工具结果的稳定比较视图，不改变模型回执和真实引用身份。"""

from copy import deepcopy


def investigation_result(value):
    if not isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, (list, tuple)):
        return [investigation_result(item) for item in value]
    ephemeral = {
        "retrieval_id",
        "evidence_set_id",
        "retrieval_query_ids",
        "context_window_ids",
        "retrieval_provenance",
        "query_id",
        "context_window_id",
    }
    return {key: investigation_result(item) for key, item in value.items() if key not in ephemeral}


def comparison_for(name, artifacts):
    def project(result):
        value = deepcopy(result)
        if isinstance(value, dict) and value.get("evidence_set_id"):
            # 工具可能只返回分页目录；比较其实际正文，不能把两个不同证据集判成重复。
            try:
                value["comparison_messages"] = artifacts.evidence_set_rows(value["evidence_set_id"])
            except ValueError:
                # 错误回执可能携带无效编号；比较阶段不能截断模型修正参数的机会。
                value["comparison_messages"] = None
        return investigation_result(value)

    return (
        project
        if name
        in {
            "search_world",
            "locate_source_messages",
            "get_message_context",
            "read_evidence_page",
        }
        else lambda value: value
    )
