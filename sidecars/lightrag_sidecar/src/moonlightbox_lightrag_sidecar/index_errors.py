"""将供应商异常和持久化抽取错误转换为稳定契约，不向客户端返回原始正文。"""

import re
from collections.abc import Mapping


class IndexFailure(RuntimeError):
    def __init__(
        self, code: str, message: str, status_code: int,
        document_id: str | None = None, chunk_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = status_code in {429, 503, 504}
        self.detail = dict(
            code=code, message=message, document_id=document_id, chunk_id=chunk_id,
        )


def index_failure(error: object, document_id: str | None = None) -> IndexFailure:
    texts = []
    statuses = set()
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        texts.append(str(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            statuses.add(status)
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    text = "\n".join(texts)
    lower = text.lower()
    chunk = re.search(r"(?<![\w-])(?:[\w-]+-)?chunk-[\w-]+", text)
    statuses.update(int(code) for code in re.findall(
        r"(?:http|error code|status(?:_code)?)\s*[:=]?\s*([45]\d{2})\b", lower,
    ))
    # 本地配置/初始化错误不是供应商临时故障，必须终止自动恢复。
    if any(marker in lower for marker in (
        "未配置 lightrag_sidecar_llm_api_key", "未配置 lightrag_sidecar_embedding_api_key",
        "missing lightrag_sidecar_llm_api_key", "missing lightrag_sidecar_embedding_api_key",
    )):
        code, message, status = (
            "lightrag_configuration_missing",
            "LightRAG Sidecar 缺少模型服务配置，请补充凭据后重试。", 424,
        )
    elif any(marker in lower for marker in (
        "new_sensitive", "content_filter", "content_policy_violation",
    )):
        code, message, status = (
            "lightrag_content_rejected",
            "模型服务拒绝处理部分聊天内容，资料已保留。请检查受阻片段后再恢复。", 422,
        )
    elif statuses & {401, 403} or "authenticationerror" in lower:
        code, message, status = (
            "lightrag_provider_authentication",
            "图谱模型服务认证失败，请检查抽取服务的凭据后再恢复。", 422,
        )
    elif statuses & {400, 422} or any(marker in lower for marker in (
        "unprocessableentityerror", "badrequesterror",
    )):
        code, message, status = (
            "lightrag_provider_request_rejected",
            "图谱模型服务拒绝了请求，请检查模型配置及受阻资料后再恢复。", 422,
        )
    elif 429 in statuses or "ratelimiterror" in lower:
        code, message, status = (
            "lightrag_rate_limited", "图谱模型服务暂时限流，将稍后恢复。", 429,
        )
    elif 504 in statuses or "timeout" in lower or "timed out" in lower:
        code, message, status = "lightrag_timeout", "图谱模型请求超时，将稍后恢复。", 504
    elif any(s >= 500 for s in statuses) or any(marker in lower for marker in (
        "connectionerror", "connecterror",
    )):
        code, message, status = (
            "lightrag_unavailable", "图谱模型服务暂时无法连接，将稍后恢复。", 503,
        )
    else:
        code, message, status = (
            "lightrag_index_error",
            "资料整理遇到内部错误，已有进度已保留，需要检查后恢复。", 422,
        )
    return IndexFailure(code, message, status, document_id, chunk.group() if chunk else None)


def document_failures(statuses: Mapping[str, object]) -> list[IndexFailure]:
    failures = []
    for document_id, record in statuses.items():
        status = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
        if getattr(status, "value", status) == "failed":
            error = (
                record.get("error_msg") if isinstance(record, dict)
                else getattr(record, "error_msg", None)
            )
            failures.append(index_failure(error or "unknown index failure", document_id))
    return failures
