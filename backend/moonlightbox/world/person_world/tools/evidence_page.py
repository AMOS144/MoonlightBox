"""给原生工具回环返回真正可阅读的消息正文，而不是只返回消息 ID。"""

from langchain_core.tools import StructuredTool
from pydantic import Field

from ..contracts.context_modules import ProfileModel


class EvidencePageArgs(ProfileModel):
    evidence_set_id: str = Field(
        description="locate_source_messages/get_message_context 返回的 evidence_set_id；不是 retrieval_id 或 message_id"
    )
    offset: int = Field(
        default=0, ge=0, description="消息条目偏移，首次 0；续页原样使用 next_offset，不是字符偏移"
    )
    limit: int = Field(
        default=20, ge=1, le=40, description="本页最多消息条数；next_offset=null 表示已读到末尾"
    )


def evidence_page(artifacts, evidence_set_id, offset=0, limit=20):
    from moonlightbox.agent_runtime.tool_errors import ToolInputError

    try:
        rows = artifacts.evidence_set_rows(evidence_set_id)
    except (KeyError, ValueError) as error:
        raise ToolInputError(
            "evidence_set_id 不属于当前调查；使用原文工具返回的证据集合 ID，不是 retrieval_id 或消息 UUID",
            field="evidence_set_id",
        ) from error
    if offset > len(rows):
        raise ToolInputError(f"offset 超出证据集合长度 {len(rows)}；首次填 0，续页用 next_offset", field="offset")
    page = rows[offset : offset + limit]
    return {
        "evidence_set_id": evidence_set_id,
        "messages": list(page),
        "total": len(rows),
        "next_offset": offset + len(page) if offset + len(page) < len(rows) else None,
    }


def build_evidence_page_tool(artifacts):
    def read_evidence_page(evidence_set_id, offset=0, limit=20):
        return evidence_page(artifacts, evidence_set_id, offset, limit)

    return StructuredTool.from_function(
        read_evidence_page,
        name="read_evidence_page",
        args_schema=EvidencePageArgs,
        description=(
            "分页读取刚定位的原始消息正文及发送者/时间；用 next_offset 继续，不把首屏当全部历史。"
        ),
    )


def project_native_search(result):
    if not isinstance(result, dict):
        return result
    return {
        **{key: value for key, value in result.items() if key != "references"},
        "reference_count": len(result.get("references", [])),
        "references": [
            {key: value for key, value in ref.items() if key != "chunk_content"}
            for ref in result.get("references", [])
        ],
    }


def project_native_messages(artifacts, result):
    if not isinstance(result, dict) or not result.get("evidence_set_id"):
        return result
    return {**result, **evidence_page(artifacts, result["evidence_set_id"])}
