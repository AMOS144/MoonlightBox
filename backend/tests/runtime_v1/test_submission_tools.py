"""直接调用真实工具，检查迁移后错误仍在工具交互内返回。"""

from types import SimpleNamespace

import pytest
from moonlightbox.agent_runtime.submission import submission_context
from moonlightbox.runtime_v1.tools.submit_day_plan import build_submit_day_plan_tool
from moonlightbox.runtime_v1.tools.submit_expression import ExpressionSubmission
from pydantic import ValidationError


def test_expression_tool_rejects_unknown_candidate_then_accepts_text():
    submission = ExpressionSubmission({})
    tool = submission.build_tool().tool
    with submission_context(SimpleNamespace(tool_results=[])):
        rejected = tool.invoke(
            {
                "result": {
                    "status": "ready",
                    "messages": [{"kind": "sticker", "asset_ref": "invented"}],
                }
            }
        )
        accepted = tool.invoke(
            {"result": {"status": "ready", "messages": [{"kind": "text", "text": "好呀"}]}}
        )
    assert rejected["status"] == "rejected"
    assert accepted["status"] == "accepted"
    assert accepted["committed"] is False


def test_expression_tool_reports_unavailable_resource():
    def unavailable(ref):
        raise ValueError("asset_unavailable")

    submission = ExpressionSubmission(
        {"sticker_candidates": [{"available": True, "asset_ref": "known", "usage_refs": ["m1"]}]},
        asset_validator=unavailable,
    )
    with submission_context(SimpleNamespace(tool_results=[])):
        result = submission.build_tool().tool.invoke(
            {"result": {"status": "ready", "messages": [{"kind": "sticker", "asset_ref": "known"}]}}
        )
    assert result["status"] == "rejected"
    assert "asset_unavailable" in result["message"]


def test_plan_evidence_requirement_is_checked_by_tool_schema():
    # 该规则已在共享领域模型中，工具在执行回调前即拒绝，而不是等 Executor 才发现。
    from datetime import date
    tool = build_submit_day_plan_tool(SimpleNamespace(target_date=date(2026, 9, 13))).tool
    with pytest.raises(ValidationError, match="需要已提供的真实引用"):
        tool.invoke(
            {
                "result": {
                    "proposal": {
                        "blocks": [
                            {
                                "start": "00:00",
                                "end": "24:00",
                                "activity": "工作",
                                "basis": "profile_inference",
                                "evidence_ids": [],
                            }
                        ],
                    }
                }
            }
        )
