"""MiniMax 官方错误码到本地错误码的映射。

依据 platform.minimax.cn/docs/api-reference/errorcode：同属 HTTP 429 的
频率超限（1002，可稍后重试）与 Token Plan 用量上限（2056，须等计费窗口）
必须区分，否则配额耗尽会被当作普通限流反复重试。
"""

from types import SimpleNamespace

import httpx
import pytest

from moonlightbox.agent_runtime.resilience import classify_failure
from moonlightbox.runtime_v1.cloud_models import (
    RuntimeCloudInferenceError,
    _minimax_status,
    _response_message,
)


def openai_style_429(minimax_code: int) -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": f"已达到 Token Plan 用量上限 ({minimax_code})",
                "http_code": "429",
            },
        },
    )


def test_token_plan_quota_is_not_plain_rate_limit():
    with pytest.raises(RuntimeCloudInferenceError) as caught:
        _response_message(openai_style_429(2056), is_minimax=True)
    assert caught.value.code == "quota_exhausted"
    assert caught.value.diagnostic["minimax_status"] == 2056


def test_minimax_frequency_limit_stays_retryable():
    with pytest.raises(RuntimeCloudInferenceError) as caught:
        _response_message(openai_style_429(1002), is_minimax=True)
    assert caught.value.code == "rate_limited"


def test_minimax_base_resp_business_error_on_http_200():
    response = httpx.Response(
        200, json={"base_resp": {"status_code": 1026, "status_msg": "输入内容涉敏"}}
    )
    with pytest.raises(RuntimeCloudInferenceError) as caught:
        _response_message(response, is_minimax=True)
    assert caught.value.code == "provider_input_rejected"


def test_minimax_codes_require_minimax_host():
    # 非 MiniMax 供应商的 429 不得套用 MiniMax 错误码表。
    with pytest.raises(RuntimeCloudInferenceError) as caught:
        _response_message(openai_style_429(2056), is_minimax=False)
    assert caught.value.code == "rate_limited"


def test_unknown_minimax_code_falls_back_to_http_mapping():
    with pytest.raises(RuntimeCloudInferenceError) as caught:
        _response_message(openai_style_429(9999), is_minimax=True)
    assert caught.value.code == "rate_limited"


def test_minimax_status_ignores_success_payload():
    assert _minimax_status({"base_resp": {"status_code": 0}, "choices": []}) is None
    assert _minimax_status({"choices": []}) is None
    assert _minimax_status(None) is None


def test_quota_exhausted_is_terminal_for_recovery():
    failure = classify_failure(SimpleNamespace(code="quota_exhausted"))
    assert failure.retryable is False
    assert failure.category == "quota"
