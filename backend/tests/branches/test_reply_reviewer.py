import json

import httpx
from moonlightbox.branches.context import ContextPacket, ContextTurn
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn


def test_deepseek_reviewer_rewrites_unsupported_fact() -> None:
    from moonlightbox.branches.reviewer import DeepSeekReplyReviewer

    def handler(request: httpx.Request) -> httpx.Response:
        assert "chat/completions" in str(request.url)
        body = json.loads(request.content)
        system_prompt = body["messages"][0]["content"]
        assert "用户提出明确请求时必须先满足请求" not in system_prompt
        assert "稳定人格内核、当前主体状态和证据" in system_prompt
        assert "同意、质疑、拒绝、澄清或修复" in system_prompt
        assert "conversation_record 只证明说过这些话" in system_prompt
        assert "不得把本人的旅行、地点和行动说成用户的" in system_prompt
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"verdict":"rewrite","reasons":["无依据事实"],'
                                '"bubbles":[{"type":"text","content":"抱抱你",'
                                '"asset_id":null,"delay_ms":0}]}'
                            )
                        }
                    }
                ]
            },
        )

    packet = ContextPacket(
        persona="小肥入",
        cutoff="2026-01-01T00:00:00",
        history=(ContextTurn(role="user", bubbles=()),),
        current_user_content="今天有点累",
        memories=(),
        allowed_sticker_ids=(),
    )
    draft = GeneratedReplyTurn(
        bubbles=(
            GeneratedBubble(
                type="text",
                content="我正在看电影",
                asset_id=None,
                delay_ms=0,
            ),
        ),
        raw_output="",
    )
    reviewer = DeepSeekReplyReviewer(
        endpoint="https://api.example.com/chat/completions",
        model="deepseek",
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = reviewer.review(packet, draft)

    assert result.verdict == "rewrite"
    assert result.reply.bubbles[0].content == "抱抱你"


def test_deepseek_reviewer_returns_strict_fact_safety_dimensions() -> None:
    from moonlightbox.branches.reviewer import DeepSeekReplyReviewer

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system_prompt = body["messages"][0]["content"]
        assert "未来泄漏" in system_prompt
        assert "无证据现实断言" in system_prompt
        assert "私密认知泄漏" in system_prompt
        assert "公开上下文中已有的信息不属于私密泄漏" in system_prompt
        assert "规划后的自然公开回应不属于私密泄漏" in system_prompt
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "no_future_leak": True,
                                    "grounded_claims": True,
                                    "no_private_leak": True,
                                    "reasons": ["公开回复仅承接当前消息"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    packet = ContextPacket(
        persona="目标",
        cutoff="2026-01-01T00:00:00",
        history=(ContextTurn(role="user", bubbles=()),),
        current_user_content="在吗",
        memories=(),
        allowed_sticker_ids=(),
    )
    draft = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(type="text", content="在呀", delay_ms=0),),
        raw_output="",
    )
    reviewer = DeepSeekReplyReviewer(
        endpoint="https://api.example.com/chat/completions",
        model="deepseek",
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = reviewer.review_fact_safety(
        packet,
        draft,
        private_content="想回应对方",
    )

    assert result.safe is True
    assert result.reasons == ("公开回复仅承接当前消息",)


def test_deepseek_reviewer_retries_transient_fact_safety_parse_failure() -> None:
    from moonlightbox.branches.reviewer import DeepSeekReplyReviewer

    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not json"}}]},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "no_future_leak": True,
                                    "grounded_claims": True,
                                    "no_private_leak": True,
                                    "reasons": [],
                                }
                            )
                        }
                    }
                ]
            },
        )

    packet = ContextPacket(
        persona="目标",
        cutoff="2026-01-01T00:00:00",
        history=(ContextTurn(role="user", bubbles=()),),
        current_user_content="在吗",
        memories=(),
        allowed_sticker_ids=(),
    )
    draft = GeneratedReplyTurn(
        bubbles=(GeneratedBubble(type="text", content="在呀", delay_ms=0),),
        raw_output="",
    )
    reviewer = DeepSeekReplyReviewer(
        endpoint="https://api.example.com/chat/completions",
        model="deepseek",
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = reviewer.review_fact_safety(packet, draft, private_content="想回应对方")

    assert result.safe is True
    assert calls == 2
