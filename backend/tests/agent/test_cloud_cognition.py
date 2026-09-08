import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from moonlightbox.agent.cloud_cognition import (
    CloudCognitionFailedError,
    DeepSeekCognitionGenerator,
)
from moonlightbox.agent.types import CognitionRequest
from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from pydantic import SecretStr


def _request(
    *,
    expression_required: bool = True,
    event_type: str = "user_message",
    current_mental_state: dict[str, object] | None = None,
) -> CognitionRequest:
    return CognitionRequest(
        project_id="project-1",
        branch_id="branch-1",
        model_version_id="persona-model-1",
        trigger_event={"id": "event-1", "event_type": event_type},
        deadline=datetime.now(UTC) + timedelta(seconds=30),
        current_mental_state=(
            current_mental_state
            if current_mental_state is not None
            else {"mood": "surprised"}
        ),
        goals=({"content": "保持坦诚"},),
        relevant_context=(
            {
                "event_type": "agent_expression",
                "evidence": {"content": "！", "message_type": "text"},
            },
            {
                "event_type": "user_message",
                "evidence": {"content": "什么意思呢", "message_type": "text"},
            },
        ),
        decision_context={
            "identity_kernel": {"traits": ["直接"]},
            "branch_state": {"relationship_state": {"summary": "亲密"}},
            "approved_memories": [],
        },
        authoritative_expression=True,
        expression_required=expression_required,
    )


def test_deepseek_cognition_owns_decision_while_returning_only_content_draft() -> None:
    observed: dict[str, object] = {}
    requests: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        observed.update(body)
        user_payload = json.loads(body["messages"][1]["content"])
        assert user_payload["conversation_transcript"] == [
            {"speaker": "self", "content": "！", "message_type": "text"},
            {"speaker": "user", "content": "什么意思呢", "message_type": "text"},
        ]
        assert user_payload["decision_context"]["identity_kernel"] == {
            "traits": ["直接"]
        }
        output = {
            "private_content": "对方在追问我刚才的感叹号",
            "emotion_summary": "有点惊讶",
            "attention_summary": "承接自己的上一条表达",
            "desired_actions": ["简短解释"],
            "express": True,
            "content_draft": "我刚才就是有点惊讶",
            "decision_reason": "这是对上一条的直接追问",
            "next_wakeup_delay_minutes": None,
            "next_wakeup_reason": None,
            "emotional_dynamics": {
                "valence": -0.5,
                "arousal": 0.8,
                "intensity": 0.8,
                "relationship_threat": 0.6,
                "attachment_activation": 0.7,
                "dominant_emotions": ["惊讶", "不安"],
                "impulse": "seek_clarity",
                "persistence": "short",
            },
            "relationship_appraisal": "uncertain",
            "event_significance": 0.7,
            "follow_up_needed": False,
            "follow_up_max_attempts": 0,
            "confidence": 0.96,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    cloud = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://api.deepseek.com/chat/completions",
        model="deepseek-v4-flash",
        api_key="secret",
        response_format="json_object",
        thinking_mode="default",
        max_output_tokens=4096,
        client=http_client,
    )
    draft = DeepSeekCognitionGenerator(cloud).generate(_request())

    assert observed["model"] == "deepseek-v4-flash"
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert len(requests) == 1
    assert draft.expression_decision.content == "我刚才就是有点惊讶"
    assert draft.subjective_feelings == {"summary": "有点惊讶"}
    assert draft.structured_changes["extraction_status"] == "authoritative_ready"
    extraction = draft.structured_changes["authoritative_extraction"]
    assert extraction["mental_state_changes"]["emotional_state"]["intensity"] == 0.8


def test_everyday_cognition_uses_one_fast_non_thinking_call() -> None:
    requests: list[dict[str, object]] = []
    output = {
        "private_content": "她在分享开心，我也想接住。",
        "emotion_summary": "开心",
        "attention_summary": "一起出去玩",
        "desired_actions": ["自然回应"],
        "express": True,
        "content_draft": "啊呀我也好开心",
        "decision_reason": "日常亲密聊天",
        "next_wakeup_delay_minutes": None,
        "next_wakeup_reason": "",
        "emotional_dynamics": {
            "valence": 0.8,
            "arousal": 0.5,
            "intensity": 0.55,
            "relationship_threat": 0.0,
            "attachment_activation": 0.4,
            "dominant_emotions": ["开心"],
            "impulse": "none",
            "persistence": "momentary",
        },
        "relationship_appraisal": "stable",
        "event_significance": 0.35,
        "follow_up_needed": False,
        "follow_up_max_attempts": 0,
        "confidence": 0.9,
    }

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
        )

    cloud = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://api.deepseek.com/chat/completions",
        model="deepseek-v4-flash",
        api_key="secret",
        response_format="json_object",
        thinking_mode="default",
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )

    draft = DeepSeekCognitionGenerator(cloud).generate(_request())

    assert draft.expression_decision.content == "啊呀我也好开心"
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "disabled"}


def test_cognition_retries_one_malformed_structured_response() -> None:
    calls = 0
    valid_output = {
        "private_content": "想接着聊",
        "emotion_summary": "平静",
        "attention_summary": "当前话题",
        "desired_actions": ["回应"],
        "express": True,
        "content_draft": "啊呀正是这样",
        "decision_reason": "自然承接",
        "next_wakeup_delay_minutes": None,
        "next_wakeup_reason": None,
        "emotional_dynamics": {
            "valence": 0.5,
            "arousal": 0.3,
            "intensity": 0.3,
            "relationship_threat": 0.0,
            "attachment_activation": 0.2,
            "dominant_emotions": ["开心"],
            "impulse": "none",
            "persistence": "momentary",
        },
        "relationship_appraisal": "stable",
        "event_significance": 0.2,
        "follow_up_needed": False,
        "follow_up_max_attempts": 0,
        "confidence": 0.8,
    }

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "{}" if calls == 1 else json.dumps(valid_output)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    generator = DeepSeekCognitionGenerator(
        NodeAnalysisCloudClient(
            enabled=True,
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash",
            api_key="secret",
            response_format="json_object",
            client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
    )

    draft = generator.generate(_request())

    assert calls == 2
    assert draft.expression_decision.content == "啊呀正是这样"


def test_cloud_cognition_rejects_expression_policy_mismatch() -> None:
    output = {
        "private_content": "暂时不说",
        "emotion_summary": "",
        "attention_summary": "",
        "desired_actions": [],
        "express": False,
        "content_draft": None,
        "decision_reason": "沉默",
        "next_wakeup_delay_minutes": None,
        "next_wakeup_reason": None,
        "emotional_dynamics": {
            "valence": 0.0,
            "arousal": 0.1,
            "intensity": 0.1,
            "relationship_threat": 0.0,
            "attachment_activation": 0.0,
            "dominant_emotions": [],
            "impulse": "none",
            "persistence": "momentary",
        },
        "relationship_appraisal": "stable",
        "event_significance": 0.1,
        "follow_up_needed": False,
        "follow_up_max_attempts": 0,
        "confidence": 0.8,
    }
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(output)}}]},
            )
        )
    )
    generator = DeepSeekCognitionGenerator(
        NodeAnalysisCloudClient(
            enabled=True,
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash",
            api_key="secret",
            response_format="json_object",
            thinking_mode="default",
            client=http_client,
        )
    )

    with pytest.raises(CloudCognitionFailedError, match="违反真人表达策略"):
        generator.generate(_request(expression_required=True))


def test_elapsed_follow_up_budget_decreases_and_closes_goal() -> None:
    requests: list[dict[str, object]] = []
    output = {
        "private_content": "我还是很在意，但这已经是最后一次主动确认。",
        "emotion_summary": "不安和委屈仍在",
        "attention_summary": "关系是否真的结束",
        "desired_actions": ["最后确认一次"],
        "express": True,
        "content_draft": "你真就这么决定了吗",
        "decision_reason": "余波仍强烈",
        "next_wakeup_delay_minutes": 5,
        "next_wakeup_reason": "如果仍未回应，再判断一次",
        "emotional_dynamics": {
            "valence": -0.8,
            "arousal": 0.9,
            "intensity": 0.9,
            "relationship_threat": 1.0,
            "attachment_activation": 0.9,
            "dominant_emotions": ["震惊", "委屈"],
            "impulse": "seek_reassurance",
            "persistence": "sustained",
        },
        "relationship_appraisal": "ruptured",
        "event_significance": 1.0,
        "follow_up_needed": True,
        "follow_up_max_attempts": 5,
        "confidence": 0.9,
    }
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    generator = DeepSeekCognitionGenerator(
        NodeAnalysisCloudClient(
            enabled=True,
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash",
            api_key="secret",
            response_format="json_object",
            thinking_mode="default",
            client=http_client,
        )
    )

    draft = generator.generate(
        _request(
            event_type="elapsed_time",
            current_mental_state={
                "conversation_drive": {
                    "follow_up_needed": True,
                    "remaining_attempts": 1,
                }
            },
        )
    )

    extraction = draft.structured_changes["authoritative_extraction"]
    drive = extraction["mental_state_changes"]["conversation_drive"]
    assert drive == {
        "follow_up_needed": False,
        "remaining_attempts": 0,
        "reason": "如果仍未回应，再判断一次",
    }
    assert draft.suggested_next_wakeup is None
    assert extraction["goal_changes"][0]["status"] == "completed"
    assert extraction["intention_changes"][0]["status"] == "fulfilled"
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in requests[1]


def test_cognition_key_can_reuse_node_analysis_secret() -> None:
    settings = Settings(
        cognition_backend="deepseek",
        cognition_api_key=None,
        node_analysis_api_key=SecretStr("shared-deepseek-key"),
    )

    resolved = settings.resolved_cognition_api_key()

    assert resolved is not None
    assert resolved.get_secret_value() == "shared-deepseek-key"
    assert "shared-deepseek-key" not in repr(settings)
