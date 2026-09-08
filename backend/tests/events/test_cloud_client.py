import json
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import format_datetime
from typing import Any, Literal, cast

import httpx
import pytest
from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import (
    NodeAnalysisCloudClient,
    NodeAnalysisCloudError,
    NodeAnalysisCloudErrorCode,
)
from moonlightbox.events.reviewer import EventCandidateBatch, EventCandidateReview
from pydantic import BaseModel, Field, RootModel, SecretStr, ValidationError


class ExampleResult(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class NestedResult(BaseModel):
    note: str | None = None


class ComplexResult(BaseModel):
    child: NestedResult | None = None
    items: list[NestedResult] = Field(default_factory=list)
    status: str = "new"


class OpenMapResult(BaseModel):
    metadata: dict[str, str]


class DefaultFieldResult(BaseModel):
    default: str = "保留字段"


class ConstrainedResult(BaseModel):
    score: float = Field(gt=0, le=1)
    text: str = Field(min_length=2, max_length=10, pattern=r"^[a-z]+$")
    items: list[int] = Field(min_length=1, max_length=3)
    created_at: datetime


class StringListResult(RootModel[list[str]]):
    pass


class ExampleChild(BaseModel):
    status: Literal["ready", "done"]
    tags: list[str]


class RecursiveExampleResult(BaseModel):
    child: ExampleChild
    value: int | None


def make_http_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_cloud_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **overrides: Any,
) -> NodeAnalysisCloudClient:
    options: dict[str, Any] = {
        "enabled": True,
        "endpoint": "https://example.com/v1/chat/completions",
        "model": "analysis-model",
        "api_key": "secret-key",
        "max_retries": 0,
        "client": make_http_client(handler),
    }
    options.update(overrides)
    return NodeAnalysisCloudClient(**options)


def complete(client: NodeAnalysisCloudClient) -> ExampleResult:
    return client.create_structured_completion(
        system_content="系统内容",
        user_content="用户内容",
        response_model=ExampleResult,
        operation_id="operation-1",
        window_id="window-1",
        run_id="run-1",
    )


def assert_error_code(
    client: NodeAnalysisCloudClient,
    expected: NodeAnalysisCloudErrorCode,
) -> NodeAnalysisCloudError:
    with pytest.raises(NodeAnalysisCloudError) as caught:
        complete(client)
    assert caught.value.code is expected
    return caught.value


def test_public_error_codes_are_exactly_the_eight_stable_codes() -> None:
    assert list(NodeAnalysisCloudErrorCode) == [
        NodeAnalysisCloudErrorCode.DISABLED,
        NodeAnalysisCloudErrorCode.MISSING_API_KEY,
        NodeAnalysisCloudErrorCode.AUTHENTICATION,
        NodeAnalysisCloudErrorCode.RATE_LIMIT,
        NodeAnalysisCloudErrorCode.SERVER,
        NodeAnalysisCloudErrorCode.TIMEOUT,
        NodeAnalysisCloudErrorCode.NETWORK,
        NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
    ]


def test_successfully_returns_schema_validated_model() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"label": "冲突", "score": 0.8})}}]
            },
        )
    )

    result = complete(client)

    assert result == ExampleResult(label="冲突", score=0.8)


def test_request_uses_strict_json_schema_and_deterministic_messages() -> None:
    requests: list[httpx.Request] = []
    custom_schema = ExampleResult.model_json_schema()
    strict_schema = json.loads(json.dumps(custom_schema))
    strict_schema["additionalProperties"] = False
    strict_schema["properties"]["score"].pop("minimum")
    strict_schema["properties"]["score"].pop("maximum")

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"和解","score":1}'}}]},
        )

    client = make_cloud_client(handle)
    result = client.create_structured_completion(
        system_content="只判断关系变化",
        user_content="窗口消息",
        response_model=ExampleResult,
        json_schema=custom_schema,
    )

    assert result.label == "和解"
    request = requests[0]
    assert request.headers["authorization"] == "Bearer secret-key"
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.content)
    assert body == {
        "model": "analysis-model",
        "messages": [
            {"role": "system", "content": "只判断关系变化"},
            {"role": "user", "content": "窗口消息"},
        ],
        "temperature": 0,
        "max_tokens": 8192,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ExampleResult",
                "strict": True,
                "schema": strict_schema,
            },
        },
    }


def test_deepseek_json_object_request_is_exact_and_locally_validated() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"和解","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        model="deepseek-v4-flash",
        response_format="json_object",
        thinking_mode="disabled",
        max_output_tokens=4096,
    )

    assert complete(client) == ExampleResult(label="和解", score=1)
    body = json.loads(requests[0].content)
    assert body["model"] == "deepseek-v4-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 4096
    assert "json_schema" not in json.dumps(body["response_format"])
    final_prompt = "".join(message["content"] for message in body["messages"])
    assert "json" in final_prompt.lower()
    assert "schema" in final_prompt.lower()
    assert final_prompt.lower().count("json schema") == 1
    assert final_prompt.count('"properties"') == 1
    example = json.loads(body["messages"][0]["content"].split("最小合法 JSON 示例：", 1)[1])
    assert ExampleResult.model_validate(example, strict=True)
    assert example == {"label": "", "score": 0}


def test_json_object_example_recursively_covers_schema_types() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": ('{"child":{"status":"ready","tags":[]},"value":0}')}}
                ]
            },
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
    )

    client.create_structured_completion(
        system_content="系统",
        user_content="用户",
        response_model=RecursiveExampleResult,
    )

    system_prompt = json.loads(requests[0].content)["messages"][0]["content"]
    example = json.loads(system_prompt.split("最小合法 JSON 示例：", 1)[1])
    assert example == {
        "child": {"status": "ready", "tags": [""]},
        "value": 0,
    }
    assert RecursiveExampleResult.model_validate(example, strict=True)


@pytest.mark.parametrize(
    ("response_model", "array_path"),
    [
        (EventCandidateBatch, ("candidates",)),
        (EventCandidateReview, ("evidence_ids",)),
    ],
)
def test_json_object_example_arrays_are_non_empty_and_model_valid(
    response_model: type[BaseModel],
    array_path: tuple[str, ...],
) -> None:
    examples: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        system_prompt = json.loads(request.content)["messages"][0]["content"]
        example = json.loads(system_prompt.split("最小合法 JSON 示例：", 1)[1])
        examples.append(example)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(example)}}]},
        )

    result = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
    ).create_structured_completion(
        system_content="系统",
        user_content="用户",
        response_model=response_model,
    )

    array_value: Any = examples[0]
    for key in array_path:
        array_value = array_value[key]
    assert array_value
    assert response_model.model_validate(result.model_dump(), strict=True)

    if response_model is EventCandidateBatch:
        candidate = examples[0]["candidates"][0]
        candidate_properties = EventCandidateBatch.model_json_schema()["$defs"][
            "RawEventCandidate"
        ]["properties"]
        assert set(candidate) == set(candidate_properties)
        assert candidate["emotion_labels"] == [""]
        assert candidate["evidence_ids"] == [""]


def test_json_object_example_honors_zero_max_items_and_caps_large_min_items() -> None:
    requests: list[httpx.Request] = []
    schema = {
        "type": "object",
        "properties": {
            "empty": {"type": "array", "items": {"type": "string"}, "maxItems": 0},
            "bounded": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 10000,
            },
        },
        "required": ["empty", "bounded"],
        "additionalProperties": False,
    }

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"empty":[],"bounded":[]}'}}]},
        )

    with pytest.raises(NodeAnalysisCloudError):
        make_cloud_client(
            handle,
            response_format="json_object",
        ).create_structured_completion(
            system_content="系统",
            user_content="用户",
            response_model=ExampleResult,
            json_schema=schema,
        )

    prompt = json.loads(requests[0].content)["messages"][0]["content"]
    example = json.loads(prompt.split("最小合法 JSON 示例：", 1)[1])
    assert example["empty"] == []
    assert 1 < len(example["bounded"]) <= 16


def test_json_object_final_prompt_respects_exact_character_budget() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"预算","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
    )
    system_content = "系统"
    user_content = "用户"
    budget = (
        len(system_content)
        + len(user_content)
        + client.estimate_structured_prompt_overhead(response_model=ExampleResult)
    )

    result = client.create_structured_completion(
        system_content=system_content,
        user_content=user_content,
        response_model=ExampleResult,
        max_prompt_chars=budget,
    )

    body = json.loads(requests[0].content)
    final_size = sum(len(message["content"]) for message in body["messages"])
    assert result.label == "预算"
    assert final_size == budget


def test_json_object_prompt_over_budget_fails_without_transport_call() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
    )

    with pytest.raises(NodeAnalysisCloudError) as caught:
        client.create_structured_completion(
            system_content="系统",
            user_content="用户",
            response_model=ExampleResult,
            max_prompt_chars=1,
        )

    assert caught.value.code is NodeAnalysisCloudErrorCode.INVALID_RESPONSE
    assert caught.value.attempts == 0
    assert caught.value.diagnostic == {"configuration": "prompt_budget_exceeded"}
    assert attempts == 0


def test_deepseek_default_thinking_mode_omits_thinking_field() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"正常","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
        thinking_mode="default",
    )

    assert complete(client).label == "正常"
    assert "thinking" not in json.loads(requests[0].content)


def test_minimax_request_splits_reasoning_from_structured_content() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"正常","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.minimaxi.com/v1/chat/completions",
        model="MiniMax-M2.7-highspeed",
        response_format="json_object",
        thinking_mode="disabled",
    )

    assert complete(client).label == "正常"
    body = json.loads(requests[0].content)
    assert body["reasoning_split"] is True
    assert "thinking" not in body


@pytest.mark.parametrize(
    "content",
    [
        "",
        '{"label":"截断"',
        '{"label":"字段错误","score":2}',
        '{"label":"类型错误","score":"0.5"}',
    ],
)
def test_deepseek_invalid_json_object_response_is_rejected(content: str) -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        ),
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
        thinking_mode="disabled",
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)

    assert error.attempts == 1
    assert error.retryable is False


def test_deepseek_blank_content_retries_then_succeeds() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = " \n\t " if attempts == 1 else '{"label":"恢复","score":1}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
        max_retries=2,
        backoff_seconds=0.25,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    assert complete(client).label == "恢复"
    assert attempts == 2
    assert sleeps == [0.25]


def test_deepseek_blank_content_exhausts_retry_budget() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "   "}}]},
        )

    client = make_cloud_client(
        handle,
        endpoint="https://api.deepseek.com/chat/completions",
        response_format="json_object",
        max_retries=2,
        backoff_seconds=0.5,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)

    assert error.attempts == 3
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_json_schema_blank_content_keeps_non_retrying_contract() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "   "}}]},
        )

    client = make_cloud_client(handle, max_retries=3)

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)

    assert error.attempts == 1
    assert attempts == 1


def test_injected_http_client_receives_explicit_request_timeout() -> None:
    timeout_extensions: list[dict[str, float | None]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        timeout_extensions.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"正常","score":1}'}}]},
        )

    client = make_cloud_client(handle, timeout_seconds=17.5)

    assert complete(client).label == "正常"
    assert timeout_extensions == [{"connect": 17.5, "read": 17.5, "write": 17.5, "pool": 17.5}]


def test_request_normalizes_nested_optional_schema_for_openai_strict_mode() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"child":null,"items":[],"status":"new"}'}}]
            },
        )

    result = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="analysis-model",
        api_key="secret-key",
        max_retries=0,
        client=make_http_client(handle),
    ).create_structured_completion(
        system_content="系统内容",
        user_content="用户内容",
        response_model=ComplexResult,
    )

    schema = json.loads(requests[0].content)["response_format"]["json_schema"]["schema"]
    nested_schema = schema["$defs"]["NestedResult"]
    assert result.child is None
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["child", "items", "status"]
    assert nested_schema["additionalProperties"] is False
    assert nested_schema["required"] == ["note"]
    assert schema["properties"]["child"]["anyOf"][1] == {"type": "null"}
    assert schema["properties"]["items"]["items"] == {"$ref": "#/$defs/NestedResult"}
    assert "default" not in json.dumps(schema)


def test_strict_schema_keeps_property_named_default() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"default":"字段值"}'}}]},
        )

    result = make_cloud_client(handle).create_structured_completion(
        system_content="系统内容",
        user_content="用户内容",
        response_model=DefaultFieldResult,
    )

    schema = json.loads(requests[0].content)["response_format"]["json_schema"]["schema"]
    assert result.default == "字段值"
    assert "default" in schema["properties"]
    assert "default" not in schema["properties"]["default"]
    assert schema["required"] == ["default"]


def test_open_map_schema_is_rejected_without_sending_request() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(handle)

    with pytest.raises(NodeAnalysisCloudError) as caught:
        client.create_structured_completion(
            system_content="系统内容",
            user_content="用户内容",
            response_model=OpenMapResult,
        )

    assert caught.value.code is NodeAnalysisCloudErrorCode.INVALID_RESPONSE
    assert caught.value.attempts == 0
    assert caught.value.retryable is False
    assert caught.value.diagnostic == {"configuration": "unsupported_open_map"}
    assert attempts == 0


def test_request_schema_removes_provider_unsupported_validation_keywords() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"score":0.5,"text":"valid","items":[1],'
                                '"created_at":"2026-07-18T12:00:00Z"}'
                            )
                        }
                    }
                ]
            },
        )

    result = make_cloud_client(handle).create_structured_completion(
        system_content="系统内容",
        user_content="用户内容",
        response_model=ConstrainedResult,
    )

    schema = json.loads(requests[0].content)["response_format"]["json_schema"]["schema"]
    serialized = json.dumps(schema)
    unsupported = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    assert result.score == 0.5
    assert all(f'"{keyword}":' not in serialized for keyword in unsupported)


def test_pattern_properties_schema_is_rejected_without_sending_request() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(handle)
    pattern_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["label", "score"],
        "patternProperties": {"^x-": {"type": "string"}},
    }

    with pytest.raises(NodeAnalysisCloudError) as caught:
        client.create_structured_completion(
            system_content="系统内容",
            user_content="用户内容",
            response_model=ExampleResult,
            json_schema=pattern_schema,
        )

    assert caught.value.code is NodeAnalysisCloudErrorCode.INVALID_RESPONSE
    assert caught.value.attempts == 0
    assert caught.value.diagnostic == {"configuration": "unsupported_schema_structure"}
    assert attempts == 0


@pytest.mark.parametrize(
    "schema",
    [
        True,
        False,
        {
            "type": "object",
            "properties": {"label": False, "score": {"type": "number"}},
        },
        {
            "type": "object",
            "properties": {
                "label": {"anyOf": [{"type": "string"}, True]},
                "score": {"type": "number"},
            },
        },
        {
            "type": "object",
            "properties": {
                "label": {"type": "array", "items": False},
                "score": {"type": "number"},
            },
        },
        {
            "type": "object",
            "$defs": {"Nested": False},
            "properties": {
                "label": {"$ref": "#/$defs/Nested"},
                "score": {"type": "number"},
            },
        },
    ],
)
def test_boolean_subschemas_are_rejected_without_sending_request(
    schema: object,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(handle)

    with pytest.raises(NodeAnalysisCloudError) as caught:
        client.create_structured_completion(
            system_content="系统内容",
            user_content="用户内容",
            response_model=ExampleResult,
            json_schema=cast(Any, schema),
        )

    assert caught.value.code is NodeAnalysisCloudErrorCode.INVALID_RESPONSE
    assert caught.value.attempts == 0
    assert caught.value.diagnostic == {"configuration": "unsupported_boolean_schema"}
    assert attempts == 0


def test_root_array_schema_is_rejected_without_sending_request() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(handle)

    with pytest.raises(NodeAnalysisCloudError) as caught:
        client.create_structured_completion(
            system_content="系统内容",
            user_content="用户内容",
            response_model=StringListResult,
        )

    assert caught.value.code is NodeAnalysisCloudErrorCode.INVALID_RESPONSE
    assert caught.value.attempts == 0
    assert caught.value.diagnostic == {"configuration": "root_schema_must_be_object"}
    assert attempts == 0


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_missing_api_key_fails_without_sending_request(api_key: str | None) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(handle, api_key=api_key)

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.MISSING_API_KEY)
    assert attempts == 0
    assert error.attempts == 0
    assert error.retryable is False


@pytest.mark.parametrize(
    ("endpoint", "model"),
    [
        ("", "analysis-model"),
        ("https://example.com/v1/chat/completions", ""),
    ],
)
def test_incomplete_endpoint_or_model_fails_without_sending_request(
    endpoint: str,
    model: str,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    with pytest.raises(ValueError, match="endpoint 和 model"):
        make_cloud_client(handle, endpoint=endpoint, model=model)
    assert attempts == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_seconds", 7201),
        ("max_retries", 11),
        ("backoff_seconds", 61),
        ("max_backoff_seconds", 301),
    ],
)
def test_client_rejects_retry_storm_configuration(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        make_cloud_client(lambda request: httpx.Response(200), **{field: value})


def test_client_accepts_two_hour_timeout_for_world_compilation() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(200),
        timeout_seconds=7200,
    )

    assert client.is_closed is False
    client.close()


def test_disabled_client_fails_without_sending_request() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    client = make_cloud_client(handle, enabled=False)

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.DISABLED)
    assert attempts == 0
    assert error.attempts == 0
    assert error.retryable is False


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, NodeAnalysisCloudErrorCode.AUTHENTICATION),
        (403, NodeAnalysisCloudErrorCode.AUTHENTICATION),
    ],
)
def test_authentication_errors_fail_immediately(
    status_code: int,
    expected: NodeAnalysisCloudErrorCode,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code, text="sensitive response")

    client = make_cloud_client(handle, max_retries=3)

    error = assert_error_code(client, expected)
    assert attempts == 1
    assert error.attempts == 1
    assert error.retryable is False
    assert error.diagnostic == {"http_status": status_code}


def test_rate_limit_retries_then_succeeds_with_exponential_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"升级","score":0.9}'}}]},
        )

    client = make_cloud_client(
        handle,
        max_retries=3,
        backoff_seconds=0.25,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    assert complete(client).label == "升级"
    assert attempts == 3
    assert sleeps == [0.25, 0.5]


def test_rate_limit_exhausts_retry_budget_with_retry_metadata() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429)

    client = make_cloud_client(
        handle,
        max_retries=2,
        backoff_seconds=0.5,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.RATE_LIMIT)

    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert error.attempts == 3
    assert error.retryable is True


def test_server_error_exhausts_retry_budget() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    client = make_cloud_client(
        handle,
        max_retries=2,
        backoff_seconds=1,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.SERVER)
    assert attempts == 3
    assert sleeps == [1, 2]
    assert error.attempts == 3
    assert error.retryable is True


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (httpx.ReadTimeout("timed out"), NodeAnalysisCloudErrorCode.TIMEOUT),
        (httpx.ConnectError("network down"), NodeAnalysisCloudErrorCode.NETWORK),
    ],
)
def test_transport_errors_are_classified_and_retried(
    exception: httpx.RequestError,
    expected: NodeAnalysisCloudErrorCode,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise exception

    client = make_cloud_client(
        handle,
        max_retries=1,
        backoff_seconds=0.1,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    error = assert_error_code(client, expected)
    assert attempts == 2
    assert sleeps == [0.1]
    assert error.attempts == 2
    assert error.retryable is True
    assert error.diagnostic == {"exception_type": type(exception).__name__}


def test_retry_after_seconds_takes_precedence_over_exponential_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"恢复","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        max_retries=1,
        backoff_seconds=1,
        sleep=sleeps.append,
        jitter=lambda: 0.5,
    )

    assert complete(client).label == "恢复"
    assert sleeps == [3]


def test_retry_after_http_date_uses_injected_clock() -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    retry_at = datetime(2026, 7, 18, 12, 0, 5, tzinfo=UTC)
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                headers={"Retry-After": format_datetime(retry_at, usegmt=True)},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"恢复","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        max_retries=1,
        backoff_seconds=1,
        sleep=sleeps.append,
        clock=lambda: now.timestamp(),
        jitter=lambda: 0,
    )

    assert complete(client).label == "恢复"
    assert sleeps == [5]


def test_malformed_retry_after_falls_back_to_capped_jittered_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, headers={"Retry-After": "not-a-delay"})

    client = make_cloud_client(
        handle,
        max_retries=2,
        backoff_seconds=10,
        max_backoff_seconds=12,
        sleep=sleeps.append,
        jitter=lambda: 1,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.SERVER)

    assert error.attempts == 3
    assert attempts == 3
    assert sleeps == [12, 12]


def test_retry_after_is_not_truncated_by_local_backoff_cap() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "120"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"恢复","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        max_retries=1,
        backoff_seconds=10,
        max_backoff_seconds=12,
        sleep=sleeps.append,
        jitter=lambda: 1,
    )

    assert complete(client).label == "恢复"
    assert sleeps == [120]


@pytest.mark.parametrize("header", ["1e309", "NaN", "-1"])
def test_non_finite_or_negative_retry_after_falls_back_to_local_backoff(
    header: str,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": header})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"恢复","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        max_retries=1,
        backoff_seconds=2,
        max_backoff_seconds=10,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    assert complete(client).label == "恢复"
    assert sleeps == [2]


def test_large_finite_retry_after_uses_independent_safety_cap() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": "1000000"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"恢复","score":1}'}}]},
        )

    client = make_cloud_client(
        handle,
        max_retries=1,
        backoff_seconds=1,
        max_backoff_seconds=5,
        max_retry_after_seconds=30,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    assert complete(client).label == "恢复"
    assert sleeps == [30]


def test_invalid_message_json_is_classified_without_retry() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json"}}]},
        ),
        max_retries=3,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)
    assert error.attempts == 1
    assert error.retryable is False
    assert error.diagnostic == {"http_status": 200}


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"label":"正常","score":1}\n```',
        '<think>先组织结果</think>\n{"label":"正常","score":1}',
        '结果如下：\n```json\n{"label":"正常","score":1}\n```',
    ],
)
def test_json_object_transport_wrappers_are_removed(content: str) -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        ),
        response_format="json_object",
    )

    assert complete(client) == ExampleResult(label="正常", score=1)


def test_invalid_response_body_json_is_classified_without_retry() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(200, content=b"not-json"),
        max_retries=3,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)
    assert error.attempts == 1
    assert error.retryable is False


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": {"label": "冲突", "score": 1}}}]},
    ],
)
def test_invalid_response_structure_is_classified(body: object) -> None:
    client = make_cloud_client(lambda request: httpx.Response(200, json=body))

    assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)


def test_schema_validation_failure_is_classified_without_retry() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"冲突","score":2}'}}]},
        ),
        max_retries=3,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)
    assert error.attempts == 1
    assert error.retryable is False


def test_message_parsed_object_takes_precedence_over_content() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "parsed": {"label": "已解析", "score": 0.7},
                            "content": "not-json",
                        }
                    }
                ]
            },
        )
    )

    assert complete(client) == ExampleResult(label="已解析", score=0.7)


def test_non_empty_refusal_is_rejected_even_when_parsed_is_present() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "refusal": "sensitive refusal",
                            "parsed": {"label": "不应采用", "score": 1},
                        }
                    }
                ]
            },
        )
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)

    assert error.attempts == 1
    assert error.retryable is False


def test_text_block_array_content_is_joined_and_validated() -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": '{"label":"数组",'},
                                {"type": "text", "text": '"score":1}'},
                            ]
                        }
                    }
                ]
            },
        )
    )

    assert complete(client) == ExampleResult(label="数组", score=1)


@pytest.mark.parametrize(
    "message",
    [
        {},
        {"parsed": None, "content": []},
        {"content": [{"type": "image", "image_url": "sensitive"}]},
        {"content": [{"type": "text", "text": 123}]},
    ],
)
def test_message_without_parseable_content_is_invalid(message: object) -> None:
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": message}]},
        )
    )

    assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)


def test_unexpected_client_error_is_an_invalid_response_without_retry() -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(422, text="sensitive validation details")

    client = make_cloud_client(handle, max_retries=3)

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)

    assert attempts == 1
    assert error.attempts == 1
    assert error.retryable is False


def test_error_strings_do_not_leak_secrets_or_payloads() -> None:
    secrets = [
        "ultra-secret-key",
        "Bearer ultra-secret-key",
        "完整系统提示",
        "完整用户消息",
        "完整敏感响应",
    ]
    client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="analysis-model",
        api_key="ultra-secret-key",
        max_retries=0,
        client=make_http_client(lambda request: httpx.Response(500, text="完整敏感响应")),
    )

    with pytest.raises(NodeAnalysisCloudError) as caught:
        client.create_structured_completion(
            system_content="完整系统提示",
            user_content="完整用户消息",
            response_model=ExampleResult,
            operation_id="safe-operation",
        )

    rendered = f"{caught.value!s} {caught.value!r}"
    assert all(secret not in rendered for secret in secrets)
    assert caught.value.context == {"operation_id": "safe-operation"}


def test_transport_diagnostic_contains_only_exception_type() -> None:
    secret = "never-leak-transport-message"
    secret_url = "https://secret.example/private-path"

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            secret,
            request=httpx.Request("GET", secret_url),
        )

    client = make_cloud_client(handle)
    error = assert_error_code(client, NodeAnalysisCloudErrorCode.NETWORK)

    rendered = f"{error!s} {error!r} {error.diagnostic!r}"
    assert error.diagnostic == {"exception_type": "ConnectError"}
    assert secret not in rendered
    assert secret_url not in rendered


def test_invalid_url_is_mapped_to_invalid_response_without_retry() -> None:
    sleeps: list[float] = []
    client = make_cloud_client(
        lambda request: httpx.Response(200),
        endpoint="https://example.com:not-a-port/chat",
        max_retries=3,
        sleep=sleeps.append,
    )

    error = assert_error_code(client, NodeAnalysisCloudErrorCode.INVALID_RESPONSE)

    assert error.attempts == 1
    assert error.retryable is False
    assert error.diagnostic == {"exception_type": "InvalidURL"}
    assert sleeps == []


def test_schema_error_traceback_does_not_include_complete_response() -> None:
    sensitive_response = "完整敏感响应"
    client = make_cloud_client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"label": sensitive_response, "score": 2})}}
                ]
            },
        )
    )

    with pytest.raises(NodeAnalysisCloudError) as caught:
        complete(client)

    rendered = "".join(
        traceback.format_exception(
            caught.type,
            caught.value,
            caught.tb,
        )
    )
    assert sensitive_response not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_settings_define_independent_node_analysis_configuration() -> None:
    settings = Settings(
        node_analysis_enabled=True,
        node_analysis_endpoint="https://node.example/v1/chat/completions",
        node_analysis_model="node-model",
        node_analysis_api_key="node-secret",
        node_analysis_timeout_seconds=12.5,
        node_analysis_max_retries=4,
        node_analysis_backoff_seconds=0.75,
        node_analysis_max_backoff_seconds=12,
        node_analysis_max_retry_after_seconds=600,
        node_analysis_response_format="json_object",
        node_analysis_thinking_mode="default",
        node_analysis_max_output_tokens=4096,
        cloud_evaluation_endpoint="https://evaluation.example/v1/chat/completions",
        cloud_evaluation_api_key="evaluation-secret",
    )

    assert settings.node_analysis_enabled is True
    assert settings.node_analysis_endpoint == "https://node.example/v1/chat/completions"
    assert settings.node_analysis_model == "node-model"
    assert settings.node_analysis_api_key is not None
    assert settings.node_analysis_api_key.get_secret_value() == "node-secret"
    assert settings.node_analysis_timeout_seconds == 12.5
    assert settings.node_analysis_max_retries == 4
    assert settings.node_analysis_backoff_seconds == 0.75
    assert settings.node_analysis_max_backoff_seconds == 12
    assert settings.node_analysis_max_retry_after_seconds == 600
    assert settings.node_analysis_response_format == "json_object"
    assert settings.node_analysis_thinking_mode == "default"
    assert settings.node_analysis_max_output_tokens == 4096
    assert settings.node_analysis_api_key.get_secret_value() != settings.cloud_evaluation_api_key


def test_settings_use_safe_structured_output_defaults() -> None:
    settings = Settings.model_construct()

    assert settings.node_analysis_response_format == "json_schema"
    assert settings.node_analysis_thinking_mode == "disabled"
    assert settings.node_analysis_max_output_tokens == 8192


def test_settings_hide_node_analysis_api_key_from_repr_and_dump() -> None:
    secret = "never-print-this-key"
    settings = Settings(node_analysis_api_key=secret)

    assert isinstance(settings.node_analysis_api_key, SecretStr)
    assert secret not in repr(settings)
    assert secret not in repr(settings.model_dump())
    assert secret not in json.dumps(settings.model_dump(mode="json"), default=str)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_analysis_endpoint", ""),
        ("node_analysis_endpoint", "ftp://example.com/chat"),
        ("node_analysis_endpoint", "https://example.com:not-a-port/chat"),
        ("node_analysis_endpoint", "https:///missing-host"),
        ("node_analysis_model", "   "),
        ("node_analysis_timeout_seconds", 0),
        ("node_analysis_timeout_seconds", 301),
        ("node_analysis_max_retries", -1),
        ("node_analysis_max_retries", 11),
        ("node_analysis_backoff_seconds", 0),
        ("node_analysis_backoff_seconds", 61),
        ("node_analysis_max_backoff_seconds", 0),
        ("node_analysis_max_backoff_seconds", 301),
        ("node_analysis_max_retry_after_seconds", 0),
        ("node_analysis_response_format", "yaml"),
        ("node_analysis_thinking_mode", "enabled"),
        ("node_analysis_max_output_tokens", 0),
        ("node_analysis_max_output_tokens", 65537),
    ],
)
def test_invalid_node_analysis_settings_fail_during_loading(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_endpoint_credentials_are_rejected_without_error_repr_leak() -> None:
    username = "private-user"
    password = "private-password"

    with pytest.raises(ValidationError) as caught:
        Settings(
            node_analysis_endpoint=(
                f"https://{username}:{password}@example.com/v1/chat/completions"
            )
        )

    rendered = f"{caught.value!s} {caught.value!r}"
    assert username not in rendered
    assert password not in rendered


def test_client_accepts_secret_str_api_key_without_exposing_it() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"安全","score":1}'}}]},
        )

    client = make_cloud_client(handle, api_key=SecretStr("secret-value"))

    assert complete(client).label == "安全"
    assert requests[0].headers["authorization"] == "Bearer secret-value"
    assert "secret-value" not in repr(client)


def test_owned_http_client_is_closed_but_injected_client_is_not() -> None:
    injected = make_http_client(lambda request: httpx.Response(500))
    external_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="model",
        api_key="secret",
        client=injected,
    )
    owned_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="model",
        api_key="secret",
    )

    external_client.close()
    owned_client.close()

    assert injected.is_closed is False
    assert owned_client.is_closed is True
