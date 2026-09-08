"""Runtime v1 到 Linux 人格服务的窄 HTTP 客户端。"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


class RuntimeInferenceUnavailableError(RuntimeError):
    """独立人格服务不可达或响应不符合 Runtime 传输协议。"""


class RuntimeInferenceClient:
    """仅暴露 Director、PersonaActor 与 tokenizer 三种 Runtime 调用。"""

    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def generate_runtime_director(
        self, model_version_id: str, system_prompt: str, messages: list[dict[str, str]]
    ) -> str:
        return self._content(
            "/v1/inference/runtime-director",
            "runtime_director",
            model_version_id,
            {"system_prompt": system_prompt, "messages": messages},
            priority=3,
        )

    def generate_runtime_actor(
        self, model_version_id: str, system_prompt: str, messages: list[dict[str, str]]
    ) -> str:
        return self._content(
            "/v1/inference/runtime-actor",
            "runtime_actor",
            model_version_id,
            {"system_prompt": system_prompt, "messages": messages},
            priority=0,
        )

    def count_runtime_director_tokens(self, model_version_id: str, text: str) -> int:
        output = self._post(
            "/v1/inference/runtime-token-count",
            "runtime_token_count",
            model_version_id,
            {"text": text},
            priority=3,
        )
        count = output.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeInferenceUnavailableError("人格服务返回了无效 token 数")
        return count

    def _content(
        self,
        path: str,
        request_type: str,
        model_version_id: str,
        payload: dict[str, object],
        *,
        priority: int,
    ) -> str:
        output = self._post(path, request_type, model_version_id, payload, priority=priority)
        content = output.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeInferenceUnavailableError("人格服务返回了空内容")
        return content

    def _post(
        self,
        path: str,
        request_type: str,
        model_version_id: str,
        payload: dict[str, object],
        *,
        priority: int,
    ) -> dict[str, object]:
        body = {
            "request_id": str(uuid4()),
            "request_type": request_type,
            "priority": priority,
            "deadline": (datetime.now(UTC) + timedelta(seconds=self._timeout)).isoformat(),
            "model_version_id": model_version_id,
            "payload": payload,
        }
        request = Request(
            self._base_url + path,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                decoded = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeInferenceUnavailableError("人格推理服务不可用") from error
        if not isinstance(decoded, dict) or not isinstance(decoded.get("output"), dict):
            raise RuntimeInferenceUnavailableError("人格服务响应协议无效")
        return decoded["output"]
