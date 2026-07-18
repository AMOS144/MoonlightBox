import json
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field, ValidationError


@dataclass(frozen=True)
class CloudEvaluationCase:
    prompt: str
    reference_answer: str
    candidate_answer: str


class CloudEvaluationScore(BaseModel):
    style_score: float = Field(ge=0.0, le=1.0)
    semantic_score: float = Field(ge=0.0, le=1.0)
    groundedness_score: float = Field(ge=0.0, le=1.0)
    reason: str


class CloudEvaluationError(RuntimeError):
    pass


class CloudEvaluator:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=60)

    def evaluate(self, case: CloudEvaluationCase) -> CloudEvaluationScore:
        prompt = json.dumps(
            {
                "任务": "比较候选回复与真实回复，只返回 JSON 分数",
                "用户消息": case.prompt,
                "真实回复": case.reference_answer,
                "候选回复": case.candidate_answer,
                "输出字段": [
                    "style_score",
                    "semantic_score",
                    "groundedness_score",
                    "reason",
                ],
            },
            ensure_ascii=False,
        )
        response = self._client.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        try:
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return CloudEvaluationScore.model_validate_json(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValidationError) as error:
            raise CloudEvaluationError("云端评测响应无效") from error
