import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings


def test_openai_compatible_evaluator_returns_structured_scores() -> None:
    from moonlightbox.evaluation.cloud import CloudEvaluationCase, CloudEvaluator

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "style_score": 0.8,
                                    "semantic_score": 0.9,
                                    "groundedness_score": 0.95,
                                    "reason": "语气接近且事实一致",
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    evaluator = CloudEvaluator(
        endpoint="https://example.com/v1/chat/completions",
        api_key="secret",
        model="judge-model",
        client=client,
    )

    score = evaluator.evaluate(CloudEvaluationCase("今晚吃什么", "随便啦", "都可以呀"))

    assert score.style_score == 0.8
    assert score.groundedness_score == 0.95


def test_cloud_evaluation_can_be_disabled(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'evaluation.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
        cloud_evaluation_enabled=False,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/evaluation/cloud",
            json={
                "prompt": "问题",
                "reference_answer": "真实",
                "candidate_answer": "候选",
            },
        )

    assert response.status_code == 403
