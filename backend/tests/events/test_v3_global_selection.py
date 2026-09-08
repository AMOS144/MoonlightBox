from typing import Any


class FakeSelectionClient:
    def __init__(self) -> None:
        self.system_content = ""
        self.user_content = ""

    def create_structured_completion(
        self,
        *,
        system_content: str,
        user_content: str,
        response_model: type[Any],
        **_kwargs: object,
    ) -> Any:
        self.system_content = system_content
        self.user_content = user_content
        return response_model.model_validate(
            {
                "selected": [
                    {
                        "candidate_key": "travel-key",
                        "relative_importance": 0.91,
                        "reason": "这段经历改变了后续共同计划",
                    }
                ]
            }
        )


def test_global_selector_compares_candidates_without_type_hard_gates() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    client = FakeSelectionClient()
    result = DualChannelEventReviewer(client).select_globally(
        candidates=[
            {
                "candidate_key": "game-key",
                "title": "一起打王者",
                "summary": "临时打了一局游戏",
                "evidence": ["要不要打王者", "上号吧"],
            },
            {
                "candidate_key": "travel-key",
                "title": "共同旅行",
                "summary": "共同完成旅行并形成后续计划",
                "evidence": ["到酒店了", "下次还想一起去"],
            },
        ],
        rejected_examples=[{"title": "普通吃饭", "reason": "用户标记为误报"}],
        maximum_nodes=10,
        run_id="run-1",
    )

    assert [item.candidate_key for item in result.selected] == ["travel-key"]
    assert "删除后是否会改变关系轨迹" in client.system_content
    assert "仅仅证明“发生过”不等于“重要”" in client.system_content
    assert "临时玩一局游戏" in client.system_content
    assert "票务买卖" in client.system_content
    assert "前后状态相同" in client.system_content
    assert "普通吃饭" in client.user_content
    assert "王者" not in client.system_content
