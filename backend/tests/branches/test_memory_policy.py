import pytest


@pytest.mark.parametrize(
    "content",
    (
        "你在干嘛",
        "现在在哪里呢",
        "吃了吗",
        "你现在还在炒股吗",
        "是否可以电话呢",
        "方便接电话吗",
    ),
)
def test_current_state_bypasses_vector_memory(content: str) -> None:
    from moonlightbox.branches.memory_policy import retrieval_policy

    policy = retrieval_policy(content)

    assert policy.intent == "current_state"
    assert policy.retrieve_project_history is False
    assert policy.retrieve_branch_continuity is False


@pytest.mark.parametrize(
    ("content", "intent"),
    (
        ("你还记得那次去海边吗", "past_event"),
        ("你之前答应我的事呢", "commitment"),
        ("你以前说过不喜欢旅行吗", "prior_utterance"),
        ("你到底在不在乎我", "relationship"),
        ("你喜欢吃什么", "preference"),
    ),
)
def test_historical_intents_keep_memory_retrieval(content: str, intent: str) -> None:
    from moonlightbox.branches.memory_policy import retrieval_policy

    policy = retrieval_policy(content)

    assert policy.intent == intent
    assert policy.retrieve_project_history is True
    assert policy.retrieve_branch_continuity is True


@pytest.mark.parametrize("content", ("你现在还在乎我吗", "你目前还爱我吗"))
def test_relationship_feeling_is_not_misrouted_as_world_state(content: str) -> None:
    from moonlightbox.branches.memory_policy import retrieval_policy

    policy = retrieval_policy(content)

    assert policy.intent == "relationship"


def test_shared_past_question_prefers_event_recall_over_relationship_mode() -> None:
    from moonlightbox.branches.memory_policy import retrieval_policy

    assert retrieval_policy("我们以前一起去过首尔吗").intent == "past_event"


def test_quoted_preference_does_not_change_direct_question_intent() -> None:
    from moonlightbox.branches.memory_policy import retrieval_policy

    assert retrieval_policy("是谁呢\n> Feather：好讨厌这种半生不熟的人").intent == "general"
    assert retrieval_policy(
        "[引用消息：好讨厌这种半生不熟的人]\n是谁呢"
    ).intent == "general"


def test_quoted_current_state_does_not_disable_history_for_direct_question() -> None:
    from moonlightbox.branches.memory_policy import retrieval_policy

    policy = retrieval_policy("那个人是谁\n> 你现在在干嘛")

    assert policy.intent == "general"
    assert policy.retrieve_project_history is True
