def test_quality_report_rejects_immediate_repetition() -> None:
    from moonlightbox.training.conversation_quality import evaluate_conversations

    report = evaluate_conversations(
        [
            {
                "previous": "你今晚睡了吗？",
                "user": "我也正是这样",
                "candidate": "你今晚睡了吗？",
            }
        ]
    )

    assert report.immediate_repeat_rate == 1.0
    assert report.passed is False
