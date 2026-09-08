from datetime import UTC, datetime, timedelta


def test_proactive_contact_requires_new_psychological_impetus() -> None:
    from moonlightbox.branches.proactive import ProactiveDecisionEngine

    engine = ProactiveDecisionEngine()
    now = datetime.now(UTC)
    state = {
        "approach_motivation": {
            "score": 0.65,
            "trigger": "user-message-12",
        },
        "inhibition": {"score": 0.05},
        "open_sequences": [{"topic": "还没聊完的旅行"}],
    }

    first = engine.decide(
        state=state,
        last_message_at=now - timedelta(hours=2),
        unanswered_assistant_count=1,
        now=now,
    )
    state["approach_motivation"] = {
        "score": 0.65,
        "trigger": "user-message-12",
        "consumed_trigger": "user-message-12",
    }
    repeated = engine.decide(
        state=state,
        last_message_at=now - timedelta(hours=3),
        unanswered_assistant_count=2,
        now=now,
    )

    assert first.should_speak is True
    assert repeated.should_speak is False


def test_proactive_decision_has_no_daily_frequency_counter() -> None:
    from inspect import signature

    from moonlightbox.branches.proactive import ProactiveDecisionEngine

    parameters = signature(ProactiveDecisionEngine.decide).parameters

    assert "daily_count" not in parameters
    assert "daily_limit" not in parameters


def test_proactive_contact_respects_authentic_behavioral_rhythm() -> None:
    from moonlightbox.branches.proactive import ProactiveDecisionEngine

    engine = ProactiveDecisionEngine()
    profile = {
        "sample_count": 100,
        "confidence": 1.0,
        "timezone_offset_minutes": 0,
        "active_hours": [9, 10],
        "proactive_turn_rate": 0.8,
    }
    state = {
        "approach_motivation": {"score": 0.55, "trigger": "new-event"},
        "inhibition": {"score": 0.05},
        "open_sequences": [{"topic": "未完成话题"}],
    }
    off_hours = datetime(2026, 8, 5, 2, tzinfo=UTC)

    deferred = engine.decide(
        state=state,
        last_message_at=off_hours - timedelta(hours=2),
        unanswered_assistant_count=1,
        behavioral_rhythm=profile,
        now=off_hours,
    )
    active = engine.decide(
        state=state,
        last_message_at=off_hours - timedelta(hours=9),
        unanswered_assistant_count=1,
        behavioral_rhythm=profile,
        now=off_hours.replace(hour=9),
    )

    assert deferred.should_speak is False
    assert "常主动交流的时段" in deferred.reason
    assert deferred.next_review_at.hour == 9
    assert active.should_speak is True


def test_endogenous_impetus_can_come_from_goal_or_habit_without_user_trigger() -> None:
    from moonlightbox.branches.proactive import derive_endogenous_impetus

    now = datetime(2026, 8, 5, 9, tzinfo=UTC)
    goal = derive_endogenous_impetus(
        state={"approach_motivation": {}, "open_sequences": []},
        behavioral_rhythm=None,
        active_goals=(("goal-1", 0.8),),
        now=now,
    )
    habitual = derive_endogenous_impetus(
        state={"approach_motivation": {}, "open_sequences": []},
        behavioral_rhythm={
            "sample_count": 100,
            "proactive_turn_rate": 0.6,
            "active_hours": [9],
            "timezone_offset_minutes": 0,
        },
        now=now,
    )

    assert goal is not None and goal.trigger.startswith("goal:goal-1")
    assert habitual is not None and habitual.trigger.startswith("habitual-sharing:")


def test_consumed_endogenous_trigger_is_not_recreated_in_same_slot() -> None:
    from moonlightbox.branches.proactive import derive_endogenous_impetus

    now = datetime(2026, 8, 5, 9, tzinfo=UTC)
    trigger = "habitual-sharing:2026-08-05:9"
    impetus = derive_endogenous_impetus(
        state={
            "approach_motivation": {
                "trigger": trigger,
                "consumed_trigger": trigger,
            },
            "open_sequences": [],
        },
        behavioral_rhythm={
            "sample_count": 100,
            "proactive_turn_rate": 0.6,
            "active_hours": [9],
            "timezone_offset_minutes": 0,
        },
        now=now,
    )

    assert impetus is None


def test_stronger_endogenous_impetus_replaces_stale_weak_user_motive() -> None:
    from moonlightbox.branches.proactive import derive_endogenous_impetus

    now = datetime(2026, 8, 5, 9, tzinfo=UTC)
    impetus = derive_endogenous_impetus(
        state={
            "approach_motivation": {
                "score": 0.45,
                "trigger": "user-message-2",
            },
            "open_sequences": [],
        },
        behavioral_rhythm={
            "sample_count": 100,
            "proactive_turn_rate": 0.6,
            "active_hours": [9],
            "timezone_offset_minutes": 0,
        },
        now=now,
    )

    assert impetus is not None
    assert impetus.trigger == "habitual-sharing:2026-08-05:9"
    assert impetus.score > 0.45


def test_weaker_endogenous_impetus_does_not_replace_stronger_pending_motive() -> None:
    from moonlightbox.branches.proactive import derive_endogenous_impetus

    impetus = derive_endogenous_impetus(
        state={
            "approach_motivation": {
                "score": 0.8,
                "trigger": "important-current-goal",
            },
            "open_sequences": [],
        },
        behavioral_rhythm=None,
        recent_reflection_id="reflection-1",
        now=datetime(2026, 8, 5, 9, tzinfo=UTC),
    )

    assert impetus is None
