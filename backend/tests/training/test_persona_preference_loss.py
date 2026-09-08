import pytest

# 这是保留给历史 MLX 实现的数值单测；Linux 运行时使用
# ``peft_preference_trainer``，没有 MLX 时应跳过而不能阻断 PEFT 测试收集。
mx = pytest.importorskip("mlx.core", reason="MLX 仅用于历史 macOS 偏好训练测试")


def test_simpo_objective_rewards_larger_chosen_margin() -> None:
    from moonlightbox.training.persona_preference_loss import _simpo_objective

    good = _simpo_objective(
        mx.array([-1.0, -1.2]),
        mx.array([-3.0, -2.8]),
        beta=2.0,
        target_margin=0.5,
    )
    bad = _simpo_objective(
        mx.array([-3.0, -2.8]),
        mx.array([-1.0, -1.2]),
        beta=2.0,
        target_margin=0.5,
    )

    assert good.item() < bad.item()


def test_simpo_target_margin_prevents_zero_gap_from_passing() -> None:
    from moonlightbox.training.persona_preference_loss import _simpo_objective

    zero_gap = _simpo_objective(
        mx.array([-2.0]),
        mx.array([-2.0]),
        beta=2.0,
        target_margin=0.5,
    )
    positive_gap = _simpo_objective(
        mx.array([-1.0]),
        mx.array([-2.0]),
        beta=2.0,
        target_margin=0.5,
    )

    assert positive_gap.item() < zero_gap.item()
