import mlx.core as mx
import mlx.nn as nn


def persona_simpo_loss(
    model: nn.Module,
    chosen_batch: mx.array,
    chosen_lengths: mx.array,
    rejected_batch: mx.array,
    rejected_lengths: mx.array,
    *,
    beta: float = 2.0,
    target_margin: float = 0.5,
    sft_weight: float = 0.2,
) -> tuple[mx.array, mx.array]:
    """Reference-free preference loss plus a chosen-response SFT anchor."""

    chosen_logps, chosen_mask = _completion_token_logps(
        model,
        chosen_batch,
        chosen_lengths,
    )
    rejected_logps, rejected_mask = _completion_token_logps(
        model,
        rejected_batch,
        rejected_lengths,
    )
    chosen_average = _masked_average(chosen_logps, chosen_mask, axis=1)
    rejected_average = _masked_average(rejected_logps, rejected_mask, axis=1)
    preference = _simpo_objective(
        chosen_average,
        rejected_average,
        beta=beta,
        target_margin=target_margin,
    )
    chosen_sft = -_masked_average(chosen_logps, chosen_mask)
    loss = preference + sft_weight * chosen_sft
    return loss, mx.array(chosen_batch.shape[0])


def _simpo_objective(
    chosen_average_logps: mx.array,
    rejected_average_logps: mx.array,
    *,
    beta: float,
    target_margin: float,
) -> mx.array:
    margin = beta * (chosen_average_logps - rejected_average_logps) - target_margin
    return mx.logaddexp(mx.zeros_like(margin), -margin).mean()


def _completion_token_logps(
    model: nn.Module,
    batch: mx.array,
    lengths: mx.array,
) -> tuple[mx.array, mx.array]:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    logps = mx.take_along_axis(
        nn.log_softmax(logits, axis=-1),
        targets[..., None],
        axis=-1,
    ).squeeze(-1)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(
        steps >= lengths[:, 0:1],
        steps < lengths[:, 1:2],
    )
    return logps, mask


def _masked_average(
    values: mx.array,
    mask: mx.array,
    *,
    axis: int | None = None,
) -> mx.array:
    numerator = (values * mask).astype(mx.float32).sum(axis=axis)
    denominator = mx.maximum(mask.sum(axis=axis), mx.array(1))
    return numerator / denominator
