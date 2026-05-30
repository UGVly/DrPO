"""Loss helpers for one-step SPO training."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def pairwise_spo_loss(
    current_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    rewards: torch.Tensor,
    *,
    beta: float,
    clip_range: float,
    max_log_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute pairwise sample optimization loss for adjacent sample pairs."""
    if current_logp.ndim != 1 or ref_logp.ndim != 1 or rewards.ndim != 1:
        raise ValueError("current_logp, ref_logp, and rewards must be rank-1 tensors.")
    if current_logp.shape != ref_logp.shape or current_logp.shape != rewards.shape:
        raise ValueError("current_logp, ref_logp, and rewards must have the same shape.")
    if current_logp.numel() % 2:
        raise ValueError("SPO expects an even number of generated samples per prompt.")

    log_ratio = (current_logp.float() - ref_logp.float()).clamp(-max_log_ratio, max_log_ratio).view(-1, 2)
    if clip_range > 0:
        ratio = torch.exp(log_ratio).clamp(1.0 - clip_range, 1.0 + clip_range)
        log_ratio = torch.log(ratio)
    else:
        ratio = torch.exp(log_ratio)

    reward_pairs = rewards.float().view(-1, 2)
    preference = torch.zeros_like(reward_pairs)
    first_wins = reward_pairs[:, 0] > reward_pairs[:, 1]
    second_wins = reward_pairs[:, 1] > reward_pairs[:, 0]
    preference[first_wins, 0] = 1.0
    preference[first_wins, 1] = -1.0
    preference[second_wins, 0] = -1.0
    preference[second_wins, 1] = 1.0
    non_tie = preference.abs().sum(dim=1) > 0

    pair_logits = beta * (log_ratio * preference).sum(dim=1)
    if non_tie.any():
        loss = -F.logsigmoid(pair_logits[non_tie]).mean()
    else:
        loss = log_ratio.sum() * 0.0

    reward_margin = (reward_pairs[:, 0] - reward_pairs[:, 1]).abs()
    model_margin = log_ratio[:, 0] - log_ratio[:, 1]
    reward_sign = torch.sign(reward_pairs[:, 0] - reward_pairs[:, 1])
    pair_accuracy = ((model_margin * reward_sign) > 0).float()[non_tie].mean() if non_tie.any() else log_ratio.sum() * 0.0
    winner_log_ratio = (log_ratio * (preference > 0).float()).sum(dim=1)[non_tie]
    loser_log_ratio = (log_ratio * (preference < 0).float()).sum(dim=1)[non_tie]

    stats = {
        "spo_logit": pair_logits[non_tie].mean() if non_tie.any() else log_ratio.sum() * 0.0,
        "reward_margin": reward_margin.mean(),
        "pair_accuracy": pair_accuracy,
        "ratio_mean": ratio.mean(),
        "winner_log_ratio": winner_log_ratio.mean() if non_tie.any() else log_ratio.sum() * 0.0,
        "loser_log_ratio": loser_log_ratio.mean() if non_tie.any() else log_ratio.sum() * 0.0,
        "tie_fraction": 1.0 - non_tie.float().mean(),
    }
    return loss, stats
