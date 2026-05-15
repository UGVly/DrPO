"""Loss helpers for one-step Neighbor GRPO training."""

from __future__ import annotations

import math
from typing import Literal

import torch

DistanceReduction = Literal["mean", "sum"]


def validate_neighbor_sigma(sigma: float) -> None:
    if not (0.0 < sigma <= 1.0):
        raise ValueError(f"neighbor_sigma must satisfy 0 < sigma <= 1, got {sigma}")


def construct_neighbor_noises(
    base_noise: torch.Tensor,
    deltas: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Construct correlated Neighbor-GRPO initial noises.

    Args:
        base_noise: Base noise with shape ``(... )``.
        deltas: Independent noise with shape ``(G, *base_noise.shape)``.
        sigma: Neighborhood strength in ``(0, 1]``.
    """

    validate_neighbor_sigma(float(sigma))
    if deltas.ndim != base_noise.ndim + 1:
        raise ValueError(
            "deltas must have shape (group, *base_noise.shape); "
            f"got base={tuple(base_noise.shape)} deltas={tuple(deltas.shape)}"
        )
    if tuple(deltas.shape[1:]) != tuple(base_noise.shape):
        raise ValueError(
            "deltas must have shape (group, *base_noise.shape); "
            f"got base={tuple(base_noise.shape)} deltas={tuple(deltas.shape)}"
        )
    base_scale = math.sqrt(max(0.0, 1.0 - float(sigma) ** 2))
    return base_scale * base_noise.unsqueeze(0) + float(sigma) * deltas


def quasi_norm_advantages(
    rewards: torch.Tensor,
    p: float = 0.8,
    scale: float = 1.0,
    clip: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return centered group advantages with Neighbor-GRPO quasi-norm scaling."""

    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D, got shape={tuple(rewards.shape)}")
    if rewards.numel() < 2:
        raise ValueError("at least two rewards are required for group advantages")
    if not (0.0 < p <= 2.0):
        raise ValueError(f"p must satisfy 0 < p <= 2, got {p}")
    centered = rewards.float() - rewards.float().mean()
    denom = centered.abs().pow(float(p)).sum().pow(1.0 / float(p))
    if denom <= eps:
        advantages = torch.zeros_like(centered)
    else:
        advantages = math.sqrt(float(rewards.numel())) * centered / denom.clamp_min(eps)
    advantages = advantages * float(scale)
    if clip > 0:
        advantages = advantages.clamp(-float(clip), float(clip))
    return advantages.to(device=rewards.device)


def latent_distances(
    candidates: torch.Tensor,
    anchor: torch.Tensor,
    reduction: DistanceReduction = "mean",
) -> torch.Tensor:
    """Compute per-candidate squared latent distances to an anchor latent."""

    if candidates.ndim < 2:
        raise ValueError(f"candidates must have shape (group, ...), got {tuple(candidates.shape)}")
    if anchor.shape == candidates.shape[1:]:
        anchor_expanded = anchor.unsqueeze(0)
    elif anchor.shape == (1, *candidates.shape[1:]):
        anchor_expanded = anchor
    else:
        raise ValueError(
            "anchor must have shape candidates.shape[1:] or (1, *candidates.shape[1:]); "
            f"got anchor={tuple(anchor.shape)} candidates={tuple(candidates.shape)}"
        )
    diff = (candidates.float() - anchor_expanded.float()).square().flatten(1)
    if reduction == "mean":
        return diff.mean(dim=1)
    if reduction == "sum":
        return diff.sum(dim=1)
    raise ValueError(f"Unsupported distance reduction: {reduction}")


def softmax_distance_log_probs(
    candidates: torch.Tensor,
    anchor: torch.Tensor,
    temperature: float = 1.0,
    reduction: DistanceReduction = "mean",
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    distances = latent_distances(candidates, anchor, reduction=reduction)
    logits = -distances / float(temperature)
    return torch.log_softmax(logits, dim=0)


def neighbor_grpo_loss(
    candidates: torch.Tensor,
    current_anchor: torch.Tensor,
    old_anchor: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    max_log_ratio: float = 10.0,
    temperature: float = 1.0,
    reduction: DistanceReduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO-shaped Neighbor GRPO loss for one trainable anchor."""

    if advantages.ndim != 1:
        raise ValueError(f"advantages must be 1D, got {tuple(advantages.shape)}")
    if candidates.shape[0] != advantages.numel():
        raise ValueError(
            f"candidate count ({candidates.shape[0]}) must equal advantage count ({advantages.numel()})"
        )
    if clip_range <= 0:
        raise ValueError(f"clip_range must be > 0, got {clip_range}")

    current_log_probs = softmax_distance_log_probs(
        candidates.detach(),
        current_anchor,
        temperature=temperature,
        reduction=reduction,
    )
    old_log_probs = softmax_distance_log_probs(
        candidates.detach(),
        old_anchor.detach(),
        temperature=temperature,
        reduction=reduction,
    ).detach()
    log_ratio = (current_log_probs - old_log_probs).clamp(-float(max_log_ratio), float(max_log_ratio))
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
    adv = advantages.detach().to(device=candidates.device, dtype=torch.float32)

    surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
    policy_loss = -surrogate.mean()
    approx_kl = (ratio - 1.0 - log_ratio).mean()
    entropy = -(current_log_probs.exp() * current_log_probs).sum()
    stats = {
        "ratio_mean": ratio.detach().mean(),
        "ratio_std": ratio.detach().std(unbiased=False),
        "clipfrac": ratio.detach().ne(clipped_ratio.detach()).float().mean(),
        "approx_kl": approx_kl.detach(),
        "approx_kl_loss": approx_kl,
        "entropy": entropy.detach(),
        "old_entropy": (-(old_log_probs.exp() * old_log_probs).sum()).detach(),
    }
    return policy_loss, stats
