from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

DRIFT_KERNELS = ("laplacian", "exponential", "rbf", "cosine")


def normalize_drift_kernel(kernel: str) -> str:
    kernel = kernel.lower()
    if kernel == "exponential":
        return "laplacian"
    if kernel not in DRIFT_KERNELS:
        raise ValueError(f"Unknown drift kernel: {kernel}. Available: {DRIFT_KERNELS}")
    return kernel


@dataclass(frozen=True)
class DriftWeights:
    positive: float = 10.0
    negative: float = 10.0
    reference: float = 0.0
    reference_loss: float = 0.2
    frozen_feature_l2: float = 0.0


@dataclass(frozen=True)
class DriftRadii:
    preference: tuple[float, ...] = (0.02, 0.05, 0.2)
    reference: tuple[float, ...] = (1e-5,)


def pairwise_l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x2 = torch.einsum("bnd,bnd->bn", x, x)
    y2 = torch.einsum("bmd,bmd->bm", y, y)
    xy = torch.einsum("bnd,bmd->bnm", x, y)
    return (x2[:, :, None] + y2[:, None, :] - 2.0 * xy).clamp_min(eps).sqrt()


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1, eps=eps)
    y = F.normalize(y.float(), dim=-1, eps=eps)
    return torch.einsum("bnd,bmd->bnm", x, y)


def kernel_logits(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    radius: float,
    kernel: str = "laplacian",
    scale: torch.Tensor | float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return pairwise kernel logits where larger values mean stronger affinity."""
    if radius <= 0:
        raise ValueError(f"Kernel radius must be > 0, got {radius}.")
    kernel = normalize_drift_kernel(kernel)
    if kernel == "cosine":
        return pairwise_cosine(x, y, eps=eps) / float(radius)
    distances = pairwise_l2(x, y, eps=eps)
    if scale is not None:
        distances = distances / torch.as_tensor(scale, device=distances.device, dtype=distances.dtype).clamp_min(eps)
    if kernel == "laplacian":
        return -distances / float(radius)
    if kernel == "rbf":
        normalized = distances / float(radius)
        return -0.5 * normalized.square()
    raise AssertionError(f"Unhandled drift kernel: {kernel}")


def _as_weight(batch: int, count: int, value: float | torch.Tensor, device, dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        weight = value.to(device=device, dtype=dtype)
        if weight.ndim == 1:
            weight = weight.unsqueeze(0).expand(batch, -1)
        if weight.shape != (batch, count):
            raise ValueError(f"Expected weight with shape {(batch, count)}, got {tuple(weight.shape)}.")
        return weight
    return torch.full((batch, count), float(value), device=device, dtype=dtype)


def drift_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    positive_weight: float,
    negative_weight: float,
    radii: Sequence[float],
    mask_negative_self: bool = False,
    kernel: str = "laplacian",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the feature-space DrPO attraction/repulsion loss.

    Shapes are ``(B, N, D)`` for generated features, ``(B, P, D)`` for positive
    anchors, and ``(B, M, D)`` for negative anchors.
    """
    generated = generated.float()
    positive = positive.float()
    negative = negative.float()
    if generated.ndim != 3 or positive.ndim != 3 or negative.ndim != 3:
        raise ValueError("generated, positive, and negative must be rank-3 tensors.")
    if generated.shape[0] != positive.shape[0] or generated.shape[0] != negative.shape[0]:
        raise ValueError("All feature tensors must have the same batch size.")
    batch, num_generated, dim = generated.shape
    num_positive = positive.shape[1]
    num_negative = negative.shape[1]
    if num_positive == 0 or num_negative == 0:
        raise ValueError("Drift loss requires at least one positive and one negative anchor.")

    old_generated = generated.detach()
    targets = torch.cat([old_generated, negative, positive], dim=1)
    target_weights = torch.cat(
        [
            torch.ones((batch, num_generated), device=generated.device, dtype=generated.dtype),
            _as_weight(batch, num_negative, negative_weight, generated.device, generated.dtype),
            _as_weight(batch, num_positive, positive_weight, generated.device, generated.dtype),
        ],
        dim=1,
    )

    distances = pairwise_l2(old_generated, targets)
    scale = ((distances * target_weights[:, None, :]).mean() / target_weights.mean().clamp_min(1e-8)).clamp_min(1e-3)
    scale_per_dim = (scale / math.sqrt(dim)).clamp_min(1e-3)

    eye = torch.eye(num_generated, device=generated.device, dtype=generated.dtype)[None]
    mask = torch.cat(
        [eye, torch.zeros((1, num_generated, num_negative + num_positive), device=generated.device, dtype=generated.dtype)],
        dim=2,
    )
    if mask_negative_self:
        if num_negative != num_generated:
            raise ValueError("mask_negative_self requires negative anchors to match generated count.")
        mask = mask + torch.cat(
            [
                torch.zeros((1, num_generated, num_generated), device=generated.device, dtype=generated.dtype),
                eye,
                torch.zeros((1, num_generated, num_positive), device=generated.device, dtype=generated.dtype),
            ],
            dim=2,
        )
    masked_logit_penalty = mask

    old_scaled = old_generated / scale_per_dim
    targets_scaled = targets / scale_per_dim
    force = torch.zeros_like(old_scaled)
    normalized_kernel = normalize_drift_kernel(kernel)
    info = {
        "feature_scale": scale.detach(),
        "feature_scale_per_dim": scale_per_dim.detach(),
        "drift_kernel": generated.new_tensor(float(DRIFT_KERNELS.index(normalized_kernel))),
    }
    positive_start = num_generated + num_negative

    for radius in radii:
        logits = kernel_logits(
            old_generated,
            targets,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        logits = logits - (100.0 / float(radius)) * masked_logit_penalty
        affinity = (torch.softmax(logits, dim=-1) * torch.softmax(logits, dim=-2)).clamp_min(1e-6).sqrt()
        affinity = affinity * target_weights[:, None, :]
        affinity_negative = affinity[:, :, :positive_start]
        affinity_positive = affinity[:, :, positive_start:]
        positive_mass = affinity_positive.sum(dim=-1, keepdim=True)
        negative_mass = affinity_negative.sum(dim=-1, keepdim=True)
        coeff = torch.cat([-affinity_negative * positive_mass, affinity_positive * negative_mass], dim=2)
        force_radius = torch.einsum("bnt,btd->bnd", coeff, targets_scaled)
        force_radius = force_radius - coeff.sum(dim=-1, keepdim=True) * old_scaled
        rms = force_radius.pow(2).mean().clamp_min(1e-8).sqrt()
        info[f"drift_rms_{radius}"] = rms.detach()
        force = force + force_radius / rms

    goal = (old_scaled + force).detach()
    loss = ((generated / scale_per_dim - goal) ** 2).mean(dim=(-1, -2))
    return loss, info


def reward_kernel_drift_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
    *,
    reward_alpha: float,
    radii: Sequence[float],
    kernel: str = "laplacian",
    reward_logit_clip: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute reward-weighted per-sample drift without binary pos/neg labels.

    For every generated feature ``x``, the drift direction follows:

      reward-weighted mean over generated samples
      - x
      + kernel mean over reference samples
      - unweighted kernel mean over generated samples

    The reward weights are ``exp(score / reward_alpha)`` inside the kernel
    normalization. Targets are computed from detached features, matching
    ``drift_loss``'s stop-gradient force construction.
    """
    if reward_alpha <= 0:
        raise ValueError(f"reward_alpha must be > 0, got {reward_alpha}.")
    generated = generated.float()
    reference = reference.float()
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("generated and reference must be rank-3 tensors.")
    if generated.shape != reference.shape:
        raise ValueError(
            "reward_kernel_drift_loss expects generated and reference to have "
            f"the same shape, got {tuple(generated.shape)} and {tuple(reference.shape)}."
        )
    batch, num_generated, dim = generated.shape
    scores = scores.to(device=generated.device, dtype=generated.dtype)
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.shape != (batch, num_generated):
        raise ValueError(f"Expected scores with shape {(batch, num_generated)}, got {tuple(scores.shape)}.")

    old_generated = generated.detach()
    old_reference = reference.detach()
    targets = torch.cat([old_generated, old_reference], dim=1)
    distances = pairwise_l2(old_generated, targets)
    scale = distances.mean().clamp_min(1e-3)
    scale_per_dim = (scale / math.sqrt(dim)).clamp_min(1e-3)

    old_scaled = old_generated / scale_per_dim
    reference_scaled = old_reference / scale_per_dim
    force = torch.zeros_like(old_scaled)
    normalized_kernel = normalize_drift_kernel(kernel)

    reward_logits = scores / float(reward_alpha)
    reward_logits = reward_logits - reward_logits.max(dim=-1, keepdim=True).values
    if reward_logit_clip > 0:
        reward_logits = reward_logits.clamp(min=-float(reward_logit_clip), max=float(reward_logit_clip))
    reward_probs = torch.softmax(reward_logits, dim=-1)
    reward_effective_count = (1.0 / reward_probs.square().sum(dim=-1).clamp_min(1e-8)).mean()

    info = {
        "feature_scale": scale.detach(),
        "feature_scale_per_dim": scale_per_dim.detach(),
        "drift_kernel": generated.new_tensor(float(DRIFT_KERNELS.index(normalized_kernel))),
        "reward_kernel_alpha": generated.new_tensor(float(reward_alpha)),
        "reward_kernel_effective_count": reward_effective_count.detach(),
        "reward_kernel_weight_max": reward_probs.max(dim=-1).values.mean().detach(),
        "reward_kernel_weight_min": reward_probs.min(dim=-1).values.mean().detach(),
    }

    for radius in radii:
        generated_logits = kernel_logits(
            old_generated,
            old_generated,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        reference_logits = kernel_logits(
            old_generated,
            old_reference,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        reward_affinity = torch.softmax(generated_logits + reward_logits[:, None, :], dim=-1)
        generated_affinity = torch.softmax(generated_logits, dim=-1)
        reference_affinity = torch.softmax(reference_logits, dim=-1)

        reward_mean = torch.einsum("bnm,bmd->bnd", reward_affinity, old_scaled)
        generated_mean = torch.einsum("bnm,bmd->bnd", generated_affinity, old_scaled)
        reference_mean = torch.einsum("bnm,bmd->bnd", reference_affinity, reference_scaled)
        force_radius = reward_mean - old_scaled + reference_mean - generated_mean
        rms = force_radius.pow(2).mean().clamp_min(1e-8).sqrt()
        info[f"reward_kernel_drift_rms_{radius}"] = rms.detach()
        force = force + force_radius / rms

    goal = (old_scaled + force).detach()
    loss = ((generated / scale_per_dim - goal) ** 2).mean(dim=(-1, -2))
    return loss, info


def reward_contrastive_kernel_drift_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
    *,
    reward_alpha: float,
    radii: Sequence[float],
    kernel: str = "laplacian",
    reward_logit_clip: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute signed AWR drift with soft high-score and low-score anchors.

    This keeps the per-sample AWR property, but makes the force closer to
    binary DrPO: high-reward generated samples form a soft positive set and
    low-reward generated samples form a soft negative set.
    """
    if reward_alpha <= 0:
        raise ValueError(f"reward_alpha must be > 0, got {reward_alpha}.")
    generated = generated.float()
    reference = reference.float()
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("generated and reference must be rank-3 tensors.")
    if generated.shape != reference.shape:
        raise ValueError(
            "reward_contrastive_kernel_drift_loss expects generated and reference to have "
            f"the same shape, got {tuple(generated.shape)} and {tuple(reference.shape)}."
        )
    batch, num_generated, dim = generated.shape
    scores = scores.to(device=generated.device, dtype=generated.dtype)
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.shape != (batch, num_generated):
        raise ValueError(f"Expected scores with shape {(batch, num_generated)}, got {tuple(scores.shape)}.")

    old_generated = generated.detach()
    old_reference = reference.detach()
    targets = torch.cat([old_generated, old_reference], dim=1)
    distances = pairwise_l2(old_generated, targets)
    scale = distances.mean().clamp_min(1e-3)
    scale_per_dim = (scale / math.sqrt(dim)).clamp_min(1e-3)

    old_scaled = old_generated / scale_per_dim
    reference_scaled = old_reference / scale_per_dim
    force = torch.zeros_like(old_scaled)
    normalized_kernel = normalize_drift_kernel(kernel)

    pos_logits = scores / float(reward_alpha)
    neg_logits = -scores / float(reward_alpha)
    pos_logits = pos_logits - pos_logits.max(dim=-1, keepdim=True).values
    neg_logits = neg_logits - neg_logits.max(dim=-1, keepdim=True).values
    if reward_logit_clip > 0:
        clip = float(reward_logit_clip)
        pos_logits = pos_logits.clamp(min=-clip, max=clip)
        neg_logits = neg_logits.clamp(min=-clip, max=clip)
    pos_probs = torch.softmax(pos_logits, dim=-1)
    neg_probs = torch.softmax(neg_logits, dim=-1)
    pos_effective_count = (1.0 / pos_probs.square().sum(dim=-1).clamp_min(1e-8)).mean()
    neg_effective_count = (1.0 / neg_probs.square().sum(dim=-1).clamp_min(1e-8)).mean()

    info = {
        "feature_scale": scale.detach(),
        "feature_scale_per_dim": scale_per_dim.detach(),
        "drift_kernel": generated.new_tensor(float(DRIFT_KERNELS.index(normalized_kernel))),
        "reward_kernel_alpha": generated.new_tensor(float(reward_alpha)),
        "reward_kernel_effective_count": pos_effective_count.detach(),
        "reward_kernel_weight_max": pos_probs.max(dim=-1).values.mean().detach(),
        "reward_kernel_weight_min": pos_probs.min(dim=-1).values.mean().detach(),
        "reward_kernel_negative_effective_count": neg_effective_count.detach(),
        "reward_kernel_negative_weight_max": neg_probs.max(dim=-1).values.mean().detach(),
    }

    for radius in radii:
        generated_logits = kernel_logits(
            old_generated,
            old_generated,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        reference_logits = kernel_logits(
            old_generated,
            old_reference,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        pos_affinity = torch.softmax(generated_logits + pos_logits[:, None, :], dim=-1)
        neg_affinity = torch.softmax(generated_logits + neg_logits[:, None, :], dim=-1)
        reference_affinity = torch.softmax(reference_logits, dim=-1)

        pos_mean = torch.einsum("bnm,bmd->bnd", pos_affinity, old_scaled)
        neg_mean = torch.einsum("bnm,bmd->bnd", neg_affinity, old_scaled)
        reference_mean = torch.einsum("bnm,bmd->bnd", reference_affinity, reference_scaled)
        force_radius = pos_mean - old_scaled + reference_mean - neg_mean
        rms = force_radius.pow(2).mean().clamp_min(1e-8).sqrt()
        info[f"reward_kernel_contrastive_drift_rms_{radius}"] = rms.detach()
        force = force + force_radius / rms

    goal = (old_scaled + force).detach()
    loss = ((generated / scale_per_dim - goal) ** 2).mean(dim=(-1, -2))
    return loss, info


def reward_topk_contrastive_kernel_drift_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
    *,
    reward_alpha: float,
    radii: Sequence[float],
    kernel: str = "laplacian",
    reward_logit_clip: float = 20.0,
    top_fraction: float = 0.5,
    force_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a stronger signed AWR drift using soft top/bottom anchors.

    This is still per-sample AWR rather than a hard binary classification loss:
    anchors inside the top and bottom sets are weighted by their reward scores.
    The top/bottom split only prevents the full-batch softmax from washing out
    the update direction when score differences are small.
    """
    if reward_alpha <= 0:
        raise ValueError(f"reward_alpha must be > 0, got {reward_alpha}.")
    if not (0 < top_fraction <= 1):
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
    if force_scale <= 0:
        raise ValueError(f"force_scale must be > 0, got {force_scale}.")
    generated = generated.float()
    reference = reference.float()
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("generated and reference must be rank-3 tensors.")
    if generated.shape != reference.shape:
        raise ValueError(
            "reward_topk_contrastive_kernel_drift_loss expects generated and reference to have "
            f"the same shape, got {tuple(generated.shape)} and {tuple(reference.shape)}."
        )
    batch, num_generated, dim = generated.shape
    scores = scores.to(device=generated.device, dtype=generated.dtype)
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.shape != (batch, num_generated):
        raise ValueError(f"Expected scores with shape {(batch, num_generated)}, got {tuple(scores.shape)}.")

    old_generated = generated.detach()
    old_reference = reference.detach()
    targets = torch.cat([old_generated, old_reference], dim=1)
    distances = pairwise_l2(old_generated, targets)
    scale = distances.mean().clamp_min(1e-3)
    scale_per_dim = (scale / math.sqrt(dim)).clamp_min(1e-3)

    top_count = max(1, min(num_generated, math.ceil(num_generated * float(top_fraction))))
    bottom_count = top_count
    top_idx = scores.topk(k=top_count, dim=-1, largest=True).indices
    bottom_idx = scores.topk(k=bottom_count, dim=-1, largest=False).indices
    gather_top = top_idx[..., None].expand(-1, -1, dim)
    gather_bottom = bottom_idx[..., None].expand(-1, -1, dim)
    top_generated = torch.gather(old_generated, dim=1, index=gather_top)
    bottom_generated = torch.gather(old_generated, dim=1, index=gather_bottom)
    top_scores = torch.gather(scores, dim=1, index=top_idx)
    bottom_scores = torch.gather(scores, dim=1, index=bottom_idx)

    old_scaled = old_generated / scale_per_dim
    reference_scaled = old_reference / scale_per_dim
    top_scaled = top_generated / scale_per_dim
    bottom_scaled = bottom_generated / scale_per_dim
    force = torch.zeros_like(old_scaled)
    normalized_kernel = normalize_drift_kernel(kernel)

    pos_logits = top_scores / float(reward_alpha)
    neg_logits = -bottom_scores / float(reward_alpha)
    pos_logits = pos_logits - pos_logits.max(dim=-1, keepdim=True).values
    neg_logits = neg_logits - neg_logits.max(dim=-1, keepdim=True).values
    if reward_logit_clip > 0:
        clip = float(reward_logit_clip)
        pos_logits = pos_logits.clamp(min=-clip, max=clip)
        neg_logits = neg_logits.clamp(min=-clip, max=clip)
    pos_probs = torch.softmax(pos_logits, dim=-1)
    neg_probs = torch.softmax(neg_logits, dim=-1)
    pos_effective_count = (1.0 / pos_probs.square().sum(dim=-1).clamp_min(1e-8)).mean()
    neg_effective_count = (1.0 / neg_probs.square().sum(dim=-1).clamp_min(1e-8)).mean()

    info = {
        "feature_scale": scale.detach(),
        "feature_scale_per_dim": scale_per_dim.detach(),
        "drift_kernel": generated.new_tensor(float(DRIFT_KERNELS.index(normalized_kernel))),
        "reward_kernel_alpha": generated.new_tensor(float(reward_alpha)),
        "reward_kernel_effective_count": pos_effective_count.detach(),
        "reward_kernel_weight_max": pos_probs.max(dim=-1).values.mean().detach(),
        "reward_kernel_weight_min": pos_probs.min(dim=-1).values.mean().detach(),
        "reward_kernel_negative_effective_count": neg_effective_count.detach(),
        "reward_kernel_negative_weight_max": neg_probs.max(dim=-1).values.mean().detach(),
        "reward_kernel_top_count": generated.new_tensor(float(top_count)),
        "reward_kernel_force_scale": generated.new_tensor(float(force_scale)),
    }

    for radius in radii:
        positive_logits = kernel_logits(
            old_generated,
            top_generated,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        negative_logits = kernel_logits(
            old_generated,
            bottom_generated,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )
        reference_logits = kernel_logits(
            old_generated,
            old_reference,
            radius=float(radius),
            kernel=kernel,
            scale=scale,
        )

        positive_affinity = torch.softmax(positive_logits + pos_logits[:, None, :], dim=-1)
        negative_affinity = torch.softmax(negative_logits + neg_logits[:, None, :], dim=-1)
        reference_affinity = torch.softmax(reference_logits, dim=-1)

        positive_mean = torch.einsum("bnm,bmd->bnd", positive_affinity, top_scaled)
        negative_mean = torch.einsum("bnm,bmd->bnd", negative_affinity, bottom_scaled)
        reference_mean = torch.einsum("bnm,bmd->bnd", reference_affinity, reference_scaled)
        force_radius = positive_mean - old_scaled + reference_mean - negative_mean
        rms = force_radius.pow(2).mean().clamp_min(1e-8).sqrt()
        info[f"reward_kernel_topk_drift_rms_{radius}"] = rms.detach()
        force = force + float(force_scale) * force_radius / rms

    goal = (old_scaled + force).detach()
    loss = ((generated / scale_per_dim - goal) ** 2).mean(dim=(-1, -2))
    return loss, info


def reward_topk_weighted_binary_drift_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
    *,
    reward_alpha: float,
    positive_weight: float,
    negative_weight: float,
    radii: Sequence[float],
    kernel: str = "laplacian",
    reward_logit_clip: float = 20.0,
    top_fraction: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """AWR rescue objective using binary DrPO's strong positive/negative force.

    Top and bottom anchors are still weighted per sample by reward advantages,
    but the actual feature-space drift uses the same weighted positive/negative
    kernel construction as binary DrPO. This keeps the AWR signal continuous
    while avoiding the weak full-batch barycenter force seen in early runs.
    """
    if reward_alpha <= 0:
        raise ValueError(f"reward_alpha must be > 0, got {reward_alpha}.")
    if not (0 < top_fraction <= 1):
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
    generated = generated.float()
    reference = reference.float()
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("generated and reference must be rank-3 tensors.")
    if generated.shape != reference.shape:
        raise ValueError(
            "reward_topk_weighted_binary_drift_loss expects generated and reference to have "
            f"the same shape, got {tuple(generated.shape)} and {tuple(reference.shape)}."
        )
    batch, num_generated, dim = generated.shape
    scores = scores.to(device=generated.device, dtype=generated.dtype)
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.shape != (batch, num_generated):
        raise ValueError(f"Expected scores with shape {(batch, num_generated)}, got {tuple(scores.shape)}.")

    top_count = max(1, min(num_generated, math.ceil(num_generated * float(top_fraction))))
    bottom_count = top_count
    top_idx = scores.topk(k=top_count, dim=-1, largest=True).indices
    bottom_idx = scores.topk(k=bottom_count, dim=-1, largest=False).indices
    gather_top = top_idx[..., None].expand(-1, -1, dim)
    gather_bottom = bottom_idx[..., None].expand(-1, -1, dim)
    top_generated = torch.gather(generated.detach(), dim=1, index=gather_top)
    bottom_generated = torch.gather(generated.detach(), dim=1, index=gather_bottom)
    top_scores = torch.gather(scores, dim=1, index=top_idx)
    bottom_scores = torch.gather(scores, dim=1, index=bottom_idx)

    pos_logits = top_scores / float(reward_alpha)
    neg_logits = -bottom_scores / float(reward_alpha)
    pos_logits = pos_logits - pos_logits.max(dim=-1, keepdim=True).values
    neg_logits = neg_logits - neg_logits.max(dim=-1, keepdim=True).values
    if reward_logit_clip > 0:
        clip = float(reward_logit_clip)
        pos_logits = pos_logits.clamp(min=-clip, max=clip)
        neg_logits = neg_logits.clamp(min=-clip, max=clip)
    pos_probs = torch.softmax(pos_logits, dim=-1)
    neg_probs = torch.softmax(neg_logits, dim=-1)
    # Preserve the scalar DrPO weight scale while distributing it by AWR mass.
    pos_anchor_weights = float(positive_weight) * float(top_count) * pos_probs
    neg_anchor_weights = float(negative_weight) * float(bottom_count) * neg_probs

    loss, info = drift_loss(
        generated,
        top_generated,
        bottom_generated,
        positive_weight=pos_anchor_weights,
        negative_weight=neg_anchor_weights,
        radii=radii,
        kernel=kernel,
    )
    info.update(
        {
            "reward_kernel_alpha": generated.new_tensor(float(reward_alpha)),
            "reward_kernel_effective_count": (1.0 / pos_probs.square().sum(dim=-1).clamp_min(1e-8)).mean().detach(),
            "reward_kernel_weight_max": pos_probs.max(dim=-1).values.mean().detach(),
            "reward_kernel_weight_min": pos_probs.min(dim=-1).values.mean().detach(),
            "reward_kernel_negative_effective_count": (1.0 / neg_probs.square().sum(dim=-1).clamp_min(1e-8)).mean().detach(),
            "reward_kernel_negative_weight_max": neg_probs.max(dim=-1).values.mean().detach(),
            "reward_kernel_top_count": generated.new_tensor(float(top_count)),
            "reward_kernel_force_scale": generated.new_tensor(1.0),
        }
    )
    return loss, info


def reward_advantage_sign_weighted_binary_drift_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
    *,
    reward_alpha: float,
    positive_weight: float,
    negative_weight: float,
    radii: Sequence[float],
    kernel: str = "laplacian",
    reward_logit_clip: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """AWR objective that splits anchors by the sign of per-prompt advantage.

    Unlike the hard top/bottom variant, this does not force a fixed positive
    and negative fraction. Positive anchors are samples with above-mean reward,
    negative anchors are samples with below-mean reward, and both sides receive
    AWR softmax weights within their own signed partition.
    """
    if reward_alpha <= 0:
        raise ValueError(f"reward_alpha must be > 0, got {reward_alpha}.")
    generated = generated.float()
    reference = reference.float()
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("generated and reference must be rank-3 tensors.")
    if generated.shape != reference.shape:
        raise ValueError(
            "reward_advantage_sign_weighted_binary_drift_loss expects generated and reference to have "
            f"the same shape, got {tuple(generated.shape)} and {tuple(reference.shape)}."
        )
    batch, num_generated, _ = generated.shape
    scores = scores.to(device=generated.device, dtype=generated.dtype)
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.shape != (batch, num_generated):
        raise ValueError(f"Expected scores with shape {(batch, num_generated)}, got {tuple(scores.shape)}.")

    advantage = scores - scores.mean(dim=-1, keepdim=True)
    positive_mask = advantage > 0
    negative_mask = advantage < 0
    # Degenerate equal-score batches are rare with z-scored rewards, but keep a
    # deterministic fallback so the objective never produces all-zero anchors.
    no_positive = ~positive_mask.any(dim=-1)
    no_negative = ~negative_mask.any(dim=-1)
    if no_positive.any():
        positive_mask = positive_mask.clone()
        positive_mask[no_positive] = False
        positive_mask[no_positive, scores[no_positive].argmax(dim=-1)] = True
    if no_negative.any():
        negative_mask = negative_mask.clone()
        negative_mask[no_negative] = False
        negative_mask[no_negative, scores[no_negative].argmin(dim=-1)] = True

    pos_count = positive_mask.sum(dim=-1).to(generated.dtype).clamp_min(1.0)
    neg_count = negative_mask.sum(dim=-1).to(generated.dtype).clamp_min(1.0)
    pos_logits = advantage / float(reward_alpha)
    neg_logits = -advantage / float(reward_alpha)
    if reward_logit_clip > 0:
        clip = float(reward_logit_clip)
        pos_logits = pos_logits.clamp(min=-clip, max=clip)
        neg_logits = neg_logits.clamp(min=-clip, max=clip)
    pos_logits = pos_logits.masked_fill(~positive_mask, torch.finfo(pos_logits.dtype).min)
    neg_logits = neg_logits.masked_fill(~negative_mask, torch.finfo(neg_logits.dtype).min)
    pos_probs = torch.softmax(pos_logits, dim=-1).masked_fill(~positive_mask, 0.0)
    neg_probs = torch.softmax(neg_logits, dim=-1).masked_fill(~negative_mask, 0.0)

    pos_anchor_weights = float(positive_weight) * pos_count[:, None] * pos_probs
    neg_anchor_weights = float(negative_weight) * neg_count[:, None] * neg_probs

    loss, info = drift_loss(
        generated,
        generated.detach(),
        generated.detach(),
        positive_weight=pos_anchor_weights,
        negative_weight=neg_anchor_weights,
        radii=radii,
        kernel=kernel,
    )
    info.update(
        {
            "reward_kernel_alpha": generated.new_tensor(float(reward_alpha)),
            "reward_kernel_effective_count": (1.0 / pos_probs.square().sum(dim=-1).clamp_min(1e-8)).mean().detach(),
            "reward_kernel_weight_max": pos_probs.max(dim=-1).values.mean().detach(),
            "reward_kernel_weight_min": pos_probs.masked_fill(~positive_mask, 1.0).min(dim=-1).values.mean().detach(),
            "reward_kernel_negative_effective_count": (1.0 / neg_probs.square().sum(dim=-1).clamp_min(1e-8)).mean().detach(),
            "reward_kernel_negative_weight_max": neg_probs.max(dim=-1).values.mean().detach(),
            "reward_kernel_top_count": pos_count.mean().detach(),
            "reward_kernel_negative_count": neg_count.mean().detach(),
            "reward_kernel_force_scale": generated.new_tensor(1.0),
        }
    )
    return loss, info


def dual_drift_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: DriftWeights,
    radii: DriftRadii,
    kernel: str = "laplacian",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pref_loss, pref_info = drift_loss(
        generated,
        positive,
        negative,
        positive_weight=weights.positive,
        negative_weight=weights.negative,
        radii=radii.preference,
        kernel=kernel,
    )
    ref_loss, ref_info = drift_loss(
        generated,
        reference,
        generated.detach(),
        positive_weight=weights.reference,
        negative_weight=weights.negative,
        radii=radii.reference,
        mask_negative_self=True,
        kernel=kernel,
    )
    feature_l2 = F.mse_loss(generated.float(), reference.float())
    total = pref_loss.mean() + weights.reference_loss * ref_loss.mean() + weights.frozen_feature_l2 * feature_l2
    return total, {
        "pref_loss": pref_loss.mean().detach(),
        "ref_loss": ref_loss.mean().detach(),
        "feature_l2": feature_l2.detach(),
        "pref_feature_scale": pref_info["feature_scale"],
        "ref_feature_scale": ref_info["feature_scale"],
    }
