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


def _as_weight(batch: int, count: int, value: float, device, dtype) -> torch.Tensor:
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
