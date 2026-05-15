from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def parse_csv_floats(value: str) -> tuple[float, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float.")
    return tuple(float(item) for item in items)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    return tuple(int(item) for item in items)


def parse_csv_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def safe_std(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var.clamp_min(0.0) + eps).to(dtype=x.dtype)


def choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups:
        groups -= 1
    return max(groups, 1)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def normalize_prompts(prompt: str | Sequence[str], count: int) -> list[str]:
    if isinstance(prompt, str):
        return [prompt] * count
    prompts = list(prompt)
    if len(prompts) != count:
        raise ValueError(f"Expected {count} prompts, got {len(prompts)}.")
    return prompts


def select_top_fraction_indices(scores: torch.Tensor, fraction: float, min_count: int = 1) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {tuple(scores.shape)}.")
    if scores.numel() == 0:
        raise ValueError("Expected at least one score.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"Expected fraction in (0, 1], got {fraction}.")
    keep = min(scores.numel(), max(min_count, math.ceil(scores.numel() * fraction)))
    return torch.argsort(scores.float(), descending=True)[:keep]


def remove_indices(source_idx: torch.Tensor, remove_idx: torch.Tensor) -> torch.Tensor:
    if source_idx.numel() == 0 or remove_idx.numel() == 0:
        return source_idx
    keep_mask = ~(source_idx[:, None] == remove_idx[None, :]).any(dim=1)
    return source_idx[keep_mask]


def select_disjoint_pref_indices(
    scores: torch.Tensor,
    *,
    num_pos: int,
    num_neg: int,
    feature_top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape={tuple(scores.shape)}")
    total = int(scores.numel())
    if total < 2:
        raise ValueError("Preference ranking requires at least two generated candidates.")
    if num_pos < 1 or num_neg < 1:
        raise ValueError("num_pos and num_neg must both be >= 1.")
    if num_pos + num_neg > total:
        raise ValueError(
            f"Require num_pos + num_neg <= number of candidates, got {num_pos} + {num_neg} > {total}."
        )
    order = torch.argsort(scores.float(), descending=True)
    best_idx = order[:num_pos]
    worst_idx = order[-num_neg:]
    if feature_top_fraction >= 1.0:
        feature_top_idx = torch.arange(total, device=scores.device)
        return best_idx, worst_idx, feature_top_idx
    feature_top_idx = select_top_fraction_indices(scores, feature_top_fraction)
    feature_top_idx = remove_indices(feature_top_idx, worst_idx)
    if feature_top_idx.numel() == 0:
        feature_top_idx = best_idx[:1]
    return best_idx, worst_idx, feature_top_idx


def add_rank_selection_stats(
    info: dict[str, torch.Tensor],
    scores: torch.Tensor,
    best_idx: torch.Tensor,
    worst_idx: torch.Tensor,
    feature_top_idx: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    scores_f = scores.float()
    best_scores = scores_f.index_select(0, best_idx)
    worst_scores = scores_f.index_select(0, worst_idx)
    feature_scores = scores_f.index_select(0, feature_top_idx)
    info[f"{prefix}_best_score"] = best_scores.mean().detach()
    info[f"{prefix}_worst_score"] = worst_scores.mean().detach()
    info[f"{prefix}_selected_score_gap"] = (best_scores.mean() - worst_scores.mean()).detach()
    info[f"{prefix}_feature_top_score_mean"] = feature_scores.mean().detach()
    info[f"{prefix}_selected_pos_count"] = scores_f.new_tensor(float(best_idx.numel()))
    info[f"{prefix}_selected_neg_count"] = scores_f.new_tensor(float(worst_idx.numel()))
    info[f"{prefix}_feature_top_count"] = scores_f.new_tensor(float(feature_top_idx.numel()))
    return info


def strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        output[clean_key] = value
    return output


def feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0, :]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        first = output[0]
        return first[:, 0, :] if first.ndim == 3 else first
    raise TypeError(f"Unsupported model output type: {type(output)}")
