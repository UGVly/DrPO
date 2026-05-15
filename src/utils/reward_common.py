from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_model_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)


def normalize_prompts(prompt: str | Sequence[str], count: int) -> list[str]:
    if isinstance(prompt, str):
        return [prompt] * count
    prompts = list(prompt)
    if len(prompts) != count:
        raise ValueError(f"Expected {count} prompts, got {len(prompts)}.")
    return prompts


def strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "_orig_mod."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value
    return normalized


def select_top_fraction_indices(scores: torch.Tensor, fraction: float, min_count: int = 1) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D reward scores, got shape {tuple(scores.shape)}.")
    if scores.numel() == 0:
        raise ValueError("Expected at least one reward score.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"Expected fraction in (0, 1], got {fraction}.")
    if min_count < 1:
        raise ValueError(f"Expected min_count >= 1, got {min_count}.")

    keep_count = min(scores.shape[0], max(min_count, math.ceil(scores.shape[0] * fraction)))
    return torch.topk(scores, k=keep_count, largest=True).indices


def to_feature_tensor(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out

    image_embeds = getattr(out, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds

    text_embeds = getattr(out, "text_embeds", None)
    if isinstance(text_embeds, torch.Tensor):
        return text_embeds

    pooler_output = getattr(out, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    last_hidden_state = getattr(out, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0, :]

    if isinstance(out, (tuple, list)) and out:
        first = out[0]
        if isinstance(first, torch.Tensor):
            return first[:, 0, :] if first.ndim == 3 else first

    raise TypeError(f"Unsupported feature output type: {type(out)}")
