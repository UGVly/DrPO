import argparse
import copy
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from PIL import Image
from tqdm.auto import tqdm

from drpo.data import Batch, PromptDataset, collate_preference_batch
from drpo.methods.sdxl_drpo.trainer import (
    _decode_latents_to_tensor,
    _encode_prompts,
    _sdxl_one_step_latents,
    _tensor_to_pil,
)
from drpo.methods.sdxl_common import (
    create_accelerator,
    dtype_for_mixed_precision,
    make_lora_trainable,
    maybe_enable_xformers,
    parse_floats,
    parse_names,
    resume_global_step,
    save_runtime_snapshot,
    save_unet_checkpoint,
    setup_training_logging,
    TrainingStepCallbacks,
    trainable_parameters,
)
from drpo.paths import project_root, require_local_path
from drpo.rewards import build_choice_selectors, resolve_choice_model_weights, resolve_choice_models

logger = logging.getLogger(__name__)
DistanceReduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class SDXLGRPOConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    prompt_file: str
    model_variant: str | None = "fp16"
    choice_model: str = "pickscore"
    choice_models: tuple[str, ...] = ()
    choice_model_weights: tuple[float, ...] = ()
    choice_score_normalize: str = "zscore"
    pickscore_model_name_or_path: str | None = None
    pickscore_processor_name_or_path: str | None = None
    pickscore_allow_remote: bool = False
    seed: int = 42
    resolution: int = 512
    train_batch_size: int = 1
    batchsize_gen: int = 24
    num_inference_steps: int = 1
    guidance_scale: float = 0.0
    max_train_steps: int = 5000
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 2
    max_train_samples: int | None = None
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    checkpointing_steps: int = 100
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    use_lora: bool = True
    resume_from_checkpoint: str | None = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    neighbor_sigma: float = 0.3
    neighbor_num_anchors: int = 4
    neighbor_p_norm: float = 0.8
    neighbor_distance_temperature: float = 1.0
    neighbor_distance_reduction: str = "mean"
    ppo_clip_range: float = 0.2
    max_log_ratio: float = 10.0
    advantage_clip: float = 5.0
    advantage_scale: float = 1.0
    policy_kl_weight: float = 0.0
    ref_model_l2_weight: float = 0.02
    vae_decode_chunk_size: int = 4
    reward_score_batch_size: int = 128
    reward_cache_dir: str | None = None
    reward_cache_interval: int = 1
    reward_cache_images: bool = False


def _parse_names(value: str | None) -> tuple[str, ...]:
    return parse_names(value)


def _parse_floats(value: str) -> tuple[float, ...]:
    return parse_floats(value)


def _dtype_for(config: SDXLGRPOConfig) -> torch.dtype:
    return dtype_for_mixed_precision(config.mixed_precision)


def validate_neighbor_sigma(sigma: float) -> None:
    if not (0.0 < sigma <= 1.0):
        raise ValueError(f"neighbor_sigma must satisfy 0 < sigma <= 1, got {sigma}")


def construct_neighbor_noises(base_noise: torch.Tensor, deltas: torch.Tensor, sigma: float) -> torch.Tensor:
    validate_neighbor_sigma(float(sigma))
    if deltas.ndim != base_noise.ndim + 1 or tuple(deltas.shape[1:]) != tuple(base_noise.shape):
        raise ValueError(
            "deltas must have shape (group, *base_noise.shape); "
            f"got base={tuple(base_noise.shape)} deltas={tuple(deltas.shape)}"
        )
    base_scale = math.sqrt(max(0.0, 1.0 - float(sigma) ** 2))
    return base_scale * base_noise.unsqueeze(0) + float(sigma) * deltas


def quasi_norm_advantages(rewards: torch.Tensor, p: float = 0.8, scale: float = 1.0, clip: float = 5.0, eps: float = 1e-6) -> torch.Tensor:
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


def latent_distances(candidates: torch.Tensor, anchor: torch.Tensor, reduction: DistanceReduction = "mean") -> torch.Tensor:
    if anchor.shape == candidates.shape[1:]:
        anchor = anchor.unsqueeze(0)
    elif anchor.shape != (1, *candidates.shape[1:]):
        raise ValueError(
            "anchor must have shape candidates.shape[1:] or (1, *candidates.shape[1:]); "
            f"got anchor={tuple(anchor.shape)} candidates={tuple(candidates.shape)}"
        )
    diff = (candidates.float() - anchor.float()).square().flatten(1)
    if reduction == "mean":
        return diff.mean(dim=1)
    if reduction == "sum":
        return diff.sum(dim=1)
    raise ValueError(f"Unsupported distance reduction: {reduction}")


def softmax_distance_log_probs(candidates: torch.Tensor, anchor: torch.Tensor, temperature: float = 1.0, reduction: DistanceReduction = "mean") -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return torch.log_softmax(-latent_distances(candidates, anchor, reduction=reduction) / float(temperature), dim=0)


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
    if advantages.ndim != 1:
        raise ValueError(f"advantages must be 1D, got {tuple(advantages.shape)}")
    if candidates.shape[0] != advantages.numel():
        raise ValueError(f"candidate count ({candidates.shape[0]}) must equal advantage count ({advantages.numel()})")
    if clip_range <= 0:
        raise ValueError(f"clip_range must be > 0, got {clip_range}")
    current_log_probs = softmax_distance_log_probs(candidates.detach(), current_anchor, temperature=temperature, reduction=reduction)
    old_log_probs = softmax_distance_log_probs(candidates.detach(), old_anchor.detach(), temperature=temperature, reduction=reduction).detach()
    log_ratio = (current_log_probs - old_log_probs).clamp(-float(max_log_ratio), float(max_log_ratio))
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
    adv = advantages.detach().to(device=candidates.device, dtype=torch.float32)
    policy_loss = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()
    approx_kl = (ratio - 1.0 - log_ratio).mean()
    entropy = -(current_log_probs.exp() * current_log_probs).sum()
    return policy_loss, {
        "ratio_mean": ratio.detach().mean(),
        "ratio_std": ratio.detach().std(unbiased=False),
        "clipfrac": ratio.detach().ne(clipped_ratio.detach()).float().mean(),
        "approx_kl": approx_kl.detach(),
        "approx_kl_loss": approx_kl,
        "entropy": entropy.detach(),
    }


def batched_neighbor_grpo_loss(
    candidates: torch.Tensor,
    current_anchors: torch.Tensor,
    old_anchors: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    max_log_ratio: float = 10.0,
    temperature: float = 1.0,
    reduction: DistanceReduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if candidates.ndim < 3:
        raise ValueError(f"candidates must have shape [B, G, ...], got {tuple(candidates.shape)}")
    if current_anchors.ndim != candidates.ndim:
        raise ValueError(f"current_anchors must have shape [B, A, ...], got {tuple(current_anchors.shape)}")
    if old_anchors.shape != current_anchors.shape:
        raise ValueError(f"old_anchors shape {tuple(old_anchors.shape)} must match current_anchors {tuple(current_anchors.shape)}")
    if advantages.shape != candidates.shape[:2]:
        raise ValueError(f"advantages must have shape {tuple(candidates.shape[:2])}, got {tuple(advantages.shape)}")
    if clip_range <= 0:
        raise ValueError(f"clip_range must be > 0, got {clip_range}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    current_diff = (candidates.detach().float().unsqueeze(1) - current_anchors.float().unsqueeze(2)).square().flatten(3)
    old_diff = (candidates.detach().float().unsqueeze(1) - old_anchors.detach().float().unsqueeze(2)).square().flatten(3)
    if reduction == "mean":
        current_dist = current_diff.mean(dim=3)
        old_dist = old_diff.mean(dim=3)
    elif reduction == "sum":
        current_dist = current_diff.sum(dim=3)
        old_dist = old_diff.sum(dim=3)
    else:
        raise ValueError(f"Unsupported distance reduction: {reduction}")

    current_log_probs = torch.log_softmax(-current_dist / float(temperature), dim=2)
    old_log_probs = torch.log_softmax(-old_dist / float(temperature), dim=2).detach()
    log_ratio = (current_log_probs - old_log_probs).clamp(-float(max_log_ratio), float(max_log_ratio))
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
    adv = advantages.detach().to(device=candidates.device, dtype=torch.float32).unsqueeze(1)
    policy_loss = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()
    approx_kl = ratio - 1.0 - log_ratio
    entropy = -(current_log_probs.exp() * current_log_probs).sum(dim=2).mean()
    return policy_loss, {
        "ratio_mean": ratio.detach().mean(),
        "ratio_std": ratio.detach().std(dim=2, unbiased=False).mean(),
        "clipfrac": ratio.detach().ne(clipped_ratio.detach()).float().mean(),
        "approx_kl": approx_kl.detach().mean(),
        "approx_kl_loss": approx_kl.mean(),
        "entropy": entropy.detach(),
    }


def _resolve_unet_lora_dir(path: str | Path) -> Path:
    checkpoint_dir = Path(path).expanduser()
    if checkpoint_dir.is_file() and checkpoint_dir.name == "adapter_model.safetensors":
        return checkpoint_dir.parent
    if checkpoint_dir.is_dir() and checkpoint_dir.name == "unet_lora":
        return checkpoint_dir
    if checkpoint_dir.is_dir() and (checkpoint_dir / "unet_lora" / "adapter_model.safetensors").is_file():
        return checkpoint_dir / "unet_lora"
    raise FileNotFoundError(f"Expected checkpoint dir or unet_lora adapter path, got: {checkpoint_dir}")


def _resume_global_step(path: str | Path | None) -> int:
    if not path:
        return 0
    checkpoint_dir = Path(path).expanduser()
    if checkpoint_dir.name == "unet_lora":
        checkpoint_dir = checkpoint_dir.parent
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.is_file():
        return 0
    return int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0))


def _make_trainable(unet, config: SDXLGRPOConfig):
    return make_lora_trainable(unet, config)


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return trainable_parameters(model)


def _save_checkpoint(accelerator: Accelerator, unet, config: SDXLGRPOConfig, checkpoint_dir: Path, global_step: int, checkpoint_type: str) -> None:
    save_unet_checkpoint(
        accelerator,
        unet,
        config,
        checkpoint_dir,
        global_step=global_step,
        checkpoint_type=checkpoint_type,
        metadata={"model_type": "sdxl-turbo-grpo-neighbor"},
    )


def _setup_logging(config: SDXLGRPOConfig, accelerator: Accelerator) -> None:
    setup_training_logging(config, accelerator, logger_name=__name__)


def _save_runtime_snapshot(config: SDXLGRPOConfig, accelerator: Accelerator) -> None:
    save_runtime_snapshot(config, accelerator)


def _validate_config(config: SDXLGRPOConfig) -> None:
    if config.batchsize_gen < 2:
        raise ValueError("batchsize_gen must be >= 2.")
    if config.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1.")
    if config.resolution % 8:
        raise ValueError("resolution must be divisible by 8.")
    if not (0.0 < config.neighbor_sigma <= 1.0):
        raise ValueError("neighbor_sigma must satisfy 0 < sigma <= 1.")
    if config.neighbor_num_anchors < 1 or config.neighbor_num_anchors > config.batchsize_gen:
        raise ValueError("neighbor_num_anchors must be in [1, batchsize_gen].")
    if not (0.0 < config.neighbor_p_norm <= 2.0):
        raise ValueError("neighbor_p_norm must satisfy 0 < p <= 2.")
    if config.neighbor_distance_temperature <= 0:
        raise ValueError("neighbor_distance_temperature must be > 0.")
    if config.ppo_clip_range <= 0:
        raise ValueError("ppo_clip_range must be > 0.")
    if config.vae_decode_chunk_size < 1:
        raise ValueError("vae_decode_chunk_size must be >= 1.")
    if config.reward_score_batch_size < 1:
        raise ValueError("reward_score_batch_size must be >= 1.")
    if config.reward_cache_interval < 0:
        raise ValueError("reward_cache_interval must be >= 0.")


def _normalize_group_rewards(raw: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    if mode == "zscore":
        std = torch.clamp(raw.std(dim=1, unbiased=False, keepdim=True), min=eps)
        return (raw - raw.mean(dim=1, keepdim=True)) / std
    if mode == "none":
        return raw
    raise ValueError(f"Unknown choice score normalization mode: {mode}")


def _score_selector_batches(
    selector,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    *,
    batch_size: int,
    image_paths: Sequence[str] | None = None,
) -> list[float]:
    if len(images) != len(prompts):
        raise ValueError(f"images/prompts length mismatch: {len(images)} vs {len(prompts)}")
    if image_paths is not None and len(image_paths) != len(images):
        raise ValueError(f"image_paths/images length mismatch: {len(image_paths)} vs {len(images)}")
    scores: list[float] = []
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(images), chunk_size):
        end = min(len(images), start + chunk_size)
        prompt_chunk = prompts[start:end]
        if image_paths is not None and hasattr(selector, "score_paths"):
            raw_scores = selector.score_paths(image_paths[start:end], prompt_chunk)
        else:
            raw_scores = selector.score(images[start:end], prompt_chunk)
        scores.extend(float(score) for score in raw_scores)
    return scores


def _as_float_list(raw_scores) -> list[float]:
    if isinstance(raw_scores, torch.Tensor):
        return [float(score) for score in raw_scores.detach().float().cpu().flatten().tolist()]
    return [float(score) for score in raw_scores]


def _score_selector_tensor_batches(
    selector,
    image_tensor: torch.Tensor,
    prompts: Sequence[str],
    *,
    batch_size: int,
) -> list[float]:
    if image_tensor.shape[0] != len(prompts):
        raise ValueError(f"image tensor/prompts length mismatch: {image_tensor.shape[0]} vs {len(prompts)}")
    scores: list[float] = []
    chunk_size = max(1, int(batch_size))
    for start in range(0, image_tensor.shape[0], chunk_size):
        end = min(image_tensor.shape[0], start + chunk_size)
        raw_scores = selector.score_tensor(image_tensor[start:end], prompts[start:end])
        scores.extend(_as_float_list(raw_scores))
    return scores


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _score_grouped_selector(
    model_name: str,
    selector,
    images: Sequence[Image.Image] | None,
    prompts: Sequence[str],
    *,
    batch_size: int,
    image_paths: Sequence[str] | None,
    image_tensor: torch.Tensor | None,
) -> tuple[str, list[float], float]:
    start = time.perf_counter()
    if image_paths is None and image_tensor is not None and hasattr(selector, "score_tensor"):
        scores = _score_selector_tensor_batches(selector, image_tensor, prompts, batch_size=batch_size)
    else:
        if images is None:
            raise ValueError(f"Selector {model_name} requires PIL/path images but none were provided.")
        scores = _score_selector_batches(
            selector,
            images,
            prompts,
            batch_size=batch_size,
            image_paths=image_paths,
        )
    return model_name, scores, time.perf_counter() - start


def _score_reward_ensemble_grouped(
    selectors,
    weights: Sequence[float],
    images: Sequence[Image.Image] | None,
    prompts: Sequence[str],
    *,
    group_size: int,
    batch_size: int,
    normalize: str,
    image_paths: Sequence[str] | None = None,
    image_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(selectors) == 0:
        raise ValueError("Expected at least one reward selector.")
    if len(selectors) != len(weights):
        raise ValueError(f"Expected {len(selectors)} weights, got {len(weights)}.")
    if group_size < 2:
        raise ValueError("group_size must be >= 2.")
    image_count = image_tensor.shape[0] if image_tensor is not None else (len(images) if images is not None else 0)
    if image_count != len(prompts):
        raise ValueError(f"images/prompts length mismatch: {image_count} vs {len(prompts)}")
    if image_count % group_size:
        raise ValueError(f"image count {image_count} must be divisible by group_size {group_size}")

    num_groups = image_count // group_size
    weighted_terms: list[torch.Tensor] = []
    info: dict[str, torch.Tensor] = {}
    selector_items = list(selectors.items())
    if _truthy_env("DRPO_PARALLEL_REWARD_SELECTORS") and len(selector_items) > 1:
        max_workers = min(len(selector_items), max(1, int(os.getenv("DRPO_PARALLEL_REWARD_WORKERS", str(len(selector_items))))))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            raw_by_model = {
                model_name: (scores, elapsed)
                for model_name, scores, elapsed in pool.map(
                    lambda item: _score_grouped_selector(
                        item[0],
                        item[1],
                        images,
                        prompts,
                        batch_size=batch_size,
                        image_paths=image_paths,
                        image_tensor=image_tensor,
                    ),
                    selector_items,
                )
            }
    else:
        raw_by_model = {
            model_name: (scores, elapsed)
            for model_name, scores, elapsed in (
                _score_grouped_selector(
                    model_name,
                    selector,
                    images,
                    prompts,
                    batch_size=batch_size,
                    image_paths=image_paths,
                    image_tensor=image_tensor,
                )
                for model_name, selector in selector_items
            )
        }
    for model_name, weight in zip(selectors.keys(), weights):
        raw_scores, elapsed = raw_by_model[model_name]
        raw = torch.tensor(raw_scores, dtype=torch.float32).reshape(num_groups, group_size)
        normed = _normalize_group_rewards(raw, normalize)
        weighted_terms.append(float(weight) * normed)
        safe_name = model_name.replace("/", "_").replace("-", "_")
        info[f"online_reward_{safe_name}_raw_mean"] = raw.mean(dim=1).mean().detach()
        info[f"online_reward_{safe_name}_raw_std"] = raw.std(dim=1, unbiased=False).mean().detach()
        info[f"online_reward_{safe_name}_norm_mean"] = normed.mean(dim=1).mean().detach()
        info[f"online_reward_{safe_name}_norm_std"] = normed.std(dim=1, unbiased=False).mean().detach()
        info[f"online_reward_{safe_name}_weight"] = raw.new_tensor(float(weight))
        info[f"online_reward_{safe_name}_score_seconds"] = raw.new_tensor(float(elapsed))
    ensemble = torch.stack(weighted_terms, dim=0).sum(dim=0)
    info["online_reward_ensemble_mean"] = ensemble.mean(dim=1).mean().detach()
    info["online_reward_ensemble_std"] = ensemble.std(dim=1, unbiased=False).mean().detach()
    info["online_reward_ensemble_max"] = ensemble.max(dim=1).values.mean().detach()
    info["online_reward_ensemble_min"] = ensemble.min(dim=1).values.mean().detach()
    return ensemble, info


def _cache_and_score_rewards(
    *,
    config: SDXLGRPOConfig,
    selectors,
    image_tensor: torch.Tensor,
    images: Sequence[Image.Image] | None,
    prompts: Sequence[str],
    global_step: int,
    process_index: int,
    call_index: int,
    enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image_paths: list[str] | None = None
    records: list[dict[str, object]] = []
    if enabled:
        if config.reward_cache_dir is None:
            raise ValueError("reward_cache_dir is required when reward_cache_images is enabled.")
        if images is None:
            images = _tensor_to_pil(image_tensor)
        step_dir = (
            Path(config.reward_cache_dir)
            / f"step-{int(global_step):06d}"
            / f"rank-{int(process_index):03d}"
            / f"call-{int(call_index):06d}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for flat_index, (image, prompt) in enumerate(zip(images, prompts)):
            local_prompt_index = flat_index // config.batchsize_gen
            candidate_index = flat_index % config.batchsize_gen
            image_path = step_dir / f"p{local_prompt_index:04d}_g{candidate_index:03d}.png"
            image.save(image_path)
            image_paths.append(str(image_path))
            records.append(
                {
                    "step": int(global_step),
                    "rank": int(process_index),
                    "call_index": int(call_index),
                    "local_prompt_index": int(local_prompt_index),
                    "candidate_index": int(candidate_index),
                    "group_size": int(config.batchsize_gen),
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "width": int(image.width),
                    "height": int(image.height),
                    "choice_models": list(config.choice_models),
                    "choice_model_weights": [float(weight) for weight in config.choice_model_weights],
                }
            )

    scores, reward_info = _score_reward_ensemble_grouped(
        selectors,
        config.choice_model_weights,
        images,
        prompts,
        group_size=config.batchsize_gen,
        batch_size=config.reward_score_batch_size,
        normalize=config.choice_score_normalize,
        image_paths=image_paths,
        image_tensor=image_tensor,
    )

    if enabled and records:
        manifest_path = Path(records[0]["image_path"]).parent / "manifest.jsonl"
        tmp_path = manifest_path.with_suffix(".jsonl.tmp")
        flat_scores = scores.flatten().tolist()
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for record, score in zip(records, flat_scores):
                row = dict(record)
                row["score"] = float(score)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, manifest_path)

    return scores, reward_info


def _batch_terms(
    config: SDXLGRPOConfig,
    pipe: StableDiffusionXLPipeline,
    reference_unet,
    selectors,
    latent_channels: int,
    latent_size: int,
    prompts: Sequence[str],
    device: torch.device,
    *,
    global_step: int,
    process_index: int,
    reward_cache_call_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prompt_list = list(prompts)
    batch_size = len(prompt_list)
    if batch_size < 1:
        raise ValueError("Expected at least one prompt.")
    group_size = config.batchsize_gen
    with torch.no_grad():
        prompt_embeds_once, pooled_prompt_embeds_once = _encode_prompts(pipe, prompt_list, device)
    prompt_embeds = prompt_embeds_once.repeat_interleave(group_size, dim=0)
    pooled_prompt_embeds = pooled_prompt_embeds_once.repeat_interleave(group_size, dim=0)

    base_noise = torch.randn((batch_size, latent_channels, latent_size, latent_size), device=device, dtype=prompt_embeds.dtype)
    deltas = torch.randn((batch_size, group_size, latent_channels, latent_size, latent_size), device=device, dtype=prompt_embeds.dtype)
    validate_neighbor_sigma(float(config.neighbor_sigma))
    base_scale = math.sqrt(max(0.0, 1.0 - float(config.neighbor_sigma) ** 2))
    initial_latents = base_scale * base_noise.unsqueeze(1) + float(config.neighbor_sigma) * deltas
    flat_initial_latents = initial_latents.reshape(batch_size * group_size, latent_channels, latent_size, latent_size)

    anchor_indices = torch.stack(
        [torch.randperm(group_size, device=device)[: config.neighbor_num_anchors] for _ in range(batch_size)],
        dim=0,
    )
    anchor_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * group_size
    flat_anchor_indices = (anchor_offsets + anchor_indices).reshape(-1)
    anchor_latents = flat_initial_latents.index_select(0, flat_anchor_indices)
    anchor_prompt_embeds = prompt_embeds.index_select(0, flat_anchor_indices)
    anchor_pooled_prompt_embeds = pooled_prompt_embeds.index_select(0, flat_anchor_indices)

    with torch.no_grad():
        candidate_old_flat = _sdxl_one_step_latents(
            pipe=pipe,
            unet=pipe.unet,
            latents=flat_initial_latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            resolution=config.resolution,
            num_inference_steps=config.num_inference_steps,
        ).detach()
        reference_anchors = _sdxl_one_step_latents(
            pipe=pipe,
            unet=reference_unet,
            latents=anchor_latents,
            prompt_embeds=anchor_prompt_embeds,
            pooled_prompt_embeds=anchor_pooled_prompt_embeds,
            resolution=config.resolution,
            num_inference_steps=config.num_inference_steps,
        ).detach()
        reward_tensor = _decode_latents_to_tensor(pipe.vae, candidate_old_flat, chunk_size=config.vae_decode_chunk_size)
        flat_reward_prompts = [prompt for prompt in prompt_list for _ in range(group_size)]
        cache_enabled = (
            config.reward_cache_images
            and config.reward_cache_interval > 0
            and int(global_step) % int(config.reward_cache_interval) == 0
        )
        scores, reward_info = _cache_and_score_rewards(
            config=config,
            selectors=selectors,
            image_tensor=reward_tensor,
            images=None,
            prompts=flat_reward_prompts,
            global_step=global_step,
            process_index=process_index,
            call_index=reward_cache_call_index,
            enabled=cache_enabled,
        )
    scores = scores.to(device=device, dtype=torch.float32)
    advantages = torch.stack(
        [
            quasi_norm_advantages(group_scores, p=config.neighbor_p_norm, scale=config.advantage_scale, clip=config.advantage_clip)
            for group_scores in scores
        ],
        dim=0,
    ).detach()

    current_anchors = _sdxl_one_step_latents(
        pipe=pipe,
        unet=pipe.unet,
        latents=anchor_latents,
        prompt_embeds=anchor_prompt_embeds,
        pooled_prompt_embeds=anchor_pooled_prompt_embeds,
        resolution=config.resolution,
        num_inference_steps=config.num_inference_steps,
    )

    candidate_old = candidate_old_flat.reshape(batch_size, group_size, latent_channels, latent_size, latent_size)
    current_anchors = current_anchors.reshape(batch_size, config.neighbor_num_anchors, latent_channels, latent_size, latent_size)
    reference_anchors = reference_anchors.reshape(batch_size, config.neighbor_num_anchors, latent_channels, latent_size, latent_size)
    batch_index = torch.arange(batch_size, device=device).unsqueeze(1)
    old_anchors = candidate_old[batch_index, anchor_indices]

    policy_loss, stats = batched_neighbor_grpo_loss(
        candidate_old,
        current_anchors,
        old_anchors,
        advantages,
        clip_range=config.ppo_clip_range,
        max_log_ratio=config.max_log_ratio,
        temperature=config.neighbor_distance_temperature,
        reduction=config.neighbor_distance_reduction,
    )
    ppo_kl_loss = stats["approx_kl_loss"]
    ref_l2 = F.mse_loss(current_anchors.float(), reference_anchors.float())
    loss = policy_loss + config.policy_kl_weight * ppo_kl_loss + config.ref_model_l2_weight * ref_l2
    logs = {
        "policy_loss": policy_loss.detach(),
        "reward_mean": scores.mean().detach(),
        "reward_std": scores.std(unbiased=False).detach(),
        "adv_abs": advantages.abs().mean().detach(),
        "ratio_mean": stats["ratio_mean"].detach(),
        "clipfrac": stats["clipfrac"].detach(),
        "ppo_kl": stats["approx_kl"].detach(),
        "ref_l2": ref_l2.detach(),
        "surrogate_entropy": stats["entropy"].detach(),
    }
    for key, value in reward_info.items():
        logs[key] = value.to(device=device).detach()
    return loss, logs


def train(config: SDXLGRPOConfig) -> None:
    if config.reward_cache_dir is None:
        config = SDXLGRPOConfig(**{**asdict(config), "reward_cache_dir": str(Path(config.output_dir) / "reward_cache")})
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = create_accelerator(config, output_dir)
    _setup_logging(config, accelerator)
    if config.seed is not None:
        set_seed(config.seed + accelerator.process_index)
    _save_runtime_snapshot(config, accelerator)

    model_path = str(require_local_path(config.pretrained_model_name_or_path, description="SDXL Turbo model", must_be_file=False))
    dtype = _dtype_for(config)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype, variant=config.model_variant, local_files_only=True)
    pipe.scheduler.set_timesteps(config.num_inference_steps)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    reference_unet = copy.deepcopy(pipe.unet).eval().requires_grad_(False)
    pipe.unet = _make_trainable(pipe.unet, config)
    if hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()
    maybe_enable_xformers(pipe.unet, logger)

    choice_models = resolve_choice_models(config.choice_models, config.choice_model)
    choice_model_weights = resolve_choice_model_weights(config.choice_model_weights, choice_models)
    config = SDXLGRPOConfig(**{**asdict(config), "choice_models": choice_models, "choice_model_weights": choice_model_weights})

    dataset = PromptDataset(config.prompt_file, pipe.tokenizer, max_samples=config.max_train_samples, seed=config.seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_preference_batch,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.dataloader_num_workers > 0,
    )
    selectors = build_choice_selectors(
        choice_models,
        accelerator.device,
        pickscore_model_path=config.pickscore_model_name_or_path,
        pickscore_processor_path=config.pickscore_processor_name_or_path or config.pickscore_model_name_or_path,
        local_files_only=not config.pickscore_allow_remote,
    )
    optimizer = torch.optim.AdamW(
        _trainable_parameters(pipe.unet),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    pipe.unet, reference_unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet,
        reference_unet,
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    pipe.unet.eval()
    reference_unet.eval()
    latent_channels = int(accelerator.unwrap_model(pipe.unet).config.in_channels)
    latent_size = config.resolution // int(getattr(pipe, "vae_scale_factor", 8) or 8)

    if accelerator.is_main_process:
        tracker_config = {key: json.dumps(value) if isinstance(value, (list, tuple)) else value for key, value in asdict(config).items()}
        accelerator.init_trackers("sdxl-turbo-grpo", tracker_config)

    step = resume_global_step(config.resume_from_checkpoint)
    reward_cache_call_index = 0
    progress = tqdm(total=config.max_train_steps, initial=step, disable=not accelerator.is_local_main_process)
    callbacks = TrainingStepCallbacks(
        accelerator=accelerator,
        progress=progress,
        config=config,
        output_dir=output_dir,
        unet=pipe.unet,
        save_checkpoint=_save_checkpoint,
    )
    while step < config.max_train_steps:
        for batch in dataloader:
            if step >= config.max_train_steps:
                break
            assert isinstance(batch, Batch)
            with accelerator.accumulate(pipe.unet):
                loss, logs = _batch_terms(
                    config,
                    pipe,
                    reference_unet,
                    selectors,
                    latent_channels,
                    latent_size,
                    batch.prompts,
                    accelerator.device,
                    global_step=step,
                    process_index=accelerator.process_index,
                    reward_cache_call_index=reward_cache_call_index,
                )
                reward_cache_call_index += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(_trainable_parameters(pipe.unet), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            log_values = {key: value.detach().float() for key, value in logs.items()}
            step, should_stop = callbacks.finish_step(
                step=step,
                loss=loss,
                lr=lr_scheduler.get_last_lr()[0],
                log_values=log_values,
            )
            if should_stop:
                break
    callbacks.finish_training(step)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train SDXL Turbo with fused neighbor-based GRPO.")
    parser.add_argument("--pretrained_model_name_or_path", default=str(root / "models" / "stable-diffusion-xl-turbo"))
    parser.add_argument("--output_dir", default=str(root / "outputs" / "sdxl-turbo-lora" / "grpo" / "pickscore" / "lr1e-5_bs24_ga4_steps5000"))
    parser.add_argument("--prompt_file", default=str(root / "data" / "pickscore" / "train.txt"))
    parser.add_argument("--model_variant", default="fp16")
    parser.add_argument("--choice_model", default="pickscore")
    parser.add_argument("--choice_models", default="")
    parser.add_argument("--choice_model_weights", default="")
    parser.add_argument("--choice_score_normalize", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--pickscore_model_name_or_path", default=str(root / "models" / "PickScore_v1"))
    parser.add_argument("--pickscore_processor_name_or_path", default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--batchsize_gen", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--neighbor_sigma", type=float, default=0.3)
    parser.add_argument("--neighbor_num_anchors", type=int, default=4)
    parser.add_argument("--neighbor_p_norm", type=float, default=0.8)
    parser.add_argument("--neighbor_distance_temperature", type=float, default=1.0)
    parser.add_argument("--neighbor_distance_reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2)
    parser.add_argument("--max_log_ratio", type=float, default=10.0)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--advantage_scale", type=float, default=1.0)
    parser.add_argument("--policy_kl_weight", type=float, default=0.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.02)
    parser.add_argument("--vae_decode_chunk_size", type=int, default=4)
    parser.add_argument("--reward_score_batch_size", type=int, default=128)
    parser.add_argument("--reward_cache_dir", default=None)
    parser.add_argument("--reward_cache_interval", type=int, default=1)
    parser.add_argument("--reward_cache_images", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_config(argv: list[str] | None = None) -> SDXLGRPOConfig:
    args = build_parser().parse_args(argv)
    reward_cache_dir = args.reward_cache_dir or str(Path(args.output_dir) / "reward_cache")
    return SDXLGRPOConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        model_variant=args.model_variant or None,
        choice_model=args.choice_model,
        choice_models=_parse_names(args.choice_models),
        choice_model_weights=_parse_floats(args.choice_model_weights) if args.choice_model_weights else (),
        choice_score_normalize=args.choice_score_normalize,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        pickscore_processor_name_or_path=args.pickscore_processor_name_or_path,
        pickscore_allow_remote=args.pickscore_allow_remote,
        seed=args.seed,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        batchsize_gen=args.batchsize_gen,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        max_train_samples=args.max_train_samples,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        checkpointing_steps=args.checkpointing_steps,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        use_lora=args.use_lora,
        resume_from_checkpoint=args.resume_from_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_parse_names(args.lora_target_modules),
        neighbor_sigma=args.neighbor_sigma,
        neighbor_num_anchors=args.neighbor_num_anchors,
        neighbor_p_norm=args.neighbor_p_norm,
        neighbor_distance_temperature=args.neighbor_distance_temperature,
        neighbor_distance_reduction=args.neighbor_distance_reduction,
        ppo_clip_range=args.ppo_clip_range,
        max_log_ratio=args.max_log_ratio,
        advantage_clip=args.advantage_clip,
        advantage_scale=args.advantage_scale,
        policy_kl_weight=args.policy_kl_weight,
        ref_model_l2_weight=args.ref_model_l2_weight,
        vae_decode_chunk_size=args.vae_decode_chunk_size,
        reward_score_batch_size=args.reward_score_batch_size,
        reward_cache_dir=reward_cache_dir,
        reward_cache_interval=args.reward_cache_interval,
        reward_cache_images=args.reward_cache_images,
    )


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
