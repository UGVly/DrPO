from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shlex
import socket
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

from drpo.data import Batch, PromptDataset, collate_preference_batch
from drpo.methods.sdxl_drpo.trainer import (
    _decode_latents_to_tensor,
    _encode_prompts,
    _maybe_enable_xformers,
    _sdxl_one_step_latents,
    _tensor_to_pil,
)
from drpo.paths import project_root, require_local_path
from drpo.rewards import build_choice_selectors, resolve_choice_model_weights, resolve_choice_models, score_reward_ensemble

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
    gradient_accumulation_steps: int = 4
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
    vae_decode_chunk_size: int = 1


def _parse_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _dtype_for(config: SDXLGRPOConfig) -> torch.dtype:
    if config.mixed_precision == "bf16":
        return torch.bfloat16
    if config.mixed_precision == "fp16":
        return torch.float16
    return torch.float32


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
    if not config.use_lora:
        unet.requires_grad_(True)
        return unet
    if config.resume_from_checkpoint:
        return PeftModel.from_pretrained(unet, str(_resolve_unet_lora_dir(config.resume_from_checkpoint)), is_trainable=True)
    unet.requires_grad_(False)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )
    return get_peft_model(unet, lora_config)


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _save_checkpoint(accelerator: Accelerator, unet, config: SDXLGRPOConfig, checkpoint_dir: Path, global_step: int, checkpoint_type: str) -> None:
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        if config.use_lora:
            unwrapped.save_pretrained(checkpoint_dir / "unet_lora")
        else:
            unwrapped.save_pretrained(checkpoint_dir / "unet")
        metadata = {
            "checkpoint_type": checkpoint_type,
            "global_step": global_step,
            "created_at": datetime.now().isoformat(),
            "model_type": "sdxl-turbo-grpo-neighbor",
            "contains_accelerate_state": False,
            "contains_model_state": True,
            "model_variant": config.model_variant,
            "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        }
        (checkpoint_dir / "training_state.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    accelerator.wait_for_everyone()


def _setup_logging(config: SDXLGRPOConfig, accelerator: Accelerator) -> None:
    log_dir = Path(config.output_dir) / config.logging_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(log_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=f"%(asctime)s | %(levelname)s | rank={accelerator.process_index}/{accelerator.num_processes} | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _save_runtime_snapshot(config: SDXLGRPOConfig, accelerator: Accelerator) -> None:
    if not accelerator.is_main_process:
        return
    out = Path(config.output_dir) / "run_metadata"
    out.mkdir(parents=True, exist_ok=True)
    safe_config = asdict(config)
    (out / "resolved_config.json").write_text(json.dumps(safe_config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    runtime = {
        "created_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "world_size": accelerator.num_processes,
        "device": str(accelerator.device),
        "torch_version": torch.__version__,
    }
    (out / "runtime_info.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    (out / "launch_command.sh").write_text("#!/usr/bin/env bash\n" + shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")


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


def _prompt_terms(
    config: SDXLGRPOConfig,
    pipe: StableDiffusionXLPipeline,
    reference_unet,
    selectors,
    latent_channels: int,
    latent_size: int,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prompts = [prompt] * config.batchsize_gen
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = _encode_prompts(pipe, prompts, device)
    base_noise = torch.randn((latent_channels, latent_size, latent_size), device=device, dtype=prompt_embeds.dtype)
    deltas = torch.randn((config.batchsize_gen, latent_channels, latent_size, latent_size), device=device, dtype=prompt_embeds.dtype)
    initial_latents = construct_neighbor_noises(base_noise, deltas, config.neighbor_sigma).to(dtype=prompt_embeds.dtype)

    with torch.no_grad():
        candidate_old = _sdxl_one_step_latents(
            pipe=pipe,
            unet=pipe.unet,
            latents=initial_latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            resolution=config.resolution,
            num_inference_steps=config.num_inference_steps,
        ).detach()
        reference_latents = _sdxl_one_step_latents(
            pipe=pipe,
            unet=reference_unet,
            latents=initial_latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            resolution=config.resolution,
            num_inference_steps=config.num_inference_steps,
        ).detach()
        reward_tensor = _decode_latents_to_tensor(pipe.vae, candidate_old, chunk_size=config.vae_decode_chunk_size)
        reward_images = _tensor_to_pil(reward_tensor)
        scores, reward_info = score_reward_ensemble(selectors, config.choice_model_weights, reward_images, prompt, normalize=config.choice_score_normalize)
    scores = scores.to(device=device, dtype=torch.float32)
    advantages = quasi_norm_advantages(scores, p=config.neighbor_p_norm, scale=config.advantage_scale, clip=config.advantage_clip).detach()

    anchor_indices = torch.randperm(config.batchsize_gen, device=device)[: config.neighbor_num_anchors]
    anchor_latents = initial_latents.index_select(0, anchor_indices)
    anchor_prompt_embeds = prompt_embeds.index_select(0, anchor_indices)
    anchor_pooled_prompt_embeds = pooled_prompt_embeds.index_select(0, anchor_indices)
    current_anchors = _sdxl_one_step_latents(
        pipe=pipe,
        unet=pipe.unet,
        latents=anchor_latents,
        prompt_embeds=anchor_prompt_embeds,
        pooled_prompt_embeds=anchor_pooled_prompt_embeds,
        resolution=config.resolution,
        num_inference_steps=config.num_inference_steps,
    )

    policy_loss_sum = 0.0
    ratio_sum = 0.0
    clipfrac_sum = 0.0
    kl_sum = 0.0
    kl_loss_sum = 0.0
    entropy_sum = 0.0
    ref_l2_sum = 0.0
    for anchor_pos, anchor_index in enumerate(anchor_indices.tolist()):
        policy_loss_anchor, stats = neighbor_grpo_loss(
            candidate_old,
            current_anchors[anchor_pos],
            candidate_old[anchor_index],
            advantages,
            clip_range=config.ppo_clip_range,
            max_log_ratio=config.max_log_ratio,
            temperature=config.neighbor_distance_temperature,
            reduction=config.neighbor_distance_reduction,
        )
        ref_l2_anchor = F.mse_loss(current_anchors[anchor_pos].float(), reference_latents[anchor_index].float())
        policy_loss_sum = policy_loss_sum + policy_loss_anchor
        ratio_sum = ratio_sum + stats["ratio_mean"]
        clipfrac_sum = clipfrac_sum + stats["clipfrac"]
        kl_sum = kl_sum + stats["approx_kl"]
        kl_loss_sum = kl_loss_sum + stats["approx_kl_loss"]
        entropy_sum = entropy_sum + stats["entropy"]
        ref_l2_sum = ref_l2_sum + ref_l2_anchor

    policy_loss = policy_loss_sum / config.neighbor_num_anchors
    ppo_kl_loss = kl_loss_sum / config.neighbor_num_anchors
    ref_l2 = ref_l2_sum / config.neighbor_num_anchors
    loss = policy_loss + config.policy_kl_weight * ppo_kl_loss + config.ref_model_l2_weight * ref_l2
    logs = {
        "policy_loss": policy_loss.detach(),
        "reward_mean": scores.mean().detach(),
        "reward_std": scores.std(unbiased=False).detach(),
        "adv_abs": advantages.abs().mean().detach(),
        "ratio_mean": (ratio_sum / config.neighbor_num_anchors).detach(),
        "clipfrac": (clipfrac_sum / config.neighbor_num_anchors).detach(),
        "ppo_kl": (kl_sum / config.neighbor_num_anchors).detach(),
        "ref_l2": ref_l2.detach(),
        "surrogate_entropy": (entropy_sum / config.neighbor_num_anchors).detach(),
    }
    for key, value in reward_info.items():
        logs[key] = value.to(device=device).detach()
    return loss, logs


def train(config: SDXLGRPOConfig) -> None:
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
        log_with=config.report_to,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / config.logging_dir)),
    )
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
    _maybe_enable_xformers(pipe.unet)

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

    step = _resume_global_step(config.resume_from_checkpoint)
    progress = tqdm(total=config.max_train_steps, initial=step, disable=not accelerator.is_local_main_process)
    while step < config.max_train_steps:
        for batch in dataloader:
            if step >= config.max_train_steps:
                break
            assert isinstance(batch, Batch)
            with accelerator.accumulate(pipe.unet):
                sample_losses: list[torch.Tensor] = []
                logs: dict[str, list[torch.Tensor]] = {}
                for prompt in batch.prompts:
                    loss_i, logs_i = _prompt_terms(config, pipe, reference_unet, selectors, latent_channels, latent_size, prompt, accelerator.device)
                    sample_losses.append(loss_i)
                    for key, value in logs_i.items():
                        logs.setdefault(key, []).append(value.detach().float())
                loss = torch.stack(sample_losses).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(_trainable_parameters(pipe.unet), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                log_values = {"train_loss": loss.detach(), "lr": torch.tensor(lr_scheduler.get_last_lr()[0], device=accelerator.device)}
                for key, values in logs.items():
                    if values:
                        log_values[key] = torch.stack(values).mean()
                accelerator.log({key: float(value.detach().cpu()) for key, value in log_values.items()}, step=step)
                progress.update(1)
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
                if config.checkpointing_steps > 0 and step % config.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, pipe.unet, config, output_dir / f"checkpoint-{step}", step, "intermediate")
                if step >= config.max_train_steps:
                    break
    _save_checkpoint(accelerator, pipe.unet, config, output_dir / "final", step, "final")
    accelerator.end_training()


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train SDXL Turbo with fused neighbor-based GRPO.")
    parser.add_argument("--pretrained_model_name_or_path", default=str(root / "models" / "stable-diffusion-xl-turbo"))
    parser.add_argument("--output_dir", default=str(root / "outputs" / "sdxl-turbo-lora" / "grpo" / "pickscore" / "lr1e-5_bs24_ga4_steps5000"))
    parser.add_argument("--prompt_file", default=str(root / "data" / "prompts" / "pickapicv2_test_unique.txt"))
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
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
    parser.add_argument("--vae_decode_chunk_size", type=int, default=1)
    return parser


def parse_config(argv: list[str] | None = None) -> SDXLGRPOConfig:
    args = build_parser().parse_args(argv)
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
    )


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
