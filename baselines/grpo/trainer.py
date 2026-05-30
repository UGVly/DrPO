#!/usr/bin/env python
# coding=utf-8

"""Neighbor-based one-step SD-Turbo GRPO trainer.

This is the fused GRPO baseline: the historical plain GRPO and Neighbor-GRPO
paths are represented as one algorithm, with Neighbor-GRPO as the core update.
The implementation is self-contained under ``baselines`` and does not import
project-internal training modules.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import (  # noqa: E402
    PairsPromptDataset,
    build_tracker_config,
    collate_fn,
    compute_grad_norm,
    decode_latents_to_pil,
    enforce_zero_terminal_snr,
    evaluate_fixed_prompts,
    load_prompt_file,
    one_step_clean_latent,
    resume_training_state,
    cache_and_score_pil_rewards,
    save_runtime_snapshot,
    save_training_checkpoint,
    setup_logging,
)
from baselines.grpo.losses import (  # noqa: E402
    batched_neighbor_grpo_loss,
    construct_neighbor_noises,
    neighbor_grpo_loss,
    quasi_norm_advantages,
)

check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


def pred_to_onestep_latent(noisy_latents: torch.Tensor, model_pred: torch.Tensor) -> torch.Tensor:
    return one_step_clean_latent(noisy_latents, model_pred)



def build_reward_selector(args, device: torch.device):
    if args.choice_model == "pickscore":
        from utils.pickscore_utils import Selector

        return Selector(
            device,
            model_name_or_path=args.pickscore_model_name_or_path,
            processor_name_or_path=args.pickscore_processor_name_or_path or args.pickscore_model_name_or_path,
            local_files_only=not args.pickscore_allow_remote,
        )
    if args.choice_model == "clip":
        from utils.clip_utils import Selector

        return Selector(
            device,
            model_name_or_path=args.clip_model_name_or_path,
            local_files_only=not args.clip_allow_remote,
        )
    if args.choice_model == "aes":
        from utils.aes_utils import Selector

        return Selector(
            device,
            clip_model_path=args.aesthetic_clip_model_path,
            ckpt_path=args.aesthetic_ckpt_path,
            local_files_only=not args.aesthetic_allow_remote,
        )
    if args.choice_model in {"hps", "hpsv2"}:
        from utils.hps_utils import Selector

        return Selector(
            device,
            open_clip_pretrained_path=args.hps_open_clip_pretrained_path,
            checkpoint_path=args.hps_checkpoint_path,
        )
    raise ValueError(f"Unsupported choice_model: {args.choice_model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SD-Turbo neighbor-based one-step GRPO trainer")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "sd-turbo"))
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "grpo", "pickscore", "default"))
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs_jsonl",
        "--train_prompt_file",
        dest="pairs_jsonl",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "pickscore", "train.txt"),
        help="Prompt txt for online mode, or preference-pair JSONL for offline modes.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--batchsize_gen", type=int, default=24)
    parser.add_argument("--generation_timestep", type=int, default=999)
    parser.add_argument("--generation_target_timestep", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--choice_model", type=str, default="pickscore", choices=["pickscore", "clip", "aes", "hps", "hpsv2"])
    parser.add_argument("--pickscore_model_name_or_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "PickScore_v1"))
    parser.add_argument("--pickscore_processor_name_or_path", type=str, default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")
    parser.add_argument("--clip_model_name_or_path", type=str, default=None)
    parser.add_argument("--clip_allow_remote", action="store_true")
    parser.add_argument("--aesthetic_clip_model_path", type=str, default=None)
    parser.add_argument("--aesthetic_ckpt_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "Aesthetic", "sac+logos+ava1-l14-linearMSE.pth"))
    parser.add_argument("--aesthetic_allow_remote", action="store_true")
    parser.add_argument("--hps_open_clip_pretrained_path", type=str, default=None)
    parser.add_argument("--hps_checkpoint_path", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--neighbor_sigma", type=float, default=0.3)
    parser.add_argument("--neighbor_num_anchors", type=int, default=4)
    parser.add_argument("--neighbor_p_norm", type=float, default=0.8)
    parser.add_argument("--neighbor_distance_temperature", type=float, default=1.0)
    parser.add_argument("--neighbor_distance_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--ppo_clip_range", type=float, default=0.2)
    parser.add_argument("--max_log_ratio", type=float, default=10.0)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--advantage_scale", type=float, default=1.0)
    parser.add_argument("--policy_kl_weight", type=float, default=0.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.02)
    parser.add_argument("--vae_decode_chunk_size", type=int, default=4)
    parser.add_argument("--reward_score_batch_size", type=int, default=128)
    parser.add_argument("--reward_cache_dir", type=str, default=None)
    parser.add_argument("--reward_cache_interval", type=int, default=1)
    parser.add_argument("--reward_cache_images", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eval_prompt_file", type=str, default=os.path.join(PROJECT_ROOT, "data", "pickscore", "test.txt"))
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--eval_output_subdir", type=str, default="eval_samples")
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if args.pickscore_processor_name_or_path is None:
        args.pickscore_processor_name_or_path = args.pickscore_model_name_or_path
    if args.batchsize_gen < 2:
        raise ValueError("--batchsize_gen must be >= 2 for group-relative advantages.")
    if args.ppo_clip_range <= 0:
        raise ValueError("--ppo_clip_range must be > 0.")
    if args.generation_timestep < 0:
        raise ValueError("--generation_timestep must be >= 0.")
    if args.generation_target_timestep >= 0 and args.generation_target_timestep > args.generation_timestep:
        raise ValueError("--generation_target_timestep must be <= --generation_timestep or -1.")
    if not (0.0 < args.neighbor_sigma <= 1.0):
        raise ValueError("--neighbor_sigma must satisfy 0 < sigma <= 1.")
    if args.neighbor_num_anchors < 1 or args.neighbor_num_anchors > args.batchsize_gen:
        raise ValueError("--neighbor_num_anchors must be in [1, batchsize_gen].")
    if not (0.0 < args.neighbor_p_norm <= 2.0):
        raise ValueError("--neighbor_p_norm must satisfy 0 < p <= 2.")
    if args.neighbor_distance_temperature <= 0:
        raise ValueError("--neighbor_distance_temperature must be > 0.")
    if args.vae_decode_chunk_size < 1:
        raise ValueError("--vae_decode_chunk_size must be >= 1.")
    if args.reward_score_batch_size < 1:
        raise ValueError("--reward_score_batch_size must be >= 1.")
    if args.reward_cache_interval < 0:
        raise ValueError("--reward_cache_interval must be >= 0.")
    if args.reward_cache_dir is None:
        args.reward_cache_dir = os.path.join(args.output_dir, "reward_cache")
    return args


def _make_lora_unet(unet, args):
    if not args.use_lora:
        return unet
    unet.requires_grad_(False)
    lora_target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    if not lora_target_modules:
        raise ValueError("Expected at least one LoRA target module.")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=lora_target_modules,
    )
    return get_peft_model(unet, lora_config)


def _maybe_enable_xformers(unet) -> None:
    if not is_xformers_available():
        logger.warning("xformers is not available; continuing without memory efficient attention.")
        return
    import xformers

    if version.parse(xformers.__version__) < version.parse("0.0.17"):
        return
    try:
        unet.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory efficient attention.")
    except Exception as exc:
        logger.warning(f"Failed to enable xformers memory efficient attention: {exc}")


def _group_prompt_embeddings(text_encoder, prompt_ids: torch.Tensor, group_size: int) -> torch.Tensor:
    with torch.no_grad():
        encoder_hidden_states_one = text_encoder(prompt_ids)[0]
    return encoder_hidden_states_one.repeat_interleave(group_size, dim=0)



def _neighbor_based_grpo_terms(args, unet, ref_unet, vae, text_encoder, prompt: str, prompt_ids: torch.Tensor, weight_dtype: torch.dtype, device: torch.device):
    with torch.no_grad():
        encoder_hidden_states_one = text_encoder(prompt_ids)[0]
    encoder_hidden_states = encoder_hidden_states_one.repeat_interleave(args.batchsize_gen, dim=0)
    base_noise = torch.randn((4, args.resolution // 8, args.resolution // 8), device=device, dtype=weight_dtype)
    deltas = torch.randn((args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8), device=device, dtype=weight_dtype)
    noisy_latents = construct_neighbor_noises(base_noise, deltas, args.neighbor_sigma).to(dtype=weight_dtype)
    timesteps = torch.full((args.batchsize_gen,), args.generation_timestep, device=device, dtype=torch.long)
    anchor_indices = torch.randperm(args.batchsize_gen, device=device)[: args.neighbor_num_anchors]
    anchor_noisy = noisy_latents.index_select(0, anchor_indices)
    anchor_timesteps = timesteps.index_select(0, anchor_indices)
    anchor_encoder_hidden_states = encoder_hidden_states_one.repeat_interleave(args.neighbor_num_anchors, dim=0)

    with torch.no_grad():
        rollout_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        candidate_x0_old = pred_to_onestep_latent(noisy_latents, rollout_pred).detach()
        ref_pred = ref_unet(anchor_noisy, anchor_timesteps, anchor_encoder_hidden_states).sample
        ref_anchor_x0 = pred_to_onestep_latent(anchor_noisy, ref_pred).detach()

    sampled_images = decode_latents_to_pil(vae, candidate_x0_old, chunk_size=args.vae_decode_chunk_size)
    rewards = torch.tensor(args.reward_selector.score(sampled_images, prompt), device=device, dtype=torch.float32)
    advantages = quasi_norm_advantages(
        rewards,
        p=args.neighbor_p_norm,
        scale=args.advantage_scale,
        clip=args.advantage_clip,
    ).detach()

    current_pred = unet(anchor_noisy, anchor_timesteps, anchor_encoder_hidden_states).sample
    current_anchor_x0 = pred_to_onestep_latent(anchor_noisy, current_pred)

    policy_loss_sum = 0.0
    ratio_sum = 0.0
    clipfrac_sum = 0.0
    kl_sum = 0.0
    kl_loss_sum = 0.0
    entropy_sum = 0.0
    ref_l2_sum = 0.0
    for anchor_pos, anchor_index in enumerate(anchor_indices.tolist()):
        policy_loss_anchor, stats = neighbor_grpo_loss(
            candidate_x0_old,
            current_anchor_x0[anchor_pos],
            candidate_x0_old[anchor_index],
            advantages,
            clip_range=args.ppo_clip_range,
            max_log_ratio=args.max_log_ratio,
            temperature=args.neighbor_distance_temperature,
            reduction=args.neighbor_distance_reduction,
        )
        ref_l2_anchor = F.mse_loss(current_anchor_x0[anchor_pos].float(), ref_anchor_x0[anchor_pos].float())
        policy_loss_sum = policy_loss_sum + policy_loss_anchor
        ratio_sum = ratio_sum + stats["ratio_mean"]
        clipfrac_sum = clipfrac_sum + stats["clipfrac"]
        kl_sum = kl_sum + stats["approx_kl"]
        kl_loss_sum = kl_loss_sum + stats["approx_kl_loss"]
        entropy_sum = entropy_sum + stats["entropy"]
        ref_l2_sum = ref_l2_sum + ref_l2_anchor

    policy_loss = policy_loss_sum / args.neighbor_num_anchors
    ppo_kl_loss = kl_loss_sum / args.neighbor_num_anchors
    ref_l2 = ref_l2_sum / args.neighbor_num_anchors
    loss = policy_loss + args.ref_model_l2_weight * ref_l2 + args.policy_kl_weight * ppo_kl_loss
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_std": rewards.std(unbiased=False).detach(),
        "adv_abs": advantages.abs().mean().detach(),
        "ratio_mean": (ratio_sum / args.neighbor_num_anchors).detach(),
        "clipfrac": (clipfrac_sum / args.neighbor_num_anchors).detach(),
        "ppo_kl": (kl_sum / args.neighbor_num_anchors).detach(),
        "ref_l2": ref_l2.detach(),
        "entropy": (entropy_sum / args.neighbor_num_anchors).detach(),
    }


def _expand_group_embeddings(encoder_hidden_states_one: torch.Tensor, group_size: int) -> torch.Tensor:
    return encoder_hidden_states_one.repeat_interleave(group_size, dim=0)


def _neighbor_based_grpo_batch_terms(
    args,
    unet,
    ref_unet,
    vae,
    text_encoder,
    prompts: Sequence[str],
    prompt_ids: torch.Tensor,
    weight_dtype: torch.dtype,
    device: torch.device,
    *,
    global_step: int,
    reward_cache_call_index: int,
):
    batch_size = len(prompts)
    group_size = args.batchsize_gen
    num_anchors = args.neighbor_num_anchors
    latent_shape = (4, args.resolution // 8, args.resolution // 8)

    if batch_size < 1:
        raise ValueError("Expected at least one prompt.")
    if prompt_ids.shape[0] != batch_size:
        raise ValueError(f"prompt_ids batch mismatch: {prompt_ids.shape[0]} vs {batch_size}")

    with torch.no_grad():
        encoder_hidden_states_one = text_encoder(prompt_ids)[0]
    encoder_hidden_states = _expand_group_embeddings(encoder_hidden_states_one, group_size)

    base_noise = torch.randn((batch_size, *latent_shape), device=device, dtype=weight_dtype)
    deltas = torch.randn((batch_size, group_size, *latent_shape), device=device, dtype=weight_dtype)
    base_scale = (max(0.0, 1.0 - float(args.neighbor_sigma) ** 2)) ** 0.5
    noisy_latents = base_scale * base_noise[:, None, :, :, :] + float(args.neighbor_sigma) * deltas
    noisy_latents_flat = noisy_latents.reshape(batch_size * group_size, *latent_shape)
    timesteps_flat = torch.full((batch_size * group_size,), args.generation_timestep, device=device, dtype=torch.long)

    anchor_indices = torch.stack(
        [torch.randperm(group_size, device=device)[:num_anchors] for _ in range(batch_size)],
        dim=0,
    )
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    anchor_noisy = noisy_latents[batch_indices, anchor_indices].reshape(batch_size * num_anchors, *latent_shape)
    anchor_timesteps = torch.full((batch_size * num_anchors,), args.generation_timestep, device=device, dtype=torch.long)
    anchor_encoder_hidden_states = _expand_group_embeddings(encoder_hidden_states_one, num_anchors)

    with torch.no_grad():
        rollout_pred = unet(noisy_latents_flat, timesteps_flat, encoder_hidden_states).sample
        candidate_x0_old = pred_to_onestep_latent(noisy_latents_flat, rollout_pred).detach()
        ref_pred = ref_unet(anchor_noisy, anchor_timesteps, anchor_encoder_hidden_states).sample
        ref_anchor_x0 = pred_to_onestep_latent(anchor_noisy, ref_pred).detach().reshape(batch_size, num_anchors, *latent_shape)

    current_pred = unet(anchor_noisy, anchor_timesteps, anchor_encoder_hidden_states).sample
    current_anchor_x0 = pred_to_onestep_latent(anchor_noisy, current_pred).reshape(batch_size, num_anchors, *latent_shape)
    candidate_x0_old = candidate_x0_old.reshape(batch_size, group_size, *latent_shape)
    old_anchor_x0 = candidate_x0_old[batch_indices, anchor_indices]

    sampled_images = decode_latents_to_pil(
        vae,
        candidate_x0_old.reshape(batch_size * group_size, *latent_shape),
        chunk_size=args.vae_decode_chunk_size,
    )
    flat_prompts = [prompt for prompt in prompts for _ in range(group_size)]
    cache_enabled = bool(args.reward_cache_images) and (
        args.reward_cache_interval > 0 and int(global_step) % int(args.reward_cache_interval) == 0
    )
    score_values = cache_and_score_pil_rewards(
        selector=args.reward_selector,
        images=sampled_images,
        prompts=flat_prompts,
        cache_dir=args.reward_cache_dir,
        global_step=global_step,
        process_index=getattr(args, "process_index", 0),
        call_index=reward_cache_call_index,
        group_size=group_size,
        choice_model=args.choice_model,
        score_batch_size=args.reward_score_batch_size,
        enabled=cache_enabled,
    )
    rewards = torch.tensor(score_values, device=device, dtype=torch.float32).reshape(batch_size, group_size)
    advantages = torch.stack(
        [
            quasi_norm_advantages(
                prompt_rewards,
                p=args.neighbor_p_norm,
                scale=args.advantage_scale,
                clip=args.advantage_clip,
            )
            for prompt_rewards in rewards
        ],
        dim=0,
    ).detach()

    policy_loss, stats = batched_neighbor_grpo_loss(
        candidate_x0_old,
        current_anchor_x0,
        old_anchor_x0,
        advantages,
        clip_range=args.ppo_clip_range,
        max_log_ratio=args.max_log_ratio,
        temperature=args.neighbor_distance_temperature,
        reduction=args.neighbor_distance_reduction,
    )
    ppo_kl_loss = stats["approx_kl_loss"]
    ref_l2 = F.mse_loss(current_anchor_x0.float(), ref_anchor_x0.float())
    loss = policy_loss + args.ref_model_l2_weight * ref_l2 + args.policy_kl_weight * ppo_kl_loss
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_std": rewards.std(dim=1, unbiased=False).mean().detach(),
        "adv_abs": advantages.abs().mean().detach(),
        "ratio_mean": stats["ratio_mean"].detach(),
        "clipfrac": stats["clipfrac"].detach(),
        "ppo_kl": stats["approx_kl"].detach(),
        "ref_l2": ref_l2.detach(),
        "entropy": stats["entropy"].detach(),
    }


def _average_terms(terms: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = terms[0].keys()
    return {key: torch.stack([item[key] for item in terms]).mean() for key in keys}


def main() -> None:
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    args.process_index = accelerator.process_index
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_logging(args, accelerator)
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        snapshot = save_runtime_snapshot(args, accelerator)
        logger.info(f"Training log will be written to {log_path}")
        logger.info(f"Runtime snapshot saved to {snapshot['snapshot_dir']}")
    accelerator.wait_for_everyone()

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if "turbo" in args.pretrained_model_name_or_path.lower():
        enforce_zero_terminal_snr(noise_scheduler)

    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    ref_unet.requires_grad_(False).eval()
    _maybe_enable_xformers(unet)
    unet = _make_lora_unet(unet, args)

    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    dataset = PairsPromptDataset(
        pairs_jsonl=args.pairs_jsonl,
        tokenizer=tokenizer,
        train_mode="online",
        image_size=args.resolution,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.dataloader_num_workers > 0,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)
    unet.eval()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    args.reward_selector = build_reward_selector(args, accelerator.device)
    eval_selector = args.reward_selector if accelerator.is_main_process else None
    global_step = resume_training_state(accelerator, args.output_dir, args.resume_from_checkpoint, logger=logger)
    eval_prompts = load_prompt_file(args.eval_prompt_file, max_prompts=args.num_eval_prompts)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "eval_prompts.txt"), "w", encoding="utf-8") as handle:
            for prompt in eval_prompts:
                handle.write(prompt + "\n")
        accelerator.init_trackers("sdturbo-grpo", build_tracker_config(args))

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, disable=not accelerator.is_local_main_process)
    if global_step == 0 and args.eval_every_steps > 0:
        eval_summary = evaluate_fixed_prompts(
            args=args,
            accelerator=accelerator,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            selector=eval_selector,
            generation_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            prompts=eval_prompts,
            global_step=0,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and eval_summary is not None:
            metric = args.choice_model
            accelerator.log({f"eval_avg_{metric}": eval_summary[f"avg_{metric}"], f"eval_best_{metric}": eval_summary[f"best_{metric}"], f"eval_worst_{metric}": eval_summary[f"worst_{metric}"]}, step=0)

    reward_cache_call_index = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                prompt_batch = batch["prompt"]
                input_ids = batch["input_ids"]
                terms = _neighbor_based_grpo_batch_terms(
                    args,
                    unet,
                    ref_unet,
                    vae,
                    text_encoder,
                    prompt_batch,
                    input_ids,
                    weight_dtype,
                    accelerator.device,
                    global_step=global_step,
                    reward_cache_call_index=reward_cache_call_index,
                )
                reward_cache_call_index += 1
                loss = terms["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm_value = compute_grad_norm(unet.parameters()).to(accelerator.device)
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                gathered: Dict[str, float] = {}
                for key, value in terms.items():
                    gathered[key] = accelerator.gather(value.detach().repeat(args.train_batch_size)).mean().item()
                avg_grad = accelerator.gather(grad_norm_value.repeat(args.train_batch_size)).mean().item()
                log_payload: Dict[str, float] = {
                    "train_loss": gathered["loss"],
                    "policy_loss_unaccumulated": gathered["policy_loss"],
                    "reward_mean_unaccumulated": gathered["reward_mean"],
                    "reward_std_unaccumulated": gathered["reward_std"],
                    "adv_abs_unaccumulated": gathered["adv_abs"],
                    "ratio_mean_unaccumulated": gathered["ratio_mean"],
                    "clipfrac_unaccumulated": gathered["clipfrac"],
                    "ppo_kl_unaccumulated": gathered["ppo_kl"],
                    "ref_l2_unaccumulated": gathered["ref_l2"],
                    "surrogate_entropy_unaccumulated": gathered["entropy"],
                    "grad_norm_unaccumulated": avg_grad,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(log_payload, step=global_step)
                progress_bar.set_postfix(
                    loss=gathered["loss"],
                    reward=gathered["reward_mean"],
                    ratio=gathered["ratio_mean"],
                    clipfrac=gathered["clipfrac"],
                    kl=gathered["ppo_kl"],
                    ref_l2=gathered["ref_l2"],
                    grad=avg_grad,
                    lr=lr_scheduler.get_last_lr()[0],
                )
                progress_bar.update(1)

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    eval_summary = evaluate_fixed_prompts(
                        args=args,
                        accelerator=accelerator,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        selector=eval_selector,
                        generation_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        prompts=eval_prompts,
                        global_step=global_step,
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and eval_summary is not None:
                        metric = args.choice_model
                        accelerator.log({f"eval_avg_{metric}": eval_summary[f"avg_{metric}"], f"eval_best_{metric}": eval_summary[f"best_{metric}"], f"eval_worst_{metric}": eval_summary[f"worst_{metric}"]}, step=global_step)

                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    save_training_checkpoint(
                        accelerator,
                        unet,
                        args,
                        os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                        global_step,
                        "periodic",
                        logger=logger,
                    )
            if global_step >= args.max_train_steps:
                break

    if global_step > 0:
        save_training_checkpoint(
            accelerator,
            unet,
            args,
            os.path.join(args.output_dir, "final"),
            global_step,
            "final",
            logger=logger,
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
