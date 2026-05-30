#!/usr/bin/env python
"""One-step SD-Turbo VGG-Flow trainer.

The official VGG-Flow code builds a velocity target by adding a clipped reward
gradient to a reference flow field. This one-step adaptation trains the SD-Turbo
clean-latent prediction toward the analogous target:

    x0_target = x0_ref + eta(t) * reward_scale * grad_x0 reward(decode(x0)).

The value-network consistency branch from the multi-step official code is not
used here because the distilled SD-Turbo setup trains one transition.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

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
_SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from reward_gradient import (  # noqa: E402
    DifferentiableAestheticReward,
    DifferentiablePickScoreReward,
    compute_pickscore_reward_gradient,
    eta_multiplier,
)
from baselines.common import (  # noqa: E402
    PairsPromptDataset,
    build_tracker_config,
    collate_fn,
    compute_grad_norm,
    enforce_zero_terminal_snr,
    evaluate_fixed_prompts,
    load_prompt_file,
    save_runtime_snapshot,
    one_step_clean_latent,
    resume_training_state,
    save_training_checkpoint,
    setup_logging,
)

check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser("SD-Turbo one-step VGG-Flow trainer")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "sd-turbo"))
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "vggflow", "pickscore", "default"))
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
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
    parser.add_argument("--batchsize_gen", type=int, default=8)
    parser.add_argument("--generation_timestep", type=int, default=999)
    parser.add_argument("--generation_target_timestep", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--choice_model", type=str, default="pickscore", choices=["pickscore", "aes"])
    parser.add_argument("--pickscore_model_name_or_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "PickScore_v1"))
    parser.add_argument("--pickscore_processor_name_or_path", type=str, default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")
    parser.add_argument("--aesthetic_clip_model_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "CLIP-ViT-L-14"))
    parser.add_argument("--aesthetic_ckpt_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "Aesthetic", "sac+logos+ava1-l14-linearMSE.pth"))
    parser.add_argument("--aesthetic_allow_remote", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_lora", action="store_true", default=True)
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

    parser.add_argument("--reward_scale", type=float, default=1e3)
    parser.add_argument("--eta_mode", type=str, default="constant", choices=["constant", "linear", "quad"])
    parser.add_argument("--quantile_clipping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgrad_clip_threshold", type=float, default=1.0)
    parser.add_argument("--rgrad_quantile", type=float, default=0.8)
    parser.add_argument("--rgrad_jitter_count", type=int, default=1)
    parser.add_argument("--rgrad_jitter_std", type=float, default=0.0)
    parser.add_argument("--reward_masking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward_mask_threshold", type=float, default=0.0)
    parser.add_argument("--unet_reg_scale", type=float, default=0.0)
    parser.add_argument("--vae_decode_chunk_size", type=int, default=4)

    parser.add_argument("--eval_prompt_file", type=str, default=os.path.join(PROJECT_ROOT, "data", "pickscore", "test.txt"))
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--eval_output_subdir", type=str, default="eval_samples")

    args = parser.parse_args()
    if args.pickscore_processor_name_or_path is None:
        args.pickscore_processor_name_or_path = args.pickscore_model_name_or_path
    if args.generation_timestep < 0:
        raise ValueError("--generation_timestep must be >= 0.")
    if args.generation_target_timestep not in {-1, 0}:
        raise ValueError("One-step VGG-Flow currently supports --generation_target_timestep -1 or 0.")
    if args.rgrad_jitter_count < 1:
        raise ValueError("--rgrad_jitter_count must be >= 1.")
    if not (0.0 < args.rgrad_quantile <= 1.0):
        raise ValueError("--rgrad_quantile must be in (0, 1].")
    return args


def build_eval_selector(args, device: torch.device):
    if args.choice_model == "aes":
        from utils.aes_utils import Selector

        return Selector(
            device,
            clip_model_path=args.aesthetic_clip_model_path,
            ckpt_path=args.aesthetic_ckpt_path,
            local_files_only=not args.aesthetic_allow_remote,
        )

    from utils.pickscore_utils import Selector

    return Selector(
        device,
        model_name_or_path=args.pickscore_model_name_or_path,
        processor_name_or_path=args.pickscore_processor_name_or_path,
        local_files_only=not args.pickscore_allow_remote,
    )


def build_reward_model(args, device: torch.device):
    if args.choice_model == "aes":
        return DifferentiableAestheticReward(
            device,
            clip_model_path=args.aesthetic_clip_model_path,
            ckpt_path=args.aesthetic_ckpt_path,
            local_files_only=not args.aesthetic_allow_remote,
        )
    return DifferentiablePickScoreReward(
        device,
        model_name_or_path=args.pickscore_model_name_or_path,
        processor_name_or_path=args.pickscore_processor_name_or_path,
        local_files_only=not args.pickscore_allow_remote,
    )


def pred_to_onestep_latent(noisy_latents: torch.Tensor, model_pred: torch.Tensor) -> torch.Tensor:
    return one_step_clean_latent(noisy_latents, model_pred)


def main():
    args = parse_args()
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_logging(args, accelerator)

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        runtime_snapshot = save_runtime_snapshot(args, accelerator)
        logger.info("Training log will be written to %s", log_path)
        logger.info("Runtime snapshot saved to %s", runtime_snapshot["snapshot_dir"])
        logger.info("Using one-step VGG-Flow without the official multi-step value-net consistency branch.")
    accelerator.wait_for_everyone()

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    ref_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if "turbo" in args.pretrained_model_name_or_path.lower():
        enforce_zero_terminal_snr(noise_scheduler)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    ref_unet.eval()

    if is_xformers_available():
        import xformers

        if version.parse(xformers.__version__) >= version.parse("0.0.17"):
            try:
                unet.enable_xformers_memory_efficient_attention()
                ref_unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention.")
            except Exception as exc:
                logger.warning("Failed to enable xformers memory efficient attention: %s", exc)
    else:
        logger.warning("xformers is not available; continuing without memory efficient attention.")

    if args.use_lora:
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
        unet = get_peft_model(unet, lora_config)

    trainable_parameters = [param for param in unet.parameters() if param.requires_grad]
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

    reward_model = build_reward_model(args, accelerator.device).to(accelerator.device)
    reward_model.eval()
    eval_selector = build_eval_selector(args, accelerator.device) if accelerator.is_main_process else None

    global_step = resume_training_state(accelerator, args.output_dir, args.resume_from_checkpoint, logger=logger)
    rgrad_threshold = float(args.rgrad_clip_threshold)

    eval_prompts = load_prompt_file(args.eval_prompt_file, max_prompts=args.num_eval_prompts)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "eval_prompts.txt"), "w", encoding="utf-8") as handle:
            for prompt in eval_prompts:
                handle.write(prompt + "\n")
        accelerator.init_trackers("sdturbo-vggflow", build_tracker_config(args))

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )

    if args.eval_every_steps > 0 and global_step == 0:
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
            eval_metric_key = args.choice_model
            accelerator.log(
                {
                    f"eval_avg_{eval_metric_key}": eval_summary[f"avg_{eval_metric_key}"],
                    f"eval_best_{eval_metric_key}": eval_summary[f"best_{eval_metric_key}"],
                    f"eval_worst_{eval_metric_key}": eval_summary[f"worst_{eval_metric_key}"],
                },
                step=0,
            )
    elif args.eval_every_steps > 0 and accelerator.is_main_process:
        logger.info("Skipping step-0 eval after resume at global step %s.", global_step)

    sigma_value = float(args.generation_timestep) / 1000.0
    eta_value = eta_multiplier(args.eta_mode, torch.tensor(sigma_value)).item()

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                prompt_batch = batch["prompt"]
                input_ids = batch["input_ids"]
                bsz = input_ids.shape[0]

                total_loss = 0.0
                total_vgg_loss = 0.0
                total_ref_l2 = 0.0
                total_reward_mean = 0.0
                total_reward_std = 0.0
                total_rgrad_norm = 0.0
                total_mask_fraction = 0.0
                rgrad_norm_batches = []

                for i in range(bsz):
                    prompt = prompt_batch[i]
                    prompt_ids = input_ids[i : i + 1]
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(prompt_ids)[0]
                    encoder_hidden_states = encoder_hidden_states.repeat_interleave(args.batchsize_gen, dim=0)

                    noisy_latents = torch.randn(
                        (args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8),
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    timesteps = torch.full(
                        (args.batchsize_gen,),
                        args.generation_timestep,
                        device=accelerator.device,
                        dtype=torch.long,
                    )

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    policy_x0 = pred_to_onestep_latent(noisy_latents, model_pred)

                    with torch.no_grad():
                        ref_pred = ref_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        ref_x0 = pred_to_onestep_latent(noisy_latents, ref_pred)

                    reward_grad = compute_pickscore_reward_gradient(
                        vae=vae,
                        reward_model=reward_model,
                        latents=policy_x0,
                        prompts=[prompt] * args.batchsize_gen,
                        clip_threshold=rgrad_threshold,
                        quantile_clipping=args.quantile_clipping,
                        reward_mask_threshold=args.reward_mask_threshold,
                        jitter_count=args.rgrad_jitter_count,
                        jitter_std=args.rgrad_jitter_std,
                        decode_chunk_size=args.vae_decode_chunk_size,
                    )
                    target_x0 = ref_x0.detach().float() + eta_value * args.reward_scale * reward_grad.gradient

                    per_sample_loss = (policy_x0.float() - target_x0).pow(2).flatten(1).mean(dim=1)
                    if args.reward_masking:
                        vgg_loss_i = (per_sample_loss * reward_grad.reward_mask).sum() / reward_grad.reward_mask.sum().clamp_min(1.0)
                    else:
                        vgg_loss_i = per_sample_loss.mean()
                    ref_l2_i = F.mse_loss(policy_x0.float(), ref_x0.float())
                    loss_i = vgg_loss_i + args.unet_reg_scale * ref_l2_i

                    total_loss = total_loss + loss_i
                    total_vgg_loss = total_vgg_loss + vgg_loss_i.detach()
                    total_ref_l2 = total_ref_l2 + ref_l2_i.detach()
                    total_reward_mean = total_reward_mean + reward_grad.rewards.mean()
                    total_reward_std = total_reward_std + reward_grad.rewards.std(unbiased=False)
                    total_rgrad_norm = total_rgrad_norm + reward_grad.gradient_norm.mean()
                    total_mask_fraction = total_mask_fraction + reward_grad.reward_mask.mean()
                    rgrad_norm_batches.append(reward_grad.gradient_norm)

                loss = total_loss / bsz
                vgg_loss = total_vgg_loss / bsz
                ref_l2 = total_ref_l2 / bsz
                reward_mean = total_reward_mean / bsz
                reward_std = total_reward_std / bsz
                rgrad_norm = total_rgrad_norm / bsz
                mask_fraction = total_mask_fraction / bsz
                rgrad_norm_values = torch.cat(rgrad_norm_batches)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm_value = compute_grad_norm(unet.parameters()).to(accelerator.device)
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                gathered_rgrad_norm = accelerator.gather(rgrad_norm_values.detach()).float()
                if args.quantile_clipping and gathered_rgrad_norm.numel() > 0:
                    rgrad_threshold = float(torch.quantile(gathered_rgrad_norm, args.rgrad_quantile).item())

                avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_vgg_loss = accelerator.gather(vgg_loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_ref_l2 = accelerator.gather(ref_l2.detach().repeat(args.train_batch_size)).mean().item()
                avg_reward_mean = accelerator.gather(reward_mean.detach().repeat(args.train_batch_size)).mean().item()
                avg_reward_std = accelerator.gather(reward_std.detach().repeat(args.train_batch_size)).mean().item()
                avg_rgrad_norm = accelerator.gather(rgrad_norm.detach().repeat(args.train_batch_size)).mean().item()
                avg_mask_fraction = accelerator.gather(mask_fraction.detach().repeat(args.train_batch_size)).mean().item()
                avg_grad = accelerator.gather(grad_norm_value.repeat(args.train_batch_size)).mean().item()

                log_payload: Dict[str, float] = {
                    "train_loss": avg_loss,
                    "vgg_loss_unaccumulated": avg_vgg_loss,
                    "ref_l2_unaccumulated": avg_ref_l2,
                    "reward_mean_unaccumulated": avg_reward_mean,
                    "reward_std_unaccumulated": avg_reward_std,
                    "rgrad_norm_unaccumulated": avg_rgrad_norm,
                    "rgrad_threshold": rgrad_threshold,
                    "reward_mask_fraction": avg_mask_fraction,
                    "eta": eta_value,
                    "grad_norm_unaccumulated": avg_grad,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(log_payload, step=global_step)
                progress_bar.set_postfix(
                    loss=avg_loss,
                    reward=avg_reward_mean,
                    rgrad=avg_rgrad_norm,
                    thr=rgrad_threshold,
                    ref_l2=avg_ref_l2,
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
                        eval_metric_key = args.choice_model
                        accelerator.log(
                            {
                                f"eval_avg_{eval_metric_key}": eval_summary[f"avg_{eval_metric_key}"],
                                f"eval_best_{eval_metric_key}": eval_summary[f"best_{eval_metric_key}"],
                                f"eval_worst_{eval_metric_key}": eval_summary[f"worst_{eval_metric_key}"],
                            },
                            step=global_step,
                        )

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
