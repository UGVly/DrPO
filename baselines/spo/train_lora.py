#!/usr/bin/env python
"""One-step SD-Turbo SPO/PSO baseline trainer.

This implements the pairwise sample objective from Pairwise Sample Optimization
for the local one-step SD-Turbo setup. The original PSO trainer estimates
per-step diffusion log-probability ratios across a multi-step SDXL-Turbo
trajectory. Here the trajectory has one distilled transition, so the sampled
clean latent is treated as the action under a Gaussian policy centered at the
UNet-predicted one-step clean latent.
"""

import argparse
import json
import logging
import os
import shlex
import shutil
import socket
import sys
from datetime import datetime
from typing import Dict

import accelerate
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

from losses import pairwise_spo_loss  # noqa: E402
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
    save_training_checkpoint,
    setup_logging,
)

check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


def _make_json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)


def save_runtime_snapshot(args, accelerator: Accelerator) -> Dict[str, object]:
    snapshot_dir = os.path.join(args.output_dir, "run_metadata")
    code_dir = os.path.join(snapshot_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    resolved_config = {key: _make_json_safe(value) for key, value in vars(args).items()}
    config_json_path = os.path.join(snapshot_dir, "resolved_config.json")
    with open(config_json_path, "w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2, ensure_ascii=False, sort_keys=True)

    config_txt_path = os.path.join(snapshot_dir, "resolved_config.txt")
    with open(config_txt_path, "w", encoding="utf-8") as handle:
        for key in sorted(resolved_config):
            handle.write(f"{key}={resolved_config[key]}\n")

    command_path = os.path.join(snapshot_dir, "launch_command.sh")
    with open(command_path, "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write(shlex.join([sys.executable, *sys.argv]) + "\n")

    runtime_info = {
        "created_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "script_path": os.path.abspath(__file__),
        "world_size": accelerator.num_processes,
        "main_process_index": accelerator.process_index,
        "device": str(accelerator.device),
        "mixed_precision": accelerator.mixed_precision,
        "torch_version": torch.__version__,
        "accelerate_version": accelerate.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    runtime_info_path = os.path.join(snapshot_dir, "runtime_info.json")
    with open(runtime_info_path, "w", encoding="utf-8") as handle:
        json.dump(runtime_info, handle, indent=2, ensure_ascii=False, sort_keys=True)

    copied_files = []
    for source_path in [os.path.abspath(__file__)]:
        if not os.path.exists(source_path):
            continue
        relative_path = os.path.relpath(source_path, PROJECT_ROOT)
        target_path = os.path.join(code_dir, relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_files.append(relative_path)

    manifest_path = os.path.join(snapshot_dir, "code_snapshot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"files": copied_files}, handle, indent=2, ensure_ascii=False, sort_keys=True)

    return {
        "snapshot_dir": snapshot_dir,
        "config_json_path": config_json_path,
        "runtime_info_path": runtime_info_path,
        "manifest_path": manifest_path,
    }


def pred_to_onestep_latent(noisy_latents: torch.Tensor, model_pred: torch.Tensor) -> torch.Tensor:
    return one_step_clean_latent(noisy_latents, model_pred)




def derive_noise_policy_offset(noisy_latents: torch.Tensor) -> torch.Tensor:
    offset = noisy_latents.float()
    mean = offset.mean(dim=(1, 2, 3), keepdim=True)
    std = offset.std(dim=(1, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    return ((offset - mean) / std).to(dtype=noisy_latents.dtype)


def gaussian_logprob(action: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"Expected positive std, got {std}")
    diff = (action.float() - mean.float()) / std
    return -0.5 * diff.square().flatten(1).mean(dim=1)


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

def parse_args():
    parser = argparse.ArgumentParser("SD-Turbo one-step SPO trainer")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "sd-turbo"))
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "spo", "pickscore", "default"))
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
    parser.add_argument("--batchsize_gen", type=int, default=8, help="Even number of generated samples per prompt; adjacent samples form SPO pairs.")
    parser.add_argument("--generation_timestep", type=int, default=999)
    parser.add_argument("--generation_target_timestep", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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

    parser.add_argument("--rollout_action_std", type=float, default=0.05)
    parser.add_argument("--spo_beta", type=float, default=50.0)
    parser.add_argument("--spo_clip_range", type=float, default=0.1)
    parser.add_argument("--max_log_ratio", type=float, default=10.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.0)

    parser.add_argument("--eval_prompt_file", type=str, default=os.path.join(PROJECT_ROOT, "data", "pickscore", "test.txt"))
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--eval_output_subdir", type=str, default="eval_samples")

    args = parser.parse_args()
    if args.pickscore_processor_name_or_path is None:
        args.pickscore_processor_name_or_path = args.pickscore_model_name_or_path
    if args.batchsize_gen < 2 or args.batchsize_gen % 2:
        raise ValueError("--batchsize_gen must be an even integer >= 2 for SPO pair construction.")
    if args.rollout_action_std <= 0:
        raise ValueError("--rollout_action_std must be > 0.")
    if args.spo_clip_range < 0 or args.spo_clip_range >= 1:
        raise ValueError("--spo_clip_range must be in [0, 1).")
    if args.spo_beta <= 0:
        raise ValueError("--spo_beta must be > 0.")
    if args.max_log_ratio <= 0:
        raise ValueError("--max_log_ratio must be > 0.")
    if args.generation_timestep < 0:
        raise ValueError("--generation_timestep must be >= 0.")
    if args.generation_target_timestep >= 0 and args.generation_target_timestep > args.generation_timestep:
        raise ValueError("--generation_target_timestep must be <= --generation_timestep (or -1).")
    return args


def main():
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.logging_dir))
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
        logger.info(f"Training log will be written to {log_path}")
        logger.info(f"Runtime snapshot saved to {runtime_snapshot['snapshot_dir']}")
    accelerator.wait_for_everyone()

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

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
                logger.info("Enabled xformers memory efficient attention.")
            except Exception as exc:
                logger.warning(f"Failed to enable xformers memory efficient attention: {exc}")
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

    reward_selector = build_reward_selector(args, accelerator.device)
    eval_selector = reward_selector if accelerator.is_main_process else None
    global_step = resume_training_state(accelerator, args.output_dir, args.resume_from_checkpoint, logger=logger)

    eval_prompts = load_prompt_file(args.eval_prompt_file, max_prompts=args.num_eval_prompts)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "eval_prompts.txt"), "w", encoding="utf-8") as handle:
            for prompt in eval_prompts:
                handle.write(prompt + "\n")
        accelerator.init_trackers("sdturbo-spo", build_tracker_config(args))

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

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                prompt_batch = batch["prompt"]
                input_ids = batch["input_ids"]
                bsz = input_ids.shape[0]

                total_loss = 0.0
                total_spo_loss = 0.0
                total_reward_mean = 0.0
                total_reward_margin = 0.0
                total_logit = 0.0
                total_ratio_mean = 0.0
                total_pair_accuracy = 0.0
                total_winner_log_ratio = 0.0
                total_loser_log_ratio = 0.0
                total_tie_fraction = 0.0
                total_ref_l2 = 0.0

                for i in range(bsz):
                    prompt = prompt_batch[i]
                    prompt_ids = input_ids[i : i + 1]
                    encoder_hidden_states = text_encoder(prompt_ids)[0].repeat_interleave(args.batchsize_gen, dim=0)
                    noisy_latents = torch.randn(
                        (args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8),
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    timesteps = torch.ones((args.batchsize_gen,), device=accelerator.device, dtype=torch.long) * args.generation_timestep

                    with torch.no_grad():
                        rollout_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        rollout_x0_mean = pred_to_onestep_latent(noisy_latents, rollout_pred)
                        ref_pred = ref_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        ref_x0_mean = pred_to_onestep_latent(noisy_latents, ref_pred)

                    action_offset = derive_noise_policy_offset(noisy_latents)
                    sampled_x0 = rollout_x0_mean.detach() + args.rollout_action_std * action_offset
                    sampled_images = decode_latents_to_pil(vae, sampled_x0)
                    reward_values = torch.tensor(
                        reward_selector.score(sampled_images, prompt),
                        device=accelerator.device,
                        dtype=torch.float32,
                    )

                    current_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    current_x0_mean = pred_to_onestep_latent(noisy_latents, current_pred)
                    current_logp = gaussian_logprob(sampled_x0.detach(), current_x0_mean, args.rollout_action_std)
                    ref_logp = gaussian_logprob(sampled_x0.detach(), ref_x0_mean.detach(), args.rollout_action_std)
                    spo_loss_i, spo_stats = pairwise_spo_loss(
                        current_logp,
                        ref_logp,
                        reward_values,
                        beta=args.spo_beta,
                        clip_range=args.spo_clip_range,
                        max_log_ratio=args.max_log_ratio,
                    )
                    ref_l2_i = F.mse_loss(current_x0_mean.float(), ref_x0_mean.float())
                    loss_i = spo_loss_i + args.ref_model_l2_weight * ref_l2_i

                    total_loss = total_loss + loss_i
                    total_spo_loss = total_spo_loss + spo_loss_i.detach()
                    total_reward_mean = total_reward_mean + reward_values.mean().detach()
                    total_reward_margin = total_reward_margin + spo_stats["reward_margin"].detach()
                    total_logit = total_logit + spo_stats["spo_logit"].detach()
                    total_ratio_mean = total_ratio_mean + spo_stats["ratio_mean"].detach()
                    total_pair_accuracy = total_pair_accuracy + spo_stats["pair_accuracy"].detach()
                    total_winner_log_ratio = total_winner_log_ratio + spo_stats["winner_log_ratio"].detach()
                    total_loser_log_ratio = total_loser_log_ratio + spo_stats["loser_log_ratio"].detach()
                    total_tie_fraction = total_tie_fraction + spo_stats["tie_fraction"].detach()
                    total_ref_l2 = total_ref_l2 + ref_l2_i.detach()

                loss = total_loss / bsz
                spo_loss = total_spo_loss / bsz
                reward_mean = total_reward_mean / bsz
                reward_margin = total_reward_margin / bsz
                spo_logit = total_logit / bsz
                ratio_mean = total_ratio_mean / bsz
                pair_accuracy = total_pair_accuracy / bsz
                winner_log_ratio = total_winner_log_ratio / bsz
                loser_log_ratio = total_loser_log_ratio / bsz
                tie_fraction = total_tie_fraction / bsz
                ref_l2 = total_ref_l2 / bsz

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm_value = compute_grad_norm(unet.parameters()).to(accelerator.device)
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_spo_loss = accelerator.gather(spo_loss.repeat(args.train_batch_size)).mean().item()
                avg_reward_mean = accelerator.gather(reward_mean.repeat(args.train_batch_size)).mean().item()
                avg_reward_margin = accelerator.gather(reward_margin.repeat(args.train_batch_size)).mean().item()
                avg_spo_logit = accelerator.gather(spo_logit.repeat(args.train_batch_size)).mean().item()
                avg_ratio_mean = accelerator.gather(ratio_mean.repeat(args.train_batch_size)).mean().item()
                avg_pair_accuracy = accelerator.gather(pair_accuracy.repeat(args.train_batch_size)).mean().item()
                avg_winner_log_ratio = accelerator.gather(winner_log_ratio.repeat(args.train_batch_size)).mean().item()
                avg_loser_log_ratio = accelerator.gather(loser_log_ratio.repeat(args.train_batch_size)).mean().item()
                avg_tie_fraction = accelerator.gather(tie_fraction.repeat(args.train_batch_size)).mean().item()
                avg_ref_l2 = accelerator.gather(ref_l2.repeat(args.train_batch_size)).mean().item()
                avg_grad = accelerator.gather(grad_norm_value.repeat(args.train_batch_size)).mean().item()

                log_payload: Dict[str, float] = {
                    "train_loss": avg_loss,
                    "spo_loss_unaccumulated": avg_spo_loss,
                    "reward_mean_unaccumulated": avg_reward_mean,
                    "reward_pair_margin_unaccumulated": avg_reward_margin,
                    "spo_logit_unaccumulated": avg_spo_logit,
                    "ratio_mean_unaccumulated": avg_ratio_mean,
                    "pair_accuracy_unaccumulated": avg_pair_accuracy,
                    "winner_log_ratio_unaccumulated": avg_winner_log_ratio,
                    "loser_log_ratio_unaccumulated": avg_loser_log_ratio,
                    "tie_fraction_unaccumulated": avg_tie_fraction,
                    "ref_l2_unaccumulated": avg_ref_l2,
                    "grad_norm_unaccumulated": avg_grad,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(log_payload, step=global_step)
                progress_bar.set_postfix(
                    loss=avg_loss,
                    spo=avg_spo_loss,
                    reward=avg_reward_mean,
                    margin=avg_reward_margin,
                    acc=avg_pair_accuracy,
                    ratio=avg_ratio_mean,
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
                        logger.info(
                            f"Eval step {global_step}: avg_{eval_metric_key}={eval_summary[f'avg_{eval_metric_key}']:.6f}, "
                            f"best={eval_summary[f'best_{eval_metric_key}']:.6f}, "
                            f"worst={eval_summary[f'worst_{eval_metric_key}']:.6f}"
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
