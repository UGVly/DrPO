#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, CLIPTextModel, CLIPTokenizer

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from baselines.common import (  # noqa: E402
    PairsPromptDataset,
    _build_fallback_pickscore_selector,
    build_tracker_config,
    collate_fn,
    compute_grad_norm,
    decode_latents_to_pil,
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



class DifferentiablePickScoreScorer(nn.Module):
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str,
        processor_name_or_path: str,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).eval().to(device)
        self.model.requires_grad_(False)

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("Expected AutoProcessor with an image_processor for PickScore.")

        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))

        def resolve_image_size(size_cfg, keys):
            if hasattr(size_cfg, "get"):
                for key in keys:
                    value = size_cfg.get(key)
                    if value is not None:
                        return int(value)
                return 224
            return int(size_cfg)

        size_cfg = getattr(image_processor, "crop_size", None)
        if hasattr(size_cfg, "get"):
            self.image_size = resolve_image_size(size_cfg, ("height", "width"))
        elif isinstance(size_cfg, int):
            self.image_size = int(size_cfg)
        else:
            size_cfg = getattr(image_processor, "size", {"shortest_edge": 224})
            if hasattr(size_cfg, "get"):
                self.image_size = resolve_image_size(size_cfg, ("shortest_edge", "height", "width"))
            else:
                self.image_size = int(size_cfg)

    def _to_tensor(self, out):
        if isinstance(out, torch.Tensor):
            return out
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        if getattr(out, "last_hidden_state", None) is not None:
            return out.last_hidden_state[:, 0, :]
        raise TypeError(f"Unsupported feature output type: {type(out)}")

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images = ((images.clamp(-1, 1) + 1.0) / 2.0).float()
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image_mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        image_std = self.image_std.to(device=images.device, dtype=images.dtype)
        return (images - image_mean) / image_std

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embs = self._to_tensor(self.model.get_text_features(**text_inputs))
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        return text_embs.detach()

    def score_from_image_tensor(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        pixel_values = self.preprocess_images(images).to(device=self.device, dtype=self.model.dtype)
        text_embs = self.encode_text(prompts)
        image_embs = self._to_tensor(self.model.get_image_features(pixel_values=pixel_values))
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * torch.sum(text_embs * image_embs, dim=-1)
        return logits.float()


def decode_latents_to_tensor(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    chunk_size: int = 1,
) -> torch.Tensor:
    scaled_latents = (latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)
    if scaled_latents.shape[0] <= chunk_size:
        decoded = vae.decode(scaled_latents).sample
        return decoded.float().clamp(-1, 1)

    decoded_chunks = [vae.decode(chunk).sample for chunk in scaled_latents.split(chunk_size)]
    return torch.cat(decoded_chunks, dim=0).float().clamp(-1, 1)


def parse_args():
    parser = argparse.ArgumentParser("SD-Turbo PickScore direct-backprop trainer")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "sd-turbo"),
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/sdturbo-pickscore-direct")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs_jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "pairs.jsonl"))
    parser.add_argument(
        "--pickscore_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "PickScore_v1"),
    )
    parser.add_argument("--pickscore_processor_name_or_path", type=str, default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")
    parser.add_argument("--choice_model", type=str, default="pickscore", choices=["pickscore"])

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--batchsize_gen", type=int, default=8)
    parser.add_argument(
        "--generation_timestep",
        type=int,
        default=999,
        help="Timestep fed to the one-step UNet during training/eval generation.",
    )
    parser.add_argument(
        "--generation_target_timestep",
        type=int,
        default=0,
        help="Project the predicted clean sample back to this timestep before VAE decode. Use -1 to decode x0 directly.",
    )
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=512)

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

    parser.add_argument("--pickscore_loss_weight", type=float, default=1.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.0)
    parser.add_argument("--score_std_weight", type=float, default=0.0)

    parser.add_argument(
        "--eval_prompt_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "prompts", "pickapicv2_test_unique.txt"),
    )
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--eval_output_subdir", type=str, default="eval_samples")
    args = parser.parse_args()

    if args.generation_timestep < 0:
        raise ValueError("--generation_timestep must be >= 0.")
    if args.generation_target_timestep >= 0 and args.generation_target_timestep > args.generation_timestep:
        raise ValueError("--generation_target_timestep must be <= --generation_timestep (or -1).")
    if args.pickscore_processor_name_or_path is None:
        args.pickscore_processor_name_or_path = args.pickscore_model_name_or_path
    return args


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

    runtime_snapshot = None
    if accelerator.is_main_process:
        runtime_snapshot = save_runtime_snapshot(args, accelerator)
        logger.info(f"Training log will be written to {log_path}")
        logger.info(f"Resolved config saved to {runtime_snapshot['config_json_path']}")
        logger.info(f"Runtime snapshot saved to {runtime_snapshot['snapshot_dir']}")
    accelerator.wait_for_everyone()

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if "turbo" in args.pretrained_model_name_or_path.lower():
        enforce_zero_terminal_snr(noise_scheduler)
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
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
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

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)

    if is_xformers_available():
        import xformers

        if version.parse(xformers.__version__) >= version.parse("0.0.17"):
            unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is required")

    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

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

    trainable_parameters = [p for p in unet.parameters() if p.requires_grad]
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
    dataloader = torch.utils.data.DataLoader(
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    local_files_only = not args.pickscore_allow_remote
    if args.pickscore_allow_remote and accelerator.is_main_process:
        _ = DifferentiablePickScoreScorer(
            accelerator.device,
            model_name_or_path=args.pickscore_model_name_or_path,
            processor_name_or_path=args.pickscore_processor_name_or_path,
            local_files_only=False,
        )
        accelerator.wait_for_everyone()
        local_files_only = True

    diff_pickscore = DifferentiablePickScoreScorer(
        accelerator.device,
        model_name_or_path=args.pickscore_model_name_or_path,
        processor_name_or_path=args.pickscore_processor_name_or_path,
        local_files_only=local_files_only,
    ).to(accelerator.device)
    eval_selector = None
    if accelerator.is_main_process:
        eval_selector = _build_fallback_pickscore_selector(
            accelerator.device,
            model_name_or_path=args.pickscore_model_name_or_path,
            processor_name_or_path=args.pickscore_processor_name_or_path,
            local_files_only=local_files_only,
        )

    global_step = resume_training_state(accelerator, args.output_dir, args.resume_from_checkpoint, logger=logger)

    eval_prompts = load_prompt_file(args.eval_prompt_file, max_prompts=args.num_eval_prompts)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "eval_prompts.txt"), "w", encoding="utf-8") as f:
            for prompt in eval_prompts:
                f.write(prompt + "\n")
        accelerator.init_trackers("sdturbo-pickscore-direct", build_tracker_config(args))

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )

    if global_step == 0:
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
            accelerator.log(
                {
                    "eval_avg_pickscore": eval_summary["avg_pickscore"],
                    "eval_best_pickscore": eval_summary["best_pickscore"],
                    "eval_worst_pickscore": eval_summary["worst_pickscore"],
                },
                step=0,
            )
            logger.info(
                f"Eval step 0: avg_pickscore={eval_summary['avg_pickscore']:.6f}, "
                f"best={eval_summary['best_pickscore']:.6f}, worst={eval_summary['worst_pickscore']:.6f}"
            )
    elif accelerator.is_main_process:
        logger.info("Skipping step-0 eval after resume at global step %s.", global_step)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                prompt_batch = batch["prompt"]
                input_ids = batch["input_ids"]
                bsz = input_ids.shape[0]
                total_loss = 0.0
                total_pickscore_loss = 0.0
                total_pickscore_mean = 0.0
                total_pickscore_std = 0.0
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
                    timesteps = torch.ones((args.batchsize_gen,), device=accelerator.device, dtype=torch.long) * 999
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    model_pred_latent = one_step_clean_latent(noisy_latents, model_pred)

                    with torch.no_grad():
                        ref_pred = ref_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        ref_pred_latent = one_step_clean_latent(noisy_latents, ref_pred)

                    decoded = decode_latents_to_tensor(vae, model_pred_latent)
                    scores = diff_pickscore.score_from_image_tensor(decoded, [prompt] * args.batchsize_gen)
                    pickscore_mean_i = scores.mean()
                    pickscore_std_i = scores.std(unbiased=False)
                    pickscore_loss_i = -pickscore_mean_i
                    ref_l2_i = F.mse_loss(model_pred_latent.float(), ref_pred_latent.float())
                    loss_i = (
                        args.pickscore_loss_weight * pickscore_loss_i
                        + args.ref_model_l2_weight * ref_l2_i
                        - args.score_std_weight * pickscore_std_i
                    )

                    total_loss = total_loss + loss_i
                    total_pickscore_loss = total_pickscore_loss + pickscore_loss_i
                    total_pickscore_mean = total_pickscore_mean + pickscore_mean_i
                    total_pickscore_std = total_pickscore_std + pickscore_std_i
                    total_ref_l2 = total_ref_l2 + ref_l2_i

                loss = total_loss / bsz
                pickscore_loss = total_pickscore_loss / bsz
                pickscore_mean = total_pickscore_mean / bsz
                pickscore_std = total_pickscore_std / bsz
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
                avg_pickscore_loss = accelerator.gather(pickscore_loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_pickscore_mean = accelerator.gather(pickscore_mean.detach().repeat(args.train_batch_size)).mean().item()
                avg_pickscore_std = accelerator.gather(pickscore_std.detach().repeat(args.train_batch_size)).mean().item()
                avg_ref_l2 = accelerator.gather(ref_l2.detach().repeat(args.train_batch_size)).mean().item()
                avg_grad = accelerator.gather(grad_norm_value.repeat(args.train_batch_size)).mean().item()

                log_payload: Dict[str, float] = {
                    "train_loss": avg_loss,
                    "pickscore_loss_unaccumulated": avg_pickscore_loss,
                    "pickscore_mean_unaccumulated": avg_pickscore_mean,
                    "pickscore_std_unaccumulated": avg_pickscore_std,
                    "ref_l2_unaccumulated": avg_ref_l2,
                    "grad_norm_unaccumulated": avg_grad,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(log_payload, step=global_step)
                progress_bar.set_postfix(
                    step_loss=avg_loss,
                    pickscore=avg_pickscore_mean,
                    pickstd=avg_pickscore_std,
                    ref_l2=avg_ref_l2,
                    grad_norm=avg_grad,
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
                        accelerator.log(
                            {
                                "eval_avg_pickscore": eval_summary["avg_pickscore"],
                                "eval_best_pickscore": eval_summary["best_pickscore"],
                                "eval_worst_pickscore": eval_summary["worst_pickscore"],
                            },
                            step=global_step,
                        )
                        logger.info(
                            f"Eval step {global_step}: avg_pickscore={eval_summary['avg_pickscore']:.6f}, "
                            f"best={eval_summary['best_pickscore']:.6f}, worst={eval_summary['worst_pickscore']:.6f}"
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
