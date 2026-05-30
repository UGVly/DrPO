import argparse
import copy
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from drpo.pickscore import DifferentiablePickScoreScorer
from drpo.data import Batch, PromptDataset, collate_preference_batch
from drpo.methods.sdxl_drpo.trainer import (
    _decode_latents_to_tensor,
    _encode_prompts,
    _sdxl_one_step_latents,
)
from drpo.methods.sdxl_common import (
    create_accelerator,
    dtype_for_mixed_precision,
    make_lora_trainable,
    maybe_enable_xformers,
    parse_names,
    resolve_unet_lora_dir,
    resume_global_step,
    save_runtime_snapshot,
    save_unet_checkpoint,
    setup_training_logging,
    TrainingStepCallbacks,
    trainable_parameters,
)
from drpo.paths import project_root, require_local_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SDXLDraftConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    prompt_file: str
    model_variant: str | None = "fp16"
    choice_model: str = "pickscore"
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
    pickscore_loss_weight: float = 1.0
    ref_model_l2_weight: float = 0.02
    score_std_weight: float = 0.0
    generation_chunk_size: int = 4
    vae_decode_chunk_size: int = 1


def _dtype_for(config: SDXLDraftConfig) -> torch.dtype:
    return dtype_for_mixed_precision(config.mixed_precision)


def _validate_config(config: SDXLDraftConfig) -> None:
    if config.choice_model != "pickscore":
        raise ValueError("SDXL-Draft currently supports only --choice_model pickscore.")
    if config.batchsize_gen < 1:
        raise ValueError("batchsize_gen must be >= 1.")
    if config.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1.")
    if config.resolution % 8:
        raise ValueError("resolution must be divisible by 8.")
    if config.guidance_scale != 0:
        raise ValueError("This one-step SDXL-Draft trainer is aligned to SDXL-Turbo with guidance_scale=0.")
    if config.ref_model_l2_weight < 0:
        raise ValueError("ref_model_l2_weight must be non-negative.")
    if config.generation_chunk_size < 1:
        raise ValueError("generation_chunk_size must be >= 1.")


def _setup_logging(config: SDXLDraftConfig, accelerator: Accelerator) -> None:
    setup_training_logging(config, accelerator, logger_name=__name__)


def _safe_tracker_config(config: SDXLDraftConfig) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in asdict(config).items():
        if isinstance(value, tuple):
            values[key] = ",".join(str(item) for item in value)
        elif value is None:
            values[key] = ""
        else:
            values[key] = value
    return values


def _save_runtime_snapshot(config: SDXLDraftConfig, accelerator: Accelerator) -> None:
    save_runtime_snapshot(config, accelerator)


def _save_checkpoint(accelerator: Accelerator, unet, config: SDXLDraftConfig, checkpoint_dir: Path, global_step: int, checkpoint_type: str) -> None:
    save_unet_checkpoint(
        accelerator,
        unet,
        config,
        checkpoint_dir,
        global_step=global_step,
        checkpoint_type=checkpoint_type,
        metadata={
            "model_type": "sdxl-turbo-draft-pickscore",
            "choice_model": config.choice_model,
            "pickscore_model_name_or_path": config.pickscore_model_name_or_path,
        },
    )


def _load_pickscore(config: SDXLDraftConfig, device: torch.device) -> DifferentiablePickScoreScorer:
    model_path = str(require_local_path(config.pickscore_model_name_or_path or "", description="PickScore model", must_be_file=False))
    processor_path = str(
        require_local_path(
            config.pickscore_processor_name_or_path or config.pickscore_model_name_or_path or "",
            description="PickScore processor",
            must_be_file=False,
        )
    )
    return DifferentiablePickScoreScorer(
        device,
        model_name_or_path=model_path,
        processor_name_or_path=processor_path,
        local_files_only=not config.pickscore_allow_remote,
    ).eval()


def train(config: SDXLDraftConfig) -> None:
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

    pipe.unet = make_lora_trainable(pipe.unet, config)
    reference_unet = None if config.use_lora else copy.deepcopy(pipe.unet).eval().requires_grad_(False)
    if hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()
    maybe_enable_xformers(pipe.unet, logger)

    dataset = PromptDataset(config.prompt_file, pipe.tokenizer, max_samples=config.max_train_samples, seed=config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_preference_batch,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(pipe.unet),
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

    if reference_unet is None:
        pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
            optimizer,
            dataloader,
            lr_scheduler,
        )
    else:
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
    pickscore = _load_pickscore(config, accelerator.device)
    latent_channels = int(accelerator.unwrap_model(pipe.unet).config.in_channels)
    latent_size = config.resolution // int(getattr(pipe, "vae_scale_factor", 8) or 8)

    if accelerator.is_main_process:
        accelerator.init_trackers("sdxl-turbo-draft", _safe_tracker_config(config))
        trainable = sum(parameter.numel() for parameter in trainable_parameters(accelerator.unwrap_model(pipe.unet)))
        logger.info("Starting SDXL-Draft training with %d trainable parameters.", trainable)
        logger.info("Output dir: %s", output_dir)
        if config.resume_from_checkpoint:
            logger.info("Resumed LoRA from %s", resolve_unet_lora_dir(config.resume_from_checkpoint))

    step = resume_global_step(config.resume_from_checkpoint)
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
                prompt_count = max(1, len(batch.prompts))
                batch_loss = torch.zeros((), device=accelerator.device)
                metric_weight = 0.0
                metric_sums: dict[str, torch.Tensor] = {}
                saw_chunk = False
                for prompt in batch.prompts:
                    remaining = config.batchsize_gen
                    while remaining > 0:
                        chunk_size = min(config.generation_chunk_size, remaining)
                        prompts = [prompt] * chunk_size
                        with torch.no_grad():
                            prompt_embeds, pooled_prompt_embeds = _encode_prompts(pipe, prompts, accelerator.device)
                        latents = torch.randn(
                            (chunk_size, latent_channels, latent_size, latent_size),
                            device=accelerator.device,
                            dtype=prompt_embeds.dtype,
                        )
                        generated_latents = _sdxl_one_step_latents(
                            pipe=pipe,
                            unet=pipe.unet,
                            latents=latents,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            resolution=config.resolution,
                            num_inference_steps=config.num_inference_steps,
                        )
                        if reference_unet is None:
                            base_unet = accelerator.unwrap_model(pipe.unet)
                            adapter_context = base_unet.disable_adapter() if hasattr(base_unet, "disable_adapter") else nullcontext()
                            ref_forward_unet = pipe.unet
                        else:
                            adapter_context = nullcontext()
                            ref_forward_unet = reference_unet

                        with torch.no_grad(), adapter_context:
                            reference_latents = _sdxl_one_step_latents(
                                pipe=pipe,
                                unet=ref_forward_unet,
                                latents=latents,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                resolution=config.resolution,
                                num_inference_steps=config.num_inference_steps,
                            )
                        decoded = _decode_latents_to_tensor(pipe.vae, generated_latents, chunk_size=config.vae_decode_chunk_size)
                        scores = pickscore.score_from_image_tensor(decoded, prompts)
                        pickscore_mean = scores.mean()
                        pickscore_std = scores.std(unbiased=False)
                        ref_l2 = F.mse_loss(generated_latents.float(), reference_latents.float())
                        loss_i = (
                            -config.pickscore_loss_weight * pickscore_mean
                            + config.ref_model_l2_weight * ref_l2
                            - config.score_std_weight * pickscore_std
                        )
                        chunk_weight = float(chunk_size) / float(config.batchsize_gen * prompt_count)
                        accelerator.backward(loss_i * chunk_weight)
                        batch_loss = batch_loss + loss_i.detach().float() * chunk_weight
                        metric_weight += chunk_weight
                        for key, value in {
                            "pickscore_mean": pickscore_mean,
                            "pickscore_std": pickscore_std,
                            "ref_model_l2": ref_l2,
                        }.items():
                            metric_sums[key] = metric_sums.get(key, torch.zeros((), device=accelerator.device))
                            metric_sums[key] = metric_sums[key] + value.detach().float() * chunk_weight
                        saw_chunk = True
                        remaining -= chunk_size

                if not saw_chunk:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters(pipe.unet), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            log_values = {}
            if metric_weight > 0:
                for key, value in metric_sums.items():
                    log_values[key] = value / metric_weight
            postfix = {"loss": batch_loss.detach()}
            if "pickscore_mean" in log_values:
                postfix["pickscore"] = f"{float(log_values['pickscore_mean'].detach().cpu()):.3f}"
            step, should_stop = callbacks.finish_step(
                step=step,
                loss=batch_loss,
                lr=lr_scheduler.get_last_lr()[0],
                log_values=log_values,
                postfix=postfix,
            )
            if should_stop:
                break

    callbacks.finish_training(step)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train SDXL-Turbo with DRaFT-style differentiable PickScore reward.")
    parser.add_argument("--pretrained_model_name_or_path", default=str(root / "models" / "stable-diffusion-xl-turbo"))
    parser.add_argument(
        "--output_dir",
        default=str(root / "outputs" / "sdxl-turbo-lora" / "draft" / "pickscore" / datetime.now().strftime("%Y%m%d%H%M%S")),
    )
    parser.add_argument("--prompt_file", default=str(root / "data" / "pickscore" / "train.txt"))
    parser.add_argument("--model_variant", default="fp16")
    parser.add_argument("--choice_model", default="pickscore", choices=["pickscore"])
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
    parser.add_argument("--pickscore_loss_weight", type=float, default=1.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.02)
    parser.add_argument("--score_std_weight", type=float, default=0.0)
    parser.add_argument("--generation_chunk_size", type=int, default=4)
    parser.add_argument("--vae_decode_chunk_size", type=int, default=1)
    return parser


def parse_config(argv: list[str] | None = None) -> SDXLDraftConfig:
    args = build_parser().parse_args(argv)
    return SDXLDraftConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        model_variant=args.model_variant or None,
        choice_model=args.choice_model,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        pickscore_processor_name_or_path=args.pickscore_processor_name_or_path or args.pickscore_model_name_or_path,
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
        lora_target_modules=parse_names(args.lora_target_modules),
        pickscore_loss_weight=args.pickscore_loss_weight,
        ref_model_l2_weight=args.ref_model_l2_weight,
        score_std_weight=args.score_std_weight,
        generation_chunk_size=args.generation_chunk_size,
        vae_decode_chunk_size=args.vae_decode_chunk_size,
    )


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
