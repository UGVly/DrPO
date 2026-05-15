#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import diffusers.utils.import_utils as diffusers_import_utils

    diffusers_import_utils._xformers_available = False
except Exception:
    pass

from drpo.config import DrPOConfig
from drpo.features import FrozenMAELatentFeatureExtractor
from drpo.paths import project_root, require_local_path
from drpo.sdturbo import decode_latents_to_tensor, load_sdturbo_components, one_step_clean_latent
from drpo.training.trainer import _compute_prompt_terms
from drpo.utils.tensors import normalize_prompts, select_disjoint_pref_indices


COMPONENTS = (
    "text_encode",
    "student_unet_generate",
    "reference_unet_generate",
    "vae_decode",
    "hpsv3_score",
    "rank_select",
    "feature_extract",
    "loss_build",
    "backward",
    "optimizer_step",
    "eval_generate",
    "eval_hpsv3_score",
)


class CudaTimer:
    def __init__(self) -> None:
        self.totals = defaultdict(float)

    def sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def measure(self, name: str):
        timer = self

        class _Measure:
            def __enter__(self):
                timer.sync()
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                timer.sync()
                timer.totals[name] += time.perf_counter() - self.start

        return _Measure()

    def vector(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.totals.get(name, 0.0) for name in COMPONENTS], device=device, dtype=torch.float64)


class DifferentiableHPSv3Scorer(nn.Module):
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str | os.PathLike[str],
        base_model_path: str | os.PathLike[str],
        config_path: str | os.PathLike[str] | None = None,
        *,
        differentiable: bool,
        gradient_checkpointing: bool,
        score_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.score_chunk_size = max(int(score_chunk_size), 1)
        self.differentiable = bool(differentiable)

        import yaml
        from hpsv3 import HPSv3RewardInferencer
        from hpsv3.inference import _MODEL_CONFIG_PATH

        root = project_root()
        self.checkpoint_path = require_local_path(
            checkpoint_path or root / "models" / "HPSv3" / "HPSv3.safetensors",
            description="HPSv3 checkpoint",
            must_be_file=True,
        )
        self.base_model_path = require_local_path(
            base_model_path or root / "models" / "Qwen2-VL-7B-Instruct",
            description="HPSv3 Qwen2-VL base model",
            must_be_file=False,
        )
        self.config_path = self._local_config_path(config_path, _MODEL_CONFIG_PATH, yaml)
        self.inferencer = HPSv3RewardInferencer(
            config_path=str(self.config_path),
            checkpoint_path=str(self.checkpoint_path),
            device=str(self.device),
            differentiable=self.differentiable,
        )
        self.model = self.inferencer.model
        self.model.requires_grad_(False)
        self.model.eval()
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()

    def _local_config_path(self, config_path, model_config_path, yaml_module) -> Path:
        if config_path:
            return require_local_path(config_path, description="HPSv3 config", must_be_file=True)
        default_config = Path(model_config_path) / "HPSv3_7B.yaml"
        with default_config.open("r", encoding="utf-8") as handle:
            config = yaml_module.safe_load(handle)
        config["model_name_or_path"] = str(self.base_model_path)
        config["output_dir"] = str(project_root() / "outputs" / "hpsv3-wallclock-profile")
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="wallclock_hpsv3_", delete=False, encoding="utf-8")
        with handle:
            yaml_module.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)

    def score_from_image_tensor(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        prompts = normalize_prompts(prompts, images.shape[0])
        hps_images = (images.clamp(-1, 1) + 1.0) * 127.5
        hps_images = hps_images.to(device=self.device, dtype=torch.float32)
        prompt_list = list(prompts)
        chunks = []
        for start in range(0, hps_images.shape[0], self.score_chunk_size):
            end = start + self.score_chunk_size
            rewards = self.inferencer.reward([image for image in hps_images[start:end]], prompt_list[start:end])
            values = rewards[:, 0] if rewards.ndim > 1 else rewards
            chunks.append(values.float())
        return torch.cat(chunks, dim=0)


def install_hpsv3_checkpoint_key_patch() -> None:
    import transformers
    from packaging import version

    if version.parse(transformers.__version__) < version.parse("4.55.0"):
        return
    import safetensors.torch

    if getattr(safetensors.torch.load_file, "_drpo_hpsv3_patched", False):
        return
    original_load_file = safetensors.torch.load_file

    def patched_load_file(filename, *args, **kwargs):
        state_dict = original_load_file(filename, *args, **kwargs)
        if Path(str(filename)).name != "HPSv3.safetensors":
            return state_dict
        if any(key.startswith("model.language_model.") for key in state_dict):
            return state_dict
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                remapped[f"model.visual.{key[len('visual.') :]}"] = value
            elif key.startswith("model."):
                remapped[f"model.language_model.{key[len('model.') :]}"] = value
            else:
                remapped[key] = value
        return remapped

    patched_load_file._drpo_hpsv3_patched = True
    safetensors.torch.load_file = patched_load_file


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["drpo", "draft"], required=True)
    parser.add_argument("--pretrained_model_name_or_path", default=str(root / "models" / "sd-turbo"))
    parser.add_argument("--prompt_file", default=str(root / "data" / "prompts" / "pickapicv2_test_unique.txt"))
    parser.add_argument("--drifting_mae_path", default=str(root / "drifting" / "mae_latent_256_torch.pth"))
    parser.add_argument("--hpsv3_checkpoint_path", default=str(root / "models" / "HPSv3" / "HPSv3.safetensors"))
    parser.add_argument("--hpsv3_base_model_path", default=str(root / "models" / "Qwen2-VL-7B-Instruct"))
    parser.add_argument("--hpsv3_config_path", default=None)
    parser.add_argument("--output_dir", default=str(root / "analysis" / "profiling" / "hpsv3_wallclock_4gpu"))
    parser.add_argument("--batchsize_gen", type=int, default=24)
    parser.add_argument("--num_pos_images", type=int, default=12)
    parser.add_argument("--num_neg_images", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_updates", type=int, default=20)
    parser.add_argument("--eval_every_updates", type=int, default=2)
    parser.add_argument("--num_eval_prompts", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mixed_precision", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--vae_decode_chunk_size", type=int, default=1)
    parser.add_argument("--hpsv3_score_chunk_size", type=int, default=1)
    parser.add_argument("--drifting_pos_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_loss_weight", type=float, default=0.0)
    parser.add_argument("--draft_ref_model_l2_weight", type=float, default=0.0)
    parser.add_argument("--draft_compute_ref", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def rank_info() -> tuple[int, int, int]:
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def load_prompts(path: str) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}.")
    return prompts


def add_lora(unet, args):
    unet.requires_grad_(False)
    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(unet, lora_config)


def load_base(args, device: torch.device, dtype: torch.dtype):
    tokenizer, text_encoder, vae, unet, scheduler = load_sdturbo_components(args.pretrained_model_name_or_path)
    text_encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    unet.to(device=device, dtype=dtype)
    unet = add_lora(unet, args).train()
    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=args.learning_rate)
    return tokenizer, text_encoder, vae, unet, scheduler, optimizer


def load_ref_unet(args, device: torch.device, dtype: torch.dtype):
    _, _, _, ref_unet, scheduler = load_sdturbo_components(args.pretrained_model_name_or_path)
    ref_unet.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return ref_unet, scheduler


def encode_prompt(tokenizer, text_encoder, prompt: str, device: torch.device, repeat: int, timer: CudaTimer) -> torch.Tensor:
    with timer.measure("text_encode"):
        input_ids = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        embeds = text_encoder(input_ids)[0]
        return embeds.repeat_interleave(repeat, dim=0)


def run_unet(unet, latents, embeds, timer: CudaTimer, name: str):
    with timer.measure(name):
        timesteps = torch.full((latents.shape[0],), 999, device=latents.device, dtype=torch.long)
        model_pred = unet(latents, timesteps, embeds).sample
        clean = one_step_clean_latent(latents, model_pred)
    return clean


def train_drpo_microstep(
    args,
    prompt: str,
    tokenizer,
    text_encoder,
    vae,
    unet,
    ref_unet,
    reward,
    extractor,
    feature_keys,
    config: DrPOConfig,
    device: torch.device,
    dtype: torch.dtype,
    timer: CudaTimer,
    loss_scale: float,
) -> tuple[float, int, float]:
    embeds = encode_prompt(tokenizer, text_encoder, prompt, device, args.batchsize_gen, timer)
    latents = torch.randn((args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8), device=device, dtype=dtype)
    clean = run_unet(unet, latents, embeds, timer, "student_unet_generate")
    with torch.no_grad():
        ref_clean = run_unet(ref_unet, latents, embeds, timer, "reference_unet_generate")
        with timer.measure("vae_decode"):
            decoded = decode_latents_to_tensor(vae, clean.detach(), chunk_size=args.vae_decode_chunk_size)
        with timer.measure("hpsv3_score"):
            scores = reward.score_from_image_tensor(decoded, [prompt] * decoded.shape[0]).detach()
    with timer.measure("rank_select"):
        best_idx, worst_idx, feature_top_idx = select_disjoint_pref_indices(
            scores,
            num_pos=min(args.num_pos_images, args.batchsize_gen // 2),
            num_neg=min(args.num_neg_images, args.batchsize_gen // 2),
            feature_top_fraction=1.0,
        )
    with timer.measure("feature_extract"):
        generated_features = extractor.vector_features(clean.index_select(0, feature_top_idx), keys=feature_keys)
        reference_features = extractor.vector_features(ref_clean.index_select(0, feature_top_idx).detach(), keys=feature_keys)
        positive_features = extractor.vector_features(clean.index_select(0, best_idx).detach(), keys=feature_keys)
        negative_features = extractor.vector_features(clean.index_select(0, worst_idx).detach(), keys=feature_keys)
    with timer.measure("loss_build"):
        terms = _compute_prompt_terms(feature_keys, generated_features, reference_features, positive_features, negative_features, config)
        loss = terms["loss"]
    with timer.measure("backward"):
        (loss * loss_scale).backward()
    return float(scores.sum().item()), int(scores.numel()), float(loss.detach().item())


def train_draft_microstep(
    args,
    prompt: str,
    tokenizer,
    text_encoder,
    vae,
    unet,
    ref_unet,
    reward,
    device: torch.device,
    dtype: torch.dtype,
    timer: CudaTimer,
    loss_scale: float,
) -> tuple[float, int, float]:
    embeds = encode_prompt(tokenizer, text_encoder, prompt, device, 1, timer)
    score_sum = 0.0
    loss_sum = 0.0
    for _ in range(args.batchsize_gen):
        latents = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), device=device, dtype=dtype)
        clean = run_unet(unet, latents, embeds, timer, "student_unet_generate")
        if ref_unet is not None:
            with torch.no_grad():
                ref_clean = run_unet(ref_unet, latents, embeds, timer, "reference_unet_generate")
        with timer.measure("vae_decode"):
            decoded = decode_latents_to_tensor(vae, clean, chunk_size=args.vae_decode_chunk_size)
        with timer.measure("hpsv3_score"):
            score = reward.score_from_image_tensor(decoded, [prompt]).mean()
        with timer.measure("loss_build"):
            reward_loss = -score
            if ref_unet is not None and args.draft_ref_model_l2_weight:
                ref_l2 = F.mse_loss(clean.float(), ref_clean.float())
                loss = reward_loss + args.draft_ref_model_l2_weight * ref_l2
            else:
                loss = reward_loss
        with timer.measure("backward"):
            (loss * loss_scale / args.batchsize_gen).backward()
        score_sum += float(score.detach().item())
        loss_sum += float(loss.detach().item()) / args.batchsize_gen
    return score_sum, args.batchsize_gen, loss_sum


@torch.no_grad()
def evaluate_hpsv3(
    args,
    prompts: Sequence[str],
    tokenizer,
    text_encoder,
    vae,
    unet,
    reward,
    device: torch.device,
    dtype: torch.dtype,
    timer: CudaTimer,
) -> float:
    module = unet.module if isinstance(unet, DDP) else unet
    was_training = module.training
    module.eval()
    scores = []
    for prompt in prompts:
        with timer.measure("eval_generate"):
            embeds = encode_prompt(tokenizer, text_encoder, prompt, device, 1, timer)
            latents = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), device=device, dtype=dtype)
            clean = run_unet(module, latents, embeds, timer, "eval_generate")
            decoded = decode_latents_to_tensor(vae, clean, chunk_size=args.vae_decode_chunk_size)
        with timer.measure("eval_hpsv3_score"):
            score = reward.score_from_image_tensor(decoded, [prompt]).mean()
        scores.append(score.detach())
    if was_training:
        module.train()
    return float(torch.stack(scores).mean().item())


def reduce_update_metrics(device: torch.device, timer: CudaTimer, score_sum: float, score_count: int, loss_sum: float, loss_count: int):
    metric = torch.tensor([score_sum, float(score_count), loss_sum, float(loss_count)], device=device, dtype=torch.float64)
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    comp_sum = timer.vector(device)
    comp_mean = comp_sum.clone()
    comp_max = comp_sum.clone()
    dist.all_reduce(comp_mean, op=dist.ReduceOp.SUM)
    comp_mean /= dist.get_world_size()
    dist.all_reduce(comp_max, op=dist.ReduceOp.MAX)
    train_hpsv3 = metric[0].item() / max(metric[1].item(), 1.0)
    loss_mean = metric[2].item() / max(metric[3].item(), 1.0)
    return train_hpsv3, loss_mean, comp_mean.detach().cpu().tolist(), comp_max.detach().cpu().tolist()


def write_outputs(args, rows: list[dict], summary: dict) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_hpsv3_4gpu_k{args.batchsize_gen}_acc{args.gradient_accumulation_steps}_updates{args.max_updates}"
    csv_path = out_dir / f"{stem}_curve.csv"
    json_path = out_dir / f"{stem}_summary.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path), **summary}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, local_rank, world_size = rank_info()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    torch.manual_seed(args.seed + rank)

    prompts = load_prompts(args.prompt_file)
    eval_prompts = prompts[: args.num_eval_prompts]
    tokenizer, text_encoder, vae, unet, _, optimizer = load_base(args, device, dtype)
    ref_unet = None
    if args.method == "drpo" or args.draft_compute_ref:
        ref_unet, _ = load_ref_unet(args, device, dtype)

    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    trainable_parameters = [p for p in unet.parameters() if p.requires_grad]

    install_hpsv3_checkpoint_key_patch()
    reward = DifferentiableHPSv3Scorer(
        device=device,
        checkpoint_path=args.hpsv3_checkpoint_path,
        base_model_path=args.hpsv3_base_model_path,
        config_path=args.hpsv3_config_path,
        differentiable=True,
        gradient_checkpointing=args.method == "draft",
        score_chunk_size=args.hpsv3_score_chunk_size,
    )

    extractor = None
    feature_keys = ("layer4_mean", "layer4_std", "layer4_mean_2", "layer4_std_2", "layer4_mean_4", "layer4_std_4")
    config = None
    if args.method == "drpo":
        extractor = FrozenMAELatentFeatureExtractor(
            args.drifting_mae_path,
            feature_key="layer4_mean",
            block_stride=2,
            patch_pool_sizes=(2, 4),
            include_input_sq_mean=False,
            include_spatial_features=False,
        ).to(device=device, dtype=torch.float32).eval()
        config = DrPOConfig(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            output_dir=args.output_dir,
            train_mode="online",
            pairs_jsonl="",
            prompt_file=args.prompt_file,
            drifting_mae_path=args.drifting_mae_path,
            batchsize_gen=args.batchsize_gen,
            num_pos_images=args.num_pos_images,
            num_neg_images=args.num_neg_images,
            drifting_pos_weight=args.drifting_pos_weight,
            drifting_neg_weight=args.drifting_neg_weight,
            drifting_ref_weight=0.0,
            drifting_ref_loss_weight=args.drifting_ref_loss_weight,
            frozen_feature_l2_weight=0.0,
            mixed_precision=args.mixed_precision,
        )

    dist.barrier()
    train_start = time.perf_counter()
    rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats(device)

    for update in range(1, args.max_updates + 1):
        dist.barrier()
        update_start = time.perf_counter()
        timer = CudaTimer()
        score_sum = 0.0
        score_count = 0
        loss_sum = 0.0
        loss_count = 0
        optimizer.zero_grad(set_to_none=True)
        for accum in range(args.gradient_accumulation_steps):
            prompt_index = ((update - 1) * args.gradient_accumulation_steps * world_size + accum * world_size + rank) % len(prompts)
            prompt = prompts[prompt_index]
            sync_context = nullcontext() if accum == args.gradient_accumulation_steps - 1 else unet.no_sync()
            with sync_context:
                if args.method == "drpo":
                    s_sum, s_count, loss_value = train_drpo_microstep(
                        args,
                        prompt,
                        tokenizer,
                        text_encoder,
                        vae,
                        unet,
                        ref_unet,
                        reward,
                        extractor,
                        feature_keys,
                        config,
                        device,
                        dtype,
                        timer,
                        1.0 / args.gradient_accumulation_steps,
                    )
                else:
                    s_sum, s_count, loss_value = train_draft_microstep(
                        args,
                        prompt,
                        tokenizer,
                        text_encoder,
                        vae,
                        unet,
                        ref_unet,
                        reward,
                        device,
                        dtype,
                        timer,
                        1.0 / args.gradient_accumulation_steps,
                    )
            score_sum += s_sum
            score_count += s_count
            loss_sum += loss_value
            loss_count += 1

        with timer.measure("optimizer_step"):
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_hpsv3, loss_mean, comp_mean, comp_max = reduce_update_metrics(device, timer, score_sum, score_count, loss_sum, loss_count)
        dist.barrier()
        update_wall_sec = time.perf_counter() - update_start

        eval_hpsv3 = None
        eval_wall_sec = 0.0
        if args.eval_every_updates > 0 and update % args.eval_every_updates == 0:
            dist.barrier()
            eval_start = time.perf_counter()
            if rank == 0:
                eval_timer = CudaTimer()
                eval_hpsv3 = evaluate_hpsv3(args, eval_prompts, tokenizer, text_encoder, vae, unet, reward, device, dtype, eval_timer)
                eval_vec = eval_timer.vector(device).detach().cpu().tolist()
            else:
                eval_vec = [0.0 for _ in COMPONENTS]
            dist.barrier()
            eval_wall_sec = time.perf_counter() - eval_start
        else:
            eval_vec = [0.0 for _ in COMPONENTS]

        elapsed_sec = time.perf_counter() - train_start
        if rank == 0:
            row = {
                "method": args.method,
                "update": update,
                "world_size": world_size,
                "batchsize_gen": args.batchsize_gen,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "global_prompt_batch_per_update": world_size * args.gradient_accumulation_steps,
                "global_candidate_budget_per_update": world_size * args.gradient_accumulation_steps * args.batchsize_gen,
                "elapsed_sec": elapsed_sec,
                "update_wall_sec": update_wall_sec,
                "eval_wall_sec": eval_wall_sec,
                "train_hpsv3_mean": train_hpsv3,
                "eval_hpsv3_mean": eval_hpsv3,
                "loss_mean": loss_mean,
            }
            for name, value in zip(COMPONENTS, comp_mean):
                row[f"mean_{name}_sec"] = value
            for name, value in zip(COMPONENTS, comp_max):
                row[f"max_{name}_sec"] = value
            for name, value in zip(COMPONENTS, eval_vec):
                if value:
                    row[f"rank0_{name}_sec"] = value
            rows.append(row)
            print(json.dumps(row), flush=True)

    peak = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device) / (1024**3),
            torch.cuda.max_memory_reserved(device) / (1024**3),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        avg_update_wall = sum(row["update_wall_sec"] for row in rows) / max(len(rows), 1)
        avg_components = {
            name: sum(row.get(f"mean_{name}_sec", 0.0) for row in rows) / max(len(rows), 1)
            for name in COMPONENTS
        }
        summary = {
            "method": args.method,
            "world_size": world_size,
            "batchsize_gen": args.batchsize_gen,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "global_prompt_batch_per_update": world_size * args.gradient_accumulation_steps,
            "global_candidate_budget_per_update": world_size * args.gradient_accumulation_steps * args.batchsize_gen,
            "max_updates": args.max_updates,
            "avg_update_wall_sec": avg_update_wall,
            "avg_components_sec": avg_components,
            "peak_cuda_memory_allocated_gb": peak[0].item(),
            "peak_cuda_memory_reserved_gb": peak[1].item(),
            "notes": "4-GPU DDP run. Wall-clock starts after model loading and includes periodic eval barriers. Train HPSv3 is the mean HPSv3 score over the training candidates consumed by each update; eval HPSv3 is rank-0 fixed-prompt one-image evaluation.",
        }
        write_outputs(args, rows, summary)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
