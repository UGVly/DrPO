#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# H200's shared mydiffusers env currently has an xformers build that is
# ABI-incompatible with its PyTorch. Diffusers only needs the availability flag
# during import here, so disable xformers before importing SD-Turbo helpers.
try:
    import diffusers.utils.import_utils as diffusers_import_utils

    diffusers_import_utils._xformers_available = False
except Exception:
    pass

from drpo.config import DrPOConfig
from drpo.features import FrozenMAELatentFeatureExtractor
from drpo.paths import project_root, require_local_path
from drpo.rewards import HPSv3Selector
from drpo.sdturbo import decode_latents_to_pil, decode_latents_to_tensor, load_sdturbo_components, one_step_clean_latent
from drpo.training.trainer import _compute_prompt_terms
from drpo.utils.tensors import normalize_prompts, select_disjoint_pref_indices


def install_hpsv3_checkpoint_key_patch() -> None:
    """Adapt HPSv3 checkpoint keys for newer Transformers Qwen2-VL modules."""
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


class CudaTimer:
    def __init__(self) -> None:
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

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
                timer.counts[name] += 1

        return _Measure()

    def averages(self, steps: int) -> dict[str, float]:
        return {name: value / max(steps, 1) for name, value in sorted(self.totals.items())}


class DifferentiableHPSv3Scorer(nn.Module):
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str | os.PathLike[str],
        base_model_path: str | os.PathLike[str],
        config_path: str | os.PathLike[str] | None = None,
        gradient_checkpointing: bool = True,
        score_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.score_chunk_size = max(int(score_chunk_size), 1)

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
            differentiable=True,
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
        config["output_dir"] = str(project_root() / "outputs" / "hpsv3-profile")
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="profile_hpsv3_", delete=False, encoding="utf-8")
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
    parser.add_argument("--output_dir", default=str(root / "analysis" / "profiling" / "hpsv3_speed"))
    parser.add_argument("--batchsize_gen", type=int, default=24)
    parser.add_argument("--num_pos_images", type=int, default=12)
    parser.add_argument("--num_neg_images", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
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
    parser.add_argument("--drpo_hpsv3_backend", choices=["tensor", "selector"], default="tensor")
    return parser.parse_args()


def load_prompts(path: str, count: int) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < count:
        raise ValueError(f"Need at least {count} prompts in {path}, found {len(prompts)}.")
    return prompts[:count]


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


def profile_drpo(args, prompts: Sequence[str], device: torch.device, dtype: torch.dtype) -> dict:
    tokenizer, text_encoder, vae, unet, _, optimizer = load_base(args, device, dtype)
    ref_unet, _ = load_ref_unet(args, device, dtype)
    install_hpsv3_checkpoint_key_patch()
    if args.drpo_hpsv3_backend == "selector":
        selector = HPSv3Selector(
            device=device,
            checkpoint_path=args.hpsv3_checkpoint_path,
            base_model_path=args.hpsv3_base_model_path,
            config_path=args.hpsv3_config_path,
        )
    else:
        selector = DifferentiableHPSv3Scorer(
            device=device,
            checkpoint_path=args.hpsv3_checkpoint_path,
            base_model_path=args.hpsv3_base_model_path,
            config_path=args.hpsv3_config_path,
            gradient_checkpointing=False,
            score_chunk_size=args.hpsv3_score_chunk_size,
        )
    extractor = FrozenMAELatentFeatureExtractor(
        args.drifting_mae_path,
        feature_key="layer4_mean",
        block_stride=2,
        patch_pool_sizes=(2, 4),
        include_input_sq_mean=False,
        include_spatial_features=False,
    ).to(device=device, dtype=torch.float32).eval()
    feature_keys = ("layer4_mean", "layer4_std", "layer4_mean_2", "layer4_std_2", "layer4_mean_4", "layer4_std_4")
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
    timer = CudaTimer()
    rows = []
    measured = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step, prompt in enumerate(prompts):
        is_measured = step >= args.warmup_steps
        if not is_measured:
            local_timer = CudaTimer()
        else:
            local_timer = timer
            measured += 1
        total_start = time.perf_counter()
        embeds = encode_prompt(tokenizer, text_encoder, prompt, device, args.batchsize_gen, local_timer)
        latents = torch.randn((args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8), device=device, dtype=dtype)
        clean = run_unet(unet, latents, embeds, local_timer, "student_unet_generate")
        with torch.no_grad():
            ref_clean = run_unet(ref_unet, latents, embeds, local_timer, "reference_unet_generate")
        if args.drpo_hpsv3_backend == "selector":
            with local_timer.measure("vae_decode"):
                images = decode_latents_to_pil(vae, clean.detach(), chunk_size=args.vae_decode_chunk_size)
            with local_timer.measure("hpsv3_score"):
                scores = torch.tensor(selector.score(images, [prompt] * len(images)), device=device, dtype=torch.float32)
        else:
            with torch.no_grad():
                with local_timer.measure("vae_decode"):
                    decoded = decode_latents_to_tensor(vae, clean.detach(), chunk_size=args.vae_decode_chunk_size)
                with local_timer.measure("hpsv3_score"):
                    scores = selector.score_from_image_tensor(decoded, [prompt] * decoded.shape[0]).detach()
        with local_timer.measure("rank_select"):
            best_idx, worst_idx, feature_top_idx = select_disjoint_pref_indices(
                scores,
                num_pos=min(args.num_pos_images, args.batchsize_gen // 2),
                num_neg=min(args.num_neg_images, args.batchsize_gen // 2),
                feature_top_fraction=1.0,
            )
        with local_timer.measure("feature_extract"):
            generated_features = extractor.vector_features(clean.index_select(0, feature_top_idx), keys=feature_keys)
            reference_features = extractor.vector_features(ref_clean.index_select(0, feature_top_idx).detach(), keys=feature_keys)
            positive_features = extractor.vector_features(clean.index_select(0, best_idx).detach(), keys=feature_keys)
            negative_features = extractor.vector_features(clean.index_select(0, worst_idx).detach(), keys=feature_keys)
        with local_timer.measure("loss_build"):
            terms = _compute_prompt_terms(feature_keys, generated_features, reference_features, positive_features, negative_features, config)
            loss = terms["loss"]
        with local_timer.measure("backward"):
            loss.backward()
        with local_timer.measure("optimizer_step"):
            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if is_measured:
            timer.sync()
            rows.append({"step": measured, "total": time.perf_counter() - total_start, "reward_mean": float(scores.mean().item())})
    return finish_result(args, "drpo", timer, rows, measured, device)


def profile_draft(args, prompts: Sequence[str], device: torch.device, dtype: torch.dtype) -> dict:
    tokenizer, text_encoder, vae, unet, _, optimizer = load_base(args, device, dtype)
    ref_unet = None
    if args.draft_compute_ref:
        ref_unet, _ = load_ref_unet(args, device, dtype)
    install_hpsv3_checkpoint_key_patch()
    reward = DifferentiableHPSv3Scorer(
        device=device,
        checkpoint_path=args.hpsv3_checkpoint_path,
        base_model_path=args.hpsv3_base_model_path,
        config_path=args.hpsv3_config_path,
        gradient_checkpointing=True,
        score_chunk_size=args.hpsv3_score_chunk_size,
    )
    timer = CudaTimer()
    rows = []
    measured = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step, prompt in enumerate(prompts):
        is_measured = step >= args.warmup_steps
        local_timer = timer if is_measured else CudaTimer()
        if is_measured:
            measured += 1
        total_start = time.perf_counter()
        embeds = encode_prompt(tokenizer, text_encoder, prompt, device, 1, local_timer)
        prompt_scores = []
        prompt_loss = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(args.batchsize_gen):
            latents = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), device=device, dtype=dtype)
            clean = run_unet(unet, latents, embeds, local_timer, "student_unet_generate")
            if ref_unet is not None:
                with torch.no_grad():
                    ref_clean = run_unet(ref_unet, latents, embeds, local_timer, "reference_unet_generate")
            with local_timer.measure("vae_decode"):
                decoded = decode_latents_to_tensor(vae, clean, chunk_size=args.vae_decode_chunk_size)
            with local_timer.measure("hpsv3_score"):
                score = reward.score_from_image_tensor(decoded, [prompt]).mean()
            with local_timer.measure("loss_build"):
                reward_loss = -score
                if ref_unet is not None and args.draft_ref_model_l2_weight:
                    ref_l2 = F.mse_loss(clean.float(), ref_clean.float())
                    loss = reward_loss + args.draft_ref_model_l2_weight * ref_l2
                else:
                    loss = reward_loss
            with local_timer.measure("backward"):
                (loss / args.batchsize_gen).backward()
            prompt_loss = prompt_loss + loss.detach() / args.batchsize_gen
            prompt_scores.append(score.detach())
        with local_timer.measure("optimizer_step"):
            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if is_measured:
            timer.sync()
            rows.append({
                "step": measured,
                "total": time.perf_counter() - total_start,
                "reward_mean": float(torch.stack(prompt_scores).mean().item()),
                "loss": float(prompt_loss.item()),
            })
    return finish_result(args, "draft", timer, rows, measured, device)


def finish_result(args, method: str, timer: CudaTimer, rows: list[dict], measured: int, device: torch.device) -> dict:
    avg_components = timer.averages(measured)
    avg_total = sum(row["total"] for row in rows) / max(len(rows), 1)
    component_sum = sum(avg_components.values())
    result = {
        "method": method,
        "batchsize_gen": args.batchsize_gen,
        "drpo_hpsv3_backend": getattr(args, "drpo_hpsv3_backend", None) if method == "drpo" else None,
        "warmup_steps": args.warmup_steps,
        "measured_steps": measured,
        "avg_total_sec": avg_total,
        "avg_components_sec": avg_components,
        "avg_component_sum_sec": component_sum,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
        "rows": rows,
        "notes": "Load time excluded. DRaFT streams K HPSv3 candidates as in the strict HPSv3 trainer; DrPO batches K candidates. DrPO tensor backend uses the same in-memory HPSv3 tensor path without reward gradients; selector backend uses the repo PIL/tempfile selector path.",
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_hpsv3_k{args.batchsize_gen}_steps{measured}"
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / f"{stem}_steps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    prompts = load_prompts(args.prompt_file, args.warmup_steps + args.steps)
    if args.method == "drpo":
        profile_drpo(args, prompts, device, dtype)
    else:
        profile_draft(args, prompts, device, dtype)


if __name__ == "__main__":
    main()
