# coding=utf-8
"""Self-contained helpers for baseline trainers.

The baselines package intentionally does not import project-internal training modules.
Shared utilities that the historical baseline scripts used live here instead, so
each baseline remains runnable as an independent implementation.
"""

import argparse
import json
import logging
import math
import os
import random
import shlex
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)


SD_TURBO_TIMESTEP = 999


def one_step_clean_latent(noisy_latent: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
    """SD-Turbo one-step latent conversion used by the original training code."""
    return ((noisy_latent - 0.9977 * noise_pred) / 0.0683) * 0.9996 + 0.0292 * noise_pred


def is_default_sdturbo_projection(timestep: int, target_timestep: int) -> bool:
    return int(timestep) == SD_TURBO_TIMESTEP and int(target_timestep) in {-1, 0}


def decode_latents_to_tensor(vae: AutoencoderKL, latents: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    scaled_latents = (latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)
    decoded_chunks = [vae.decode(chunk).sample for chunk in scaled_latents.split(max(1, int(chunk_size)))]
    return torch.cat(decoded_chunks, dim=0).float().clamp(-1, 1)

from utils.reward_common import select_top_fraction_indices


check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


def parse_float_list(list_str: str) -> Tuple[float, ...]:
    values = [v.strip() for v in list_str.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return tuple(float(v) for v in values)


def parse_int_list(list_str: str) -> Tuple[int, ...]:
    values = [v.strip() for v in list_str.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return tuple(int(v) for v in values)


def parse_name_list(list_str: str) -> Tuple[str, ...]:
    values = [v.strip() for v in list_str.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one name.")
    return tuple(values)


def load_prompt_file(prompt_file: str, max_prompts: int = None) -> List[str]:
    prompts: List[str] = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    if not prompts:
        raise ValueError(f"No valid prompts found in: {prompt_file}")
    return prompts


def resolve_weight_file(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Weight file or directory not found: {path}")

    if (path / "unet").is_dir():
        path = path / "unet"

    preferred_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    )
    for name in preferred_names:
        candidate = path / name
        if candidate.is_file():
            return candidate

    candidates = sorted(path.glob("*.safetensors"))
    if not candidates:
        candidates = sorted(path.glob("*.bin"))
    if not candidates:
        candidates = sorted(path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No weight file found under {path}")
    if len(candidates) > 1:
        raise ValueError(f"Found multiple candidate weight files under {path}; pass a specific file instead.")
    return candidates[0]


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        raise ValueError("Empty state dict.")

    prefixes = ("unet.", "model.diffusion_model.")
    keys = list(state_dict.keys())
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in keys):
            return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _compute_lora_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    out_shape: torch.Size,
) -> torch.Tensor:
    if lora_a.ndim == 2 and lora_b.ndim == 2:
        return lora_b.float() @ lora_a.float()
    if lora_a.ndim == 4 and lora_b.ndim == 4:
        if lora_a.shape[2:] == (1, 1) and lora_b.shape[2:] == (1, 1):
            a_matrix = lora_a.float().reshape(lora_a.shape[0], lora_a.shape[1])
            b_matrix = lora_b.float().reshape(lora_b.shape[0], lora_b.shape[1])
            return (b_matrix @ a_matrix).reshape(out_shape)
        if lora_b.shape[2:] == (1, 1):
            return torch.einsum("orxy,rihw->oihw", lora_b.float(), lora_a.float())
        if lora_a.shape[2:] == (1, 1):
            return torch.einsum("orhw,rixy->oihw", lora_b.float(), lora_a.float())
    raise ValueError(
        f"Unsupported LoRA weight shapes for merge: A={tuple(lora_a.shape)} B={tuple(lora_b.shape)} out={tuple(out_shape)}"
    )


def merge_base_layer_lora_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(".base_layer." in key for key in state_dict):
        return state_dict

    merged_state_dict: Dict[str, torch.Tensor] = {}
    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        if ".base_layer." in key:
            merged_state_dict[key.replace(".base_layer.", ".")] = value
            continue
        if ".lora_A." in key:
            prefix = key.split(".lora_A.", 1)[0]
            lora_pairs.setdefault(prefix, {})["A"] = value
            continue
        if ".lora_B." in key:
            prefix = key.split(".lora_B.", 1)[0]
            lora_pairs.setdefault(prefix, {})["B"] = value
            continue
        merged_state_dict[key] = value

    for prefix, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA pair for {prefix}; found keys={sorted(pair)}")
        weight_key = f"{prefix}.weight"
        if weight_key not in merged_state_dict:
            raise KeyError(f"Missing base weight for merged LoRA key: {weight_key}")
        base_weight = merged_state_dict[weight_key]
        delta = _compute_lora_delta(pair["A"], pair["B"], out_shape=base_weight.shape).to(dtype=base_weight.dtype)
        merged_state_dict[weight_key] = base_weight + delta

    return merged_state_dict


def load_state_dict_file(weight_path: Path) -> Dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        state_dict = load_safetensors_file(str(weight_path), device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict from {weight_path}, got {type(state_dict)}")
    state_dict = normalize_state_dict_keys(state_dict)
    return merge_base_layer_lora_weights(state_dict)


def load_unet_checkpoint_weights(unet: UNet2DConditionModel, checkpoint_path: str) -> Path:
    weight_path = resolve_weight_file(checkpoint_path)
    state_dict = load_state_dict_file(weight_path)
    incompatible_keys = unet.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        raise ValueError(
            "UNet checkpoint does not match the base model.\n"
            f"path={weight_path}\n"
            f"missing_keys={incompatible_keys.missing_keys[:8]}\n"
            f"unexpected_keys={incompatible_keys.unexpected_keys[:8]}"
        )
    return weight_path


def choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return max(groups, 1)


def safe_std_torch(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps).to(dtype=x.dtype)


def enforce_zero_terminal_snr(scheduler: DDPMScheduler):
    alphas = 1 - scheduler.betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_t
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0)


def _scheduler_alpha_prod(
    scheduler: DDPMScheduler,
    timestep: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    return scheduler.alphas_cumprod[int(timestep)].to(dtype=dtype)


def project_model_output_to_target_timestep(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    scheduler: DDPMScheduler,
    *,
    timestep: int,
    target_timestep: int,
) -> torch.Tensor:
    if is_default_sdturbo_projection(timestep, target_timestep):
        return one_step_clean_latent(sample, model_output)

    alpha_prod_t = _scheduler_alpha_prod(
        scheduler,
        timestep,
        device=sample.device,
        dtype=sample.dtype,
    )
    beta_prod_t = 1 - alpha_prod_t

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    else:
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")

    if target_timestep < 0:
        return pred_original_sample

    alpha_prod_target = _scheduler_alpha_prod(
        scheduler,
        target_timestep,
        device=sample.device,
        dtype=sample.dtype,
    )
    beta_prod_target = 1 - alpha_prod_target
    return alpha_prod_target.sqrt() * pred_original_sample + beta_prod_target.sqrt() * pred_epsilon


def run_one_step_unet(
    unet,
    noisy_latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    scheduler: DDPMScheduler,
    *,
    timestep: int,
    target_timestep: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.full(
        (noisy_latents.shape[0],),
        int(timestep),
        device=noisy_latents.device,
        dtype=torch.long,
    )
    model_output = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    projected_latents = project_model_output_to_target_timestep(
        sample=noisy_latents,
        model_output=model_output,
        scheduler=scheduler,
        timestep=timestep,
        target_timestep=target_timestep,
    )
    return model_output, projected_latents


class PairsPromptDataset(Dataset):
    """
    train_mode == offline: yields prompt + chosen/rejected image tensors.
    train_mode == offline_distance: yields prompt + chosen/rejected image tensors,
        but uses them as feature-distance references for pseudo-online ranking.
    train_mode == online: yields prompt only.
    """

    def __init__(
        self,
        pairs_jsonl: str,
        tokenizer: CLIPTokenizer,
        train_mode: str,
        image_size: int,
        max_train_samples: int = None,
        seed: int = None,
        proportion_empty_prompts: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        self.proportion_empty_prompts = proportion_empty_prompts
        self.base_dir = os.path.dirname(os.path.abspath(pairs_jsonl))
        self.rows: List[Dict[str, str]] = []
        with open(pairs_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
        if max_train_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(self.rows)
            self.rows = self.rows[:max_train_samples]

        self.empty_input_ids = tokenizer(
            [""],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def _read_rgb(self, rel_path: str) -> Image.Image:
        path = rel_path if os.path.isabs(rel_path) else os.path.join(self.base_dir, rel_path)
        with Image.open(path) as img:
            return img.convert("RGB")

    def _normalize_image_list(self, value) -> List[str]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise TypeError(f"Expected image path or list of paths, got {type(value)}")
        items = [item for item in items if item]
        if not items:
            raise ValueError("Expected at least one image path.")
        return items

    def __getitem__(self, idx):
        rec = self.rows[idx]
        prompt = rec.get("prompt", "")
        if random.random() < self.proportion_empty_prompts:
            input_ids = self.empty_input_ids.clone()
        else:
            input_ids = self.tokenizer(
                [prompt],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        out = {"prompt": prompt, "input_ids": input_ids}
        if self.train_mode in {"offline", "offline_distance"}:
            chosen = torch.stack(
                [self.train_transforms(self._read_rgb(path)) for path in self._normalize_image_list(rec["chosen"])]
            )
            rejected = torch.stack(
                [self.train_transforms(self._read_rgb(path)) for path in self._normalize_image_list(rec["rejected"])]
            )
            out["chosen"] = chosen
            out["rejected"] = rejected
        return out


def collate_fn(examples):
    batch = {
        "prompt": [e["prompt"] for e in examples],
        "input_ids": torch.stack([e["input_ids"] for e in examples]),
    }
    if "chosen" in examples[0]:
        batch["chosen"] = [e["chosen"].float() for e in examples]
        batch["rejected"] = [e["rejected"].float() for e in examples]
    return batch


def _load_mae_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
        metadata = ckpt.get("metadata", {})
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state_dict = ckpt["model_state_dict"]
        metadata = ckpt.get("metadata", {})
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
        metadata = {}
    else:
        raise ValueError(f"Invalid MAE checkpoint format: {checkpoint_path}")

    normalized_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "_orig_mod."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict, metadata


class FrozenMAELatentFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        feature_key: str = "layer4_mean",
        block_stride: int = 2,
        patch_pool_sizes: Sequence[int] = (2, 4),
        include_input_sq_mean: bool = False,
        include_spatial_features: bool = False,
        include_mean_features: bool = True,
        include_std_features: bool = True,
        patch_mean_sizes: Sequence[int] | None = None,
        patch_std_sizes: Sequence[int] | None = None,
    ):
        super().__init__()
        self.params, meta = _load_mae_checkpoint(checkpoint_path)
        cfg = meta.get("model_config", {})
        self.layers: Sequence[int] = cfg.get("layers", [3, 4, 6, 3])
        conv1_weight = self.params.get("encoder.conv1.kernel")
        if conv1_weight is not None:
            inferred_in_channels = int(conv1_weight.shape[2])
        else:
            conv1_weight = self.params.get("encoder.conv1.weight")
            if conv1_weight is None:
                raise KeyError("Missing encoder.conv1.kernel/weight in MAE checkpoint")
            inferred_in_channels = int(conv1_weight.shape[1])
        self.in_channels = int(cfg.get("in_channels", inferred_in_channels))
        self.feature_key = feature_key
        self.block_stride = max(int(block_stride), 0)
        patch_pool_sizes = tuple(sorted({int(size) for size in patch_pool_sizes if int(size) > 1}))
        self.patch_mean_sizes = tuple(sorted({int(size) for size in (patch_mean_sizes or patch_pool_sizes) if int(size) > 1}))
        self.patch_std_sizes = tuple(sorted({int(size) for size in (patch_std_sizes or patch_pool_sizes) if int(size) > 1}))
        self.patch_pool_sizes = tuple(sorted(set(self.patch_mean_sizes) | set(self.patch_std_sizes)))
        self.include_input_sq_mean = include_input_sq_mean
        self.include_spatial_features = include_spatial_features
        self.include_mean_features = include_mean_features
        self.include_std_features = include_std_features
        self.available_feature_keys = {"conv1", "conv1_mean", "conv1_std"}
        self.available_vector_feature_keys = set()
        self._register_feature_name("conv1")
        for stage_idx, num_blocks in enumerate(self.layers):
            layer_name = f"layer{stage_idx + 1}"
            if self.block_stride > 0:
                for block_num in range(self.block_stride, num_blocks, self.block_stride):
                    self._register_feature_name(f"{layer_name}_blk{block_num}")
            self._register_feature_name(layer_name)
        if self.include_input_sq_mean:
            self.available_vector_feature_keys.add("input_x2_mean")
            self.available_vector_feature_keys.add("norm_x")
        if self.feature_key not in self.available_vector_feature_keys:
            available = ", ".join(sorted(self.available_vector_feature_keys))
            raise ValueError(f"Unknown drifting feature key: {self.feature_key}. Available keys: {available}")
        self.requires_grad_(False)

    def _register_feature_name(self, name: str) -> None:
        self.available_feature_keys.add(name)
        self.available_feature_keys.add(f"{name}_mean")
        self.available_feature_keys.add(f"{name}_std")
        self.available_vector_feature_keys.add(name)
        self.available_vector_feature_keys.add(f"{name}_mean")
        self.available_vector_feature_keys.add(f"{name}_std")
        for patch_size in self.patch_mean_sizes:
            self.available_vector_feature_keys.add(f"{name}_mean_{patch_size}")
            self.available_vector_feature_keys.add(f"{name}_patch{patch_size}_mean")
        for patch_size in self.patch_std_sizes:
            self.available_vector_feature_keys.add(f"{name}_std_{patch_size}")
            self.available_vector_feature_keys.add(f"{name}_patch{patch_size}_std")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.params = {k: v.to(*args, **kwargs) for k, v in self.params.items()}
        return self

    def _conv2d(self, x: torch.Tensor, prefix: str, stride: int = 1, padding: int = 1):
        kernel = self.params.get(f"{prefix}.kernel")
        if kernel is None:
            kernel = self.params.get(f"{prefix}.weight")
        if kernel is None:
            raise KeyError(f"Missing conv weights for {prefix}")
        if kernel.ndim != 4:
            raise ValueError(f"Expected 4D conv kernel for {prefix}, got shape={tuple(kernel.shape)}")
        if kernel.shape[2] == x.shape[1]:
            w = kernel.permute(3, 2, 0, 1).contiguous()
        elif kernel.shape[1] == x.shape[1]:
            w = kernel.contiguous()
        else:
            raise ValueError(
                f"Cannot infer conv layout for {prefix}: kernel_shape={tuple(kernel.shape)}, input_channels={x.shape[1]}"
            )
        return F.conv2d(x, w, bias=None, stride=stride, padding=padding)

    def _gn(self, x: torch.Tensor, prefix: str):
        weight = self.params.get(f"{prefix}.scale")
        if weight is None:
            weight = self.params.get(f"{prefix}.weight")
        bias = self.params.get(f"{prefix}.bias")
        if weight is None or bias is None:
            raise KeyError(f"Missing group norm weights for {prefix}")
        return F.group_norm(x, choose_gn_groups(x.shape[1]), weight, bias, eps=1e-5)

    def _basic_block(self, x: torch.Tensor, prefix: str, stride: int):
        residual = x
        y = self._conv2d(x, f"{prefix}.conv1", stride=stride, padding=1)
        y = self._gn(y, f"{prefix}.gn1")
        y = F.relu(y, inplace=False)
        y = self._conv2d(y, f"{prefix}.conv2", stride=1, padding=1)
        y = self._gn(y, f"{prefix}.gn2")
        if residual.shape != y.shape:
            residual = self._conv2d(residual, f"{prefix}.proj_conv", stride=stride, padding=0)
            residual = self._gn(residual, f"{prefix}.proj_gn")
        return F.relu(residual + y, inplace=False)

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        x = self._conv2d(x, "encoder.conv1", stride=1, padding=1)
        x = self._gn(x, "encoder.gn1")
        x = F.relu(x, inplace=False)
        feats["conv1"] = x
        for s, nb in enumerate(self.layers):
            for b in range(nb):
                stride = 2 if (s > 0 and b == 0) else 1
                x = self._basic_block(x, f"encoder.stages_{s}.layers_{b}", stride=stride)
                block_num = b + 1
                if self.block_stride > 0 and block_num % self.block_stride == 0 and block_num < nb:
                    feats[f"layer{s+1}_blk{block_num}"] = x
            name = f"layer{s+1}"
            x = self._gn(x, f"encoder.{name}_norm")
            feats[name] = x
        return feats

    def get_vector_features(self, latents: torch.Tensor, feature_keys: Sequence[str] = None) -> Dict[str, torch.Tensor]:
        if latents.ndim != 4:
            raise ValueError(f"Expected latents with shape (B, C, H, W), got {tuple(latents.shape)}")
        if latents.shape[1] != self.in_channels:
            raise ValueError(
                f"Latent channel mismatch: expected {self.in_channels} channels from checkpoint, got {latents.shape[1]}"
            )
        conv1_kernel = self.params.get("encoder.conv1.kernel")
        if conv1_kernel is None:
            conv1_kernel = self.params["encoder.conv1.weight"]
        dtype = conv1_kernel.dtype
        if latents.dtype != dtype:
            latents = latents.to(dtype=dtype)

        feats = self.get_features(latents)
        vector_features: Dict[str, torch.Tensor] = {}

        if self.include_input_sq_mean:
            norm_x = torch.sqrt(torch.clamp(latents.float().pow(2).mean(dim=(2, 3), keepdim=False), min=0.0) + 1e-6)
            norm_x = norm_x.to(dtype=latents.dtype).unsqueeze(1)
            vector_features["norm_x"] = norm_x
            vector_features["input_x2_mean"] = latents.square().mean(dim=(2, 3), keepdim=False).unsqueeze(1)

        for name, feat in feats.items():
            b, c, h, w = feat.shape
            if self.include_spatial_features:
                spatial_vectors = feat.flatten(2).transpose(1, 2).contiguous()
                vector_features[name] = spatial_vectors
            if self.include_mean_features:
                vector_features[f"{name}_mean"] = feat.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
            if self.include_std_features:
                vector_features[f"{name}_std"] = safe_std_torch(feat, dim=(2, 3), keepdim=False).unsqueeze(1)
            for patch_size in self.patch_mean_sizes:
                if h % patch_size != 0 or w % patch_size != 0:
                    continue
                reshaped = feat.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                reshaped = reshaped.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, patch_size * patch_size, c)
                patch_mean = reshaped.mean(dim=2)
                vector_features[f"{name}_mean_{patch_size}"] = patch_mean
                vector_features[f"{name}_patch{patch_size}_mean"] = patch_mean
            for patch_size in self.patch_std_sizes:
                if h % patch_size != 0 or w % patch_size != 0:
                    continue
                reshaped = feat.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                reshaped = reshaped.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, patch_size * patch_size, c)
                patch_std = safe_std_torch(reshaped, dim=2, keepdim=False)
                vector_features[f"{name}_std_{patch_size}"] = patch_std
                vector_features[f"{name}_patch{patch_size}_std"] = patch_std

        if feature_keys is None:
            return vector_features

        missing_keys = [key for key in feature_keys if key not in vector_features]
        if missing_keys:
            available = ", ".join(sorted(vector_features.keys()))
            raise ValueError(f"Requested drifting feature keys not available: {missing_keys}. Available keys: {available}")
        return {key: vector_features[key] for key in feature_keys}

    def get_default_multifeature_keys(self) -> Tuple[str, ...]:
        keys: List[str] = []
        if self.include_input_sq_mean:
            keys.append("norm_x")
        for stage_idx, num_blocks in enumerate(self.layers):
            layer_name = f"layer{stage_idx + 1}"
            if self.block_stride > 0:
                for block_num in range(self.block_stride, num_blocks, self.block_stride):
                    base_name = f"{layer_name}_blk{block_num}"
                    keys.extend(self._expand_multifeature_names(base_name))
            keys.extend(self._expand_multifeature_names(layer_name))
        return tuple(keys)

    def _expand_multifeature_names(self, base_name: str) -> List[str]:
        names: List[str] = []
        if self.include_spatial_features:
            names.append(base_name)
        if self.include_mean_features:
            names.append(f"{base_name}_mean")
        if self.include_std_features:
            names.append(f"{base_name}_std")
        for patch_size in self.patch_mean_sizes:
            names.append(f"{base_name}_mean_{patch_size}")
        for patch_size in self.patch_std_sizes:
            names.append(f"{base_name}_std_{patch_size}")
        return names

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        vectors = self.get_vector_features(latents, feature_keys=(self.feature_key,))
        value = vectors[self.feature_key]
        if value.shape[1] == 1:
            return value[:, 0, :]
        return value.flatten(1)


def torch_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """Memory-friendlier Euclidean cdist matching drifting's JAX implementation."""
    x = x.float()
    y = y.float()
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drifting_loss_torch(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor,
    weight_gen: torch.Tensor,
    weight_pos: torch.Tensor,
    weight_neg: torch.Tensor,
    r_values: Sequence[float],
    mask_neg_self: bool = False,
):
    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    _, c_gen, feat_dim = gen.shape
    c_pos = fixed_pos.shape[1]
    c_neg = fixed_neg.shape[1]
    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # A.6 Feature normalization: share one feature scale per feature set so
    # all spatial locations / pooled vectors from the same feature map live on
    # the same distance scale.
    dist = torch_cdist(old_gen, targets)
    weighted_dist = dist * targets_w[:, None, :]
    feature_scale = weighted_dist.mean() / torch.clamp(targets_w.mean(), min=1e-8)
    feature_scale = torch.clamp(feature_scale, min=1e-3)
    feature_scale_per_dim = torch.clamp(feature_scale / math.sqrt(feat_dim), min=1e-3)
    old_gen_scaled = old_gen / feature_scale_per_dim
    targets_scaled = targets / feature_scale_per_dim
    dist_normed = dist / feature_scale

    diag_mask = torch.eye(c_gen, device=gen.device, dtype=gen.dtype)[None, :, :]
    block_mask = torch.cat([diag_mask, torch.zeros((1, c_gen, c_neg + c_pos), device=gen.device, dtype=gen.dtype)], dim=2)
    if mask_neg_self:
        if c_neg != c_gen:
            raise ValueError("mask_neg_self requires fixed_neg to have the same size as gen along dim 1")
        neg_self_mask = torch.cat(
            [torch.zeros((1, c_gen, c_gen), device=gen.device, dtype=gen.dtype), diag_mask, torch.zeros((1, c_gen, c_pos), device=gen.device, dtype=gen.dtype)],
            dim=2,
        )
        block_mask = block_mask + neg_self_mask
    dist_normed = dist_normed + block_mask * 100.0

    force_all = torch.zeros_like(old_gen_scaled)
    info = {
        "feature_scale": feature_scale.detach(),
        "feature_scale_per_dim": feature_scale_per_dim.detach(),
    }
    split_idx = c_gen + c_neg
    for r in r_values:
        # A.6 / B.2 kernel normalization: softmax over both axes.
        logits = -dist_normed / r
        aff_row = torch.softmax(logits, dim=-1)
        aff_col = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(aff_row * aff_col, min=1e-6))
        affinity = affinity * targets_w[:, None, :]
        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        sum_pos = aff_pos.sum(dim=-1, keepdim=True)
        sum_neg = aff_neg.sum(dim=-1, keepdim=True)
        coeff_neg = -aff_neg * sum_pos
        coeff_pos = aff_pos * sum_neg
        coeff = torch.cat([coeff_neg, coeff_pos], dim=2)
        force_r = torch.einsum("biy,byx->bix", coeff, targets_scaled)
        force_r = force_r - coeff.sum(dim=-1, keepdim=True) * old_gen_scaled
        # A.6 Drift normalization: normalize each drift field by its RMS so
        # every feature contributes on a comparable energy scale.
        drift_rms = torch.sqrt(torch.clamp((force_r ** 2).mean(), min=1e-8))
        info[f"drift_rms_{r}"] = drift_rms.detach()
        force_all = force_all + force_r / drift_rms

    goal_scaled = (old_gen_scaled + force_all).detach()
    gen_scaled = gen / feature_scale_per_dim
    loss = ((gen_scaled - goal_scaled) ** 2).mean(dim=(-1, -2))
    return loss, info


def resolve_drifting_feature_keys(args, extractor: FrozenMAELatentFeatureExtractor) -> Tuple[str, ...]:
    if args.drifting_feature_keys:
        return parse_name_list(args.drifting_feature_keys)
    if args.drifting_feature_mode == "multi":
        return extractor.get_default_multifeature_keys()
    return (args.drifting_feature_key,)


def decode_latents_to_pil(vae: AutoencoderKL, latents: torch.Tensor) -> List[Image.Image]:
    with torch.no_grad():
        imgs = vae.decode((latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)).sample
    imgs = ((imgs.clamp(-1, 1) + 1) / 2).detach().cpu()
    imgs = (imgs * 255).round().to(torch.uint8)
    out = []
    for img in imgs:
        out.append(Image.fromarray(img.permute(1, 2, 0).numpy()))
    return out


def encode_images_to_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    encode_mode: str = "sample",
) -> torch.Tensor:
    images = images.to(device=device, dtype=dtype)
    latent_dist = vae.encode(images).latent_dist
    if encode_mode == "sample":
        latents = latent_dist.sample()
    elif encode_mode == "mode":
        latents = latent_dist.mode()
    else:
        raise ValueError(f"Unknown VAE latent encode mode: {encode_mode}")
    return latents * vae.config.scaling_factor


def _normalize_vector_for_ranking(
    values: torch.Tensor,
    mode: str = "zscore",
    eps: float = 1e-6,
) -> torch.Tensor:
    values = values.float()
    if mode == "none":
        return values
    if mode == "zscore":
        std = torch.clamp(values.std(unbiased=False), min=eps)
        return (values - values.mean()) / std
    raise ValueError(f"Unknown offline-distance score normalization mode: {mode}")


def _feature_distance_to_refs(
    cand_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return one distance value per generated candidate.

    cand_feat: [G, D] or [G, T, D]
    ref_feat:  [R, D] or [R, T, D]

    The token/spatial dimension is flattened intentionally. This keeps ranking
    simple and works for layer_mean, layer_std, patch features, and optional
    spatial features as long as candidate and reference feature shapes match.
    """
    if cand_feat.ndim < 2:
        raise ValueError(f"Expected cand_feat with ndim >= 2, got shape={tuple(cand_feat.shape)}")
    if ref_feat.ndim < 2:
        raise ValueError(f"Expected ref_feat with ndim >= 2, got shape={tuple(ref_feat.shape)}")

    cand_flat = cand_feat.float().reshape(cand_feat.shape[0], -1)
    ref_flat = ref_feat.float().reshape(ref_feat.shape[0], -1)

    if cand_flat.shape[1] != ref_flat.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch for offline-distance ranking: "
            f"cand={tuple(cand_flat.shape)}, ref={tuple(ref_flat.shape)}"
        )

    # [G, R]
    dist = torch.cdist(cand_flat, ref_flat, p=2)

    if reduction == "mean":
        return dist.mean(dim=1)
    if reduction == "min":
        return dist.min(dim=1).values
    raise ValueError(f"Unknown reference distance reduction: {reduction}")


def _feature_cosine_similarity_to_refs(
    cand_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one cosine-similarity value per generated candidate."""
    if cand_feat.ndim < 2:
        raise ValueError(f"Expected cand_feat with ndim >= 2, got shape={tuple(cand_feat.shape)}")
    if ref_feat.ndim < 2:
        raise ValueError(f"Expected ref_feat with ndim >= 2, got shape={tuple(ref_feat.shape)}")

    cand_flat = cand_feat.float().reshape(cand_feat.shape[0], -1)
    ref_flat = ref_feat.float().reshape(ref_feat.shape[0], -1)

    if cand_flat.shape[1] != ref_flat.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch for offline cosine ranking: "
            f"cand={tuple(cand_flat.shape)}, ref={tuple(ref_flat.shape)}"
        )

    cand_flat = F.normalize(cand_flat, p=2, dim=1, eps=eps)
    ref_flat = F.normalize(ref_flat, p=2, dim=1, eps=eps)
    sim = cand_flat @ ref_flat.t()

    if reduction == "mean":
        return sim.mean(dim=1)
    if reduction == "min":
        return sim.max(dim=1).values
    raise ValueError(f"Unknown reference similarity reduction: {reduction}")


def compute_offline_distance_scores(
    cand_feat_dict: Dict[str, torch.Tensor],
    pos_feat_dict: Dict[str, torch.Tensor],
    neg_feat_dict: Dict[str, torch.Tensor],
    feature_keys: Sequence[str],
    score_mode: str = "margin",
    normalize: str = "zscore",
    aggregation: str = "mean",
    ref_reduction: str = "mean",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Rank generated candidates by distance to offline chosen/rejected anchors.

    For every candidate x_j, compute:

        score(j) = normalize(d_neg(j) - d_pos(j))

    where d_pos is the feature distance to offline chosen images and d_neg is
    the feature distance to offline rejected images. Higher score means the
    candidate is closer to chosen and farther from rejected, so it becomes a
    pseudo-online positive sample.

    score_mode="separate_zscore" preserves the older behavior:

        score(j) = normalize(d_neg(j)) - normalize(d_pos(j))

    score_mode="cosine_softmax_diff" ranks candidates by:

        score(j) = softmax(sim_pos)(j) - softmax(sim_neg)(j)
    """
    score_terms: List[torch.Tensor] = []
    raw_d_pos_terms: List[torch.Tensor] = []
    raw_d_neg_terms: List[torch.Tensor] = []
    raw_margin_terms: List[torch.Tensor] = []
    raw_sim_pos_terms: List[torch.Tensor] = []
    raw_sim_neg_terms: List[torch.Tensor] = []

    for feature_name in feature_keys:
        cand_feat = cand_feat_dict[feature_name]
        pos_feat = pos_feat_dict[feature_name]
        neg_feat = neg_feat_dict[feature_name]

        d_pos = _feature_distance_to_refs(cand_feat, pos_feat, reduction=ref_reduction)
        d_neg = _feature_distance_to_refs(cand_feat, neg_feat, reduction=ref_reduction)
        margin = d_neg - d_pos

        if score_mode == "margin":
            score = _normalize_vector_for_ranking(margin, mode=normalize, eps=eps)
        elif score_mode == "separate_zscore":
            z_pos = _normalize_vector_for_ranking(d_pos, mode=normalize, eps=eps)
            z_neg = _normalize_vector_for_ranking(d_neg, mode=normalize, eps=eps)
            score = z_neg - z_pos
        elif score_mode == "cosine_softmax_diff":
            sim_pos = _feature_cosine_similarity_to_refs(cand_feat, pos_feat, reduction=ref_reduction, eps=eps)
            sim_neg = _feature_cosine_similarity_to_refs(cand_feat, neg_feat, reduction=ref_reduction, eps=eps)
            score = F.softmax(sim_pos, dim=0) - F.softmax(sim_neg, dim=0)
            raw_sim_pos_terms.append(sim_pos)
            raw_sim_neg_terms.append(sim_neg)
        else:
            raise ValueError(f"Unknown offline-distance score mode: {score_mode}")

        score_terms.append(score)
        raw_d_pos_terms.append(d_pos)
        raw_d_neg_terms.append(d_neg)
        raw_margin_terms.append(margin)

    score_stack = torch.stack(score_terms, dim=0)
    d_pos_stack = torch.stack(raw_d_pos_terms, dim=0)
    d_neg_stack = torch.stack(raw_d_neg_terms, dim=0)
    margin_stack = torch.stack(raw_margin_terms, dim=0)
    sim_pos_stack = torch.stack(raw_sim_pos_terms, dim=0) if raw_sim_pos_terms else None
    sim_neg_stack = torch.stack(raw_sim_neg_terms, dim=0) if raw_sim_neg_terms else None

    if aggregation == "mean":
        final_score = score_stack.mean(dim=0)
    elif aggregation == "sum":
        final_score = score_stack.sum(dim=0)
    else:
        raise ValueError(f"Unknown offline-distance score aggregation: {aggregation}")

    final_d_pos = d_pos_stack.mean(dim=0)
    final_d_neg = d_neg_stack.mean(dim=0)
    final_margin = margin_stack.mean(dim=0)

    info = {
        "offline_distance_score_mean": final_score.mean().detach(),
        "offline_distance_score_std": final_score.std(unbiased=False).detach(),
        "offline_distance_score_max": final_score.max().detach(),
        "offline_distance_score_min": final_score.min().detach(),
        "offline_distance_d_pos_mean": final_d_pos.mean().detach(),
        "offline_distance_d_neg_mean": final_d_neg.mean().detach(),
        "offline_distance_margin_mean": final_margin.mean().detach(),
        "offline_distance_margin_std": final_margin.std(unbiased=False).detach(),
    }
    if sim_pos_stack is not None and sim_neg_stack is not None:
        final_sim_pos = sim_pos_stack.mean(dim=0)
        final_sim_neg = sim_neg_stack.mean(dim=0)
        info["offline_distance_cos_pos_mean"] = final_sim_pos.mean().detach()
        info["offline_distance_cos_neg_mean"] = final_sim_neg.mean().detach()
        info["offline_distance_cos_margin_mean"] = (final_sim_pos - final_sim_neg).mean().detach()
    return final_score, info


def evaluate_fixed_prompts(
    args,
    accelerator: Accelerator,
    unet,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    selector,
    generation_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    prompts: List[str],
    global_step: int,
):
    if not accelerator.is_main_process:
        return None

    step_dir = os.path.join(args.output_dir, args.eval_output_subdir, f"step-{global_step:04d}")
    os.makedirs(step_dir, exist_ok=True)

    unet_to_eval = accelerator.unwrap_model(unet)
    was_training = unet_to_eval.training
    unet_to_eval.eval()

    records = []
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts):
            prompt_ids = tokenizer(
                [prompt],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            encoder_hidden_states = text_encoder(prompt_ids)[0]
            generator = torch.Generator(device=accelerator.device).manual_seed(args.eval_seed + prompt_idx)
            noisy_latents = torch.randn(
                (1, 4, args.resolution // 8, args.resolution // 8),
                generator=generator,
                device=accelerator.device,
                dtype=weight_dtype,
            )
            _, model_pred_latent = run_one_step_unet(
                unet_to_eval,
                noisy_latents,
                encoder_hidden_states,
                generation_scheduler,
                timestep=args.generation_timestep,
                target_timestep=args.generation_target_timestep,
            )
            image = decode_latents_to_pil(vae, model_pred_latent)[0]
            file_name = f"{prompt_idx:03d}.png"
            image.save(os.path.join(step_dir, file_name))
            score_value = float(selector.score([image], prompt)[0])
            score_key = args.choice_model
            records.append(
                {
                    "step": global_step,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "file_name": file_name,
                    "seed": args.eval_seed + prompt_idx,
                    score_key: score_value,
                }
            )

    if was_training:
        unet_to_eval.train()

    scores_path = os.path.join(step_dir, "score_records.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    score_key = args.choice_model
    avg_score = sum(record[score_key] for record in records) / len(records)
    summary = {
        "step": global_step,
        "scores_file": scores_path,
        "samples": records,
    }
    summary[f"avg_{score_key}"] = avg_score
    summary[f"best_{score_key}"] = max(record[score_key] for record in records)
    summary[f"worst_{score_key}"] = min(record[score_key] for record in records)
    metrics_path = os.path.join(args.output_dir, "eval_metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary


def compute_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    if norm_type == float("inf"):
        return max(g.abs().max() for g in grads)
    norms = torch.stack([torch.norm(g, norm_type) for g in grads])
    return torch.norm(norms, norm_type)


def compute_loss_grad_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    sq_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
    for grad in grads:
        if grad is None:
            continue
        sq_norm = sq_norm + grad.detach().float().pow(2).sum()
    return torch.sqrt(sq_norm)


def build_sample_grad_norm_filter(
    sample_losses: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    discard_quantile: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if sample_losses.ndim != 1:
        raise ValueError(f"Expected 1D sample losses, got shape={tuple(sample_losses.shape)}")
    if sample_losses.numel() == 0:
        raise ValueError("Expected at least one sample loss.")

    sample_grad_norms = torch.stack([compute_loss_grad_norm(loss_i, parameters) for loss_i in sample_losses.unbind()])
    detached_norms = sample_grad_norms.detach()
    safe_norms = torch.nan_to_num(detached_norms, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    finite_mask = torch.isfinite(detached_norms)

    if safe_norms.numel() == 1:
        threshold = safe_norms[0]
        keep_mask = finite_mask.clone()
        if not bool(keep_mask.any()):
            keep_mask = torch.ones_like(safe_norms, dtype=torch.bool)
    else:
        threshold = torch.quantile(safe_norms, discard_quantile)
        keep_mask = finite_mask & (safe_norms <= threshold)
        if not bool(keep_mask.any()):
            keep_mask = torch.zeros_like(safe_norms, dtype=torch.bool)
            keep_mask[torch.argmin(safe_norms)] = True

    keep_count = keep_mask.sum().to(dtype=torch.float32)
    stats = {
        "sample_grad_norm_mean": safe_norms.mean(),
        "sample_grad_norm_max": safe_norms.max(),
        "sample_grad_norm_threshold": threshold,
        "sample_grad_keep_ratio": keep_mask.float().mean(),
        "sample_grad_keep_count": keep_count,
        "sample_grad_drop_count": safe_norms.new_tensor(float(safe_norms.numel())) - keep_count,
    }
    return keep_mask, stats


def _flatten_task_grads(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    flat_chunks = []
    device = loss.device
    for param, grad in zip(parameters, grads):
        if grad is None:
            flat_chunks.append(torch.zeros(param.numel(), device=device, dtype=torch.float32))
        else:
            flat_chunks.append(grad.detach().float().reshape(-1))
    if not flat_chunks:
        return torch.zeros(1, device=device, dtype=torch.float32)
    return torch.cat(flat_chunks)


def compute_task_gradient_stats(
    pref_loss: torch.Tensor,
    ref_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    ref_weight: float,
) -> Dict[str, torch.Tensor]:
    pref_grad = _flatten_task_grads(pref_loss, parameters)
    ref_grad = _flatten_task_grads(ref_loss, parameters)
    weighted_ref_grad = ref_grad * ref_weight

    pref_norm = torch.norm(pref_grad)
    ref_norm = torch.norm(ref_grad)
    weighted_ref_norm = torch.norm(weighted_ref_grad)
    denom = torch.clamp(pref_norm * ref_norm, min=1e-12)
    cosine = torch.sum(pref_grad * ref_grad) / denom
    return {
        "pref_grad_norm": pref_norm.detach(),
        "ref_grad_norm": ref_norm.detach(),
        "weighted_ref_grad_norm": weighted_ref_norm.detach(),
        "pref_ref_grad_cosine": cosine.detach(),
    }


def _make_json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)


def build_resolved_config(args) -> Dict[str, object]:
    return {key: _make_json_safe(value) for key, value in vars(args).items()}


def _make_tracker_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(_make_json_safe(value), ensure_ascii=False, sort_keys=True)


def build_tracker_config(args) -> Dict[str, object]:
    return {key: _make_tracker_safe(value) for key, value in vars(args).items()}


def setup_logging(args, accelerator: Accelerator) -> str:
    log_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=f"%(asctime)s | %(levelname)s | rank={accelerator.process_index}/{accelerator.num_processes} | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log_path


def save_runtime_snapshot(args, accelerator: Accelerator) -> Dict[str, object]:
    snapshot_dir = os.path.join(args.output_dir, "run_metadata")
    code_dir = os.path.join(snapshot_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    resolved_config = build_resolved_config(args)
    config_json_path = os.path.join(snapshot_dir, "resolved_config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=False, sort_keys=True)

    config_txt_path = os.path.join(snapshot_dir, "resolved_config.txt")
    with open(config_txt_path, "w", encoding="utf-8") as f:
        for key in sorted(resolved_config):
            f.write(f"{key}={resolved_config[key]}\n")

    command_path = os.path.join(snapshot_dir, "launch_command.sh")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(shlex.join([sys.executable, *sys.argv]) + "\n")

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
    with open(runtime_info_path, "w", encoding="utf-8") as f:
        json.dump(runtime_info, f, indent=2, ensure_ascii=False, sort_keys=True)

    copied_files: List[str] = []
    snapshot_sources = [
        os.path.abspath(__file__),
        os.path.join(PROJECT_ROOT, "scripts", "inference", "sample_sdturbo_lora.sh"),
    ]
    for source_path in snapshot_sources:
        if not os.path.exists(source_path):
            continue
        relative_path = os.path.relpath(source_path, PROJECT_ROOT)
        target_path = os.path.join(code_dir, relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_files.append(relative_path)

    manifest_path = os.path.join(snapshot_dir, "code_snapshot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"files": copied_files}, f, indent=2, ensure_ascii=False, sort_keys=True)

    return {
        "snapshot_dir": snapshot_dir,
        "config_json_path": config_json_path,
        "config_txt_path": config_txt_path,
        "command_path": command_path,
        "runtime_info_path": runtime_info_path,
        "manifest_path": manifest_path,
        "copied_files": copied_files,
    }


def _build_fallback_pickscore_selector(
    device: torch.device,
    model_name_or_path: str,
    processor_name_or_path: str,
    local_files_only: bool = True,
):
    """Fallback selector when local utils.pickscore_utils is unavailable."""

    class _FallbackPickScoreSelector:
        def __init__(self, device):
            self.device = device
            self.processor = AutoProcessor.from_pretrained(
                processor_name_or_path,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            ).eval().to(device)

        def _to_tensor(self, out):
            if isinstance(out, torch.Tensor):
                return out
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            if getattr(out, "last_hidden_state", None) is not None:
                return out.last_hidden_state[:, 0, :]
            raise TypeError(f"Unsupported feature output type: {type(out)}")

        def score(self, images, prompt):
            if isinstance(prompt, str):
                texts = [prompt] * len(images)
            else:
                texts = prompt
            image_inputs = self.processor(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            text_inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                image_embs = self._to_tensor(self.model.get_image_features(**image_inputs))
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                text_embs = self._to_tensor(self.model.get_text_features(**text_inputs))
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                logits = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
                return logits.diag().detach().cpu().tolist()

    return _FallbackPickScoreSelector(device)


def parse_args():
    parser = argparse.ArgumentParser("SD-Turbo Drifting trainer with offline/online modes")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "sd-turbo"),
    )
    parser.add_argument(
        "--initial_unet_path",
        type=str,
        default=None,
        help="Optional external UNet weights or directory loaded into both student and reference UNets.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/sdturbo-diop-hybrid")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_mode",
        type=str,
        default="offline",
        choices=["offline", "online", "offline_distance"],
        help=(
            "offline: use dataset chosen/rejected directly as drifting anchors; "
            "online: use reward model to rank generated candidates; "
            "offline_distance: use dataset chosen/rejected as feature-distance references "
            "to rank generated candidates and build pseudo-online preference pairs."
        ),
    )
    parser.add_argument("--pairs_jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "pairs.jsonl"))
    parser.add_argument(
        "--choice_model",
        type=str,
        default="pickscore",
        choices=["aes", "clip", "hps", "hpsv2", "pickscore"],
    )
    parser.add_argument(
        "--pickscore_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "PickScore_v1"),
    )
    parser.add_argument(
        "--pickscore_processor_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pickscore_allow_remote",
        action="store_true",
        help="Allow downloading PickScore assets from HuggingFace when local files are missing.",
    )
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

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--use_lora", action="store_true")
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
    parser.add_argument(
        "--sample_grad_norm_discard_quantile",
        type=float,
        default=0.85,
        help="Drop samples whose per-sample grad norm exceeds this within-batch quantile. Set to 1.0 to disable.",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=200)

    parser.add_argument(
        "--drifting_mae_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "drifting", "mae_latent_256_torch.pth"),
    )
    parser.add_argument("--drifting_feature_mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--drifting_feature_key", type=str, default="layer4_mean")
    parser.add_argument(
        "--drifting_feature_keys",
        type=str,
        default="",
        help="Comma-separated feature keys. Overrides --drifting_feature_mode/--drifting_feature_key when set.",
    )
    parser.add_argument(
        "--drifting_feature_block_stride",
        type=int,
        default=2,
        help="Export intermediate residual block features every N blocks in multi-feature mode.",
    )
    parser.add_argument(
        "--drifting_feature_patch_sizes",
        type=str,
        default="2,4",
        help="Comma-separated patch sizes for pooled mean/std multi-features.",
    )
    parser.add_argument(
        "--drifting_include_input_sq_mean",
        action="store_true",
        help="Include encoder-input x^2 mean as an additional drifting feature.",
    )
    parser.add_argument(
        "--drifting_include_spatial_features",
        action="store_true",
        help="Include raw spatial token features in multi-feature mode. Disabled by default in PyTorch to avoid OOM.",
    )
    parser.add_argument(
        "--drifting_feature_aggregation",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="How to aggregate per-feature drifting losses when multiple features are active.",
    )
    parser.add_argument("--drifting_pref_r_list", type=str, default="0.02,0.05,0.2")
    parser.add_argument("--drifting_ref_r_list", type=str, default="0.02,0.05,0.2")
    parser.add_argument("--num_pos_images", type=int, default=1)
    parser.add_argument("--num_neg_images", type=int, default=1)
    parser.add_argument(
        "--online_feature_top_fraction",
        type=float,
        default=0.5,
        help="In online mode, only the reward top fraction of generated candidates contributes feature-level gradients. Use 1.0 to disable filtering.",
    )
    parser.add_argument(
        "--offline_distance_top_fraction",
        type=float,
        default=0.5,
        help=(
            "In offline_distance mode, only the top fraction of generated candidates "
            "by offline-distance score contributes feature-level gradients. Use 1.0 to disable filtering."
        ),
    )
    parser.add_argument(
        "--offline_distance_score_mode",
        type=str,
        default="margin",
        choices=["margin", "separate_zscore", "cosine_softmax_diff"],
        help=(
            "Score generated candidates in offline_distance mode. "
            "cosine_softmax_diff matches the cossoftmax proxy: "
            "softmax(sim_to_chosen) - softmax(sim_to_rejected)."
        ),
    )
    parser.add_argument(
        "--offline_distance_score_normalize",
        type=str,
        default="zscore",
        choices=["zscore", "none"],
        help="Normalize L2 distance scores. Ignored by cosine_softmax_diff.",
    )
    parser.add_argument(
        "--offline_distance_score_aggregation",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Aggregate offline-distance scores across drifting feature keys.",
    )
    parser.add_argument(
        "--offline_distance_ref_reduction",
        type=str,
        default="mean",
        choices=["mean", "min"],
        help="Reduce distances to multiple offline chosen/rejected references by mean or min.",
    )
    parser.add_argument(
        "--offline_latent_encode_mode",
        type=str,
        default="sample",
        choices=["sample", "mode"],
        help="How to encode offline chosen/rejected images with VAE. sample preserves old behavior; mode is deterministic.",
    )
    parser.add_argument(
        "--eval_prompt_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "prompts", "pickapicv2_test_unique.txt"),
    )
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_output_subdir", type=str, default="eval_samples")
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--drifting_ref_loss_weight", type=float, default=0.2)
    parser.add_argument("--drifting_ref_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_pos_weight", type=float, default=3000.0)
    parser.add_argument(
        "--log_task_grad_stats",
        action="store_true",
        help="Log gradient norm / cosine statistics for pref_loss vs ref_loss on trainable parameters.",
    )
    parser.add_argument(
        "--task_grad_log_interval",
        type=int,
        default=50,
        help="When --log_task_grad_stats is enabled, compute task-gradient stats every N optimizer steps.",
    )
    parser.add_argument(
        "--ref_model_l2_weight",
        type=float,
        default=0.0,
        help="Weight for MSE(student one-step latent, frozen ref one-step latent). 0 disables.",
    )
    parser.add_argument(
        "--frozen_feature_l2_weight",
        type=float,
        default=0.0,
        help="Weight for MSE(student vs ref) per drifting feature key; aggregate with --drifting_feature_aggregation (mean/sum). 0 disables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.drifting_pref_r_list = parse_float_list(args.drifting_pref_r_list)
    args.drifting_ref_r_list = parse_float_list(args.drifting_ref_r_list)
    args.drifting_feature_patch_sizes = parse_int_list(args.drifting_feature_patch_sizes)
    if args.num_pos_images < 1 or args.num_neg_images < 1:
        raise ValueError("--num_pos_images and --num_neg_images must be >= 1")
    if not (0.0 < args.online_feature_top_fraction <= 1.0):
        raise ValueError("--online_feature_top_fraction must be in (0, 1].")
    if not (0.0 < args.offline_distance_top_fraction <= 1.0):
        raise ValueError("--offline_distance_top_fraction must be in (0, 1].")
    if args.num_eval_prompts < 1:
        raise ValueError("--num_eval_prompts must be >= 1")
    if args.eval_every_steps < 0:
        raise ValueError("--eval_every_steps must be >= 0")
    if not (0.0 < args.sample_grad_norm_discard_quantile <= 1.0):
        raise ValueError("--sample_grad_norm_discard_quantile must be in (0, 1].")
    if args.generation_timestep < 0:
        raise ValueError("--generation_timestep must be >= 0.")
    if args.generation_target_timestep >= 0 and args.generation_target_timestep > args.generation_timestep:
        raise ValueError("--generation_target_timestep must be <= --generation_timestep (or -1).")
    if args.pickscore_processor_name_or_path is None:
        args.pickscore_processor_name_or_path = args.pickscore_model_name_or_path
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

    runtime_snapshot = None
    if accelerator.is_main_process:
        runtime_snapshot = save_runtime_snapshot(args, accelerator)
        logger.info(f"Training log will be written to {log_path}")
        logger.info(f"Resolved config saved to {runtime_snapshot['config_json_path']}")
        logger.info(f"Runtime snapshot saved to {runtime_snapshot['snapshot_dir']}")
        logger.info(
            "Copied code snapshot files: %s",
            ", ".join(runtime_snapshot["copied_files"]) if runtime_snapshot["copied_files"] else "(none)",
        )
    accelerator.wait_for_everyone()
    logger.info(
        "Starting training with mode=%s, output_dir=%s, num_processes=%s, batch_size=%s, grad_accum=%s, mixed_precision=%s",
        args.train_mode,
        args.output_dir,
        accelerator.num_processes,
        args.train_batch_size,
        args.gradient_accumulation_steps,
        args.mixed_precision,
    )

    generation_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if "turbo" in args.pretrained_model_name_or_path.lower():
        enforce_zero_terminal_snr(generation_scheduler)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    if args.generation_timestep >= generation_scheduler.config.num_train_timesteps:
        raise ValueError(
            f"--generation_timestep must be < num_train_timesteps ({generation_scheduler.config.num_train_timesteps})."
        )
    if args.generation_target_timestep >= generation_scheduler.config.num_train_timesteps:
        raise ValueError(
            f"--generation_target_timestep must be < num_train_timesteps ({generation_scheduler.config.num_train_timesteps}) or -1."
        )
    if args.initial_unet_path:
        init_weight_path = load_unet_checkpoint_weights(ref_unet, args.initial_unet_path)
        load_unet_checkpoint_weights(unet, args.initial_unet_path)
        logger.info("Initialized student/reference UNets from %s", init_weight_path)
    logger.info(
        "One-step generation config: timestep=%s target_timestep=%s prediction_type=%s",
        args.generation_timestep,
        args.generation_target_timestep,
        getattr(generation_scheduler.config, "prediction_type", "epsilon"),
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)

    if is_xformers_available():
        import xformers

        if version.parse(xformers.__version__) >= version.parse("0.0.17"):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention.")
            except Exception as exc:
                logger.warning(f"Failed to enable xformers memory efficient attention: {exc}")
        else:
            logger.warning(f"xformers version {xformers.__version__} is too old; continuing without it.")
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
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet.parameters())
            logger.info(
                f"Enabled LoRA on UNet: trainable_params={trainable_params}, total_params={total_params}, "
                f"ratio={trainable_params / total_params:.6f}"
            )

    drifting_feature_extractor = FrozenMAELatentFeatureExtractor(
        args.drifting_mae_path,
        args.drifting_feature_key,
        block_stride=args.drifting_feature_block_stride,
        patch_pool_sizes=args.drifting_feature_patch_sizes,
        include_input_sq_mean=args.drifting_include_input_sq_mean,
        include_spatial_features=args.drifting_include_spatial_features,
    ).to(accelerator.device)
    drifting_feature_extractor.eval()
    drifting_feature_keys = resolve_drifting_feature_keys(args, drifting_feature_extractor)
    logger.info(
        "Using %d drifting feature(s) with mode=%s, aggregation=%s: %s",
        len(drifting_feature_keys),
        args.drifting_feature_mode,
        args.drifting_feature_aggregation,
        ", ".join(drifting_feature_keys),
    )

    trainable_unet_parameters = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_unet_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = PairsPromptDataset(
        pairs_jsonl=args.pairs_jsonl,
        tokenizer=tokenizer,
        train_mode=args.train_mode,
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
    trainable_unet_parameters = [p for p in unet.parameters() if p.requires_grad]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)
    drifting_feature_extractor.to(accelerator.device, dtype=weight_dtype)

    eval_prompts = load_prompt_file(args.eval_prompt_file, max_prompts=args.num_eval_prompts)
    if accelerator.is_main_process:
        eval_prompt_path = os.path.join(args.output_dir, "eval_prompts.txt")
        with open(eval_prompt_path, "w", encoding="utf-8") as f:
            for prompt in eval_prompts:
                f.write(prompt + "\n")

    selector = None
    if args.train_mode == "online":
        try:
            if args.choice_model in {"hps", "hpsv2"}:
                from utils.hps_utils import Selector
            elif args.choice_model == "clip":
                from utils.clip_utils import Selector
            elif args.choice_model == "pickscore":
                from utils.pickscore_utils import Selector
            else:
                from utils.aes_utils import Selector
            selector = Selector(accelerator.device)
        except ModuleNotFoundError:
            if args.choice_model != "pickscore":
                raise
            logger.warning("utils.pickscore_utils not found. Falling back to built-in PickScore selector.")
            local_files_only = not args.pickscore_allow_remote
            if args.pickscore_allow_remote and accelerator.is_main_process:
                # Avoid all ranks hitting remote simultaneously:
                # rank0 warms cache first, then others load from local cache only.
                _ = _build_fallback_pickscore_selector(
                    accelerator.device,
                    model_name_or_path=args.pickscore_model_name_or_path,
                    processor_name_or_path=args.pickscore_processor_name_or_path,
                    local_files_only=False,
                )
                accelerator.wait_for_everyone()
                local_files_only = True
            selector = _build_fallback_pickscore_selector(
                accelerator.device,
                model_name_or_path=args.pickscore_model_name_or_path,
                processor_name_or_path=args.pickscore_processor_name_or_path,
                local_files_only=local_files_only,
            )

    eval_selector = None
    if accelerator.is_main_process:
        eval_selector = _build_fallback_pickscore_selector(
            accelerator.device,
            model_name_or_path=args.pickscore_model_name_or_path,
            processor_name_or_path=args.pickscore_processor_name_or_path,
            local_files_only=not args.pickscore_allow_remote,
        )

    if accelerator.is_main_process:
        tracker_cfg = build_tracker_config(args)
        accelerator.init_trackers("sdturbo-diop-hybrid", tracker_cfg)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    eval_summary = evaluate_fixed_prompts(
        args=args,
        accelerator=accelerator,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        selector=eval_selector,
        generation_scheduler=generation_scheduler,
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
        logger.info(
            f"Eval step 0: avg_{eval_metric_key}={eval_summary[f'avg_{eval_metric_key}']:.6f}, "
            f"best={eval_summary[f'best_{eval_metric_key}']:.6f}, "
            f"worst={eval_summary[f'worst_{eval_metric_key}']:.6f}"
        )

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                prompt_batch = batch["prompt"]
                input_ids = batch["input_ids"]
                bsz = input_ids.shape[0]
                sample_losses: List[torch.Tensor] = []
                sample_pref_losses: List[torch.Tensor] = []
                sample_ref_losses: List[torch.Tensor] = []
                sample_ref_l2_losses: List[torch.Tensor] = []
                sample_frozen_feat_l2_losses: List[torch.Tensor] = []
                sample_dpos_values: List[torch.Tensor] = []
                sample_dneg_values: List[torch.Tensor] = []
                sample_dref_values: List[torch.Tensor] = []
                sample_v_var_values: List[torch.Tensor] = []
                sample_ref_v_var_values: List[torch.Tensor] = []
                sample_offline_distance_stats: List[Dict[str, torch.Tensor]] = []

                for i in range(bsz):
                    prompt = prompt_batch[i]
                    prompt_ids = input_ids[i : i + 1]
                    encoder_hidden_states = text_encoder(prompt_ids)[0].repeat_interleave(args.batchsize_gen, dim=0)
                    noisy_latents = torch.randn((args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8), device=accelerator.device, dtype=weight_dtype)
                    model_pred, model_pred_latent = run_one_step_unet(
                        unet,
                        noisy_latents,
                        encoder_hidden_states,
                        generation_scheduler,
                        timestep=args.generation_timestep,
                        target_timestep=args.generation_target_timestep,
                    )
                    with torch.no_grad():
                        ref_pred, ref_pred_latent = run_one_step_unet(
                            ref_unet,
                            noisy_latents,
                            encoder_hidden_states,
                            generation_scheduler,
                            timestep=args.generation_timestep,
                            target_timestep=args.generation_target_timestep,
                        )
                    sample_v_var_values.append(model_pred.float().var(unbiased=False))
                    sample_ref_v_var_values.append(ref_pred.float().var(unbiased=False))

                    # Debug: convert model_pred_latent to PNG and save for inspection
                    # import os
                    # from datetime import datetime
                    # imgs = decode_latents_to_pil(vae, model_pred_latent)
                    # save_dir = "./debug_pred_png"
                    # os.makedirs(save_dir, exist_ok=True)
                    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    # for idx, img in enumerate(imgs):
                    #     img.save(os.path.join(save_dir, f"model_pred_{timestamp}_{idx}.png"))

                    # exit(-1)
                    
                    ref_l2_i = F.mse_loss(model_pred_latent.float(), ref_pred_latent.float())

                    if args.train_mode in {"offline", "offline_distance"}:
                        chosen = batch["chosen"][i]
                        rejected = batch["rejected"][i]
                        with torch.no_grad():
                            win_data_latent = encode_images_to_latents(
                                vae=vae,
                                images=chosen,
                                device=accelerator.device,
                                dtype=weight_dtype,
                                encode_mode=args.offline_latent_encode_mode,
                            )
                            lose_data_latent = encode_images_to_latents(
                                vae=vae,
                                images=rejected,
                                device=accelerator.device,
                                dtype=weight_dtype,
                                encode_mode=args.offline_latent_encode_mode,
                            )

                        if args.train_mode == "offline":
                            # Original offline behavior: dataset chosen/rejected are fixed drifting anchors.
                            win_latent = win_data_latent
                            lose_latent = lose_data_latent
                            feature_model_pred_latent = model_pred_latent
                            feature_ref_pred_latent = ref_pred_latent
                        else:
                            # New offline_distance behavior:
                            # 1) Generate candidates with the current student.
                            # 2) Score candidates by normalized feature-distance difference to offline data:
                            #       score = z(distance_to_rejected) - z(distance_to_chosen)
                            # 3) Treat best/worst generated candidates as pseudo-online preference pair.
                            with torch.no_grad():
                                cand_score_feat_dict = drifting_feature_extractor.get_vector_features(
                                    model_pred_latent.detach(),
                                    feature_keys=drifting_feature_keys,
                                )
                                pos_score_feat_dict = drifting_feature_extractor.get_vector_features(
                                    win_data_latent,
                                    feature_keys=drifting_feature_keys,
                                )
                                neg_score_feat_dict = drifting_feature_extractor.get_vector_features(
                                    lose_data_latent,
                                    feature_keys=drifting_feature_keys,
                                )
                                scores_t, offline_distance_info = compute_offline_distance_scores(
                                    cand_feat_dict=cand_score_feat_dict,
                                    pos_feat_dict=pos_score_feat_dict,
                                    neg_feat_dict=neg_score_feat_dict,
                                    feature_keys=drifting_feature_keys,
                                    score_mode=args.offline_distance_score_mode,
                                    normalize=args.offline_distance_score_normalize,
                                    aggregation=args.offline_distance_score_aggregation,
                                    ref_reduction=args.offline_distance_ref_reduction,
                                )
                                sample_offline_distance_stats.append(offline_distance_info)

                                pos_count = min(args.num_pos_images, scores_t.shape[0])
                                neg_count = min(args.num_neg_images, scores_t.shape[0])
                                best_idx = torch.topk(scores_t, k=pos_count, largest=True).indices
                                worst_idx = torch.topk(scores_t, k=neg_count, largest=False).indices
                                feature_top_idx = select_top_fraction_indices(
                                    scores_t,
                                    args.offline_distance_top_fraction,
                                )

                                win_latent = model_pred_latent.index_select(0, best_idx).detach()
                                lose_latent = model_pred_latent.index_select(0, worst_idx).detach()

                            feature_model_pred_latent = model_pred_latent.index_select(0, feature_top_idx)
                            feature_ref_pred_latent = ref_pred_latent.index_select(0, feature_top_idx)
                    else:
                        cand_images = decode_latents_to_pil(vae, model_pred_latent)
                        scores = selector.score(cand_images, prompt)
                        scores_t = torch.tensor(scores, device=accelerator.device, dtype=torch.float32)
                        pos_count = min(args.num_pos_images, scores_t.shape[0])
                        neg_count = min(args.num_neg_images, scores_t.shape[0])
                        best_idx = torch.topk(scores_t, k=pos_count, largest=True).indices
                        worst_idx = torch.topk(scores_t, k=neg_count, largest=False).indices
                        feature_top_idx = select_top_fraction_indices(scores_t, args.online_feature_top_fraction)
                        with torch.no_grad():
                            win_latent = model_pred_latent.index_select(0, best_idx).detach()
                            lose_latent = model_pred_latent.index_select(0, worst_idx).detach()
                        feature_model_pred_latent = model_pred_latent.index_select(0, feature_top_idx)
                        feature_ref_pred_latent = ref_pred_latent.index_select(0, feature_top_idx)

                    x_gen_dict = drifting_feature_extractor.get_vector_features(feature_model_pred_latent, feature_keys=drifting_feature_keys)
                    with torch.no_grad():
                        x_ref_dict = drifting_feature_extractor.get_vector_features(feature_ref_pred_latent, feature_keys=drifting_feature_keys)
                        y_pos_dict = drifting_feature_extractor.get_vector_features(win_latent, feature_keys=drifting_feature_keys)
                        y_neg_dict = drifting_feature_extractor.get_vector_features(lose_latent, feature_keys=drifting_feature_keys)

                    pref_loss_terms = []
                    ref_loss_terms = []
                    frozen_feat_l2_terms = []
                    d_pos_terms = []
                    d_neg_terms = []
                    d_ref_terms = []
                    for feature_name in drifting_feature_keys:
                        x_gen = x_gen_dict[feature_name].reshape(-1, x_gen_dict[feature_name].shape[-1])
                        x_ref = x_ref_dict[feature_name].reshape(-1, x_ref_dict[feature_name].shape[-1])
                        frozen_feat_l2_terms.append(F.mse_loss(x_gen.float(), x_ref.float()))
                        y_pos = y_pos_dict[feature_name].reshape(-1, y_pos_dict[feature_name].shape[-1])
                        y_neg = y_neg_dict[feature_name].reshape(-1, y_neg_dict[feature_name].shape[-1])

                        gen_feat = x_gen.unsqueeze(0)
                        weight_gen = torch.ones((1, gen_feat.shape[1]), device=accelerator.device, dtype=gen_feat.dtype)

                        pref_fixed_pos = y_pos.unsqueeze(0)
                        pref_fixed_neg = y_neg.unsqueeze(0)
                        pref_weight_pos = torch.full(
                            (1, y_pos.shape[0]),
                            args.drifting_pos_weight,
                            device=accelerator.device,
                            dtype=gen_feat.dtype,
                        )
                        pref_weight_neg = torch.full(
                            (1, y_neg.shape[0]),
                            args.drifting_neg_weight,
                            device=accelerator.device,
                            dtype=gen_feat.dtype,
                        )
                        pref_loss_vec, _ = drifting_loss_torch(
                            gen=gen_feat,
                            fixed_pos=pref_fixed_pos,
                            fixed_neg=pref_fixed_neg,
                            weight_gen=weight_gen,
                            weight_pos=pref_weight_pos,
                            weight_neg=pref_weight_neg,
                            r_values=args.drifting_pref_r_list,
                        )

                        ref_fixed_pos = x_ref.unsqueeze(0)
                        ref_fixed_neg = x_gen.detach().unsqueeze(0)
                        ref_weight_pos = torch.full(
                            (1, x_ref.shape[0]),
                            args.drifting_ref_weight,
                            device=accelerator.device,
                            dtype=gen_feat.dtype,
                        )
                        ref_weight_neg = torch.full(
                            (1, x_gen.shape[0]),
                            args.drifting_ref_neg_weight,
                            device=accelerator.device,
                            dtype=gen_feat.dtype,
                        )
                        ref_loss_vec, _ = drifting_loss_torch(
                            gen=gen_feat,
                            fixed_pos=ref_fixed_pos,
                            fixed_neg=ref_fixed_neg,
                            weight_gen=weight_gen,
                            weight_pos=ref_weight_pos,
                            weight_neg=ref_weight_neg,
                            r_values=args.drifting_ref_r_list,
                            mask_neg_self=True,
                        )
                        pref_loss_terms.append(pref_loss_vec.mean())
                        ref_loss_terms.append(ref_loss_vec.mean())
                        d_pos_terms.append(torch.cdist(x_gen.float(), y_pos.float(), p=1).mean())
                        d_neg_terms.append(torch.cdist(x_gen.float(), y_neg.float(), p=1).mean())
                        d_ref_terms.append(torch.cdist(x_gen.float(), x_ref.float(), p=1).mean())

                    if args.drifting_feature_aggregation == "mean":
                        pref_loss_i = torch.stack(pref_loss_terms).mean()
                        ref_loss_i = torch.stack(ref_loss_terms).mean()
                        frozen_feat_l2_i = torch.stack(frozen_feat_l2_terms).mean()
                        d_pos_i = torch.stack(d_pos_terms).mean()
                        d_neg_i = torch.stack(d_neg_terms).mean()
                        d_ref_i = torch.stack(d_ref_terms).mean()
                    else:
                        pref_loss_i = torch.stack(pref_loss_terms).sum()
                        ref_loss_i = torch.stack(ref_loss_terms).sum()
                        frozen_feat_l2_i = torch.stack(frozen_feat_l2_terms).sum()
                        d_pos_i = torch.stack(d_pos_terms).mean()
                        d_neg_i = torch.stack(d_neg_terms).mean()
                        d_ref_i = torch.stack(d_ref_terms).mean()

                    loss_i = (
                        pref_loss_i
                        + args.drifting_ref_loss_weight * ref_loss_i
                        + args.ref_model_l2_weight * ref_l2_i
                        + args.frozen_feature_l2_weight * frozen_feat_l2_i
                    )
                    sample_losses.append(loss_i)
                    sample_pref_losses.append(pref_loss_i)
                    sample_ref_losses.append(ref_loss_i)
                    sample_ref_l2_losses.append(ref_l2_i)
                    sample_frozen_feat_l2_losses.append(frozen_feat_l2_i)
                    sample_dpos_values.append(d_pos_i)
                    sample_dneg_values.append(d_neg_i)
                    sample_dref_values.append(d_ref_i)

                loss_values = torch.stack(sample_losses)
                pref_loss_values = torch.stack(sample_pref_losses)
                ref_loss_values = torch.stack(sample_ref_losses)
                ref_l2_values = torch.stack(sample_ref_l2_losses)
                frozen_feat_l2_values = torch.stack(sample_frozen_feat_l2_losses)
                d_pos_values = torch.stack(sample_dpos_values)
                d_neg_values = torch.stack(sample_dneg_values)
                d_ref_values = torch.stack(sample_dref_values)
                v_var_values = torch.stack(sample_v_var_values)
                ref_v_var_values = torch.stack(sample_ref_v_var_values)

                sample_grad_filter_stats = None
                keep_mask = torch.ones_like(loss_values, dtype=torch.bool)
                if args.sample_grad_norm_discard_quantile < 1.0:
                    keep_mask, sample_grad_filter_stats = build_sample_grad_norm_filter(
                        sample_losses=loss_values,
                        parameters=trainable_unet_parameters,
                        discard_quantile=args.sample_grad_norm_discard_quantile,
                    )

                loss = loss_values[keep_mask].mean()
                pref_loss = pref_loss_values[keep_mask].mean()
                ref_loss = ref_loss_values[keep_mask].mean()
                ref_l2 = ref_l2_values[keep_mask].mean()
                frozen_feat_l2 = frozen_feat_l2_values[keep_mask].mean()
                d_pos = d_pos_values[keep_mask].mean()
                d_neg = d_neg_values[keep_mask].mean()
                d_ref = d_ref_values[keep_mask].mean()
                d_margin = d_neg - d_pos
                v_var = v_var_values[keep_mask].mean()
                ref_v_var = ref_v_var_values[keep_mask].mean()
                task_grad_stats = None

                if (
                    args.log_task_grad_stats
                    and accelerator.sync_gradients
                    and args.task_grad_log_interval > 0
                    and (global_step + 1) % args.task_grad_log_interval == 0
                ):
                    task_grad_stats = compute_task_gradient_stats(
                        pref_loss=pref_loss,
                        ref_loss=ref_loss,
                        parameters=trainable_unet_parameters,
                        ref_weight=args.drifting_ref_loss_weight,
                    )

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
                avg_pref_loss = accelerator.gather(pref_loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_ref_loss = accelerator.gather(ref_loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_ref_l2 = accelerator.gather(ref_l2.detach().repeat(args.train_batch_size)).mean().item()
                avg_frozen_feat_l2 = accelerator.gather(frozen_feat_l2.detach().repeat(args.train_batch_size)).mean().item()
                avg_dpos = accelerator.gather(d_pos.detach().repeat(args.train_batch_size)).mean().item()
                avg_dneg = accelerator.gather(d_neg.detach().repeat(args.train_batch_size)).mean().item()
                avg_dref = accelerator.gather(d_ref.detach().repeat(args.train_batch_size)).mean().item()
                avg_margin = accelerator.gather(d_margin.detach().repeat(args.train_batch_size)).mean().item()
                avg_v_var = accelerator.gather(v_var.detach().repeat(args.train_batch_size)).mean().item()
                avg_ref_v_var = accelerator.gather(ref_v_var.detach().repeat(args.train_batch_size)).mean().item()
                avg_grad = accelerator.gather(grad_norm_value.repeat(args.train_batch_size)).mean().item()
                log_payload = {
                    "train_loss": avg_loss,
                    "pref_loss_unaccumulated": avg_pref_loss,
                    "ref_loss_unaccumulated": avg_ref_loss,
                    "ref_l2_unaccumulated": avg_ref_l2,
                    "ref_l2_weighted_unaccumulated": avg_ref_l2 * args.ref_model_l2_weight,
                    "frozen_feat_l2_unaccumulated": avg_frozen_feat_l2,
                    "frozen_feat_l2_weighted_unaccumulated": avg_frozen_feat_l2 * args.frozen_feature_l2_weight,
                    "grad_norm_unaccumulated": avg_grad,
                    "d_pos_unaccumulated": avg_dpos,
                    "d_neg_unaccumulated": avg_dneg,
                    "d_ref_unaccumulated": avg_dref,
                    "d_margin_unaccumulated": avg_margin,
                    "v_variance_unaccumulated": avg_v_var,
                    "ref_v_variance_unaccumulated": avg_ref_v_var,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                avg_keep_ratio = None
                if sample_grad_filter_stats is not None:
                    for key, value in sample_grad_filter_stats.items():
                        avg_value = accelerator.gather(value.repeat(args.train_batch_size)).mean().item()
                        log_payload[f"{key}_unaccumulated"] = avg_value
                        if key == "sample_grad_keep_ratio":
                            avg_keep_ratio = avg_value
                if task_grad_stats is not None:
                    for key, value in task_grad_stats.items():
                        avg_value = accelerator.gather(value.repeat(args.train_batch_size)).mean().item()
                        log_payload[f"{key}_unaccumulated"] = avg_value
                if sample_offline_distance_stats:
                    offline_stat_keys = sample_offline_distance_stats[0].keys()
                    for key in offline_stat_keys:
                        value = torch.stack([stats[key] for stats in sample_offline_distance_stats]).mean()
                        avg_value = accelerator.gather(value.repeat(args.train_batch_size)).mean().item()
                        log_payload[f"{key}_unaccumulated"] = avg_value
                accelerator.log(log_payload, step=global_step)
                postfix = dict(
                    step_loss=avg_loss,
                    pref_loss=avg_pref_loss,
                    ref_loss=avg_ref_loss,
                    ref_l2=avg_ref_l2,
                    feat_l2=avg_frozen_feat_l2,
                    grad_norm=avg_grad,
                    d_pos=avg_dpos,
                    d_neg=avg_dneg,
                    d_ref=avg_dref,
                    d_margin=avg_margin,
                    v_var=avg_v_var,
                    ref_v_var=avg_ref_v_var,
                    lr=lr_scheduler.get_last_lr()[0],
                )
                if avg_keep_ratio is not None:
                    postfix["keep_ratio"] = avg_keep_ratio
                progress_bar.set_postfix(
                    **postfix,
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
                        generation_scheduler=generation_scheduler,
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

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unet_to_save = accelerator.unwrap_model(unet)
                        if args.use_lora:
                            unet_to_save.save_pretrained(os.path.join(save_path, "unet_lora"))
                        else:
                            # Also export a diffusers-style UNet folder for direct inference loading.
                            unet_to_save.save_pretrained(os.path.join(save_path, "unet"))
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()


def _checkpoint_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_checkpoint_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _checkpoint_json_safe(item) for key, item in value.items()}
    return str(value)


def checkpoint_step(checkpoint_dir: str | Path) -> int:
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "training_state.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            return int(metadata.get("global_step", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    name = checkpoint_path.name
    if name.startswith("checkpoint-"):
        suffix = name.removeprefix("checkpoint-")
        if suffix.isdigit():
            return int(suffix)
    return 0


def latest_checkpoint(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    candidates = [
        (step, path)
        for path in output_path.glob("checkpoint-*")
        if path.is_dir() and (step := checkpoint_step(path)) > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {output_path}.")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_resume_checkpoint(output_dir: str | Path, resume_from_checkpoint: str | None) -> Path | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        return latest_checkpoint(output_dir)

    checkpoint_path = Path(resume_from_checkpoint).expanduser()
    if not checkpoint_path.exists() and not checkpoint_path.is_absolute():
        output_relative = Path(output_dir) / checkpoint_path
        if output_relative.exists():
            checkpoint_path = output_relative
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume_from_checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def resume_training_state(accelerator, output_dir: str | Path, resume_from_checkpoint: str | None, logger=None) -> int:
    checkpoint_path = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    if checkpoint_path is None:
        return 0
    if logger is not None:
        logger.info("Resuming training state from %s", checkpoint_path)
    accelerator.load_state(str(checkpoint_path))
    step = checkpoint_step(checkpoint_path)
    if logger is not None:
        logger.info("Resumed training state at global step %s.", step)
    return step


def save_training_checkpoint(
    accelerator,
    unet,
    args,
    checkpoint_dir: str | Path,
    global_step: int,
    checkpoint_type: str,
    logger=None,
) -> None:
    checkpoint_path = Path(checkpoint_dir)
    if accelerator.is_main_process:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(checkpoint_path))

    if accelerator.is_main_process:
        unet_to_save = accelerator.unwrap_model(unet)
        if getattr(args, "use_lora", False):
            unet_to_save.save_pretrained(checkpoint_path / "unet_lora")
        else:
            unet_to_save.save_pretrained(checkpoint_path / "unet")

        metadata = {
            "checkpoint_type": checkpoint_type,
            "global_step": int(global_step),
            "max_train_steps": int(getattr(args, "max_train_steps", 0)),
            "checkpointing_steps": int(getattr(args, "checkpointing_steps", 0)),
            "resume_from_checkpoint": str(checkpoint_path),
            "contains_accelerate_state": True,
            "contains_optimizer_state": True,
            "contains_scheduler_state": True,
            "args": {key: _checkpoint_json_safe(value) for key, value in vars(args).items()},
        }
        with (checkpoint_path / "training_state.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        if logger is not None:
            logger.info("Saved %s checkpoint to %s", checkpoint_type, checkpoint_path)

    accelerator.wait_for_everyone()


def _resolve_processor_size(image_processor) -> int:
    for size_dict in (getattr(image_processor, "crop_size", None), getattr(image_processor, "size", None)):
        if isinstance(size_dict, dict):
            if "height" in size_dict:
                return int(size_dict["height"])
            if "shortest_edge" in size_dict:
                return int(size_dict["shortest_edge"])
        elif isinstance(size_dict, int):
            return int(size_dict)
    return 224


class FrozenDinoV2BFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        processor_name_or_path: str = None,
        feature_key: str = "layer12_patch_mean",
        block_stride: int = 3,
        patch_pool_sizes: Sequence[int] = (2, 4),
        local_files_only: bool = False,
    ):
        super().__init__()
        if processor_name_or_path is None:
            processor_name_or_path = model_name_or_path

        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_name_or_path,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.feature_key = feature_key
        self.block_stride = max(int(block_stride), 0)
        self.patch_pool_sizes = tuple(sorted({int(size) for size in patch_pool_sizes if int(size) > 1}))
        self.input_size = _resolve_processor_size(self.image_processor)
        image_mean = torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        image_std = torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("image_mean", image_mean, persistent=False)
        self.register_buffer("image_std", image_std, persistent=False)

        self.num_hidden_layers = int(getattr(self.model.config, "num_hidden_layers", 12))
        self.available_vector_feature_keys = set()
        for layer_idx in range(1, self.num_hidden_layers + 1):
            self._register_feature_base(f"layer{layer_idx}_patch", allow_patch_pool=True)
            self._register_feature_base(f"layer{layer_idx}_cls", allow_patch_pool=False)

        if self.feature_key not in self.available_vector_feature_keys:
            available = ", ".join(sorted(self.available_vector_feature_keys))
            raise ValueError(f"Unknown DINOv2 feature key: {self.feature_key}. Available keys: {available}")

    def _register_feature_base(self, base_name: str, allow_patch_pool: bool) -> None:
        self.available_vector_feature_keys.add(base_name)
        self.available_vector_feature_keys.add(f"{base_name}_mean")
        self.available_vector_feature_keys.add(f"{base_name}_std")
        if allow_patch_pool:
            for patch_size in self.patch_pool_sizes:
                self.available_vector_feature_keys.add(f"{base_name}_patch{patch_size}_mean")
                self.available_vector_feature_keys.add(f"{base_name}_patch{patch_size}_std")

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape (B, 3, H, W), got {tuple(images.shape)}")
        images = images.clamp(-1, 1)
        images = (images + 1.0) / 2.0
        if images.shape[-2] != self.input_size or images.shape[-1] != self.input_size:
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        images = (images - self.image_mean.to(dtype=images.dtype)) / self.image_std.to(dtype=images.dtype)
        return images

    def _extract_hidden_states(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        dtype = next(self.model.parameters()).dtype
        pixel_values = self._preprocess(images).to(dtype=dtype)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("DINOv2 model did not return hidden states.")
        return hidden_states

    def get_vector_features(self, images: torch.Tensor, feature_keys: Sequence[str] = None) -> Dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = self.get_default_multifeature_keys()
        requested = set(feature_keys)
        hidden_states = self._extract_hidden_states(images)
        vector_features: Dict[str, torch.Tensor] = {}

        for layer_idx in range(1, self.num_hidden_layers + 1):
            layer_prefix = f"layer{layer_idx}_"
            if not any(key.startswith(layer_prefix) for key in requested):
                continue
            hidden = hidden_states[layer_idx]
            cls_token = hidden[:, :1, :]
            patch_tokens = hidden[:, 1:, :]
            if patch_tokens.shape[1] == 0:
                raise ValueError(f"DINOv2 hidden state at layer {layer_idx} does not contain patch tokens.")

            patch_base = f"layer{layer_idx}_patch"
            cls_base = f"layer{layer_idx}_cls"

            if patch_base in requested:
                vector_features[patch_base] = patch_tokens
            if f"{patch_base}_mean" in requested:
                vector_features[f"{patch_base}_mean"] = patch_tokens.mean(dim=1, keepdim=True)
            if f"{patch_base}_std" in requested:
                vector_features[f"{patch_base}_std"] = patch_tokens.std(dim=1, keepdim=True)
            if cls_base in requested:
                vector_features[cls_base] = cls_token
            if f"{cls_base}_mean" in requested:
                vector_features[f"{cls_base}_mean"] = cls_token
            if f"{cls_base}_std" in requested:
                vector_features[f"{cls_base}_std"] = torch.zeros_like(cls_token)

            patch_pool_keys = [
                (patch_size, f"{patch_base}_patch{patch_size}_mean", f"{patch_base}_patch{patch_size}_std")
                for patch_size in self.patch_pool_sizes
            ]
            if any(mean_key in requested or std_key in requested for _, mean_key, std_key in patch_pool_keys):
                token_count = patch_tokens.shape[1]
                grid_size = int(round(math.sqrt(token_count)))
                if grid_size * grid_size != token_count:
                    continue
                patch_map = patch_tokens.transpose(1, 2).reshape(patch_tokens.shape[0], patch_tokens.shape[2], grid_size, grid_size)
                patch_map_sq = patch_map * patch_map
                for patch_size, mean_key, std_key in patch_pool_keys:
                    if grid_size < patch_size:
                        continue
                    if mean_key not in requested and std_key not in requested:
                        continue
                    pooled_mean = F.avg_pool2d(patch_map, kernel_size=patch_size, stride=patch_size)
                    if mean_key in requested:
                        vector_features[mean_key] = pooled_mean.flatten(2).transpose(1, 2).contiguous()
                    if std_key in requested:
                        pooled_sq_mean = F.avg_pool2d(patch_map_sq, kernel_size=patch_size, stride=patch_size)
                        pooled_var = torch.clamp(pooled_sq_mean - pooled_mean * pooled_mean, min=1e-8)
                        pooled_std = torch.sqrt(pooled_var)
                        vector_features[std_key] = pooled_std.flatten(2).transpose(1, 2).contiguous()

        missing_keys = [key for key in feature_keys if key not in vector_features]
        if missing_keys:
            available = ", ".join(sorted(vector_features.keys()))
            raise ValueError(f"Requested DINOv2 feature keys not available: {missing_keys}. Available keys: {available}")
        return {key: vector_features[key] for key in feature_keys}

    def get_default_multifeature_keys(self) -> Tuple[str, ...]:
        keys: List[str] = []
        selected_layers = list(range(1, self.num_hidden_layers + 1))
        if self.block_stride > 0:
            selected_layers = [idx for idx in selected_layers if idx % self.block_stride == 0]
        if self.num_hidden_layers not in selected_layers:
            selected_layers.append(self.num_hidden_layers)
        for layer_idx in selected_layers:
            patch_base = f"layer{layer_idx}_patch"
            cls_base = f"layer{layer_idx}_cls"
            keys.extend([f"{patch_base}_mean", f"{patch_base}_std", cls_base])
        return tuple(keys)

    def _expand_multifeature_names(self, base_name: str) -> List[str]:
        names = [base_name, f"{base_name}_mean", f"{base_name}_std"]
        for patch_size in self.patch_pool_sizes:
            names.append(f"{base_name}_patch{patch_size}_mean")
            names.append(f"{base_name}_patch{patch_size}_std")
        return names

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        vectors = self.get_vector_features(images, feature_keys=(self.feature_key,))
        value = vectors[self.feature_key]
        if value.shape[1] == 1:
            return value[:, 0, :]
        return value.flatten(1)

