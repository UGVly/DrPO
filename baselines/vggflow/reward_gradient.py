"""Reward-gradient helpers for one-step VGG-Flow.

This mirrors the official VGG-Flow core idea in the SD-Turbo one-step setting:
compute a differentiable reward gradient in clean-latent space, clip it by a
running norm threshold, and build a reference-plus-gradient target.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from utils.aes_utils import AestheticMLP
from utils.reward_common import normalize_features, strip_state_dict_prefixes, to_feature_tensor
from baselines.common import decode_latents_to_tensor


def eta_multiplier(mode: str, sigma: torch.Tensor | float) -> torch.Tensor:
    sigma_t = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(float(sigma))
    if mode == "constant":
        return torch.ones_like(sigma_t)
    if mode == "linear":
        return 1.0 - sigma_t
    if mode == "quad":
        return (1.0 - sigma_t).pow(2)
    raise ValueError(f"Unsupported eta_mode: {mode}")


def clip_gradient_by_norm(gradient: torch.Tensor, threshold: float, enabled: bool = True) -> torch.Tensor:
    if not enabled or threshold <= 0:
        return gradient
    grad_norm = torch.linalg.norm(gradient.float().flatten(1), dim=1).clamp_min(1e-8)
    scale = grad_norm.clamp(max=float(threshold)) / grad_norm
    return gradient * scale.view(-1, 1, 1, 1).to(dtype=gradient.dtype)


@dataclass
class RewardGradientOutput:
    gradient: torch.Tensor
    gradient_norm: torch.Tensor
    rewards: torch.Tensor
    reward_mask: torch.Tensor


class DifferentiablePickScoreReward(nn.Module):
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str,
        processor_name_or_path: str,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("Expected PickScore AutoProcessor with an image_processor.")

        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))

        size_cfg = getattr(image_processor, "crop_size", None) or getattr(image_processor, "size", 224)
        if hasattr(size_cfg, "get"):
            self.image_size = int(size_cfg.get("height") or size_cfg.get("width") or size_cfg.get("shortest_edge") or 224)
        else:
            self.image_size = int(size_cfg)

    def _to_tensor(self, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        if getattr(output, "last_hidden_state", None) is not None:
            return output.last_hidden_state[:, 0, :]
        raise TypeError(f"Unsupported feature output type: {type(output)}")

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
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        with torch.no_grad():
            text_features = self._to_tensor(self.model.get_text_features(**text_inputs))
            text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        return text_features.detach()

    def forward(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = self.preprocess_images(images).to(device=self.device, dtype=model_dtype)
        text_features = self.encode_text(prompts)
        image_features = self._to_tensor(self.model.get_image_features(pixel_values=pixel_values))
        image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        return (self.model.logit_scale.exp() * torch.sum(text_features * image_features, dim=-1)).float()


class DifferentiableAestheticReward(nn.Module):
    def __init__(
        self,
        device: torch.device,
        clip_model_path: str,
        ckpt_path: str,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(
            clip_model_path,
            local_files_only=local_files_only,
        )
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(
            clip_model_path,
            local_files_only=local_files_only,
            torch_dtype=model_dtype,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("Expected CLIPProcessor with an image_processor.")
        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))

        size_cfg = getattr(image_processor, "crop_size", None) or getattr(image_processor, "size", 224)
        if hasattr(size_cfg, "get"):
            self.image_size = int(size_cfg.get("height") or size_cfg.get("width") or size_cfg.get("shortest_edge") or 224)
        else:
            self.image_size = int(size_cfg)

        projection_dim = int(getattr(self.model.config, "projection_dim", 768))
        self.head = AestheticMLP(input_size=projection_dim).eval().to(self.device, dtype=torch.float32)
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported aesthetic checkpoint format: {ckpt_path}")
        self.head.load_state_dict(strip_state_dict_prefixes(state), strict=True)
        self.head.requires_grad_(False)

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

    def forward(self, images: torch.Tensor, prompts: Sequence[str] | None = None) -> torch.Tensor:
        del prompts
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = self.preprocess_images(images).to(device=self.device, dtype=model_dtype)
        image_features = to_feature_tensor(self.model.get_image_features(pixel_values=pixel_values))
        image_features = normalize_features(image_features.float())
        return self.head(image_features).flatten().float()


def compute_pickscore_reward_gradient(
    *,
    vae,
    reward_model: DifferentiablePickScoreReward,
    latents: torch.Tensor,
    prompts: Sequence[str],
    clip_threshold: float,
    quantile_clipping: bool,
    reward_mask_threshold: float,
    jitter_count: int = 1,
    jitter_std: float = 0.0,
    decode_chunk_size: int = 4,
) -> RewardGradientOutput:
    if len(prompts) != latents.shape[0]:
        raise ValueError(f"Prompt count {len(prompts)} does not match latent batch {latents.shape[0]}.")
    if jitter_count < 1:
        raise ValueError("jitter_count must be >= 1.")

    latent_leaf = latents.detach().float()
    if jitter_count > 1 or jitter_std > 0:
        jittered_latents = (
            latent_leaf.unsqueeze(1)
            + float(jitter_std) * torch.randn(
                latent_leaf.shape[0],
                jitter_count,
                *latent_leaf.shape[1:],
                device=latent_leaf.device,
                dtype=latent_leaf.dtype,
            )
        ).reshape(-1, *latent_leaf.shape[1:])
        reward_prompts = [prompt for prompt in prompts for _ in range(jitter_count)]
    else:
        jittered_latents = latent_leaf
        reward_prompts = list(prompts)

    flat_latents = jittered_latents.reshape(-1, *latent_leaf.shape[1:])
    reward_chunks = []
    gradient_chunks = []
    for start in range(0, flat_latents.shape[0], decode_chunk_size):
        end = min(flat_latents.shape[0], start + decode_chunk_size)
        chunk_leaf = flat_latents[start:end].detach().float().requires_grad_(True)
        images = decode_latents_to_tensor(
            vae,
            chunk_leaf,
            chunk_size=decode_chunk_size,
        )
        rewards_chunk = reward_model(images, reward_prompts[start:end])
        gradient_chunk = torch.autograd.grad(rewards_chunk.sum(), chunk_leaf, retain_graph=False)[0]
        reward_chunks.append(rewards_chunk.detach())
        gradient_chunks.append(gradient_chunk.detach().float())
    rewards_flat = torch.cat(reward_chunks, dim=0)
    gradient_flat = torch.cat(gradient_chunks, dim=0)
    rewards = rewards_flat.view(latent_leaf.shape[0], jitter_count).mean(dim=1)
    gradient = gradient_flat.view(latent_leaf.shape[0], jitter_count, *latent_leaf.shape[1:]).mean(dim=1)
    gradient_norm = torch.linalg.norm(gradient.flatten(1), dim=1)
    clipped = clip_gradient_by_norm(gradient, clip_threshold, enabled=quantile_clipping)
    reward_mask = (rewards.detach() >= float(reward_mask_threshold)).float()

    return RewardGradientOutput(
        gradient=clipped,
        gradient_norm=gradient_norm.detach(),
        rewards=rewards.detach().float(),
        reward_mask=reward_mask,
    )
