from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from drpo.utils.tensors import choose_gn_groups, safe_std, strip_state_dict_prefixes


def load_mae_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
        metadata = checkpoint.get("metadata", {})
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {})
    elif isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(f"Unsupported MAE checkpoint format: {path}")
    return strip_state_dict_prefixes(state_dict), metadata


class FrozenMAELatentFeatureExtractor(nn.Module):
    """Frozen ResNet-style feature extractor exported from the MAE latent model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        feature_key: str = "layer4_mean",
        block_stride: int = 2,
        patch_pool_sizes: Sequence[int] = (2, 4),
        include_input_sq_mean: bool = False,
        include_spatial_features: bool = False,
    ) -> None:
        super().__init__()
        self.params, metadata = load_mae_checkpoint(checkpoint_path)
        config = metadata.get("model_config", {}) if isinstance(metadata, dict) else {}
        self.layers = tuple(int(x) for x in config.get("layers", [3, 4, 6, 3]))
        self.feature_key = feature_key
        self.block_stride = max(int(block_stride), 0)
        self.patch_pool_sizes = tuple(sorted({int(size) for size in patch_pool_sizes if int(size) > 1}))
        self.include_input_sq_mean = include_input_sq_mean
        self.include_spatial_features = include_spatial_features
        conv1 = self.params.get("encoder.conv1.kernel", self.params.get("encoder.conv1.weight"))
        if conv1 is None:
            raise KeyError("MAE checkpoint is missing encoder.conv1 weights.")
        self.in_channels = int(config.get("in_channels", conv1.shape[2] if conv1.shape[2] <= 16 else conv1.shape[1]))
        self.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.params = {key: value.to(*args, **kwargs) for key, value in self.params.items()}
        return self

    def _conv2d(self, x: torch.Tensor, prefix: str, stride: int = 1, padding: int = 1) -> torch.Tensor:
        kernel = self.params.get(f"{prefix}.kernel", self.params.get(f"{prefix}.weight"))
        if kernel is None:
            raise KeyError(f"Missing convolution weights for {prefix}.")
        if kernel.shape[2] == x.shape[1]:
            weight = kernel.permute(3, 2, 0, 1).contiguous()
        elif kernel.shape[1] == x.shape[1]:
            weight = kernel.contiguous()
        else:
            raise ValueError(f"Cannot infer kernel layout for {prefix}: {tuple(kernel.shape)}")
        return F.conv2d(x, weight, bias=None, stride=stride, padding=padding)

    def _group_norm(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        weight = self.params.get(f"{prefix}.scale", self.params.get(f"{prefix}.weight"))
        bias = self.params.get(f"{prefix}.bias")
        if weight is None or bias is None:
            raise KeyError(f"Missing group norm weights for {prefix}.")
        return F.group_norm(x, choose_gn_groups(x.shape[1]), weight, bias, eps=1e-5)

    def _block(self, x: torch.Tensor, prefix: str, stride: int) -> torch.Tensor:
        residual = x
        y = F.relu(self._group_norm(self._conv2d(x, f"{prefix}.conv1", stride=stride), f"{prefix}.gn1"))
        y = self._group_norm(self._conv2d(y, f"{prefix}.conv2"), f"{prefix}.gn2")
        if residual.shape != y.shape:
            residual = self._group_norm(self._conv2d(residual, f"{prefix}.proj_conv", stride=stride, padding=0), f"{prefix}.proj_gn")
        return F.relu(residual + y)

    def spatial_features(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        if latents.ndim != 4:
            raise ValueError(f"Expected latents with shape (B, C, H, W), got {tuple(latents.shape)}")
        if latents.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {latents.shape[1]}.")
        first_kernel = self.params.get("encoder.conv1.kernel", self.params.get("encoder.conv1.weight"))
        if first_kernel is not None and latents.dtype != first_kernel.dtype:
            latents = latents.to(dtype=first_kernel.dtype)
        features: dict[str, torch.Tensor] = {}
        x = F.relu(self._group_norm(self._conv2d(latents, "encoder.conv1"), "encoder.gn1"))
        features["conv1"] = x
        for stage_index, blocks in enumerate(self.layers):
            for block_index in range(blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                x = self._block(x, f"encoder.stages_{stage_index}.layers_{block_index}", stride)
                block_number = block_index + 1
                if self.block_stride and block_number % self.block_stride == 0 and block_number < blocks:
                    features[f"layer{stage_index + 1}_blk{block_number}"] = x
            name = f"layer{stage_index + 1}"
            x = self._group_norm(x, f"encoder.{name}_norm")
            features[name] = x
        return features

    def vector_features(self, latents: torch.Tensor, keys: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        if self.include_input_sq_mean:
            features["input_x2_mean"] = latents.square().mean(dim=(2, 3)).unsqueeze(1)
            features["norm_x"] = latents.float().square().mean(dim=(2, 3)).clamp_min(0.0).sqrt().to(latents.dtype).unsqueeze(1)
        for name, value in self.spatial_features(latents).items():
            batch, channels, height, width = value.shape
            if self.include_spatial_features:
                features[name] = value.flatten(2).transpose(1, 2).contiguous()
            features[f"{name}_mean"] = value.mean(dim=(2, 3)).unsqueeze(1)
            features[f"{name}_std"] = safe_std(value, dim=(2, 3)).unsqueeze(1)
            for patch_size in self.patch_pool_sizes:
                if height % patch_size or width % patch_size:
                    continue
                patches = value.view(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
                patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch_size * patch_size, channels)
                features[f"{name}_mean_{patch_size}"] = patches.mean(dim=2)
                features[f"{name}_std_{patch_size}"] = safe_std(patches, dim=2)
        if keys is None:
            return features
        missing = [key for key in keys if key not in features]
        if missing:
            raise ValueError(f"Missing feature keys {missing}. Available: {sorted(features)}")
        return {key: features[key] for key in keys}

    def default_feature_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        if self.include_input_sq_mean:
            keys.append("norm_x")
        for stage_index, blocks in enumerate(self.layers):
            names = [f"layer{stage_index + 1}"]
            if self.block_stride:
                names.extend(
                    f"layer{stage_index + 1}_blk{block}"
                    for block in range(self.block_stride, blocks, self.block_stride)
                )
            for name in names:
                if self.include_spatial_features:
                    keys.append(name)
                keys.extend([f"{name}_mean", f"{name}_std"])
                for patch_size in self.patch_pool_sizes:
                    keys.extend([f"{name}_mean_{patch_size}", f"{name}_std_{patch_size}"])
        return tuple(keys)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        value = self.vector_features(latents, keys=(self.feature_key,))[self.feature_key]
        return value[:, 0, :] if value.shape[1] == 1 else value.flatten(1)


class DirectLatentFeatureExtractor(nn.Module):
    """Use VAE latents directly as DrPO features."""

    def __init__(
        self,
        *,
        feature_key: str = "latent",
        patch_pool_sizes: Sequence[int] = (2, 4),
        include_spatial_features: bool = True,
        spatial_stride: int | None = None,
    ) -> None:
        super().__init__()
        self.feature_key = feature_key
        self.patch_pool_sizes = tuple(sorted({int(size) for size in patch_pool_sizes if int(size) > 1}))
        self.include_spatial_features = include_spatial_features
        if spatial_stride is None:
            spatial_stride = int(os.environ.get("DRPO_LATENT_SPATIAL_STRIDE", "1"))
        self.spatial_stride = max(int(spatial_stride), 1)

    def vector_features(self, latents: torch.Tensor, keys: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        if latents.ndim != 4:
            raise ValueError(f"Expected latents with shape (B, C, H, W), got {tuple(latents.shape)}")
        batch, channels, height, width = latents.shape
        features: dict[str, torch.Tensor] = {}
        if self.include_spatial_features:
            latent_spatial = latents[:, :, :: self.spatial_stride, :: self.spatial_stride].contiguous()
            features["latent"] = latent_spatial.flatten(2).transpose(1, 2).contiguous()
        features["latent_mean"] = latents.mean(dim=(2, 3)).unsqueeze(1)
        features["latent_std"] = safe_std(latents, dim=(2, 3)).unsqueeze(1)
        features["latent_norm"] = latents.float().square().mean(dim=(2, 3)).clamp_min(0.0).sqrt().to(latents.dtype).unsqueeze(1)
        for patch_size in self.patch_pool_sizes:
            if height % patch_size or width % patch_size:
                continue
            patches = latents.view(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
            patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch_size * patch_size, channels)
            features[f"latent_mean_{patch_size}"] = patches.mean(dim=2)
            features[f"latent_std_{patch_size}"] = safe_std(patches, dim=2)
        if keys is None:
            return features
        missing = [key for key in keys if key not in features]
        if missing:
            raise ValueError(f"Missing latent feature keys {missing}. Available: {sorted(features)}")
        return {key: features[key] for key in keys}

    def default_feature_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        if self.include_spatial_features:
            keys.append("latent")
        keys.extend(["latent_mean", "latent_std"])
        for patch_size in self.patch_pool_sizes:
            keys.extend([f"latent_mean_{patch_size}", f"latent_std_{patch_size}"])
        return tuple(keys)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        value = self.vector_features(latents, keys=(self.feature_key,))[self.feature_key]
        return value[:, 0, :] if value.shape[1] == 1 else value.flatten(1)


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


class FrozenDinoImageFeatureExtractor(nn.Module):
    """Frozen DINO/DINOv2 image feature extractor with MAE-style vector features."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        processor_name_or_path: str | Path | None = None,
        feature_key: str = "layer12_patch_mean",
        block_stride: int = 3,
        patch_pool_sizes: Sequence[int] = (2, 4),
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        processor_name_or_path = processor_name_or_path or model_name_or_path
        self.image_processor = AutoImageProcessor.from_pretrained(
            str(processor_name_or_path),
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            str(model_name_or_path),
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.feature_key = feature_key
        self.block_stride = max(int(block_stride), 0)
        self.patch_pool_sizes = tuple(sorted({int(size) for size in patch_pool_sizes if int(size) > 1}))
        self.input_size = _resolve_processor_size(self.image_processor)
        self.num_hidden_layers = int(getattr(self.model.config, "num_hidden_layers", 12))
        image_mean = torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        image_std = torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("image_mean", image_mean, persistent=False)
        self.register_buffer("image_std", image_std, persistent=False)
        self.requires_grad_(False)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape (B, 3, H, W), got {tuple(images.shape)}")
        images = images.clamp(-1, 1)
        images = (images + 1.0) / 2.0
        if images.shape[-2] != self.input_size or images.shape[-1] != self.input_size:
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        return (images - self.image_mean.to(device=images.device, dtype=images.dtype)) / self.image_std.to(device=images.device, dtype=images.dtype)

    def _extract_hidden_states(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        dtype = next(self.model.parameters()).dtype
        pixel_values = self._preprocess(images).to(dtype=dtype)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("DINO model did not return hidden states.")
        return tuple(hidden_states)

    def vector_features(self, images: torch.Tensor, keys: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = self.default_feature_keys()
        requested = set(keys)
        hidden_states = self._extract_hidden_states(images)
        features: dict[str, torch.Tensor] = {}
        for layer_idx in range(1, self.num_hidden_layers + 1):
            layer_prefix = f"layer{layer_idx}_"
            if not any(key.startswith(layer_prefix) for key in requested):
                continue
            hidden = hidden_states[layer_idx]
            cls_token = hidden[:, :1, :]
            patch_tokens = hidden[:, 1:, :]
            if patch_tokens.shape[1] == 0:
                raise ValueError(f"DINO hidden state at layer {layer_idx} does not contain patch tokens.")

            patch_base = f"layer{layer_idx}_patch"
            cls_base = f"layer{layer_idx}_cls"
            if patch_base in requested:
                features[patch_base] = patch_tokens
            if f"{patch_base}_mean" in requested:
                features[f"{patch_base}_mean"] = patch_tokens.mean(dim=1, keepdim=True)
            if f"{patch_base}_std" in requested:
                features[f"{patch_base}_std"] = safe_std(patch_tokens, dim=1).unsqueeze(1)
            if cls_base in requested:
                features[cls_base] = cls_token
            if f"{cls_base}_mean" in requested:
                features[f"{cls_base}_mean"] = cls_token
            if f"{cls_base}_std" in requested:
                features[f"{cls_base}_std"] = torch.zeros_like(cls_token)

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
                patch_map_sq = patch_map.square()
                for patch_size, mean_key, std_key in patch_pool_keys:
                    if grid_size < patch_size:
                        continue
                    if mean_key not in requested and std_key not in requested:
                        continue
                    pooled_mean = F.avg_pool2d(patch_map, kernel_size=patch_size, stride=patch_size)
                    if mean_key in requested:
                        features[mean_key] = pooled_mean.flatten(2).transpose(1, 2).contiguous()
                    if std_key in requested:
                        pooled_sq_mean = F.avg_pool2d(patch_map_sq, kernel_size=patch_size, stride=patch_size)
                        pooled_std = torch.clamp(pooled_sq_mean - pooled_mean.square(), min=1e-8).sqrt()
                        features[std_key] = pooled_std.flatten(2).transpose(1, 2).contiguous()

        missing = [key for key in keys if key not in features]
        if missing:
            raise ValueError(f"Missing DINO feature keys {missing}. Available: {sorted(features)}")
        return {key: features[key] for key in keys}

    def default_feature_keys(self) -> tuple[str, ...]:
        selected_layers = list(range(1, self.num_hidden_layers + 1))
        if self.block_stride > 0:
            selected_layers = [idx for idx in selected_layers if idx % self.block_stride == 0]
        if self.num_hidden_layers not in selected_layers:
            selected_layers.append(self.num_hidden_layers)
        keys: list[str] = []
        for layer_idx in selected_layers:
            patch_base = f"layer{layer_idx}_patch"
            cls_base = f"layer{layer_idx}_cls"
            keys.extend([f"{patch_base}_mean", f"{patch_base}_std", cls_base])
        return tuple(keys)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = self.vector_features(images, keys=(self.feature_key,))[self.feature_key]
        return value[:, 0, :] if value.shape[1] == 1 else value.flatten(1)
