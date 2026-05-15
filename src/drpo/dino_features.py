from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


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

