from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


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

        size_cfg = getattr(image_processor, "crop_size", None)
        if hasattr(size_cfg, "get"):
            self.image_size = self._resolve_image_size(size_cfg, ("height", "width"))
        elif isinstance(size_cfg, int):
            self.image_size = int(size_cfg)
        else:
            size_cfg = getattr(image_processor, "size", {"shortest_edge": 224})
            if hasattr(size_cfg, "get"):
                self.image_size = self._resolve_image_size(size_cfg, ("shortest_edge", "height", "width"))
            else:
                self.image_size = int(size_cfg)

    @staticmethod
    def _resolve_image_size(size_cfg, keys: tuple[str, ...]) -> int:
        for key in keys:
            value = size_cfg.get(key)
            if value is not None:
                return int(value)
        return 224

    @staticmethod
    def _to_tensor(out):
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
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
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
