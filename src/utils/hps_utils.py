from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

import open_clip

from .reward_common import PROJECT_ROOT, normalize_prompts


DEFAULT_OPEN_CLIP_WEIGHTS = (
    PROJECT_ROOT / "models" / "CLIP-ViT-H-14-laion2B-s32B-b79K" / "open_clip_pytorch_model.bin"
)
DEFAULT_HPS_CKPT = PROJECT_ROOT / "models" / "HPSv2" / "HPS_v2_compressed.pt"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_existing_path(value: str | os.PathLike[str] | None, default_path: Path, env_name: str) -> Path:
    resolved = Path(value or os.getenv(env_name) or default_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected file at {resolved}. Override with {env_name}.")
    return resolved


class Selector:
    def __init__(
        self,
        device,
        open_clip_pretrained_path: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ):
        self.device = torch.device(device)
        self.open_clip_pretrained_path = _resolve_existing_path(
            open_clip_pretrained_path,
            DEFAULT_OPEN_CLIP_WEIGHTS,
            "HPS_OPEN_CLIP_PRETRAINED_PATH",
        )
        self.checkpoint_path = _resolve_existing_path(
            checkpoint_path,
            DEFAULT_HPS_CKPT,
            "HPS_CKPT_PATH",
        )
        precision = "amp" if self.device.type == "cuda" else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(self.open_clip_pretrained_path),
            precision=precision,
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            output_dict=True,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def score(self, images: Sequence[Image.Image], prompt: str | Sequence[str]) -> list[float]:
        if not images:
            return []

        prompts = normalize_prompts(prompt, len(images))
        image_tensor = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        text_tensor = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor, text_tensor)
            scores = torch.sum(outputs["image_features"] * outputs["text_features"], dim=-1)
        return scores.detach().float().cpu().tolist()
