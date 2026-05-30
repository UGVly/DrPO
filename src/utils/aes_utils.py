import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .reward_common import (
    PROJECT_ROOT,
    choose_model_dtype,
    normalize_features,
    strip_state_dict_prefixes,
    to_feature_tensor,
)

DEFAULT_AES_CKPT = PROJECT_ROOT / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth"
DEFAULT_LOCAL_CLIP_MODEL = PROJECT_ROOT / "models" / "CLIP-ViT-L-14"
DEFAULT_REMOTE_CLIP_MODEL = "openai/clip-vit-large-patch14"

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _resolve_clip_model_path(value: str | None, local_files_only: bool) -> str:
    if value:
        return value

    if DEFAULT_LOCAL_CLIP_MODEL.is_dir():
        return str(DEFAULT_LOCAL_CLIP_MODEL)

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--clip-vit-large-patch14"
        / "snapshots"
    )
    if cache_root.is_dir():
        snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

    if local_files_only:
        raise FileNotFoundError(
            "CLIP model not found in local HuggingFace cache. "
            "Set AESTHETIC_CLIP_MODEL_PATH or pass clip_model_path explicitly."
        )
    return DEFAULT_REMOTE_CLIP_MODEL


class Selector:
    def __init__(
        self,
        device,
        clip_model_path: str | None = None,
        ckpt_path: str | os.PathLike[str] | None = None,
        local_files_only: bool = True,
    ):
        self.device = torch.device(device)
        self.local_files_only = local_files_only
        self.clip_model_path = _resolve_clip_model_path(
            clip_model_path or os.getenv("AESTHETIC_CLIP_MODEL_PATH"),
            local_files_only=local_files_only,
        )
        resolved_ckpt = Path(ckpt_path or os.getenv("AESTHETIC_CKPT_PATH") or DEFAULT_AES_CKPT)
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(
                f"Aesthetic checkpoint not found: {resolved_ckpt}. "
                "Set AESTHETIC_CKPT_PATH or place the file under models/Aesthetic/."
            )
        self.ckpt_path = resolved_ckpt

        self.processor = CLIPProcessor.from_pretrained(
            self.clip_model_path,
            local_files_only=self.local_files_only,
        )
        model_dtype = choose_model_dtype(self.device)
        self.model = CLIPModel.from_pretrained(
            self.clip_model_path,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
        ).eval().to(self.device)

        projection_dim = int(self.model.config.projection_dim)
        self.head = AestheticMLP(input_size=projection_dim)
        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported aesthetic checkpoint format: {self.ckpt_path}")
        self.head.load_state_dict(strip_state_dict_prefixes(state), strict=True)
        self.head.eval().to(device=self.device, dtype=torch.float32)

    def score(self, images: Sequence[Image.Image], prompt=None) -> list[float]:
        del prompt
        if not images:
            return []

        image_inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        with torch.no_grad():
            image_embs = to_feature_tensor(self.model.get_image_features(pixel_values=pixel_values))
            image_embs = normalize_features(image_embs)
            scores = self.head(image_embs.float()).flatten()
        return scores.detach().float().cpu().tolist()
