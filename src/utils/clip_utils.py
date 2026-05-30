import os
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .reward_common import PROJECT_ROOT, choose_model_dtype, normalize_features, normalize_prompts, to_feature_tensor


DEFAULT_CLIP_MODEL = PROJECT_ROOT / "models" / "CLIP-ViT-L-14"

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_model_path(value: str | None) -> str:
    resolved = Path(value or os.getenv("CLIP_REWARD_MODEL_PATH") or DEFAULT_CLIP_MODEL)
    if not resolved.exists():
        raise FileNotFoundError(
            f"CLIP reward model path not found: {resolved}. "
            "Place the model under models/CLIP-ViT-L-14 or set CLIP_REWARD_MODEL_PATH."
        )
    return str(resolved)


class Selector:
    def __init__(
        self,
        device,
        model_name_or_path: str | None = None,
        local_files_only: bool = True,
    ):
        self.device = torch.device(device)
        self.local_files_only = local_files_only
        self.model_name_or_path = _resolve_model_path(model_name_or_path)
        model_dtype = choose_model_dtype(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
        ).eval().to(self.device)

    def score(self, images: Sequence[Image.Image], prompt: str | Sequence[str]) -> list[float]:
        if not images:
            return []

        prompts = normalize_prompts(prompt, len(images))
        image_inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            image_embs = normalize_features(
                to_feature_tensor(self.model.get_image_features(pixel_values=image_inputs["pixel_values"]))
            )
            text_embs = normalize_features(
                to_feature_tensor(
                    self.model.get_text_features(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                    )
                )
            )
            scores = torch.sum(text_embs * image_embs, dim=-1)
        return scores.detach().float().cpu().tolist()
