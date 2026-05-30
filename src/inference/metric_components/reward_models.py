import os
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from drpo.paths import project_root, require_local_path
from drpo.rewards import AestheticHead, build_selector
from drpo.utils.tensors import feature_tensor, normalize_features, strip_state_dict_prefixes
from inference.metric_components.common import batched, close_images, num_batches, open_images


def score_with_selector(
    rows: list[dict[str, Any]],
    *,
    selector_name: str,
    device: str,
    batch_size: int,
    image_load_workers: int = 0,
    selector: Any | None = None,
) -> None:
    selector = selector or build_selector(selector_name, device)
    key = "hpsv2" if selector_name == "hpsv2" else selector_name
    for batch in tqdm(batched(rows, batch_size), total=num_batches(len(rows), batch_size), desc=f"score {key}", dynamic_ncols=True):
        images = open_images(batch, num_workers=image_load_workers)
        prompts = [str(row["prompt"]) for row in batch]
        try:
            scores = selector.score(images, prompts)
        finally:
            close_images(images)
        for row, score in zip(batch, scores):
            row[key] = float(score)


def build_clip_aes_models(device: str) -> tuple[Any, Any, AestheticHead, torch.device]:
    from packaging.version import Version
    from transformers import CLIPModel, CLIPProcessor
    from transformers import __version__ as transformers_version

    root = project_root()
    clip_path = require_local_path(
        os.getenv("CLIP_MODEL_PATH") or root / "models" / "CLIP-ViT-L-14",
        description="CLIP reward model",
        must_be_file=False,
    )
    aesthetic_head_path = require_local_path(
        os.getenv("AES_HEAD_PATH") or root / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth",
        description="Aesthetic reward head",
        must_be_file=True,
    )
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    dtype_key = "dtype" if Version(transformers_version).major >= 5 else "torch_dtype"
    processor = CLIPProcessor.from_pretrained(str(clip_path), local_files_only=True)
    model = CLIPModel.from_pretrained(str(clip_path), **{dtype_key: dtype}, local_files_only=True).to(device_obj).eval()
    projection_dim = int(getattr(model.config, "projection_dim", 768))
    head = AestheticHead(projection_dim).to(device_obj).eval()
    head.load_state_dict(strip_state_dict_prefixes(torch.load(aesthetic_head_path, map_location="cpu")))
    return processor, model, head, device_obj


@torch.no_grad()
def score_clip_aes(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    processor: Any,
    model: Any,
    head: AestheticHead,
    device_obj: torch.device,
    image_load_workers: int = 0,
) -> None:
    for batch in tqdm(batched(rows, batch_size), total=num_batches(len(rows), batch_size), desc="score clip/aes", dynamic_ncols=True):
        images = open_images(batch, num_workers=image_load_workers)
        prompts = [str(row["prompt"]) for row in batch]
        try:
            image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device_obj)
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device_obj)
            image_features = normalize_features(feature_tensor(model.get_image_features(**image_inputs)).float())
            text_features = normalize_features(feature_tensor(model.get_text_features(**text_inputs)).float())
            clip_scores = (image_features * text_features).sum(dim=-1).float().cpu().tolist()
            aes_scores = head(image_features).flatten().float().cpu().tolist()
        finally:
            close_images(images)
        for row, clip_score, aes_score in zip(batch, clip_scores, aes_scores):
            row["clip"] = float(clip_score)
            row["aes"] = float(aes_score)
