from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from packaging.version import Version
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, __version__ as transformers_version

from drpo.paths import project_root, require_local_path
from drpo.utils.tensors import feature_tensor, normalize_features, normalize_prompts, strip_state_dict_prefixes


def _dtype_for(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def _dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    key = "dtype" if Version(transformers_version).major >= 5 else "torch_dtype"
    return {key: dtype}


class RewardSelector(ABC):
    @abstractmethod
    def score(self, images: Sequence[Image.Image], prompts: Any) -> list[float]:
        """Return one scalar reward per image."""


class PickScoreSelector(RewardSelector):
    def __init__(self, device, model_path: str | Path | None = None, processor_path: str | Path | None = None, local_files_only: bool = True):
        self.device = torch.device(device)
        root = project_root()
        self.model_path = str(require_local_path(model_path or os.getenv("PICKSCORE_MODEL_PATH") or root / "models" / "PickScore_v1", description="PickScore model", must_be_file=False))
        self.processor_path = str(require_local_path(processor_path or self.model_path, description="PickScore processor", must_be_file=False))
        dtype = _dtype_for(self.device)
        self.processor = AutoProcessor.from_pretrained(self.processor_path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(self.model_path, **_dtype_kwargs(dtype), local_files_only=local_files_only).to(self.device).eval()
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("Expected PickScore processor to expose image_processor.")
        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        size_cfg = getattr(image_processor, "crop_size", None)
        if hasattr(size_cfg, "get"):
            self.image_size = int(size_cfg.get("height") or size_cfg.get("width") or 224)
        elif isinstance(size_cfg, int):
            self.image_size = int(size_cfg)
        else:
            size_cfg = getattr(image_processor, "size", {"shortest_edge": 224})
            if hasattr(size_cfg, "get"):
                self.image_size = int(size_cfg.get("shortest_edge") or size_cfg.get("height") or size_cfg.get("width") or 224)
            else:
                self.image_size = int(size_cfg or 224)

    def _preprocess_tensor_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected image tensor with shape (B, 3, H, W), got {tuple(images.shape)}")
        images = ((images.clamp(-1, 1) + 1.0) / 2.0).to(device=self.device, dtype=torch.float32)
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        image_mean = self.image_mean.to(device=self.device, dtype=images.dtype)
        image_std = self.image_std.to(device=self.device, dtype=images.dtype)
        model_dtype = next(self.model.parameters()).dtype
        return ((images - image_mean) / image_std).to(dtype=model_dtype)

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(images))
        image_inputs = self.processor(images=list(images), padding=True, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        image_features = normalize_features(feature_tensor(self.model.get_image_features(**image_inputs)).float())
        text_features = normalize_features(feature_tensor(self.model.get_text_features(**text_inputs)).float())
        return (image_features * text_features).sum(dim=-1).float().cpu().tolist()

    @torch.no_grad()
    def score_tensor(self, images: torch.Tensor, prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, int(images.shape[0]))
        pixel_values = self._preprocess_tensor_images(images)
        text_inputs = self.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        image_features = normalize_features(feature_tensor(self.model.get_image_features(pixel_values=pixel_values)).float())
        text_features = normalize_features(feature_tensor(self.model.get_text_features(**text_inputs)).float())
        return (image_features * text_features).sum(dim=-1).float().cpu().tolist()


class CLIPSelector(RewardSelector):
    def __init__(self, device, model_path: str | Path | None = None, local_files_only: bool = True):
        self.device = torch.device(device)
        root = project_root()
        self.model_path = str(require_local_path(model_path or os.getenv("CLIP_REWARD_MODEL_PATH") or root / "models" / "CLIP-ViT-L-14", description="CLIP reward model", must_be_file=False))
        dtype = _dtype_for(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_path, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(self.model_path, **_dtype_kwargs(dtype), local_files_only=local_files_only).to(self.device).eval()

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(images))
        image_inputs = self.processor(images=list(images), return_tensors="pt", padding=True).to(self.device)
        text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        image_features = normalize_features(feature_tensor(self.model.get_image_features(**image_inputs)).float())
        text_features = normalize_features(feature_tensor(self.model.get_text_features(**text_inputs)).float())
        return (image_features * text_features).sum(dim=-1).float().cpu().tolist()


class AestheticHead(nn.Module):
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


class AestheticSelector(RewardSelector):
    def __init__(self, device, clip_model_path: str | Path | None = None, head_path: str | Path | None = None, local_files_only: bool = True):
        self.device = torch.device(device)
        root = project_root()
        self.clip_model_path = str(require_local_path(clip_model_path or os.getenv("AESTHETIC_CLIP_MODEL_PATH") or root / "models" / "CLIP-ViT-L-14", description="Aesthetic CLIP model", must_be_file=False))
        self.head_path = require_local_path(head_path or os.getenv("AESTHETIC_CKPT_PATH") or root / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth", description="Aesthetic reward head", must_be_file=True)
        dtype = _dtype_for(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_path, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(self.clip_model_path, **_dtype_kwargs(dtype), local_files_only=local_files_only).to(self.device).eval()
        projection_dim = int(getattr(self.model.config, "projection_dim", 768))
        self.head = AestheticHead(projection_dim).to(self.device).eval()
        self.head.load_state_dict(strip_state_dict_prefixes(torch.load(self.head_path, map_location="cpu")))

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        del prompts
        inputs = self.processor(images=list(images), return_tensors="pt", padding=True).to(self.device)
        features = normalize_features(feature_tensor(self.model.get_image_features(**inputs)).float())
        return self.head(features).flatten().float().cpu().tolist()


class HPSSelector(RewardSelector):
    def __init__(self, device, open_clip_path: str | Path | None = None, checkpoint_path: str | Path | None = None):
        import open_clip

        self.device = torch.device(device)
        root = project_root()
        open_clip_path = require_local_path(open_clip_path or os.getenv("HPS_OPEN_CLIP_PRETRAINED_PATH") or root / "models" / "CLIP-ViT-H-14-laion2B-s32B-b79K" / "open_clip_pytorch_model.bin", description="HPS OpenCLIP weights", must_be_file=True)
        checkpoint_path = require_local_path(checkpoint_path or os.getenv("HPS_CKPT_PATH") or root / "models" / "HPSv2" / "HPS_v2_compressed.pt", description="HPSv2 checkpoint", must_be_file=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=str(open_clip_path), device=self.device)
        self.model.load_state_dict(strip_state_dict_prefixes(torch.load(checkpoint_path, map_location=self.device)), strict=False)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(images))
        image_tensor = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        text_tensor = self.tokenizer(prompts).to(self.device)
        image_features = normalize_features(self.model.encode_image(image_tensor).float())
        text_features = normalize_features(self.model.encode_text(text_tensor).float())
        return (image_features * text_features).sum(dim=-1).float().cpu().tolist()


CHOICE_MODEL_NAMES = {
    "aes",
    "clip",
    "hps",
    "hpsv2",
    "pickscore",
}


def resolve_choice_models(choice_models: str | Sequence[str] | None, fallback_choice_model: str) -> tuple[str, ...]:
    if isinstance(choice_models, str):
        models = tuple(item.strip() for item in choice_models.split(",") if item.strip())
    elif choice_models is None:
        models = ()
    else:
        models = tuple(choice_models)
    if not models:
        models = (fallback_choice_model,)
    invalid = [name for name in models if name not in CHOICE_MODEL_NAMES]
    if invalid:
        raise ValueError(f"Unknown choice model(s): {invalid}. Available: {sorted(CHOICE_MODEL_NAMES)}")
    if len(set(models)) != len(models):
        raise ValueError(f"Duplicate choice models are not allowed: {models}")
    return models


def resolve_choice_model_weights(choice_model_weights: str | Sequence[float] | None, models: Sequence[str]) -> tuple[float, ...]:
    if isinstance(choice_model_weights, str):
        weights = tuple(float(item.strip()) for item in choice_model_weights.split(",") if item.strip())
    elif choice_model_weights is None:
        weights = ()
    else:
        weights = tuple(float(item) for item in choice_model_weights)
    if not weights:
        weights = tuple(1.0 for _ in models)
    if len(weights) != len(models):
        raise ValueError(f"Expected {len(models)} choice model weights, got {len(weights)}.")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Choice model weights must be non-negative, got {weights}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("At least one choice model weight must be > 0.")
    return tuple(weight / total for weight in weights)


def build_selector(
    name: str,
    device,
    *,
    pickscore_model_path: str | Path | None = None,
    pickscore_processor_path: str | Path | None = None,
    local_files_only: bool = True,
) -> RewardSelector:
    normalized = name.lower()
    if normalized == "pickscore":
        return PickScoreSelector(device, pickscore_model_path, pickscore_processor_path, local_files_only=local_files_only)
    if normalized == "clip":
        return CLIPSelector(device, local_files_only=local_files_only)
    if normalized == "aes":
        return AestheticSelector(device, local_files_only=local_files_only)
    if normalized in {"hps", "hpsv2"}:
        return HPSSelector(device)
    raise ValueError(f"Unknown reward selector: {name}")


def build_choice_selectors(
    models: Sequence[str],
    device,
    *,
    pickscore_model_path: str | Path | None = None,
    pickscore_processor_path: str | Path | None = None,
    local_files_only: bool = True,
) -> dict[str, RewardSelector]:
    device_overrides = _parse_choice_model_device_overrides(os.getenv("DRPO_CHOICE_MODEL_DEVICES", ""))
    return {
        model: build_selector(
            model,
            device_overrides.get(model, device),
            pickscore_model_path=pickscore_model_path,
            pickscore_processor_path=pickscore_processor_path,
            local_files_only=local_files_only,
        )
        for model in models
    }


def _parse_choice_model_device_overrides(value: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, _, device = item.partition("=")
        elif ":" in item:
            key, _, device = item.partition(":")
        else:
            raise ValueError("DRPO_CHOICE_MODEL_DEVICES entries must use model=device.")
        key = key.strip()
        device = device.strip()
        if key not in CHOICE_MODEL_NAMES:
            raise ValueError(f"Unknown choice model in DRPO_CHOICE_MODEL_DEVICES: {key}")
        if not device:
            raise ValueError(f"Missing device for DRPO_CHOICE_MODEL_DEVICES entry: {item}")
        overrides[key] = device
    return overrides


def _payload_for_selector(model_name: str, prompt: Any) -> Any:
    if not isinstance(prompt, Mapping):
        return prompt
    return prompt.get("prompt", prompt)


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _score_one_selector(
    model_name: str,
    selector: RewardSelector,
    images: Sequence[Image.Image],
    prompt: Any,
) -> tuple[str, torch.Tensor, float]:
    start = time.perf_counter()
    scores = selector.score(images, _payload_for_selector(model_name, prompt))
    return model_name, torch.tensor(scores, dtype=torch.float32), time.perf_counter() - start


def score_reward_ensemble(
    selectors: dict[str, RewardSelector],
    weights: Sequence[float],
    images: Sequence[Image.Image],
    prompt: Any,
    *,
    normalize: str = "zscore",
    eps: float = 1e-6,
    parallel_models: bool | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(selectors) == 0:
        raise ValueError("Expected at least one reward selector.")
    if len(selectors) != len(weights):
        raise ValueError(f"Expected {len(selectors)} weights, got {len(weights)}.")
    weighted_terms: list[torch.Tensor] = []
    info: dict[str, torch.Tensor] = {}
    selector_items = list(selectors.items())
    if parallel_models is None:
        parallel_models = _truthy_env("DRPO_PARALLEL_REWARD_SELECTORS")
    if parallel_models and len(selector_items) > 1:
        max_workers = min(len(selector_items), max(1, int(os.getenv("DRPO_PARALLEL_REWARD_WORKERS", str(len(selector_items))))))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            raw_by_model = {
                model_name: (raw, elapsed)
                for model_name, raw, elapsed in pool.map(
                    lambda item: _score_one_selector(item[0], item[1], images, prompt),
                    selector_items,
                )
            }
    else:
        raw_by_model = {
            model_name: (raw, elapsed)
            for model_name, raw, elapsed in (
                _score_one_selector(model_name, selector, images, prompt)
                for model_name, selector in selector_items
            )
        }
    for model_name, weight in zip(selectors.keys(), weights):
        raw, elapsed = raw_by_model[model_name]
        if raw.ndim != 1 or raw.numel() != len(images):
            raise ValueError(
                f"Selector {model_name} returned invalid score shape {tuple(raw.shape)} for {len(images)} images."
            )
        if normalize == "zscore":
            std = torch.clamp(raw.std(unbiased=False), min=eps)
            normed = (raw - raw.mean()) / std
        elif normalize == "none":
            normed = raw
        else:
            raise ValueError(f"Unknown choice score normalization mode: {normalize}")
        weighted_terms.append(float(weight) * normed)
        safe_name = model_name.replace("/", "_").replace("-", "_")
        info[f"online_reward_{safe_name}_raw_mean"] = raw.mean().detach()
        info[f"online_reward_{safe_name}_raw_std"] = raw.std(unbiased=False).detach()
        info[f"online_reward_{safe_name}_norm_mean"] = normed.mean().detach()
        info[f"online_reward_{safe_name}_norm_std"] = normed.std(unbiased=False).detach()
        info[f"online_reward_{safe_name}_weight"] = raw.new_tensor(float(weight))
        info[f"online_reward_{safe_name}_score_seconds"] = raw.new_tensor(float(elapsed))
    ensemble = torch.stack(weighted_terms, dim=0).sum(dim=0)
    info["online_reward_ensemble_mean"] = ensemble.mean().detach()
    info["online_reward_ensemble_std"] = ensemble.std(unbiased=False).detach()
    info["online_reward_ensemble_max"] = ensemble.max().detach()
    info["online_reward_ensemble_min"] = ensemble.min().detach()
    return ensemble, info
