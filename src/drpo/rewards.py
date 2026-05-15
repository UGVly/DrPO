from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(images))
        image_inputs = self.processor(images=list(images), padding=True, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        image_features = normalize_features(feature_tensor(self.model.get_image_features(**image_inputs)).float())
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


class HPSv3Selector(RewardSelector):
    def __init__(
        self,
        device,
        checkpoint_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ):
        try:
            import yaml
            from hpsv3 import HPSv3RewardInferencer
            from hpsv3.inference import _MODEL_CONFIG_PATH
        except ImportError as exc:  # pragma: no cover - optional eval env
            raise ImportError("HPSv3 is not installed in the active conda environment.") from exc

        self.device = torch.device(device)
        root = project_root()
        self.checkpoint_path = require_local_path(
            checkpoint_path or os.getenv("HPSV3_CKPT_PATH") or root / "models" / "HPSv3" / "HPSv3.safetensors",
            description="HPSv3 checkpoint",
            must_be_file=True,
        )
        self.base_model_path = require_local_path(
            base_model_path or os.getenv("HPSV3_BASE_MODEL_PATH") or root / "models" / "Qwen2-VL-7B-Instruct",
            description="HPSv3 Qwen2-VL base model",
            must_be_file=False,
        )
        self.config_path = self._local_config_path(config_path, _MODEL_CONFIG_PATH, yaml)
        self.model = HPSv3RewardInferencer(
            config_path=str(self.config_path),
            checkpoint_path=str(self.checkpoint_path),
            device=str(self.device),
        )

    def _local_config_path(self, config_path, model_config_path, yaml_module) -> Path:
        if config_path is not None or os.getenv("HPSV3_CONFIG_PATH"):
            return require_local_path(config_path or os.getenv("HPSV3_CONFIG_PATH"), description="HPSv3 config", must_be_file=True)
        default_config = Path(model_config_path) / "HPSv3_7B.yaml"
        with default_config.open("r", encoding="utf-8") as handle:
            config = yaml_module.safe_load(handle)
        config["model_name_or_path"] = str(self.base_model_path)
        config["output_dir"] = str(project_root() / "outputs" / "hpsv3-eval")
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="drpo_hpsv3_", delete=False, encoding="utf-8")
        with handle:
            yaml_module.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)

    @torch.no_grad()
    def score_paths(self, image_paths: Sequence[str | Path], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(image_paths))
        rewards = self.model.reward([str(path) for path in image_paths], prompts)
        if isinstance(rewards, torch.Tensor):
            values = rewards[:, 0] if rewards.ndim > 1 else rewards
            return values.float().detach().cpu().tolist()
        return [float(reward[0].item() if isinstance(reward, torch.Tensor) and reward.ndim else reward) for reward in rewards]

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        with tempfile.TemporaryDirectory(prefix="drpo_hpsv3_images_") as tmp:
            paths = []
            for index, image in enumerate(images):
                path = Path(tmp) / f"{index:06d}.png"
                image.save(path)
                paths.append(path)
            return self.score_paths(paths, prompts)


class ImageRewardSelector(RewardSelector):
    def __init__(
        self,
        device,
        checkpoint_path: str | Path | None = None,
        med_config_path: str | Path | None = None,
        bert_tokenizer_path: str | Path | None = None,
    ):
        try:
            import ImageReward as RM
        except ImportError as exc:  # pragma: no cover - optional eval env
            raise ImportError("ImageReward is not installed in the active conda environment.") from exc

        self.device = torch.device(device)
        root = project_root()
        self.checkpoint_path = require_local_path(
            checkpoint_path or os.getenv("IMAGEREWARD_CKPT_PATH") or root / "models" / "ImageReward" / "ImageReward.pt",
            description="ImageReward checkpoint",
            must_be_file=True,
        )
        self.med_config_path = require_local_path(
            med_config_path or os.getenv("IMAGEREWARD_MED_CONFIG_PATH") or root / "models" / "ImageReward" / "med_config.json",
            description="ImageReward med_config",
            must_be_file=True,
        )
        self.bert_tokenizer_path = require_local_path(
            bert_tokenizer_path or os.getenv("IMAGEREWARD_BERT_TOKENIZER_PATH") or root / "models" / "bert-base-uncased",
            description="ImageReward BERT tokenizer",
            must_be_file=False,
        )
        self._patch_local_bert_tokenizer()
        self.model = RM.load(str(self.checkpoint_path), device=str(self.device), med_config=str(self.med_config_path))

    def _patch_local_bert_tokenizer(self) -> None:
        from transformers import BertTokenizer
        import ImageReward.models.BLIP.blip as blip
        import ImageReward.models.BLIP.blip_pretrain as blip_pretrain

        def init_tokenizer():
            tokenizer = BertTokenizer.from_pretrained(str(self.bert_tokenizer_path), local_files_only=True)
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
            tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
            return tokenizer

        blip.init_tokenizer = init_tokenizer
        blip_pretrain.init_tokenizer = init_tokenizer

    @torch.no_grad()
    def _score_pil_batch(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(images))
        if not images:
            return []

        text_input = self.model.blip.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        image_tensor = torch.stack([self.model.preprocess(image) for image in images], dim=0).to(self.device)

        image_embeds = self.model.blip.visual_encoder(image_tensor)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
        text_output = self.model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.model.mlp(txt_features)
        rewards = (rewards - self.model.mean) / self.model.std
        return rewards.flatten().float().detach().cpu().tolist()

    @torch.no_grad()
    def score_paths(self, image_paths: Sequence[str | Path], prompts: str | Sequence[str]) -> list[float]:
        images: list[Image.Image] = []
        try:
            for path in image_paths:
                images.append(Image.open(path).convert("RGB"))
            return self._score_pil_batch(images, prompts)
        finally:
            for image in images:
                image.close()

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        return self._score_pil_batch(images, prompts)


CHOICE_MODEL_NAMES = {"aes", "clip", "geneval", "hps", "hpsv2", "imagereward", "pickscore"}


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
    geneval_repo: str | Path | None = None,
    geneval_detector_path: str | Path | None = None,
    geneval_model_config: str | Path | None = None,
    geneval_options: str = "",
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
    if normalized == "geneval":
        from utils.geneval_utils import Selector as GenevalSelector

        return GenevalSelector(
            torch.device(device),
            geneval_repo=str(geneval_repo) if geneval_repo is not None else "",
            geneval_detector_path=str(geneval_detector_path) if geneval_detector_path is not None else "",
            geneval_model_config=str(geneval_model_config) if geneval_model_config is not None else None,
            geneval_options=geneval_options,
        )
    if normalized == "hpsv3":
        return HPSv3Selector(device)
    if normalized in {"imagereward", "image_reward"}:
        return ImageRewardSelector(device)
    raise ValueError(f"Unknown reward selector: {name}")


def build_choice_selectors(
    models: Sequence[str],
    device,
    *,
    pickscore_model_path: str | Path | None = None,
    pickscore_processor_path: str | Path | None = None,
    geneval_repo: str | Path | None = None,
    geneval_detector_path: str | Path | None = None,
    geneval_model_config: str | Path | None = None,
    geneval_options: str = "",
    local_files_only: bool = True,
) -> dict[str, RewardSelector]:
    return {
        model: build_selector(
            model,
            device,
            pickscore_model_path=pickscore_model_path,
            pickscore_processor_path=pickscore_processor_path,
            geneval_repo=geneval_repo,
            geneval_detector_path=geneval_detector_path,
            geneval_model_config=geneval_model_config,
            geneval_options=geneval_options,
            local_files_only=local_files_only,
        )
        for model in models
    }


def score_reward_ensemble(
    selectors: dict[str, RewardSelector],
    weights: Sequence[float],
    images: Sequence[Image.Image],
    prompt: Any,
    *,
    normalize: str = "zscore",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(selectors) == 0:
        raise ValueError("Expected at least one reward selector.")
    if len(selectors) != len(weights):
        raise ValueError(f"Expected {len(selectors)} weights, got {len(weights)}.")
    weighted_terms: list[torch.Tensor] = []
    info: dict[str, torch.Tensor] = {}
    for (model_name, selector), weight in zip(selectors.items(), weights):
        raw = torch.tensor(selector.score(images, prompt), dtype=torch.float32)
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
    ensemble = torch.stack(weighted_terms, dim=0).sum(dim=0)
    info["online_reward_ensemble_mean"] = ensemble.mean().detach()
    info["online_reward_ensemble_std"] = ensemble.std(unbiased=False).detach()
    info["online_reward_ensemble_max"] = ensemble.max().detach()
    info["online_reward_ensemble_min"] = ensemble.min().detach()
    return ensemble, info
