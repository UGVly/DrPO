from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
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
        self._supports_in_memory_images: bool | None = None

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
        prompts = normalize_prompts(prompts, len(images))
        if self._supports_in_memory_images is not False:
            try:
                image_tensors = [torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8)) for image in images]
                rewards = self.model.reward(image_tensors, prompts)
                self._supports_in_memory_images = True
                if isinstance(rewards, torch.Tensor):
                    values = rewards[:, 0] if rewards.ndim > 1 else rewards
                    return values.float().detach().cpu().tolist()
                return [float(reward[0].item() if isinstance(reward, torch.Tensor) and reward.ndim else reward) for reward in rewards]
            except Exception:
                self._supports_in_memory_images = False
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
            import transformers.modeling_utils as modeling_utils
            import transformers.pytorch_utils as pytorch_utils

            for name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                    setattr(modeling_utils, name, getattr(pytorch_utils, name))
        except Exception:
            pass
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


class DiffusionNFTGeneValSelector(RewardSelector):
    """GeneVal reward adapter that reuses DiffusionNFT/Flow-GRPO's evaluator."""

    def __init__(
        self,
        device,
        diffusionnft_repo_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        train_metric: str = "score",
    ):
        self.device = torch.device(device)
        self.train_metric = train_metric
        self.diffusionnft_repo_path = self._resolve_diffusionnft_repo(diffusionnft_repo_path)
        self._add_diffusionnft_paths(self.diffusionnft_repo_path)
        from flow_grpo.rewards import multi_score

        self.scorer = multi_score(self.device, {"geneval": 1.0})
        self.metadata_by_prompt = self._load_metadata(metadata_path)

    def _resolve_diffusionnft_repo(self, path: str | Path | None) -> Path:
        candidates: list[str | Path | None] = [
            path,
            os.getenv("DIFFUSIONNFT_REPO_DIR"),
            os.getenv("GENEVAL_DIFFUSIONNFT_REPO"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            repo = Path(candidate).expanduser().resolve()
            if (repo / "flow_grpo" / "rewards.py").is_file():
                return require_local_path(repo, description="DiffusionNFT repo", must_be_file=False)
        raise FileNotFoundError("Could not find a DiffusionNFT/Mutistepalign repo containing flow_grpo/rewards.py.")

    def _add_diffusionnft_paths(self, repo: Path) -> None:
        for path in (repo, repo / "mmdetection", repo / "mmcv"):
            value = str(path)
            if path.exists() and value not in sys.path:
                sys.path.insert(0, value)

    def _load_metadata(self, metadata_path: str | Path | None) -> dict[str, dict[str, Any]]:
        path_value = metadata_path or os.getenv("DIFFUSIONNFT_GENEVAL_METADATA_PATH")
        if path_value is None:
            default_path = self.diffusionnft_repo_path / "dataset" / "geneval" / "train_metadata.jsonl"
            path_value = default_path if default_path.is_file() else None
        if path_value is None:
            return {}

        path = require_local_path(path_value, description="DiffusionNFT GeneVal metadata", must_be_file=True)
        rows: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"GeneVal metadata row {line_number} must be an object.")
                prompt = str(row.get("prompt", "")).strip()
                if prompt:
                    rows[prompt] = row
        return rows

    def _payload_to_metadata(self, payload: Any, count: int) -> tuple[list[str], list[dict[str, Any]]]:
        if isinstance(payload, Mapping):
            prompt_text = str(payload.get("prompt", "")).strip()
            metadata = payload.get("geneval_metadata") or payload.get("metadata")
            if metadata is None and "tag" in payload:
                metadata = payload
            if metadata is None:
                raise ValueError("DiffusionNFT GeneVal reward requires geneval_metadata for each prompt.")
            if isinstance(metadata, Mapping):
                metadata_list = [dict(metadata) for _ in range(count)]
            elif isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
                metadata_list = [dict(item) for item in metadata]
                if len(metadata_list) != count:
                    raise ValueError(f"Expected {count} GeneVal metadata entries, got {len(metadata_list)}.")
            else:
                raise TypeError(f"Unsupported GeneVal metadata payload: {type(metadata)}")
            prompts = [prompt_text or str(item.get("prompt", "")) for item in metadata_list]
            return prompts, metadata_list

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            values = list(payload)
            if len(values) == count and all(isinstance(item, Mapping) for item in values):
                metadata_list = [dict(item) for item in values]
                prompts = [str(item.get("prompt", "")) for item in metadata_list]
                return prompts, metadata_list

        prompts = normalize_prompts(payload, count)
        metadata_list = []
        for prompt in prompts:
            metadata = self.metadata_by_prompt.get(prompt)
            if metadata is None:
                raise KeyError(f"No GeneVal metadata found for prompt: {prompt!r}")
            metadata_list.append(dict(metadata))
        return prompts, metadata_list

    def _metric_values(self, score_details: dict[str, Any]) -> Any:
        metric = self.train_metric.strip().lower()
        if metric in {"score", "soft", "soft_score", "avg", "geneval"}:
            return score_details["avg"]
        if metric in {"strict", "strict_accuracy"}:
            return score_details["strict_accuracy"]
        if metric in {"accuracy", "relaxed_accuracy"}:
            return score_details["accuracy"]
        raise ValueError("geneval_train_metric must be one of: score, strict_accuracy, accuracy.")

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: Any) -> list[float]:
        prompt_list, metadata_list = self._payload_to_metadata(prompts, len(images))
        image_array = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images], axis=0)
        score_details, _ = self.scorer(image_array, prompt_list, metadata_list, only_strict=True)
        values = self._metric_values(score_details)
        if isinstance(values, torch.Tensor):
            return values.float().detach().cpu().tolist()
        return [float(value) for value in values]


class OCRSelector(RewardSelector):
    """PaddleOCR text-rendering reward compatible with DiffusionNFT OCR prompts."""

    _quoted_text = re.compile(r"[\"']([^\"']+)[\"']")
    _subprocess_code = r"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from Levenshtein import distance
from paddleocr import PaddleOCR
from PIL import Image

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_root = Path.home() / ".paddleocr" / "whl"
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    use_gpu=False,
    show_log=False,
    det_model_dir=str(model_root / "det" / "en" / "en_PP-OCRv3_det_infer"),
    rec_model_dir=str(model_root / "rec" / "en" / "en_PP-OCRv4_rec_infer"),
    cls_model_dir=str(model_root / "cls" / "ch" / "ch_ppocr_mobile_v2.0_cls_infer"),
)
scores = []
for item in payload:
    target = re.sub(r"\s+", "", item["target"]).lower()
    image = np.asarray(Image.open(item["image"]).convert("RGB"), dtype=np.uint8)
    try:
        result = ocr.ocr(image, cls=False)
        recognized = "".join(res[1][0] if res[1][1] > 0 else "" for res in (result[0] or []))
        recognized = re.sub(r"\s+", "", recognized).lower()
        dist = 0 if target in recognized else min(distance(recognized, target), len(target))
    except Exception as exc:
        print(f"OCR processing failed: {exc}", file=sys.stderr)
        dist = len(target)
    scores.append(1.0 - dist / max(len(target), 1))
print(json.dumps(scores))
"""

    def __init__(self, device=None, use_gpu: bool | None = None):
        self.subprocess_python = os.getenv("DRPO_OCR_SUBPROCESS_PYTHON", "").strip()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if self.subprocess_python:
            self.ocr = None
            self._distance = self._edit_distance
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - optional OCR env
            raise ImportError("PaddleOCR is not installed in the active environment.") from exc

        if use_gpu is None:
            use_gpu = os.getenv("DRPO_OCR_USE_GPU", "").strip().lower() in {"1", "true", "yes", "on"}
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=bool(use_gpu), show_log=False)
        try:
            from Levenshtein import distance as levenshtein_distance
        except ImportError:
            levenshtein_distance = self._edit_distance
        self._distance = levenshtein_distance

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        if left == right:
            return 0
        prev = list(range(len(right) + 1))
        for i, left_ch in enumerate(left, start=1):
            cur = [i]
            for j, right_ch in enumerate(right, start=1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (left_ch != right_ch)))
            prev = cur
        return prev[-1]

    @classmethod
    def _target_text(cls, prompt: str) -> str:
        match = cls._quoted_text.search(prompt)
        if match is None:
            raise ValueError(f"OCR reward prompt must contain target text in quotes: {prompt!r}")
        return match.group(1)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", "", value).lower()

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], prompts: str | Sequence[str]) -> list[float]:
        prompts = normalize_prompts(prompts, len(images))
        targets = [self._normalize_text(self._target_text(prompt)) for prompt in prompts]
        if self.subprocess_python:
            with tempfile.TemporaryDirectory(prefix="drpo_ocr_") as tmp:
                tmp_path = Path(tmp)
                payload = []
                for index, (image, target) in enumerate(zip(images, targets)):
                    image_path = tmp_path / f"{index:04d}.png"
                    image.convert("RGB").save(image_path)
                    payload.append({"image": str(image_path), "target": target})
                payload_path = tmp_path / "payload.json"
                payload_path.write_text(json.dumps(payload), encoding="utf-8")
                result = subprocess.run(
                    [self.subprocess_python, "-c", self._subprocess_code, str(payload_path)],
                    check=True,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                return [float(value) for value in json.loads(result.stdout)]
        rewards: list[float] = []
        for image, target in zip(images, targets):
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            try:
                result = self.ocr.ocr(array, cls=False)
                recognized = "".join(res[1][0] if res[1][1] > 0 else "" for res in (result[0] or []))
                recognized = self._normalize_text(recognized)
                distance = 0 if target in recognized else int(self._distance(recognized, target))
                distance = min(distance, len(target))
            except Exception as exc:
                print(f"OCR processing failed: {exc}")
                distance = len(target)
            rewards.append(1.0 - distance / max(len(target), 1))
        return rewards


CHOICE_MODEL_NAMES = {
    "aes",
    "clip",
    "diffusionnft_geneval",
    "diffusionnft_ocr",
    "geneval",
    "geneval_flow",
    "hps",
    "hpsv2",
    "hpsv3",
    "imagereward",
    "image_reward",
    "ocr",
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
    geneval_repo: str | Path | None = None,
    geneval_detector_path: str | Path | None = None,
    geneval_model_config: str | Path | None = None,
    geneval_options: str = "",
    diffusionnft_repo_path: str | Path | None = None,
    geneval_metadata_path: str | Path | None = None,
    geneval_train_metric: str = "score",
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
    if normalized in {"diffusionnft_geneval", "geneval_flow"}:
        return DiffusionNFTGeneValSelector(
            device,
            diffusionnft_repo_path=diffusionnft_repo_path,
            metadata_path=geneval_metadata_path,
            train_metric=geneval_train_metric,
        )
    if normalized in {"ocr", "diffusionnft_ocr"}:
        return OCRSelector(device)
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
    diffusionnft_repo_path: str | Path | None = None,
    geneval_metadata_path: str | Path | None = None,
    geneval_train_metric: str = "score",
    local_files_only: bool = True,
) -> dict[str, RewardSelector]:
    device_overrides = _parse_choice_model_device_overrides(os.getenv("DRPO_CHOICE_MODEL_DEVICES", ""))
    return {
        model: build_selector(
            model,
            device_overrides.get(model, device),
            pickscore_model_path=pickscore_model_path,
            pickscore_processor_path=pickscore_processor_path,
            geneval_repo=geneval_repo,
            geneval_detector_path=geneval_detector_path,
            geneval_model_config=geneval_model_config,
            geneval_options=geneval_options,
            diffusionnft_repo_path=diffusionnft_repo_path,
            geneval_metadata_path=geneval_metadata_path,
            geneval_train_metric=geneval_train_metric,
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
    normalized = model_name.lower()
    if normalized == "geneval":
        return prompt.get("geneval_metadata") or prompt.get("metadata") or prompt
    if normalized in {"diffusionnft_geneval", "geneval_flow"}:
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
