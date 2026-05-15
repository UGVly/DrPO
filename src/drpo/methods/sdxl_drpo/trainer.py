from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shlex
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from drpo.data import Batch, PromptDataset, collate_preference_batch
from drpo.drift import drift_loss, pairwise_l2
from drpo.paths import project_root, require_local_path
from drpo.rewards import build_choice_selectors, resolve_choice_model_weights, resolve_choice_models, score_reward_ensemble
from drpo.utils.tensors import add_rank_selection_stats, safe_std, select_disjoint_pref_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SDXLDrPOConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    prompt_file: str
    mae_model_name_or_path: str
    model_variant: str | None = "fp16"
    choice_model: str = "pickscore"
    choice_models: tuple[str, ...] = ()
    choice_model_weights: tuple[float, ...] = ()
    choice_score_normalize: str = "zscore"
    pickscore_model_name_or_path: str | None = None
    pickscore_processor_name_or_path: str | None = None
    pickscore_allow_remote: bool = False
    seed: int = 42
    resolution: int = 512
    train_batch_size: int = 1
    batchsize_gen: int = 8
    num_inference_steps: int = 1
    guidance_scale: float = 0.0
    max_train_steps: int = 1000
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 2
    max_train_samples: int | None = None
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    checkpointing_steps: int = 100
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    use_lora: bool = True
    resume_from_checkpoint: str | None = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    feature_extractor: str = "mae"
    mae_feature_keys: tuple[str, ...] = ("layer12_patch_mean", "layer12_patch_std", "layer12_cls")
    teacher_feature_layers: tuple[str, ...] = ("down_blocks.2", "mid_block", "up_blocks.0")
    teacher_feature_noise: float = 0.1
    teacher_feature_timestep: int = 100
    teacher_feature_pool_size: int = 4
    drifting_kernel: str = "laplacian"
    drifting_pref_r_list: tuple[float, ...] = (0.02, 0.05, 0.2)
    drifting_ref_r_list: tuple[float, ...] = (0.02, 0.05, 0.2)
    drifting_pos_weight: float = 3000.0
    drifting_neg_weight: float = 3000.0
    drifting_ref_weight: float = 3000.0
    drifting_ref_neg_weight: float = 3000.0
    drifting_ref_loss_weight: float = 0.2
    ref_model_l2_weight: float = 0.02
    num_pos_images: int = 2
    num_neg_images: int = 2
    online_feature_top_fraction: float = 1.0
    vae_decode_chunk_size: int = 1
    mae_chunk_size: int = 4


def _parse_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


class FrozenViTMAEImageFeatureExtractor(nn.Module):
    """Frozen official pixel MAE encoder features for DrPO image-space drift."""

    def __init__(self, model_name_or_path: str | Path, *, feature_keys: Sequence[str], local_files_only: bool = True) -> None:
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(str(model_name_or_path), local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(str(model_name_or_path), local_files_only=local_files_only)
        self.model.eval().requires_grad_(False)
        self.feature_keys = tuple(feature_keys)
        self.num_hidden_layers = int(getattr(self.model.config, "num_hidden_layers", 12))
        size = getattr(self.processor, "crop_size", None) or getattr(self.processor, "size", None) or {}
        if hasattr(size, "get"):
            self.input_size = int(size.get("height") or size.get("shortest_edge") or size.get("width") or 224)
        else:
            self.input_size = int(size or 224)
        mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(self.processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape (B, 3, H, W), got {tuple(images.shape)}")
        images = (images.clamp(-1, 1) + 1.0) / 2.0
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def vector_features(self, images: torch.Tensor, keys: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        keys = tuple(keys or self.feature_keys)
        requested = set(keys)
        dtype = next(self.model.parameters()).dtype
        pixel_values = self._preprocess(images).to(dtype=dtype)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("MAE model did not return hidden states.")
        features: dict[str, torch.Tensor] = {}
        for layer_idx in range(1, self.num_hidden_layers + 1):
            prefix = f"layer{layer_idx}"
            if not any(key.startswith(prefix) for key in requested):
                continue
            hidden = hidden_states[layer_idx]
            cls_token = hidden[:, :1, :]
            patch_tokens = hidden[:, 1:, :]
            if f"{prefix}_patch" in requested:
                features[f"{prefix}_patch"] = patch_tokens
            if f"{prefix}_patch_mean" in requested:
                features[f"{prefix}_patch_mean"] = patch_tokens.mean(dim=1, keepdim=True)
            if f"{prefix}_patch_std" in requested:
                features[f"{prefix}_patch_std"] = safe_std(patch_tokens, dim=1).unsqueeze(1)
            if f"{prefix}_cls" in requested:
                features[f"{prefix}_cls"] = cls_token
        missing = [key for key in keys if key not in features]
        if missing:
            raise ValueError(f"Missing MAE feature keys {missing}. Available: {sorted(features)}")
        return {key: features[key] for key in keys}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.vector_features(images, self.feature_keys)
        return torch.cat([value.flatten(1) for value in values.values()], dim=1)


class SDXLTeacherUNetFeatureExtractor(nn.Module):
    """Frozen SDXL teacher U-Net hidden states for latent-space DrPO drift."""

    def __init__(
        self,
        ref_unet,
        *,
        feature_layers: Sequence[str],
        feature_noise: float = 0.1,
        feature_timestep: int = 100,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        if not feature_layers:
            raise ValueError("At least one teacher feature layer is required.")
        if feature_noise < 0:
            raise ValueError("teacher feature noise must be >= 0.")
        if feature_timestep < 0:
            raise ValueError("teacher feature timestep must be >= 0.")
        if pool_size < 1:
            raise ValueError("teacher feature pool size must be >= 1.")

        self.ref_unet = ref_unet
        self.feature_layers = tuple(feature_layers)
        self.feature_noise = float(feature_noise)
        self.feature_timestep = int(feature_timestep)
        self.pool_size = int(pool_size)
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self.ref_unet.eval().requires_grad_(False)
        self._register_hooks()

    def _root_module(self):
        return getattr(self.ref_unet, "module", self.ref_unet)

    def _resolve_layer(self, name: str) -> nn.Module:
        module: object = self._root_module()
        for part in name.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        if not isinstance(module, nn.Module):
            raise TypeError(f"Teacher feature layer {name!r} did not resolve to an nn.Module.")
        return module

    def _register_hooks(self) -> None:
        for name in self.feature_layers:
            self._hooks.append(self._resolve_layer(name).register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            feature = output
            if isinstance(feature, (tuple, list)):
                feature = feature[0]
            if hasattr(feature, "sample"):
                feature = feature.sample
            if not torch.is_tensor(feature):
                raise TypeError(f"Teacher feature layer {name!r} returned unsupported output type: {type(feature)}")
            self._features[name] = feature

        return hook

    def _add_feature_noise(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_noise > 0:
            latents = latents + self.feature_noise * torch.randn_like(latents)
        timesteps = torch.full((latents.shape[0],), self.feature_timestep, device=latents.device, dtype=torch.long)
        return latents, timesteps

    def _postprocess_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 4 and self.pool_size > 1:
            if feature.shape[-2] >= self.pool_size and feature.shape[-1] >= self.pool_size:
                feature = F.avg_pool2d(feature, kernel_size=self.pool_size, stride=self.pool_size)
            else:
                feature = F.adaptive_avg_pool2d(feature, output_size=(1, 1))
        if feature.ndim < 2:
            raise ValueError(f"Expected teacher feature rank >= 2, got {tuple(feature.shape)}")
        return feature.flatten(start_dim=1).unsqueeze(1)

    def vector_features(
        self,
        latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        require_grad: bool,
    ) -> dict[str, torch.Tensor]:
        self._features = {}
        noisy_latents, feature_timestep = self._add_feature_noise(latents)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds.to(device=latents.device, dtype=prompt_embeds.dtype),
            "time_ids": add_time_ids.to(device=latents.device, dtype=prompt_embeds.dtype),
        }

        def run_forward() -> dict[str, torch.Tensor]:
            _ = self.ref_unet(
                noisy_latents,
                feature_timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            missing = [name for name in self.feature_layers if name not in self._features]
            if missing:
                raise ValueError(f"Teacher U-Net hooks did not capture feature layers: {missing}")
            return {name: self._postprocess_feature(self._features[name]) for name in self.feature_layers}

        if require_grad:
            return run_forward()
        with torch.no_grad():
            return run_forward()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def _dtype_for(config: SDXLDrPOConfig) -> torch.dtype:
    if config.mixed_precision == "bf16":
        return torch.bfloat16
    if config.mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def _decode_latents_to_tensor(vae, latents: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 0.13025))
    latents = (latents / scaling).to(device=vae.device, dtype=vae.dtype)
    chunks = []
    for chunk in latents.split(max(1, chunk_size)):
        chunks.append(vae.decode(chunk).sample)
    return torch.cat(chunks, dim=0).float().clamp(-1, 1)


@torch.no_grad()
def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    array = ((images.detach().cpu().clamp(-1, 1) + 1.0) / 2.0).permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((image * 255).round().astype("uint8")) for image in array]


def _encode_prompts(pipe: StableDiffusionXLPipeline, prompts: Sequence[str], device: torch.device):
    result = pipe.encode_prompt(
        prompt=list(prompts),
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds, _, pooled_prompt_embeds, _ = result
    return prompt_embeds, pooled_prompt_embeds


def _add_time_ids(pipe: StableDiffusionXLPipeline, batch_size: int, resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = [resolution, resolution, 0, 0, resolution, resolution]
    add_time_ids = torch.tensor([value], device=device, dtype=dtype)
    return add_time_ids.repeat(batch_size, 1)


def _select_condition(
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        prompt_embeds.index_select(0, indices),
        pooled_prompt_embeds.index_select(0, indices),
        add_time_ids.index_select(0, indices),
    )


def _sdxl_one_step_latents(
    *,
    pipe: StableDiffusionXLPipeline,
    unet,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    resolution: int,
    num_inference_steps: int,
) -> torch.Tensor:
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    sample = latents * scheduler.init_noise_sigma
    add_text_embeds = pooled_prompt_embeds.to(device=latents.device, dtype=prompt_embeds.dtype)
    add_time_ids = _add_time_ids(pipe, sample.shape[0], resolution, sample.device, prompt_embeds.dtype)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    for timestep in scheduler.timesteps:
        latent_model_input = scheduler.scale_model_input(sample, timestep)
        noise_pred = unet(
            latent_model_input,
            timestep.expand(sample.shape[0]) if timestep.ndim == 0 else timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        sample = scheduler.step(noise_pred, timestep, sample, return_dict=False)[0]
    return sample



def _feature_set(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor with shape (B, N, D), got {tuple(features.shape)}.")
    return features.reshape(1, -1, features.shape[-1])


def _active_feature_keys(config: SDXLDrPOConfig) -> tuple[str, ...]:
    if config.feature_extractor == "mae":
        return config.mae_feature_keys
    if config.feature_extractor == "teacher_unet":
        return config.teacher_feature_layers
    raise ValueError(f"Unknown SDXL feature extractor: {config.feature_extractor}")


def _resolve_unet_lora_dir(path: str | Path) -> Path:
    checkpoint_dir = Path(path).expanduser()
    if checkpoint_dir.is_file() and checkpoint_dir.name == "adapter_model.safetensors":
        return checkpoint_dir.parent
    if checkpoint_dir.is_dir() and checkpoint_dir.name == "unet_lora":
        return checkpoint_dir
    if checkpoint_dir.is_dir() and (checkpoint_dir / "unet_lora" / "adapter_model.safetensors").is_file():
        return checkpoint_dir / "unet_lora"
    raise FileNotFoundError(f"Expected checkpoint dir or unet_lora adapter path, got: {checkpoint_dir}")


def _resume_global_step(path: str | Path | None) -> int:
    if not path:
        return 0
    checkpoint_dir = Path(path).expanduser()
    if checkpoint_dir.name == "unet_lora":
        checkpoint_dir = checkpoint_dir.parent
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.is_file():
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state.get("global_step", 0))


def _make_trainable(unet, config: SDXLDrPOConfig):
    if not config.use_lora:
        unet.requires_grad_(True)
        return unet
    if config.resume_from_checkpoint:
        return PeftModel.from_pretrained(unet, str(_resolve_unet_lora_dir(config.resume_from_checkpoint)), is_trainable=True)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )
    return get_peft_model(unet, lora_config)


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _maybe_enable_xformers(unet) -> None:
    if not is_xformers_available():
        return
    import xformers

    if version.parse(xformers.__version__) < version.parse("0.0.17"):
        return
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        logger.warning("Could not enable xformers: %s", exc)


def _compute_prompt_terms(
    feature_keys: tuple[str, ...],
    generated_features: dict[str, torch.Tensor],
    reference_features: dict[str, torch.Tensor],
    positive_features: dict[str, torch.Tensor],
    negative_features: dict[str, torch.Tensor],
    config: SDXLDrPOConfig,
) -> dict[str, torch.Tensor]:
    pref_losses = []
    ref_losses = []
    feature_l2_terms = []
    d_pos_terms = []
    d_neg_terms = []
    d_ref_terms = []
    for key in feature_keys:
        generated = _feature_set(generated_features[key]).float()
        reference = _feature_set(reference_features[key]).float()
        positive = _feature_set(positive_features[key]).float()
        negative = _feature_set(negative_features[key]).float()
        pref_loss_vec, _ = drift_loss(
            generated,
            positive,
            negative,
            positive_weight=config.drifting_pos_weight,
            negative_weight=config.drifting_neg_weight,
            radii=config.drifting_pref_r_list,
            kernel=config.drifting_kernel,
        )
        ref_loss_vec, _ = drift_loss(
            generated,
            reference,
            generated.detach(),
            positive_weight=config.drifting_ref_weight,
            negative_weight=config.drifting_ref_neg_weight,
            radii=config.drifting_ref_r_list,
            mask_negative_self=True,
            kernel=config.drifting_kernel,
        )
        pref_losses.append(pref_loss_vec.mean())
        ref_losses.append(ref_loss_vec.mean())
        feature_l2_terms.append(F.mse_loss(generated, reference))
        d_pos_terms.append(pairwise_l2(generated.detach(), positive.detach()).mean())
        d_neg_terms.append(pairwise_l2(generated.detach(), negative.detach()).mean())
        d_ref_terms.append(pairwise_l2(generated.detach(), reference.detach()).mean())
    pref_loss = torch.stack(pref_losses).mean()
    ref_loss = torch.stack(ref_losses).mean()
    feature_l2 = torch.stack(feature_l2_terms).mean()
    loss = pref_loss + config.drifting_ref_loss_weight * ref_loss
    return {
        "loss": loss,
        "pref_loss": pref_loss.detach(),
        "ref_loss": ref_loss.detach(),
        "feature_l2": feature_l2.detach(),
        "d_pos": torch.stack(d_pos_terms).mean().detach(),
        "d_neg": torch.stack(d_neg_terms).mean().detach(),
        "d_ref": torch.stack(d_ref_terms).mean().detach(),
    }


def _save_checkpoint(accelerator: Accelerator, unet, config: SDXLDrPOConfig, checkpoint_dir: Path, global_step: int, checkpoint_type: str) -> None:
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        if config.use_lora:
            unwrapped.save_pretrained(checkpoint_dir / "unet_lora")
        else:
            unwrapped.save_pretrained(checkpoint_dir / "unet")
        metadata = {
            "checkpoint_type": checkpoint_type,
            "global_step": global_step,
            "created_at": datetime.now().isoformat(),
            "model_type": f"sdxl-turbo-drpo-{config.feature_extractor}",
            "feature_extractor": config.feature_extractor,
            "contains_accelerate_state": False,
            "contains_model_state": True,
            "model_variant": config.model_variant,
            "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
            "mae_model_name_or_path": config.mae_model_name_or_path,
        }
        (checkpoint_dir / "training_state.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    accelerator.wait_for_everyone()


def _setup_logging(config: SDXLDrPOConfig, accelerator: Accelerator) -> None:
    log_dir = Path(config.output_dir) / config.logging_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(log_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=f"%(asctime)s | %(levelname)s | rank={accelerator.process_index}/{accelerator.num_processes} | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _save_runtime_snapshot(config: SDXLDrPOConfig, accelerator: Accelerator) -> None:
    if not accelerator.is_main_process:
        return
    out = Path(config.output_dir) / "run_metadata"
    out.mkdir(parents=True, exist_ok=True)
    safe_config = asdict(config)
    (out / "resolved_config.json").write_text(json.dumps(safe_config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    runtime = {
        "created_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "world_size": accelerator.num_processes,
        "device": str(accelerator.device),
        "torch_version": torch.__version__,
    }
    (out / "runtime_info.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    (out / "launch_command.sh").write_text("#!/usr/bin/env bash\n" + shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")


def _validate_config(config: SDXLDrPOConfig) -> None:
    if config.batchsize_gen < config.num_pos_images + config.num_neg_images:
        raise ValueError("batchsize_gen must be >= num_pos_images + num_neg_images.")
    if config.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1.")
    if config.resolution % 8:
        raise ValueError("resolution must be divisible by 8.")
    if config.feature_extractor not in {"mae", "teacher_unet"}:
        raise ValueError("feature_extractor must be either 'mae' or 'teacher_unet'.")
    if config.feature_extractor == "mae" and not config.mae_feature_keys:
        raise ValueError("mae_feature_keys must not be empty when feature_extractor=mae.")
    if config.feature_extractor == "teacher_unet":
        if not config.teacher_feature_layers:
            raise ValueError("teacher_feature_layers must not be empty when feature_extractor=teacher_unet.")
        if config.teacher_feature_noise < 0:
            raise ValueError("teacher_feature_noise must be >= 0.")
        if config.teacher_feature_timestep < 0:
            raise ValueError("teacher_feature_timestep must be >= 0.")
        if config.teacher_feature_pool_size < 1:
            raise ValueError("teacher_feature_pool_size must be >= 1.")


def train(config: SDXLDrPOConfig) -> None:
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
        log_with=config.report_to,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / config.logging_dir)),
    )
    _setup_logging(config, accelerator)
    if config.seed is not None:
        set_seed(config.seed + accelerator.process_index)
    _save_runtime_snapshot(config, accelerator)

    model_path = str(require_local_path(config.pretrained_model_name_or_path, description="SDXL Turbo model", must_be_file=False))
    mae_path = (
        str(require_local_path(config.mae_model_name_or_path, description="facebook-vit-mae-base", must_be_file=False))
        if config.feature_extractor == "mae"
        else None
    )
    dtype = _dtype_for(config)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype, variant=config.model_variant, local_files_only=True)
    pipe.scheduler.set_timesteps(config.num_inference_steps)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    reference_unet = copy.deepcopy(pipe.unet).eval().requires_grad_(False)
    pipe.unet = _make_trainable(pipe.unet, config)
    if hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()
    _maybe_enable_xformers(pipe.unet)

    choice_models = resolve_choice_models(config.choice_models, config.choice_model)
    choice_model_weights = resolve_choice_model_weights(config.choice_model_weights, choice_models)
    config = SDXLDrPOConfig(**{**asdict(config), "choice_models": choice_models, "choice_model_weights": choice_model_weights})

    dataset = PromptDataset(config.prompt_file, pipe.tokenizer, max_samples=config.max_train_samples, seed=config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_preference_batch,
        drop_last=True,
    )
    extractor = FrozenViTMAEImageFeatureExtractor(mae_path, feature_keys=config.mae_feature_keys) if mae_path is not None else None
    selectors = build_choice_selectors(
        choice_models,
        accelerator.device,
        pickscore_model_path=config.pickscore_model_name_or_path,
        pickscore_processor_path=config.pickscore_processor_name_or_path or config.pickscore_model_name_or_path,
        local_files_only=not config.pickscore_allow_remote,
    )
    optimizer = torch.optim.AdamW(
        _trainable_parameters(pipe.unet),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    if extractor is None:
        pipe.unet, reference_unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet,
            reference_unet,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
            optimizer,
            dataloader,
            lr_scheduler,
        )
        extractor = SDXLTeacherUNetFeatureExtractor(
            reference_unet,
            feature_layers=config.teacher_feature_layers,
            feature_noise=config.teacher_feature_noise,
            feature_timestep=config.teacher_feature_timestep,
            pool_size=config.teacher_feature_pool_size,
        )
    else:
        pipe.unet, reference_unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, extractor, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet,
            reference_unet,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
            extractor,
            optimizer,
            dataloader,
            lr_scheduler,
        )
    latent_channels = int(accelerator.unwrap_model(pipe.unet).config.in_channels)
    latent_size = config.resolution // int(getattr(pipe.vae.config, "vae_scale_factor", 8) or 8)
    if hasattr(pipe, "vae_scale_factor"):
        latent_size = config.resolution // int(pipe.vae_scale_factor)

    if accelerator.is_main_process:
        accelerator.init_trackers("sdxl-turbo-drpo", {key: json.dumps(value) if isinstance(value, (list, tuple)) else value for key, value in asdict(config).items()})

    step = _resume_global_step(config.resume_from_checkpoint)
    progress = tqdm(total=config.max_train_steps, initial=step, disable=not accelerator.is_local_main_process)
    while step < config.max_train_steps:
        for batch in dataloader:
            if step >= config.max_train_steps:
                break
            assert isinstance(batch, Batch)
            with accelerator.accumulate(pipe.unet):
                sample_losses = []
                logs: dict[str, list[torch.Tensor]] = {}
                for prompt in batch.prompts:
                    prompts = [prompt] * config.batchsize_gen
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds = _encode_prompts(pipe, prompts, accelerator.device)
                    latents = torch.randn(
                        (config.batchsize_gen, latent_channels, latent_size, latent_size),
                        device=accelerator.device,
                        dtype=prompt_embeds.dtype,
                    )
                    generated_latents = _sdxl_one_step_latents(
                        pipe=pipe,
                        unet=pipe.unet,
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        resolution=config.resolution,
                        num_inference_steps=config.num_inference_steps,
                    )
                    with torch.no_grad():
                        reference_latents = _sdxl_one_step_latents(
                            pipe=pipe,
                            unet=reference_unet,
                            latents=latents,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            resolution=config.resolution,
                            num_inference_steps=config.num_inference_steps,
                        )
                    reward_images_tensor = _decode_latents_to_tensor(pipe.vae, generated_latents.detach(), chunk_size=config.vae_decode_chunk_size)
                    reward_images = _tensor_to_pil(reward_images_tensor)
                    scores, reward_info = score_reward_ensemble(selectors, config.choice_model_weights, reward_images, prompt, normalize=config.choice_score_normalize)
                    scores = scores.to(device=accelerator.device)
                    best_idx, worst_idx, feature_top_idx = select_disjoint_pref_indices(
                        scores,
                        num_pos=config.num_pos_images,
                        num_neg=config.num_neg_images,
                        feature_top_fraction=config.online_feature_top_fraction,
                    )
                    rank_info = add_rank_selection_stats(reward_info, scores, best_idx, worst_idx, feature_top_idx, prefix="online_reward")
                    feature_generated_latents = generated_latents.index_select(0, feature_top_idx)
                    feature_reference_latents = reference_latents.index_select(0, feature_top_idx)
                    positive_latents = generated_latents.index_select(0, best_idx).detach()
                    negative_latents = generated_latents.index_select(0, worst_idx).detach()

                    feature_keys = _active_feature_keys(config)
                    if config.feature_extractor == "mae":
                        generated_images = _decode_latents_to_tensor(pipe.vae, feature_generated_latents, chunk_size=config.vae_decode_chunk_size)
                        with torch.no_grad():
                            reference_images = _decode_latents_to_tensor(pipe.vae, feature_reference_latents, chunk_size=config.vae_decode_chunk_size)
                            positive_images = _decode_latents_to_tensor(pipe.vae, positive_latents, chunk_size=config.vae_decode_chunk_size)
                            negative_images = _decode_latents_to_tensor(pipe.vae, negative_latents, chunk_size=config.vae_decode_chunk_size)
                        generated_features = extractor.vector_features(generated_images, config.mae_feature_keys)
                        with torch.no_grad():
                            reference_features = extractor.vector_features(reference_images, config.mae_feature_keys)
                            positive_features = extractor.vector_features(positive_images, config.mae_feature_keys)
                            negative_features = extractor.vector_features(negative_images, config.mae_feature_keys)
                    else:
                        add_time_ids = _add_time_ids(pipe, config.batchsize_gen, config.resolution, accelerator.device, prompt_embeds.dtype)
                        feature_prompt_embeds, feature_pooled_prompt_embeds, feature_add_time_ids = _select_condition(
                            prompt_embeds,
                            pooled_prompt_embeds,
                            add_time_ids,
                            feature_top_idx,
                        )
                        pos_prompt_embeds, pos_pooled_prompt_embeds, pos_add_time_ids = _select_condition(
                            prompt_embeds,
                            pooled_prompt_embeds,
                            add_time_ids,
                            best_idx,
                        )
                        neg_prompt_embeds, neg_pooled_prompt_embeds, neg_add_time_ids = _select_condition(
                            prompt_embeds,
                            pooled_prompt_embeds,
                            add_time_ids,
                            worst_idx,
                        )
                        generated_features = extractor.vector_features(
                            feature_generated_latents,
                            prompt_embeds=feature_prompt_embeds,
                            pooled_prompt_embeds=feature_pooled_prompt_embeds,
                            add_time_ids=feature_add_time_ids,
                            require_grad=True,
                        )
                        reference_features = extractor.vector_features(
                            feature_reference_latents,
                            prompt_embeds=feature_prompt_embeds,
                            pooled_prompt_embeds=feature_pooled_prompt_embeds,
                            add_time_ids=feature_add_time_ids,
                            require_grad=False,
                        )
                        positive_features = extractor.vector_features(
                            positive_latents,
                            prompt_embeds=pos_prompt_embeds,
                            pooled_prompt_embeds=pos_pooled_prompt_embeds,
                            add_time_ids=pos_add_time_ids,
                            require_grad=False,
                        )
                        negative_features = extractor.vector_features(
                            negative_latents,
                            prompt_embeds=neg_prompt_embeds,
                            pooled_prompt_embeds=neg_pooled_prompt_embeds,
                            add_time_ids=neg_add_time_ids,
                            require_grad=False,
                        )
                    terms = _compute_prompt_terms(
                        feature_keys,
                        generated_features,
                        reference_features,
                        positive_features,
                        negative_features,
                        config,
                    )
                    ref_l2 = F.mse_loss(feature_generated_latents.float(), feature_reference_latents.float())
                    loss_i = terms["loss"] + config.ref_model_l2_weight * ref_l2
                    sample_losses.append(loss_i)
                    for key, value in {**terms, **rank_info, "ref_model_l2": ref_l2.detach()}.items():
                        if torch.is_tensor(value):
                            logs.setdefault(key, []).append(value.detach().float())
                if not sample_losses:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = torch.stack(sample_losses).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(_trainable_parameters(pipe.unet), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                log_values = {"train_loss": loss.detach(), "lr": torch.tensor(lr_scheduler.get_last_lr()[0], device=accelerator.device)}
                for key, values in logs.items():
                    if values:
                        log_values[key] = torch.stack(values).mean()
                accelerator.log({key: float(value.detach().cpu()) for key, value in log_values.items()}, step=step)
                progress.update(1)
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
                if step % config.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, pipe.unet, config, output_dir / f"checkpoint-{step}", step, "intermediate")
                if step >= config.max_train_steps:
                    break
    _save_checkpoint(accelerator, pipe.unet, config, output_dir / "final", step, "final")
    accelerator.end_training()


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train SDXL Turbo with online DrPO and MAE or teacher U-Net features.")
    parser.add_argument("--pretrained_model_name_or_path", default=str(root / "models" / "stable-diffusion-xl-turbo"))
    parser.add_argument("--output_dir", default=str(root / "outputs" / "sdxl-turbo-lora" / "drpo" / "mae" / datetime.now().strftime("%Y%m%d%H%M%S")))
    parser.add_argument("--prompt_file", default=str(root / "data" / "prompts" / "pickapicv2_test_unique.txt"))
    parser.add_argument("--mae_model_name_or_path", default=str(root / "models" / "facebook-vit-mae-base"))
    parser.add_argument("--model_variant", default="fp16")
    parser.add_argument("--choice_model", default="pickscore")
    parser.add_argument("--choice_models", default="")
    parser.add_argument("--choice_model_weights", default="")
    parser.add_argument("--choice_score_normalize", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--pickscore_model_name_or_path", default=str(root / "models" / "PickScore_v1"))
    parser.add_argument("--pickscore_processor_name_or_path", default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--batchsize_gen", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--feature_extractor", choices=["mae", "teacher_unet"], default="mae")
    parser.add_argument("--mae_feature_keys", default="layer12_patch_mean,layer12_patch_std,layer12_cls")
    parser.add_argument("--teacher_feature_layers", default="down_blocks.2,mid_block,up_blocks.0")
    parser.add_argument("--teacher_feature_noise", type=float, default=0.1)
    parser.add_argument("--teacher_feature_timestep", type=int, default=100)
    parser.add_argument("--teacher_feature_pool_size", type=int, default=4)
    parser.add_argument("--drifting_kernel", choices=["laplacian", "exponential", "rbf", "cosine"], default="laplacian")
    parser.add_argument("--drifting_pref_r_list", default="0.02,0.05,0.2")
    parser.add_argument("--drifting_ref_r_list", default="0.02,0.05,0.2")
    parser.add_argument("--drifting_pos_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_loss_weight", type=float, default=0.2)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.02)
    parser.add_argument("--num_pos_images", type=int, default=2)
    parser.add_argument("--num_neg_images", type=int, default=2)
    parser.add_argument("--online_feature_top_fraction", type=float, default=1.0)
    parser.add_argument("--vae_decode_chunk_size", type=int, default=1)
    parser.add_argument("--mae_chunk_size", type=int, default=4)
    return parser


def parse_config(argv: list[str] | None = None) -> SDXLDrPOConfig:
    args = build_parser().parse_args(argv)
    return SDXLDrPOConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        mae_model_name_or_path=args.mae_model_name_or_path,
        model_variant=args.model_variant or None,
        choice_model=args.choice_model,
        choice_models=_parse_names(args.choice_models),
        choice_model_weights=_parse_floats(args.choice_model_weights) if args.choice_model_weights else (),
        choice_score_normalize=args.choice_score_normalize,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        pickscore_processor_name_or_path=args.pickscore_processor_name_or_path,
        pickscore_allow_remote=args.pickscore_allow_remote,
        seed=args.seed,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        batchsize_gen=args.batchsize_gen,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        max_train_samples=args.max_train_samples,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        checkpointing_steps=args.checkpointing_steps,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        use_lora=args.use_lora,
        resume_from_checkpoint=args.resume_from_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_parse_names(args.lora_target_modules),
        feature_extractor=args.feature_extractor,
        mae_feature_keys=_parse_names(args.mae_feature_keys),
        teacher_feature_layers=_parse_names(args.teacher_feature_layers),
        teacher_feature_noise=args.teacher_feature_noise,
        teacher_feature_timestep=args.teacher_feature_timestep,
        teacher_feature_pool_size=args.teacher_feature_pool_size,
        drifting_kernel=args.drifting_kernel,
        drifting_pref_r_list=_parse_floats(args.drifting_pref_r_list),
        drifting_ref_r_list=_parse_floats(args.drifting_ref_r_list),
        drifting_pos_weight=args.drifting_pos_weight,
        drifting_neg_weight=args.drifting_neg_weight,
        drifting_ref_weight=args.drifting_ref_weight,
        drifting_ref_neg_weight=args.drifting_ref_neg_weight,
        drifting_ref_loss_weight=args.drifting_ref_loss_weight,
        ref_model_l2_weight=args.ref_model_l2_weight,
        num_pos_images=args.num_pos_images,
        num_neg_images=args.num_neg_images,
        online_feature_top_fraction=args.online_feature_top_fraction,
        vae_decode_chunk_size=args.vae_decode_chunk_size,
        mae_chunk_size=args.mae_chunk_size,
    )


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
