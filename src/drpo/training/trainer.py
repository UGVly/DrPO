from __future__ import annotations

import copy
import json
import logging
import math
import os
import shlex
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from packaging import version
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from drpo.config import DrPOConfig
from drpo.data import Batch, PreferenceDataset, PromptDataset, collate_preference_batch, load_prompt_file
from drpo.drift import DRIFT_KERNELS, DriftRadii, drift_loss, pairwise_l2
from drpo.features import DirectLatentFeatureExtractor, FrozenDinoImageFeatureExtractor, FrozenMAELatentFeatureExtractor
from drpo.rewards import (
    build_choice_selectors,
    resolve_choice_models,
    resolve_choice_model_weights,
    score_reward_ensemble,
)
from drpo.sdturbo import (
    decode_latents_to_pil,
    decode_latents_to_tensor,
    encode_images,
    encode_prompts,
    load_sdturbo_components,
    load_unet_checkpoint_weights,
    run_one_step_unet,
)
from drpo.utils.tensors import (
    add_rank_selection_stats,
    select_disjoint_pref_indices,
)


logger = logging.getLogger(__name__)


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _maybe_enable_xformers(unet) -> None:
    if not is_xformers_available():
        logger.warning("xformers is not available; continuing without memory efficient attention.")
        return
    import xformers

    if version.parse(xformers.__version__) < version.parse("0.0.17"):
        logger.warning("xformers version %s is too old; continuing without it.", xformers.__version__)
        return
    try:
        unet.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory efficient attention.")
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("Failed to enable xformers memory efficient attention: %s", exc)


def _maybe_enable_gradient_checkpointing(unet) -> None:
    if not hasattr(unet, "enable_gradient_checkpointing"):
        return
    try:
        unet.enable_gradient_checkpointing()
        logger.info("Enabled UNet gradient checkpointing.")
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("Failed to enable UNet gradient checkpointing: %s", exc)


def _maybe_enable_vae_memory_optimizations(vae) -> None:
    for method_name, log_label in (("enable_slicing", "VAE slicing"), ("enable_tiling", "VAE tiling")):
        if not hasattr(vae, method_name):
            continue
        try:
            getattr(vae, method_name)()
            logger.info("Enabled %s.", log_label)
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("Failed to enable %s: %s", log_label, exc)


def _maybe_add_lora(unet, config: DrPOConfig):
    if not config.use_lora:
        return unet
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )
    return get_peft_model(unet, lora_config)


def _setup_logging(config: DrPOConfig, accelerator: Accelerator) -> str:
    log_dir = Path(config.output_dir) / config.logging_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=f"%(asctime)s | %(levelname)s | rank={accelerator.process_index}/{accelerator.num_processes} | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return str(log_path)


def _make_json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)


def _make_tracker_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(_make_json_safe(value), ensure_ascii=False, sort_keys=True)


def _save_runtime_snapshot(config: DrPOConfig, accelerator: Accelerator) -> None:
    snapshot_dir = Path(config.output_dir) / "run_metadata"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {key: _make_json_safe(value) for key, value in vars(config).items()}
    with (snapshot_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with (snapshot_dir / "resolved_config.txt").open("w", encoding="utf-8") as handle:
        for key in sorted(resolved_config):
            handle.write(f"{key}={resolved_config[key]}\n")
    with (snapshot_dir / "runtime_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "hostname": socket.gethostname(),
                "cwd": os.getcwd(),
                "python_executable": sys.executable,
                "script_path": os.path.abspath(__file__),
                "world_size": accelerator.num_processes,
                "main_process_index": accelerator.process_index,
                "device": str(accelerator.device),
                "mixed_precision": accelerator.mixed_precision,
                "torch_version": torch.__version__,
            },
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    with (snapshot_dir / "launch_command.sh").open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write(shlex.join([sys.executable, *sys.argv]) + "\n")


def _checkpoint_step(checkpoint_dir: Path) -> int:
    metadata_path = checkpoint_dir / "training_state.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return max(int(metadata.get("global_step", 0)), 0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to read checkpoint metadata from %s.", metadata_path)
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        suffix = name.removeprefix("checkpoint-")
        if suffix.isdigit():
            return int(suffix)
    return 0


def _latest_checkpoint(output_dir: Path) -> Path:
    candidates = [(step, path) for path in output_dir.glob("checkpoint-*") if path.is_dir() and (step := _checkpoint_step(path)) > 0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {output_dir}.")
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_resume_checkpoint(config: DrPOConfig) -> Path | None:
    if not config.resume_from_checkpoint:
        return None
    if config.resume_from_checkpoint == "latest":
        return _latest_checkpoint(Path(config.output_dir))
    checkpoint_dir = Path(config.resume_from_checkpoint).expanduser()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"resume_from_checkpoint does not exist: {checkpoint_dir}")
    return checkpoint_dir


def _save_training_checkpoint(
    accelerator: Accelerator,
    unet,
    config: DrPOConfig,
    checkpoint_dir: Path,
    global_step: int,
    checkpoint_type: str,
) -> None:
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_to_save = accelerator.unwrap_model(unet)
        if config.use_lora:
            unet_to_save.save_pretrained(checkpoint_dir / "unet_lora")
        else:
            unet_to_save.save_pretrained(checkpoint_dir / "unet")
        metadata = {
            "checkpoint_type": checkpoint_type,
            "created_at": datetime.now().isoformat(),
            "global_step": global_step,
            "max_train_steps": config.max_train_steps,
            "checkpointing_steps": config.checkpointing_steps,
            "output_dir": config.output_dir,
            "resume_from_checkpoint": str(checkpoint_dir),
            "contains_accelerate_state": True,
            "contains_model_state": True,
            "contains_optimizer_state": True,
            "contains_scheduler_state": True,
            "contains_rng_state": True,
        }
        with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
    accelerator.wait_for_everyone()


def _validate_config(config: DrPOConfig) -> None:
    resolved_choice_models = config.choice_models or (config.choice_model,)
    if config.num_pos_images < 1 or config.num_neg_images < 1:
        raise ValueError("num_pos_images and num_neg_images must both be >= 1.")
    required_pos_images = config.num_pos_images
    if config.num_pos_images_min is not None:
        if config.num_pos_images_min < 1:
            raise ValueError("num_pos_images_min must be >= 1 when set.")
        if config.num_pos_images_min > config.num_pos_images:
            raise ValueError("num_pos_images_min must be <= num_pos_images.")
        required_pos_images = config.num_pos_images_min
    if config.train_mode in {"online", "offline_distance"}:
        if config.batchsize_gen < 2:
            raise ValueError("batchsize_gen must be >= 2 for online/offline_distance ranking.")
        max_rank_candidates = config.batchsize_gen
        if config.train_mode == "online" and resolved_choice_models == ("geneval",):
            max_rank_candidates = config.batchsize_gen * config.geneval_max_rollout_rounds
        if required_pos_images + config.num_neg_images > max_rank_candidates:
            raise ValueError(
                "num_pos_images + num_neg_images must fit the available generated candidates "
                f"({required_pos_images} + {config.num_neg_images} > {max_rank_candidates})."
            )
        max_feature_candidates = max_rank_candidates - config.num_neg_images
        if config.online_feature_top_count > 0 and config.online_feature_top_count > max_feature_candidates:
            raise ValueError(
                "online_feature_top_count must fit the available optimization candidates after removing negative anchors "
                f"({config.online_feature_top_count} > {max_feature_candidates})."
            )
    if config.online_feature_top_count < 0:
        raise ValueError("online_feature_top_count must be >= 0.")
    if not (0.0 < config.online_feature_top_fraction <= 1.0):
        raise ValueError("online_feature_top_fraction must be in (0, 1].")
    if not (0.0 < config.offline_distance_top_fraction <= 1.0):
        raise ValueError("offline_distance_top_fraction must be in (0, 1].")
    if config.resolution % 8:
        raise ValueError(f"Resolution must be divisible by 8, got {config.resolution}.")
    if config.drifting_kernel not in DRIFT_KERNELS:
        raise ValueError(f"drifting_kernel must be one of {DRIFT_KERNELS}, got {config.drifting_kernel}.")
    if config.num_eval_prompts < 1:
        raise ValueError("num_eval_prompts must be >= 1.")
    if config.eval_every_steps < 0:
        raise ValueError("eval_every_steps must be >= 0.")
    if not (0.0 < config.sample_grad_norm_discard_quantile <= 1.0):
        raise ValueError("sample_grad_norm_discard_quantile must be in (0, 1].")
    if config.per_sample_threshold_quantile >= 1.0 or config.per_sample_threshold_quantile == 0.0:
        raise ValueError("per_sample_threshold_quantile must be negative to disable or in (0, 1) to enable.")
    if not (0.0 <= config.per_sample_threshold_ema_decay < 1.0):
        raise ValueError("per_sample_threshold_ema_decay must be in [0, 1).")
    if config.generation_timestep < 0:
        raise ValueError("generation_timestep must be >= 0.")
    if config.generation_target_timestep > config.generation_timestep:
        raise ValueError("generation_target_timestep must be <= generation_timestep, or -1.")
    if config.geneval_max_rollout_rounds < 1:
        raise ValueError("geneval_max_rollout_rounds must be >= 1.")
    if "geneval" in resolved_choice_models:
        if config.train_mode != "online":
            raise ValueError("choice_model=geneval is only supported with train_mode=online.")
        if len(resolved_choice_models) != 1:
            raise ValueError("choice_model=geneval does not support reward ensembles.")


def _feature_keys(config: DrPOConfig, extractor) -> tuple[str, ...]:
    if config.drifting_feature_keys:
        return config.drifting_feature_keys
    if config.drifting_feature_mode == "single":
        return (config.drifting_feature_key,)
    return extractor.default_feature_keys()


def _build_feature_extractor(config: DrPOConfig):
    if config.drifting_feature_extractor == "mae":
        return FrozenMAELatentFeatureExtractor(
            config.drifting_mae_path,
            feature_key=config.drifting_feature_key,
            block_stride=config.drifting_feature_block_stride,
            patch_pool_sizes=config.drifting_feature_patch_sizes,
            include_input_sq_mean=config.drifting_include_input_sq_mean,
            include_spatial_features=config.drifting_include_spatial_features,
        )
    if config.drifting_feature_extractor == "dino":
        return FrozenDinoImageFeatureExtractor(
            config.drifting_dino_model_name_or_path,
            processor_name_or_path=config.drifting_dino_processor_name_or_path,
            feature_key=config.drifting_feature_key,
            block_stride=config.drifting_feature_block_stride,
            patch_pool_sizes=config.drifting_feature_patch_sizes,
            local_files_only=not config.drifting_dino_allow_remote,
        )
    if config.drifting_feature_extractor == "latent":
        return DirectLatentFeatureExtractor(
            feature_key=config.drifting_feature_key,
            patch_pool_sizes=config.drifting_feature_patch_sizes,
            include_spatial_features=True,
        )
    raise ValueError(f"Unknown drifting_feature_extractor: {config.drifting_feature_extractor}")


def _vector_features_from_latents(extractor, vae, latents: torch.Tensor, feature_keys: tuple[str, ...], config: DrPOConfig) -> dict[str, torch.Tensor]:
    if config.drifting_feature_extractor == "dino":
        if not getattr(vae, "_drpo_dino_memory_efficient_decode", False):
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            setattr(vae, "_drpo_dino_memory_efficient_decode", True)
        chunk_size = max(1, int(os.environ.get("DRPO_DINO_FEATURE_CHUNK_SIZE", "1")))
        chunks = []
        for latent_chunk in latents.split(chunk_size):
            images = decode_latents_to_tensor(vae, latent_chunk, chunk_size=chunk_size)
            chunks.append(extractor.vector_features(images, feature_keys))
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in feature_keys}
    return extractor.vector_features(latents, feature_keys)


def _vector_features_from_images_or_latents(extractor, vae, images: torch.Tensor, feature_keys: tuple[str, ...], config: DrPOConfig) -> dict[str, torch.Tensor]:
    if config.drifting_feature_extractor == "dino":
        return extractor.vector_features(images, feature_keys)
    latents = encode_images(vae, images, mode=config.offline_latent_encode_mode)
    return extractor.vector_features(latents, feature_keys)


def _normalize_vector_for_ranking(values: torch.Tensor, mode: str = "zscore", eps: float = 1e-6) -> torch.Tensor:
    values = values.float()
    if mode == "none":
        return values
    if mode == "zscore":
        std = torch.clamp(values.std(unbiased=False), min=eps)
        return (values - values.mean()) / std
    raise ValueError(f"Unknown offline-distance score normalization mode: {mode}")


def _feature_distance_to_refs(cand_feat: torch.Tensor, ref_feat: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    cand_flat = cand_feat.float().reshape(cand_feat.shape[0], -1)
    ref_flat = ref_feat.float().reshape(ref_feat.shape[0], -1)
    if cand_flat.shape[1] != ref_flat.shape[1]:
        raise ValueError(
            "Feature dimension mismatch for offline-distance ranking: "
            f"cand={tuple(cand_flat.shape)}, ref={tuple(ref_flat.shape)}"
        )
    distances = torch.cdist(cand_flat, ref_flat, p=2)
    if reduction == "mean":
        return distances.mean(dim=1)
    if reduction == "min":
        return distances.min(dim=1).values
    raise ValueError(f"Unknown reference distance reduction: {reduction}")


def _feature_cosine_similarity_to_refs(
    cand_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    cand_flat = cand_feat.float().reshape(cand_feat.shape[0], -1)
    ref_flat = ref_feat.float().reshape(ref_feat.shape[0], -1)
    if cand_flat.shape[1] != ref_flat.shape[1]:
        raise ValueError(
            "Feature dimension mismatch for offline cosine ranking: "
            f"cand={tuple(cand_flat.shape)}, ref={tuple(ref_flat.shape)}"
        )
    cand_flat = F.normalize(cand_flat, p=2, dim=1, eps=eps)
    ref_flat = F.normalize(ref_flat, p=2, dim=1, eps=eps)
    similarities = cand_flat @ ref_flat.t()
    if reduction == "mean":
        return similarities.mean(dim=1)
    if reduction == "min":
        return similarities.max(dim=1).values
    raise ValueError(f"Unknown reference similarity reduction: {reduction}")


def compute_offline_distance_scores(
    cand_feat_dict: dict[str, torch.Tensor],
    pos_feat_dict: dict[str, torch.Tensor],
    neg_feat_dict: dict[str, torch.Tensor],
    feature_keys: tuple[str, ...],
    *,
    score_mode: str,
    normalize: str,
    aggregation: str,
    ref_reduction: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    score_terms: list[torch.Tensor] = []
    d_pos_terms: list[torch.Tensor] = []
    d_neg_terms: list[torch.Tensor] = []
    margin_terms: list[torch.Tensor] = []
    sim_pos_terms: list[torch.Tensor] = []
    sim_neg_terms: list[torch.Tensor] = []
    for feature_name in feature_keys:
        cand_feat = cand_feat_dict[feature_name]
        pos_feat = pos_feat_dict[feature_name]
        neg_feat = neg_feat_dict[feature_name]
        d_pos = _feature_distance_to_refs(cand_feat, pos_feat, reduction=ref_reduction)
        d_neg = _feature_distance_to_refs(cand_feat, neg_feat, reduction=ref_reduction)
        margin = d_neg - d_pos
        if score_mode == "margin":
            score = _normalize_vector_for_ranking(margin, mode=normalize)
        elif score_mode == "separate_zscore":
            score = _normalize_vector_for_ranking(d_neg, mode=normalize) - _normalize_vector_for_ranking(d_pos, mode=normalize)
        elif score_mode == "cosine_softmax_diff":
            sim_pos = _feature_cosine_similarity_to_refs(cand_feat, pos_feat, reduction=ref_reduction)
            sim_neg = _feature_cosine_similarity_to_refs(cand_feat, neg_feat, reduction=ref_reduction)
            score = F.softmax(sim_pos, dim=0) - F.softmax(sim_neg, dim=0)
            sim_pos_terms.append(sim_pos)
            sim_neg_terms.append(sim_neg)
        else:
            raise ValueError(f"Unknown offline-distance score mode: {score_mode}")
        score_terms.append(score)
        d_pos_terms.append(d_pos)
        d_neg_terms.append(d_neg)
        margin_terms.append(margin)
    score_stack = torch.stack(score_terms, dim=0)
    d_pos_stack = torch.stack(d_pos_terms, dim=0)
    d_neg_stack = torch.stack(d_neg_terms, dim=0)
    margin_stack = torch.stack(margin_terms, dim=0)
    sim_pos_stack = torch.stack(sim_pos_terms, dim=0) if sim_pos_terms else None
    sim_neg_stack = torch.stack(sim_neg_terms, dim=0) if sim_neg_terms else None
    if aggregation == "mean":
        final_score = score_stack.mean(dim=0)
    elif aggregation == "sum":
        final_score = score_stack.sum(dim=0)
    else:
        raise ValueError(f"Unknown offline-distance score aggregation: {aggregation}")
    final_d_pos = d_pos_stack.mean(dim=0)
    final_d_neg = d_neg_stack.mean(dim=0)
    final_margin = margin_stack.mean(dim=0)
    info = {
        "offline_distance_score_mean": final_score.mean().detach(),
        "offline_distance_score_std": final_score.std(unbiased=False).detach(),
        "offline_distance_score_max": final_score.max().detach(),
        "offline_distance_score_min": final_score.min().detach(),
        "offline_distance_d_pos_mean": final_d_pos.mean().detach(),
        "offline_distance_d_neg_mean": final_d_neg.mean().detach(),
        "offline_distance_margin_mean": final_margin.mean().detach(),
        "offline_distance_margin_std": final_margin.std(unbiased=False).detach(),
    }
    if sim_pos_stack is not None and sim_neg_stack is not None:
        final_sim_pos = sim_pos_stack.mean(dim=0)
        final_sim_neg = sim_neg_stack.mean(dim=0)
        info["offline_distance_cos_pos_mean"] = final_sim_pos.mean().detach()
        info["offline_distance_cos_neg_mean"] = final_sim_neg.mean().detach()
        info["offline_distance_cos_margin_mean"] = (final_sim_pos - final_sim_neg).mean().detach()
    return final_score, info


def _feature_set(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor with shape (B, N, D), got {tuple(features.shape)}.")
    return features.reshape(1, -1, features.shape[-1])


def _compute_prompt_terms(
    feature_keys: tuple[str, ...],
    generated_features: dict[str, torch.Tensor],
    reference_features: dict[str, torch.Tensor],
    positive_features: dict[str, torch.Tensor],
    negative_features: dict[str, torch.Tensor],
    config: DrPOConfig,
) -> dict[str, torch.Tensor]:
    pref_terms: list[torch.Tensor] = []
    ref_terms: list[torch.Tensor] = []
    feat_l2_terms: list[torch.Tensor] = []
    d_pos_terms: list[torch.Tensor] = []
    d_neg_terms: list[torch.Tensor] = []
    d_ref_terms: list[torch.Tensor] = []
    for key in feature_keys:
        gen = _feature_set(generated_features[key])
        ref = _feature_set(reference_features[key])
        pos = _feature_set(positive_features[key])
        neg = _feature_set(negative_features[key])
        pref_loss_vec, _ = drift_loss(
            gen,
            pos,
            neg,
            positive_weight=config.drifting_pos_weight,
            negative_weight=config.drifting_neg_weight,
            radii=DriftRadii(config.drifting_pref_r_list, config.drifting_ref_r_list).preference,
            kernel=config.drifting_kernel,
        )
        ref_loss_vec, _ = drift_loss(
            gen,
            ref,
            gen.detach(),
            positive_weight=config.drifting_ref_weight,
            negative_weight=config.drifting_ref_neg_weight,
            radii=DriftRadii(config.drifting_pref_r_list, config.drifting_ref_r_list).reference,
            mask_negative_self=True,
            kernel=config.drifting_kernel,
        )
        pref_terms.append(pref_loss_vec.mean())
        ref_terms.append(ref_loss_vec.mean())
        feat_l2_terms.append(F.mse_loss(gen.float(), ref.float()))
        d_pos_terms.append(pairwise_l2(gen.float(), pos.float()).mean())
        d_neg_terms.append(pairwise_l2(gen.float(), neg.float()).mean())
        d_ref_terms.append(pairwise_l2(gen.float(), ref.float()).mean())
    reduce = torch.mean if config.drifting_feature_aggregation == "mean" else torch.sum
    pref_loss = reduce(torch.stack(pref_terms))
    ref_loss = reduce(torch.stack(ref_terms))
    feat_l2 = reduce(torch.stack(feat_l2_terms))
    d_pos = torch.stack(d_pos_terms).mean()
    d_neg = torch.stack(d_neg_terms).mean()
    d_ref = torch.stack(d_ref_terms).mean()
    loss = pref_loss
    if config.drifting_ref_loss_weight != 0.0:
        loss = loss + config.drifting_ref_loss_weight * ref_loss
    if config.frozen_feature_l2_weight != 0.0:
        loss = loss + config.frozen_feature_l2_weight * feat_l2
    return {
        "loss": loss,
        "pref_loss": pref_loss.detach(),
        "ref_loss": ref_loss.detach(),
        "feature_l2": feat_l2.detach(),
        "d_pos": d_pos.detach(),
        "d_neg": d_neg.detach(),
        "d_ref": d_ref.detach(),
    }


def _compute_loss_grad_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    sq_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
    for grad in grads:
        if grad is not None:
            sq_norm = sq_norm + grad.detach().float().pow(2).sum()
    return torch.sqrt(sq_norm)


def _build_sample_grad_norm_filter(
    sample_losses: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    discard_quantile: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    norms = torch.stack([_compute_loss_grad_norm(loss_i, parameters) for loss_i in sample_losses.unbind()])
    safe_norms = torch.nan_to_num(norms.detach(), nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    if safe_norms.numel() == 1:
        threshold = safe_norms[0]
        keep_mask = torch.ones_like(safe_norms, dtype=torch.bool)
    else:
        threshold = torch.quantile(safe_norms, discard_quantile)
        keep_mask = safe_norms <= threshold
        if not bool(keep_mask.any()):
            keep_mask[torch.argmin(safe_norms)] = True
    return keep_mask, {
        "sample_grad_norm_mean": safe_norms.mean().detach(),
        "sample_grad_norm_max": safe_norms.max().detach(),
        "sample_grad_norm_threshold": threshold.detach(),
        "sample_grad_keep_ratio": keep_mask.float().mean().detach(),
    }


def _build_per_sample_threshold_filter(
    sample_losses: torch.Tensor,
    threshold_quantile: float,
    *,
    threshold_ema: torch.Tensor | None,
    ema_decay: float,
    use_sqrt: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    detached_losses = sample_losses.detach().float()
    scores = torch.sqrt(torch.clamp(detached_losses, min=0.0)) if use_sqrt else detached_losses
    finite_mask = torch.isfinite(scores)
    finite_scores = scores[finite_mask]
    threshold = torch.quantile(finite_scores, threshold_quantile) if finite_scores.numel() > 0 else scores.new_tensor(float("inf"))
    if threshold_ema is None:
        threshold_ema = threshold.detach()
    else:
        threshold_ema = ema_decay * threshold_ema + (1.0 - ema_decay) * threshold.detach()
    keep_mask = finite_mask & (scores < threshold_ema)
    if not bool(keep_mask.any()):
        masked = torch.where(finite_mask, scores, torch.full_like(scores, float("inf")))
        keep_mask = torch.zeros_like(finite_mask, dtype=torch.bool)
        keep_mask[torch.argmin(masked)] = True
    return keep_mask, {
        "per_sample_threshold_raw": threshold.detach(),
        "per_sample_threshold_ema": threshold_ema.detach(),
        "per_sample_threshold_keep_ratio": keep_mask.float().mean().detach(),
    }, threshold_ema.detach()


def _reduce_sample_values(values: torch.Tensor, keep_mask: torch.Tensor, reduction: str) -> torch.Tensor:
    mask_f = keep_mask.to(device=values.device, dtype=values.dtype)
    if reduction == "masked_mean":
        return (values * mask_f).mean()
    if reduction == "kept_mean":
        return (values * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)
    raise ValueError(f"Unknown reduction: {reduction}")


def _compute_task_gradient_stats(
    pref_loss: torch.Tensor,
    ref_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    ref_weight: float,
) -> dict[str, torch.Tensor]:
    pref_grads = torch.autograd.grad(pref_loss, parameters, retain_graph=True, allow_unused=True)
    ref_grads = torch.autograd.grad(ref_loss, parameters, retain_graph=True, allow_unused=True)
    pref_flat = torch.cat([(grad.detach().float().reshape(-1) if grad is not None else torch.zeros(param.numel(), device=pref_loss.device)) for param, grad in zip(parameters, pref_grads)])
    ref_flat = torch.cat([(grad.detach().float().reshape(-1) if grad is not None else torch.zeros(param.numel(), device=ref_loss.device)) for param, grad in zip(parameters, ref_grads)])
    pref_norm = torch.norm(pref_flat)
    ref_norm = torch.norm(ref_flat)
    cosine = torch.sum(pref_flat * ref_flat) / torch.clamp(pref_norm * ref_norm, min=1e-12)
    return {
        "pref_grad_norm": pref_norm.detach(),
        "ref_grad_norm": ref_norm.detach(),
        "weighted_ref_grad_norm": (ref_norm * ref_weight).detach(),
        "pref_ref_grad_cosine": cosine.detach(),
    }


def _geneval_feature_target_count(config: DrPOConfig) -> int:
    return config.online_feature_top_count if config.online_feature_top_count > 0 else 0


def _geneval_stop_requirements_met(
    *,
    pos_count: int,
    neg_count: int,
    config: DrPOConfig,
) -> bool:
    required_pos = config.num_pos_images if config.num_pos_images_min is None else config.num_pos_images_min
    if pos_count < required_pos or neg_count < config.num_neg_images:
        return False
    feature_target_count = _geneval_feature_target_count(config)
    if feature_target_count <= 0:
        return True
    available_feature_count = pos_count + max(0, neg_count - config.num_neg_images)
    return available_feature_count >= feature_target_count


def _append_capped_geneval_candidates(
    existing_latents: torch.Tensor | None,
    existing_scores: torch.Tensor | None,
    new_latents: torch.Tensor,
    new_scores: torch.Tensor,
    *,
    keep_cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents_cpu = new_latents.detach().cpu()
    scores_cpu = new_scores.detach().cpu()
    if existing_latents is not None:
        latents_cpu = torch.cat([existing_latents, latents_cpu], dim=0)
        scores_cpu = torch.cat([existing_scores, scores_cpu], dim=0)
    if keep_cap > 0 and latents_cpu.shape[0] > keep_cap:
        keep_idx = torch.randperm(latents_cpu.shape[0])[:keep_cap]
        latents_cpu = latents_cpu.index_select(0, keep_idx)
        scores_cpu = scores_cpu.index_select(0, keep_idx)
    return latents_cpu, scores_cpu


def _sample_geneval_pref_indices(
    raw_scores: torch.Tensor,
    *,
    num_pos_min: int | None,
    num_pos: int,
    num_neg: int,
    feature_top_count: int,
    feature_top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_scores.ndim != 1:
        raise ValueError(f"Expected 1D raw_scores, got shape={tuple(raw_scores.shape)}")
    pos_idx = torch.nonzero(raw_scores >= 0.5, as_tuple=False).flatten()
    neg_idx = torch.nonzero(raw_scores < 0.5, as_tuple=False).flatten()
    required_pos = num_pos if num_pos_min is None else num_pos_min
    if pos_idx.numel() < required_pos or neg_idx.numel() < num_neg:
        raise ValueError("Not enough positive or negative Geneval samples to draw from.")

    pos_order = torch.randperm(pos_idx.numel(), device=raw_scores.device)
    neg_order = torch.randperm(neg_idx.numel(), device=raw_scores.device)
    shuffled_pos_idx = pos_idx.index_select(0, pos_order)
    shuffled_neg_idx = neg_idx.index_select(0, neg_order)
    pos_count = min(pos_idx.numel(), num_pos)
    best_idx = shuffled_pos_idx[:pos_count]
    worst_idx = shuffled_neg_idx[:num_neg]

    available_feature_idx = torch.cat([shuffled_pos_idx, shuffled_neg_idx], dim=0)
    available_feature_idx = available_feature_idx[~(available_feature_idx[:, None] == worst_idx[None, :]).any(dim=1)]
    if available_feature_idx.numel() == 0:
        feature_top_idx = best_idx[:1]
    elif feature_top_count > 0:
        if available_feature_idx.numel() < feature_top_count:
            raise ValueError("Not enough optimization candidates after removing negative anchors.")
        feature_top_idx = available_feature_idx[:feature_top_count]
    elif feature_top_fraction >= 1.0:
        feature_top_idx = available_feature_idx
    else:
        top_count = min(raw_scores.numel(), max(1, math.ceil(raw_scores.numel() * feature_top_fraction)))
        feature_top_idx = available_feature_idx[: min(top_count, available_feature_idx.numel())]
        if feature_top_idx.numel() == 0:
            feature_top_idx = best_idx[:1]
    return best_idx, worst_idx, feature_top_idx


def _collect_geneval_online_latents(
    *,
    config: DrPOConfig,
    accelerator: Accelerator,
    unet,
    reference_unet,
    vae,
    scheduler,
    selectors,
    prompt_embedding: torch.Tensor,
    prompt_metadata: dict[str, object],
    unet_in_channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | None:
    feature_target_count = _geneval_feature_target_count(config)
    pos_keep_cap = max(config.num_pos_images, config.num_pos_images_min or 0, feature_target_count)
    neg_keep_cap = config.num_neg_images + feature_target_count
    pos_noisy_latents: torch.Tensor | None = None
    pos_scores: torch.Tensor | None = None
    neg_noisy_latents: torch.Tensor | None = None
    neg_scores: torch.Tensor | None = None
    rollout_rounds = 0
    total_candidates = 0
    total_positive = 0
    raw_score_sum = 0.0
    raw_score_sum_sq = 0.0

    if "geneval" not in selectors or len(selectors) != 1:
        raise ValueError("Geneval rollout collection requires choice_models=('geneval',).")

    for rollout_round in range(config.geneval_max_rollout_rounds):
        del rollout_round
        rollout_rounds += 1
        noisy_latents = torch.randn(
            (config.batchsize_gen, unet_in_channels, config.resolution // 8, config.resolution // 8),
            device=accelerator.device,
            dtype=prompt_embedding.dtype,
        )
        with torch.no_grad():
            _, generated_latents = run_one_step_unet(
                unet,
                noisy_latents,
                prompt_embedding,
                scheduler,
                timestep=config.generation_timestep,
                target_timestep=config.generation_target_timestep,
            )
        images = decode_latents_to_pil(vae, generated_latents.detach())
        raw_scores = torch.tensor(selectors["geneval"].score(images, prompt_metadata), device=generated_latents.device, dtype=torch.float32)
        pos_mask = raw_scores >= 0.5
        neg_mask = ~pos_mask
        total_candidates += int(raw_scores.numel())
        total_positive += int(pos_mask.sum().item())
        raw_score_sum += float(raw_scores.sum().item())
        raw_score_sum_sq += float(raw_scores.float().pow(2).sum().item())
        if bool(pos_mask.any()):
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
            pos_noisy_latents, pos_scores = _append_capped_geneval_candidates(
                pos_noisy_latents,
                pos_scores,
                noisy_latents.index_select(0, pos_idx),
                raw_scores.index_select(0, pos_idx),
                keep_cap=pos_keep_cap,
            )
        if bool(neg_mask.any()):
            neg_idx = torch.nonzero(neg_mask, as_tuple=False).flatten()
            neg_noisy_latents, neg_scores = _append_capped_geneval_candidates(
                neg_noisy_latents,
                neg_scores,
                noisy_latents.index_select(0, neg_idx),
                raw_scores.index_select(0, neg_idx),
                keep_cap=neg_keep_cap,
            )
        pos_count = 0 if pos_scores is None else int(pos_scores.numel())
        neg_count = 0 if neg_scores is None else int(neg_scores.numel())
        if _geneval_stop_requirements_met(pos_count=pos_count, neg_count=neg_count, config=config):
            break

    if pos_noisy_latents is None or pos_scores is None or neg_noisy_latents is None or neg_scores is None:
        return None
    all_noisy_latents = torch.cat([pos_noisy_latents, neg_noisy_latents], dim=0)
    all_raw_scores = torch.cat([pos_scores, neg_scores], dim=0).to(device=accelerator.device, dtype=torch.float32)
    all_scores = all_raw_scores
    pos_mask = all_raw_scores >= 0.5
    neg_mask = ~pos_mask
    required_pos = config.num_pos_images if config.num_pos_images_min is None else config.num_pos_images_min
    if int(pos_mask.sum().item()) < required_pos or int(neg_mask.sum().item()) < config.num_neg_images:
        return None

    best_idx, worst_idx, feature_top_idx = _sample_geneval_pref_indices(
        all_raw_scores,
        num_pos_min=config.num_pos_images_min,
        num_pos=config.num_pos_images,
        num_neg=config.num_neg_images,
        feature_top_count=config.online_feature_top_count,
        feature_top_fraction=config.online_feature_top_fraction,
    )
    # Keep reward screening graph-free, then rebuild gradients only for candidates
    # that actually participate in the loss.
    selected_idx = torch.unique(torch.cat([best_idx, worst_idx, feature_top_idx]), sorted=True)
    remap = torch.full((all_scores.numel(),), -1, device=all_scores.device, dtype=torch.long)
    remap[selected_idx] = torch.arange(selected_idx.numel(), device=all_scores.device)
    selected_noisy_latents = all_noisy_latents.index_select(0, selected_idx.detach().cpu()).to(
        device=accelerator.device,
        dtype=prompt_embedding.dtype,
    )
    selected_prompt_embedding = prompt_embedding[:1].expand(selected_noisy_latents.shape[0], -1, -1)
    _, selected_generated = run_one_step_unet(
        unet,
        selected_noisy_latents,
        selected_prompt_embedding,
        scheduler,
        timestep=config.generation_timestep,
        target_timestep=config.generation_target_timestep,
    )
    with torch.no_grad():
        _, selected_reference = run_one_step_unet(
            reference_unet,
            selected_noisy_latents,
            selected_prompt_embedding,
            scheduler,
            timestep=config.generation_timestep,
            target_timestep=config.generation_target_timestep,
        )
    best_local_idx = remap.index_select(0, best_idx)
    worst_local_idx = remap.index_select(0, worst_idx)
    feature_local_idx = remap.index_select(0, feature_top_idx)
    raw_mean = raw_score_sum / max(total_candidates, 1)
    raw_var = max(raw_score_sum_sq / max(total_candidates, 1) - raw_mean * raw_mean, 0.0)
    reward_info: dict[str, torch.Tensor] = {
        "online_reward_geneval_raw_mean": all_scores.new_tensor(raw_mean),
        "online_reward_geneval_raw_std": all_scores.new_tensor(raw_var).sqrt(),
        "online_reward_geneval_positive_rate": all_scores.new_tensor(total_positive / max(total_candidates, 1)),
        "online_geneval_rollout_rounds": all_scores.new_tensor(float(rollout_rounds)),
        "online_geneval_total_candidates": all_scores.new_tensor(float(total_candidates)),
    }
    reward_info = add_rank_selection_stats(reward_info, all_scores, best_idx, worst_idx, feature_top_idx, prefix="online_reward")
    return (
        selected_generated.index_select(0, best_local_idx).detach(),
        selected_generated.index_select(0, worst_local_idx).detach(),
        selected_generated.index_select(0, feature_local_idx),
        selected_reference.index_select(0, feature_local_idx),
        all_scores,
        reward_info,
    )


def _evaluate_fixed_prompts(
    config: DrPOConfig,
    accelerator: Accelerator,
    unet,
    vae,
    text_encoder,
    tokenizer,
    eval_selectors,
    scheduler,
    prompts: list[str],
    global_step: int,
) -> dict[str, float] | None:
    if not accelerator.is_main_process or not eval_selectors:
        return None
    step_dir = Path(config.output_dir) / config.eval_output_subdir / f"step-{global_step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    unet_to_eval = accelerator.unwrap_model(unet)
    was_training = unet_to_eval.training
    unet_to_eval.eval()
    records = []
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts):
            prompt_ids = tokenizer(
                [prompt],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            hidden = text_encoder(prompt_ids)[0]
            generator = torch.Generator(device=accelerator.device).manual_seed(config.eval_seed + prompt_idx)
            noisy_latents = torch.randn(
                (1, 4, config.resolution // 8, config.resolution // 8),
                generator=generator,
                device=accelerator.device,
                dtype=hidden.dtype,
            )
            _, pred_latent = run_one_step_unet(
                unet_to_eval,
                noisy_latents,
                hidden,
                scheduler,
                timestep=config.generation_timestep,
                target_timestep=config.generation_target_timestep,
            )
            image = decode_latents_to_pil(vae, pred_latent)[0]
            file_name = f"{prompt_idx:03d}.png"
            image.save(step_dir / file_name)
            record = {"prompt": prompt, "file_name": file_name}
            ensemble = 0.0
            for model_name, weight in zip(config.choice_models, config.choice_model_weights):
                value = float(eval_selectors[model_name].score([image], prompt)[0])
                record[model_name] = value
                ensemble += float(weight) * value
            record["ensemble"] = ensemble
            records.append(record)
    if was_training:
        unet_to_eval.train()
    with (step_dir / "score_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary: dict[str, float] = {}
    for metric_key in (*config.choice_models, "ensemble"):
        values = [float(record[metric_key]) for record in records]
        summary[f"eval_avg_{metric_key}"] = sum(values) / len(values)
        summary[f"eval_best_{metric_key}"] = max(values)
        summary[f"eval_worst_{metric_key}"] = min(values)
    with (Path(config.output_dir) / "eval_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": global_step, **summary}, ensure_ascii=False) + "\n")
    return summary


def _evaluate_geneval_prompts(
    config: DrPOConfig,
    accelerator: Accelerator,
    unet,
    vae,
    text_encoder,
    tokenizer,
    eval_selectors,
    scheduler,
    eval_rows: list[dict[str, Any]],
    global_step: int,
) -> dict[str, float] | None:
    if not accelerator.is_main_process or not eval_selectors:
        return None

    step_dir = Path(config.output_dir) / config.eval_output_subdir / f"step-{global_step:04d}"
    images_dir = step_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    unet_to_eval = accelerator.unwrap_model(unet)
    was_training = unet_to_eval.training
    unet_to_eval.eval()

    records: list[dict[str, Any]] = []
    per_tag_total: dict[str, int] = {}
    per_tag_correct: dict[str, float] = {}

    with torch.no_grad():
        for prompt_idx, row in enumerate(eval_rows):
            prompt = str(row["prompt"])
            metadata = row.get("geneval_metadata") or row
            if not isinstance(metadata, dict):
                raise TypeError(f"Expected Geneval metadata dict, got {type(metadata)}")

            prompt_ids = tokenizer(
                [prompt],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            hidden = text_encoder(prompt_ids)[0]
            generator = torch.Generator(device=accelerator.device).manual_seed(config.eval_seed + prompt_idx)
            noisy_latents = torch.randn(
                (1, 4, config.resolution // 8, config.resolution // 8),
                generator=generator,
                device=accelerator.device,
                dtype=hidden.dtype,
            )
            _, pred_latent = run_one_step_unet(
                unet_to_eval,
                noisy_latents,
                hidden,
                scheduler,
                timestep=config.generation_timestep,
                target_timestep=config.generation_target_timestep,
            )
            image = decode_latents_to_pil(vae, pred_latent)[0]
            file_name = f"{prompt_idx:06d}.png"
            image.save(images_dir / file_name)

            score = float(eval_selectors["geneval"].score([image], metadata)[0])
            tag = str(metadata.get("tag", "unknown"))
            per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
            per_tag_correct[tag] = per_tag_correct.get(tag, 0.0) + score
            records.append(
                {
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "file_name": file_name,
                    "tag": tag,
                    "geneval_score": score,
                    "metadata": metadata,
                }
            )

    if was_training:
        unet_to_eval.train()

    with (step_dir / "score_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_scored = len(records)
    overall_accuracy = sum(record["geneval_score"] for record in records) / total_scored if total_scored else 0.0
    per_tag_accuracy = {
        tag: (per_tag_correct[tag] / per_tag_total[tag]) if per_tag_total[tag] else 0.0
        for tag in sorted(per_tag_total)
    }
    task_score = sum(per_tag_accuracy.values()) / len(per_tag_accuracy) if per_tag_accuracy else 0.0

    summary: dict[str, float] = {
        "eval_geneval_image_accuracy": overall_accuracy,
        "eval_geneval_task_score": task_score,
        "eval_geneval_total_images": float(total_scored),
    }
    for tag, accuracy in per_tag_accuracy.items():
        summary[f"eval_geneval_{tag}_accuracy"] = accuracy

    with (step_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with (Path(config.output_dir) / "eval_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": global_step, **summary}, ensure_ascii=False) + "\n")
    return summary


def train(config: DrPOConfig) -> None:
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
        log_with=config.report_to,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / config.logging_dir)),
    )
    log_path = _setup_logging(config, accelerator)
    if config.seed is not None:
        set_seed(config.seed + accelerator.process_index)
    if accelerator.is_main_process:
        _save_runtime_snapshot(config, accelerator)
        logger.info("Training log will be written to %s", log_path)

    tokenizer, text_encoder, vae, unet, scheduler = load_sdturbo_components(config.pretrained_model_name_or_path, config.revision)
    if config.generation_timestep >= scheduler.config.num_train_timesteps:
        raise ValueError(
            f"generation_timestep must be < num_train_timesteps ({scheduler.config.num_train_timesteps})."
        )
    if config.generation_target_timestep >= scheduler.config.num_train_timesteps:
        raise ValueError(
            "generation_target_timestep must be < scheduler num_train_timesteps or -1."
        )
    reference_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    if config.initial_unet_path:
        load_unet_checkpoint_weights(reference_unet, config.initial_unet_path)
        load_unet_checkpoint_weights(unet, config.initial_unet_path)
    unet = _maybe_add_lora(unet, config)
    _maybe_enable_gradient_checkpointing(unet)
    _maybe_enable_xformers(unet)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    reference_unet.eval()
    _maybe_enable_vae_memory_optimizations(vae)

    choice_models = resolve_choice_models(config.choice_models, config.choice_model)
    choice_model_weights = resolve_choice_model_weights(config.choice_model_weights, choice_models)
    config = DrPOConfig(**{**vars(config), "choice_models": choice_models, "choice_model_weights": choice_model_weights})

    if config.train_mode == "online":
        dataset = PromptDataset(config.prompt_file, tokenizer, max_samples=config.max_train_samples, seed=config.seed)
    else:
        dataset = PreferenceDataset(
            config.pairs_jsonl,
            tokenizer,
            config.train_mode,
            config.resolution,
            max_samples=config.max_train_samples,
            seed=config.seed,
            empty_prompt_probability=config.proportion_empty_prompts,
        )
    if config.train_mode == "online" and "geneval" in choice_models and not getattr(dataset, "has_geneval_metadata", False):
        raise ValueError("choice_model=geneval requires prompt_file to be a JSONL with prompt metadata rows.")
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_mode != "online",
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_preference_batch,
        drop_last=True,
    )

    extractor = _build_feature_extractor(config)
    feature_keys = _feature_keys(config, extractor)
    use_geneval_online = config.train_mode == "online" and choice_models == ("geneval",)
    selectors = None
    if config.train_mode == "online":
        selectors = build_choice_selectors(
            choice_models,
            accelerator.device,
            pickscore_model_path=config.pickscore_model_name_or_path,
            pickscore_processor_path=config.pickscore_processor_name_or_path or config.pickscore_model_name_or_path,
            geneval_repo=config.geneval_repo,
            geneval_detector_path=config.geneval_detector_path,
            geneval_model_config=config.geneval_model_config,
            geneval_options=config.geneval_options,
            local_files_only=not config.pickscore_allow_remote,
        )
    eval_prompts: list[str] = []
    eval_geneval_rows: list[dict[str, Any]] = []
    eval_selectors = None
    if config.eval_every_steps > 0:
        if use_geneval_online:
            eval_dataset = PromptDataset(
                config.eval_prompt_file or config.prompt_file,
                tokenizer,
                max_samples=config.num_eval_prompts,
                seed=None,
            )
            if not eval_dataset.has_geneval_metadata:
                raise ValueError("Geneval eval requires eval_prompt_file to be a JSONL with prompt metadata rows.")
            eval_geneval_rows = eval_dataset.rows
        else:
            eval_prompts = load_prompt_file(config.eval_prompt_file or config.prompt_file, limit=config.num_eval_prompts)
    if config.eval_every_steps > 0 and accelerator.is_main_process:
        eval_selectors = build_choice_selectors(
            choice_models,
            accelerator.device,
            pickscore_model_path=config.pickscore_model_name_or_path,
            pickscore_processor_path=config.pickscore_processor_name_or_path or config.pickscore_model_name_or_path,
            geneval_repo=config.geneval_repo,
            geneval_detector_path=config.geneval_detector_path,
            geneval_model_config=config.geneval_model_config,
            geneval_options=config.geneval_options,
            local_files_only=not config.pickscore_allow_remote,
        )

    optimizer = torch.optim.AdamW(
        _trainable_parameters(unet),
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
    unet, text_encoder, vae, reference_unet, extractor, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, vae, reference_unet, extractor, optimizer, dataloader, lr_scheduler
    )
    unet_in_channels = accelerator.unwrap_model(unet).config.in_channels
    trainable_unet_parameters = _trainable_parameters(unet)

    step = 0
    resume_checkpoint = _resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        logger.info("Resuming training state from %s", resume_checkpoint)
        accelerator.load_state(str(resume_checkpoint))
        step = _checkpoint_step(resume_checkpoint)
        logger.info("Resumed training state at global step %s.", step)

    if accelerator.is_main_process:
        accelerator.init_trackers("sdturbo-drpo", {key: _make_tracker_safe(value) for key, value in vars(config).items()})

    progress = tqdm(range(config.max_train_steps), initial=step, disable=not accelerator.is_local_main_process)
    threshold_ema = None
    while step < config.max_train_steps:
        for batch in dataloader:
            if step >= config.max_train_steps:
                break
            assert isinstance(batch, Batch)
            with accelerator.accumulate(unet):
                text_embeddings = encode_prompts(text_encoder, batch.input_ids.to(accelerator.device))
                sample_losses: list[torch.Tensor] = []
                sample_pref_losses: list[torch.Tensor] = []
                sample_ref_losses: list[torch.Tensor] = []
                sample_feat_l2_losses: list[torch.Tensor] = []
                sample_ref_l2_losses: list[torch.Tensor] = []
                sample_vggflow_unet_reg_losses: list[torch.Tensor] = []
                sample_dpos: list[torch.Tensor] = []
                sample_dnegs: list[torch.Tensor] = []
                sample_drefs: list[torch.Tensor] = []
                sample_v_var_values: list[torch.Tensor] = []
                sample_ref_v_var_values: list[torch.Tensor] = []
                sample_used_flags: list[torch.Tensor] = []
                sample_skipped_flags: list[torch.Tensor] = []
                online_stats: list[dict[str, torch.Tensor]] = []
                offline_stats: list[dict[str, torch.Tensor]] = []
                for prompt_index, prompt in enumerate(batch.prompts):
                    prompt_embedding = text_embeddings[prompt_index: prompt_index + 1].expand(config.batchsize_gen, -1, -1)
                    ref_l2_source_generated = None
                    ref_l2_source_reference = None
                    if config.train_mode == "online":
                        if use_geneval_online:
                            if batch.geneval_metadata is None or batch.geneval_metadata[prompt_index] is None:
                                raise ValueError("choice_model=geneval requires geneval_metadata for every online prompt.")
                            geneval_result = _collect_geneval_online_latents(
                                config=config,
                                accelerator=accelerator,
                                unet=unet,
                                reference_unet=reference_unet,
                                vae=vae,
                                scheduler=scheduler,
                                selectors=selectors,
                                prompt_embedding=prompt_embedding,
                                prompt_metadata=batch.geneval_metadata[prompt_index],
                                unet_in_channels=unet_in_channels,
                            )
                            if geneval_result is None:
                                sample_skipped_flags.append(torch.tensor(1.0, device=accelerator.device))
                                sample_used_flags.append(torch.tensor(0.0, device=accelerator.device))
                                continue
                            positive_latents, negative_latents, feature_generated_latents, feature_reference_latents, scores, reward_info = geneval_result
                            sample_v_var_values.append(feature_generated_latents.float().var(unbiased=False))
                            sample_ref_v_var_values.append(feature_reference_latents.float().var(unbiased=False))
                            online_stats.append(reward_info)
                            ref_l2_source_generated = feature_generated_latents
                            ref_l2_source_reference = feature_reference_latents
                        else:
                            noisy_latents = torch.randn(
                                (config.batchsize_gen, unet_in_channels, config.resolution // 8, config.resolution // 8),
                                device=accelerator.device,
                                dtype=prompt_embedding.dtype,
                            )
                            _, generated_latents = run_one_step_unet(
                                unet,
                                noisy_latents,
                                prompt_embedding,
                                scheduler,
                                timestep=config.generation_timestep,
                                target_timestep=config.generation_target_timestep,
                            )
                            with torch.no_grad():
                                _, reference_latents = run_one_step_unet(
                                    reference_unet,
                                    noisy_latents,
                                    prompt_embedding,
                                    scheduler,
                                    timestep=config.generation_timestep,
                                    target_timestep=config.generation_target_timestep,
                                )
                            sample_v_var_values.append(generated_latents.float().var(unbiased=False))
                            sample_ref_v_var_values.append(reference_latents.float().var(unbiased=False))
                            images = decode_latents_to_pil(vae, generated_latents.detach())
                            scores, reward_info = score_reward_ensemble(
                                selectors,
                                config.choice_model_weights,
                                images,
                                prompt,
                                normalize=config.choice_score_normalize,
                            )
                            scores = scores.to(device=generated_latents.device)
                            best_idx, worst_idx, feature_top_idx = select_disjoint_pref_indices(
                                scores,
                                num_pos=config.num_pos_images,
                                num_neg=config.num_neg_images,
                                feature_top_fraction=config.online_feature_top_fraction,
                            )
                            online_stats.append(add_rank_selection_stats(reward_info, scores, best_idx, worst_idx, feature_top_idx, prefix="online_reward"))
                            positive_latents = generated_latents.index_select(0, best_idx).detach()
                            negative_latents = generated_latents.index_select(0, worst_idx).detach()
                            feature_generated_latents = generated_latents.index_select(0, feature_top_idx)
                            feature_reference_latents = reference_latents.index_select(0, feature_top_idx)
                            ref_l2_source_generated = generated_latents
                            ref_l2_source_reference = reference_latents
                    elif config.train_mode == "offline_distance":
                        noisy_latents = torch.randn(
                            (config.batchsize_gen, unet_in_channels, config.resolution // 8, config.resolution // 8),
                            device=accelerator.device,
                            dtype=prompt_embedding.dtype,
                        )
                        _, generated_latents = run_one_step_unet(
                            unet,
                            noisy_latents,
                            prompt_embedding,
                            scheduler,
                            timestep=config.generation_timestep,
                            target_timestep=config.generation_target_timestep,
                        )
                        with torch.no_grad():
                            _, reference_latents = run_one_step_unet(
                                reference_unet,
                                noisy_latents,
                                prompt_embedding,
                                scheduler,
                                timestep=config.generation_timestep,
                                target_timestep=config.generation_target_timestep,
                            )
                        sample_v_var_values.append(generated_latents.float().var(unbiased=False))
                        sample_ref_v_var_values.append(reference_latents.float().var(unbiased=False))
                        chosen_images = batch.chosen[prompt_index].to(device=accelerator.device, dtype=generated_latents.dtype)
                        rejected_images = batch.rejected[prompt_index].to(device=accelerator.device, dtype=generated_latents.dtype)
                        positive_data_latents = encode_images(vae, chosen_images, mode=config.offline_latent_encode_mode)
                        negative_data_latents = encode_images(vae, rejected_images, mode=config.offline_latent_encode_mode)
                        with torch.no_grad():
                            cand_features = _vector_features_from_latents(extractor, vae, generated_latents.detach(), feature_keys, config)
                            pos_features = _vector_features_from_images_or_latents(extractor, vae, chosen_images, feature_keys, config)
                            neg_features = _vector_features_from_images_or_latents(extractor, vae, rejected_images, feature_keys, config)
                            scores, info = compute_offline_distance_scores(
                                cand_features,
                                pos_features,
                                neg_features,
                                feature_keys,
                                score_mode=config.offline_distance_score_mode,
                                normalize=config.offline_distance_score_normalize,
                                aggregation=config.offline_distance_score_aggregation,
                                ref_reduction=config.offline_distance_ref_reduction,
                            )
                            best_idx, worst_idx, feature_top_idx = select_disjoint_pref_indices(
                                scores,
                                num_pos=config.num_pos_images,
                                num_neg=config.num_neg_images,
                                feature_top_fraction=config.offline_distance_top_fraction,
                            )
                            info = add_rank_selection_stats(info, scores, best_idx, worst_idx, feature_top_idx, prefix="offline_distance")
                            offline_stats.append(info)
                        positive_latents = generated_latents.index_select(0, best_idx).detach()
                        negative_latents = generated_latents.index_select(0, worst_idx).detach()
                        if config.offline_distance_use_data_anchors:
                            positive_latents = torch.cat([positive_latents, positive_data_latents.detach()], dim=0)
                            negative_latents = torch.cat([negative_latents, negative_data_latents.detach()], dim=0)
                        feature_generated_latents = generated_latents.index_select(0, feature_top_idx)
                        feature_reference_latents = reference_latents.index_select(0, feature_top_idx)
                        ref_l2_source_generated = generated_latents
                        ref_l2_source_reference = reference_latents
                    else:
                        noisy_latents = torch.randn(
                            (config.batchsize_gen, unet_in_channels, config.resolution // 8, config.resolution // 8),
                            device=accelerator.device,
                            dtype=prompt_embedding.dtype,
                        )
                        _, generated_latents = run_one_step_unet(
                            unet,
                            noisy_latents,
                            prompt_embedding,
                            scheduler,
                            timestep=config.generation_timestep,
                            target_timestep=config.generation_target_timestep,
                        )
                        with torch.no_grad():
                            _, reference_latents = run_one_step_unet(
                                reference_unet,
                                noisy_latents,
                                prompt_embedding,
                                scheduler,
                                timestep=config.generation_timestep,
                                target_timestep=config.generation_target_timestep,
                            )
                        sample_v_var_values.append(generated_latents.float().var(unbiased=False))
                        sample_ref_v_var_values.append(reference_latents.float().var(unbiased=False))
                        positive_latents = encode_images(
                            vae,
                            batch.chosen[prompt_index].to(device=accelerator.device, dtype=generated_latents.dtype),
                            mode=config.offline_latent_encode_mode,
                        )
                        negative_latents = encode_images(
                            vae,
                            batch.rejected[prompt_index].to(device=accelerator.device, dtype=generated_latents.dtype),
                            mode=config.offline_latent_encode_mode,
                        )
                        feature_generated_latents = generated_latents
                        feature_reference_latents = reference_latents
                        ref_l2_source_generated = generated_latents
                        ref_l2_source_reference = reference_latents

                    generated_features = _vector_features_from_latents(extractor, vae, feature_generated_latents, feature_keys, config)
                    reference_features = _vector_features_from_latents(extractor, vae, feature_reference_latents, feature_keys, config)
                    with torch.no_grad():
                        positive_features = _vector_features_from_latents(extractor, vae, positive_latents, feature_keys, config)
                        negative_features = _vector_features_from_latents(extractor, vae, negative_latents, feature_keys, config)
                    prompt_terms = _compute_prompt_terms(
                        feature_keys,
                        generated_features,
                        reference_features,
                        positive_features,
                        negative_features,
                        config,
                    )
                    ref_l2 = F.mse_loss(ref_l2_source_generated.float(), ref_l2_source_reference.float())
                    vggflow_unet_reg = ref_l2
                    loss_i = prompt_terms["loss"]
                    if config.ref_model_l2_weight != 0.0:
                        loss_i = loss_i + config.ref_model_l2_weight * ref_l2
                    if config.vggflow_unet_reg_scale != 0.0:
                        loss_i = loss_i + config.vggflow_unet_reg_scale * vggflow_unet_reg
                    sample_used_flags.append(torch.tensor(1.0, device=accelerator.device))
                    sample_skipped_flags.append(torch.tensor(0.0, device=accelerator.device))
                    sample_losses.append(loss_i)
                    sample_pref_losses.append(prompt_terms["pref_loss"])
                    sample_ref_losses.append(prompt_terms["ref_loss"])
                    sample_feat_l2_losses.append(prompt_terms["feature_l2"])
                    sample_ref_l2_losses.append(ref_l2.detach())
                    sample_vggflow_unet_reg_losses.append(vggflow_unet_reg.detach())
                    sample_dpos.append(prompt_terms["d_pos"])
                    sample_dnegs.append(prompt_terms["d_neg"])
                    sample_drefs.append(prompt_terms["d_ref"])
                if not sample_losses:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss_values = torch.stack(sample_losses)
                keep_mask = torch.ones_like(loss_values, dtype=torch.bool)
                grad_filter_stats = None
                threshold_stats = None
                if config.per_sample_threshold_quantile > 0.0:
                    keep_mask, threshold_stats, threshold_ema = _build_per_sample_threshold_filter(
                        loss_values,
                        config.per_sample_threshold_quantile,
                        threshold_ema=threshold_ema,
                        ema_decay=config.per_sample_threshold_ema_decay,
                        use_sqrt=config.per_sample_threshold_use_sqrt,
                    )
                if config.sample_grad_norm_discard_quantile < 1.0:
                    grad_keep_mask, grad_filter_stats = _build_sample_grad_norm_filter(
                        loss_values,
                        trainable_unet_parameters,
                        config.sample_grad_norm_discard_quantile,
                    )
                    keep_mask = keep_mask & grad_keep_mask
                    if not bool(keep_mask.any()):
                        keep_mask = grad_keep_mask

                loss = _reduce_sample_values(loss_values, keep_mask, config.per_sample_threshold_reduction)
                pref_loss = _reduce_sample_values(torch.stack(sample_pref_losses), keep_mask, config.per_sample_threshold_reduction)
                ref_loss = _reduce_sample_values(torch.stack(sample_ref_losses), keep_mask, config.per_sample_threshold_reduction)
                feature_l2 = _reduce_sample_values(torch.stack(sample_feat_l2_losses), keep_mask, config.per_sample_threshold_reduction)
                ref_l2 = _reduce_sample_values(torch.stack(sample_ref_l2_losses), keep_mask, config.per_sample_threshold_reduction)
                vggflow_unet_reg = _reduce_sample_values(torch.stack(sample_vggflow_unet_reg_losses), keep_mask, config.per_sample_threshold_reduction)
                d_pos = _reduce_sample_values(torch.stack(sample_dpos), keep_mask, "kept_mean")
                d_neg = _reduce_sample_values(torch.stack(sample_dnegs), keep_mask, "kept_mean")
                d_ref = _reduce_sample_values(torch.stack(sample_drefs), keep_mask, "kept_mean")
                v_var = _reduce_sample_values(torch.stack(sample_v_var_values), keep_mask, "kept_mean")
                ref_v_var = _reduce_sample_values(torch.stack(sample_ref_v_var_values), keep_mask, "kept_mean")
                d_margin = d_neg - d_pos
                accelerator.backward(loss)
                grad_norm_value = torch.tensor(0.0, device=accelerator.device)
                if accelerator.sync_gradients:
                    grad_norm_value = accelerator.clip_grad_norm_(trainable_unet_parameters, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                progress.update(1)

                def gather_mean(value: torch.Tensor) -> float:
                    if not torch.is_tensor(value):
                        value = torch.as_tensor(value)
                    value = value.detach().to(device=accelerator.device).reshape(-1).contiguous()
                    return accelerator.gather(value.repeat(config.train_batch_size)).float().mean().item()

                log_payload = {
                    "train_loss": gather_mean(loss),
                    "pref_loss_unaccumulated": gather_mean(pref_loss),
                    "ref_loss_unaccumulated": gather_mean(ref_loss),
                    "feature_l2_unaccumulated": gather_mean(feature_l2),
                    "ref_l2_unaccumulated": gather_mean(ref_l2),
                    "vggflow_unet_reg_unaccumulated": gather_mean(vggflow_unet_reg),
                    "vggflow_unet_reg_weighted_unaccumulated": gather_mean(vggflow_unet_reg) * config.vggflow_unet_reg_scale,
                    "d_pos_unaccumulated": gather_mean(d_pos),
                    "d_neg_unaccumulated": gather_mean(d_neg),
                    "d_ref_unaccumulated": gather_mean(d_ref),
                    "d_margin_unaccumulated": gather_mean(d_margin),
                    "v_variance_unaccumulated": gather_mean(v_var),
                    "ref_v_variance_unaccumulated": gather_mean(ref_v_var),
                    "grad_norm_unaccumulated": gather_mean(grad_norm_value),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if sample_used_flags:
                    used_ratio = torch.stack(sample_used_flags).mean()
                    skipped_ratio = torch.stack(sample_skipped_flags).mean()
                    log_payload["online_prompt_used_ratio_unaccumulated"] = gather_mean(used_ratio)
                    log_payload["online_prompt_skipped_ratio_unaccumulated"] = gather_mean(skipped_ratio)
                if threshold_stats is not None:
                    for key, value in threshold_stats.items():
                        log_payload[f"{key}_unaccumulated"] = gather_mean(value)
                if grad_filter_stats is not None:
                    for key, value in grad_filter_stats.items():
                        log_payload[f"{key}_unaccumulated"] = gather_mean(value)
                if online_stats:
                    for key in online_stats[0]:
                        value = torch.stack([item[key] for item in online_stats]).mean()
                        log_payload[f"{key}_unaccumulated"] = gather_mean(value)
                if offline_stats:
                    for key in offline_stats[0]:
                        value = torch.stack([item[key] for item in offline_stats]).mean()
                        log_payload[f"{key}_unaccumulated"] = gather_mean(value)
                if config.log_task_grad_stats and step % config.task_grad_log_interval == 0:
                    task_stats = _compute_task_gradient_stats(pref_loss, ref_loss, trainable_unet_parameters, config.drifting_ref_loss_weight)
                    for key, value in task_stats.items():
                        log_payload[f"{key}_unaccumulated"] = gather_mean(value)
                if accelerator.is_main_process:
                    accelerator.log(log_payload, step=step)
                progress.set_postfix(
                    loss=f"{log_payload['train_loss']:.4f}",
                    d_margin=f"{log_payload['d_margin_unaccumulated']:.4f}",
                    v_var=f"{log_payload['v_variance_unaccumulated']:.4f}",
                )

                if config.eval_every_steps > 0 and step % config.eval_every_steps == 0:
                    if use_geneval_online and eval_geneval_rows:
                        eval_summary = _evaluate_geneval_prompts(
                            config,
                            accelerator,
                            unet,
                            vae,
                            text_encoder,
                            tokenizer,
                            eval_selectors,
                            scheduler,
                            eval_geneval_rows,
                            step,
                        )
                    elif eval_prompts:
                        eval_summary = _evaluate_fixed_prompts(
                            config,
                            accelerator,
                            unet,
                            vae,
                            text_encoder,
                            tokenizer,
                            eval_selectors,
                            scheduler,
                            eval_prompts,
                            step,
                        )
                    else:
                        eval_summary = None
                    if eval_summary and accelerator.is_main_process:
                        accelerator.log(eval_summary, step=step)
                if config.checkpointing_steps and step % config.checkpointing_steps == 0:
                    _save_training_checkpoint(accelerator, unet, config, output_dir / f"checkpoint-{step}", step, "periodic")

    accelerator.wait_for_everyone()
    _save_training_checkpoint(accelerator, unet, config, output_dir / "final", step, "final")
