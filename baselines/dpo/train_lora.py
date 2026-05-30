#!/usr/bin/env python
# coding=utf-8

"""
One-step diffusion DPO trainer for SD-Turbo.

This adapts diffusion-style DPO to SD-Turbo's one-step setting by replacing the
usual multi-step denoising trajectory likelihood with a pseudo log-probability
defined on the one-step x0 prediction under a fixed noisy latent and prompt.

Two preference spaces are supported:
1. Raw latent space (`--preference_space latent`)
2. Frozen feature-extractor space (`--preference_space mae_latent|dino_image`)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import socket
import sys
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import accelerate
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from baselines.common import (  # noqa: E402
    PairsPromptDataset,
    _build_fallback_pickscore_selector,
    collate_fn,
    compute_grad_norm,
    decode_latents_to_pil,
    enforce_zero_terminal_snr,
    evaluate_fixed_prompts,
    load_prompt_file,
    parse_int_list,
    parse_name_list,
    FrozenDinoV2BFeatureExtractor,
    FrozenMAELatentFeatureExtractor,
    one_step_clean_latent,
    resume_training_state,
    save_training_checkpoint,
)

check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


def _get_dino_feature_extractor_cls():
    return FrozenDinoV2BFeatureExtractor


def get_module_device_dtype(module) -> Tuple[torch.device, torch.dtype]:
    param = next(module.parameters())
    return param.device, param.dtype


def _make_json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)


def build_resolved_config(args) -> Dict[str, object]:
    return {key: _make_json_safe(value) for key, value in vars(args).items()}


def _make_tracker_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(_make_json_safe(value), ensure_ascii=False, sort_keys=True)


def build_tracker_config(args) -> Dict[str, object]:
    return {key: _make_tracker_safe(value) for key, value in vars(args).items()}


def setup_logging(args, accelerator: Accelerator) -> str:
    log_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=f"%(asctime)s | %(levelname)s | rank={accelerator.process_index}/{accelerator.num_processes} | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log_path


def save_runtime_snapshot(args, accelerator: Accelerator) -> Dict[str, object]:
    snapshot_dir = os.path.join(args.output_dir, "run_metadata")
    code_dir = os.path.join(snapshot_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    resolved_config = build_resolved_config(args)
    config_json_path = os.path.join(snapshot_dir, "resolved_config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=False, sort_keys=True)

    config_txt_path = os.path.join(snapshot_dir, "resolved_config.txt")
    with open(config_txt_path, "w", encoding="utf-8") as f:
        for key in sorted(resolved_config):
            f.write(f"{key}={resolved_config[key]}\n")

    command_path = os.path.join(snapshot_dir, "launch_command.sh")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(shlex.join([sys.executable, *sys.argv]) + "\n")

    runtime_info = {
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
        "accelerate_version": accelerate.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    runtime_info_path = os.path.join(snapshot_dir, "runtime_info.json")
    with open(runtime_info_path, "w", encoding="utf-8") as f:
        json.dump(runtime_info, f, indent=2, ensure_ascii=False, sort_keys=True)

    copied_files: List[str] = []
    snapshot_sources = [
        os.path.abspath(__file__),
        os.path.join(PROJECT_ROOT, "scripts", "train", "dpo.sh"),
        os.path.join(PROJECT_ROOT, "scripts", "inference", "sample_sdturbo_lora.sh"),
    ]
    for source_path in snapshot_sources:
        if not os.path.exists(source_path):
            continue
        relative_path = os.path.relpath(source_path, PROJECT_ROOT)
        target_path = os.path.join(code_dir, relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_files.append(relative_path)

    manifest_path = os.path.join(snapshot_dir, "code_snapshot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"files": copied_files}, f, indent=2, ensure_ascii=False, sort_keys=True)

    return {
        "snapshot_dir": snapshot_dir,
        "config_json_path": config_json_path,
        "config_txt_path": config_txt_path,
        "command_path": command_path,
        "runtime_info_path": runtime_info_path,
        "manifest_path": manifest_path,
        "copied_files": copied_files,
    }


def one_step_predict_x0(noisy_latents: torch.Tensor, model_pred: torch.Tensor) -> torch.Tensor:
    return one_step_clean_latent(noisy_latents, model_pred)


def decode_latents_to_tensor(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    vae_device, vae_dtype = get_module_device_dtype(vae)
    decoded = vae.decode((latents / vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)).sample
    return decoded.float().clamp(-1, 1)


def flatten_for_preference_distance(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    return x.reshape(x.shape[0], -1)


def pairwise_preference_score(gen: torch.Tensor, target: torch.Tensor, metric: str) -> torch.Tensor:
    gen_flat = flatten_for_preference_distance(gen).float()
    target_flat = flatten_for_preference_distance(target).float()

    if metric == "l2":
        dist = (gen_flat[:, None, :] - target_flat[None, :, :]).pow(2).mean(dim=-1)
        return -dist.mean()
    if metric == "l1":
        dist = (gen_flat[:, None, :] - target_flat[None, :, :]).abs().mean(dim=-1)
        return -dist.mean()
    if metric == "cosine":
        gen_norm = F.normalize(gen_flat, dim=-1)
        target_norm = F.normalize(target_flat, dim=-1)
        similarity = gen_norm @ target_norm.T
        return similarity.mean()
    raise ValueError(f"Unsupported DPO distance metric: {metric}")


def feature_dict_preference_score(
    gen_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    feature_keys: Sequence[str],
    metric: str,
    aggregation: str,
) -> torch.Tensor:
    terms = [pairwise_preference_score(gen_dict[key], target_dict[key], metric) for key in feature_keys]
    if not terms:
        raise ValueError("Expected at least one feature key for feature-space DPO.")
    values = torch.stack(terms)
    if aggregation == "sum":
        return values.sum()
    if aggregation == "mean":
        return values.mean()
    raise ValueError(f"Unsupported feature aggregation mode: {aggregation}")


def resolve_feature_keys(args, extractor) -> Tuple[str, ...]:
    if args.feature_keys:
        return parse_name_list(args.feature_keys)
    if args.feature_mode == "multi":
        return extractor.get_default_multifeature_keys()
    return (args.feature_key,)


def encode_images_to_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    *,
    encode_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    posterior = vae.encode(images.to(device=device, dtype=dtype)).latent_dist
    if encode_mode == "sample":
        latents = posterior.sample()
    elif encode_mode == "mode":
        latents = posterior.mode()
    else:
        raise ValueError(f"Unsupported anchor encode mode: {encode_mode}")
    return latents * vae.config.scaling_factor


def maybe_prepare_dino_cache(args, accelerator: Accelerator) -> bool:
    local_files_only = args.dino_local_files_only
    if args.preference_space != "dino_image":
        return local_files_only
    if local_files_only:
        return True
    if accelerator.is_main_process:
        dino_extractor_cls = _get_dino_feature_extractor_cls()
        _ = dino_extractor_cls(
            model_name_or_path=args.dino_model_name_or_path,
            processor_name_or_path=args.dino_processor_name_or_path,
            feature_key=args.feature_key,
            block_stride=args.feature_block_stride,
            patch_pool_sizes=args.feature_patch_sizes,
            local_files_only=False,
        )
    accelerator.wait_for_everyone()
    return True


def build_online_selector(args, accelerator: Accelerator):
    selector = None
    if args.train_mode != "online":
        return selector

    try:
        if args.choice_model in {"hps", "hpsv2"}:
            from utils.hps_utils import Selector
        elif args.choice_model == "clip":
            from utils.clip_utils import Selector
        elif args.choice_model == "pickscore":
            from utils.pickscore_utils import Selector
        else:
            from utils.aes_utils import Selector
        selector = Selector(accelerator.device)
    except ModuleNotFoundError:
        if args.choice_model != "pickscore":
            raise
        logger.warning("utils.pickscore_utils not found. Falling back to built-in PickScore selector.")
        local_files_only = not args.pickscore_allow_remote
        if args.pickscore_allow_remote and accelerator.is_main_process:
            _ = _build_fallback_pickscore_selector(
                accelerator.device,
                model_name_or_path=args.pickscore_model_name_or_path,
                processor_name_or_path=args.pickscore_processor_name_or_path,
                local_files_only=False,
            )
            accelerator.wait_for_everyone()
            local_files_only = True
        selector = _build_fallback_pickscore_selector(
            accelerator.device,
            model_name_or_path=args.pickscore_model_name_or_path,
            processor_name_or_path=args.pickscore_processor_name_or_path,
            local_files_only=local_files_only,
        )
    return selector


def compute_space_scores(
    *,
    args,
    vae: AutoencoderKL,
    extractor,
    feature_keys: Sequence[str],
    policy_latents: torch.Tensor,
    ref_latents: torch.Tensor,
    chosen_targets: torch.Tensor,
    rejected_targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.preference_space == "latent":
        policy_pos = pairwise_preference_score(policy_latents, chosen_targets, args.dpo_distance_metric)
        policy_neg = pairwise_preference_score(policy_latents, rejected_targets, args.dpo_distance_metric)
        with torch.no_grad():
            ref_pos = pairwise_preference_score(ref_latents, chosen_targets, args.dpo_distance_metric)
            ref_neg = pairwise_preference_score(ref_latents, rejected_targets, args.dpo_distance_metric)
        return policy_pos, policy_neg, ref_pos, ref_neg

    if args.preference_space == "mae_latent":
        policy_features = extractor.get_vector_features(policy_latents, feature_keys=feature_keys)
        with torch.no_grad():
            ref_features = extractor.get_vector_features(ref_latents, feature_keys=feature_keys)
            chosen_features = extractor.get_vector_features(chosen_targets, feature_keys=feature_keys)
            rejected_features = extractor.get_vector_features(rejected_targets, feature_keys=feature_keys)
        policy_pos = feature_dict_preference_score(
            policy_features,
            chosen_features,
            feature_keys,
            args.dpo_distance_metric,
            args.feature_aggregation,
        )
        policy_neg = feature_dict_preference_score(
            policy_features,
            rejected_features,
            feature_keys,
            args.dpo_distance_metric,
            args.feature_aggregation,
        )
        with torch.no_grad():
            ref_pos = feature_dict_preference_score(
                ref_features,
                chosen_features,
                feature_keys,
                args.dpo_distance_metric,
                args.feature_aggregation,
            )
            ref_neg = feature_dict_preference_score(
                ref_features,
                rejected_features,
                feature_keys,
                args.dpo_distance_metric,
                args.feature_aggregation,
            )
        return policy_pos, policy_neg, ref_pos, ref_neg

    if args.preference_space == "dino_image":
        policy_images = decode_latents_to_tensor(vae, policy_latents)
        policy_features = extractor.get_vector_features(policy_images, feature_keys=feature_keys)
        with torch.no_grad():
            ref_images = decode_latents_to_tensor(vae, ref_latents)
            ref_features = extractor.get_vector_features(ref_images, feature_keys=feature_keys)
            chosen_features = extractor.get_vector_features(chosen_targets, feature_keys=feature_keys)
            rejected_features = extractor.get_vector_features(rejected_targets, feature_keys=feature_keys)
        policy_pos = feature_dict_preference_score(
            policy_features,
            chosen_features,
            feature_keys,
            args.dpo_distance_metric,
            args.feature_aggregation,
        )
        policy_neg = feature_dict_preference_score(
            policy_features,
            rejected_features,
            feature_keys,
            args.dpo_distance_metric,
            args.feature_aggregation,
        )
        with torch.no_grad():
            ref_pos = feature_dict_preference_score(
                ref_features,
                chosen_features,
                feature_keys,
                args.dpo_distance_metric,
                args.feature_aggregation,
            )
            ref_neg = feature_dict_preference_score(
                ref_features,
                rejected_features,
                feature_keys,
                args.dpo_distance_metric,
                args.feature_aggregation,
            )
        return policy_pos, policy_neg, ref_pos, ref_neg

    raise ValueError(f"Unsupported preference space: {args.preference_space}")


def parse_args():
    parser = argparse.ArgumentParser("SD-Turbo one-step diffusion DPO trainer")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "sd-turbo"),
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/sdturbo-diffusion-dpo")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_mode", type=str, default="online", choices=["offline", "online"])
    parser.add_argument(
        "--pairs_jsonl",
        "--train_prompt_file",
        dest="pairs_jsonl",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "pickscore", "train.txt"),
        help="Prompt txt for online mode, or preference-pair JSONL for offline modes.",
    )

    parser.add_argument(
        "--choice_model",
        type=str,
        default="pickscore",
        choices=["aes", "clip", "hps", "hpsv2", "pickscore"],
        help="Online mode only: selector used to build chosen/rejected pseudo-pairs.",
    )
    parser.add_argument(
        "--pickscore_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "PickScore_v1"),
    )
    parser.add_argument("--pickscore_processor_name_or_path", type=str, default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--batchsize_gen", type=int, default=8)
    parser.add_argument(
        "--generation_timestep",
        type=int,
        default=999,
        help="Timestep fed to the one-step UNet during training/eval generation.",
    )
    parser.add_argument(
        "--generation_target_timestep",
        type=int,
        default=0,
        help="Project the predicted clean sample back to this timestep before VAE decode. Use -1 to decode x0 directly.",
    )
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument(
        "--preference_space",
        type=str,
        default="latent",
        choices=["latent", "mae_latent", "dino_image"],
        help="DPO pseudo log-prob space: raw x0 latent or frozen feature space.",
    )
    parser.add_argument("--dpo_beta", type=float, default=1.0)
    parser.add_argument("--dpo_loss_weight", type=float, default=1.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.0)
    parser.add_argument("--reference_free_dpo", action="store_true")
    parser.add_argument("--dpo_distance_metric", type=str, default="l2", choices=["l2", "l1", "cosine"])
    parser.add_argument("--anchor_encode_mode", type=str, default="mode", choices=["mode", "sample"])

    parser.add_argument("--feature_mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--feature_key", type=str, default="layer4_mean")
    parser.add_argument("--feature_keys", type=str, default="")
    parser.add_argument("--feature_block_stride", type=int, default=2)
    parser.add_argument("--feature_patch_sizes", type=str, default="2,4")
    parser.add_argument("--feature_aggregation", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--mae_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "mae_latent_256_torch.pth"))
    parser.add_argument("--mae_include_input_sq_mean", action="store_true")
    parser.add_argument("--mae_include_spatial_features", action="store_true")

    parser.add_argument(
        "--dino_model_name_or_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "dinov2-base"),
    )
    parser.add_argument("--dino_processor_name_or_path", type=str, default=None)
    parser.add_argument("--dino_local_files_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_pos_images", type=int, default=1)
    parser.add_argument("--num_neg_images", type=int, default=1)
    parser.add_argument(
        "--eval_prompt_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "pickscore", "test.txt"),
    )
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--eval_output_subdir", type=str, default="eval_samples")
    return parser.parse_args()


def main():
    args = parse_args()
    args.feature_patch_sizes = parse_int_list(args.feature_patch_sizes)

    if args.pickscore_processor_name_or_path is None:
        args.pickscore_processor_name_or_path = args.pickscore_model_name_or_path
    if args.dino_processor_name_or_path is None:
        args.dino_processor_name_or_path = args.dino_model_name_or_path
    if args.preference_space == "dino_image" and not args.feature_keys and args.feature_key == "layer4_mean":
        args.feature_key = "layer12_patch_mean"
    if args.preference_space != "dino_image" and args.feature_key == "layer12_patch_mean":
        args.feature_key = "layer4_mean"
    if args.num_pos_images < 1 or args.num_neg_images < 1:
        raise ValueError("--num_pos_images and --num_neg_images must be >= 1")
    if args.num_eval_prompts < 1:
        raise ValueError("--num_eval_prompts must be >= 1")
    if args.eval_every_steps < 0:
        raise ValueError("--eval_every_steps must be >= 0")
    if args.generation_timestep < 0:
        raise ValueError("--generation_timestep must be >= 0.")
    if args.generation_target_timestep >= 0 and args.generation_target_timestep > args.generation_timestep:
        raise ValueError("--generation_target_timestep must be <= --generation_timestep (or -1).")

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_logging(args, accelerator)

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    accelerator.wait_for_everyone()

    runtime_snapshot = None
    if accelerator.is_main_process:
        runtime_snapshot = save_runtime_snapshot(args, accelerator)
        logger.info(f"Training log will be written to {log_path}")
        logger.info(f"Resolved config saved to {runtime_snapshot['config_json_path']}")
        logger.info(f"Runtime snapshot saved to {runtime_snapshot['snapshot_dir']}")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if "turbo" in args.pretrained_model_name_or_path.lower():
        enforce_zero_terminal_snr(noise_scheduler)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    ref_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)

    if is_xformers_available():
        import xformers

        if version.parse(xformers.__version__) >= version.parse("0.0.17"):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention.")
            except Exception as exc:
                logger.warning(f"Failed to enable xformers memory efficient attention: {exc}")
        else:
            logger.warning(f"xformers version {xformers.__version__} is too old; continuing without it.")
    else:
        logger.warning("xformers is not available; continuing without memory efficient attention.")

    if args.use_lora:
        unet.requires_grad_(False)
        lora_target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        if not lora_target_modules:
            raise ValueError("Expected at least one LoRA target module.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=lora_target_modules,
        )
        unet = get_peft_model(unet, lora_config)
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet.parameters())
            logger.info(
                "Enabled LoRA on UNet: trainable_params=%s, total_params=%s, ratio=%.6f",
                trainable_params,
                total_params,
                trainable_params / total_params,
            )

    dino_local_files_only = maybe_prepare_dino_cache(args, accelerator)

    feature_extractor = None
    feature_keys: Tuple[str, ...] = ()
    if args.preference_space == "mae_latent":
        feature_extractor = FrozenMAELatentFeatureExtractor(
            args.mae_path,
            args.feature_key,
            block_stride=args.feature_block_stride,
            patch_pool_sizes=args.feature_patch_sizes,
            include_input_sq_mean=args.mae_include_input_sq_mean,
            include_spatial_features=args.mae_include_spatial_features,
        ).to(accelerator.device)
        feature_extractor.eval()
        feature_keys = resolve_feature_keys(args, feature_extractor)
        logger.info("Using MAE latent DPO with %d feature key(s): %s", len(feature_keys), ", ".join(feature_keys))
    elif args.preference_space == "dino_image":
        dino_extractor_cls = _get_dino_feature_extractor_cls()
        feature_extractor = dino_extractor_cls(
            model_name_or_path=args.dino_model_name_or_path,
            processor_name_or_path=args.dino_processor_name_or_path,
            feature_key=args.feature_key,
            block_stride=args.feature_block_stride,
            patch_pool_sizes=args.feature_patch_sizes,
            local_files_only=dino_local_files_only,
        ).to(accelerator.device)
        feature_extractor.eval()
        feature_keys = resolve_feature_keys(args, feature_extractor)
        logger.info("Using DINO image DPO with %d feature key(s): %s", len(feature_keys), ", ".join(feature_keys))
    else:
        logger.info("Using latent-space DPO without extra feature extractor.")

    trainable_unet_parameters = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_unet_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = PairsPromptDataset(
        pairs_jsonl=args.pairs_jsonl,
        tokenizer=tokenizer,
        train_mode=args.train_mode,
        image_size=args.resolution,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)
    if feature_extractor is not None:
        if args.preference_space == "dino_image":
            feature_extractor.to(accelerator.device)
        else:
            feature_extractor.to(accelerator.device, dtype=weight_dtype)

    selector = build_online_selector(args, accelerator)

    eval_prompts = load_prompt_file(args.eval_prompt_file, max_prompts=args.num_eval_prompts)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "eval_prompts.txt"), "w", encoding="utf-8") as f:
            for prompt in eval_prompts:
                f.write(prompt + "\n")

    eval_selector = None
    eval_args = None
    if accelerator.is_main_process:
        eval_selector = _build_fallback_pickscore_selector(
            accelerator.device,
            model_name_or_path=args.pickscore_model_name_or_path,
            processor_name_or_path=args.pickscore_processor_name_or_path,
            local_files_only=not args.pickscore_allow_remote,
        )
        eval_args = argparse.Namespace(**vars(args))
        eval_args.choice_model = "pickscore"

    global_step = resume_training_state(accelerator, args.output_dir, args.resume_from_checkpoint, logger=logger)

    if accelerator.is_main_process:
        accelerator.init_trackers("sdturbo-diffusion-dpo", build_tracker_config(args))

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )

    if global_step == 0:
        eval_summary = evaluate_fixed_prompts(
            args=eval_args,
            accelerator=accelerator,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            selector=eval_selector,
            generation_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            prompts=eval_prompts,
            global_step=0,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and eval_summary is not None:
            accelerator.log(
                {
                    "eval_avg_pickscore": eval_summary["avg_pickscore"],
                    "eval_best_pickscore": eval_summary["best_pickscore"],
                    "eval_worst_pickscore": eval_summary["worst_pickscore"],
                },
                step=0,
            )
            logger.info(
                "Eval step %s: avg=%.6f best=%.6f worst=%.6f",
                0,
                eval_summary["avg_pickscore"],
                eval_summary["best_pickscore"],
                eval_summary["worst_pickscore"],
            )
    elif accelerator.is_main_process:
        logger.info("Skipping step-0 eval after resume at global step %s.", global_step)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                prompt_batch = batch["prompt"]
                input_ids = batch["input_ids"]
                bsz = input_ids.shape[0]

                total_loss = 0.0
                total_dpo_loss = 0.0
                total_policy_pos = 0.0
                total_policy_neg = 0.0
                total_ref_pos = 0.0
                total_ref_neg = 0.0
                total_dpo_logit = 0.0
                total_policy_gap = 0.0
                total_ref_gap = 0.0
                total_ref_l2 = 0.0
                total_rank_pos = 0.0
                total_rank_neg = 0.0
                rank_count = 0

                for i in range(bsz):
                    prompt = prompt_batch[i]
                    prompt_ids = input_ids[i : i + 1]
                    encoder_hidden_states = text_encoder(prompt_ids)[0].repeat_interleave(args.batchsize_gen, dim=0)
                    noisy_latents = torch.randn(
                        (args.batchsize_gen, 4, args.resolution // 8, args.resolution // 8),
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    timesteps = torch.ones((args.batchsize_gen,), device=accelerator.device, dtype=torch.long) * 999

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    policy_latents = one_step_predict_x0(noisy_latents, model_pred)

                    with torch.no_grad():
                        ref_pred = ref_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        ref_latents = one_step_predict_x0(noisy_latents, ref_pred)

                    if args.train_mode == "offline":
                        chosen_images = batch["chosen"][i].to(device=accelerator.device, dtype=weight_dtype)
                        rejected_images = batch["rejected"][i].to(device=accelerator.device, dtype=weight_dtype)
                        if args.preference_space == "dino_image":
                            chosen_targets = chosen_images
                            rejected_targets = rejected_images
                        else:
                            with torch.no_grad():
                                chosen_targets = encode_images_to_latents(
                                    vae,
                                    chosen_images,
                                    encode_mode=args.anchor_encode_mode,
                                    device=accelerator.device,
                                    dtype=weight_dtype,
                                )
                                rejected_targets = encode_images_to_latents(
                                    vae,
                                    rejected_images,
                                    encode_mode=args.anchor_encode_mode,
                                    device=accelerator.device,
                                    dtype=weight_dtype,
                                )
                    else:
                        cand_images = decode_latents_to_pil(vae, policy_latents)
                        scores = selector.score(cand_images, prompt)
                        scores_t = torch.tensor(scores, device=accelerator.device, dtype=torch.float32)
                        pos_count = min(args.num_pos_images, scores_t.shape[0])
                        neg_count = min(args.num_neg_images, scores_t.shape[0])
                        best_idx = torch.topk(scores_t, k=pos_count, largest=True).indices
                        worst_idx = torch.topk(scores_t, k=neg_count, largest=False).indices
                        total_rank_pos = total_rank_pos + scores_t[best_idx].mean()
                        total_rank_neg = total_rank_neg + scores_t[worst_idx].mean()
                        rank_count += 1

                        if args.preference_space == "dino_image":
                            decoded_policy_images = decode_latents_to_tensor(vae, policy_latents)
                            chosen_targets = decoded_policy_images[best_idx].detach()
                            rejected_targets = decoded_policy_images[worst_idx].detach()
                        else:
                            chosen_targets = policy_latents[best_idx].detach()
                            rejected_targets = policy_latents[worst_idx].detach()

                    policy_pos, policy_neg, ref_pos, ref_neg = compute_space_scores(
                        args=args,
                        vae=vae,
                        extractor=feature_extractor,
                        feature_keys=feature_keys,
                        policy_latents=policy_latents,
                        ref_latents=ref_latents,
                        chosen_targets=chosen_targets,
                        rejected_targets=rejected_targets,
                    )

                    policy_gap = policy_pos - policy_neg
                    ref_gap = ref_pos - ref_neg
                    dpo_logit = args.dpo_beta * (policy_gap if args.reference_free_dpo else (policy_gap - ref_gap))
                    dpo_loss_i = -F.logsigmoid(dpo_logit)
                    ref_l2_i = F.mse_loss(policy_latents.float(), ref_latents.float())
                    loss_i = args.dpo_loss_weight * dpo_loss_i + args.ref_model_l2_weight * ref_l2_i

                    total_loss = total_loss + loss_i
                    total_dpo_loss = total_dpo_loss + dpo_loss_i
                    total_policy_pos = total_policy_pos + policy_pos.detach()
                    total_policy_neg = total_policy_neg + policy_neg.detach()
                    total_ref_pos = total_ref_pos + ref_pos.detach()
                    total_ref_neg = total_ref_neg + ref_neg.detach()
                    total_dpo_logit = total_dpo_logit + dpo_logit.detach()
                    total_policy_gap = total_policy_gap + policy_gap.detach()
                    total_ref_gap = total_ref_gap + ref_gap.detach()
                    total_ref_l2 = total_ref_l2 + ref_l2_i.detach()

                loss = total_loss / bsz
                dpo_loss = total_dpo_loss / bsz
                policy_pos = total_policy_pos / bsz
                policy_neg = total_policy_neg / bsz
                ref_pos = total_ref_pos / bsz
                ref_neg = total_ref_neg / bsz
                dpo_logit = total_dpo_logit / bsz
                policy_gap = total_policy_gap / bsz
                ref_gap = total_ref_gap / bsz
                ref_l2 = total_ref_l2 / bsz
                if rank_count > 0:
                    rank_pos = total_rank_pos / rank_count
                    rank_neg = total_rank_neg / rank_count

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm_value = compute_grad_norm(unet.parameters()).to(accelerator.device)
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_dpo_loss = accelerator.gather(dpo_loss.detach().repeat(args.train_batch_size)).mean().item()
                avg_policy_pos = accelerator.gather(policy_pos.repeat(args.train_batch_size)).mean().item()
                avg_policy_neg = accelerator.gather(policy_neg.repeat(args.train_batch_size)).mean().item()
                avg_ref_pos = accelerator.gather(ref_pos.repeat(args.train_batch_size)).mean().item()
                avg_ref_neg = accelerator.gather(ref_neg.repeat(args.train_batch_size)).mean().item()
                avg_dpo_logit = accelerator.gather(dpo_logit.repeat(args.train_batch_size)).mean().item()
                avg_policy_gap = accelerator.gather(policy_gap.repeat(args.train_batch_size)).mean().item()
                avg_ref_gap = accelerator.gather(ref_gap.repeat(args.train_batch_size)).mean().item()
                avg_ref_l2 = accelerator.gather(ref_l2.repeat(args.train_batch_size)).mean().item()
                avg_grad = accelerator.gather(grad_norm_value.repeat(args.train_batch_size)).mean().item()

                log_payload: Dict[str, float] = {
                    "train_loss": avg_loss,
                    "dpo_loss_unaccumulated": avg_dpo_loss,
                    "policy_pos_score_unaccumulated": avg_policy_pos,
                    "policy_neg_score_unaccumulated": avg_policy_neg,
                    "ref_pos_score_unaccumulated": avg_ref_pos,
                    "ref_neg_score_unaccumulated": avg_ref_neg,
                    "dpo_logit_unaccumulated": avg_dpo_logit,
                    "policy_gap_unaccumulated": avg_policy_gap,
                    "ref_gap_unaccumulated": avg_ref_gap,
                    "ref_l2_unaccumulated": avg_ref_l2,
                    "grad_norm_unaccumulated": avg_grad,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if rank_count > 0:
                    avg_rank_pos = accelerator.gather(rank_pos.detach().repeat(args.train_batch_size)).mean().item()
                    avg_rank_neg = accelerator.gather(rank_neg.detach().repeat(args.train_batch_size)).mean().item()
                    log_payload["online_chosen_score_unaccumulated"] = avg_rank_pos
                    log_payload["online_rejected_score_unaccumulated"] = avg_rank_neg

                accelerator.log(log_payload, step=global_step)
                progress_postfix = {
                    "loss": avg_loss,
                    "dpo": avg_dpo_loss,
                    "logit": avg_dpo_logit,
                    "pgap": avg_policy_gap,
                    "rgap": avg_ref_gap,
                    "ref_l2": avg_ref_l2,
                    "grad": avg_grad,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if rank_count > 0:
                    progress_postfix["ch"] = avg_rank_pos
                    progress_postfix["rej"] = avg_rank_neg
                progress_bar.set_postfix(**progress_postfix)
                progress_bar.update(1)

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    eval_summary = evaluate_fixed_prompts(
                        args=eval_args,
                        accelerator=accelerator,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        selector=eval_selector,
                        generation_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        prompts=eval_prompts,
                        global_step=global_step,
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and eval_summary is not None:
                        accelerator.log(
                            {
                                "eval_avg_pickscore": eval_summary["avg_pickscore"],
                                "eval_best_pickscore": eval_summary["best_pickscore"],
                                "eval_worst_pickscore": eval_summary["worst_pickscore"],
                            },
                            step=global_step,
                        )
                        logger.info(
                            "Eval step %s: avg=%.6f best=%.6f worst=%.6f",
                            global_step,
                            eval_summary["avg_pickscore"],
                            eval_summary["best_pickscore"],
                            eval_summary["worst_pickscore"],
                        )

                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    save_training_checkpoint(
                        accelerator,
                        unet,
                        args,
                        os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                        global_step,
                        "periodic",
                        logger=logger,
                    )

            if global_step >= args.max_train_steps:
                break

    if global_step > 0:
        save_training_checkpoint(
            accelerator,
            unet,
            args,
            os.path.join(args.output_dir, "final"),
            global_step,
            "final",
            logger=logger,
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
