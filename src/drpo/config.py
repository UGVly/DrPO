from __future__ import annotations

import argparse
from dataclasses import dataclass

from drpo.paths import project_root

DEFAULT_MAE_FEATURE_KEYS = "layer4_mean,layer4_std,layer4_mean_2,layer4_std_2,layer4_mean_4,layer4_std_4"
DEFAULT_DINO_FEATURE_KEYS = (
    "layer3_patch_mean,layer3_patch_std,"
    "layer3_cls,"
    "layer6_patch_mean,layer6_patch_std,"
    "layer6_cls,"
    "layer9_patch_mean,layer9_patch_std,"
    "layer9_cls,"
    "layer12_patch_mean,layer12_patch_std,"
    "layer12_cls"
)
DEFAULT_LATENT_FEATURE_KEYS = "latent,latent_mean,latent_std,latent_mean_2,latent_std_2,latent_mean_4,latent_std_4"


def parse_csv_floats(value: str) -> tuple[float, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float.")
    return tuple(float(item) for item in items)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    return tuple(int(item) for item in items)


def parse_csv_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_feature_keys(value: str, extractor: str) -> tuple[str, ...]:
    if value == DEFAULT_MAE_FEATURE_KEYS:
        if extractor == "dino":
            value = DEFAULT_DINO_FEATURE_KEYS
        elif extractor == "latent":
            value = DEFAULT_LATENT_FEATURE_KEYS
    return parse_csv_names(value)


def parse_feature_key(value: str, extractor: str) -> str:
    if extractor == "dino" and value == "layer4_mean":
        return "layer12_patch_mean"
    if extractor == "latent" and value == "layer4_mean":
        return "latent"
    return value


@dataclass(frozen=True)
class DrPOConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    train_mode: str
    pairs_jsonl: str
    prompt_file: str
    drifting_mae_path: str
    drifting_feature_extractor: str = "mae"
    drifting_dino_model_name_or_path: str | None = None
    drifting_dino_processor_name_or_path: str | None = None
    drifting_dino_allow_remote: bool = False
    revision: str | None = None
    initial_unet_path: str | None = None
    choice_model: str = "pickscore"
    choice_models: tuple[str, ...] = ()
    choice_model_weights: tuple[float, ...] = ()
    choice_score_normalize: str = "zscore"
    pickscore_model_name_or_path: str | None = None
    pickscore_processor_name_or_path: str | None = None
    pickscore_allow_remote: bool = False
    geneval_repo: str | None = None
    geneval_detector_path: str | None = None
    geneval_model_config: str | None = None
    geneval_options: str = ""
    geneval_max_rollout_rounds: int = 4
    seed: int = 42
    resolution: int = 512
    train_batch_size: int = 1
    batchsize_gen: int = 24
    generation_timestep: int = 999
    generation_target_timestep: int = 0
    max_train_steps: int = 1000
    gradient_accumulation_steps: int = 32
    dataloader_num_workers: int = 2
    max_train_samples: int | None = None
    proportion_empty_prompts: float = 0.0
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    checkpointing_steps: int = 100
    resume_from_checkpoint: str | None = None
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    drifting_feature_mode: str = "multi"
    drifting_feature_key: str = "layer4_mean"
    drifting_feature_keys: tuple[str, ...] = ("layer4_mean", "layer4_std", "layer4_mean_2", "layer4_std_2", "layer4_mean_4", "layer4_std_4")
    drifting_feature_block_stride: int = 2
    drifting_feature_patch_sizes: tuple[int, ...] = (2, 4)
    drifting_include_input_sq_mean: bool = False
    drifting_include_spatial_features: bool = False
    drifting_feature_aggregation: str = "sum"
    drifting_kernel: str = "laplacian"
    drifting_pref_r_list: tuple[float, ...] = (0.02, 0.05, 0.2)
    drifting_ref_r_list: tuple[float, ...] = (0.02, 0.05, 0.2)
    num_pos_images_min: int | None = None
    num_pos_images: int = 8
    num_neg_images: int = 8
    online_feature_top_count: int = 0
    online_feature_top_fraction: float = 1.0
    offline_distance_top_fraction: float = 0.5
    offline_distance_score_mode: str = "margin"
    offline_distance_score_normalize: str = "zscore"
    offline_distance_score_aggregation: str = "mean"
    offline_distance_ref_reduction: str = "mean"
    offline_distance_use_data_anchors: bool = False
    offline_latent_encode_mode: str = "mode"
    drifting_pos_weight: float = 3000.0
    drifting_neg_weight: float = 3000.0
    drifting_ref_weight: float = 3000.0
    drifting_ref_neg_weight: float = 3000.0
    drifting_ref_loss_weight: float = 0.2
    ref_model_l2_weight: float = 0.2
    vggflow_unet_reg_scale: float = 0.0
    frozen_feature_l2_weight: float = 0.0
    eval_prompt_file: str | None = None
    num_eval_prompts: int = 10
    eval_every_steps: int = 0
    eval_output_subdir: str = "eval_samples"
    eval_seed: int = 1234
    sample_grad_norm_discard_quantile: float = 1.0
    per_sample_threshold_quantile: float = -1.0
    per_sample_threshold_ema_decay: float = 0.99
    per_sample_threshold_use_sqrt: bool = True
    per_sample_threshold_reduction: str = "masked_mean"
    log_task_grad_stats: bool = False
    task_grad_log_interval: int = 50


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train SD-Turbo with DrPO.")
    parser.add_argument("--pretrained_model_name_or_path", default=str(root / "models" / "sd-turbo"))
    parser.add_argument("--initial_unet_path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", default=str(root / "outputs" / "drpo-sdturbo"))
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--train_mode", choices=["offline", "online", "offline_distance"], default="online")
    parser.add_argument("--pairs_jsonl", default=str(root / "data" / "pairs.jsonl"))
    parser.add_argument("--prompt_file", default=str(root / "data" / "prompts" / "pickapicv2_test_unique.txt"))
    parser.add_argument("--drifting_mae_path", default=str(root / "drifting" / "mae_latent_256_torch.pth"))
    parser.add_argument("--drifting_feature_extractor", choices=["mae", "dino", "latent"], default="mae")
    parser.add_argument("--drifting_dino_model_name_or_path", default=str(root / "models" / "dinov2-base"))
    parser.add_argument("--drifting_dino_processor_name_or_path", default=None)
    parser.add_argument("--drifting_dino_allow_remote", action="store_true")
    parser.add_argument(
        "--choice_model",
        choices=["pickscore", "clip", "aes", "hps", "hpsv2", "imagereward", "geneval"],
        default="pickscore",
    )
    parser.add_argument("--choice_models", default="")
    parser.add_argument("--choice_model_weights", default="")
    parser.add_argument("--choice_score_normalize", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--pickscore_model_name_or_path", default=str(root / "models" / "PickScore_v1"))
    parser.add_argument("--pickscore_processor_name_or_path", default=None)
    parser.add_argument("--pickscore_allow_remote", action="store_true")
    parser.add_argument("--geneval_repo", default=str(root / "third_party" / "geneval"))
    parser.add_argument("--geneval_detector_path", default=str(root / "models" / "geneval_detector"))
    parser.add_argument("--geneval_model_config", default=None)
    parser.add_argument("--geneval_options", default="")
    parser.add_argument("--geneval_max_rollout_rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--batchsize_gen", type=int, default=24)
    parser.add_argument("--generation_timestep", type=int, default=999)
    parser.add_argument("--generation_target_timestep", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--use_lora", dest="use_lora", action="store_true", default=True)
    parser.add_argument("--no_use_lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--drifting_feature_mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--drifting_feature_key", default="layer4_mean")
    parser.add_argument("--drifting_feature_keys", default=DEFAULT_MAE_FEATURE_KEYS)
    parser.add_argument("--drifting_feature_block_stride", type=int, default=2)
    parser.add_argument("--drifting_feature_patch_sizes", default="2,4")
    parser.add_argument("--drifting_include_input_sq_mean", action="store_true")
    parser.add_argument("--drifting_include_spatial_features", action="store_true")
    parser.add_argument("--drifting_feature_aggregation", choices=["sum", "mean"], default="sum")
    parser.add_argument("--drifting_kernel", choices=["laplacian", "exponential", "rbf", "cosine"], default="laplacian")
    parser.add_argument("--drifting_pref_r_list", default="0.02,0.05,0.2")
    parser.add_argument("--drifting_ref_r_list", default="0.02,0.05,0.2")
    parser.add_argument("--drifting_pos_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_neg_weight", type=float, default=3000.0)
    parser.add_argument("--drifting_ref_loss_weight", type=float, default=0.2)
    parser.add_argument("--frozen_feature_l2_weight", type=float, default=0.0)
    parser.add_argument("--ref_model_l2_weight", type=float, default=0.2)
    parser.add_argument(
        "--vggflow_unet_reg_scale",
        type=float,
        default=0.0,
        help="VGG-Flow-style clean-latent MSE regularization between policy x0 and frozen reference x0.",
    )
    parser.add_argument("--num_pos_images_min", type=int, default=None)
    parser.add_argument("--num_pos_images", type=int, default=8)
    parser.add_argument("--num_neg_images", type=int, default=8)
    parser.add_argument("--online_feature_top_count", type=int, default=0)
    parser.add_argument("--online_feature_top_fraction", type=float, default=1.0)
    parser.add_argument("--offline_distance_top_fraction", type=float, default=0.5)
    parser.add_argument(
        "--offline_distance_score_mode",
        choices=["margin", "separate_zscore", "cosine_softmax_diff"],
        default="margin",
    )
    parser.add_argument("--offline_distance_score_normalize", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--offline_distance_score_aggregation", choices=["mean", "sum"], default="mean")
    parser.add_argument("--offline_distance_ref_reduction", choices=["mean", "min"], default="mean")
    parser.add_argument("--offline_distance_use_data_anchors", action="store_true")
    parser.add_argument("--offline_latent_encode_mode", choices=["sample", "mode"], default="mode")
    parser.add_argument("--eval_prompt_file", default=str(root / "data" / "prompts" / "pickapicv2_test_unique.txt"))
    parser.add_argument("--num_eval_prompts", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_output_subdir", default="eval_samples")
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--sample_grad_norm_discard_quantile", type=float, default=1.0)
    parser.add_argument("--per_sample_threshold_quantile", type=float, default=-1.0)
    parser.add_argument("--per_sample_threshold_ema_decay", type=float, default=0.99)
    parser.add_argument("--per_sample_threshold_use_sqrt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per_sample_threshold_reduction", choices=["masked_mean", "kept_mean"], default="masked_mean")
    parser.add_argument("--log_task_grad_stats", action="store_true")
    parser.add_argument("--task_grad_log_interval", type=int, default=50)
    return parser


def parse_config(argv: list[str] | None = None, *, force_use_lora: bool | None = None) -> DrPOConfig:
    args = build_parser().parse_args(argv)
    use_lora = args.use_lora if force_use_lora is None else force_use_lora
    return DrPOConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        initial_unet_path=args.initial_unet_path,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        train_mode=args.train_mode,
        pairs_jsonl=args.pairs_jsonl,
        prompt_file=args.prompt_file,
        drifting_mae_path=args.drifting_mae_path,
        drifting_feature_extractor=args.drifting_feature_extractor,
        drifting_dino_model_name_or_path=args.drifting_dino_model_name_or_path,
        drifting_dino_processor_name_or_path=args.drifting_dino_processor_name_or_path,
        drifting_dino_allow_remote=args.drifting_dino_allow_remote,
        revision=args.revision,
        choice_model=args.choice_model,
        choice_models=parse_csv_names(args.choice_models),
        choice_model_weights=parse_csv_floats(args.choice_model_weights) if args.choice_model_weights else (),
        choice_score_normalize=args.choice_score_normalize,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        pickscore_processor_name_or_path=args.pickscore_processor_name_or_path,
        pickscore_allow_remote=args.pickscore_allow_remote,
        geneval_repo=args.geneval_repo,
        geneval_detector_path=args.geneval_detector_path,
        geneval_model_config=args.geneval_model_config,
        geneval_options=args.geneval_options,
        geneval_max_rollout_rounds=args.geneval_max_rollout_rounds,
        seed=args.seed,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        batchsize_gen=args.batchsize_gen,
        generation_timestep=args.generation_timestep,
        generation_target_timestep=args.generation_target_timestep,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        max_train_samples=args.max_train_samples,
        proportion_empty_prompts=args.proportion_empty_prompts,
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
        resume_from_checkpoint=args.resume_from_checkpoint or None,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_csv_names(args.lora_target_modules),
        drifting_feature_mode=args.drifting_feature_mode,
        drifting_feature_key=parse_feature_key(args.drifting_feature_key, args.drifting_feature_extractor),
        drifting_feature_keys=parse_feature_keys(args.drifting_feature_keys, args.drifting_feature_extractor),
        drifting_feature_block_stride=args.drifting_feature_block_stride,
        drifting_feature_patch_sizes=parse_csv_ints(args.drifting_feature_patch_sizes),
        drifting_include_input_sq_mean=args.drifting_include_input_sq_mean,
        drifting_include_spatial_features=args.drifting_include_spatial_features,
        drifting_feature_aggregation=args.drifting_feature_aggregation,
        drifting_kernel=args.drifting_kernel,
        drifting_pref_r_list=parse_csv_floats(args.drifting_pref_r_list),
        drifting_ref_r_list=parse_csv_floats(args.drifting_ref_r_list),
        drifting_pos_weight=args.drifting_pos_weight,
        drifting_neg_weight=args.drifting_neg_weight,
        drifting_ref_weight=args.drifting_ref_weight,
        drifting_ref_neg_weight=args.drifting_ref_neg_weight,
        drifting_ref_loss_weight=args.drifting_ref_loss_weight,
        frozen_feature_l2_weight=args.frozen_feature_l2_weight,
        ref_model_l2_weight=args.ref_model_l2_weight,
        vggflow_unet_reg_scale=args.vggflow_unet_reg_scale,
        num_pos_images_min=args.num_pos_images_min,
        num_pos_images=args.num_pos_images,
        num_neg_images=args.num_neg_images,
        online_feature_top_count=args.online_feature_top_count,
        online_feature_top_fraction=args.online_feature_top_fraction,
        offline_distance_top_fraction=args.offline_distance_top_fraction,
        offline_distance_score_mode=args.offline_distance_score_mode,
        offline_distance_score_normalize=args.offline_distance_score_normalize,
        offline_distance_score_aggregation=args.offline_distance_score_aggregation,
        offline_distance_ref_reduction=args.offline_distance_ref_reduction,
        offline_distance_use_data_anchors=args.offline_distance_use_data_anchors,
        offline_latent_encode_mode=args.offline_latent_encode_mode,
        eval_prompt_file=args.eval_prompt_file,
        num_eval_prompts=args.num_eval_prompts,
        eval_every_steps=args.eval_every_steps,
        eval_output_subdir=args.eval_output_subdir,
        eval_seed=args.eval_seed,
        sample_grad_norm_discard_quantile=args.sample_grad_norm_discard_quantile,
        per_sample_threshold_quantile=args.per_sample_threshold_quantile,
        per_sample_threshold_ema_decay=args.per_sample_threshold_ema_decay,
        per_sample_threshold_use_sqrt=args.per_sample_threshold_use_sqrt,
        per_sample_threshold_reduction=args.per_sample_threshold_reduction,
        log_task_grad_stats=args.log_task_grad_stats,
        task_grad_log_interval=args.task_grad_log_interval,
    )
