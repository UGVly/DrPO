from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drpo.paths import project_root


@dataclass(frozen=True)
class AssetSpec:
    name: str
    path: Path
    kind: str


def required_assets(root: Path | None = None) -> list[AssetSpec]:
    root = root or project_root()
    return [
        AssetSpec("sd_turbo", root / "models" / "sd-turbo", "dir"),
        AssetSpec("sdxl_turbo", root / "models" / "sdxl-turbo", "dir"),
        AssetSpec("sdxl_vae_decoder", root / "models" / "sdxl-turbo-vae-decoder", "dir"),
        AssetSpec("mae_vit_base", root / "models" / "mae-vit-base", "dir"),
        AssetSpec("pickscore", root / "models" / "PickScore_v1", "dir"),
        AssetSpec("clip_l14", root / "models" / "CLIP-ViT-L-14", "dir"),
        AssetSpec("hps_openclip", root / "models" / "CLIP-ViT-H-14-laion2B-s32B-b79K" / "open_clip_pytorch_model.bin", "file"),
        AssetSpec("hpsv2", root / "models" / "HPSv2" / "HPS_v2_compressed.pt", "file"),
        AssetSpec("aesthetic_head", root / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth", "file"),
        AssetSpec("mae_latent_256", root / "drifting" / "mae_latent_256_torch.pth", "file"),
    ]


def eval_assets(root: Path | None = None) -> list[AssetSpec]:
    root = root or project_root()
    return [
        AssetSpec("hpsv3_checkpoint", root / "models" / "HPSv3" / "HPSv3.safetensors", "file"),
        AssetSpec("hpsv3_base_qwen", root / "models" / "Qwen2-VL-7B-Instruct", "dir"),
        AssetSpec("imagereward_checkpoint", root / "models" / "ImageReward" / "ImageReward.pt", "file"),
        AssetSpec("imagereward_med_config", root / "models" / "ImageReward" / "med_config.json", "file"),
        AssetSpec("imagereward_bert_tokenizer", root / "models" / "bert-base-uncased", "dir"),
    ]


def check_assets(root: Path | None = None, *, include_eval: bool = False) -> list[tuple[AssetSpec, bool]]:
    status: list[tuple[AssetSpec, bool]] = []
    assets = required_assets(root)
    if include_eval:
        assets.extend(eval_assets(root))
    for asset in assets:
        exists = asset.path.is_file() if asset.kind == "file" else asset.path.is_dir()
        status.append((asset, exists))
    return status
