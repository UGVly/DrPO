from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drpo.paths import project_root


@dataclass(frozen=True)
class AssetSpec:
    name: str
    path: Path
    kind: str
    optional: bool = False


def required_assets(root: Path | None = None, *, include_optional: bool = False) -> list[AssetSpec]:
    root = root or project_root()
    assets = [
        AssetSpec("sd_turbo", root / "models" / "sd-turbo", "dir"),
        AssetSpec("sdxl_turbo", root / "models" / "stable-diffusion-xl-turbo", "dir"),
        AssetSpec("mae_vit_base", root / "models" / "facebook-vit-mae-base", "dir"),
        AssetSpec("pickscore", root / "models" / "PickScore_v1", "dir"),
        AssetSpec("prompt_file", root / "data" / "prompts" / "pickapicv2_test_unique.txt", "file"),
        AssetSpec("pairs_jsonl", root / "data" / "pairs.jsonl", "file"),
        AssetSpec("clip_l14", root / "models" / "CLIP-ViT-L-14", "dir", optional=True),
        AssetSpec(
            "hps_openclip",
            root / "models" / "CLIP-ViT-H-14-laion2B-s32B-b79K" / "open_clip_pytorch_model.bin",
            "file",
            optional=True,
        ),
        AssetSpec("hpsv2", root / "models" / "HPSv2" / "HPS_v2_compressed.pt", "file", optional=True),
        AssetSpec(
            "aesthetic_head",
            root / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth",
            "file",
            optional=True,
        ),
        AssetSpec("mae_latent_256", root / "models" / "mae_latent_256_torch.pth", "file", optional=True),
    ]
    if include_optional:
        return assets
    return [asset for asset in assets if not asset.optional]


def check_assets(root: Path | None = None, *, include_optional: bool = False) -> list[tuple[AssetSpec, bool]]:
    status: list[tuple[AssetSpec, bool]] = []
    assets = required_assets(root, include_optional=include_optional)
    for asset in assets:
        exists = asset.path.is_file() if asset.kind == "file" else asset.path.is_dir()
        status.append((asset, exists))
    return status
