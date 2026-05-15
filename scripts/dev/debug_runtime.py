from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch
from PIL import Image

from drpo.assets import check_assets
from drpo.paths import project_root
from drpo.rewards import build_selector
from drpo.sdturbo import load_sdturbo_components


MODULES = (
    "drpo.config",
    "drpo.data",
    "drpo.drift",
    "drpo.features",
    "drpo.paths",
    "drpo.rewards",
    "drpo.sdturbo",
    "drpo.training.trainer",
    "drpo.training.sdturbo_drpo",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def check_imports() -> None:
    for module in MODULES:
        importlib.import_module(module)
        print(f"OK      import            {module}")


def check_local_assets() -> None:
    missing = []
    for asset, exists in check_assets():
        state = "OK" if exists else "MISSING"
        print(f"{state:7s} asset             {asset.name:18s} {asset.path}")
        if not exists:
            missing.append(asset)
    if missing:
        names = ", ".join(asset.name for asset in missing)
        raise FileNotFoundError(f"Missing local assets: {names}")


def smoke_sdturbo() -> None:
    tokenizer, text_encoder, vae, unet, scheduler = load_sdturbo_components(str(project_root() / "models" / "sd-turbo"))
    print(f"OK      sdturbo           tokenizer={type(tokenizer).__name__} max_length={tokenizer.model_max_length}")
    print(f"OK      sdturbo           text_encoder={type(text_encoder).__name__}")
    print(f"OK      sdturbo           vae={type(vae).__name__} scaling_factor={vae.config.scaling_factor}")
    print(f"OK      sdturbo           unet={type(unet).__name__} in_channels={unet.config.in_channels}")
    print(f"OK      sdturbo           scheduler={type(scheduler).__name__} steps={len(scheduler.betas)}")


def smoke_rewards(names: list[str], device: torch.device) -> None:
    image = Image.new("RGB", (224, 224), "white")
    for name in names:
        selector = build_selector(name, device)
        score = selector.score([image], ["a white square"])[0]
        print(f"OK      reward            {name:9s} score={score:.6f} device={device}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug DrPO runtime wiring without starting a full experiment.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--eval-assets", action="store_true", help="also check HPSv3 and ImageReward assets")
    parser.add_argument("--skip-imports", action="store_true")
    parser.add_argument("--sdturbo", action="store_true", help="load SD-Turbo components from local models/sd-turbo")
    parser.add_argument(
        "--reward",
        action="append",
        choices=["pickscore", "clip", "aes", "hps", "hpsv2", "hpsv3", "imagereward", "all", "all-eval"],
        default=[],
        help="smoke-test one reward selector; repeat for multiple selectors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.skip_imports:
        check_imports()
    if not args.skip_assets:
        check_local_assets()
        if args.eval_assets:
            missing = []
            for asset, exists in check_assets(include_eval=True)[len(check_assets()) :]:
                state = "OK" if exists else "MISSING"
                print(f"{state:7s} asset             {asset.name:18s} {asset.path}")
                if not exists:
                    missing.append(asset)
            if missing:
                names = ", ".join(asset.name for asset in missing)
                raise FileNotFoundError(f"Missing local eval assets: {names}")
    if args.sdturbo:
        smoke_sdturbo()
    reward_names = args.reward
    if "all" in reward_names:
        reward_names = ["pickscore", "clip", "aes", "hps"]
    if "all-eval" in reward_names:
        reward_names = ["hpsv3", "imagereward"]
    if reward_names:
        smoke_rewards(reward_names, _device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
