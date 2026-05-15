from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.paths import project_root


def download_hpsv3(root: Path) -> None:
    from huggingface_hub import snapshot_download

    hpsv3_dir = root / "models" / "HPSv3"
    qwen_dir = root / "models" / "Qwen2-VL-7B-Instruct"
    print(f"Downloading HPSv3 checkpoint to {hpsv3_dir}")
    snapshot_download(
        repo_id="MizzenAI/HPSv3",
        repo_type="model",
        local_dir=hpsv3_dir,
        allow_patterns=["HPSv3.safetensors", "config.json", "README.md"],
    )
    print(f"Downloading Qwen2-VL base model to {qwen_dir}")
    snapshot_download(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        repo_type="model",
        local_dir=qwen_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )


def download_imagereward(root: Path) -> None:
    from huggingface_hub import snapshot_download

    target = root / "models" / "ImageReward"
    bert_target = root / "models" / "bert-base-uncased"
    print(f"Downloading ImageReward assets to {target}")
    snapshot_download(
        repo_id="THUDM/ImageReward",
        repo_type="model",
        local_dir=target,
        allow_patterns=["ImageReward.pt", "med_config.json", "README.md"],
    )
    print(f"Downloading ImageReward BERT tokenizer to {bert_target}")
    snapshot_download(
        repo_id="bert-base-uncased",
        repo_type="model",
        local_dir=bert_target,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download optional local evaluation reward assets.")
    parser.add_argument("--model", choices=["hpsv3", "imagereward", "all"], default="all")
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--endpoint", default=None, help="optional Hugging Face endpoint, for example https://hf-mirror.com")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    root = Path(args.root).resolve()
    if args.model in {"hpsv3", "all"}:
        download_hpsv3(root)
    if args.model in {"imagereward", "all"}:
        download_imagereward(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
