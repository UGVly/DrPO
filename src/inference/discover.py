from __future__ import annotations

import argparse
import json
from pathlib import Path


def discover_checkpoints(outputs_dir: str | Path = "outputs") -> list[Path]:
    root = Path(outputs_dir)
    checkpoints = []
    for adapter in root.glob("**/checkpoint-*/unet_lora/adapter_model.safetensors"):
        checkpoints.append(adapter.parent.parent)
    return sorted(checkpoints, key=lambda path: path.as_posix())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover LoRA checkpoints for inference.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--write-jsonl", default=None)
    parser.add_argument("--print-paths", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = discover_checkpoints(args.outputs_dir)
    if args.write_jsonl:
        output = Path(args.write_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for checkpoint in checkpoints:
                handle.write(json.dumps({"checkpoint": checkpoint.as_posix()}, ensure_ascii=False) + "\n")
    if args.print_paths or not args.write_jsonl:
        for checkpoint in checkpoints:
            print(checkpoint.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

