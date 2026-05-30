import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

from PIL import Image

warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.rewards import build_selector


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Row {line_number} must be a JSON object.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise TypeError("Image field must be a path string or a list of path strings.")


def _image_values(row: dict[str, Any], preferred_key: str) -> list[str]:
    for key in (preferred_key, "image_path", "images", "chosen", "rejected"):
        if key in row:
            return _as_list(row[key])
    raise KeyError(f"Row is missing image field. Tried: {preferred_key}, image_path, images, chosen, rejected.")


def _resolve_images(paths: list[str], base_dir: Path) -> list[Path]:
    resolved = []
    for path in paths:
        value = Path(path)
        resolved.append(value if value.is_absolute() else (base_dir / value).resolve())
    return resolved


def _score_paths_or_images(selector, paths: list[Path], prompt: str) -> list[float]:
    if hasattr(selector, "score_paths"):
        return selector.score_paths(paths, prompt)
    images = []
    try:
        for path in paths:
            images.append(Image.open(path).convert("RGB"))
        return selector.score(images, prompt)
    finally:
        for image in images:
            image.close()


def evaluate_rows(args) -> list[dict[str, Any]]:
    selector = build_selector(args.selector, args.device)
    input_path = Path(args.input_jsonl)
    base_dir = Path(args.base_dir).resolve() if args.base_dir else input_path.resolve().parent
    score_key = args.score_key or f"{args.selector.replace('-', '_')}_scores"
    rows = _read_jsonl(input_path)
    outputs = []
    for row in rows:
        prompt = str(row[args.prompt_key])
        paths = _resolve_images(_image_values(row, args.image_key), base_dir)
        scores = _score_paths_or_images(selector, paths, prompt)
        output = dict(row)
        output[score_key] = scores
        if len(scores) == 1:
            output[score_key[:-1] if score_key.endswith("s") else f"{score_key}_value"] = scores[0]
        outputs.append(output)
    return outputs


def evaluate_single(args) -> list[dict[str, Any]]:
    selector = build_selector(args.selector, args.device)
    paths = [Path(path).resolve() for path in args.image]
    scores = _score_paths_or_images(selector, paths, args.prompt)
    return [{"prompt": args.prompt, "images": [str(path) for path in paths], f"{args.selector}_scores": scores}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score generated images with local reward models.")
    parser.add_argument("--selector", required=True, choices=["pickscore", "clip", "aes", "hps", "hpsv2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--score-key", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_jsonl:
        if not args.output_jsonl:
            raise ValueError("--output-jsonl is required with --input-jsonl.")
        rows = evaluate_rows(args)
        _write_jsonl(Path(args.output_jsonl), rows)
    else:
        if not args.prompt or not args.image:
            raise ValueError("Provide either --input-jsonl/--output-jsonl or --prompt with one or more --image.")
        rows = evaluate_single(args)
        print(json.dumps(rows[0], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
