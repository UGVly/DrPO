#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SAMPLE_ROOTS = (Path("/datapool"), Path.cwd())
DEFAULT_DRPO = "samples/sd-turbo-lora/drpo/online/reference_ablation/ref-drift-002/20260429134304/checkpoint-1000/manifest.jsonl"
DEFAULT_DRAFT_200 = "samples/sd-turbo-lora/draft/sdturbo_lora/default/20260426135052/checkpoint-200/manifest.jsonl"
DEFAULT_DRAFT_1000 = "samples/sd-turbo-lora/draft/sdturbo_lora/default/20260426135052/checkpoint-1000/manifest.jsonl"


JUDGE_PROMPT = """You are an expert text-to-image preference judge.

Task: compare two generated images for the same text prompt and choose the image that better satisfies the prompt while being visually high quality.

Prompt:
{prompt}

Evaluation criteria, in priority order:
1. Prompt alignment: objects, attributes, relations, style, counting, and composition requested by the prompt.
2. Visual quality: fewer artifacts, cleaner anatomy/geometry/textures, coherent lighting, and realistic or stylistically consistent rendering.
3. Overall appeal: choose the more compelling image only after considering prompt alignment and artifact severity.

Important instructions:
- Do not prefer an image because it is labeled A or B.
- Penalize obvious artifacts, broken shapes, extra/missing key objects, unreadable required text, and prompt-irrelevant content.
- If both images are genuinely comparable, answer tie.
- Return only valid JSON, with no markdown.

Output schema:
{{"winner":"A"|"B"|"tie","confidence":0.0-1.0,"reason":"short reason under 25 words"}}
"""


@dataclass(frozen=True)
class ManifestRow:
    key: tuple[int, int]
    prompt: str
    image_path: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class Pair:
    key: tuple[int, int]
    prompt: str
    left_name: str
    left_image: Path
    right_name: str
    right_image: Path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_existing(path: str | Path, roots: list[Path]) -> Path:
    path = Path(path)
    if path.exists():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve path: {path}")


def image_path_from_row(row: dict[str, Any], manifest_path: Path, roots: list[Path]) -> Path:
    value = row.get("image_path") or row.get("path") or row.get("file_name")
    if not value:
        raise ValueError(f"Manifest row has no image path: {row}")
    candidates = [
        Path(value),
        manifest_path.parent / str(value),
        manifest_path.parents[0] / str(value),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for root in roots:
        candidate = root / str(value)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image path from manifest {manifest_path}: {value}")


def load_manifest(path: Path, roots: list[Path]) -> dict[tuple[int, int], ManifestRow]:
    lookup: dict[tuple[int, int], ManifestRow] = {}
    for row in load_jsonl(path):
        prompt_id = int(row["prompt_id"])
        seed = int(row.get("seed", row.get("latent_seed", 0)))
        key = (prompt_id, seed)
        lookup[key] = ManifestRow(
            key=key,
            prompt=str(row["prompt"]),
            image_path=image_path_from_row(row, path, roots),
            raw=row,
        )
    return lookup


def build_pairs(
    left_manifest: Path,
    right_manifest: Path,
    left_name: str,
    right_name: str,
    roots: list[Path],
    max_pairs: int | None,
    seed: int,
) -> list[Pair]:
    left = load_manifest(left_manifest, roots)
    right = load_manifest(right_manifest, roots)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise RuntimeError("No matched (prompt_id, seed) pairs found.")
    rng = random.Random(seed)
    rng.shuffle(keys)
    if max_pairs is not None:
        keys = keys[:max_pairs]
    pairs = []
    for key in keys:
        if left[key].prompt != right[key].prompt:
            raise ValueError(f"Prompt mismatch for key {key}")
        pairs.append(
            Pair(
                key=key,
                prompt=left[key].prompt,
                left_name=left_name,
                left_image=left[key].image_path,
                right_name=right_name,
                right_image=right[key].image_path,
            )
        )
    return pairs


def deterministic_swap(pair: Pair, salt: str) -> bool:
    digest = hashlib.sha256(f"{salt}:{pair.key[0]}:{pair.key[1]}".encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def parse_judgment(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        lower = text.lower()
        if re.search(r"\bimage\s*a\b|\ba\b", lower) and not re.search(r"\bimage\s*b\b|\bb\b", lower):
            data = {"winner": "A", "confidence": None, "reason": text[:200]}
        elif re.search(r"\bimage\s*b\b|\bb\b", lower) and not re.search(r"\bimage\s*a\b|\ba\b", lower):
            data = {"winner": "B", "confidence": None, "reason": text[:200]}
        elif "tie" in lower:
            data = {"winner": "tie", "confidence": None, "reason": text[:200]}
        else:
            data = {"winner": "invalid", "confidence": None, "reason": text[:200]}
    winner = str(data.get("winner", "invalid")).strip().upper()
    if winner not in {"A", "B"}:
        winner = "tie" if str(data.get("winner", "")).lower() == "tie" else "invalid"
    return {
        "winner": winner,
        "confidence": data.get("confidence"),
        "reason": str(data.get("reason", "")),
        "raw_text": text,
    }


def load_model(model_name: str, device_map: str, torch_dtype: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def judge_pair(
    model: Any,
    processor: Any,
    pair: Pair,
    swapped: bool,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str, str, Path, Path]:
    if swapped:
        image_a_name, image_a_path = pair.right_name, pair.right_image
        image_b_name, image_b_path = pair.left_name, pair.left_image
    else:
        image_a_name, image_a_path = pair.left_name, pair.left_image
        image_b_name, image_b_path = pair.right_name, pair.right_image

    prompt = JUDGE_PROMPT.format(prompt=pair.prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_a_path)},
                {"type": "image", "image": str(image_b_path)},
                {"type": "text", "text": f"Image A is the first image. Image B is the second image.\n\n{prompt}"},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=max_new_tokens,
    )
    generated = output_ids[:, inputs["input_ids"].shape[-1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return parse_judgment(text), image_a_name, image_b_name, image_a_path, image_b_path


def summarize(results: list[dict[str, Any]], left_name: str, right_name: str) -> dict[str, Any]:
    valid = [r for r in results if r["winner_method"] in {left_name, right_name, "tie"}]
    left_wins = sum(r["winner_method"] == left_name for r in valid)
    right_wins = sum(r["winner_method"] == right_name for r in valid)
    ties = sum(r["winner_method"] == "tie" for r in valid)
    n = len(valid)
    denom = max(1, n - ties)
    return {
        "left_name": left_name,
        "right_name": right_name,
        "num_pairs": len(results),
        "num_valid": n,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "invalid": len(results) - n,
        "left_win_rate_excluding_ties": left_wins / denom,
        "right_win_rate_excluding_ties": right_wins / denom,
        "left_preference_share_ties_half": (left_wins + 0.5 * ties) / max(1, n),
        "right_preference_share_ties_half": (right_wins + 0.5 * ties) / max(1, n),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    args: argparse.Namespace,
    label: str,
    left_manifest: Path,
    right_manifest: Path,
    model: Any | None = None,
    processor: Any | None = None,
) -> dict[str, Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    roots = [Path(root) for root in args.sample_root]
    pairs = build_pairs(
        left_manifest=left_manifest,
        right_manifest=right_manifest,
        left_name=label,
        right_name=args.right_name,
        roots=roots,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    if args.num_shards > 1:
        pairs = pairs[args.shard_index :: args.num_shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_file = args.output_dir / f"{args.run_name}_{label}_pairs.jsonl"
    with pair_file.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": pair.key[0],
                        "seed": pair.key[1],
                        "prompt": pair.prompt,
                        label: str(pair.left_image),
                        args.right_name: str(pair.right_image),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if args.dry_run:
        return {
            "comparison": f"{label} vs {args.right_name}",
            "pairs": len(pairs),
            "pair_file": str(pair_file),
            "dry_run": True,
        }

    if model is None or processor is None:
        model, processor = load_model(args.model, args.device_map, args.torch_dtype)
    rows = []
    for pair in tqdm(pairs, desc=f"Qwen judge {label} vs {args.right_name}"):
        swapped = deterministic_swap(pair, args.swap_salt)
        judgment, image_a_name, image_b_name, image_a_path, image_b_path = judge_pair(
            model=model,
            processor=processor,
            pair=pair,
            swapped=swapped,
            max_new_tokens=args.max_new_tokens,
        )
        winner = judgment["winner"]
        if winner == "A":
            winner_method = image_a_name
        elif winner == "B":
            winner_method = image_b_name
        else:
            winner_method = winner
        rows.append(
            {
                "comparison": f"{label} vs {args.right_name}",
                "prompt_id": pair.key[0],
                "seed": pair.key[1],
                "prompt": pair.prompt,
                "image_a_method": image_a_name,
                "image_b_method": image_b_name,
                "image_a_path": str(image_a_path),
                "image_b_path": str(image_b_path),
                "winner_ab": winner,
                "winner_method": winner_method,
                "confidence": judgment["confidence"],
                "reason": judgment["reason"],
                "raw_text": judgment["raw_text"],
            }
        )
    result_file = args.output_dir / f"{args.run_name}_{label}_judgments.jsonl"
    with result_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_file = args.output_dir / f"{args.run_name}_{label}_judgments.csv"
    write_csv(csv_file, rows)
    summary = summarize(rows, label, args.right_name)
    summary.update(
        {
            "comparison": f"{label} vs {args.right_name}",
            "model": args.model,
            "left_manifest": str(left_manifest),
            "right_manifest": str(right_manifest),
            "pair_file": str(pair_file),
            "judgment_file": str(result_file),
            "csv_file": str(csv_file),
        }
    )
    write_summary(args.output_dir / f"{args.run_name}_{label}_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise Qwen3-VL win-rate judge for DrPO comparisons.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-root", action="append", default=[str(p) for p in DEFAULT_SAMPLE_ROOTS])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/qwen_vl_winrate"))
    parser.add_argument("--run-name", default="draft_drpo_qwen3vl")
    parser.add_argument("--right-name", default="DrPO")
    parser.add_argument("--drpo-manifest", default=DEFAULT_DRPO)
    parser.add_argument("--draft-200-manifest", default=DEFAULT_DRAFT_200)
    parser.add_argument("--draft-1000-manifest", default=DEFAULT_DRAFT_1000)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--swap-salt", default="qwen3vl_pairwise_v1")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    return args


def main() -> None:
    args = parse_args()
    roots = [Path(root) for root in args.sample_root]
    drpo_manifest = resolve_existing(args.drpo_manifest, roots)
    comparisons = [
        ("DRaFT-200", resolve_existing(args.draft_200_manifest, roots)),
        ("DRaFT-1000", resolve_existing(args.draft_1000_manifest, roots)),
    ]
    summaries = []
    model = None
    processor = None
    if not args.dry_run:
        model, processor = load_model(args.model, args.device_map, args.torch_dtype)
    for label, manifest in comparisons:
        summaries.append(run_comparison(args, label, manifest, drpo_manifest, model=model, processor=processor))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / f"{args.run_name}_summary.json", {"comparisons": summaries})
    print(json.dumps({"comparisons": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
