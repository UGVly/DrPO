from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from drpo.paths import project_root
from drpo.rewards import build_selector
from inference.metric_components.common import (
    CORE_REWARD_NAMES,
    ImageTensorDataset,
    batched,
    build_manifest_groups,
    build_summary,
    component_scores_path,
    discover_manifests,
    feature_cache_path,
    flatten_groups,
    image_batch_to_tensor,
    image_tensor_loader,
    manifest_metric_dir,
    num_batches,
    open_images,
    read_jsonl,
    resolve_image_path,
    resolve_requested_manifests,
    scalar_summary,
    write_jsonl,
    write_summary_csv,
)
from inference.metric_components.imagereward import evaluate_imagereward
from inference.metric_components.reward_models import build_clip_aes_models, score_clip_aes, score_with_selector

CORE_COMPONENTS = ("pickscore", "clip_aes", "hpsv2", "summarize")

__all__ = [
    "ImageTensorDataset",
    "batched",
    "build_summary",
    "discover_manifests",
    "evaluate_core",
    "evaluate_imagereward",
    "feature_cache_path",
    "image_batch_to_tensor",
    "image_tensor_loader",
    "manifest_metric_dir",
    "num_batches",
    "open_images",
    "read_jsonl",
    "resolve_image_path",
    "scalar_summary",
    "write_jsonl",
    "write_summary_csv",
]


def parse_component_list(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested or requested == ["all"] or "all" in requested:
        return list(CORE_COMPONENTS)
    unknown = sorted(set(requested) - set(CORE_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown components: {', '.join(unknown)}")
    return requested


def component_exists(metric_dir: Path, component: str) -> bool:
    if component == "pickscore":
        return component_scores_path(metric_dir, "pickscore").is_file()
    if component == "clip_aes":
        return component_scores_path(metric_dir, "clip_aes").is_file()
    if component == "hpsv2":
        return component_scores_path(metric_dir, "hpsv2").is_file()
    if component == "summarize":
        return (metric_dir / "scores.jsonl").is_file() and (metric_dir / "summary.json").is_file()
    return False


def groups_needing_component(
    groups: list[dict[str, Any]],
    *,
    samples_dir: Path,
    metrics_dir: Path,
    component: str,
    force: bool,
) -> list[dict[str, Any]]:
    if force:
        return groups
    pending = []
    for group in groups:
        metric_dir = manifest_metric_dir(samples_dir, metrics_dir, group["manifest"])
        if not component_exists(metric_dir, component):
            pending.append(group)
    return pending


def write_component_scores(
    groups: list[dict[str, Any]],
    *,
    samples_dir: Path,
    metrics_dir: Path,
    component: str,
) -> None:
    for group in groups:
        metric_dir = manifest_metric_dir(samples_dir, metrics_dir, group["manifest"])
        metric_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(component_scores_path(metric_dir, component), group["rows"])


def _merge_component_scores(rows: list[dict[str, Any]], metric_dir: Path) -> None:
    for component in ("pickscore", "clip_aes", "hpsv2"):
        path = component_scores_path(metric_dir, component)
        if not path.is_file():
            continue
        component_rows = read_jsonl(path)
        if len(component_rows) != len(rows):
            raise ValueError(f"Component row count mismatch for {path}")
        for row, component_row in zip(rows, component_rows):
            for key in CORE_REWARD_NAMES:
                if key in component_row:
                    row[key] = component_row[key]


def evaluate_core(args) -> None:
    samples_dir = Path(args.samples_dir)
    metrics_dir = Path(args.metrics_dir)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    components = parse_component_list(getattr(args, "components", "all"))
    manifests = resolve_requested_manifests(args, samples_dir)
    groups, _ = build_manifest_groups(manifests)
    if not groups:
        write_summary_csv(metrics_dir)
        return

    if "pickscore" in components:
        selected = groups_needing_component(groups, samples_dir=samples_dir, metrics_dir=metrics_dir, component="pickscore", force=args.force)
        if selected:
            rows = flatten_groups(selected)
            score_with_selector(
                rows,
                selector_name="pickscore",
                device=args.device,
                batch_size=args.reward_batch_size,
                image_load_workers=args.image_load_workers,
                selector=build_selector("pickscore", args.device),
            )
            write_component_scores(selected, samples_dir=samples_dir, metrics_dir=metrics_dir, component="pickscore")

    if "clip_aes" in components:
        selected = groups_needing_component(groups, samples_dir=samples_dir, metrics_dir=metrics_dir, component="clip_aes", force=args.force)
        if selected:
            rows = flatten_groups(selected)
            clip_processor, clip_model, aes_head, device_obj = build_clip_aes_models(args.device)
            score_clip_aes(
                rows,
                batch_size=args.reward_batch_size,
                processor=clip_processor,
                model=clip_model,
                head=aes_head,
                device_obj=device_obj,
                image_load_workers=args.image_load_workers,
            )
            write_component_scores(selected, samples_dir=samples_dir, metrics_dir=metrics_dir, component="clip_aes")

    if "hpsv2" in components:
        selected = groups_needing_component(groups, samples_dir=samples_dir, metrics_dir=metrics_dir, component="hpsv2", force=args.force)
        if selected:
            rows = flatten_groups(selected)
            score_with_selector(
                rows,
                selector_name="hpsv2",
                device=args.device,
                batch_size=args.reward_batch_size,
                image_load_workers=args.image_load_workers,
                selector=build_selector("hpsv2", args.device),
            )
            write_component_scores(selected, samples_dir=samples_dir, metrics_dir=metrics_dir, component="hpsv2")

    if "summarize" in components:
        selected = groups_needing_component(groups, samples_dir=samples_dir, metrics_dir=metrics_dir, component="summarize", force=args.force)
        for group in selected:
            manifest = group["manifest"]
            rows = read_jsonl(manifest)
            metric_dir = manifest_metric_dir(samples_dir, metrics_dir, manifest)
            _merge_component_scores(rows, metric_dir)
            summary = build_summary(rows)
            metric_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(metric_dir / "scores.jsonl", rows)
            with (metric_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_csv(metrics_dir)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate generated samples.")
    parser.add_argument("--metric-set", choices=["core", "imagereward"], default="core")
    parser.add_argument("--samples-dir", default=str(root / "samples"))
    parser.add_argument("--metrics-dir", default=str(root / "samples" / "metrics"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--manifest-list", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-load-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated core metric components: pickscore,clip_aes,hpsv2,summarize, or all.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_set == "core":
        evaluate_core(args)
    else:
        evaluate_imagereward(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
