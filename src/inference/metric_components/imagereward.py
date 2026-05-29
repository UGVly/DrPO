from __future__ import annotations

import json
from pathlib import Path

from inference.metric_components.common import (
    build_summary,
    manifest_metric_dir,
    read_jsonl,
    resolve_requested_manifests,
    write_jsonl,
    write_summary_csv,
)
from inference.metric_components.reward_models import score_with_selector


def evaluate_imagereward(args) -> None:
    samples_dir = Path(args.samples_dir)
    metrics_dir = Path(args.metrics_dir)
    manifests = resolve_requested_manifests(args, samples_dir)
    for manifest in manifests:
        metric_dir = manifest_metric_dir(samples_dir, metrics_dir, manifest)
        scores_path = metric_dir / "scores.jsonl"
        summary_path = metric_dir / "summary.json"
        rows = read_jsonl(scores_path if scores_path.is_file() else manifest)
        if rows and "imagereward" in rows[0] and not args.force:
            continue
        score_with_selector(
            rows,
            selector_name="imagereward",
            device=args.device,
            batch_size=args.reward_batch_size,
            image_load_workers=args.image_load_workers,
        )
        summary = {}
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        summary.update(build_summary(rows))
        metric_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(scores_path, rows)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_csv(metrics_dir)
