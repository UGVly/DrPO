from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from drpo.paths import project_root

CORE_REWARD_NAMES = ("pickscore", "clip", "aes", "hpsv2")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def batched(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start: start + batch_size]


def num_batches(num_items: int, batch_size: int) -> int:
    return math.ceil(num_items / batch_size) if batch_size > 0 else 0


def open_one_image(row: dict[str, Any]) -> Image.Image:
    with Image.open(resolve_image_path(row["image_path"])) as image:
        return image.convert("RGB").copy()


def open_images(rows: list[dict[str, Any]], *, num_workers: int = 0) -> list[Image.Image]:
    if num_workers <= 1 or len(rows) <= 1:
        return [open_one_image(row) for row in rows]
    workers = min(num_workers, len(rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(open_one_image, rows))


def close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def image_batch_to_tensor(images: list[Image.Image], *, size: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    tensors = []
    for image in images:
        resized = image.resize((size, size), resample=Image.BICUBIC)
        tensors.append(TF.pil_to_tensor(resized).float() / 255.0)
    return torch.stack(tensors, dim=0)


class ImageTensorDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, size: int):
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        path = resolve_image_path(self.rows[index]["image_path"])
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.size, self.size), resample=Image.BICUBIC)
            return TF.pil_to_tensor(image).float().div_(255.0)


def image_tensor_loader(
    rows: list[dict[str, Any]],
    *,
    size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ImageTensorDataset(rows, size=size), **kwargs)


def manifest_cache_key(manifest: Path) -> str:
    resolved = manifest.resolve()
    stat = resolved.stat()
    payload = f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def feature_cache_path(cache_dir: Path | None, *, prefix: str, manifest: Path | None) -> Path | None:
    if cache_dir is None or manifest is None:
        return None
    return cache_dir / f"{prefix}-{manifest_cache_key(manifest)}.pt"


def scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan}
    return {"mean": float(mean(values)), "std": float(pstdev(values)) if len(values) > 1 else 0.0}


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_images": len(rows),
        "model_type": rows[0].get("model_type") if rows else None,
        "checkpoint_path": rows[0].get("checkpoint_path") if rows else None,
    }
    for key in CORE_REWARD_NAMES:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if values:
            stat = scalar_summary(values)
            summary[f"{key}_mean"] = stat["mean"]
            summary[f"{key}_std"] = stat["std"]
    return summary


def manifest_metric_dir(samples_dir: Path, metrics_dir: Path, manifest: Path) -> Path:
    manifest_parent = manifest.parent.resolve()
    root = samples_dir.resolve()
    try:
        relative = manifest_parent.relative_to(root)
    except ValueError:
        relative = Path(manifest_parent.name)
    return metrics_dir / relative


def discover_manifests(samples_dir: Path) -> list[Path]:
    manifests = sorted((samples_dir / "sd-turbo-baseline").glob("**/manifest.jsonl"))
    manifests.extend(sorted((samples_dir / "sd-turbo-lora").glob("**/manifest.jsonl")))
    manifests.extend(sorted((samples_dir / "sdxl-turbo-lora").glob("**/manifest.jsonl")))
    manifests.extend(sorted((samples_dir / "diffusers-baseline").glob("**/manifest.jsonl")))
    return manifests


def resolve_requested_manifests(args, samples_dir: Path) -> list[Path]:
    manifest_list = getattr(args, "manifest_list", None)
    manifest = getattr(args, "manifest", None)
    if manifest_list:
        with Path(manifest_list).open("r", encoding="utf-8") as handle:
            return [Path(line.strip()) for line in handle if line.strip()]
    if manifest:
        return [Path(manifest)]
    return discover_manifests(samples_dir)


def build_manifest_groups(manifests: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
    all_rows = []
    for manifest in manifests:
        rows = read_jsonl(manifest)
        start = len(all_rows)
        all_rows.extend(rows)
        groups.append({"manifest": manifest, "rows": rows, "start": start, "end": len(all_rows)})
    return groups, all_rows


def flatten_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        rows.extend(group["rows"])
    return rows


def component_scores_path(metric_dir: Path, component: str) -> Path:
    return metric_dir / f"scores.{component}.jsonl"


def write_summary_csv(metrics_dir: Path) -> None:
    summaries = []
    for summary_path in sorted(metrics_dir.glob("**/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["summary_path"] = summary_path.relative_to(metrics_dir).as_posix()
        summaries.append(row)
    if not summaries:
        return
    keys = sorted({key for row in summaries for key in row})
    output = metrics_dir / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
