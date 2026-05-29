from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from drpo.paths import project_root, require_local_path
from drpo.rewards import AestheticHead, build_selector
from drpo.utils.tensors import feature_tensor, normalize_features, strip_state_dict_prefixes


CORE_REWARD_NAMES = ("pickscore", "clip", "aes", "hpsv2")
CORE_COMPONENTS = ("pickscore", "clip_aes", "hpsv2", "dino", "fid", "summarize")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
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


def score_with_selector(
    rows: list[dict[str, Any]],
    *,
    selector_name: str,
    device: str,
    batch_size: int,
    image_load_workers: int = 0,
    selector: Any | None = None,
) -> None:
    selector = selector or build_selector(selector_name, device)
    key = "hpsv2" if selector_name == "hpsv2" else selector_name
    for batch in tqdm(batched(rows, batch_size), total=num_batches(len(rows), batch_size), desc=f"score {key}", dynamic_ncols=True):
        images = open_images(batch, num_workers=image_load_workers)
        prompts = [str(row["prompt"]) for row in batch]
        try:
            scores = selector.score(images, prompts)
        finally:
            close_images(images)
        for row, score in zip(batch, scores):
            row[key] = float(score)


@torch.no_grad()
def score_clip_aes_and_features(
    rows: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    model_path: str | Path | None = None,
    head_path: str | Path | None = None,
    processor: Any | None = None,
    model: Any | None = None,
    head: AestheticHead | None = None,
    device_obj: torch.device | None = None,
    image_load_workers: int = 0,
) -> torch.Tensor:
    from packaging.version import Version
    from transformers import CLIPModel, CLIPProcessor
    from transformers import __version__ as transformers_version

    root = project_root()
    clip_path = require_local_path(
        model_path or os.getenv("CLIP_MODEL_PATH") or root / "models" / "CLIP-ViT-L-14",
        description="CLIP reward model",
        must_be_file=False,
    )
    aesthetic_head_path = require_local_path(
        head_path
        or os.getenv("AES_HEAD_PATH")
        or root / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth",
        description="Aesthetic reward head",
        must_be_file=True,
    )
    device_obj = device_obj or torch.device(device)
    if processor is None or model is None or head is None:
        dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
        dtype_key = "dtype" if Version(transformers_version).major >= 5 else "torch_dtype"
        processor = CLIPProcessor.from_pretrained(str(clip_path), local_files_only=True)
        model = CLIPModel.from_pretrained(str(clip_path), **{dtype_key: dtype}, local_files_only=True).to(device_obj).eval()
        projection_dim = int(getattr(model.config, "projection_dim", 768))
        head = AestheticHead(projection_dim).to(device_obj).eval()
        head.load_state_dict(strip_state_dict_prefixes(torch.load(aesthetic_head_path, map_location="cpu")))

    features = []
    for batch in tqdm(batched(rows, batch_size), total=num_batches(len(rows), batch_size), desc="score clip/aes", dynamic_ncols=True):
        images = open_images(batch, num_workers=image_load_workers)
        prompts = [str(row["prompt"]) for row in batch]
        try:
            image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device_obj)
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device_obj)
            image_features = normalize_features(feature_tensor(model.get_image_features(**image_inputs)).float())
            text_features = normalize_features(feature_tensor(model.get_text_features(**text_inputs)).float())
            clip_scores = (image_features * text_features).sum(dim=-1).float().cpu().tolist()
            aes_scores = head(image_features).flatten().float().cpu().tolist()
            features.append(image_features.cpu())
        finally:
            close_images(images)
        for row, clip_score, aes_score in zip(batch, clip_scores, aes_scores):
            row["clip"] = float(clip_score)
            row["aes"] = float(aes_score)
    return torch.cat(features, dim=0)


def image_batch_to_tensor(images: list[Image.Image], *, size: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    tensors = []
    for image in images:
        resized = image.resize((size, size), resample=Image.BICUBIC)
        tensor = TF.pil_to_tensor(resized).float() / 255.0
        tensors.append(tensor)
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


@torch.no_grad()
def clip_image_features(rows: list[dict[str, Any]], *, device: str, batch_size: int, model_path: str | Path | None = None) -> torch.Tensor:
    from transformers import CLIPModel, CLIPProcessor
    from transformers import __version__ as transformers_version
    from packaging.version import Version

    root = project_root()
    path = require_local_path(model_path or root / "models" / "CLIP-ViT-L-14", description="CLIP reward model", must_be_file=False)
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    processor = CLIPProcessor.from_pretrained(str(path), local_files_only=True)
    dtype_key = "dtype" if Version(transformers_version).major >= 5 else "torch_dtype"
    model = CLIPModel.from_pretrained(str(path), **{dtype_key: dtype}, local_files_only=True).to(device_obj).eval()
    features = []
    for batch in tqdm(batched(rows, batch_size), total=num_batches(len(rows), batch_size), desc="clip features", dynamic_ncols=True):
        images = open_images(batch)
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device_obj)
            values = model.get_image_features(**inputs).float()
            features.append(F.normalize(values, dim=-1).cpu())
        finally:
            close_images(images)
    return torch.cat(features, dim=0)


@torch.no_grad()
def dino_image_features(
    rows: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    model_path: str | Path | None = None,
    processor_path: str | Path | None = None,
    feature_key: str = "layer12_patch_mean",
    model: Any | None = None,
    device_obj: torch.device | None = None,
) -> torch.Tensor:
    from drpo.features import FrozenDinoImageFeatureExtractor

    root = project_root()
    device_obj = device_obj or torch.device(device)
    model = model or FrozenDinoImageFeatureExtractor(
        model_path or root / "models" / "dinov2-base",
        processor_name_or_path=processor_path,
        feature_key=feature_key,
    ).to(device_obj)
    model.eval()
    features = []
    loader = image_tensor_loader(
        rows,
        size=model.input_size,
        batch_size=batch_size,
        device=device_obj,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    for tensor in tqdm(loader, desc="dino features", dynamic_ncols=True):
        tensor = tensor.to(device_obj, non_blocking=True)
        tensor = tensor * 2.0 - 1.0
        values = model(tensor).float()
        features.append(F.normalize(values, dim=-1).cpu())
    return torch.cat(features, dim=0)


def pairwise_cosine_distance_mean(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return 0.0
    features = F.normalize(features.float(), dim=-1)
    sim = features @ features.T
    n = sim.shape[0]
    off_diag_sum = sim.sum() - sim.diag().sum()
    mean_sim = off_diag_sum / (n * (n - 1))
    return float((1.0 - mean_sim).item())


def grouped_diversity(rows: list[dict[str, Any]], features: torch.Tensor) -> tuple[float, dict[str, float]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["seed"])].append(index)
    per_seed = {}
    for seed, indices in sorted(groups.items()):
        per_seed[str(seed)] = pairwise_cosine_distance_mean(features[indices])
    return pairwise_cosine_distance_mean(features), per_seed


def prompt_seed_diversity(
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    min_seeds_per_prompt: int = 25,
    min_prompts: int = 1,
) -> tuple[float | None, dict[str, float], dict[str, int]]:
    """Mean within-prompt diversity across seeds, then average over valid prompts."""
    groups: dict[int, list[int]] = defaultdict(list)
    seeds_by_prompt: dict[int, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        prompt_id = int(row.get("prompt_id", index))
        groups[prompt_id].append(index)
        if "seed" in row:
            seeds_by_prompt[prompt_id].add(int(row["seed"]))

    per_prompt: dict[str, float] = {}
    seed_counts: dict[str, int] = {}
    required_seeds = max(2, int(min_seeds_per_prompt))
    for prompt_id, indices in sorted(groups.items()):
        seed_count = len(seeds_by_prompt[prompt_id])
        seed_counts[str(prompt_id)] = seed_count
        if len(indices) < 2 or seed_count < required_seeds:
            continue
        per_prompt[str(prompt_id)] = pairwise_cosine_distance_mean(features[indices])

    if len(per_prompt) < max(1, int(min_prompts)):
        return None, per_prompt, seed_counts
    return float(mean(per_prompt.values())), per_prompt, seed_counts


def diversity_summary(
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    prefix: str,
    min_seeds_per_prompt: int,
    min_prompts: int,
) -> dict[str, Any]:
    prompt_mean, per_prompt, seed_counts = prompt_seed_diversity(
        rows,
        features,
        min_seeds_per_prompt=min_seeds_per_prompt,
        min_prompts=min_prompts,
    )
    global_diversity, per_seed = grouped_diversity(rows, features)
    output: dict[str, Any] = {
        f"{prefix}_diversity_global": global_diversity,
        f"{prefix}_diversity_by_seed": per_seed,
        f"{prefix}_diversity_by_prompt": per_prompt,
        f"{prefix}_diversity_seed_count_by_prompt": seed_counts,
        f"{prefix}_diversity_min_seeds_per_prompt": int(min_seeds_per_prompt),
        f"{prefix}_diversity_min_prompts": int(min_prompts),
        f"{prefix}_diversity_valid_prompt_count": len(per_prompt),
        f"{prefix}_diversity_prompt_count": len(seed_counts),
    }
    if seed_counts:
        output[f"{prefix}_diversity_mean_seeds_per_prompt"] = float(mean(seed_counts.values()))
    if prompt_mean is None:
        output[f"{prefix}_diversity"] = None
        output[f"{prefix}_diversity_protocol"] = "insufficient_prompt_seed_groups"
    else:
        output[f"{prefix}_diversity"] = prompt_mean
        output[f"{prefix}_diversity_protocol"] = (
            f"prompt_mean_pairwise_cosine_distance_across_{int(min_seeds_per_prompt)}_seeds"
        )
    return output


@torch.no_grad()
def inception_features(
    rows: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    cache_path: Path | None = None,
    model: Any | None = None,
    mean_tensor: torch.Tensor | None = None,
    std_tensor: torch.Tensor | None = None,
    device_obj: torch.device | None = None,
) -> torch.Tensor:
    import torchvision.models as models

    if cache_path is not None and cache_path.is_file():
        return torch.load(cache_path, map_location="cpu")

    device_obj = device_obj or torch.device(device)
    if model is None or mean_tensor is None or std_tensor is None:
        weights = models.Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        model.to(device_obj).eval()
        mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device_obj).view(1, 3, 1, 1)
        std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device_obj).view(1, 3, 1, 1)
    features = []
    loader = image_tensor_loader(
        rows,
        size=299,
        batch_size=batch_size,
        device=device_obj,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    for tensor in tqdm(loader, desc="inception features", dynamic_ncols=True):
        tensor = tensor.to(device_obj, non_blocking=True)
        tensor = (tensor - mean_tensor) / std_tensor
        output = model(tensor)
        if hasattr(output, "logits"):
            output = output.logits
        features.append(output.float().cpu())
    result = torch.cat(features, dim=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
    return result


def covariance(features: torch.Tensor) -> torch.Tensor:
    values = features.double()
    centered = values - values.mean(dim=0, keepdim=True)
    denom = max(values.shape[0] - 1, 1)
    return centered.T @ centered / denom


def symmetric_matrix_sqrt(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    matrix = (matrix + matrix.T) / 2
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(eps).sqrt()
    return (vectors * values.unsqueeze(0)) @ vectors.T


def fid_from_features(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.double()
    cand = candidate.double()
    mu_ref = ref.mean(dim=0)
    mu_cand = cand.mean(dim=0)
    sigma_ref = covariance(ref)
    sigma_cand = covariance(cand)
    eps_eye = torch.eye(sigma_ref.shape[0], dtype=torch.double) * 1e-6
    sigma_ref = sigma_ref + eps_eye
    sigma_cand = sigma_cand + eps_eye
    sqrt_ref = symmetric_matrix_sqrt(sigma_ref)
    middle = sqrt_ref @ sigma_cand @ sqrt_ref
    trace_sqrt = torch.linalg.eigvalsh((middle + middle.T) / 2).clamp_min(0).sqrt().sum()
    value = (mu_ref - mu_cand).dot(mu_ref - mu_cand) + torch.trace(sigma_ref) + torch.trace(sigma_cand) - 2 * trace_sqrt
    return float(value.clamp_min(0).item())


def scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan}
    return {"mean": float(mean(values)), "std": float(pstdev(values)) if len(values) > 1 else 0.0}


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
    if args.manifest_list:
        manifest_list = Path(args.manifest_list)
        with manifest_list.open("r", encoding="utf-8") as handle:
            return [Path(line.strip()) for line in handle if line.strip()]
    if args.manifest:
        return [Path(args.manifest)]
    return discover_manifests(samples_dir)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_images": len(rows),
        "model_type": rows[0].get("model_type") if rows else None,
        "checkpoint_path": rows[0].get("checkpoint_path") if rows else None,
    }
    for key in (*CORE_REWARD_NAMES, "imagereward"):
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if values:
            stat = scalar_summary(values)
            summary[f"{key}_mean"] = stat["mean"]
            summary[f"{key}_std"] = stat["std"]
            if key == "imagereward":
                # The official ImageReward API returns normalized scores.
                # SD-Turbo/DrPO comparison tables report IR on the same
                # normalized scale multiplied by 10.
                summary["imagereward_x10_mean"] = stat["mean"] * 10.0
                summary["imagereward_x10_std"] = stat["std"] * 10.0
    return summary


def build_manifest_groups(manifests: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
    all_rows = []
    for manifest in manifests:
        rows = read_jsonl(manifest)
        start = len(all_rows)
        all_rows.extend(rows)
        groups.append(
            {
                "manifest": manifest,
                "rows": rows,
                "start": start,
                "end": len(all_rows),
            }
        )
    return groups, all_rows


def flatten_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        rows.extend(group["rows"])
    return rows


def split_tensor_by_groups(tensor: torch.Tensor, groups: list[dict[str, Any]]) -> dict[Path, torch.Tensor]:
    pieces = {}
    start = 0
    for group in groups:
        end = start + len(group["rows"])
        pieces[group["manifest"]] = tensor[start:end]
        start = end
    return pieces


def component_scores_path(metric_dir: Path, component: str) -> Path:
    return metric_dir / f"scores.{component}.jsonl"


def component_features_path(metric_dir: Path, name: str) -> Path:
    return metric_dir / f"features.{name}.pt"


def component_summary_path(metric_dir: Path, component: str) -> Path:
    return metric_dir / f"summary.{component}.json"


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
        return component_scores_path(metric_dir, "clip_aes").is_file() and component_features_path(metric_dir, "clip").is_file()
    if component == "hpsv2":
        return component_scores_path(metric_dir, "hpsv2").is_file()
    if component == "dino":
        return component_features_path(metric_dir, "dino").is_file()
    if component == "fid":
        return component_summary_path(metric_dir, "fid").is_file()
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


def write_component_features(
    groups: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    samples_dir: Path,
    metrics_dir: Path,
    name: str,
) -> None:
    pieces = split_tensor_by_groups(features, groups)
    for group in groups:
        metric_dir = manifest_metric_dir(samples_dir, metrics_dir, group["manifest"])
        metric_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pieces[group["manifest"]], component_features_path(metric_dir, name))


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


def build_clip_aes_models(device: str) -> tuple[Any, Any, AestheticHead]:
    from packaging.version import Version
    from transformers import CLIPModel, CLIPProcessor
    from transformers import __version__ as transformers_version

    root = project_root()
    clip_path = require_local_path(
        os.getenv("CLIP_MODEL_PATH") or root / "models" / "CLIP-ViT-L-14",
        description="CLIP reward model",
        must_be_file=False,
    )
    aesthetic_head_path = require_local_path(
        os.getenv("AES_HEAD_PATH") or root / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth",
        description="Aesthetic reward head",
        must_be_file=True,
    )
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    dtype_key = "dtype" if Version(transformers_version).major >= 5 else "torch_dtype"
    processor = CLIPProcessor.from_pretrained(str(clip_path), local_files_only=True)
    model = CLIPModel.from_pretrained(str(clip_path), **{dtype_key: dtype}, local_files_only=True).to(device_obj).eval()
    projection_dim = int(getattr(model.config, "projection_dim", 768))
    head = AestheticHead(projection_dim).to(device_obj).eval()
    head.load_state_dict(strip_state_dict_prefixes(torch.load(aesthetic_head_path, map_location="cpu")))
    return processor, model, head


def build_dino_model(device: str, *, model_path: str | Path | None = None, processor_path: str | Path | None = None, feature_key: str = "layer12_patch_mean") -> Any:
    from drpo.features import FrozenDinoImageFeatureExtractor

    root = project_root()
    return FrozenDinoImageFeatureExtractor(
        model_path or root / "models" / "dinov2-base",
        processor_name_or_path=processor_path,
        feature_key=feature_key,
    ).to(torch.device(device)).eval()


def build_inception_model(device: str) -> tuple[Any, torch.Tensor, torch.Tensor]:
    import torchvision.models as models

    device_obj = torch.device(device)
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.to(device_obj).eval()
    mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device_obj).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device_obj).view(1, 3, 1, 1)
    return model, mean_tensor, std_tensor


def evaluate_core(args) -> None:
    samples_dir = Path(args.samples_dir)
    metrics_dir = Path(args.metrics_dir)
    cache_dir = None if args.no_feature_cache else Path(args.cache_dir or metrics_dir / ".cache")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    device_obj = torch.device(args.device)
    components = parse_component_list(getattr(args, "components", "all"))
    if "dino" in components:
        require_local_path(args.dino_model_path, description="DINO diversity model", must_be_file=False)
    manifests = resolve_requested_manifests(args, samples_dir)
    groups, _ = build_manifest_groups(manifests)
    if not groups:
        write_summary_csv(metrics_dir)
        return

    if "pickscore" in components:
        selected = groups_needing_component(
            groups,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            component="pickscore",
            force=args.force,
        )
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
        selected = groups_needing_component(
            groups,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            component="clip_aes",
            force=args.force,
        )
        if selected:
            rows = flatten_groups(selected)
            clip_processor, clip_model, aes_head = build_clip_aes_models(args.device)
            clip_features = score_clip_aes_and_features(
                rows,
                device=args.device,
                batch_size=args.reward_batch_size,
                processor=clip_processor,
                model=clip_model,
                head=aes_head,
                device_obj=device_obj,
                image_load_workers=args.image_load_workers,
            )
            write_component_scores(selected, samples_dir=samples_dir, metrics_dir=metrics_dir, component="clip_aes")
            write_component_features(selected, clip_features, samples_dir=samples_dir, metrics_dir=metrics_dir, name="clip")

    if "hpsv2" in components:
        selected = groups_needing_component(
            groups,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            component="hpsv2",
            force=args.force,
        )
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

    if "dino" in components:
        selected = groups_needing_component(
            groups,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            component="dino",
            force=args.force,
        )
        if selected:
            rows = flatten_groups(selected)
            dino_model = build_dino_model(
                args.device,
                model_path=args.dino_model_path,
                processor_path=args.dino_processor_path,
                feature_key=args.dino_feature_key,
            )
            dino_features = dino_image_features(
                rows,
                device=args.device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                model=dino_model,
                device_obj=device_obj,
            )
            write_component_features(selected, dino_features, samples_dir=samples_dir, metrics_dir=metrics_dir, name="dino")

    if "fid" in components:
        selected = groups_needing_component(
            groups,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            component="fid",
            force=args.force,
        )
        baseline_manifest = Path(args.baseline_manifest) if args.baseline_manifest else samples_dir / "sd-turbo-baseline" / "default" / "manifest.jsonl"
        baseline_rows = read_jsonl(baseline_manifest) if baseline_manifest.is_file() else []
        if selected and baseline_rows:
            rows = flatten_groups(selected)
            inception_model, inception_mean, inception_std = build_inception_model(args.device)
            baseline_inception = inception_features(
                baseline_rows,
                device=args.device,
                batch_size=args.fid_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                cache_path=feature_cache_path(cache_dir, prefix="inception", manifest=baseline_manifest),
                model=inception_model,
                mean_tensor=inception_mean,
                std_tensor=inception_std,
                device_obj=device_obj,
            )
            cand_inception = inception_features(
                rows,
                device=args.device,
                batch_size=args.fid_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                cache_path=None,
                model=inception_model,
                mean_tensor=inception_mean,
                std_tensor=inception_std,
                device_obj=device_obj,
            )
            pieces = split_tensor_by_groups(cand_inception, selected)
            for group in selected:
                rows_for_group = group["rows"]
                if not rows_for_group or rows_for_group[0].get("model_type") == "sd-turbo-baseline":
                    continue
                metric_dir = manifest_metric_dir(samples_dir, metrics_dir, group["manifest"])
                metric_dir.mkdir(parents=True, exist_ok=True)
                summary = {"fid_vs_baseline": fid_from_features(baseline_inception, pieces[group["manifest"]])}
                with component_summary_path(metric_dir, "fid").open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if "summarize" in components:
        selected = groups_needing_component(
            groups,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            component="summarize",
            force=args.force,
        )
        for group in selected:
            manifest = group["manifest"]
            rows = read_jsonl(manifest)
            metric_dir = manifest_metric_dir(samples_dir, metrics_dir, manifest)
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
            summary = build_summary(rows)
            clip_path = component_features_path(metric_dir, "clip")
            if clip_path.is_file():
                clip_features = torch.load(clip_path, map_location="cpu")
                summary.update(
                    diversity_summary(
                        rows,
                        clip_features,
                        prefix="clip",
                        min_seeds_per_prompt=args.diversity_min_seeds_per_prompt,
                        min_prompts=args.diversity_min_prompts,
                    )
                )
            dino_path = component_features_path(metric_dir, "dino")
            if dino_path.is_file():
                dino_features = torch.load(dino_path, map_location="cpu")
                summary.update(
                    diversity_summary(
                        rows,
                        dino_features,
                        prefix="dino",
                        min_seeds_per_prompt=args.diversity_min_seeds_per_prompt,
                        min_prompts=args.diversity_min_prompts,
                    )
                )
            fid_path = component_summary_path(metric_dir, "fid")
            if fid_path.is_file():
                with fid_path.open("r", encoding="utf-8") as handle:
                    summary.update(json.load(handle))
            metric_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(metric_dir / "scores.jsonl", rows)
            with (metric_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_csv(metrics_dir)


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


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate generated samples.")
    parser.add_argument("--metric-set", choices=["core", "imagereward"], default="core")
    parser.add_argument("--samples-dir", default=str(root / "samples"))
    parser.add_argument("--metrics-dir", default=str(root / "samples" / "metrics"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--manifest-list", default=None)
    parser.add_argument("--baseline-manifest", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--fid-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-load-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--dino-model-path", default=str(root / "models" / "dinov2-base"))
    parser.add_argument("--dino-processor-path", default=None)
    parser.add_argument("--dino-feature-key", default="layer12_patch_mean")
    parser.add_argument(
        "--diversity-min-seeds-per-prompt",
        type=int,
        default=25,
        help="Minimum seed count required before a prompt contributes to prompt-mean diversity.",
    )
    parser.add_argument(
        "--diversity-min-prompts",
        type=int,
        default=20,
        help="Minimum number of valid prompts required before reporting the main diversity value.",
    )
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated core metric components: pickscore,clip_aes,hpsv2,dino,fid,summarize, or all.",
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
