import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from tqdm.auto import tqdm

from drpo.paths import project_root, require_local_path
from drpo.sdturbo import SDTurboOneStepSampler, decode_latents_to_pil, encode_prompts, load_sdturbo_components


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
SAMPLER_NAME = "drpo_sdturbo_onestep_wrapper"


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    prompt: str


@dataclass(frozen=True)
class SampleTask:
    prompt_id: int
    prompt: str
    seed: int
    latent_seed: int
    image_path: Path


def parse_int_list(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("Expected at least one integer value.")
        return tuple(int(item) for item in items)
    return tuple(int(item) for item in value)


def read_prompts(path: str | Path, *, max_prompts: int | None = None) -> list[PromptRecord]:
    prompt_path = require_local_path(path, description="prompt file", must_be_file=True)
    records: list[PromptRecord] = []
    with Path(prompt_path).open("r", encoding="utf-8") as handle:
        for line_id, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            prompt = raw
            prompt_id = line_id
            if raw.startswith("{"):
                payload = json.loads(raw)
                prompt = str(payload.get("prompt", "")).strip()
                prompt_id = int(payload.get("prompt_id", line_id))
            if not prompt:
                continue
            records.append(PromptRecord(prompt_id=prompt_id, prompt=prompt))
            if max_prompts is not None and len(records) >= max_prompts:
                break
    if not records:
        raise ValueError(f"No prompts found in {prompt_path}.")
    return records


def project_relative(path: Path) -> str:
    root = project_root().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(tmp_name).replace(path)


def image_path_for(output_dir: Path, prompt_id: int, seed: int) -> Path:
    return output_dir / f"seed_{seed}" / "images" / f"{prompt_id:06d}.png"


def latent_seed_for(prompt_id: int, seed: int) -> int:
    return int(seed) * 1_000_000 + int(prompt_id)


def build_tasks(output_dir: Path, prompts: list[PromptRecord], seeds: Iterable[int]) -> list[SampleTask]:
    tasks = []
    for seed in seeds:
        for record in prompts:
            tasks.append(
                SampleTask(
                    prompt_id=record.prompt_id,
                    prompt=record.prompt,
                    seed=int(seed),
                    latent_seed=latent_seed_for(record.prompt_id, int(seed)),
                    image_path=image_path_for(output_dir, record.prompt_id, int(seed)),
                )
            )
    return tasks


def manifest_rows(
    tasks: Iterable[SampleTask],
    *,
    model_type: str,
    checkpoint_path: str | Path | None = None,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
) -> list[dict]:
    rows = []
    for task in tasks:
        rows.append(
            {
                "model_type": model_type,
                "sampler": SAMPLER_NAME,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "checkpoint_path": project_relative(Path(checkpoint_path)) if checkpoint_path else None,
                "prompt_id": task.prompt_id,
                "prompt": task.prompt,
                "seed": task.seed,
                "latent_seed": task.latent_seed,
                "image_path": project_relative(task.image_path),
            }
        )
    return rows


def output_dir_needs_resample(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    first_row = json.loads(line)
                except json.JSONDecodeError:
                    return True
                return first_row.get("sampler") != SAMPLER_NAME
        return True
    return any(output_dir.glob("seed_*/images/*.png"))


def checkpoint_output_dir(samples_dir: Path, checkpoint_path: str | Path, *, outputs_dir: str | Path = "outputs") -> Path:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    outputs_root = Path(outputs_dir).resolve()
    try:
        relative = checkpoint_dir.resolve().relative_to(outputs_root)
    except ValueError:
        relative = Path(checkpoint_dir.name)
    return samples_dir / "sd-turbo-lora" / relative


def resolve_checkpoint_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "adapter_model.safetensors":
        return candidate.parent.parent
    if candidate.is_dir() and candidate.name == "unet_lora":
        return candidate.parent
    if candidate.is_dir() and (candidate / "unet_lora" / "adapter_model.safetensors").is_file():
        return candidate
    raise FileNotFoundError(f"Expected a checkpoint dir or unet_lora adapter path, got: {candidate}")


def resolve_lora_dir(path: str | Path) -> Path:
    checkpoint_dir = resolve_checkpoint_dir(path)
    lora_dir = checkpoint_dir / "unet_lora"
    require_local_path(lora_dir / "adapter_model.safetensors", description="LoRA adapter", must_be_file=True)
    return lora_dir


def _enable_xformers(unet) -> None:
    for candidate in (
        unet,
        getattr(unet, "base_model", None),
        getattr(getattr(unet, "base_model", None), "model", None),
    ):
        if candidate is None or not hasattr(candidate, "enable_xformers_memory_efficient_attention"):
            continue
        try:
            candidate.enable_xformers_memory_efficient_attention()
            return
        except Exception:
            return


def load_sampling_components(
    pretrained_model_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    revision: str | None = None,
    lora_path: str | Path | None = None,
    enable_xformers: bool = True,
):
    tokenizer, text_encoder, vae, unet, scheduler = load_sdturbo_components(str(pretrained_model_path), revision=revision)
    if lora_path is not None:
        unet = PeftModel.from_pretrained(unet, str(resolve_lora_dir(lora_path)))
    text_encoder.to(device=device, dtype=dtype).requires_grad_(False).eval()
    vae.to(device=device, dtype=dtype).requires_grad_(False).eval()
    unet.to(device=device, dtype=dtype).requires_grad_(False).eval()
    if enable_xformers:
        _enable_xformers(unet)
    return tokenizer, text_encoder, vae, unet, scheduler


def sample_tasks(
    *,
    tasks: list[SampleTask],
    pretrained_model_path: str | Path,
    device: str,
    dtype_name: str,
    batch_size: int,
    resolution: int,
    generation_timestep: int | None = None,
    generation_target_timestep: int | None = None,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
    revision: str | None = None,
    lora_path: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    device_obj = torch.device(device)
    if dtype_name == "fp16":
        dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    elif dtype_name == "bf16":
        dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32
    elif dtype_name == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    if num_inference_steps != 1:
        raise ValueError("SD-Turbo one-step wrapper only supports num_inference_steps=1.")
    if guidance_scale != 0.0:
        raise ValueError("SD-Turbo one-step wrapper only supports guidance_scale=0.0.")

    generation_timestep = 999 if generation_timestep is None else int(generation_timestep)
    generation_target_timestep = 0 if generation_target_timestep is None else int(generation_target_timestep)
    tokenizer, text_encoder, vae, unet, scheduler = load_sampling_components(
        pretrained_model_path,
        device=device_obj,
        dtype=dtype,
        revision=revision,
        lora_path=lora_path,
    )
    sampler = SDTurboOneStepSampler(
        unet=unet,
        scheduler=scheduler,
        timestep=generation_timestep,
        target_timestep=generation_target_timestep,
    )
    pending = [task for task in tasks if overwrite or not task.image_path.is_file()]
    if not pending:
        return

    latent_shape = (1, int(unet.config.in_channels), resolution // 8, resolution // 8)
    with torch.inference_mode():
        for start in tqdm(range(0, len(pending), batch_size), desc="sampling", dynamic_ncols=True):
            batch = pending[start: start + batch_size]
            prompt_ids = tokenizer(
                [task.prompt for task in batch],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device_obj)
            hidden = encode_prompts(text_encoder, prompt_ids).to(dtype=dtype)
            latent_chunks = []
            for task in batch:
                generator = torch.Generator(device=device_obj).manual_seed(task.latent_seed)
                latent_chunks.append(torch.randn(latent_shape, generator=generator, device=device_obj, dtype=dtype))
            noisy_latents = torch.cat(latent_chunks, dim=0)
            clean_latents = sampler.sample_clean_latents(noisy_latents, hidden)
            images = decode_latents_to_pil(vae, clean_latents)
            for task, image in zip(batch, images):
                task.image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(task.image_path)


def add_common_sampling_args(parser: argparse.ArgumentParser) -> None:
    root = project_root()
    parser.add_argument("--pretrained-model-path", default=str(root / "models" / "sd-turbo"))
    parser.add_argument("--prompt-file", default=str(root / "data" / "pickscore" / "test.txt"))
    parser.add_argument("--samples-dir", default=str(root / "samples"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--generation-timestep", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generation-target-timestep", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--overwrite", action="store_true")
