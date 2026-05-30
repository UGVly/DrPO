from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from tqdm.auto import tqdm

from drpo.paths import project_root, require_local_path
from inference.common import (
    SampleTask,
    atomic_write_jsonl,
    build_tasks,
    parse_int_list,
    project_relative,
    read_prompts,
)

SAMPLER_NAME = "sdxl_turbo_lora_onestep"


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def resolve_unet_lora(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file() and checkpoint.name == "adapter_model.safetensors":
        return checkpoint.parent
    if checkpoint.is_dir() and checkpoint.name == "unet_lora":
        require_local_path(checkpoint / "adapter_model.safetensors", description="SDXL UNet LoRA", must_be_file=True)
        return checkpoint
    if checkpoint.is_dir() and (checkpoint / "unet_lora" / "adapter_model.safetensors").is_file():
        return checkpoint / "unet_lora"
    raise FileNotFoundError(f"Expected checkpoint dir or unet_lora adapter path, got {checkpoint}")


def checkpoint_dir_for(path: str | Path) -> Path:
    lora_dir = resolve_unet_lora(path)
    return lora_dir.parent if lora_dir.name == "unet_lora" else lora_dir


def default_output_dir(samples_dir: Path, checkpoint: str | Path) -> Path:
    checkpoint_dir = checkpoint_dir_for(checkpoint).resolve()
    outputs_root = (project_root() / "outputs").resolve()
    try:
        relative = checkpoint_dir.relative_to(outputs_root)
    except ValueError:
        relative = Path(checkpoint_dir.name)
    return samples_dir / "sdxl-turbo-lora" / relative


def manifest_rows(
    tasks: Iterable[SampleTask],
    *,
    model_type: str,
    checkpoint_path: str | Path,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
) -> list[dict]:
    checkpoint_dir = checkpoint_dir_for(checkpoint_path)
    return [
        {
            "model_type": model_type,
            "sampler": SAMPLER_NAME,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "resolution": int(resolution),
            "checkpoint_path": project_relative(checkpoint_dir),
            "prompt_id": task.prompt_id,
            "prompt": task.prompt,
            "seed": task.seed,
            "latent_seed": task.latent_seed,
            "image_path": project_relative(task.image_path),
        }
        for task in tasks
    ]


def select_shard(tasks: list[SampleTask], *, shard_index: int, num_shards: int) -> list[SampleTask]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")
    return [task for index, task in enumerate(tasks) if index % num_shards == shard_index]


def sample(
    *,
    tasks: list[SampleTask],
    pretrained_model_path: str | Path,
    checkpoint: str | Path,
    output_dir: Path,
    device: str,
    dtype_name: str,
    model_variant: str | None,
    batch_size: int,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    overwrite: bool,
) -> None:
    device_obj = torch.device(device)
    dtype = dtype_from_name(dtype_name, device_obj)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(pretrained_model_path),
        torch_dtype=dtype,
        variant=model_variant,
        local_files_only=True,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(resolve_unet_lora(checkpoint)))
    pipe.to(device_obj)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    pending = [task for task in tasks if overwrite or not task.image_path.is_file()]
    if not pending:
        return

    with torch.inference_mode():
        for start in tqdm(range(0, len(pending), batch_size), desc="sample sdxl", dynamic_ncols=True):
            batch = pending[start : start + batch_size]
            generators = [torch.Generator(device=device_obj).manual_seed(task.latent_seed) for task in batch]
            images = pipe(
                [task.prompt for task in batch],
                height=resolution,
                width=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
                output_type="pil",
            ).images
            for task, image in zip(batch, images):
                task.image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(task.image_path)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Sample SDXL-Turbo UNet LoRA checkpoints.")
    parser.add_argument("--pretrained-model-path", default=str(root / "models" / "stable-diffusion-xl-turbo"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", default=str(root / "data" / "pickscore" / "test.txt"))
    parser.add_argument("--samples-dir", default=str(root / "samples"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--model-type", default="sdxl-turbo-lora")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--model-variant", default="fp16")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pretrained_model_path = require_local_path(args.pretrained_model_path, description="SDXL-Turbo model", must_be_file=False)
    checkpoint_dir_for(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.samples_dir), args.checkpoint)
    prompts = read_prompts(args.prompt_file, max_prompts=args.max_prompts)
    seeds = parse_int_list(args.seeds)
    all_tasks = build_tasks(output_dir, prompts, seeds)
    tasks = select_shard(all_tasks, shard_index=args.shard_index, num_shards=args.num_shards)
    sample(
        tasks=tasks,
        pretrained_model_path=pretrained_model_path,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        device=args.device,
        dtype_name=args.dtype,
        model_variant=args.model_variant or None,
        batch_size=args.batch_size,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        overwrite=args.overwrite,
    )
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / f"manifest.shard{args.shard_index:02d}.jsonl"
    atomic_write_jsonl(
        manifest_path,
        manifest_rows(
            tasks,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            resolution=args.resolution,
        ),
    )
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), "num_tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
