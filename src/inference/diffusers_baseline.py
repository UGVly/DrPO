from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from tqdm.auto import tqdm

from inference.common import DEFAULT_SEEDS, atomic_write_jsonl, parse_int_list, project_relative
from drpo.paths import project_root, require_local_path


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


def read_prompts_any(path: str | Path, *, max_prompts: int | None = None) -> list[PromptRecord]:
    prompt_path = require_local_path(path, description="prompt file", must_be_file=True)
    records: list[PromptRecord] = []
    with Path(prompt_path).open("r", encoding="utf-8") as handle:
        for line_id, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            prompt = raw
            if raw.startswith("{"):
                payload = json.loads(raw)
                prompt = str(payload.get("prompt", "")).strip()
                prompt_id = int(payload.get("prompt_id", line_id))
            else:
                prompt_id = line_id
            if not prompt:
                continue
            records.append(PromptRecord(prompt_id=prompt_id, prompt=prompt))
            if max_prompts is not None and len(records) >= max_prompts:
                break
    if not records:
        raise ValueError(f"No prompts found in {prompt_path}.")
    return records


def latent_seed_for(prompt_id: int, seed: int) -> int:
    return int(seed) * 1_000_000 + int(prompt_id)


def image_path_for(output_dir: Path, prompt_id: int, seed: int) -> Path:
    return output_dir / f"seed_{seed}" / "images" / f"{prompt_id:06d}.png"


def build_tasks(output_dir: Path, prompts: list[PromptRecord], seeds: Iterable[int]) -> list[SampleTask]:
    tasks: list[SampleTask] = []
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


def output_dir_needs_resample(output_dir: Path, sampler_name: str) -> bool:
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
                return first_row.get("sampler") != sampler_name
        return True
    return any(output_dir.glob("seed_*/images/*.png"))


def manifest_rows(
    tasks: Iterable[SampleTask],
    *,
    model_type: str,
    sampler_name: str,
    model_path: str | Path,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
    save_resolution: int | None,
    scheduler: str,
    lcm_lora_path: str | Path | None,
    disable_t5_text_encoder: bool,
) -> list[dict]:
    resolved_save_resolution = int(save_resolution) if save_resolution else int(resolution)
    rows = []
    for task in tasks:
        rows.append(
            {
                "model_type": model_type,
                "sampler": sampler_name,
                "model_path": project_relative(Path(model_path)),
                "lcm_lora_path": project_relative(Path(lcm_lora_path)) if lcm_lora_path else None,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "resolution": int(resolution),
                "save_resolution": resolved_save_resolution,
                "scheduler": scheduler,
                "disable_t5_text_encoder": bool(disable_t5_text_encoder),
                "checkpoint_path": None,
                "prompt_id": task.prompt_id,
                "prompt": task.prompt,
                "seed": task.seed,
                "latent_seed": task.latent_seed,
                "image_path": project_relative(task.image_path),
            }
        )
    return rows


def atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.png", dir=str(path.parent))
    os.close(fd)
    try:
        image.save(tmp_name)
        Path(tmp_name).replace(path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def build_pipeline(args, *, device: torch.device, dtype: torch.dtype):
    from diffusers import DiffusionPipeline, StableDiffusionPipeline

    model_path = Path(args.pretrained_model_path)
    common_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if args.disable_t5_text_encoder:
        if args.from_single_file:
            raise ValueError("--disable-t5-text-encoder is only supported with directory diffusers pipelines.")
        common_kwargs.update({"text_encoder_3": None, "tokenizer_3": None})
    if args.from_single_file:
        pipe = StableDiffusionPipeline.from_single_file(str(model_path), **common_kwargs)
    else:
        pipe = DiffusionPipeline.from_pretrained(str(model_path), **common_kwargs)
    if args.scheduler == "lcm":
        from diffusers import LCMScheduler

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddim":
        from diffusers import DDIMScheduler

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "dpmpp_2m":
        from diffusers import DPMSolverMultistepScheduler

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler != "default":
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    if args.lcm_lora_path:
        pipe.load_lora_weights(str(args.lcm_lora_path))
        if args.fuse_lora:
            pipe.fuse_lora()
    pipe = pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if args.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if args.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if args.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def sample_tasks(args, tasks: list[SampleTask]) -> None:
    device = torch.device(args.device)
    if args.dtype == "fp16":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.float32
    pipe = build_pipeline(args, device=device, dtype=dtype)
    pending = [task for task in tasks if args.overwrite or not task.image_path.is_file()]
    if not pending:
        return
    save_resolution = int(args.save_resolution) if args.save_resolution else int(args.resolution)
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    with torch.inference_mode():
        for start in tqdm(range(0, len(pending), args.batch_size), desc=f"sampling {args.model_type}", dynamic_ncols=True):
            batch = pending[start : start + args.batch_size]
            generators = [torch.Generator(device=device).manual_seed(task.latent_seed) for task in batch]
            output = pipe(
                [task.prompt for task in batch],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                generator=generators,
            )
            for task, image in zip(batch, output.images):
                if image.size != (save_resolution, save_resolution):
                    image = image.resize((save_resolution, save_resolution), resample_filter)
                atomic_save_image(image.convert("RGB"), task.image_path)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Sample generic diffusers text-to-image baselines.")
    parser.add_argument("--pretrained-model-path", required=True)
    parser.add_argument("--prompt-file", default=str(root / "data" / "pickscore" / "test.txt"))
    parser.add_argument("--samples-dir", default=str(root / "samples"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS[:1]))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save-resolution", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--scheduler", choices=["default", "ddim", "dpmpp_2m", "lcm"], default="default")
    parser.add_argument("--lcm-lora-path", default=None)
    parser.add_argument("--fuse-lora", action="store_true")
    parser.add_argument("--from-single-file", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--enable-attention-slicing", action="store_true")
    parser.add_argument("--enable-vae-slicing", action="store_true")
    parser.add_argument("--disable-t5-text-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples_dir = Path(args.samples_dir)
    output_dir = samples_dir / "diffusers-baseline" / args.run_name
    prompts = read_prompts_any(args.prompt_file, max_prompts=args.max_prompts)
    seeds = parse_int_list(args.seeds)
    tasks = build_tasks(output_dir, prompts, seeds)
    sampler_name = f"diffusers_{args.scheduler}"
    if args.disable_t5_text_encoder:
        sampler_name += "_no_t5"
    args.overwrite = args.overwrite or output_dir_needs_resample(output_dir, sampler_name)
    sample_tasks(args, tasks)
    atomic_write_jsonl(
        output_dir / "manifest.jsonl",
        manifest_rows(
            tasks,
            model_type=args.model_type,
            sampler_name=sampler_name,
            model_path=args.pretrained_model_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            resolution=args.resolution,
            save_resolution=args.save_resolution,
            scheduler=args.scheduler,
            lcm_lora_path=args.lcm_lora_path,
            disable_t5_text_encoder=args.disable_t5_text_encoder,
        ),
    )
    print(f"Wrote {args.model_type} samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
