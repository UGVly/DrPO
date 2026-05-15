from __future__ import annotations

import argparse
from pathlib import Path

from inference.common import (
    add_common_sampling_args,
    atomic_write_jsonl,
    build_tasks,
    checkpoint_output_dir,
    manifest_rows,
    output_dir_needs_resample,
    parse_int_list,
    read_prompts,
    resolve_checkpoint_dir,
    sample_tasks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample an SD-Turbo LoRA checkpoint.")
    add_common_sampling_args(parser)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir, unet_lora dir, or adapter_model.safetensors.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_output_dir(Path(args.samples_dir), checkpoint_dir, outputs_dir=args.outputs_dir)
    prompts = read_prompts(args.prompt_file, max_prompts=args.max_prompts)
    seeds = parse_int_list(args.seeds)
    tasks = build_tasks(output_dir, prompts, seeds)
    sample_tasks(
        tasks=tasks,
        pretrained_model_path=args.pretrained_model_path,
        device=args.device,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        resolution=args.resolution,
        generation_timestep=args.generation_timestep,
        generation_target_timestep=args.generation_target_timestep,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        revision=args.revision,
        lora_path=checkpoint_dir,
        overwrite=args.overwrite or output_dir_needs_resample(output_dir),
    )
    atomic_write_jsonl(
        output_dir / "manifest.jsonl",
        manifest_rows(
            tasks,
            model_type="sd-turbo-lora",
            checkpoint_path=checkpoint_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ),
    )
    print(f"Wrote LoRA samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
