from __future__ import annotations

import argparse
from pathlib import Path

from inference.common import (
    add_common_sampling_args,
    atomic_write_jsonl,
    build_tasks,
    manifest_rows,
    output_dir_needs_resample,
    parse_int_list,
    read_prompts,
    sample_tasks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample the base SD-Turbo model.")
    add_common_sampling_args(parser)
    parser.add_argument("--run-name", default="default")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples_dir = Path(args.samples_dir)
    output_dir = samples_dir / "sd-turbo-baseline" / args.run_name
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
        overwrite=args.overwrite or output_dir_needs_resample(output_dir),
    )
    atomic_write_jsonl(
        output_dir / "manifest.jsonl",
        manifest_rows(
            tasks,
            model_type="sd-turbo-baseline",
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ),
    )
    print(f"Wrote baseline samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
