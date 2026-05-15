from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import torch
from tqdm.auto import tqdm

from drpo.paths import project_root, require_local_path
from drpo.sdturbo import SDTurboOneStepSampler, decode_latents_to_pil, encode_prompts
from inference.common import (
    add_common_sampling_args,
    atomic_write_jsonl,
    load_sampling_components,
    parse_int_list,
)
from utils.geneval_utils import Selector


@dataclass
class GenevalPromptState:
    prompt_id: int
    prompt: str
    metadata: dict
    solved: bool = False
    success_attempt: int | None = None
    success_image_path: str | None = None
    attempts_run: int = 0
    attempt_scores: list[float] = field(default_factory=list)


def _read_geneval_rows(path: str | Path, *, max_prompts: int | None = None) -> list[GenevalPromptState]:
    prompt_path = require_local_path(path, description="Geneval prompt metadata file", must_be_file=True)
    rows: list[GenevalPromptState] = []
    with Path(prompt_path).open("r", encoding="utf-8") as handle:
        for prompt_id, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at line {prompt_id + 1}.")
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            rows.append(GenevalPromptState(prompt_id=prompt_id, prompt=prompt, metadata=row))
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError(f"No prompts found in {prompt_path}.")
    return rows


def _iter_chunks(values: list[GenevalPromptState], chunk_size: int) -> Iterable[list[GenevalPromptState]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _attempt_image_path(output_dir: Path, attempt: int, prompt_id: int) -> Path:
    return output_dir / "attempts" / f"attempt_{attempt:02d}" / "images" / f"{prompt_id:06d}.png"


def _latent_seed(seed: int, prompt_id: int, attempt: int) -> int:
    return int(seed) * 1_000_000 + int(prompt_id) * 100 + int(attempt)


def _build_summary(states: list[GenevalPromptState], *, max_attempts: int) -> dict:
    solved = [state for state in states if state.solved]
    unsolved = [state for state in states if not state.solved]

    per_tag_total: dict[str, int] = {}
    per_tag_solved: dict[str, int] = {}
    for state in states:
        tag = str(state.metadata.get("tag", "unknown"))
        per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
        if state.solved:
            per_tag_solved[tag] = per_tag_solved.get(tag, 0) + 1

    cumulative_by_attempt = []
    for attempt in range(1, max_attempts + 1):
        solved_count = sum(1 for state in states if state.success_attempt is not None and state.success_attempt <= attempt)
        cumulative_by_attempt.append(
            {
                "attempt": attempt,
                "solved_prompts": solved_count,
                "success_rate": solved_count / len(states) if states else 0.0,
            }
        )

    attempts_to_success = [int(state.success_attempt) for state in solved if state.success_attempt is not None]
    per_tag = {}
    for tag in sorted(per_tag_total):
        total = per_tag_total[tag]
        solved_count = per_tag_solved.get(tag, 0)
        per_tag[tag] = {
            "total_prompts": total,
            "solved_prompts": solved_count,
            "success_rate": solved_count / total if total else 0.0,
        }

    overall_prompt_success_rate = len(solved) / len(states) if states else 0.0
    best_of_n_task_score = mean(item["success_rate"] for item in per_tag.values()) if per_tag else 0.0

    return {
        "total_prompts": len(states),
        "max_attempts": max_attempts,
        "solved_prompts": len(solved),
        "unsolved_prompts": len(unsolved),
        "overall_prompt_success_rate": overall_prompt_success_rate,
        "best_of_n_task_score": best_of_n_task_score,
        "mean_attempts_to_success": mean(attempts_to_success) if attempts_to_success else None,
        "median_attempts_to_success": median(attempts_to_success) if attempts_to_success else None,
        "max_attempt_needed_for_success": max(attempts_to_success) if attempts_to_success else None,
        "cumulative_success_by_attempt": cumulative_by_attempt,
        "per_tag": per_tag,
        "unsolved_prompt_ids": [state.prompt_id for state in unsolved],
        "unsolved_prompts": [state.prompt for state in unsolved],
    }


def _write_progress(output_dir: Path, states: list[GenevalPromptState], *, max_attempts: int, attempt_records: list[dict]) -> None:
    summary = _build_summary(states, max_attempts=max_attempts)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_write_jsonl(output_dir / "prompt_results.jsonl", (_prompt_row(state) for state in states))
    atomic_write_jsonl(output_dir / "attempt_results.jsonl", attempt_records)


def _prompt_row(state: GenevalPromptState) -> dict:
    return {
        "prompt_id": state.prompt_id,
        "prompt": state.prompt,
        "tag": state.metadata.get("tag", "unknown"),
        "solved": state.solved,
        "success_attempt": state.success_attempt,
        "success_image_path": state.success_image_path,
        "attempts_run": state.attempts_run,
        "attempt_scores": state.attempt_scores,
        "metadata": state.metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Best-of-N Geneval retry evaluation for the base SD-Turbo model.")
    add_common_sampling_args(parser)
    root = project_root()
    parser.set_defaults(
        prompt_file=str(root / "third_party" / "geneval" / "prompts" / "evaluation_metadata.jsonl"),
        samples_dir=str(root / "samples"),
    )
    parser.add_argument("--run-name", default="geneval_budget_base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--geneval-repo", default=str(root / "third_party" / "geneval"))
    parser.add_argument("--geneval-detector-path", default=str(root / "models" / "geneval_detector"))
    parser.add_argument("--geneval-model-config", default=None)
    parser.add_argument("--geneval-options", default="")
    parser.add_argument("--geneval-device", default=None)
    parser.add_argument("--geneval-python-bin", default=None)
    parser.add_argument("--geneval-conda-env", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >= 1.")
    if args.num_inference_steps != 1:
        raise ValueError("This script only supports SD-Turbo one-step generation.")
    if args.guidance_scale != 0.0:
        raise ValueError("This script only supports guidance_scale=0.0.")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.samples_dir) / "sd-turbo-base-geneval-budget" / args.run_name
    states = _read_geneval_rows(args.prompt_file, max_prompts=args.max_prompts)
    attempt_records: list[dict] = []

    if args.geneval_python_bin:
        Path(args.geneval_python_bin).expanduser()
        import os

        os.environ["GENEVAL_PYTHON_BIN"] = str(Path(args.geneval_python_bin).expanduser())
    if args.geneval_conda_env:
        import os

        os.environ["GENEVAL_CONDA_ENV"] = args.geneval_conda_env

    sample_device = torch.device(args.device)
    eval_device = torch.device(args.geneval_device or args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if sample_device.type != "cuda":
        dtype = torch.float32

    selector = Selector(
        device=eval_device,
        geneval_repo=str(require_local_path(args.geneval_repo, description="Geneval repo", must_be_file=False)),
        geneval_detector_path=str(require_local_path(args.geneval_detector_path, description="Geneval detector path", must_be_file=False)),
        geneval_model_config=args.geneval_model_config,
        geneval_options=args.geneval_options,
    )

    tokenizer, text_encoder, vae, unet, scheduler = load_sampling_components(
        args.pretrained_model_path,
        device=sample_device,
        dtype=dtype,
        revision=args.revision,
        lora_path=None,
    )
    sampler = SDTurboOneStepSampler(
        unet=unet,
        scheduler=scheduler,
        timestep=999 if args.generation_timestep is None else int(args.generation_timestep),
        target_timestep=0 if args.generation_target_timestep is None else int(args.generation_target_timestep),
    )

    latent_shape = (1, int(unet.config.in_channels), args.resolution // 8, args.resolution // 8)
    for attempt in range(1, args.max_attempts + 1):
        unresolved = [state for state in states if not state.solved]
        if not unresolved:
            break

        attempt_dir = output_dir / "attempts" / f"attempt_{attempt:02d}" / "images"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            for batch_states in tqdm(
                list(_iter_chunks(unresolved, args.batch_size)),
                desc=f"attempt {attempt}/{args.max_attempts}",
                dynamic_ncols=True,
            ):
                prompt_ids = tokenizer(
                    [state.prompt for state in batch_states],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(sample_device)
                hidden = encode_prompts(text_encoder, prompt_ids).to(dtype=dtype)
                latent_chunks = []
                for state in batch_states:
                    generator = torch.Generator(device=sample_device).manual_seed(_latent_seed(args.seed_base, state.prompt_id, attempt))
                    latent_chunks.append(torch.randn(latent_shape, generator=generator, device=sample_device, dtype=dtype))
                noisy_latents = torch.cat(latent_chunks, dim=0)
                clean_latents = sampler.sample_clean_latents(noisy_latents, hidden)
                images = decode_latents_to_pil(vae, clean_latents)

                for state, image in zip(batch_states, images):
                    image_path = _attempt_image_path(output_dir, attempt, state.prompt_id)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                    score = float(selector.score([image], state.metadata)[0])
                    solved_now = score >= 0.5
                    state.attempts_run += 1
                    state.attempt_scores.append(score)
                    if solved_now and not state.solved:
                        state.solved = True
                        state.success_attempt = attempt
                        state.success_image_path = str(image_path)
                    attempt_records.append(
                        {
                            "prompt_id": state.prompt_id,
                            "prompt": state.prompt,
                            "tag": state.metadata.get("tag", "unknown"),
                            "attempt": attempt,
                            "score": score,
                            "solved_now": solved_now,
                            "image_path": str(image_path),
                        }
                    )
        _write_progress(output_dir, states, max_attempts=args.max_attempts, attempt_records=attempt_records)

    print(f"Wrote retry-budget report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
