"""Shared checkpoint helpers for baseline trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    return str(value)


def checkpoint_step(checkpoint_dir: str | Path) -> int:
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "training_state.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            return int(metadata.get("global_step", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    name = checkpoint_path.name
    if name.startswith("checkpoint-"):
        suffix = name.removeprefix("checkpoint-")
        if suffix.isdigit():
            return int(suffix)
    return 0


def latest_checkpoint(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    candidates = [
        (step, path)
        for path in output_path.glob("checkpoint-*")
        if path.is_dir() and (step := checkpoint_step(path)) > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {output_path}.")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_resume_checkpoint(output_dir: str | Path, resume_from_checkpoint: str | None) -> Path | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        return latest_checkpoint(output_dir)

    checkpoint_path = Path(resume_from_checkpoint).expanduser()
    if not checkpoint_path.exists() and not checkpoint_path.is_absolute():
        output_relative = Path(output_dir) / checkpoint_path
        if output_relative.exists():
            checkpoint_path = output_relative
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume_from_checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def resume_training_state(accelerator, output_dir: str | Path, resume_from_checkpoint: str | None, logger=None) -> int:
    checkpoint_path = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    if checkpoint_path is None:
        return 0
    if logger is not None:
        logger.info("Resuming training state from %s", checkpoint_path)
    accelerator.load_state(str(checkpoint_path))
    step = checkpoint_step(checkpoint_path)
    if logger is not None:
        logger.info("Resumed training state at global step %s.", step)
    return step


def save_training_checkpoint(
    accelerator,
    unet,
    args,
    checkpoint_dir: str | Path,
    global_step: int,
    checkpoint_type: str,
    logger=None,
) -> None:
    checkpoint_path = Path(checkpoint_dir)
    if accelerator.is_main_process:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(checkpoint_path))

    if accelerator.is_main_process:
        unet_to_save = accelerator.unwrap_model(unet)
        if getattr(args, "use_lora", False):
            unet_to_save.save_pretrained(checkpoint_path / "unet_lora")
        else:
            unet_to_save.save_pretrained(checkpoint_path / "unet")

        metadata = {
            "checkpoint_type": checkpoint_type,
            "global_step": int(global_step),
            "max_train_steps": int(getattr(args, "max_train_steps", 0)),
            "checkpointing_steps": int(getattr(args, "checkpointing_steps", 0)),
            "resume_from_checkpoint": str(checkpoint_path),
            "contains_accelerate_state": True,
            "contains_optimizer_state": True,
            "contains_scheduler_state": True,
            "args": {key: _make_json_safe(value) for key, value in vars(args).items()},
        }
        with (checkpoint_path / "training_state.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        if logger is not None:
            logger.info("Saved %s checkpoint to %s", checkpoint_type, checkpoint_path)

    accelerator.wait_for_everyone()
