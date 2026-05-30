import json
import logging
import os
import shlex
import socket
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model


def parse_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def dtype_for_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def create_accelerator(config, output_dir: Path) -> Accelerator:
    return Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
        log_with=config.report_to,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / config.logging_dir)),
    )


def setup_training_logging(config, accelerator: Accelerator, *, logger_name: str | None = None) -> None:
    log_dir = Path(config.output_dir) / config.logging_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(log_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=f"%(asctime)s | %(levelname)s | rank={accelerator.process_index}/{accelerator.num_processes} | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    if logger_name:
        logging.getLogger(logger_name).debug("training logging configured")


def save_runtime_snapshot(config, accelerator: Accelerator) -> None:
    if not accelerator.is_main_process:
        return
    out = Path(config.output_dir) / "run_metadata"
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    runtime = {
        "created_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "world_size": accelerator.num_processes,
        "device": str(accelerator.device),
        "torch_version": torch.__version__,
    }
    (out / "runtime_info.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    (out / "launch_command.sh").write_text(
        "#!/usr/bin/env bash\n" + shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )


def resolve_unet_lora_dir(path: str | Path) -> Path:
    checkpoint_dir = Path(path).expanduser()
    if checkpoint_dir.is_file() and checkpoint_dir.name == "adapter_model.safetensors":
        return checkpoint_dir.parent
    if checkpoint_dir.is_dir() and checkpoint_dir.name == "unet_lora":
        return checkpoint_dir
    if checkpoint_dir.is_dir() and (checkpoint_dir / "unet_lora" / "adapter_model.safetensors").is_file():
        return checkpoint_dir / "unet_lora"
    raise FileNotFoundError(f"Expected checkpoint dir or unet_lora adapter path, got: {checkpoint_dir}")


def resume_global_step(path: str | Path | None) -> int:
    if not path:
        return 0
    checkpoint_dir = Path(path).expanduser()
    if checkpoint_dir.name == "unet_lora":
        checkpoint_dir = checkpoint_dir.parent
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.is_file():
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state.get("global_step", 0))


def make_lora_trainable(unet, config):
    if not config.use_lora:
        unet.requires_grad_(True)
        return unet
    if config.resume_from_checkpoint:
        return PeftModel.from_pretrained(unet, str(resolve_unet_lora_dir(config.resume_from_checkpoint)), is_trainable=True)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )
    return get_peft_model(unet, lora_config)


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def maybe_enable_xformers(unet, logger: logging.Logger | None = None) -> None:
    if not is_xformers_available():
        return
    import xformers

    if version.parse(xformers.__version__) < version.parse("0.0.17"):
        return
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not enable xformers: %s", exc)


def save_unet_checkpoint(
    accelerator: Accelerator,
    unet,
    config,
    checkpoint_dir: Path,
    *,
    global_step: int,
    checkpoint_type: str,
    metadata: dict[str, object],
) -> None:
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        if config.use_lora:
            unwrapped.save_pretrained(checkpoint_dir / "unet_lora")
        else:
            unwrapped.save_pretrained(checkpoint_dir / "unet")
        full_metadata = {
            "checkpoint_type": checkpoint_type,
            "global_step": global_step,
            "created_at": datetime.now().isoformat(),
            "contains_accelerate_state": False,
            "contains_model_state": True,
            "model_variant": config.model_variant,
            "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
            **metadata,
        }
        (checkpoint_dir / "training_state.json").write_text(
            json.dumps(full_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


ScalarLogValue = torch.Tensor | float | int
CheckpointFn = Callable[[Accelerator, object, object, Path, int, str], None]


def _as_float(value: ScalarLogValue) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _float_log_payload(log_values: dict[str, ScalarLogValue]) -> dict[str, float]:
    return {key: _as_float(value) for key, value in log_values.items()}


def _format_postfix(value: torch.Tensor | float | str) -> str:
    if isinstance(value, str):
        return value
    if torch.is_tensor(value):
        value = float(value.detach().cpu())
    return f"{float(value):.4f}"


class TrainingStepCallbacks:
    """Shared end-of-step logging, checkpointing, and stop handling."""

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        progress,
        config,
        output_dir: Path,
        unet,
        save_checkpoint: CheckpointFn,
    ) -> None:
        self.accelerator = accelerator
        self.progress = progress
        self.config = config
        self.output_dir = output_dir
        self.unet = unet
        self.save_checkpoint = save_checkpoint

    def finish_step(
        self,
        *,
        step: int,
        loss: torch.Tensor,
        lr: float,
        log_values: dict[str, ScalarLogValue],
        postfix: dict[str, torch.Tensor | float | str] | None = None,
    ) -> tuple[int, bool]:
        if not self.accelerator.sync_gradients:
            return step, False

        step += 1
        values = {
            "train_loss": loss.detach(),
            "lr": torch.tensor(lr, device=self.accelerator.device),
            **log_values,
        }
        self.accelerator.log(_float_log_payload(values), step=step)
        self.progress.update(1)
        self.progress.set_postfix(**{key: _format_postfix(value) for key, value in (postfix or {"loss": loss.detach()}).items()})

        checkpointing_steps = int(getattr(self.config, "checkpointing_steps", 0))
        if checkpointing_steps > 0 and step % checkpointing_steps == 0:
            self.save_checkpoint(
                self.accelerator,
                self.unet,
                self.config,
                self.output_dir / f"checkpoint-{step}",
                step,
                "intermediate",
            )
        return step, step >= int(self.config.max_train_steps)

    def finish_training(self, step: int) -> None:
        self.save_checkpoint(
            self.accelerator,
            self.unet,
            self.config,
            self.output_dir / "final",
            step,
            "final",
        )
        self.accelerator.end_training()
