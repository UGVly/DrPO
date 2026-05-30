from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from drpo.paths import require_local_path


SD_TURBO_TIMESTEP = 999


def enforce_zero_terminal_snr(scheduler: DDPMScheduler) -> None:
    alphas = 1 - scheduler.betas
    alphas_bar = alphas.cumprod(0)
    sqrt_bar = alphas_bar.sqrt()
    first = sqrt_bar[0].clone()
    last = sqrt_bar[-1].clone()
    sqrt_bar = (sqrt_bar - last) * first / (first - last)
    alphas_bar = sqrt_bar.square()
    alphas = torch.cat([alphas_bar[:1], alphas_bar[1:] / alphas_bar[:-1]])
    scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0)


def one_step_clean_latent(noisy_latent: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
    """SD-Turbo one-step latent conversion used by the original training code."""
    return ((noisy_latent - 0.9977 * noise_pred) / 0.0683) * 0.9996 + 0.0292 * noise_pred


def is_default_sdturbo_projection(timestep: int, target_timestep: int) -> bool:
    """Return true for the original SD-Turbo one-step inference rule."""
    return int(timestep) == SD_TURBO_TIMESTEP and int(target_timestep) in {-1, 0}


def resolve_weight_file(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Weight file or directory not found: {path}")
    if (path / "unet").is_dir():
        path = path / "unet"
    preferred_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    )
    for name in preferred_names:
        candidate = path / name
        if candidate.is_file():
            return candidate
    candidates = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin")) or sorted(path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No weight file found under {path}")
    if len(candidates) > 1:
        raise ValueError(f"Found multiple candidate weight files under {path}; pass a specific file instead.")
    return candidates[0]


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        raise ValueError("Empty state dict.")
    prefixes = ("unet.", "model.diffusion_model.")
    keys = list(state_dict.keys())
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in keys):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _compute_lora_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, *, out_shape: torch.Size) -> torch.Tensor:
    if lora_a.ndim == 2 and lora_b.ndim == 2:
        return lora_b.float() @ lora_a.float()
    if lora_a.ndim == 4 and lora_b.ndim == 4:
        if lora_a.shape[2:] == (1, 1) and lora_b.shape[2:] == (1, 1):
            a_matrix = lora_a.float().reshape(lora_a.shape[0], lora_a.shape[1])
            b_matrix = lora_b.float().reshape(lora_b.shape[0], lora_b.shape[1])
            return (b_matrix @ a_matrix).reshape(out_shape)
        if lora_b.shape[2:] == (1, 1):
            return torch.einsum("orxy,rihw->oihw", lora_b.float(), lora_a.float())
        if lora_a.shape[2:] == (1, 1):
            return torch.einsum("orhw,rixy->oihw", lora_b.float(), lora_a.float())
    raise ValueError(
        f"Unsupported LoRA weight shapes for merge: A={tuple(lora_a.shape)} B={tuple(lora_b.shape)} out={tuple(out_shape)}"
    )


def merge_base_layer_lora_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(".base_layer." in key for key in state_dict):
        return state_dict
    merged_state_dict: dict[str, torch.Tensor] = {}
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if ".base_layer." in key:
            merged_state_dict[key.replace(".base_layer.", ".")] = value
            continue
        if ".lora_A." in key:
            lora_pairs.setdefault(key.split(".lora_A.", 1)[0], {})["A"] = value
            continue
        if ".lora_B." in key:
            lora_pairs.setdefault(key.split(".lora_B.", 1)[0], {})["B"] = value
            continue
        merged_state_dict[key] = value
    for prefix, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA pair for {prefix}; found keys={sorted(pair)}")
        weight_key = f"{prefix}.weight"
        if weight_key not in merged_state_dict:
            raise KeyError(f"Missing base weight for merged LoRA key: {weight_key}")
        base_weight = merged_state_dict[weight_key]
        delta = _compute_lora_delta(pair["A"], pair["B"], out_shape=base_weight.shape).to(dtype=base_weight.dtype)
        merged_state_dict[weight_key] = base_weight + delta
    return merged_state_dict


def load_state_dict_file(weight_path: Path) -> dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        state_dict = load_safetensors_file(str(weight_path), device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict from {weight_path}, got {type(state_dict)}")
    return merge_base_layer_lora_weights(normalize_state_dict_keys(state_dict))


def load_unet_checkpoint_weights(unet: UNet2DConditionModel, checkpoint_path: str | Path) -> Path:
    weight_path = resolve_weight_file(checkpoint_path)
    state_dict = load_state_dict_file(weight_path)
    incompatible_keys = unet.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        raise ValueError(
            "UNet checkpoint does not match the base model.\n"
            f"path={weight_path}\n"
            f"missing_keys={incompatible_keys.missing_keys[:8]}\n"
            f"unexpected_keys={incompatible_keys.unexpected_keys[:8]}"
        )
    return weight_path


def encode_prompts(text_encoder: CLIPTextModel, input_ids: torch.Tensor) -> torch.Tensor:
    return text_encoder(input_ids)[0]


def encode_images(vae: AutoencoderKL, images: torch.Tensor, *, mode: str = "mode") -> torch.Tensor:
    latent_dist = vae.encode(images).latent_dist
    if mode == "mode":
        latents = latent_dist.mode()
    elif mode == "sample":
        latents = latent_dist.sample()
    else:
        raise ValueError(f"Unknown VAE latent encode mode: {mode}")
    return latents * vae.config.scaling_factor


def decode_latents_to_tensor(vae: AutoencoderKL, latents: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    scaled_latents = (latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)
    if scaled_latents.shape[0] <= chunk_size:
        return vae.decode(scaled_latents).sample.float().clamp(-1, 1)
    decoded_chunks = [vae.decode(chunk).sample for chunk in scaled_latents.split(chunk_size)]
    return torch.cat(decoded_chunks, dim=0).float().clamp(-1, 1)


@torch.no_grad()
def decode_latents_to_pil(vae: AutoencoderKL, latents: torch.Tensor, *, chunk_size: int = 4) -> list[Image.Image]:
    scaled_latents = (latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)
    decoded_batches = [vae.decode(chunk).sample for chunk in scaled_latents.split(chunk_size)]
    images = torch.cat(decoded_batches, dim=0)
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((image * 255).round().astype("uint8")) for image in images]


def scheduler_alpha_prod(
    scheduler: DDPMScheduler,
    timestep: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    return scheduler.alphas_cumprod[int(timestep)].to(dtype=dtype)


def project_model_output_to_target_timestep(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    scheduler: DDPMScheduler,
    *,
    timestep: int,
    target_timestep: int,
) -> torch.Tensor:
    if is_default_sdturbo_projection(timestep, target_timestep):
        return one_step_clean_latent(sample, model_output)

    sample_f = sample.float()
    model_output_f = model_output.float()
    alpha_prod_t = scheduler_alpha_prod(scheduler, timestep, device=sample.device, dtype=torch.float32).clamp(min=1e-6)
    beta_prod_t = (1 - alpha_prod_t).clamp(min=1e-6)
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_original_sample = (sample_f - beta_prod_t.sqrt() * model_output_f) / alpha_prod_t.sqrt()
        pred_epsilon = model_output_f
    elif prediction_type == "sample":
        pred_original_sample = model_output_f
        pred_epsilon = (sample_f - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample_f - beta_prod_t.sqrt() * model_output_f
        pred_epsilon = alpha_prod_t.sqrt() * model_output_f + beta_prod_t.sqrt() * sample_f
    else:
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")
    if target_timestep < 0:
        return pred_original_sample.to(dtype=sample.dtype)
    alpha_prod_target = scheduler_alpha_prod(scheduler, target_timestep, device=sample.device, dtype=torch.float32).clamp(min=1e-6)
    beta_prod_target = (1 - alpha_prod_target).clamp(min=1e-6)
    projected = alpha_prod_target.sqrt() * pred_original_sample + beta_prod_target.sqrt() * pred_epsilon
    return projected.to(dtype=sample.dtype)


@dataclass
class SDTurboOneStepSampler:
    """Small wrapper for SD-Turbo one-step latent sampling.

    The default SD-Turbo path must use the original one-step conversion constants
    instead of a scheduler-derived projection. Non-default timesteps keep the
    scheduler projection path for ablations and diagnostics.
    """

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler | None = None
    timestep: int = SD_TURBO_TIMESTEP
    target_timestep: int = 0

    def _timesteps(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), int(self.timestep), device=device, dtype=torch.long)

    def project(self, noisy_latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        if is_default_sdturbo_projection(self.timestep, self.target_timestep):
            return one_step_clean_latent(noisy_latents, model_output)
        if self.scheduler is None:
            raise ValueError("scheduler is required when using non-default SD-Turbo timestep projection.")
        return project_model_output_to_target_timestep(
            sample=noisy_latents,
            model_output=model_output,
            scheduler=self.scheduler,
            timestep=self.timestep,
            target_timestep=self.target_timestep,
        )

    def __call__(
        self,
        noisy_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self._timesteps(noisy_latents.shape[0], device=noisy_latents.device)
        model_output = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_output, self.project(noisy_latents, model_output)

    def sample_clean_latents(
        self,
        noisy_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, clean_latents = self(noisy_latents, encoder_hidden_states)
        return clean_latents


def run_one_step_unet(
    unet: UNet2DConditionModel,
    noisy_latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    scheduler: DDPMScheduler | None,
    *,
    timestep: int = SD_TURBO_TIMESTEP,
    target_timestep: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampler = SDTurboOneStepSampler(
        unet=unet,
        scheduler=scheduler,
        timestep=timestep,
        target_timestep=target_timestep,
    )
    return sampler(noisy_latents, encoder_hidden_states)


def sample_clean_latents(
    unet: UNet2DConditionModel,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    scheduler: DDPMScheduler | None = None,
    *,
    generation_timestep: int = SD_TURBO_TIMESTEP,
    generation_target_timestep: int = 0,
) -> torch.Tensor:
    sampler = SDTurboOneStepSampler(
        unet=unet,
        scheduler=scheduler,
        timestep=generation_timestep,
        target_timestep=generation_target_timestep,
    )
    return sampler.sample_clean_latents(latents, text_embeddings)


def load_sdturbo_components(model_path: str, revision: str | None = None):
    model_path = str(require_local_path(model_path, description="SD-Turbo model directory", must_be_file=False))
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=revision, local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision=revision, local_files_only=True)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=revision, local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=revision, local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", revision=revision, local_files_only=True)
    enforce_zero_terminal_snr(scheduler)
    return tokenizer, text_encoder, vae, unet, scheduler


def prompts_to_jsonl_rows(prompts: Sequence[str]) -> list[dict[str, str]]:
    return [{"prompt": prompt} for prompt in prompts]





def _self_test_projection_utils() -> None:
    """不依赖真实 SD-Turbo 模型的轻量测试。"""
    print("[1/2] Running lightweight projection utility tests...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    enforce_zero_terminal_snr(scheduler)

    batch_size = 2
    noisy_latents = torch.randn(batch_size, 4, 64, 64, device=device, dtype=dtype)
    noise_pred = torch.randn_like(noisy_latents)

    clean_latents = one_step_clean_latent(noisy_latents, noise_pred)
    assert clean_latents.shape == noisy_latents.shape
    assert clean_latents.dtype == noisy_latents.dtype

    assert is_default_sdturbo_projection(999, 0) is True
    assert is_default_sdturbo_projection(999, -1) is True
    assert is_default_sdturbo_projection(500, 0) is False

    projected_default = project_model_output_to_target_timestep(
        sample=noisy_latents,
        model_output=noise_pred,
        scheduler=scheduler,
        timestep=999,
        target_timestep=0,
    )
    assert projected_default.shape == noisy_latents.shape

    projected_non_default = project_model_output_to_target_timestep(
        sample=noisy_latents,
        model_output=noise_pred,
        scheduler=scheduler,
        timestep=500,
        target_timestep=250,
    )
    assert projected_non_default.shape == noisy_latents.shape

    print("✅ Lightweight projection utility tests passed.")
    print(
        f"clean_latents mean={clean_latents.float().mean().item():.6f}, "
        f"std={clean_latents.float().std().item():.6f}"
    )


@torch.no_grad()
def _smoke_test_sdturbo_model(
    model_path: str,
    prompt: str,
    checkpoint_path: str | None = None,
    device_str: str | None = None,
    latent_size: int = 64,
    save_preview: bool = False,
    output_dir: str = "debug_outputs",
) -> None:
    """加载本地 SD-Turbo，并跑一次 one-step UNet 前向测试。"""
    print("[2/2] Running SD-Turbo model smoke test...")

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer, text_encoder, vae, unet, scheduler = load_sdturbo_components(model_path)

    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()
    vae = vae.to(device=device, dtype=dtype).eval()
    unet = unet.to(device=device, dtype=dtype).eval()

    if checkpoint_path is not None and checkpoint_path != "":
        loaded_weight_path = load_unet_checkpoint_weights(unet, checkpoint_path)
        print(f"✅ Loaded UNet checkpoint: {loaded_weight_path}")

    tokens = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    input_ids = tokens.input_ids.to(device)
    text_embeddings = encode_prompts(text_encoder, input_ids).to(device=device, dtype=dtype)

    noisy_latents = torch.randn(
        1,
        unet.config.in_channels,
        latent_size,
        latent_size,
        device=device,
        dtype=dtype,
    )

    model_output, clean_latents = run_one_step_unet(
        unet=unet,
        noisy_latents=noisy_latents,
        encoder_hidden_states=text_embeddings,
        scheduler=scheduler,
        timestep=SD_TURBO_TIMESTEP,
        target_timestep=0,
    )

    print("✅ SD-Turbo one-step UNet smoke test passed.")
    print(f"model_output shape: {tuple(model_output.shape)}")
    print(f"clean_latents shape: {tuple(clean_latents.shape)}")
    print(
        f"clean_latents mean={clean_latents.float().mean().item():.6f}, "
        f"std={clean_latents.float().std().item():.6f}"
    )

    if save_preview:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        images = decode_latents_to_pil(vae, clean_latents)
        preview_path = output_path / "sdturbo_smoke_test.png"
        images[0].save(preview_path)

        print(f"✅ Preview image saved to: {preview_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # =========================
    # 直接在这里改测试参数
    # =========================

    # 本地 SD-Turbo 模型路径。
    # 如果留空字符串 ""，则只运行轻量函数测试，不加载真实模型。
    model_path = ""

    # 可选：你自己训练出来的 UNet checkpoint 路径。
    # 可以是具体权重文件，也可以是包含权重文件的目录。
    checkpoint_path = ""

    # 测试 prompt
    prompt = "a cute cat, high quality"

    # 运行设备：
    # None 表示自动选择 cuda / cpu；
    # 也可以手动写 "cuda", "cuda:0", "cpu"
    device = None

    # SD 512x512 通常对应 latent_size=64
    latent_size = 64

    # 是否把输出 latent decode 成图片保存
    save_preview = False

    # 图片保存目录
    output_dir = "debug_outputs"

    # =========================
    # 开始测试
    # =========================

    _self_test_projection_utils()

    if model_path != "":
        _smoke_test_sdturbo_model(
            model_path=model_path,
            prompt=prompt,
            checkpoint_path=checkpoint_path,
            device_str=device,
            latent_size=latent_size,
            save_preview=save_preview,
            output_dir=output_dir,
        )
    else:
        print("model_path is empty, skipped real SD-Turbo model smoke test.")
