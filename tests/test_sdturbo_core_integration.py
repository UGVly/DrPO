import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.sdturbo import (
    SD_TURBO_TIMESTEP,
    SDTurboOneStepSampler,
    decode_latents_to_pil,
    decode_latents_to_tensor,
    is_default_sdturbo_projection,
    merge_base_layer_lora_weights,
    normalize_state_dict_keys,
    one_step_clean_latent,
    project_model_output_to_target_timestep,
    resolve_weight_file,
)


class FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=0.5)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = 0

    def decode(self, latents):
        self.calls += 1
        return SimpleNamespace(sample=latents[:, :3].clamp(-1, 1))


class SDTurboCoreIntegrationTest(unittest.TestCase):
    def test_default_projection_accepts_zero_and_negative_one_targets(self):
        self.assertTrue(is_default_sdturbo_projection(SD_TURBO_TIMESTEP, 0))
        self.assertTrue(is_default_sdturbo_projection(SD_TURBO_TIMESTEP, -1))
        self.assertFalse(is_default_sdturbo_projection(998, 0))
        self.assertFalse(is_default_sdturbo_projection(SD_TURBO_TIMESTEP, 5))

    def test_resolve_weight_file_prefers_nested_unet_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unet").mkdir()
            preferred = root / "unet" / "diffusion_pytorch_model.safetensors"
            preferred.write_bytes(b"x")
            self.assertEqual(resolve_weight_file(root), preferred)

    def test_resolve_weight_file_rejects_ambiguous_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.safetensors").write_bytes(b"a")
            (root / "b.safetensors").write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "multiple candidate"):
                resolve_weight_file(root)

    def test_normalize_state_dict_keys_rejects_empty_dict(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            normalize_state_dict_keys({})

    def test_normalize_state_dict_keys_strips_unet_prefix(self):
        value = torch.ones(1)
        self.assertEqual(list(normalize_state_dict_keys({"unet.conv.weight": value})), ["conv.weight"])

    def test_normalize_state_dict_keys_strips_diffusion_model_prefix(self):
        value = torch.ones(1)
        result = normalize_state_dict_keys({"model.diffusion_model.conv.weight": value})
        self.assertEqual(list(result), ["conv.weight"])

    def test_merge_base_layer_lora_weights_for_linear_layer(self):
        state = {
            "proj.base_layer.weight": torch.zeros(2, 3),
            "proj.lora_A.default.weight": torch.ones(1, 3),
            "proj.lora_B.default.weight": torch.tensor([[2.0], [3.0]]),
        }
        merged = merge_base_layer_lora_weights(state)
        self.assertTrue(torch.allclose(merged["proj.weight"], torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))

    def test_merge_base_layer_lora_weights_for_one_by_one_conv(self):
        state = {
            "conv.base_layer.weight": torch.zeros(2, 3, 1, 1),
            "conv.lora_A.default.weight": torch.ones(1, 3, 1, 1),
            "conv.lora_B.default.weight": torch.tensor([[[[2.0]]], [[[3.0]]]]),
        }
        merged = merge_base_layer_lora_weights(state)
        expected = torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]).view(2, 3, 1, 1)
        self.assertTrue(torch.allclose(merged["conv.weight"], expected))

    def test_merge_base_layer_lora_weights_rejects_incomplete_pair(self):
        state = {
            "proj.base_layer.weight": torch.zeros(2, 3),
            "proj.lora_A.default.weight": torch.ones(1, 3),
        }
        with self.assertRaisesRegex(ValueError, "Incomplete LoRA pair"):
            merge_base_layer_lora_weights(state)

    def test_default_projection_matches_reference_one_step_formula(self):
        sample = torch.randn(2, 4, 4, 4)
        output = torch.randn_like(sample)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        projected = project_model_output_to_target_timestep(sample, output, scheduler, timestep=999, target_timestep=0)
        self.assertTrue(torch.allclose(projected, one_step_clean_latent(sample, output)))

    def test_epsilon_projection_to_original_sample_matches_scheduler_formula(self):
        sample = torch.randn(1, 2, 2, 2)
        epsilon = torch.randn_like(sample)
        scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type="epsilon")
        projected = project_model_output_to_target_timestep(sample, epsilon, scheduler, timestep=5, target_timestep=-1)
        alpha = scheduler.alphas_cumprod[5]
        expected = (sample.float() - (1 - alpha).sqrt() * epsilon.float()) / alpha.sqrt()
        self.assertTrue(torch.allclose(projected, expected.to(sample.dtype), atol=1e-5))

    def test_sample_prediction_projection_uses_model_output_as_original_sample(self):
        sample = torch.randn(1, 2, 2, 2)
        model_output = torch.randn_like(sample)
        scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type="sample")
        projected = project_model_output_to_target_timestep(sample, model_output, scheduler, timestep=5, target_timestep=-1)
        self.assertTrue(torch.allclose(projected, model_output, atol=1e-5))

    def test_v_prediction_projection_to_target_timestep_is_finite(self):
        sample = torch.randn(1, 2, 2, 2)
        model_output = torch.randn_like(sample)
        scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type="v_prediction")
        projected = project_model_output_to_target_timestep(sample, model_output, scheduler, timestep=5, target_timestep=2)
        self.assertEqual(tuple(projected.shape), tuple(sample.shape))
        self.assertTrue(torch.isfinite(projected).all())

    def test_sampler_requires_scheduler_for_non_default_projection(self):
        sampler = SDTurboOneStepSampler(unet=object(), scheduler=None, timestep=5, target_timestep=0)
        with self.assertRaisesRegex(ValueError, "scheduler is required"):
            sampler.project(torch.ones(1, 1), torch.zeros(1, 1))

    def test_decode_latents_to_tensor_chunks_fake_vae_decodes(self):
        vae = FakeVAE()
        latents = torch.randn(3, 3, 4, 4)
        decoded = decode_latents_to_tensor(vae, latents, chunk_size=2)
        self.assertEqual(tuple(decoded.shape), (3, 3, 4, 4))
        self.assertEqual(vae.calls, 2)
        self.assertGreaterEqual(float(decoded.min()), -1.0)
        self.assertLessEqual(float(decoded.max()), 1.0)

    def test_decode_latents_to_pil_returns_uint8_images(self):
        vae = FakeVAE()
        images = decode_latents_to_pil(vae, torch.zeros(1, 3, 2, 2))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(images[0].size, (2, 2))

    def test_decode_latents_to_pil_chunks_fake_vae_decodes(self):
        vae = FakeVAE()
        images = decode_latents_to_pil(vae, torch.zeros(3, 3, 2, 2), chunk_size=2)
        self.assertEqual(len(images), 3)
        self.assertEqual(vae.calls, 2)


if __name__ == "__main__":
    unittest.main()
