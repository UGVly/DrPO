import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.rewards import build_selector
from drpo.sdturbo import SDTurboOneStepSampler, enforce_zero_terminal_snr, one_step_clean_latent, prompts_to_jsonl_rows, run_one_step_unet


class FakeUNet:
    def __init__(self, scale: float = 0.25):
        self.scale = scale
        self.last_timesteps = None

    def __call__(self, noisy_latents, timesteps, encoder_hidden_states):
        self.last_timesteps = timesteps.detach().cpu()
        return SimpleNamespace(sample=noisy_latents * self.scale)


class RewardAndSDTurboTest(unittest.TestCase):
    def test_unknown_reward_selector_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown reward selector"):
            build_selector("missing", "cpu")

    def test_one_step_clean_latent_formula(self):
        noisy = torch.ones(1, 1, 1, 1)
        noise = torch.full((1, 1, 1, 1), 0.5)
        expected = ((noisy - 0.9977 * noise) / 0.0683) * 0.9996 + 0.0292 * noise
        self.assertTrue(torch.allclose(one_step_clean_latent(noisy, noise), expected))


    def test_sdturbo_sampler_smoke_uses_reference_one_step_rule(self):
        unet = FakeUNet(scale=0.25)
        sampler = SDTurboOneStepSampler(unet=unet)
        noisy = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        hidden = torch.randn(2, 77, 16, dtype=torch.float16)
        model_output, clean = sampler(noisy, hidden)
        expected_output = noisy * 0.25
        expected_clean = one_step_clean_latent(noisy, expected_output)
        self.assertTrue(torch.equal(unet.last_timesteps, torch.full((2,), 999, dtype=torch.long)))
        self.assertTrue(torch.allclose(model_output, expected_output))
        self.assertTrue(torch.allclose(clean, expected_clean))

    def test_run_one_step_unet_smoke_accepts_no_scheduler_for_default_rule(self):
        unet = FakeUNet(scale=-0.5)
        noisy = torch.randn(1, 4, 4, 4)
        hidden = torch.randn(1, 77, 8)
        model_output, clean = run_one_step_unet(unet, noisy, hidden, None, target_timestep=0)
        self.assertTrue(torch.allclose(model_output, noisy * -0.5))
        self.assertTrue(torch.allclose(clean, one_step_clean_latent(noisy, model_output)))

    def test_enforce_zero_terminal_snr_sets_last_alpha_to_zero(self):
        scheduler = DDPMScheduler(num_train_timesteps=8)
        enforce_zero_terminal_snr(scheduler)
        self.assertTrue(torch.isclose(scheduler.alphas_cumprod[-1], torch.tensor(0.0), atol=1e-6))

    def test_prompts_to_jsonl_rows(self):
        self.assertEqual(prompts_to_jsonl_rows(["a", "b"]), [{"prompt": "a"}, {"prompt": "b"}])


if __name__ == "__main__":
    unittest.main()
