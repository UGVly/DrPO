from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("grpo_losses", ROOT / "baselines" / "grpo" / "losses.py")
assert spec is not None and spec.loader is not None
grpo_losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grpo_losses)

construct_neighbor_noises = grpo_losses.construct_neighbor_noises
neighbor_grpo_loss = grpo_losses.neighbor_grpo_loss
quasi_norm_advantages = grpo_losses.quasi_norm_advantages
softmax_distance_log_probs = grpo_losses.softmax_distance_log_probs


class FusedGRPOLossTest(unittest.TestCase):
    def test_construct_neighbor_noises_shape_and_sigma_one(self):
        base = torch.ones(2, 3)
        deltas = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        out = construct_neighbor_noises(base, deltas, sigma=1.0)
        self.assertEqual(tuple(out.shape), (4, 2, 3))
        self.assertTrue(torch.equal(out, deltas))

    def test_construct_neighbor_noises_rejects_bad_sigma(self):
        base = torch.ones(2, 3)
        deltas = torch.ones(4, 2, 3)
        for sigma in (0.0, -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "sigma"):
                construct_neighbor_noises(base, deltas, sigma=sigma)

    def test_quasi_norm_preserves_sign_and_order(self):
        rewards = torch.tensor([0.1, 0.7, 0.2, 1.0])
        adv = quasi_norm_advantages(rewards, p=0.8, clip=0.0)
        centered = rewards - rewards.mean()
        self.assertTrue(torch.equal(adv > 0, centered > 0))
        self.assertEqual(torch.argmax(adv).item(), torch.argmax(rewards).item())
        self.assertEqual(torch.argmin(adv).item(), torch.argmin(rewards).item())

    def test_p_two_matches_l2_zscore_direction(self):
        rewards = torch.tensor([0.1, 0.7, 0.2, 1.0])
        adv = quasi_norm_advantages(rewards, p=2.0, clip=0.0)
        centered = rewards - rewards.mean()
        zscore = centered / rewards.std(unbiased=False)
        self.assertTrue(torch.allclose(adv, zscore, atol=1e-6))

    def test_quasi_norm_downweights_flat_groups(self):
        rewards = torch.tensor([1.0, 1.001, 0.999, 1.0005, 0.9995, 1.0])
        adv_p08 = quasi_norm_advantages(rewards, p=0.8, clip=0.0)
        adv_p2 = quasi_norm_advantages(rewards, p=2.0, clip=0.0)
        self.assertLess(adv_p08.abs().mean().item(), adv_p2.abs().mean().item())

    def test_softmax_ratio_is_one_when_anchor_matches_old_anchor(self):
        candidates = torch.tensor([[[[0.0]]], [[[1.0]]], [[[2.0]]]])
        old_anchor = candidates[1]
        current_logp = softmax_distance_log_probs(candidates, old_anchor, temperature=1.0)
        old_logp = softmax_distance_log_probs(candidates, old_anchor, temperature=1.0)
        self.assertTrue(torch.allclose((current_logp - old_logp).exp(), torch.ones(3)))

    def test_loss_prefers_anchor_moving_toward_high_reward_candidate(self):
        candidates = torch.tensor([[[[0.0]]], [[[2.0]]]])
        old_anchor = candidates[0]
        advantages = torch.tensor([-1.0, 1.0])
        good_loss, _ = neighbor_grpo_loss(
            candidates,
            current_anchor=torch.tensor([[[1.0]]]),
            old_anchor=old_anchor,
            advantages=advantages,
            clip_range=0.2,
            temperature=1.0,
        )
        bad_loss, _ = neighbor_grpo_loss(
            candidates,
            current_anchor=torch.tensor([[[-1.0]]]),
            old_anchor=old_anchor,
            advantages=advantages,
            clip_range=0.2,
            temperature=1.0,
        )
        self.assertLess(good_loss.item(), bad_loss.item())


if __name__ == "__main__":
    unittest.main()
