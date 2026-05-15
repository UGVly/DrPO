from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("spo_losses", ROOT / "baselines" / "spo" / "losses.py")
assert spec is not None and spec.loader is not None
spo_losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spo_losses)

pairwise_spo_loss = spo_losses.pairwise_spo_loss


class SPOLossTest(unittest.TestCase):
    def test_loss_prefers_higher_winner_log_ratio(self):
        rewards = torch.tensor([2.0, 1.0])
        good_loss, good_stats = pairwise_spo_loss(
            torch.tensor([0.5, -0.5]),
            torch.tensor([0.0, 0.0]),
            rewards,
            beta=2.0,
            clip_range=0.0,
            max_log_ratio=10.0,
        )
        bad_loss, bad_stats = pairwise_spo_loss(
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.0, 0.0]),
            rewards,
            beta=2.0,
            clip_range=0.0,
            max_log_ratio=10.0,
        )
        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertGreater(good_stats["pair_accuracy"].item(), bad_stats["pair_accuracy"].item())

    def test_requires_even_sample_count(self):
        with self.assertRaisesRegex(ValueError, "even"):
            pairwise_spo_loss(
                torch.zeros(3),
                torch.zeros(3),
                torch.zeros(3),
                beta=1.0,
                clip_range=0.0,
                max_log_ratio=10.0,
            )


if __name__ == "__main__":
    unittest.main()
