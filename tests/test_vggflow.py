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
spec = importlib.util.spec_from_file_location(
    "vggflow_reward_gradient",
    ROOT / "baselines" / "vggflow" / "reward_gradient.py",
)
assert spec is not None and spec.loader is not None
reward_gradient = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_gradient)

clip_gradient_by_norm = reward_gradient.clip_gradient_by_norm
eta_multiplier = reward_gradient.eta_multiplier


class VGGFlowHelpersTest(unittest.TestCase):
    def test_eta_modes(self):
        sigma = torch.tensor([0.25])
        self.assertAlmostEqual(eta_multiplier("constant", sigma).item(), 1.0)
        self.assertAlmostEqual(eta_multiplier("linear", sigma).item(), 0.75)
        self.assertAlmostEqual(eta_multiplier("quad", sigma).item(), 0.75**2)

    def test_clip_gradient_by_norm(self):
        gradient = torch.tensor([[[[3.0, 4.0]]], [[[0.0, 2.0]]]])
        clipped = clip_gradient_by_norm(gradient, threshold=1.0, enabled=True)
        norms = torch.linalg.norm(clipped.flatten(1), dim=1)
        self.assertTrue(torch.all(norms <= 1.0001))

    def test_clip_can_be_disabled(self):
        gradient = torch.randn(2, 1, 2, 2)
        self.assertTrue(torch.equal(clip_gradient_by_norm(gradient, threshold=1.0, enabled=False), gradient))


if __name__ == "__main__":
    unittest.main()
